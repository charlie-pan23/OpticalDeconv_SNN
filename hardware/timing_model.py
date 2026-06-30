"""Eval01: layer-wise activity and SOP trace for HIPSA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from eval.eval_utils import build_eval_model, build_loader, load_eval_context, save_csv_rows
from hardware.activity_trace import ActivityTracer, default_photonic_layer_names
from hardware.sop_counter import summarize_sops_from_activity
from utils.config_utils import save_json
from utils.data_utils import prepare_snn_batch, reset_snn_state
from utils.seed_utils import set_seed

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


def main() -> None:
    parser = argparse.ArgumentParser(description="HIPSA eval01 activity trace.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--adc-threshold-abs", type=float, default=0.0)
    parser.add_argument("--include-fc2", action="store_true")
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ctx = load_eval_context(args.config, checkpoint=args.checkpoint, run_dir=args.run_dir, hardware_config=args.hardware, device_params=args.device_params, output_dir=args.output_dir, eval_name="eval01_activity_trace", device=args.device)
    seed = int(ctx.config.get("experiment", {}).get("seed", ctx.config.get("split", {}).get("seed", 42)))
    set_seed(seed, deterministic=True, benchmark=False)

    model = build_eval_model(ctx, strict=args.strict)
    loader = build_loader(ctx.config, split_name="test", batch_size=args.batch_size, num_workers=args.num_workers, allow_no_split=args.allow_no_split)
    names = default_photonic_layer_names(model, include_fc2=args.include_fc2)
    tracer = ActivityTracer(model, target_layer_names=names, include_fc2=args.include_fc2, adc_threshold_abs=args.adc_threshold_abs)
    tracer.register()

    logger.info("=== Eval01 activity trace ===")
    logger.info("Target layers: %s", names)
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            reset_snn_state(model)
            data, target = prepare_snn_batch(data, target, ctx.config, ctx.device)
            tracer.observe_batch_input(data)
            _ = model(data)
    tracer.remove()

    activity = tracer.summary({
        "dataset": ctx.dataset,
        "config": str(args.config),
        "checkpoint": str(ctx.checkpoint),
        "adc_threshold_abs": float(args.adc_threshold_abs),
        "target_layers": names,
    })
    sop_summary = summarize_sops_from_activity(activity)

    save_json(activity, ctx.output_dir / "eval01_activity_trace.json")
    save_json(sop_summary, ctx.output_dir / "eval01_sop_summary.json")
    layer_rows = []
    for name, info in activity["layers"].items():
        row = {"layer": name, **{k: v for k, v in info.items() if not isinstance(v, (dict, list))}}
        layer_rows.append(row)
    save_csv_rows(layer_rows, ctx.output_dir / "eval01_layer_activity.csv")
    logger.info("Saved activity trace to %s", ctx.output_dir)
    logger.info("Dense SOP/image %.3e | Active SOP/image %.3e | Input activity %.4f | ADC proxy %.4f", activity["dense_sop_per_image"], activity["active_sop_per_image"], activity["input_spike_activity"], activity["adc_activity_proxy"])


if __name__ == "__main__":
    main()
