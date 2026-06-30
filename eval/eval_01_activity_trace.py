"""Eval01: layer-wise activity and SOP trace for HIPSA.

This version separates architecture-relevant activity domains:

- mvm_input_activity: input sparsity of Conv/Linear MVM, used for active SOP.
- mvm_output_nonzero_activity: raw analog Conv/Linear nonzero ratio, debug only.
- lif_spike_activity: following LIFNode output spike activity.
- adc_request_activity: comparator-threshold request proxy from raw MVM output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

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


def _nested(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def default_adc_threshold_fs(hardware_cfg: Mapping[str, Any]) -> float:
    return float(_nested(hardware_cfg, "hapr_adc_backend", "comparator_threshold_fs_default", default=0.02) or 0.02)


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
    parser.add_argument("--adc-threshold-fs", type=float, default=None, help="Comparator threshold as fraction of per-call full scale. Defaults to hardware config or 0.02.")
    parser.add_argument("--include-fc2", action="store_true")
    parser.add_argument("--no-lif-trace", action="store_true", help="Disable LIFNode hooks. Only use for debugging.")
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ctx = load_eval_context(
        args.config,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        hardware_config=args.hardware,
        device_params=args.device_params,
        output_dir=args.output_dir,
        eval_name="eval01_activity_trace",
        device=args.device,
    )
    seed = int(ctx.config.get("experiment", {}).get("seed", ctx.config.get("split", {}).get("seed", 42)))
    set_seed(seed, deterministic=True, benchmark=False)

    model = build_eval_model(ctx, strict=args.strict)
    loader = build_loader(ctx.config, split_name="test", batch_size=args.batch_size, num_workers=args.num_workers, allow_no_split=args.allow_no_split)
    names = default_photonic_layer_names(model, include_fc2=args.include_fc2)
    threshold_fs = float(args.adc_threshold_fs) if args.adc_threshold_fs is not None else default_adc_threshold_fs(ctx.hardware_cfg)

    tracer = ActivityTracer(
        model,
        target_layer_names=names,
        include_fc2=args.include_fc2,
        adc_threshold_abs=args.adc_threshold_abs,
        adc_threshold_fs=threshold_fs,
        trace_lif=(not args.no_lif_trace),
    )
    tracer.register()

    logger.info("=== Eval01 activity trace v2 ===")
    logger.info("Target MVM layers: %s", names)
    logger.info("ADC threshold: abs=%s, fs=%s", args.adc_threshold_abs, threshold_fs)
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
        "adc_threshold_fs": float(threshold_fs),
        "target_layers": names,
        "notes": {
            "mvm_input_activity": "Used for active SOP.",
            "mvm_output_nonzero_activity": "Debug only; do not interpret as spike rate.",
            "lif_spike_activity": "Post-LIF spike activity used for digital/NoC proxy.",
            "adc_request_activity": "Comparator threshold proxy used for ADC request modeling.",
        },
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
    logger.info(
        "Dense SOP/image %.3e | Active SOP/image %.3e | MVM act %.4f | LIF spike %.4f | ADC request %.4f",
        activity["dense_sop_per_image"],
        activity["active_sop_per_image"],
        activity["active_sop_ratio"],
        activity["lif_spike_activity"],
        activity["adc_request_activity"],
    )


if __name__ == "__main__":
    main()
