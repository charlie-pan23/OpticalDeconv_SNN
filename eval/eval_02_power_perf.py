"""Eval02: HIPSA architecture-level power/performance estimation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_utils import load_eval_context, save_csv_rows
from hardware.adc_request_model import estimate_adc_requests
from hardware.adc_pool_model import model_adc_pool, sweep_adc_macros
from hardware.power_model import estimate_power, sweep_mrr_power
from hardware.sop_counter import summarize_sops_from_activity
from hardware.timing_model import apply_adc_stall, estimate_timing
from utils.config_utils import load_json, save_json

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


def find_activity_file(output_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        Path("results") / "cifar10dvs" / "eval" / "eval01_activity_trace" / "eval01_activity_trace.json",
        Path("results") / "dvsgesture" / "eval" / "eval01_activity_trace" / "eval01_activity_trace.json",
        output_dir.parent / "eval01_activity_trace" / "eval01_activity_trace.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find eval01_activity_trace.json. Pass --activity explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="HIPSA eval02 power/performance.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
    parser.add_argument("--activity", default=None, help="Path to eval01_activity_trace.json")
    parser.add_argument("--hapr-group-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ctx = load_eval_context(args.config, checkpoint=args.checkpoint, run_dir=args.run_dir, hardware_config=args.hardware, device_params=args.device_params, output_dir=args.output_dir, eval_name="eval02_power_perf", device="cpu")
    activity_path = find_activity_file(ctx.output_dir, args.activity)
    activity = load_json(activity_path)

    hcfg = dict(ctx.hardware_cfg)
    if args.hapr_group_size is not None:
        backend = dict(hcfg.get("hapr_adc_backend", {}))
        backend["hapr_group_size"] = int(args.hapr_group_size)
        hcfg["hapr_adc_backend"] = backend
    hapr_g = int(hcfg.get("hapr_adc_backend", {}).get("hapr_group_size", 8))

    sop_summary = summarize_sops_from_activity(activity)
    adc_requests = estimate_adc_requests(activity, hapr_group_size=hapr_g)
    timing0 = estimate_timing(sop_summary, hcfg)
    adc_pool = model_adc_pool(adc_requests, timing0, hcfg)
    timing = apply_adc_stall(timing0, adc_pool)
    power = estimate_power(activity, sop_summary, adc_requests, adc_pool, timing, hcfg, ctx.device_params)

    result = {
        "dataset": ctx.dataset,
        "config": str(args.config),
        "activity_file": str(activity_path),
        "sop_summary": sop_summary,
        "adc_requests": adc_requests,
        "adc_pool": adc_pool,
        "timing": timing,
        "power": power,
    }
    save_json(result, ctx.output_dir / "eval02_power_perf.json")

    power_rows = [{"component": k, "power_mw": v} for k, v in power["component_power_mw"].items()]
    save_csv_rows(power_rows, ctx.output_dir / "eval02_power_breakdown.csv")
    save_csv_rows(sweep_adc_macros(adc_requests, timing0, hcfg), ctx.output_dir / "eval02_adc_pool_sweep.csv")
    save_csv_rows(sweep_mrr_power(activity, sop_summary, adc_requests, adc_pool, timing, hcfg, ctx.device_params), ctx.output_dir / "eval02_mrr_sensitivity.csv")

    logger.info("=== Eval02 HIPSA power/perf ===")
    logger.info("Activity file: %s", activity_path)
    logger.info("Latency: %.3f us/image", timing["latency_us_per_image"])
    logger.info("Throughput: %.2f img/s", timing["throughput_images_per_s"])
    logger.info("Power: %.4f W", power["total_power_w"])
    logger.info("Energy: %.4f uJ/image", power["energy_uJ_per_image"])
    logger.info("Saved: %s", ctx.output_dir / "eval02_power_perf.json")


if __name__ == "__main__":
    main()
