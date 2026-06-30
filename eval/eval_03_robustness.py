"""Eval03: device-specific robustness sweeps for HIPSA."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_utils import build_eval_model, build_loader, evaluate_clean, load_eval_context, save_csv_rows
from hardware.robustness_perturb import PerturbationContext, default_robustness_sweeps, photonic_target_names
from utils.config_utils import save_json
from utils.seed_utils import set_seed

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def selected_sweeps(mode: str) -> Dict[str, list[Dict[str, Any]]]:
    sweeps = default_robustness_sweeps()
    if mode == "all":
        return sweeps
    if mode == "fast":
        return {
            "clean": sweeps["clean"],
            "mrr": [sweeps["mrr"][0], sweeps["mrr"][2]],
            "laser": [sweeps["laser"][0], sweeps["laser"][2]],
            "adc_bits": [sweeps["adc_bits"][1], sweeps["adc_bits"][-1]],
            "threshold": [sweeps["threshold"][1], sweeps["threshold"][-1]],
        }
    if mode in sweeps:
        return {"clean": sweeps["clean"], mode: sweeps[mode]}
    raise ValueError(f"Unknown sweep mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HIPSA eval03 robustness sweeps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--mode", default="fast", choices=["fast", "all", "mrr", "laser", "wdm", "adc_bits", "threshold", "tia_noise"])
    parser.add_argument("--include-fc2", action="store_true")
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ctx = load_eval_context(args.config, checkpoint=args.checkpoint, run_dir=args.run_dir, hardware_config=args.hardware, device_params=args.device_params, output_dir=args.output_dir, eval_name="eval03_robustness", device=args.device)
    base_seed = int(ctx.config.get("experiment", {}).get("seed", ctx.config.get("split", {}).get("seed", 42)))
    set_seed(base_seed, deterministic=True, benchmark=False)

    model = build_eval_model(ctx, strict=args.strict)
    loader = build_loader(ctx.config, split_name="test", batch_size=args.batch_size, num_workers=args.num_workers, allow_no_split=args.allow_no_split)
    target_names = photonic_target_names(model, include_fc2=args.include_fc2)
    sweeps = selected_sweeps(args.mode)

    results: Dict[str, Any] = {
        "dataset": ctx.dataset,
        "config": str(args.config),
        "checkpoint": str(ctx.checkpoint),
        "mode": args.mode,
        "trials": int(args.trials),
        "target_layers": target_names,
        "groups": {},
    }
    summary_rows: List[Dict[str, Any]] = []

    for group, cases in sweeps.items():
        results["groups"][group] = []
        for case in cases:
            name = str(case["name"])
            perturbation = dict(case.get("perturbation", {}))
            trial_metrics = []
            acc_values: List[float] = []
            loss_values: List[float] = []
            n_trials = 1 if not perturbation else int(args.trials)
            logger.info("Running robustness case %s/%s: %s", group, name, perturbation)
            for t in range(n_trials):
                seed = base_seed + 1000 * len(summary_rows) + t
                set_seed(seed, deterministic=True, benchmark=False)
                with PerturbationContext(model, perturbation, target_names=target_names, seed=seed, include_fc2=args.include_fc2):
                    metrics = evaluate_clean(model, loader, ctx.config, ctx.device, max_batches=args.max_batches)
                metrics["trial"] = t
                metrics["seed"] = seed
                trial_metrics.append(metrics)
                acc_values.append(float(metrics["acc"]))
                loss_values.append(float(metrics["loss"]))
            acc = summarize(acc_values)
            loss = summarize(loss_values)
            row = {
                "group": group,
                "name": name,
                "perturbation": perturbation,
                "accuracy_mean": acc["mean"],
                "accuracy_std": acc["std"],
                "accuracy_min": acc["min"],
                "accuracy_max": acc["max"],
                "loss_mean": loss["mean"],
                "loss_std": loss["std"],
                "num_trials": n_trials,
                "num_samples": int(trial_metrics[0].get("total", 0)) if trial_metrics else 0,
                "trials": trial_metrics,
            }
            results["groups"][group].append(row)
            flat_row = {k: v for k, v in row.items() if k not in {"trials", "perturbation"}}
            flat_row.update({f"perturb_{k}": v for k, v in perturbation.items()})
            summary_rows.append(flat_row)
            logger.info("Case %s acc %.3f ± %.3f", name, acc["mean"], acc["std"])

    save_json(results, ctx.output_dir / "eval03_robustness_results.json")
    save_csv_rows(summary_rows, ctx.output_dir / "eval03_robustness_summary.csv")
    logger.info("Saved robustness results to %s", ctx.output_dir)


if __name__ == "__main__":
    main()
