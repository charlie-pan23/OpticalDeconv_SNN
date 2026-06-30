"""Eval00: clean accuracy reproduction for frozen HIPSA SNN checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_utils import build_eval_model, build_loader, evaluate_clean, load_eval_context
from utils.config_utils import save_json
from utils.seed_utils import set_seed

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


def main() -> None:
    parser = argparse.ArgumentParser(description="HIPSA eval00 clean accuracy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--allow-no-split", action="store_true", help="Debug only: evaluate full dataset if split is missing.")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ctx = load_eval_context(
        args.config,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        hardware_config=args.hardware,
        device_params=args.device_params,
        output_dir=args.output_dir,
        eval_name="eval00_clean_accuracy",
        device=args.device,
    )
    seed = int(ctx.config.get("experiment", {}).get("seed", ctx.config.get("split", {}).get("seed", 42)))
    set_seed(seed, deterministic=True, benchmark=False)

    logger.info("=== Eval00 clean accuracy ===")
    logger.info("Dataset: %s", ctx.dataset)
    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", ctx.checkpoint)
    logger.info("Output: %s", ctx.output_dir)

    model = build_eval_model(ctx, strict=args.strict)
    loader = build_loader(ctx.config, split_name="test", batch_size=args.batch_size, num_workers=args.num_workers, allow_no_split=args.allow_no_split)
    metrics = evaluate_clean(model, loader, ctx.config, ctx.device, max_batches=args.max_batches)

    result = {
        "dataset": ctx.dataset,
        "config": str(args.config),
        "checkpoint": str(ctx.checkpoint),
        "split": "test",
        "metrics": metrics,
        "artifacts": ctx.artifacts,
    }
    save_json(result, ctx.output_dir / "eval00_clean_accuracy.json")
    logger.info("Eval00 result: %s", metrics)
    logger.info("Saved: %s", ctx.output_dir / "eval00_clean_accuracy.json")


if __name__ == "__main__":
    main()
