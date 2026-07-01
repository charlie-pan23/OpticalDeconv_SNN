"""
eval_00.py

Clean workload sanity check.

This script verifies that a frozen checkpoint, config, dataset split,
and preprocessing pipeline reproduce the expected clean test accuracy.

It does NOT collect activity traces.
It does NOT call hardware models.
It does NOT generate figures.

Outputs:
  results/eval_v2/<dataset>/eval_00/
    summary.json
    predictions.csv
    confusion_matrix.csv
    per_class_accuracy.csv
    logits.npy            optional
    config_snapshot.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from eval.eval_utils import (
    build_eval_model,
    build_loader,
    load_eval_context,
    save_csv_rows,
)
from utils.config_utils import (
    copy_config_snapshot,
    dataset_tag,
    load_eval_config,
    save_json,
)
from utils.data_utils import (
    aggregate_time_logits,
    logits_aggregation_from_config,
    prepare_snn_batch,
    reset_snn_state,
)
from utils.checkpoint_utils import checkpoint_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_00: clean accuracy sanity check")

    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)

    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--output-root", default="results/eval_v2", type=str)

    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto", type=str)

    parser.add_argument("--max-batches", default=None, type=int)
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--non-strict", action="store_true")

    parser.add_argument("--save-logits", action="store_true")

    return parser.parse_args()


def build_confusion_matrix(
    targets: List[int],
    preds: List[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(targets, preds):
        matrix[int(y), int(p)] += 1
    return matrix


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    num_classes = matrix.shape[0]

    for true_cls in range(num_classes):
        row: Dict[str, Any] = {"true_class": true_cls}
        for pred_cls in range(num_classes):
            row[f"pred_{pred_cls}"] = int(matrix[true_cls, pred_cls])
        row["total"] = int(matrix[true_cls].sum())
        row["correct"] = int(matrix[true_cls, true_cls])
        row["acc_percent"] = (
            100.0 * row["correct"] / row["total"] if row["total"] > 0 else None
        )
        rows.append(row)

    save_csv_rows(rows, path)


def save_per_class_accuracy(matrix: np.ndarray, path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    num_classes = matrix.shape[0]

    for cls in range(num_classes):
        total = int(matrix[cls].sum())
        correct = int(matrix[cls, cls])
        rows.append(
            {
                "class_id": cls,
                "total": total,
                "correct": correct,
                "acc_percent": 100.0 * correct / total if total > 0 else None,
            }
        )

    save_csv_rows(rows, path)


@torch.inference_mode()
def run_clean_eval(args: argparse.Namespace) -> None:
    raw_cfg = load_eval_config(args.config)
    tag = dataset_tag(raw_cfg)
    output_dir = Path(args.output_root) / tag / "eval_00"
    output_dir.mkdir(parents=True, exist_ok=True)

    context = load_eval_context(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        eval_name="eval_00",
        device=args.device,
    )

    model = build_eval_model(context, strict=not args.non_strict)

    loader = build_loader(
        context.config,
        split_name=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_no_split=args.allow_no_split,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(reduction="none")
    aggregation = logits_aggregation_from_config(context.config)

    num_classes = int(context.artifacts.get("num_classes", 10))

    all_targets: List[int] = []
    all_preds: List[int] = []
    prediction_rows: List[Dict[str, Any]] = []
    logits_list: List[np.ndarray] = []

    total = 0
    correct = 0
    loss_sum = 0.0
    global_sample_index = 0

    for batch_idx, (data, target) in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        reset_snn_state(model)
        data, target = prepare_snn_batch(data, target, context.config, context.device)

        logits_t = model(data)
        logits = aggregate_time_logits(logits_t, aggregation)

        losses = criterion(logits, target)
        preds = logits.argmax(dim=1)

        batch_correct = (preds == target).sum().item()
        batch_total = int(target.numel())

        correct += int(batch_correct)
        total += batch_total
        loss_sum += float(losses.sum().item())

        targets_cpu = target.detach().cpu().numpy().astype(int)
        preds_cpu = preds.detach().cpu().numpy().astype(int)
        losses_cpu = losses.detach().cpu().numpy().astype(float)

        if args.save_logits:
            logits_list.append(logits.detach().cpu().numpy())

        for i in range(batch_total):
            prediction_rows.append(
                {
                    "sample_index": global_sample_index,
                    "batch_index": batch_idx,
                    "target": int(targets_cpu[i]),
                    "pred": int(preds_cpu[i]),
                    "correct": int(targets_cpu[i] == preds_cpu[i]),
                    "loss": float(losses_cpu[i]),
                }
            )
            global_sample_index += 1

        all_targets.extend(targets_cpu.tolist())
        all_preds.extend(preds_cpu.tolist())

    avg_loss = loss_sum / max(total, 1)
    acc_percent = 100.0 * correct / max(total, 1)

    confusion = build_confusion_matrix(all_targets, all_preds, num_classes)

    expected_acc = context.artifacts.get("expected_test_acc", None)
    expected_loss = context.artifacts.get("expected_test_loss", None)

    summary: Dict[str, Any] = {
        "eval_name": "eval_00",
        "purpose": "clean_accuracy_sanity_check",
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "command": " ".join(sys.argv),

        "dataset": tag,
        "split": args.split,
        "num_classes": num_classes,
        "num_samples": int(total),

        "accuracy_percent": acc_percent,
        "loss": avg_loss,
        "correct": int(correct),
        "total": int(total),

        "expected_accuracy_percent": expected_acc,
        "expected_loss": expected_loss,
        "accuracy_delta_percent": (
            acc_percent - float(expected_acc) if expected_acc is not None else None
        ),
        "loss_delta": (
            avg_loss - float(expected_loss) if expected_loss is not None else None
        ),

        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_summary": checkpoint_summary(args.checkpoint),

        "input_encoding": context.artifacts.get("input_encoding"),
        "clipped_count_max": context.artifacts.get("clipped_count_max"),
        "time_steps": context.artifacts.get("time_steps"),
        "model_class": context.artifacts.get("model_class"),

        "device": str(context.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),

        "outputs": {
            "summary_json": "summary.json",
            "predictions_csv": "predictions.csv",
            "confusion_matrix_csv": "confusion_matrix.csv",
            "per_class_accuracy_csv": "per_class_accuracy.csv",
            "logits_npy": "logits.npy" if args.save_logits else None,
        },
    }

    save_json(summary, output_dir / "summary.json")
    save_csv_rows(prediction_rows, output_dir / "predictions.csv")
    save_confusion_matrix(confusion, output_dir / "confusion_matrix.csv")
    save_per_class_accuracy(confusion, output_dir / "per_class_accuracy.csv")

    if args.save_logits and logits_list:
        np.save(output_dir / "logits.npy", np.concatenate(logits_list, axis=0))

    config_paths = [Path(args.config)]
    if Path("configs/hardware_hipsa.yaml").exists():
        config_paths.append(Path("configs/hardware_hipsa.yaml"))
    if Path("configs/device_params.yaml").exists():
        config_paths.append(Path("configs/device_params.yaml"))

    copy_config_snapshot(
        config_paths=config_paths,
        output_dir=output_dir,
        snapshot_name="config_snapshot.yaml",
        merged_config=context.config,
    )

    print("=" * 80)
    print("[eval_00] clean accuracy sanity check complete")
    print(f"dataset      : {tag}")
    print(f"split        : {args.split}")
    print(f"samples      : {total}")
    print(f"accuracy     : {acc_percent:.2f}%")
    print(f"loss         : {avg_loss:.6f}")
    print(f"output_dir   : {output_dir}")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    run_clean_eval(args)


if __name__ == "__main__":
    main()