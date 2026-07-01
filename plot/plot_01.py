"""
plot_01.py

Plot workload activity from eval_01 saved summaries.

This script does NOT run inference.
It only reads:
  results/eval_v2/<dataset>/eval_01/summary.json

Output:
  plot/results/eval_01/workload_activity.png
  plot/results/eval_01/workload_activity.pdf
  plot/results/eval_01/workload_activity_plot_data.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt
import numpy as np


DATASET_LABELS = {
    "cifar10dvs": "CIFAR10-DVS",
    "dvsgesture": "DVS Gesture",
}


METRICS = [
    ("model_input_activity", "Input activity"),
    ("active_sop_ratio", "Active SOP ratio"),
    ("lif_spike_activity", "LIF spike activity"),
    ("adc_request_activity", "ADC request activity"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval_01 workload activity")
    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="plot/results/eval_01", type=str)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10dvs", "dvsgesture"],
        type=str,
    )
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--title", default="Workload activity trace", type=str)
    return parser.parse_args()


def load_summary(input_root: Path, dataset: str) -> Dict[str, Any]:
    path = input_root / dataset / "eval_01" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"eval_01 summary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}")
    return obj


def write_plot_data(rows: List[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "metric", "value_percent", "raw_value"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {
        dataset: load_summary(input_root, dataset)
        for dataset in args.datasets
    }

    plot_rows: List[Dict[str, Any]] = []
    for dataset, summary in summaries.items():
        for key, label in METRICS:
            raw = float(summary.get(key, 0.0) or 0.0)
            plot_rows.append(
                {
                    "dataset": dataset,
                    "metric": label,
                    "raw_value": raw,
                    "value_percent": raw * 100.0,
                }
            )

    write_plot_data(plot_rows, output_root / "workload_activity_plot_data.csv")

    dataset_labels = [DATASET_LABELS.get(d, d) for d in args.datasets]
    metric_labels = [label for _, label in METRICS]

    values = np.zeros((len(args.datasets), len(METRICS)), dtype=float)
    for i, dataset in enumerate(args.datasets):
        summary = summaries[dataset]
        for j, (key, _) in enumerate(METRICS):
            values[i, j] = float(summary.get(key, 0.0) or 0.0) * 100.0

    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(args.datasets), 1)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    for i, dataset_label in enumerate(dataset_labels):
        offset = (i - (len(args.datasets) - 1) / 2.0) * width
        bars = ax.bar(x + offset, values[i], width, label=dataset_label)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(args.title)
    ax.set_ylabel("Activity (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_ylim(0, max(5.0, min(100.0, values.max() * 1.25)))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(output_root / "workload_activity.png", dpi=args.dpi)
    fig.savefig(output_root / "workload_activity.pdf")
    plt.close(fig)

    print(f"[plot_01] saved to {output_root}")


if __name__ == "__main__":
    main()