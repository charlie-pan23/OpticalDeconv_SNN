"""
plot_02.py

Plot HIPSA latency and energy overview from eval_02 saved results.

This script does NOT run inference.
This script does NOT recompute hardware power/performance.

Inputs:
  results/eval_v2/<dataset>/eval_02/latency_energy_summary.json

Outputs:
  plot/results/eval_02/plot_02_latency_energy.png
  plot/results/eval_02/plot_02_latency_energy.pdf
  plot/results/eval_02/plot_02_latency_energy_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.result_io import load_json, save_csv_rows


DATASET_LABELS = {
    "cifar10dvs": "CIFAR10-DVS",
    "dvsgesture": "DVS Gesture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval_02 latency and energy overview")

    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="plot/results/eval_02", type=str)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10dvs", "dvsgesture"],
        type=str,
    )
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--title", default="HIPSA latency and energy", type=str)

    return parser.parse_args()


def load_latency_energy(input_root: Path, dataset: str) -> Dict[str, Any]:
    path = input_root / dataset / "eval_02" / "latency_energy_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_02 latency/energy summary: {path}")
    return load_json(path)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        summary = load_latency_energy(input_root, dataset)

        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "accuracy_percent": float(summary.get("accuracy_percent", 0.0) or 0.0),
                "latency_us_per_image": float(summary.get("latency_us_per_image", 0.0) or 0.0),
                "energy_uJ_per_image": float(summary.get("energy_uJ_per_image", 0.0) or 0.0),
                "throughput_images_per_s": float(summary.get("throughput_images_per_s", 0.0) or 0.0),
                "total_power_w": float(summary.get("total_power_w", 0.0) or 0.0),
                "active_GOPS_per_W": float(summary.get("active_GOPS_per_W", 0.0) or 0.0),
                "active_sop_ratio_percent": float(summary.get("active_sop_ratio", 0.0) or 0.0) * 100.0,
                "active_sop_per_image": float(summary.get("active_sop_per_image", 0.0) or 0.0),
            }
        )

    save_csv_rows(rows, output_root / "plot_02_latency_energy_data.csv")

    labels = [row["dataset_label"] for row in rows]
    latency = np.array([row["latency_us_per_image"] for row in rows], dtype=float)
    energy = np.array([row["energy_uJ_per_image"] for row in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))

    bars1 = ax1.bar(x - width / 2, latency, width, label="Latency")
    ax1.set_ylabel("Latency (us/image)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(args.title)
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, energy, width, label="Energy")
    ax2.set_ylabel("Energy (uJ/image)")

    for bar in bars1:
        value = bar.get_height()
        ax1.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        value = bar.get_height()
        ax2.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    handles = [bars1, bars2]
    labels_legend = ["Latency", "Energy"]
    ax1.legend(handles, labels_legend, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_root / "plot_02_latency_energy.png", dpi=args.dpi)
    fig.savefig(output_root / "plot_02_latency_energy.pdf")
    plt.close(fig)

    print(f"[plot_02] saved to {output_root}")


if __name__ == "__main__":
    main()