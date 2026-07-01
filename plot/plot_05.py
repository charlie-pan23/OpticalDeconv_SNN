"""
plot_05.py

Plot eval_04 HAPR / ADC pool / MRR sensitivity.

Inputs:
  results/eval_v2/<dataset>/eval_04/adc_pool_sweep.csv
  results/eval_v2/<dataset>/eval_04/hapr_adc_sweep.csv
  results/eval_v2/<dataset>/eval_04/mrr_sensitivity.csv

Outputs:
  plot/results/eval_04/
    plot_05_adc_pool_sweep.png/pdf
    plot_05_hapr_adc_energy_<dataset>.png/pdf
    plot_05_mrr_sensitivity.png/pdf
    plot_05_*_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.result_io import load_csv_rows, save_csv_rows


DATASET_LABELS = {
    "cifar10dvs": "CIFAR10-DVS",
    "dvsgesture": "DVS Gesture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval_04 sensitivity")
    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="plot/results/eval_04", type=str)
    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--dpi", default=300, type=int)
    return parser.parse_args()


def read_rows(input_root: Path, dataset: str, filename: str) -> List[Dict[str, Any]]:
    path = input_root / dataset / "eval_04" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = load_csv_rows(path, parse_numbers=True)
    for row in rows:
        row["dataset"] = dataset
        row["dataset_label"] = DATASET_LABELS.get(dataset, dataset)
    return rows


def plot_adc_pool_sweep(all_rows: List[Dict[str, Any]], output_root: Path, dpi: int) -> None:
    save_csv_rows(all_rows, output_root / "plot_05_adc_pool_sweep_data.csv")

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.0), sharex=True)

    datasets = []
    for row in all_rows:
        ds = row["dataset"]
        if ds not in datasets:
            datasets.append(ds)

    for ds in datasets:
        rows = [r for r in all_rows if r["dataset"] == ds]
        rows = sorted(rows, key=lambda r: int(r["adc_macros"]))

        x = np.array([int(r["adc_macros"]) for r in rows], dtype=float)
        latency = np.array([float(r["latency_us_per_image"]) for r in rows], dtype=float)
        energy = np.array([float(r["energy_uJ_per_image"]) for r in rows], dtype=float)
        util = np.array([float(r["adc_macro_utilization"]) * 100.0 for r in rows], dtype=float)

        label = DATASET_LABELS.get(ds, ds)

        axes[0].plot(x, latency, marker="o", label=label)
        axes[1].plot(x, energy, marker="o", label=label)
        axes[2].plot(x, util, marker="o", label=label)

    axes[0].set_ylabel("Latency (us/image)")
    axes[0].set_title("ADC pool-size sensitivity")
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    axes[0].legend(frameon=False)

    axes[1].set_ylabel("Energy (uJ/image)")
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    axes[2].set_ylabel("ADC utilization (%)")
    axes[2].set_xlabel("ADC macros")
    axes[2].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_root / "plot_05_adc_pool_sweep.png", dpi=dpi)
    fig.savefig(output_root / "plot_05_adc_pool_sweep.pdf")
    plt.close(fig)


def plot_hapr_adc_heatmaps(all_rows: List[Dict[str, Any]], output_root: Path, dpi: int) -> None:
    save_csv_rows(all_rows, output_root / "plot_05_hapr_adc_sweep_data.csv")

    datasets = []
    for row in all_rows:
        ds = row["dataset"]
        if ds not in datasets:
            datasets.append(ds)

    for ds in datasets:
        rows = [r for r in all_rows if r["dataset"] == ds]

        hapr_values = sorted({int(r["hapr_group_size"]) for r in rows})
        adc_values = sorted({int(r["adc_macros"]) for r in rows})

        energy = np.zeros((len(hapr_values), len(adc_values)), dtype=float)
        saturated = np.zeros_like(energy)

        lookup = {
            (int(r["hapr_group_size"]), int(r["adc_macros"])): r
            for r in rows
        }

        for i, hapr in enumerate(hapr_values):
            for j, adc in enumerate(adc_values):
                r = lookup[(hapr, adc)]
                energy[i, j] = float(r["energy_uJ_per_image"])
                saturated[i, j] = float(r["adc_is_saturated"])

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        im = ax.imshow(energy)

        ax.set_xticks(np.arange(len(adc_values)))
        ax.set_yticks(np.arange(len(hapr_values)))
        ax.set_xticklabels([str(x) for x in adc_values])
        ax.set_yticklabels([str(x) for x in hapr_values])

        ax.set_xlabel("ADC macros")
        ax.set_ylabel("HAPR group size")
        ax.set_title(f"HAPR / ADC energy sensitivity: {DATASET_LABELS.get(ds, ds)}")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Energy (uJ/image)")

        for i, hapr in enumerate(hapr_values):
            for j, adc in enumerate(adc_values):
                r = lookup[(hapr, adc)]
                sat = int(r["adc_is_saturated"])
                label = f"{energy[i, j]:.0f}"
                if sat:
                    label += "*"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)

        ax.text(
            0.01,
            -0.18,
            "* indicates ADC-saturated design point",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
        )

        fig.tight_layout()
        fig.savefig(output_root / f"plot_05_hapr_adc_energy_{ds}.png", dpi=dpi)
        fig.savefig(output_root / f"plot_05_hapr_adc_energy_{ds}.pdf")
        plt.close(fig)


def plot_mrr_sensitivity(all_rows: List[Dict[str, Any]], output_root: Path, dpi: int) -> None:
    save_csv_rows(all_rows, output_root / "plot_05_mrr_sensitivity_data.csv")

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)

    datasets = []
    for row in all_rows:
        ds = row["dataset"]
        if ds not in datasets:
            datasets.append(ds)

    for ds in datasets:
        rows = [r for r in all_rows if r["dataset"] == ds]
        rows = sorted(rows, key=lambda r: float(r["mrr_stabilization_mw"]))

        x = np.array([float(r["mrr_stabilization_mw"]) / 1000.0 for r in rows], dtype=float)
        energy = np.array([float(r["energy_uJ_per_image"]) for r in rows], dtype=float)
        power = np.array([float(r["total_power_w"]) for r in rows], dtype=float)

        label = DATASET_LABELS.get(ds, ds)

        axes[0].plot(x, power, marker="o", label=label)
        axes[1].plot(x, energy, marker="o", label=label)

    axes[0].set_ylabel("Total power (W)")
    axes[0].set_title("MRR stabilization sensitivity")
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    axes[0].legend(frameon=False)

    axes[1].set_ylabel("Energy (uJ/image)")
    axes[1].set_xlabel("Added MRR stabilization power (W)")
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_root / "plot_05_mrr_sensitivity.png", dpi=dpi)
    fig.savefig(output_root / "plot_05_mrr_sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    adc_rows: List[Dict[str, Any]] = []
    hapr_rows: List[Dict[str, Any]] = []
    mrr_rows: List[Dict[str, Any]] = []

    for ds in args.datasets:
        adc_rows.extend(read_rows(input_root, ds, "adc_pool_sweep.csv"))
        hapr_rows.extend(read_rows(input_root, ds, "hapr_adc_sweep.csv"))
        mrr_rows.extend(read_rows(input_root, ds, "mrr_sensitivity.csv"))

    plot_adc_pool_sweep(adc_rows, output_root, args.dpi)
    plot_hapr_adc_heatmaps(hapr_rows, output_root, args.dpi)
    plot_mrr_sensitivity(mrr_rows, output_root, args.dpi)

    print(f"[plot_05] saved to {output_root}")


if __name__ == "__main__":
    main()