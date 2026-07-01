"""
plot_03.py

Plot HIPSA device-calibrated power breakdown from eval_02 saved results.

This script does NOT run inference.
This script does NOT recompute hardware power/performance.

Inputs:
  results/eval_v2/<dataset>/eval_02/power_breakdown.csv

Outputs:
  plot/results/eval_02/plot_03_power_breakdown.png
  plot/results/eval_02/plot_03_power_breakdown.pdf
  plot/results/eval_02/plot_03_power_breakdown_data.csv
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


COMPONENT_ORDER = [
    "cw_laser",
    "mrr_stabilization",
    "leakage_misc_io",
    "binary_modulator_driver",
    "photodiodes",
    "tia",
    "comparators",
    "hapr_selection_proxy",
    "adc_pool",
    "sram_register_files",
    "noc_bus_controller_clock",
    "digital_lif_update",
]


COMPONENT_LABELS = {
    "cw_laser": "CW laser",
    "mrr_stabilization": "MRR stabilization",
    "leakage_misc_io": "Leakage / misc. I/O",
    "binary_modulator_driver": "Binary modulator",
    "photodiodes": "Photodiodes",
    "tia": "TIA",
    "comparators": "Comparators",
    "hapr_selection_proxy": "HAPR selection",
    "adc_pool": "ADC pool",
    "sram_register_files": "SRAM / registers",
    "noc_bus_controller_clock": "NoC / control",
    "digital_lif_update": "Digital LIF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval_02 power breakdown")

    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="plot/results/eval_02", type=str)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10dvs", "dvsgesture"],
        type=str,
    )
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--title", default="HIPSA device-calibrated power breakdown", type=str)
    parser.add_argument(
        "--drop-zero",
        action="store_true",
        help="Hide components whose power is zero for all datasets.",
    )

    return parser.parse_args()


def load_power_breakdown(input_root: Path, dataset: str) -> Dict[str, float]:
    path = input_root / dataset / "eval_02" / "power_breakdown.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_02 power breakdown: {path}")

    rows = load_csv_rows(path, parse_numbers=True)

    power: Dict[str, float] = {}
    for row in rows:
        component = str(row.get("component", ""))
        power_w = float(row.get("power_w", 0.0) or 0.0)
        power[component] = power_w

    return power


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    power_by_dataset: Dict[str, Dict[str, float]] = {
        dataset: load_power_breakdown(input_root, dataset)
        for dataset in args.datasets
    }

    all_components = list(COMPONENT_ORDER)

    for dataset, power in power_by_dataset.items():
        for component in power.keys():
            if component not in all_components:
                all_components.append(component)

    if args.drop_zero:
        kept = []
        for component in all_components:
            total = sum(power_by_dataset[d].get(component, 0.0) for d in args.datasets)
            if total > 0:
                kept.append(component)
        all_components = kept

    plot_rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        total_power = sum(power_by_dataset[dataset].values())

        for component in all_components:
            power_w = float(power_by_dataset[dataset].get(component, 0.0))
            plot_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "component": component,
                    "component_label": COMPONENT_LABELS.get(component, component),
                    "power_w": power_w,
                    "power_mw": power_w * 1000.0,
                    "share_percent": 100.0 * power_w / total_power if total_power > 0 else 0.0,
                }
            )

    save_csv_rows(plot_rows, output_root / "plot_03_power_breakdown_data.csv")

    labels = [DATASET_LABELS.get(d, d) for d in args.datasets]
    y = np.arange(len(labels))

    fig_height = max(4.2, 1.0 + 0.85 * len(labels))
    fig, ax = plt.subplots(figsize=(8.4, fig_height))

    left = np.zeros(len(labels), dtype=float)

    for component in all_components:
        values = np.array(
            [power_by_dataset[d].get(component, 0.0) for d in args.datasets],
            dtype=float,
        )

        if np.all(values == 0):
            continue

        ax.barh(
            y,
            values,
            left=left,
            label=COMPONENT_LABELS.get(component, component),
        )
        left += values

    for i, total in enumerate(left):
        ax.annotate(
            f"{total:.2f} W",
            xy=(total, y[i]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Power (W)")
    ax.set_title(args.title)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_root / "plot_03_power_breakdown.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_root / "plot_03_power_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_03] saved to {output_root}")


if __name__ == "__main__":
    main()