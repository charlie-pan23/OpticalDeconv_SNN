"""
plot_final_fig7.py

Final Fig. 7:
  Combined/preview:
    fig7_robustness_summary.pdf/png
  LaTeX-ready title-free panel:
    fig7_robustness_summary_panel.pdf/png

Inputs:
  plot/results/eval_05_1/plot_07_aggregated_data_practical_none_hybrid.csv
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

DATASETS = ["cifar10dvs", "dvsgesture"]

REPRESENTATIVE_POINTS = [
    ("ADC 6-bit", "adc_bits", 6.0),
    ("MRR 3%", "mrr", 3.0),
    ("Laser 3%", "laser", 3.0),
    ("WDM -20 dB", "wdm", -20.0),
    ("TIA 2%", "tia", 2.0),
    ("Combined", "combined", 0.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final Fig. 7")
    parser.add_argument(
        "--input-csv",
        default="plot/results/eval_05_1/plot_07_aggregated_data_practical_none_hybrid.csv",
    )
    parser.add_argument("--output-root", default="plot/results/final")
    parser.add_argument("--dpi", default=300, type=int)
    return parser.parse_args()


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def find_row(rows: List[Dict[str, Any]], dataset: str, ptype: str, x_value: float) -> Dict[str, Any]:
    candidates = [
        r for r in rows
        if r.get("dataset") == dataset and r.get("perturbation_type") == ptype
    ]

    if ptype == "combined":
        if candidates:
            return candidates[0]
        raise KeyError(f"Missing combined row for {dataset}")

    for r in candidates:
        if abs(to_float(r.get("x_value")) - x_value) < 1e-6:
            return r

    raise KeyError(f"Missing row dataset={dataset}, type={ptype}, x={x_value}")


def build_plot_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plot_rows: List[Dict[str, Any]] = []
    for label, ptype, x_value in REPRESENTATIVE_POINTS:
        for ds in DATASETS:
            r = find_row(rows, ds, ptype, x_value)
            plot_rows.append(
                {
                    "dataset": ds,
                    "dataset_label": DATASET_LABELS[ds],
                    "condition": label,
                    "perturbation_type": ptype,
                    "x_value": x_value,
                    "accuracy_drop_percent": to_float(r.get("accuracy_drop_mean")),
                    "energy_change_percent": to_float(r.get("energy_change_percent_mean")),
                    "num_seeds": int(to_float(r.get("num_seeds"), 1)),
                }
            )
    return plot_rows


def draw_robustness_panel(
    ax,
    plot_rows: List[Dict[str, Any]],
    show_title: bool = True,
) -> None:
    x = np.arange(len(REPRESENTATIVE_POINTS))
    width = 0.36

    for i, ds in enumerate(DATASETS):
        vals = []
        for label, _, _ in REPRESENTATIVE_POINTS:
            hit = [r for r in plot_rows if r["dataset"] == ds and r["condition"] == label]
            vals.append(hit[0]["accuracy_drop_percent"])

        ax.bar(x + (i - 0.5) * width, vals, width, label=DATASET_LABELS[ds])

    values = [r["accuracy_drop_percent"] for r in plot_rows]
    ymin = min(values)
    ymax = max(values)

    bottom = min(-1.2, ymin * 1.25)
    top = max(3.0, ymax * 1.22)
    top = min(top, 4.0)

    ax.axhline(0, linewidth=0.8)
    ax.axhline(2, linestyle=":", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in REPRESENTATIVE_POINTS], rotation=15, ha="right")
    ax.set_ylabel("Accuracy drop from clean (%)")
    if show_title:
        ax.set_title("Representative device-specific robustness", pad=34)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_ylim(bottom, top)

    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        handlelength=1.1,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def save_panel(
    output_root: Path,
    name: str,
    plot_rows: List[Dict[str, Any]],
    show_title: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    draw_robustness_panel(ax, plot_rows, show_title=show_title)
    fig.tight_layout()
    fig.savefig(output_root / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_root / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(args.input_csv, parse_numbers=True)
    plot_rows = build_plot_rows(rows)

    save_csv_rows(plot_rows, output_root / "fig7_robustness_summary_data.csv")

    # Combined/preview version with title.
    save_panel(
        output_root,
        "fig7_robustness_summary",
        plot_rows,
        show_title=True,
        dpi=args.dpi,
    )

    # LaTeX-ready version without internal title.
    save_panel(
        output_root,
        "fig7_robustness_summary_panel",
        plot_rows,
        show_title=False,
        dpi=args.dpi,
    )

    print(f"[fig7] saved combined and title-free panel to {output_root}")


if __name__ == "__main__":
    main()