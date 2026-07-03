"""
plot_final_fig6.py

Final Fig. 6:
  Combined:
    fig6_power_adc_hapr.pdf/png
  LaTeX-ready separate panels:
    fig6a_power_breakdown.pdf/png
    fig6b_hapr_adc_energy.pdf/png
    fig6c_adc_utilization.pdf/png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eval.eval_02 import (
    derive_architecture_counts,
    estimate_adc_requests,
    estimate_power,
    estimate_timing,
    model_adc_pool,
)
from utils.result_io import load_csv_rows, load_json, load_yaml, save_csv_rows


DATASET_LABELS = {
    "cifar10dvs": "CIFAR10-DVS",
    "dvsgesture": "DVS Gesture",
}

DATASETS = ["cifar10dvs", "dvsgesture"]

DESIGN_POINTS = [
    ("Default\n8/16", 8, 16),
    ("Conservative\n8/64", 8, 64),
    ("Balanced\n16/32", 16, 32),
    ("Aggressive\n32/16", 32, 16),
]

POWER_GROUPS = {
    "Laser": ["cw_laser"],
    "Modulation": ["binary_modulator_driver"],
    "O/E frontend": ["photodiodes", "tia", "comparators", "hapr_selection_proxy", "adc_pool"],
    "Memory/digital": ["sram_register_files", "noc_bus_controller_clock", "digital_lif_update"],
    "Other static": ["leakage_misc_io", "mrr_stabilization"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final Fig. 6")
    parser.add_argument("--eval-root", default="results/eval_v2")
    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
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


def panel_legend(ax, ncol: int = 2) -> None:
    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        handlelength=1.1,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def compute_balanced_power_rows(
    dataset: str,
    eval_root: Path,
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    activity = load_json(eval_root / dataset / "eval_01" / "summary.json")

    counts = derive_architecture_counts(
        hardware_cfg,
        hapr_group_size_override=16,
        adc_macros_override=32,
    )

    adc_requests = estimate_adc_requests(activity, hapr_group_size=16)
    timing_base = estimate_timing(activity, hardware_cfg, adc_pool=None)
    adc_pool = model_adc_pool(adc_requests, timing_base, counts)
    timing = estimate_timing(activity, hardware_cfg, adc_pool=adc_pool)

    power = estimate_power(
        activity_summary=activity,
        timing=timing,
        adc_pool=adc_pool,
        counts=counts,
        device_cfg=device_cfg,
        modulator_activity_source="mvm_input_activity",
        adc_power_mode="activity_scaled",
        mrr_stabilization_mw=0.0,
    )

    rows = []
    for r in power["power_rows"]:
        item = dict(r)
        item["dataset"] = dataset
        item["dataset_label"] = DATASET_LABELS[dataset]
        rows.append(item)
    return rows


def group_power_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = []

    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        comp_power = {r["component"]: to_float(r["power_w"]) for r in ds_rows}
        total = sum(comp_power.values())

        for group_name, comps in POWER_GROUPS.items():
            value = sum(comp_power.get(c, 0.0) for c in comps)
            grouped.append(
                {
                    "dataset": ds,
                    "dataset_label": DATASET_LABELS[ds],
                    "power_group": group_name,
                    "power_w": value,
                    "share_percent": 100.0 * value / total if total > 0 else 0.0,
                }
            )
    return grouped


def load_design_points(eval_root: Path) -> List[Dict[str, Any]]:
    rows = []

    for ds in DATASETS:
        path = eval_root / ds / "eval_04" / "hapr_adc_sweep.csv"
        raw = load_csv_rows(path, parse_numbers=True)
        lookup = {
            (int(r["hapr_group_size"]), int(r["adc_macros"])): r
            for r in raw
        }

        for label, hapr, adc in DESIGN_POINTS:
            r = lookup[(hapr, adc)]
            rows.append(
                {
                    "dataset": ds,
                    "dataset_label": DATASET_LABELS[ds],
                    "design": label,
                    "hapr_group_size": hapr,
                    "adc_macros": adc,
                    "latency_us_per_image": to_float(r["latency_us_per_image"]),
                    "energy_uJ_per_image": to_float(r["energy_uJ_per_image"]),
                    "total_power_w": to_float(r["total_power_w"]),
                    "adc_macro_utilization_percent": to_float(r["adc_macro_utilization"]) * 100.0,
                    "adc_is_saturated": int(r["adc_is_saturated"]),
                }
            )
    return rows


def draw_power_breakdown_panel(
    ax,
    grouped_power: List[Dict[str, Any]],
    show_title: bool = True,
) -> None:
    y = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS), dtype=float)

    for group_name in POWER_GROUPS.keys():
        vals = []
        for ds in DATASETS:
            hit = [r for r in grouped_power if r["dataset"] == ds and r["power_group"] == group_name]
            vals.append(hit[0]["power_w"] if hit else 0.0)

        ax.barh(y, vals, left=left, label=group_name)
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=90, va="center")
    ax.set_xlabel("Power (W)")
    if show_title:
        ax.set_title("(a) Balanced power breakdown", pad=42)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_xlim(0, max(left) * 1.18)

    for i, total in enumerate(left):
        ax.text(total + 0.03, i, f"{total:.2f} W", va="center", fontsize=8)

    ax.legend(
        frameon=False,
        fontsize=6.8,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        handlelength=1.0,
        columnspacing=0.7,
        borderaxespad=0.0,
    )


def draw_energy_panel(
    ax,
    design_rows: List[Dict[str, Any]],
    show_title: bool = True,
) -> None:
    design_labels = [x[0] for x in DESIGN_POINTS]
    x = np.arange(len(design_labels))
    width = 0.36

    for i, ds in enumerate(DATASETS):
        vals = []
        for label, _, _ in DESIGN_POINTS:
            hit = [r for r in design_rows if r["dataset"] == ds and r["design"] == label]
            vals.append(hit[0]["energy_uJ_per_image"])

        ax.bar(x + (i - 0.5) * width, vals, width, label=DATASET_LABELS[ds])

    ax.set_xticks(x)
    ax.set_xticklabels(design_labels, fontsize=8)
    ax.set_ylabel("Energy (uJ/image)")
    if show_title:
        ax.set_title("(b) HAPR/ADC energy", pad=34)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_ylim(0, 265)
    panel_legend(ax, ncol=2)


def draw_adc_util_panel(
    ax,
    design_rows: List[Dict[str, Any]],
    show_title: bool = True,
) -> None:
    design_labels = [x[0] for x in DESIGN_POINTS]
    x = np.arange(len(design_labels))
    width = 0.36

    for i, ds in enumerate(DATASETS):
        vals = []
        for label, _, _ in DESIGN_POINTS:
            hit = [r for r in design_rows if r["dataset"] == ds and r["design"] == label]
            vals.append(hit[0]["adc_macro_utilization_percent"])

        ax.bar(x + (i - 0.5) * width, vals, width, label=DATASET_LABELS[ds])

    ax.axhline(100, linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(design_labels, fontsize=8)
    ax.set_ylabel("ADC utilization (%)")
    ax.set_ylim(0, 115)
    if show_title:
        ax.set_title("(c) ADC backend pressure", pad=34)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    panel_legend(ax, ncol=2)


def save_single_panel(
    output_root: Path,
    name: str,
    draw_fn,
    figsize=(4.4, 3.6),
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.tight_layout()
    fig.savefig(output_root / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_root / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    eval_root = Path(args.eval_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    hardware_cfg = load_yaml(args.hardware)
    device_cfg = load_yaml(args.device_params)

    raw_power_rows = []
    for ds in DATASETS:
        raw_power_rows.extend(compute_balanced_power_rows(ds, eval_root, hardware_cfg, device_cfg))

    grouped_power = group_power_rows(raw_power_rows)
    design_rows = load_design_points(eval_root)

    save_csv_rows(grouped_power, output_root / "fig6_power_grouped_data.csv")
    save_csv_rows(design_rows, output_root / "fig6_design_points_data.csv")

    # Combined preview figure with internal titles.
    fig = plt.figure(figsize=(14.2, 4.1))

    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.35, 2.0],
        wspace=0.15,
    )

    right = outer[0, 1].subgridspec(
        1,
        2,
        wspace=0.20,
    )

    axes = [
        fig.add_subplot(outer[0, 0]),
        fig.add_subplot(right[0, 0]),
        fig.add_subplot(right[0, 1]),
    ]

    draw_power_breakdown_panel(axes[0], grouped_power, show_title=True)
    draw_energy_panel(axes[1], design_rows, show_title=True)
    draw_adc_util_panel(axes[2], design_rows, show_title=True)

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.20,
        top=0.80,
    )

    fig.savefig(output_root / "fig6_power_adc_hapr.pdf", bbox_inches="tight")
    fig.savefig(output_root / "fig6_power_adc_hapr.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # LaTeX-ready separate panels without internal titles.
    save_single_panel(
        output_root,
        "fig6a_power_breakdown",
        lambda ax: draw_power_breakdown_panel(ax, grouped_power, show_title=False),
        figsize=(5.2, 3.45),
        dpi=args.dpi,
    )
    save_single_panel(
        output_root,
        "fig6b_hapr_adc_energy",
        lambda ax: draw_energy_panel(ax, design_rows, show_title=False),
        figsize=(4.0, 3.45),
        dpi=args.dpi,
    )
    save_single_panel(
        output_root,
        "fig6c_adc_utilization",
        lambda ax: draw_adc_util_panel(ax, design_rows, show_title=False),
        figsize=(4.0, 3.45),
        dpi=args.dpi,
    )

    print(f"[fig6] saved combined and separate panels to {output_root}")


if __name__ == "__main__":
    main()