"""
plot_final_fig5.py

Final Fig. 5:
  Combined:
    fig5_activity_performance.pdf/png
  LaTeX-ready separate panels:
    fig5a_activity.pdf/png
    fig5b_latency.pdf/png
    fig5c_energy.pdf/png

Inputs:
  results/eval_v2/<dataset>/eval_01/summary.json
  plot/results/eval_06_local/plot_06_comparison_data.csv
"""

from __future__ import annotations

import argparse
import csv
import re
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

DATASETS = ["cifar10dvs", "dvsgesture"]
PLATFORM_ORDER = ["CPU", "GPU", "HIPSA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final Fig. 5")
    parser.add_argument("--eval-root", default="results/eval_v2")
    parser.add_argument(
        "--eval06-csv",
        default="plot/results/eval_06_local/plot_06_comparison_data.csv",
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


def normalize_platform(p: str) -> str:
    s = str(p).lower()
    if "cpu" in s:
        return "CPU"
    if "gpu" in s or "cuda" in s:
        return "GPU"
    if "hipsa" in s:
        return "HIPSA"
    return str(p)


def read_csv_robust(path: Path) -> List[Dict[str, str]]:
    """Read normal CSV; fallback for accidentally flattened CSV."""

    text = path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    if rows:
        return rows

    first_match = re.search(r"\s(?=(cifar10dvs|dvsgesture),)", text)
    if first_match is None:
        return []

    header_text = text[: first_match.start()].strip()
    data_text = text[first_match.start():].strip()
    header = next(csv.reader([header_text]))

    chunks = re.split(r"\s+(?=(?:cifar10dvs|dvsgesture),)", data_text)
    out: List[Dict[str, str]] = []
    for chunk in chunks:
        values = next(csv.reader([chunk]))
        if len(values) == len(header):
            out.append(dict(zip(header, values)))
    return out


def load_activity(eval_root: Path) -> List[Dict[str, Any]]:
    metrics = [
        ("model_input_activity", "Input"),
        ("mvm_input_activity", "MVM input"),
        ("active_sop_ratio", "Active SOP"),
        ("lif_spike_activity", "LIF spike"),
        ("adc_request_activity", "ADC request"),
    ]

    rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        path = eval_root / ds / "eval_01" / "summary.json"
        data = load_json(path)
        for key, label in metrics:
            rows.append(
                {
                    "dataset": ds,
                    "dataset_label": DATASET_LABELS.get(ds, ds),
                    "metric": label,
                    "value_percent": to_float(data.get(key)) * 100.0,
                }
            )
    return rows


def load_eval06_rows(path: Path) -> List[Dict[str, Any]]:
    raw = read_csv_robust(path)
    rows: List[Dict[str, Any]] = []

    for r in raw:
        platform = normalize_platform(r.get("platform", r.get("device", "")))
        if platform not in PLATFORM_ORDER:
            continue

        ds = r.get("dataset")
        rows.append(
            {
                "dataset": ds,
                "dataset_label": DATASET_LABELS.get(ds, ds),
                "platform": platform,
                "latency_ms_per_image": to_float(r.get("latency_ms_per_image")),
                "energy_mJ_per_image": to_float(r.get("energy_mJ_per_image")),
            }
        )
    return rows


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


def draw_activity_panel(
    ax,
    activity_rows: List[Dict[str, Any]],
    show_title: bool = True,
) -> None:
    activity_metrics = ["Input", "MVM input", "Active SOP", "LIF spike", "ADC request"]
    x = np.arange(len(activity_metrics))
    width = 0.36

    for i, ds in enumerate(DATASETS):
        vals = []
        for m in activity_metrics:
            hit = [r for r in activity_rows if r["dataset"] == ds and r["metric"] == m]
            vals.append(hit[0]["value_percent"] if hit else np.nan)

        ax.bar(x + (i - 0.5) * width, vals, width, label=DATASET_LABELS[ds])

    ax.set_xticks(x)
    ax.set_xticklabels(activity_metrics, rotation=22, ha="right")
    ax.set_ylabel("Activity (%)")
    if show_title:
        ax.set_title("(a) Workload activity", pad=34)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_ylim(0, 88)
    panel_legend(ax, ncol=2)


def draw_platform_panel(
    ax,
    rows: List[Dict[str, Any]],
    value_key: str,
    ylabel: str,
    title: str,
    log: bool = False,
    show_title: bool = True,
) -> None:
    x = np.arange(len(DATASETS))
    width = 0.23

    lookup = {(r["dataset"], r["platform"]): r for r in rows}

    for i, platform in enumerate(PLATFORM_ORDER):
        vals = []
        for ds in DATASETS:
            item = lookup.get((ds, platform))
            vals.append(item[value_key] if item else np.nan)

        offset = (i - (len(PLATFORM_ORDER) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, label=platform)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=0)
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title, pad=34)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)

    if log:
        ax.set_yscale("log")

    panel_legend(ax, ncol=3)


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

    activity_rows = load_activity(eval_root)
    perf_rows = load_eval06_rows(Path(args.eval06_csv))

    save_csv_rows(
        activity_rows + perf_rows,
        output_root / "fig5_activity_performance_data.csv",
    )

    # Combined preview figure with internal titles.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    draw_activity_panel(axes[0], activity_rows, show_title=True)
    draw_platform_panel(
        axes[1],
        perf_rows,
        "latency_ms_per_image",
        "Latency (ms/image, log)",
        "(b) Latency",
        log=True,
        show_title=True,
    )
    draw_platform_panel(
        axes[2],
        perf_rows,
        "energy_mJ_per_image",
        "Energy (mJ/image, log)",
        "(c) Energy",
        log=True,
        show_title=True,
    )

    fig.tight_layout()
    fig.savefig(output_root / "fig5_activity_performance.pdf", bbox_inches="tight")
    fig.savefig(output_root / "fig5_activity_performance.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # LaTeX-ready separate panels without internal titles.
    save_single_panel(
        output_root,
        "fig5a_activity",
        lambda ax: draw_activity_panel(ax, activity_rows, show_title=False),
        figsize=(4.3, 3.6),
        dpi=args.dpi,
    )
    save_single_panel(
        output_root,
        "fig5b_latency",
        lambda ax: draw_platform_panel(
            ax,
            perf_rows,
            "latency_ms_per_image",
            "Latency (ms/image, log)",
            "(b) Latency",
            log=True,
            show_title=False,
        ),
        figsize=(4.1, 3.6),
        dpi=args.dpi,
    )
    save_single_panel(
        output_root,
        "fig5c_energy",
        lambda ax: draw_platform_panel(
            ax,
            perf_rows,
            "energy_mJ_per_image",
            "Energy (mJ/image, log)",
            "(c) Energy",
            log=True,
            show_title=False,
        ),
        figsize=(4.1, 3.6),
        dpi=args.dpi,
    )

    print(f"[fig5] saved combined and separate panels to {output_root}")


if __name__ == "__main__":
    main()