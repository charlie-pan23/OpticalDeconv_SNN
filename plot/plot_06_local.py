"""
plot_06_full.py

Final CPU/GPU/HIPSA comparison figure for eval06.

Inputs:
  Software CPU/GPU:
    <software-root>/<dataset>/eval_06/runtime_summary.json

  HIPSA:
    <hipsa-root>/<dataset>/eval_04/hapr_adc_sweep.csv
    fallback:
    <hipsa-root>/<dataset>/eval_02/latency_energy_summary.json

Outputs:
  <output-root>/plot_06_full_latency_energy.png
  <output-root>/plot_06_full_latency_energy.pdf
  <output-root>/plot_06_full_comparison_data.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.result_io import load_csv_rows, load_json, save_csv_rows


DATASET_LABELS = {
    "cifar10dvs": "CIFAR10-DVS",
    "dvsgesture": "DVS Gesture",
}

PLATFORM_ORDER = ["CPU", "GPU", "HIPSA"]

PLATFORM_LABELS = {
    "CPU": "CPU",
    "GPU": "GPU",
    "HIPSA": "HIPSA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final eval06 CPU/GPU/HIPSA latency-energy plot")

    parser.add_argument("--software-root", default="results/eval_v2_local_energy", type=str)
    parser.add_argument("--hipsa-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="plot/results/eval_06_final", type=str)

    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--hipsa-hapr", default=16, type=int)
    parser.add_argument("--hipsa-adc", default=32, type=int)

    parser.add_argument("--dpi", default=600, type=int)
    parser.add_argument("--title", default="CPU/GPU software baselines vs HIPSA", type=str)

    parser.add_argument("--latency-top-factor", default=4.5, type=float)
    parser.add_argument("--energy-top-factor", default=4.5, type=float)

    parser.add_argument(
        "--label-lift",
        default=1.18,
        type=float,
        help="Multiplicative label lift above each bar on log-scale axes.",
    )

    return parser.parse_args()


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def load_software_rows(software_root: Path, dataset: str) -> List[Dict[str, Any]]:
    path = software_root / dataset / "eval_06" / "runtime_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval06 runtime summary: {path}")

    obj = load_json(path)
    rows: List[Dict[str, Any]] = []

    for item in obj.get("device_results", []):
        if not item.get("available", False):
            continue

        device = str(item.get("device", ""))
        summary = item.get("summary", {})

        if device == "cpu":
            platform_name = "CPU"
        elif device.startswith("cuda"):
            platform_name = "GPU"
        else:
            platform_name = device.upper()

        latency_ms = safe_float(summary.get("mean_latency_ms_per_image"))
        if latency_ms is None:
            latency_ms = safe_float(summary.get("mean_latency_ms"))

        energy_mj = safe_float(summary.get("energy_mJ_per_image"))
        active_power_w = safe_float(summary.get("active_power_w"))

        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "platform": platform_name,
                "source": str(path),
                "latency_ms_per_image": latency_ms,
                "energy_mJ_per_image": energy_mj,
                "active_power_w": active_power_w,
                "accuracy_percent": safe_float(item.get("accuracy_percent_timed"), 0.0),
                "note": "measured_software_baseline",
            }
        )

    return rows


def load_hipsa_from_eval04(
    hipsa_root: Path,
    dataset: str,
    hapr: int,
    adc: int,
) -> Optional[Dict[str, Any]]:
    path = hipsa_root / dataset / "eval_04" / "hapr_adc_sweep.csv"
    if not path.exists():
        return None

    rows = load_csv_rows(path, parse_numbers=True)

    for row in rows:
        if int(row.get("hapr_group_size", -1)) == int(hapr) and int(row.get("adc_macros", -1)) == int(adc):
            return {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "platform": "HIPSA",
                "source": str(path),
                "latency_ms_per_image": float(row["latency_us_per_image"]) / 1000.0,
                "energy_mJ_per_image": float(row["energy_uJ_per_image"]) / 1000.0,
                "active_power_w": float(row.get("total_power_w", 0.0)),
                "accuracy_percent": float(row.get("accuracy_percent", 0.0)),
                "note": f"HIPSA_HAPR{hapr}_ADC{adc}",
            }

    return None


def load_hipsa_from_eval02(hipsa_root: Path, dataset: str) -> Dict[str, Any]:
    path = hipsa_root / dataset / "eval_02" / "latency_energy_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing HIPSA eval02 fallback: {path}")

    obj = load_json(path)

    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "platform": "HIPSA",
        "source": str(path),
        "latency_ms_per_image": float(obj["latency_us_per_image"]) / 1000.0,
        "energy_mJ_per_image": float(obj["energy_uJ_per_image"]) / 1000.0,
        "active_power_w": float(obj.get("total_power_w", 0.0)),
        "accuracy_percent": float(obj.get("accuracy_percent", 0.0)),
        "note": "HIPSA_eval02_fallback",
    }


def load_all_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    software_root = Path(args.software_root)
    hipsa_root = Path(args.hipsa_root)

    rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        rows.extend(load_software_rows(software_root, dataset))

        hipsa = load_hipsa_from_eval04(
            hipsa_root=hipsa_root,
            dataset=dataset,
            hapr=args.hipsa_hapr,
            adc=args.hipsa_adc,
        )
        if hipsa is None:
            hipsa = load_hipsa_from_eval02(hipsa_root, dataset)

        rows.append(hipsa)

    return rows


def ordered_platforms(rows: List[Dict[str, Any]]) -> List[str]:
    present = {row["platform"] for row in rows}
    ordered = [p for p in PLATFORM_ORDER if p in present]

    for p in sorted(present):
        if p not in ordered:
            ordered.append(p)

    return ordered


def get_value(rows: List[Dict[str, Any]], dataset: str, platform: str, key: str) -> float:
    for row in rows:
        if row["dataset"] == dataset and row["platform"] == platform:
            value = row.get(key)
            if value is None or value == "":
                return float("nan")
            return float(value)
    return float("nan")


def log_ylim(values: List[float], top_factor: float) -> tuple[float, float]:
    positive = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    if not positive:
        return 1e-3, 1.0

    ymin = min(positive)
    ymax = max(positive)

    return ymin / 2.0, ymax * top_factor


def format_latency(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2g}"
    return f"{value:.3g}"


def format_energy_sci(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        return ""

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)

    if mantissa >= 9.95:
        mantissa = 1.0
        exponent += 1

    return rf"${mantissa:.1f}\!\times\!10^{{{exponent}}}$"


def annotate_bar_top(
    ax: plt.Axes,
    x_center: float,
    height: float,
    text: str,
    lift: float,
    fontsize: float = 7.4,
) -> None:
    if not math.isfinite(height) or height <= 0:
        return

    y = height * lift

    ax.text(
        x_center,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        rotation=0,
        clip_on=False,
    )


def plot_panel(
    ax: plt.Axes,
    rows: List[Dict[str, Any]],
    datasets: List[str],
    platforms: List[str],
    key: str,
    ylabel: str,
    title: str,
    top_factor: float,
    label_type: str,
    label_lift: float,
) -> None:
    x = np.arange(len(datasets))
    width = 0.76 / max(len(platforms), 1)

    all_values: List[float] = []

    for i, platform in enumerate(platforms):
        values = [get_value(rows, dataset, platform, key) for dataset in datasets]
        all_values.extend([v for v in values if math.isfinite(v)])

        offset = (i - (len(platforms) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=PLATFORM_LABELS.get(platform, platform),
        )

        for bar, value in zip(bars, values):
            if not math.isfinite(value) or value <= 0:
                continue

            if label_type == "energy":
                label = format_energy_sci(value)
                fontsize = 7.2
            else:
                label = format_latency(value)
                fontsize = 7.5

            annotate_bar_top(
                ax=ax,
                x_center=bar.get_x() + bar.get_width() / 2.0,
                height=value,
                text=label,
                lift=label_lift,
                fontsize=fontsize,
            )

    ax.set_yscale("log")
    ax.set_ylim(*log_ylim(all_values, top_factor=top_factor))

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(dataset, dataset) for dataset in datasets])

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_all_rows(args)
    save_csv_rows(rows, output_root / "plot_06_full_comparison_data.csv")

    datasets = list(args.datasets)
    platforms = ordered_platforms(rows)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    plot_panel(
        ax=axes[0],
        rows=rows,
        datasets=datasets,
        platforms=platforms,
        key="latency_ms_per_image",
        ylabel="Latency (ms/image, log scale)",
        title="Latency",
        top_factor=args.latency_top_factor,
        label_type="latency",
        label_lift=args.label_lift,
    )

    plot_panel(
        ax=axes[1],
        rows=rows,
        datasets=datasets,
        platforms=platforms,
        key="energy_mJ_per_image",
        ylabel="Energy (mJ/image, log scale)",
        title="Energy",
        top_factor=args.energy_top_factor,
        label_type="energy",
        label_lift=args.label_lift,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(args.title, y=0.985, fontsize=12)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=len(labels),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.2,
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.15,
        left=0.09,
        right=0.985,
        wspace=0.32,
    )

    fig.savefig(output_root / "plot_06_full_latency_energy.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_root / "plot_06_full_latency_energy.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_06_full] saved to {output_root}")
    print(f"[plot_06_full] data saved to {output_root / 'plot_06_full_comparison_data.csv'}")


if __name__ == "__main__":
    main()