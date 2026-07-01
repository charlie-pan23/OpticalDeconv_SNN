"""
plot_06.py

Plot CPU/GPU/HIPSA comparison.

Inputs:
  results/eval_v2/<dataset>/eval_06/runtime_summary.json
  results/eval_v2/<dataset>/eval_04/hapr_adc_sweep.csv
  fallback: results/eval_v2/<dataset>/eval_02/latency_energy_summary.json

Outputs:
  plot/results/eval_06/plot_06_latency_comparison.png/pdf
  plot/results/eval_06/plot_06_energy_comparison.png/pdf, only if energy is available
  plot/results/eval_06/plot_06_comparison_data.csv
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CPU/GPU/HIPSA comparison")
    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="plot/results/eval_06", type=str)
    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--hipsa-hapr", default=16, type=int)
    parser.add_argument("--hipsa-adc", default=32, type=int)
    parser.add_argument("--dpi", default=300, type=int)
    return parser.parse_args()


def extract_device_results(runtime_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in runtime_summary.get("device_results", []):
        if not item.get("available", False):
            continue

        device = item.get("device")
        s = item.get("summary", {})

        latency_ms = s.get("mean_latency_ms_per_image")
        if latency_ms is None:
            latency_ms = s.get("mean_latency_ms")

        energy_mj = s.get("energy_mJ_per_image")

        rows.append(
            {
                "platform": "CPU" if device == "cpu" else "GPU",
                "device": device,
                "latency_ms_per_image": float(latency_ms) if latency_ms is not None else None,
                "energy_mJ_per_image": float(energy_mj) if energy_mj is not None else None,
                "accuracy_percent": float(item.get("accuracy_percent_timed", 0.0) or 0.0),
            }
        )

    return rows


def load_hipsa_from_eval04(input_root: Path, dataset: str, hapr: int, adc: int) -> Optional[Dict[str, Any]]:
    path = input_root / dataset / "eval_04" / "hapr_adc_sweep.csv"
    if not path.exists():
        return None

    rows = load_csv_rows(path, parse_numbers=True)
    for row in rows:
        if int(row.get("hapr_group_size", -1)) == hapr and int(row.get("adc_macros", -1)) == adc:
            return {
                "platform": f"HIPSA HAPR{hapr}/ADC{adc}",
                "device": "hipsa",
                "latency_ms_per_image": float(row["latency_us_per_image"]) / 1000.0,
                "energy_mJ_per_image": float(row["energy_uJ_per_image"]) / 1000.0,
                "accuracy_percent": float(row.get("accuracy_percent", 0.0) or 0.0),
                "source": str(path),
            }

    return None


def load_hipsa_from_eval02(input_root: Path, dataset: str) -> Dict[str, Any]:
    path = input_root / dataset / "eval_02" / "latency_energy_summary.json"
    d = load_json(path)

    return {
        "platform": "HIPSA eval02",
        "device": "hipsa",
        "latency_ms_per_image": float(d["latency_us_per_image"]) / 1000.0,
        "energy_mJ_per_image": float(d["energy_uJ_per_image"]) / 1000.0,
        "accuracy_percent": float(d.get("accuracy_percent", 0.0) or 0.0),
        "source": str(path),
    }


def load_dataset_rows(input_root: Path, dataset: str, hapr: int, adc: int) -> List[Dict[str, Any]]:
    runtime_path = input_root / dataset / "eval_06" / "runtime_summary.json"
    if not runtime_path.exists():
        raise FileNotFoundError(f"Missing eval06 runtime summary: {runtime_path}")

    runtime = load_json(runtime_path)
    rows = extract_device_results(runtime)

    hipsa = load_hipsa_from_eval04(input_root, dataset, hapr, adc)
    if hipsa is None:
        hipsa = load_hipsa_from_eval02(input_root, dataset)

    rows.append(hipsa)

    for row in rows:
        row["dataset"] = dataset
        row["dataset_label"] = DATASET_LABELS.get(dataset, dataset)

    return rows


def plot_latency(rows: List[Dict[str, Any]], output_root: Path, dpi: int) -> None:
    datasets = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    platforms = []
    for row in rows:
        if row["platform"] not in platforms:
            platforms.append(row["platform"])

    x = np.arange(len(datasets))
    width = 0.8 / max(len(platforms), 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for i, platform in enumerate(platforms):
        vals = []
        for ds in datasets:
            match = [r for r in rows if r["dataset"] == ds and r["platform"] == platform]
            vals.append(match[0]["latency_ms_per_image"] if match else np.nan)

        offset = (i - (len(platforms) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, label=platform)

    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms/image, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets])
    ax.set_title("Software baseline vs HIPSA")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_root / "plot_06_latency_comparison.png", dpi=dpi)
    fig.savefig(output_root / "plot_06_latency_comparison.pdf")
    plt.close(fig)


def plot_energy(rows: List[Dict[str, Any]], output_root: Path, dpi: int) -> None:
    energy_rows = [r for r in rows if r.get("energy_mJ_per_image") is not None]
    if not energy_rows:
        print("[plot_06] no energy data available; skipping energy plot")
        return

    datasets = []
    for row in energy_rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    platforms = []
    for row in energy_rows:
        if row["platform"] not in platforms:
            platforms.append(row["platform"])

    x = np.arange(len(datasets))
    width = 0.8 / max(len(platforms), 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for i, platform in enumerate(platforms):
        vals = []
        for ds in datasets:
            match = [r for r in energy_rows if r["dataset"] == ds and r["platform"] == platform]
            vals.append(match[0]["energy_mJ_per_image"] if match else np.nan)

        offset = (i - (len(platforms) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, label=platform)

    ax.set_yscale("log")
    ax.set_ylabel("Energy (mJ/image, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets])
    ax.set_title("Software baseline vs HIPSA energy")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_root / "plot_06_energy_comparison.png", dpi=dpi)
    fig.savefig(output_root / "plot_06_energy_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        rows.extend(load_dataset_rows(input_root, dataset, args.hipsa_hapr, args.hipsa_adc))

    save_csv_rows(rows, output_root / "plot_06_comparison_data.csv")
    plot_latency(rows, output_root, args.dpi)
    plot_energy(rows, output_root, args.dpi)

    print(f"[plot_06] saved to {output_root}")


if __name__ == "__main__":
    main()