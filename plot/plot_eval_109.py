"""Generate DATE-facing Phase 4 evidence figures from eval_109 CSV artifacts.

Outputs both PNG previews and native Matplotlib PDF vector files. No inference is
performed and no source result is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lock(out: Path, root: Path) -> None:
    rows = read_csv(root / "mrr_lock_provenance.csv")
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    for dataset, color in (("cifar10dvs", "#1f77b4"), ("dvsgesture", "#d62728")):
        subset = [r for r in rows if r.get("dataset") == dataset]
        subset.sort(key=lambda r: f(r, "locked_fraction"))
        ax.plot([f(r, "locked_fraction_percent") for r in subset], [f(r, "energy_uJ_per_image") for r in subset], "o-", label=dataset.upper(), color=color)
    ax.set_xlabel("Continuously locked MRR fraction (%)")
    ax.set_ylabel("Energy (uJ/image)")
    ax.set_title("Parameterized MRR lock-power sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out, "fig_eval109_lock_energy")


def plot_config(out: Path, root: Path) -> None:
    rows = read_csv(root / "reporting_case_matrix.csv")
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    for dataset, color in (("cifar10dvs", "#1f77b4"), ("dvsgesture", "#d62728")):
        subset = [r for r in rows if r.get("dataset") == dataset and r.get("case_id") == "configuration_sensitivity"]
        subset.sort(key=lambda r: f(r, "cycles_per_tile_load"))
        ax.plot([f(r, "cycles_per_tile_load") for r in subset], [f(r, "nominal_energy_uJ_per_image") for r in subset], "o-", label=dataset.upper(), color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Modeled tile-load cycles")
    ax.set_ylabel("Nominal energy (uJ/image)")
    ax.set_title("Configuration-cycle sensitivity")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    save(fig, out, "fig_eval109_config_sensitivity")


def plot_accuracy(out: Path, root: Path) -> None:
    rows = read_csv(root / "accuracy_statistics.csv")
    conditions = ["clean", "quantized_clean", "combined"]
    labels = ["Clean", "Quantized\nclean", "Combined"]
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    x = np.arange(len(conditions))
    width = 0.36
    for index, (dataset, color) in enumerate((("cifar10dvs", "#1f77b4"), ("dvsgesture", "#d62728"))):
        subset = {r["condition"]: r for r in rows if r.get("dataset") == dataset}
        means = [float(subset[c]["mean_accuracy_percent"]) for c in conditions]
        errors = [float(subset[c]["ci95_half_width_percentage_points"]) if subset[c]["ci95_half_width_percentage_points"] not in ("", "None") else 0.0 for c in conditions]
        ax.bar(x + (index - 0.5) * width, means, width, yerr=errors, capsize=3, label=dataset.upper(), color=color, alpha=0.88)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Paired replay accuracy statistics (N=5)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out, "fig_eval109_accuracy")


def plot_ablation(out: Path, root: Path) -> None:
    rows = read_csv(root / "orthogonal_ablation_evidence.csv")
    variants = ["adc_pooling_without_hapr", "hapr_with_64_adcs", "lazy_lif_without_hapr", "hapr_a8_without_lazy_lif", "full_hipsa"]
    short = ["ADC pool\nwithout HAPR", "HAPR\n64 ADCs", "Lazy LIF\nwithout HAPR", "HAPR+A8\nwithout Lazy LIF", "Full\nHIPSA"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=False)
    for ax, dataset in zip(axes, ("cifar10dvs", "dvsgesture")):
        subset = {r["variant"]: r for r in rows if r.get("dataset") == dataset}
        values = [float(subset[v]["energy_uJ_per_image"]) for v in variants]
        bars = ax.bar(np.arange(len(variants)), values, color=["#9ecae1", "#9ecae1", "#fdae6b", "#74c476", "#31a354"])
        ax.set_xticks(np.arange(len(variants)), short, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Energy (uJ/image)")
        ax.set_title(dataset.upper())
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Orthogonal mechanism ablation (1% lock-aware accounting)")
    save(fig, out, "fig_eval109_ablation")


def plot_area(out: Path, root: Path) -> None:
    rows = read_csv(root / "area_overhead_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    x = [f(r, "routing_thermal_overhead_multiplier") for r in rows]
    y = [f(r, "adjusted_area_mm2") for r in rows]
    ax.plot(x, y, "o-", color="#756bb1")
    ax.set_xlabel("Routing/thermal overhead multiplier")
    ax.set_ylabel("Area estimate (mm$^2$)")
    ax.set_title("Component-footprint area sensitivity")
    ax.grid(alpha=0.25)
    ax.set_xticks(x, [f"{v:g}x" for v in x])
    save(fig, out, "fig_eval109_area_overhead")


def plot_capacity(out: Path, root: Path) -> None:
    rows = read_csv(root / "g4_a8_capacity_evidence.csv")
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    lanes = f(rows[0], "post_hapr_lanes") if rows else 64
    slots = f(rows[0], "nominal_adc_slots_per_cycle") if rows else 80
    bars = ax.bar(["Post-HAPR\nlanes", "ADC slots\nper cycle"], [lanes, slots], color=["#9ecae1", "#31a354"], width=0.55)
    ax.set_ylabel("Count")
    ax.set_title("G4/A8 architecture-level admission capacity")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, (lanes, slots)):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom")
    ax.text(0.5, max(lanes, slots) * 0.55, f"margin = {slots - lanes:.0f} slots/window", ha="center", bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"})
    save(fig, out, "fig_eval109_capacity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/eval_v10/combined/eval_109")
    parser.add_argument("--output", default="plot/results/eval_109")
    args = parser.parse_args()
    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plot_lock(out, root)
    plot_config(out, root)
    plot_accuracy(out, root)
    plot_ablation(out, root)
    plot_area(out, root)
    plot_capacity(out, root)
    figures = sorted(path.name for path in out.glob("*.pdf"))
    manifest = {
        "eval_name": "eval_109",
        "status": "generated",
        "output_directory": str(out),
        "figures": [
            {"filename": name, "format": "pdf", "vector": True, "source": "matplotlib_native_pdf"}
            for name in figures
        ],
        "preview_pngs": sorted(path.name for path in out.glob("*.png")),
    }
    (root / "publication_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[plot_eval_109] saved PNG/PDF figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

