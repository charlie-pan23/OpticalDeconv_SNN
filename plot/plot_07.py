"""
plot_07_final.py

Final paper-style eval05 robustness plots.

Main idea:
- ADC precision, WDM crosstalk, and combined stress are discrete settings:
  use grouped bars.
- MRR, laser, and TIA/HAPR noise are continuous-like perturbation levels:
  use line plots.
- Main paper default hides error bars for readability.
- Appendix can enable SEM error bars.

Recommended main-paper command:
  python -m plot.plot_07_final --view practical --errorbar none

Recommended appendix command:
  python -m plot.plot_07_final --view full --errorbar sem
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.result_io import load_csv_rows, save_csv_rows


DATASET_LABELS = {
    "cifar10dvs": "CIFAR10-DVS",
    "dvsgesture": "DVS Gesture",
}

PERTURBATION_ORDER = [
    "adc_bits",
    "mrr",
    "laser",
    "wdm",
    "tia",
    "combined",
]

DISCRETE_TYPES = {"adc_bits", "wdm", "combined"}

PERTURBATION_TITLES = {
    "adc_bits": "ADC precision",
    "mrr": "MRR transmission perturbation",
    "laser": "Laser intensity fluctuation",
    "wdm": "WDM crosstalk",
    "tia": "TIA/HAPR output noise",
    "combined": "Combined stress",
}

X_LABELS = {
    "adc_bits": "ADC precision (bits)",
    "mrr": "MRR perturbation sigma (%)",
    "laser": "Laser fluctuation sigma (%)",
    "wdm": "Crosstalk level (dB)",
    "tia": "TIA/HAPR noise sigma (%)",
    "combined": "Stress case",
}

ACC_YLIMS_PRACTICAL = {
    "adc_bits": (-3, 28),
    "mrr": (-0.5, 4.0),
    "laser": (-0.5, 2.2),
    "wdm": (-2.0, 3.0),
    "tia": (-0.8, 1.2),
    "combined": (-0.5, 2.5),
}

ENERGY_YLIMS_PRACTICAL = {
    "adc_bits": (-1, 6),
    "mrr": (-0.3, 0.8),
    "laser": (-0.2, 0.4),
    "wdm": (-2.5, 0.3),
    "tia": (-0.2, 1.4),
    "combined": (-0.6, 2.0),
}

ACC_YLIMS_FULL = {
    "adc_bits": (-5, 65),
    "mrr": (-2, 10),
    "laser": (-2, 4),
    "wdm": (-4, 8),
    "tia": (-2, 4),
    "combined": (-2, 5),
}

ENERGY_YLIMS_FULL = {
    "adc_bits": (-8, 22),
    "mrr": (-2, 4),
    "laser": (-1, 2.5),
    "wdm": (-8, 2),
    "tia": (-1, 3),
    "combined": (-3, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final eval05 robustness plot")
    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="plot/results/eval_05_final", type=str)
    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--view", default="practical", choices=["practical", "full"])
    parser.add_argument("--errorbar", default="none", choices=["none", "sem", "std"])
    parser.add_argument("--dpi", default=300, type=int)
    return parser.parse_args()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1)))


def sem(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return std(values) / math.sqrt(len(values))


def load_dataset_rows(input_root: Path, dataset: str) -> List[Dict[str, Any]]:
    path = input_root / dataset / "eval_05" / "robustness_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval05 robustness summary: {path}")

    rows = load_csv_rows(path, parse_numbers=True)
    for row in rows:
        row["dataset"] = dataset
        row["dataset_label"] = DATASET_LABELS.get(dataset, dataset)
        row["seed"] = to_int(row.get("seed"), 0)
        row["level"] = to_float(row.get("level"), 0.0)
        row["accuracy_percent"] = to_float(row.get("accuracy_percent"), 0.0)
        row["energy_uJ_per_image"] = to_float(row.get("energy_uJ_per_image"), 0.0)
        row["adc_macro_utilization"] = to_float(row.get("adc_macro_utilization"), 0.0)
    return rows


def attach_baselines(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for row in rows:
        if str(row.get("perturbation_type")) == "clean":
            clean_by_key[(str(row["dataset"]), to_int(row["seed"], 0))] = row

    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        out = dict(row)
        clean = clean_by_key.get((str(row["dataset"]), to_int(row["seed"], 0)))

        if clean is None:
            out["accuracy_drop_percent_recomputed"] = ""
            out["energy_change_percent"] = ""
        else:
            clean_acc = to_float(clean.get("accuracy_percent"))
            clean_energy = to_float(clean.get("energy_uJ_per_image"))
            acc = to_float(row.get("accuracy_percent"))
            energy = to_float(row.get("energy_uJ_per_image"))

            out["clean_accuracy_percent"] = clean_acc
            out["accuracy_drop_percent_recomputed"] = clean_acc - acc
            out["clean_energy_uJ_per_image"] = clean_energy
            out["energy_norm_to_clean"] = energy / clean_energy if clean_energy > 0 else ""
            out["energy_change_percent"] = (
                (energy / clean_energy - 1.0) * 100.0 if clean_energy > 0 else ""
            )

        out_rows.append(out)

    return out_rows


def x_value_for_plot(row: Dict[str, Any]) -> float:
    ptype = str(row.get("perturbation_type"))
    level = to_float(row.get("level"))

    if ptype in {"mrr", "laser", "tia"}:
        return level * 100.0
    if ptype == "adc_bits":
        return level
    if ptype == "wdm":
        return level
    if ptype == "combined":
        return 0.0
    return level


def make_synthetic_zero_rows(rows: List[Dict[str, Any]], datasets: List[str]) -> List[Dict[str, Any]]:
    synthetic: List[Dict[str, Any]] = []

    for dataset in datasets:
        clean_rows = [
            r for r in rows
            if r["dataset"] == dataset and str(r.get("perturbation_type")) == "clean"
        ]

        for ptype in ["mrr", "laser", "tia"]:
            for r in clean_rows:
                synthetic.append(
                    {
                        **r,
                        "perturbation_type": ptype,
                        "level": 0.0,
                        "level_label": "0",
                        "accuracy_drop_percent_recomputed": 0.0,
                        "energy_change_percent": 0.0,
                    }
                )

    return rows + synthetic


def keep_for_view(row: Dict[str, Any], view: str) -> bool:
    if view == "full":
        return True

    ptype = str(row.get("perturbation_type"))
    x = x_value_for_plot(row)

    if ptype == "adc_bits":
        return x >= 5
    if ptype == "mrr":
        return x <= 3
    if ptype == "laser":
        return x <= 3
    if ptype == "wdm":
        return x <= -20
    if ptype == "tia":
        return x <= 2
    if ptype == "combined":
        return True

    return True


def aggregate_rows(rows: List[Dict[str, Any]], view: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, float, str], List[Dict[str, Any]]] = defaultdict(list)
    excluded: List[Dict[str, Any]] = []

    for row in rows:
        ptype = str(row.get("perturbation_type"))
        if ptype == "clean":
            continue

        if not keep_for_view(row, view):
            excluded.append(row)
            continue

        key = (
            str(row.get("dataset")),
            ptype,
            x_value_for_plot(row),
            str(row.get("level_label", row.get("level", ""))),
        )
        groups[key].append(row)

    aggregated: List[Dict[str, Any]] = []

    for (dataset, ptype, x_value, level_label), items in groups.items():
        acc = [to_float(x.get("accuracy_percent")) for x in items]
        drop = [
            to_float(x.get("accuracy_drop_percent_recomputed"))
            for x in items
            if x.get("accuracy_drop_percent_recomputed") != ""
        ]
        energy_change = [
            to_float(x.get("energy_change_percent"))
            for x in items
            if x.get("energy_change_percent") != ""
        ]
        energy = [to_float(x.get("energy_uJ_per_image")) for x in items]
        adc_util = [to_float(x.get("adc_macro_utilization")) for x in items]

        aggregated.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "perturbation_type": ptype,
                "level_label": level_label,
                "x_value": x_value,
                "num_seeds": len(items),

                "accuracy_mean": mean(acc),
                "accuracy_std": std(acc),
                "accuracy_sem": sem(acc),

                "accuracy_drop_mean": mean(drop),
                "accuracy_drop_std": std(drop),
                "accuracy_drop_sem": sem(drop),

                "energy_uJ_mean": mean(energy),
                "energy_uJ_std": std(energy),
                "energy_uJ_sem": sem(energy),

                "energy_change_percent_mean": mean(energy_change),
                "energy_change_percent_std": std(energy_change),
                "energy_change_percent_sem": sem(energy_change),

                "adc_utilization_mean": mean(adc_util),
                "adc_utilization_std": std(adc_util),
                "adc_utilization_sem": sem(adc_util),
            }
        )

    aggregated = sorted(
        aggregated,
        key=lambda r: (
            PERTURBATION_ORDER.index(r["perturbation_type"])
            if r["perturbation_type"] in PERTURBATION_ORDER
            else 999,
            r["dataset"],
            r["x_value"],
        ),
    )
    return aggregated, excluded


def select_rows(rows: List[Dict[str, Any]], dataset: str, ptype: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["dataset"] == dataset and r["perturbation_type"] == ptype]


def yerr_key(metric_key: str, errorbar: str) -> str | None:
    if errorbar == "none":
        return None
    if errorbar == "sem":
        if metric_key == "accuracy_drop_mean":
            return "accuracy_drop_sem"
        if metric_key == "energy_change_percent_mean":
            return "energy_change_percent_sem"
    if errorbar == "std":
        if metric_key == "accuracy_drop_mean":
            return "accuracy_drop_std"
        if metric_key == "energy_change_percent_mean":
            return "energy_change_percent_std"
    return None


def configure_axis(ax: plt.Axes, ptype: str, categorical: bool = False) -> None:
    ax.set_title(PERTURBATION_TITLES.get(ptype, ptype), fontsize=10)
    ax.set_xlabel(X_LABELS.get(ptype, "Level"), fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.tick_params(axis="both", labelsize=8)

    # Important:
    # ADC precision and WDM crosstalk are plotted as categorical grouped bars.
    # Their x positions are 0, 1, 2, ... so we must NOT overwrite ticks with
    # physical values such as 5/6/8 or -30/-25/-20 here.
    if categorical:
        return

    if ptype == "adc_bits":
        ax.set_xticks([5, 6, 8])
    elif ptype == "wdm":
        ax.set_xticks([-30, -25, -20])
        ax.set_xticklabels(["-30", "-25", "-20"])


def plot_metric_grid(
    *,
    aggregated: List[Dict[str, Any]],
    datasets: List[str],
    output_root: Path,
    metric_key: str,
    ylabel: str,
    filename: str,
    title: str,
    ylims: Dict[str, Tuple[float, float]],
    errorbar: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8))
    axes_flat = list(axes.flatten())
    err_key = yerr_key(metric_key, errorbar)

    for ax, ptype in zip(axes_flat, PERTURBATION_ORDER):
        if ptype in DISCRETE_TYPES:
            x_categories = []
            for row in aggregated:
                if row["perturbation_type"] == ptype:
                    label = row["level_label"] if ptype == "combined" else str(int(row["x_value"])) if float(row["x_value"]).is_integer() else str(row["x_value"])
                    if label not in x_categories:
                        x_categories.append(label)

            if ptype == "combined":
                x_categories = [DATASET_LABELS.get(ds, ds) for ds in datasets]

                values = []
                errors = []
                for dataset in datasets:
                    rows = select_rows(aggregated, dataset, ptype)
                    if rows:
                        row = rows[0]
                        values.append(to_float(row.get(metric_key)))
                        errors.append(to_float(row.get(err_key)) if err_key else 0.0)
                    else:
                        values.append(0.0)
                        errors.append(0.0)

                x = np.arange(len(x_categories))
                ax.bar(
                    x,
                    values,
                    yerr=errors if err_key and any(e > 0 for e in errors) else None,
                    capsize=3,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(x_categories, rotation=15, ha="right")
            else:
                # Grouped bars: categories are ADC bits or WDM dB.
                rows_by_dataset = {
                    dataset: select_rows(aggregated, dataset, ptype)
                    for dataset in datasets
                }

                x_values_sorted = sorted(
                    {
                        row["x_value"]
                        for rows in rows_by_dataset.values()
                        for row in rows
                    }
                )

                if ptype == "adc_bits":
                    category_labels = [f"{int(x)}b" for x in x_values_sorted]
                elif ptype == "wdm":
                    category_labels = [f"{int(x)} dB" for x in x_values_sorted]
                else:
                    category_labels = [str(x) for x in x_values_sorted]

                x = np.arange(len(x_values_sorted))
                width = 0.8 / max(len(datasets), 1)

                for i, dataset in enumerate(datasets):
                    rows = rows_by_dataset[dataset]
                    lookup = {row["x_value"]: row for row in rows}

                    vals = []
                    errs = []
                    for xv in x_values_sorted:
                        row = lookup.get(xv)
                        vals.append(to_float(row.get(metric_key)) if row else np.nan)
                        errs.append(to_float(row.get(err_key)) if row and err_key else 0.0)

                    offset = (i - (len(datasets) - 1) / 2.0) * width
                    ax.bar(
                        x + offset,
                        vals,
                        width,
                        yerr=errs if err_key and any(e > 0 for e in errs) else None,
                        capsize=3,
                        label=DATASET_LABELS.get(dataset, dataset),
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(category_labels)
                ax.set_xlim(-0.6, len(x_values_sorted) - 0.4)
        else:
            for dataset in datasets:
                rows = select_rows(aggregated, dataset, ptype)
                rows = sorted(rows, key=lambda r: float(r["x_value"]))
                if not rows:
                    continue

                x = np.array([to_float(r["x_value"]) for r in rows], dtype=float)
                y = np.array([to_float(r.get(metric_key)) for r in rows], dtype=float)

                if err_key:
                    yerr = np.array([to_float(r.get(err_key)) for r in rows], dtype=float)
                else:
                    yerr = None

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr if yerr is not None and np.any(yerr > 0) else None,
                    marker="o",
                    linewidth=1.7,
                    capsize=3,
                    label=DATASET_LABELS.get(dataset, dataset),
                )

        configure_axis(ax, ptype, categorical=(ptype in DISCRETE_TYPES))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, linewidth=0.8)

        if metric_key == "accuracy_drop_mean":
            ax.axhline(2, linestyle=":", linewidth=0.9)
            ax.axhline(5, linestyle=":", linewidth=0.9)
        else:
            ax.axhline(5, linestyle=":", linewidth=0.9)
            ax.axhline(-5, linestyle=":", linewidth=0.9)

        if ptype in ylims:
            ax.set_ylim(*ylims[ptype])

    axes_flat[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(output_root / f"{filename}.png", dpi=dpi)
    fig.savefig(output_root / f"{filename}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        raw_rows.extend(load_dataset_rows(input_root, dataset))

    with_baseline = attach_baselines(raw_rows)
    with_synthetic = make_synthetic_zero_rows(with_baseline, args.datasets)
    aggregated, excluded = aggregate_rows(with_synthetic, args.view)

    suffix = f"{args.view}_{args.errorbar}_hybrid"

    save_csv_rows(with_baseline, output_root / f"plot_07_raw_with_baseline_{suffix}.csv")
    save_csv_rows(aggregated, output_root / f"plot_07_aggregated_data_{suffix}.csv")
    save_csv_rows(excluded, output_root / f"plot_07_excluded_stress_points_{suffix}.csv")

    acc_ylims = ACC_YLIMS_PRACTICAL if args.view == "practical" else ACC_YLIMS_FULL
    energy_ylims = ENERGY_YLIMS_PRACTICAL if args.view == "practical" else ENERGY_YLIMS_FULL

    plot_metric_grid(
        aggregated=aggregated,
        datasets=args.datasets,
        output_root=output_root,
        metric_key="accuracy_drop_mean",
        ylabel="Accuracy drop (%)",
        filename=f"plot_07_accuracy_drop_grid_{suffix}",
        title="Device-specific robustness: accuracy drop from clean",
        ylims=acc_ylims,
        errorbar=args.errorbar,
        dpi=args.dpi,
    )

    plot_metric_grid(
        aggregated=aggregated,
        datasets=args.datasets,
        output_root=output_root,
        metric_key="energy_change_percent_mean",
        ylabel="Energy change from clean (%)",
        filename=f"plot_07_energy_change_grid_{suffix}",
        title="Device-specific robustness: energy change",
        ylims=energy_ylims,
        errorbar=args.errorbar,
        dpi=args.dpi,
    )

    print(f"[plot_07_final] saved to {output_root}")
    print(f"[plot_07_final] view={args.view}, errorbar={args.errorbar}")


if __name__ == "__main__":
    main()