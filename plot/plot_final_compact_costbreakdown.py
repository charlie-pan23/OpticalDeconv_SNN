from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


plt.rcParams.update({
    "font.weight": "bold",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    # 刻度线设置
    "xtick.direction": "in",      # x轴刻度朝内
    "ytick.direction": "in",      # y轴刻度朝内
    # "xtick.top": True,            # 显示上边框刻度
    # "ytick.right": True,          # 显示右边框刻度
    # "xtick.minor.visible": True,  # 显示x轴次刻度
    # "ytick.minor.visible": True,  # 显示y轴次刻度
    "xtick.major.width": 0.6,     # 主刻度线宽
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,     # 次刻度线宽
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.5,      # 主刻度长度
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,      # 次刻度长度
    "ytick.minor.size": 2.0,
})

DATASETS = ["cifar10dvs", "dvsgesture"]
DATASET_LABELS = {
    "cifar10dvs": "CIFAR10-DVS",
    "dvsgesture": "DVS Gesture",
}
DATASET_COLORS = {
    "cifar10dvs": "#4C78A8",
    "dvsgesture": "#59A14F",
}

# ---------------------------------------------------------------------------
# Per-figure color configuration.
# Change colors here only. This lets each final figure use its own palette
# without affecting the other panels.
# ---------------------------------------------------------------------------
FIGURE_COLORS = {
    "fig5a": {
        "component": {
            # "Laser/link": "#3EAAD0",
            # "Modulation": "#9CCEE7",
            # "O/E + ADC": "#BBD49A",
            # "Digital": "#F6D353",
            # "Other static": "#F78B15",
            "Laser/link": "#2E76B5",
            "Modulation": "#81B1D9",
            "O/E + ADC": "#A4C990",
            "Digital": "#E5C870",
            "Other static": "#D97C50",
        },
        "grid": "#D9D9D9",
        "total_text": "#333333",
    },
    "fig5b": {
        "dataset": {
            "cifar10dvs": "#6B5B95",
            "dvsgesture": "#E07A5F",
        },
        "major_grid": "#D9D9D9",
        "minor_grid": "#EEEEEE",
    },
    "fig6a": {
        "dataset": {
            "cifar10dvs": "#6B5B95",
            "dvsgesture": "#E07A5F",
        },
        "balanced_highlight": "#F2F2F2",
        "zero_line": "#666666",
        "grid": "#D9D9D9",
        "note": "#555555",
    },
    "fig6b": {
        "cmap": ["#7FA6CC", "#C7DCEA", "#FAFAFA", "#F2C8C0", "#D8837C"],
        "cell_grid": "white",
        "cell_text": "#222222",
    },
}

# The four exported figure names. The script writes both .pdf and .png.
FINAL_FIGURE_NAMES = {
    "fig5a": "Fig5a_System_Cost_Breakdown",
    "fig5b": "Fig5b_Latency_Energy_Positioning",
    "fig6a": "Fig6a_Energy_Saving_vs_Default",
    "fig6b": "Fig6b_Device_Robustness_Heatmap",
}


def figure_dataset_color(fig_key: str, dataset: str) -> str:
    """Return the dataset color for a specific final figure."""
    return FIGURE_COLORS[fig_key]["dataset"].get(
        dataset,
        DATASET_COLORS.get(dataset, "#333333"),
    )


def component_color(component: str) -> str:
    """Return the Fig.5a component color."""
    return FIGURE_COLORS["fig5a"]["component"].get(component, "#999999")


PLATFORM_ORDER = ["CPU", "GPU", "HIPSA"]

# 改 5b 点标注位置就在这里改。
# (dx, dy) 单位是 points；dx 越大越往右，dy 越大越往上。
POINT_LABEL_OFFSETS = {
    ("cifar10dvs", "CPU"): (-50, 2),
    ("cifar10dvs", "GPU"): (-52, -2),
    ("cifar10dvs", "HIPSA"): (7, -12),
    ("dvsgesture", "CPU"): (-2, -20),
    ("dvsgesture", "GPU"): (-10, -22),
    ("dvsgesture", "HIPSA"): (-7, 12),
}

DESIGN_ORDER = [
    ("Default", 8, 16),
    ("Conserv.", 8, 64),
    ("Balanced", 16, 32),
    ("Aggress.", 32, 16),
]

ROBUSTNESS_POINTS = [
    ("ADC\n6b", "adc_bits", 6.0),
    ("MRR\n3%", "mrr", 3.0),
    ("Laser\n3%", "laser", 3.0),
    ("WDM\n-20 dB", "wdm", -20.0),
    ("TIA\n2%", "tia", 2.0),
    ("Combined", "combined", 0.0),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x, default=np.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def legend_above(ax: plt.Axes, ncol: int, y: float = 1.14) -> None:
    ax.legend(
        frameon=False,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        columnspacing=0.9,
        handlelength=1.4,
        borderaxespad=0.0,
    )


def annotate_bars(ax: plt.Axes, bars, kind: str = "percent") -> None:
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        if kind == "percent":
            text = f"{h:.1f}"
        elif kind == "energy_uj":
            text = f"{h:.0f}" if h >= 100 else f"{h:.1f}"
        else:
            text = f"{h:.2f}"
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.8,
            rotation=0,
            clip_on=False,
        )


def fmt_latency_ms(v: float) -> str:
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def fmt_energy_mj(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.2f}k"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def soft_diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "soft_diverging",
        FIGURE_COLORS["fig6b"]["cmap"],
        N=256,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cost_breakdown() -> pd.DataFrame:
    """Return Fig.5a system cost breakdown data.

    Power rows are workload-dependent balanced G16/A32 power accounting.
    The area row is modeled macro area, not full chip layout area.
    """

    rows = []

    # Grouped balanced-design power values in W.
    # O/E + ADC combines photodetectors, HAPR/TIA/comparator/selection,
    # and ADC macro pool.
    power_data = {
        "cifar10dvs": {
            "row_label": "CIFAR10-DVS\npower",
            "total_label": "2.236 W",
            "components": {
                "Laser/link": 1.473,
                "Modulation": 0.073,
                "O/E + ADC": 0.502,
                "Digital": 0.098,
                "Other static": 0.090,
            },
        },
        "dvsgesture": {
            "row_label": "DVS Gesture\npower",
            "total_label": "2.404 W",
            "components": {
                "Laser/link": 1.473,
                "Modulation": 0.060,
                "O/E + ADC": 0.627,
                "Digital": 0.154,
                "Other static": 0.090,
            },
        },
    }

    for dataset, item in power_data.items():
        total = sum(item["components"].values())
        for component, value in item["components"].items():
            rows.append({
                "row_type": "power",
                "dataset": dataset,
                "row_label": item["row_label"],
                "component": component,
                "value": value,
                "share_percent": 100.0 * value / total,
                "total_value": total,
                "total_label": item["total_label"],
            })

    # Modeled macro area in mm^2.
    # This is the explicitly modeled macro/device area from the primitive table.
    # It is not a full post-layout chip area.
    area_components = {
        "Laser/link": 0.120,
        "Modulation": 1.331,
        "O/E + ADC": 0.161,
        "Digital": 0.000,
        "Other static": 0.000,
    }
    area_total = sum(area_components.values())

    for component, value in area_components.items():
        rows.append({
            "row_type": "area",
            "dataset": "modeled_macro_area",
            "row_label": "Modeled\nmacro area",
            "component": component,
            "value": value,
            "share_percent": 100.0 * value / area_total if area_total > 0 else 0.0,
            "total_value": area_total,
            "total_label": f"{area_total:.3f} mm$^2$",
        })

    return pd.DataFrame(rows)


def load_activity(input_root: Path) -> pd.DataFrame:
    metrics = [
        ("model_input_activity", "Input"),
        ("mvm_input_activity", "MVM\ninput"),
        ("active_sop_ratio", "Active\nSOP"),
        ("lif_spike_activity", "LIF\nspike"),
        ("adc_request_activity", "ADC\nrequest"),
    ]

    rows = []
    for ds in DATASETS:
        obj = load_json(input_root / ds / "eval_01" / "summary.json")
        for key, label in metrics:
            value = obj.get(key, obj.get("adc_element_request_activity"))
            rows.append({
                "dataset": ds,
                "metric": label,
                "value_percent": 100.0 * to_float(value),
            })
    return pd.DataFrame(rows)


def load_cost(cost_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cost_csv)
    df["platform"] = df["platform"].astype(str).str.upper().replace({"CUDA": "GPU"})
    return df


def load_design(input_root: Path) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        df = pd.read_csv(input_root / ds / "eval_04" / "hapr_adc_sweep.csv")
        df["hapr_group_size"] = df["hapr_group_size"].astype(float).astype(int)
        df["adc_macros"] = df["adc_macros"].astype(float).astype(int)

        for name, g, a in DESIGN_ORDER:
            hit = df[(df["hapr_group_size"] == g) & (df["adc_macros"] == a)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append({
                "dataset": ds,
                "design": name,
                "energy_uJ_per_image": to_float(r["energy_uJ_per_image"]),
                "adc_util_percent": 100.0 * to_float(r["adc_macro_utilization"]),
                "adc_is_saturated": int(to_float(r["adc_is_saturated"], 0)),
            })
    return pd.DataFrame(rows)


def load_robustness(robustness_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(robustness_csv)
    rows = []

    for ds in DATASETS:
        ds_rows = df[df["dataset"].astype(str).str.lower() == ds]
        for label, ptype, xval in ROBUSTNESS_POINTS:
            cand = ds_rows[ds_rows["perturbation_type"].astype(str).str.lower() == ptype]
            if ptype != "combined":
                cand = cand[np.isclose(cand["x_value"].astype(float), xval, atol=1e-6)]
            if cand.empty:
                raise KeyError(f"Missing robustness row: {ds}, {ptype}, {xval}")
            r = cand.iloc[0]
            rows.append({
                "dataset": ds,
                "condition": label,
                "accuracy_drop_pp": to_float(r["accuracy_drop_mean"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot panels
# ---------------------------------------------------------------------------

def draw_fig5a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Fig.5a: system power and modeled macro-area breakdown.

    This is a normalized stacked horizontal bar chart.
    Each row sums to 100%, with the absolute total shown on the right.
    """

    component_order = [
        "Laser/link",
        "Modulation",
        "O/E + ADC",
        "Digital",
        "Other static",
    ]

    row_order = [
        "CIFAR10-DVS\npower",
        "DVS Gesture\npower",
        "Modeled\nmacro area",
    ]

    y = np.arange(len(row_order), dtype=float)
    left = np.zeros(len(row_order), dtype=float)

    row_to_index = {label: i for i, label in enumerate(row_order)}

    inside_threshold = 1.0  # >=15% label inside the segment
    small_fontsize = 6.0
    inside_fontsize = 6.2

    # external label offsets for small segments (< 15%)
    # you can tune these later if needed
    outside_dx = -6  # points, horizontal shift of external label
    outside_dy_up = 20  # points, vertical shift upward
    outside_dy_down = -20  # points, vertical shift downward

    for component in component_order:
        vals = np.zeros(len(row_order), dtype=float)

        for _, r in df[df["component"] == component].iterrows():
            row_label = r["row_label"]
            vals[row_to_index[row_label]] = float(r["share_percent"])

        current_left = left.copy()

        ax.barh(
            y,
            vals,
            left=current_left,
            height=0.55,
            color=component_color(component),
            edgecolor="none",
            label=component,
            zorder=2,
        )

        # annotate every segment
        for idx, v in enumerate(vals):
            if not np.isfinite(v) or v <= 0:
                continue

            x_center = current_left[idx] + v / 2.0
            y_center = y[idx]

            text_color = "white" if component == "Laser/link" else "#222222"

            if v >= inside_threshold:
                ax.text(
                    x_center,
                    y_center,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=inside_fontsize,
                    color=text_color,
                    rotation=0,
                    zorder=6,
                    clip_on=False,
                )
            else:
                ax.text(
                    x_center,
                    y_center,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=small_fontsize,
                    color=text_color,
                    # rotation=30,
                    zorder=7,
                    clip_on=False,
                )

        left += vals

    # Absolute totals on the right.
    for row_label in row_order:
        idx = row_to_index[row_label]
        hit = df[df["row_label"] == row_label]
        if hit.empty:
            continue
        total_label = hit["total_label"].iloc[0]
        ax.text(
            102.2,
            idx,
            total_label,
            ha="left",
            va="center",
            fontsize=7.0,
            color=FIGURE_COLORS["fig5a"]["total_text"],
            # rotation=25,
            # rotation_mode="anchor",
            clip_on=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        row_order,
        rotation=30,
        ha="right",
        va="center",
        # rotation_mode="anchor",
        fontsize=7.4,
    )
    ax.tick_params(axis="y", pad=2)
    ax.invert_yaxis()

    ax.set_xlim(-0.1, 112)
    ax.set_xlabel("Share of total (%)")

    ax.grid(axis="x", color=FIGURE_COLORS["fig5a"]["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_zorder(10)
    ax.spines["bottom"].set_zorder(10)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    legend_handles = [
        Patch(facecolor=component_color(c), edgecolor="none", label=c)
        for c in component_order
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        handlelength=0.85,
        columnspacing=0.45,
        handletextpad=0.30,
        borderaxespad=0.0,
        fontsize=6.3,
    )


def draw_fig5b(ax: plt.Axes, df: pd.DataFrame) -> None:
    all_x, all_y = [], []

    for ds in DATASETS:
        sub = df[df["dataset"] == ds].copy()
        sub["rank"] = sub["platform"].map({p: i for i, p in enumerate(PLATFORM_ORDER)})
        sub = sub.sort_values("rank")

        for _, r in sub.iterrows():
            platform = str(r["platform"])
            x = to_float(r["latency_ms_per_image"])
            y = to_float(r["energy_mJ_per_image"])
            all_x.append(x)
            all_y.append(y)

            ax.scatter(
                x,
                y,
                s=60,
                marker="o",
                color=figure_dataset_color("fig5b", ds),
                edgecolor="none",
                alpha=0.92,
                label=DATASET_LABELS[ds] if platform == PLATFORM_ORDER[0] else None,
                zorder=3,
            )

            dx, dy = POINT_LABEL_OFFSETS.get((ds, platform), (6, 6))
            ax.annotate(
                f"{platform}\n({fmt_latency_ms(x)}, {fmt_energy_mj(y)})",
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=6.8,
                color=figure_dataset_color("fig5b", ds),
                clip_on=False,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latency (ms/image)")
    ax.set_ylabel("Energy (mJ/image)")

    ax.set_xlim(min(all_x) * 0.5, max(all_x) * 5)
    ax.set_ylim(min(all_y) * 0.42, max(all_y) * 7)

    ax.grid(which="major", color=FIGURE_COLORS["fig5b"]["major_grid"], linewidth=0.65, alpha=0.85)
    ax.grid(which="minor", color=FIGURE_COLORS["fig5b"]["minor_grid"], linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    legend_above(ax, ncol=2, y=1.14)

    # 在右下角添加说明框
    ax.text(
        0.98, 0.04,  # 相对坐标：右下角
        "CPU baseline: Intel Core Ultra 9 185H\nGPU baseline: NVIDIA GeForce RTX 4070 Laptop GPU",
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=5.0,
        ha='right',  # 水平右对齐
        va='bottom',  # 垂直底部对齐
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            alpha=0.85,  # 半透明背景
            edgecolor='#CCCCCC',
            linewidth=0.5
        ),
        zorder=10  # 确保显示在最上层
    )


def draw_fig6a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Fig.6a: energy-saving percentage vs. Default G8/A16.

    Main visual encoding:
      bar height = energy saving vs. dataset-specific Default (%)
      bar label  = absolute energy (uJ/image)

    Default is used as the baseline and is not drawn as a 0% bar.
    """

    def canonical_design_name(name: str) -> str:
        s = str(name).strip()

        alias = {
            "Default": "Default",
            "Default\n8/16": "Default",
            "Default G8/A16": "Default",

            "Conserv.": "ADC-rich",
            "Conservative": "ADC-rich",
            "Conservative\n8/64": "ADC-rich",
            "ADC-rich": "ADC-rich",
            "ADC-rich\nG8/A64": "ADC-rich",

            "Balanced": "Balanced",
            "Balanced\n16/32": "Balanced",
            "Balanced\nG16/A32": "Balanced",

            "Aggress.": "HAPR-heavy",
            "Aggressive": "HAPR-heavy",
            "Aggressive\n32/16": "HAPR-heavy",
            "HAPR-heavy": "HAPR-heavy",
            "HAPR-heavy\nG32/A16": "HAPR-heavy",
        }
        return alias.get(s, s)

    def format_energy_uj(v: float) -> str:
        if v >= 100:
            return f"{v:.0f} μJ"
        return f"{v:.1f} μJ"

    plot_df = df.copy()
    plot_df["design_canon"] = plot_df["design"].apply(canonical_design_name)

    # Default is the baseline for each dataset.
    default_energy = (
        plot_df[plot_df["design_canon"] == "Default"]
        .drop_duplicates(subset=["dataset"])
        .set_index("dataset")["energy_uJ_per_image"]
        .astype(float)
        .to_dict()
    )

    missing_default = [ds for ds in DATASETS if ds not in default_energy]
    if missing_default:
        raise KeyError(f"Missing Default baseline rows for: {missing_default}")

    # Only draw the three meaningful alternatives.
    draw_order = [
        ("ADC-rich\nG8/A64", "ADC-rich"),
        ("Balanced\nG16/A32", "Balanced"),
        ("HAPR-heavy\nG32/A16", "HAPR-heavy"),
    ]

    x = np.arange(len(draw_order), dtype=float)
    width = 0.34

    dataset_offsets = {
        "cifar10dvs": -width / 2,
        "dvsgesture": width / 2,
    }

    all_savings = []

    # Highlight the selected main point.
    balanced_x = 1
    ax.axvspan(
        balanced_x - 0.48,
        balanced_x + 0.48,
        color=FIGURE_COLORS["fig6a"]["balanced_highlight"],
        alpha=0.75,
        zorder=0,
    )

    for ds in DATASETS:
        sub = plot_df[plot_df["dataset"] == ds].copy()

        xs = []
        savings = []
        energies = []

        for i, (_, canon_name) in enumerate(draw_order):
            hit = sub[sub["design_canon"] == canon_name]
            if hit.empty:
                raise KeyError(f"Missing design row: dataset={ds}, design={canon_name}")

            r = hit.iloc[0]
            energy = float(r["energy_uJ_per_image"])
            saving = 100.0 * (1.0 - energy / float(default_energy[ds]))

            xs.append(x[i] + dataset_offsets[ds])
            savings.append(saving)
            energies.append(energy)
            all_savings.append(saving)

        bars = ax.bar(
            xs,
            savings,
            width=width,
            color=figure_dataset_color("fig6a", ds),
            edgecolor="none",
            alpha=0.92,
            label=DATASET_LABELS[ds],
            zorder=3,
        )

        # Top labels show absolute energy.
        for bar, energy in zip(bars, energies):
            h = bar.get_height()
            ax.annotate(
                format_energy_uj(energy),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=figure_dataset_color("fig6a", ds),
                clip_on=False,
            )

    ax.axhline(0, color=FIGURE_COLORS["fig6a"]["zero_line"], linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in draw_order])

    ax.set_ylabel("Energy saving vs. Default (%)")
    ax.set_xlabel("Selected HAPR/ADC design point")

    ymax = max(all_savings) if all_savings else 60.0
    ax.set_ylim(0, max(35.0, ymax * 1.25))

    ax.grid(axis="y", color=FIGURE_COLORS["fig6a"]["grid"], linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Small note, not a title.
    ax.text(
        0.98,
        0.96,
        "Baseline: Default G8/A16, ADC-saturated",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=FIGURE_COLORS["fig6a"]["note"],
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        columnspacing=0.9,
        handlelength=1.4,
        borderaxespad=0.0,
    )


def draw_fig6b(ax: plt.Axes, df: pd.DataFrame) -> None:
    mat = np.full((len(DATASETS), len(ROBUSTNESS_POINTS)), np.nan)
    for i, ds in enumerate(DATASETS):
        for j, (label, _, _) in enumerate(ROBUSTNESS_POINTS):
            hit = df[(df["dataset"] == ds) & (df["condition"] == label)]
            if not hit.empty:
                mat[i, j] = to_float(hit["accuracy_drop_pp"].iloc[0])

    cmap = soft_diverging_cmap()
    vmax = max(2.5, np.nanmax(np.abs(mat)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(ROBUSTNESS_POINTS)))
    ax.set_xticklabels([p[0] for p in ROBUSTNESS_POINTS])
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels(
        [DATASET_LABELS[d] for d in DATASETS],
        rotation=30,
        ha='right',
        va='center'
    )

    ax.set_xticks(np.arange(-0.5, len(ROBUSTNESS_POINTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.grid(which="minor", color=FIGURE_COLORS["fig6b"]["cell_grid"], linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.0, color=FIGURE_COLORS["fig6b"]["cell_text"])

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Accuracy drop (pp)")
    cbar.ax.tick_params(labelsize=7.2)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    cost_breakdown = load_cost_breakdown()
    cost = load_cost(Path(args.cost_csv))
    design = load_design(input_root)
    robustness = load_robustness(Path(args.robustness_csv))

    cost_breakdown.to_csv(output_root / "fig5a_system_cost_breakdown_data.csv", index=False)
    cost.to_csv(output_root / "fig5b_latency_energy_data.csv", index=False)
    design.to_csv(output_root / "fig6a_design_tradeoff_data.csv", index=False)
    robustness.to_csv(output_root / "fig6b_robustness_heatmap_data.csv", index=False)

    panels = [
        (FINAL_FIGURE_NAMES["fig5a"], draw_fig5a, cost_breakdown, (9, 1.9)),
        (FINAL_FIGURE_NAMES["fig5b"], draw_fig5b, cost, (4.20, 2.95)),
        (FINAL_FIGURE_NAMES["fig6a"], draw_fig6a, design, (4.20, 3.00)),
        (FINAL_FIGURE_NAMES["fig6b"], draw_fig6b, robustness, (4.10, 1.75)),
    ]

    for name, fn, data, size in panels:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, data)
        fig.tight_layout()
        save_fig(fig, output_root / f"{name}.pdf", dpi=args.dpi)
        plt.close(fig)

    print(f"Saved four panel figures to: {output_root.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="results/eval_v2")
    parser.add_argument("--cost-csv", default="plot/results/eval_06_local/plot_06_comparison_data.csv")
    parser.add_argument("--robustness-csv", default="plot/results/eval_05_1/plot_07_aggregated_data_practical_none_hybrid.csv")
    parser.add_argument("--output-root", default="plot/results/final_compact_panels")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    main()