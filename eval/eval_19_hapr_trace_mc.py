"""Trace-driven + parasitic-aware HAPR physical-feasibility closure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hardware.hapr_statistical_feasibility import run_joint_monte_carlo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="CSV with workload, fan_in, i_pos_a, i_neg_a")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacitance-multipliers", nargs="+", type=float, default=[1.0, 1.25, 1.5])
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.trace).open(newline="", encoding="utf-8")))
    required = {"workload", "fan_in", "i_pos_a", "i_neg_a"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Trace must contain {sorted(required)}; reconstructed aggregate proxies are not accepted.")
    if any("real_runtime" not in row.get("trace_provenance", "") for row in rows):
        raise ValueError("Publication MC requires real-runtime partial-sum provenance for every trace row.")
    device = yaml.safe_load(Path(args.device_params).read_text(encoding="utf-8"))
    cfg = dict(device["analog_validation"])
    cfg.update({
        "mrr_sigma": 0.02,
        "laser_sigma": 0.01,
        "tia_gain_sigma": 0.01,
        "branch_mismatch_sigma": 0.01,
        "pd_responsivity_sigma": 0.03,
        "adc_full_scale_sigma": 0.01,
        "settling_equivalent_resistance_ohm": cfg.get("tia_transimpedance_ohm", 200.0),
        "settling_error_fraction": 1.0 / 128.0,
    })
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fanin_rows = []
    mc_rows = []
    workloads = sorted(set(row["workload"] for row in rows))
    for workload in workloads:
        selected = [row for row in rows if row["workload"] == workload]
        counts = Counter(int(row["fan_in"]) for row in selected)
        total = sum(counts.values())
        for fan_in in (1, 2, 3, 4):
            fanin_rows.append({"workload": workload, "fan_in": fan_in, "count": counts[fan_in], "percent": 100 * counts[fan_in] / max(total, 1)})
        trace = {key: np.asarray([float(row[key]) for row in selected]) for key in ("fan_in", "i_pos_a", "i_neg_a")}
        for cap in args.capacitance_multipliers:
            result = run_joint_monte_carlo(trace, cfg, samples=args.samples, seed=args.seed, capacitance_multiplier=cap)
            result["workload"] = workload
            mc_rows.append(result)
    for name, payload in (("hapr_fanin_distribution.csv", fanin_rows), ("hapr_monte_carlo.csv", mc_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0])); writer.writeheader(); writer.writerows(payload)
    (out / "summary.json").write_text(json.dumps({
        "analysis": "trace-driven + parasitic-aware joint Monte Carlo",
        "samples_per_workload_capacitance_case": args.samples,
        "variation_assumptions": cfg,
        "fan_in": fanin_rows,
        "monte_carlo": mc_rows,
        "claim_boundary": "architecture-level statistical feasibility; not Cadence, transistor-level SPICE, foundry process MC, or silicon measurement",
    }, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    width = 0.8 / max(len(workloads), 1)
    for wi, workload in enumerate(workloads):
        vals = [next(r["percent"] for r in fanin_rows if r["workload"] == workload and r["fan_in"] == g) for g in (1,2,3,4)]
        axes[0].bar(np.arange(1,5) + (wi-(len(workloads)-1)/2)*width, vals, width=width, label=workload)
    axes[0].set(xlabel="HAPR fan-in", ylabel="Operations (%)", xticks=[1,2,3,4]); axes[0].legend(fontsize=7)
    nominal = [r for r in mc_rows if r["capacitance_multiplier"] == 1.0]
    axes[1].bar([r["workload"] for r in nominal], [r["voltage_abs_p99_v"] for r in nominal])
    axes[1].axhline(float(cfg["adc_full_scale_v"]), color="r", linestyle="--", label="ADC full scale")
    axes[1].set(ylabel="|Vout| P99 (V)"); axes[1].tick_params(axis="x", rotation=20); axes[1].legend(fontsize=7)
    for workload in workloads:
        subset = [r for r in mc_rows if r["workload"] == workload]
        axes[2].plot([r["capacitance_multiplier"] for r in subset], [r["settling_ns_p99"] for r in subset], marker="o", label=workload)
    axes[2].axhline(float(cfg["settling_time_limit_ns"]), color="r", linestyle="--")
    axes[2].set(xlabel="Parasitic-C multiplier", ylabel="Settling P99 (ns)"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_hapr_trace_mc.pdf"); fig.savefig(out / "fig_hapr_trace_mc.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
