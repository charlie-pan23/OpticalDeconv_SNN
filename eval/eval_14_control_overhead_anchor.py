"""Join the mapped digital-control anchor to the selected system points."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPA = ROOT / "results" / "hardware_validation" / "nangate45" / "ppa_summary.json"
OUT = ROOT / "results" / "eval_v3" / "combined" / "eval_14"


def main() -> int:
    ppa = json.loads(PPA.read_text(encoding="utf-8"))
    rows = []
    for dataset in ("cifar10dvs", "dvsgesture"):
        selected_path = ROOT / "results" / "eval_v3" / dataset / "eval_06" / "selected_operating_point.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        metrics = selected["metrics"]
        area = float(ppa["mapped_cell_area_mm2"])
        leakage_mw = float(ppa["library_cell_leakage_nw"]) / 1.0e6
        latency_us = float(metrics["headline_latency_us_per_image"])
        system_power_w = float(metrics["headline_power_w"])
        system_energy_uj = float(metrics["headline_energy_uJ_per_image"])
        footprint = float(metrics["area_mm2_component_lower_bound"])
        rows.append({
            "dataset": dataset,
            "publication_point": "G4/A8/W6/ADC6/Vm16",
            "control_standard_cells": int(ppa["standard_cell_count"]),
            "control_flip_flops": int(ppa["flip_flop_count"]),
            "control_area_mm2": area,
            "component_footprint_sum_mm2": footprint,
            "control_area_fraction_percent": 100.0 * area / footprint,
            "control_library_leakage_mw": leakage_mw,
            "nominal_system_power_w": system_power_w,
            "control_leakage_power_fraction_percent": 100.0 * leakage_mw / (system_power_w * 1000.0),
            "control_leakage_energy_uJ_per_image": leakage_mw * 1.0e-3 * latency_us,
            "nominal_system_energy_uJ_per_image": system_energy_uj,
            "control_leakage_energy_fraction_percent": 100.0 * (leakage_mw * 1.0e-3 * latency_us) / system_energy_uj,
            "abc_pre_layout_worst_delay_ps": ppa["abc_pre_layout_worst_combinational_delay_ps"],
            "abc_pre_layout_frequency_proxy_mhz": ppa["abc_pre_layout_frequency_proxy_mhz"],
            "timing_scope": ppa["timing_status"],
            "power_scope": "Leakage-only lower bound; workload VCD/SAIF dynamic power is unavailable and not estimated.",
            "accounting_policy": "Control leakage is reported as an overhead ratio and is not added again to system power because the existing leakage/misc-I-O proxy may already include it.",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "control_overhead_anchor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps({
        "eval_name": "eval_14_control_overhead_anchor",
        "publication_ready": False,
        "reason": "ABC pre-layout timing and library leakage are reproducible anchors; OpenSTA/post-route timing and workload-driven dynamic power remain missing.",
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[eval_14] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
