"""Convert the existing incremental ablation into paper-facing resource causality
and fixed-versus-variable energy evidence.

This evaluator does not invent per-sample behavior. It consumes the aggregate
eval_07 table and labels the provenance accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


MECHANISM = {
    "B0_dense_interface": ("Conventional WDM/MRR SNN", "reference"),
    "B1_plus_event_gating": ("+ Event gating", "optical_transactions"),
    "B2_plus_HAPR": ("+ G4 HAPR", "adc_conversions"),
    "B3_plus_ADC_pooling": ("+ A8 ADC pool", "adc_macros"),
    "B4_full_HIPSA_plus_lazy_LIF": ("+ Lazy LIF/SRAM", "sram_accesses"),
}


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def reduction(before: float, after: float) -> float:
    return 100.0 * (1.0 - after / before) if before else 0.0


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/eval_v3/combined/eval_07/incremental_ablation.csv")
    parser.add_argument("--output", default="results/eval_v3/combined/eval_12")
    parser.add_argument("--lock-power-mw", type=float, default=0.0)
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    by_dataset: Dict[str, List[Dict[str, str]]] = {}
    for row in source:
        by_dataset.setdefault(row["dataset"], []).append(row)

    resource_rows: List[Dict[str, object]] = []
    floor_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for dataset, rows in by_dataset.items():
        previous = None
        for row in rows:
            label, target = MECHANISM[row["variant"]]
            txns = f(row, "transactions_issued_per_image")
            conv = f(row, "adc_conversions_per_image")
            macros = f(row, "adc_macros")
            sram = f(row, "sram_reads_per_image") + f(row, "sram_writes_per_image")
            target_value = {"reference": 0.0, "optical_transactions": txns, "adc_conversions": conv, "adc_macros": macros, "sram_accesses": sram}[target]
            prev_value = 0.0
            if previous is not None:
                prev_value = {
                    "optical_transactions": f(previous, "transactions_issued_per_image"),
                    "adc_conversions": f(previous, "adc_conversions_per_image"),
                    "adc_macros": f(previous, "adc_macros"),
                    "sram_accesses": f(previous, "sram_reads_per_image") + f(previous, "sram_writes_per_image"),
                }.get(target, 0.0)
            resource_rows.append({
                "dataset": dataset,
                "configuration": label,
                "mechanism_target": target,
                "optical_transactions_per_image": txns,
                "adc_conversions_per_image": conv,
                "adc_macros": int(macros),
                "sram_accesses_per_image": sram,
                "latency_us_per_image": f(row, "latency_us_per_image"),
                "energy_uJ_per_image_no_lock": f(row, "energy_uJ_per_image"),
                "component_footprint_sum_mm2": f(row, "area_mm2"),
                "target_reduction_percent_vs_previous": reduction(prev_value, target_value) if previous is not None else 0.0,
                "trace_provenance": row["trace_provenance"],
            })
            previous = row

        for row in (rows[0], rows[-1]):
            latency_us = f(row, "latency_us_per_image")
            no_lock_energy = f(row, "energy_uJ_per_image")
            fixed_power_mw = f(row, "power_floor_mw") + args.lock_power_mw
            fixed_energy = fixed_power_mw / 1000.0 * latency_us
            total_energy = no_lock_energy + args.lock_power_mw / 1000.0 * latency_us
            variable_energy = total_energy - fixed_energy
            floor_rows.append({
                "dataset": dataset,
                "configuration": MECHANISM[row["variant"]][0],
                "latency_us_per_image": latency_us,
                "fixed_power_mw_laser_leakage_plus_optional_lock": fixed_power_mw,
                "fixed_energy_uJ_per_image": fixed_energy,
                "activity_dependent_and_oneshot_energy_uJ_per_image": variable_energy,
                "total_energy_uJ_per_image_with_optional_lock": total_energy,
                "fixed_energy_share_percent": 100.0 * fixed_energy / total_energy,
                "accounting_scope": f"one_pulse_mechanism_isolation; aggregate trace; optional_lock_power_mw={args.lock_power_mw}",
            })
        dense, hipsa = floor_rows[-2], floor_rows[-1]
        summaries.append({
            "dataset": dataset,
            "end_to_end_energy_reduction_percent": reduction(float(dense["total_energy_uJ_per_image_with_optional_lock"]), float(hipsa["total_energy_uJ_per_image_with_optional_lock"])),
            "controllable_energy_reduction_percent": reduction(float(dense["activity_dependent_and_oneshot_energy_uJ_per_image"]), float(hipsa["activity_dependent_and_oneshot_energy_uJ_per_image"])),
            "hipsa_fixed_energy_share_percent": hipsa["fixed_energy_share_percent"],
            "interpretation": "HIPSA reduces the controllable electronic portion more strongly than total energy because the continuously biased optical/I-O floor remains; optional lock is zero unless explicitly supplied.",
        })

    out = Path(args.output)
    write_csv(out / "mechanism_resource_causality.csv", resource_rows)
    write_csv(out / "fixed_variable_energy.csv", floor_rows)
    write_csv(out / "energy_floor_summary.csv", summaries)
    (out / "summary.json").write_text(json.dumps({"publication_ready": False, "reason": "aggregate trace only; generated evidence is accounting-level, not per-sample replay", "energy_floor_summary": summaries}, indent=2), encoding="utf-8")
    print(f"[eval_12] resource and energy-floor evidence saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
