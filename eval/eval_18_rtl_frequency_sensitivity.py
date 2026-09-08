"""Frequency sweep that refuses to invent missing RTL dynamic power."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-per-sample", required=True, type=int)
    parser.add_argument("--dynamic-power-1ghz-mw", type=float, default=None,
                        help="Must come from mapped-netlist power analysis with workload VCD/SAIF.")
    parser.add_argument("--leakage-mw", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for frequency in (0.8, 0.9, 1.0):
        latency_us = args.cycles_per_sample / (frequency * 1e3)
        dynamic = None if args.dynamic_power_1ghz_mw is None else args.dynamic_power_1ghz_mw * frequency
        total = None if dynamic is None else dynamic + args.leakage_mw
        energy_nj = None if total is None else total * latency_us
        rows.append({
            "frequency_ghz": frequency,
            "latency_us": latency_us,
            "dynamic_power_mw": dynamic,
            "leakage_power_mw": args.leakage_mw,
            "total_power_mw": total,
            "control_energy_nj": energy_nj,
            "timing_scope": "pre_layout_frequency_sensitivity_not_post_route",
            "power_scope": "unavailable_without_workload_vcd_saif" if dynamic is None else "user_supplied_mapped_netlist_dynamic_anchor_scaled_linearly_with_frequency",
        })
    with (out / "rtl_frequency_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "publication_ready": args.dynamic_power_1ghz_mw is not None,
        "rows": rows,
        "warning": "A VCD alone is not power. Provide a mapped-netlist/library power result before reporting dynamic power.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
