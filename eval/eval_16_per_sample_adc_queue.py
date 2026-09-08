"""Replay A8/A12/A16 ADC queues from eval_05 runtime request traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.per_sample_adc_queue import summarize_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adc-macros", nargs="+", type=int, default=[6, 8, 12, 16])
    parser.add_argument("--adc-sample-rate-gsps", type=float, default=10.0)
    parser.add_argument("--clock-ghz", type=float, default=1.0)
    parser.add_argument("--fifo-depth", type=int, default=4096,
                        help="Must match adc_request_fifo.sv DEPTH (default 4096).")
    parser.add_argument("--photonic-tiles", type=int, default=4)
    parser.add_argument("--outputs-per-tile", type=int, default=64,
                        help="Pre-HAPR output lanes per physical tile.")
    parser.add_argument("--hapr-group-size", type=int, default=4,
                        help="Spatial same-neuron HAPR fan-in; G4 gives 64 post-HAPR lanes for four 64-output tiles.")
    args = parser.parse_args()
    with Path(args.trace).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_samples = []
    for macros in args.adc_macros:
        samples, summary = summarize_samples(
            rows,
            adc_macros=macros,
            adc_sample_rate_gsps=args.adc_sample_rate_gsps,
            clock_ghz=args.clock_ghz,
            fifo_depth=args.fifo_depth,
            photonic_tiles=args.photonic_tiles,
            outputs_per_tile=args.outputs_per_tile,
            hapr_group_size=args.hapr_group_size,
        )
        summary["adc_macros"] = macros
        summary["architecture_feasible"] = bool(
            summary["no_hold_admission_safe_all"] and not summary["fifo_overflow_any"]
        )
        summaries.append(summary)
        all_samples.extend(samples)
    with (out / "adc_queue_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": vars(args), "design_points": summaries}, handle, indent=2)
    if all_samples:
        with (out / "per_sample_adc_queue.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_samples[0]))
            writer.writeheader()
            writer.writerows(all_samples)


if __name__ == "__main__":
    main()
