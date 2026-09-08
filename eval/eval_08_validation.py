"""Cross-stage validation report for the HIPSA paper-facing evaluation.

This is a read-only gate.  It does not train, change checkpoints, or modify
the HIPSA architecture.  It checks that the outputs used in a paper are
internally consistent and that the reviewer-sensitive assumptions are visible:
HAPR analog feasibility, MRR stabilization scenarios, count-coded input, and
power/latency arithmetic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.analog_validation import (
    derive_mrr_stabilization_budget,
    derive_optical_link_budget,
    validate_hapr_corner_sweep,
    validate_hapr_fanin,
)
from utils.result_io import load_csv_rows, load_json, load_yaml, save_csv_rows, save_json


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def add_issue(
    issues: List[Dict[str, Any]],
    dataset: str,
    check: str,
    status: str,
    message: str,
    *,
    value: Any = "",
) -> None:
    issues.append(
        {
            "dataset": dataset,
            "check": check,
            "status": status,
            "value": value,
            "message": message,
        }
    )


def validate_dataset(
    dataset: str,
    input_root: Path,
    config_path: Path,
    device_cfg: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    eval01 = input_root / dataset / "eval_01" / "summary.json"
    eval02 = input_root / dataset / "eval_02"
    eval04 = input_root / dataset / "eval_04"

    if not eval01.exists():
        add_issue(issues, dataset, "eval01_present", "error", f"Missing {eval01}")
        return issues
    if not (eval02 / "summary.json").exists():
        add_issue(issues, dataset, "eval02_present", "error", f"Missing {eval02 / 'summary.json'}")
    if not (eval04 / "summary.json").exists():
        add_issue(issues, dataset, "eval04_present", "warning", f"Missing {eval04 / 'summary.json'}")

    activity = load_json(eval01)
    if (eval02 / "summary.json").exists():
        summary = load_json(eval02 / "summary.json")
        timing = summary.get("timing", {})
        power = summary.get("power", {})
        latency_s = as_float(timing.get("latency_s_per_image"), 0.0)
        power_w = as_float(power.get("total_power_w"), 0.0)
        energy_uj = as_float(power.get("energy_uJ_per_image"), 0.0)
        expected_uj = latency_s * power_w * 1.0e6
        rel_err = abs(energy_uj - expected_uj) / max(abs(expected_uj), 1.0e-12)
        if rel_err <= 1.0e-6:
            add_issue(issues, dataset, "energy_arithmetic", "pass", "energy = power × latency", value=rel_err)
        else:
            add_issue(issues, dataset, "energy_arithmetic", "error", "Energy is inconsistent with power × latency", value=rel_err)

        mrr_reference = summary.get("mrr_reference_case", {})
        if as_float(mrr_reference.get("mrr_stabilization_mw"), 0.0) > 0.0:
            add_issue(
                issues,
                dataset,
                "mrr_main_summary_visibility",
                "pass",
                "Main eval-02 summary carries a representative nonzero MRR stabilization case.",
                value=mrr_reference,
            )
        else:
            add_issue(
                issues,
                dataset,
                "mrr_main_summary_visibility",
                "warning",
                "Main eval-02 summary has no representative nonzero MRR stabilization case.",
                value=mrr_reference,
            )

        util = as_float(timing.get("effective_utilization"), 0.0)
        accounting = hardware_cfg.get("utilization_accounting", {})
        factors = accounting.get("factors", {}) if isinstance(accounting, Mapping) else {}
        derived_util = 1.0
        if isinstance(factors, Mapping) and factors:
            for factor in factors.values():
                derived_util *= as_float(factor, 1.0)
        configured_util = as_float(
            hardware_cfg.get("photonic_tiles", {}).get("effective_utilization")
            if isinstance(hardware_cfg.get("photonic_tiles"), Mapping)
            else None,
            util,
        )
        if factors and abs(derived_util - configured_util) <= 2.0e-3:
            add_issue(
                issues,
                dataset,
                "utilization_provenance",
                "pass",
                "effective_utilization is reproduced by the declared factorized resource model.",
                value={"configured": configured_util, "derived": derived_util, "factors": factors},
            )
        else:
            add_issue(
                issues,
                dataset,
                "utilization_provenance",
                "warning",
                "effective_utilization remains an unproven architecture-model input; provide a mapping/scheduling derivation for publication.",
                value=util,
            )
        utilization_csv = eval02 / "utilization_sensitivity.csv"
        if utilization_csv.exists():
            utilization_rows = load_csv_rows(utilization_csv, parse_numbers=True)
            points = sorted({round(as_float(row.get("utilization_point"), 0.0), 6) for row in utilization_rows})
            if len(points) >= 3 and any(abs(point - util) <= 1.0e-6 for point in points):
                add_issue(
                    issues,
                    dataset,
                    "utilization_sensitivity",
                    "pass",
                    "Headline latency/energy is accompanied by an explicit utilization sensitivity sweep.",
                    value=points,
                )
            else:
                add_issue(
                    issues,
                    dataset,
                    "utilization_sensitivity",
                    "warning",
                    "Utilization sensitivity output is present but does not cover the headline point and at least two alternatives.",
                    value=points,
                )
        else:
            add_issue(
                issues,
                dataset,
                "utilization_sensitivity",
                "warning",
                f"Missing {utilization_csv}; a single utilization point is not sufficient for publication.",
            )

        area = summary.get("area", {})
        if as_float(area.get("available_area_total_mm2"), 0.0) > 0:
            area_csv = eval02 / "area_breakdown.csv"
            if area_csv.exists():
                area_rows = load_csv_rows(area_csv, parse_numbers=True)
                anchored = [
                    row
                    for row in area_rows
                    if as_float(row.get("area_total_um2"), 0.0) > 0
                    and row.get("note") not in (None, "")
                ]
                add_issue(
                    issues,
                    dataset,
                    "area_anchor_traceability",
                    "pass" if len(anchored) == len(area_rows) else "warning",
                    "Area rows expose primitive anchors, instance counts and notes; synthesis/routing remains outside the estimate.",
                    value={"rows": len(area_rows), "anchored_rows": len(anchored)},
                )
            add_issue(
                issues,
                dataset,
                "area_traceability",
                "warning",
                "Area total now includes explicit device anchors and a reference digital/control budget; still label it as a macro-area estimate until RTL/macro synthesis and packaging are included.",
                value=area.get("available_area_total_mm2"),
            )

    encoding = ""
    if config_path.exists():
        config = load_yaml(config_path)
        encoding = str(config.get("preprocess", {}).get("input_encoding", ""))
        if encoding == "clipped_count":
            add_issue(
                issues,
                dataset,
                "input_encoding_boundary",
                "warning",
                "CIFAR input is count-coded; document bit-serial/multi-pulse mapping or do not call the complete interface strictly binary.",
                value=encoding,
            )
            interface = hardware_cfg.get("interface_accounting", {})
            mapping = interface.get("cifar10dvs_mapping", "") if isinstance(interface, Mapping) else ""
            accounted = bool(interface.get("cifar10dvs_pulse_overhead_accounted", False)) if isinstance(interface, Mapping) else False
            add_issue(
                issues,
                dataset,
                "input_encoding_mapping",
                "pass" if mapping else "warning",
                "CIFAR10-DVS count-to-pulse/bit-serial mapping is explicitly declared; timing/energy accounting still needs to be enabled." if mapping else "No count-coded input mapping is declared.",
                value={"mapping": mapping, "pulse_overhead_accounted": accounted},
            )
        else:
            add_issue(issues, dataset, "input_encoding_boundary", "pass", "Input encoding is explicitly declared.", value=encoding)

    if eval04.exists() and (eval04 / "hapr_analog_validation.csv").exists():
        analog_rows = load_csv_rows(eval04 / "hapr_analog_validation.csv", parse_numbers=True)
    else:
        try:
            analog_rows = validate_hapr_fanin(device_cfg, [4, 8, 16, 32], adc_bits=6)
        except Exception as exc:
            analog_rows = []
            add_issue(issues, dataset, "hapr_analog_validation", "error", str(exc))

    if analog_rows:
        g16 = next((r for r in analog_rows if int(r.get("hapr_group_size", 0)) == 16), None)
        g32 = next((r for r in analog_rows if int(r.get("hapr_group_size", 0)) == 32), None)
        if g16 is None:
            add_issue(issues, dataset, "hapr_analog_validation", "error", "No G=16 analog validation row found")
        elif int(as_float(g16.get("feasible"), 0)) == 1:
            add_issue(issues, dataset, "hapr_analog_validation", "pass", "G=16 passes the configured analytical screen", value=g16)
        else:
            add_issue(issues, dataset, "hapr_analog_validation", "warning", "G=16 fails the configured analytical screen; revise assumptions or design point", value=g16)
        if g32 is not None and int(as_float(g32.get("feasible"), 0)) == 0:
            add_issue(issues, dataset, "hapr_analog_upper_bound", "pass", "G=32 is rejected by the same screen, making the G=16 limit explicit", value=g32)
        else:
            add_issue(issues, dataset, "hapr_analog_upper_bound", "warning", "The requested G=32 upper-bound row is not rejected by the analytical screen", value=g32 or {})

    link_budget_path = eval04 / "link_budget.json"
    if link_budget_path.exists():
        link_budget = load_json(link_budget_path)
    else:
        link_budget = derive_optical_link_budget(device_cfg)
    add_issue(
        issues,
        dataset,
        "optical_link_budget",
        "pass" if int(as_float(link_budget.get("pass"), 0)) == 1 else "warning",
        "Received per-lane optical power is derived from laser WPE, lane allocation and explicit loss terms." if int(as_float(link_budget.get("pass"), 0)) == 1 else "Optical link-budget residual exceeds the declared tolerance.",
        value=link_budget,
    )

    corner_path = eval04 / "hapr_corner_validation.csv"
    if corner_path.exists():
        corner_rows = load_csv_rows(corner_path, parse_numbers=True)
    else:
        corner_rows = validate_hapr_corner_sweep(device_cfg, [4, 8, 16, 32], adc_bits=6)
    g16_corners = [row for row in corner_rows if int(as_float(row.get("hapr_group_size"), 0)) == 16]
    nominal_g16 = next((row for row in g16_corners if row.get("corner_name") == "nominal"), None)
    non_nominal_g16 = [row for row in g16_corners if row.get("corner_name") != "nominal"]
    if nominal_g16 and int(as_float(nominal_g16.get("feasible"), 0)) == 1:
        failed = sum(int(as_float(row.get("feasible"), 0)) == 0 for row in non_nominal_g16)
        add_issue(
            issues,
            dataset,
            "hapr_corner_sensitivity",
            "pass" if failed == 0 else "warning",
            "G=16 passes the nominal and declared analog corners." if failed == 0 else "G=16 passes nominal conditions but fails at least one declared corner; report the conditional operating envelope.",
            value={"corners": len(g16_corners), "failed_non_nominal": failed},
        )
    else:
        add_issue(issues, dataset, "hapr_corner_sensitivity", "warning", "No nominal G=16 pass row is available in the corner sweep.", value=g16_corners)

    mrr_path = eval04 / "mrr_sensitivity.csv"
    if mrr_path.exists():
        mrr_rows = load_csv_rows(mrr_path, parse_numbers=True)
        if any(as_float(r.get("mrr_stabilization_mw"), 0.0) >= 19000.0 for r in mrr_rows):
            add_issue(issues, dataset, "mrr_stabilization_visibility", "pass", "Full per-ring-lock stress case is present in the paper-facing sensitivity output")
        else:
            add_issue(issues, dataset, "mrr_stabilization_visibility", "warning", "No conventional full per-ring-lock stress case found in mrr_sensitivity.csv")
    else:
        add_issue(issues, dataset, "mrr_stabilization_visibility", "error", f"Missing {mrr_path}; run eval_02/eval_04 after the update")

    mrr_budget_path = eval04 / "mrr_budget_audit.json"
    if mrr_budget_path.exists():
        mrr_budget = load_json(mrr_budget_path)
    else:
        mrr_budget = derive_mrr_stabilization_budget(device_cfg)
    budget_reconciles = int(as_float(mrr_budget.get("physical_ring_count"), 0)) == 32768
    source_declared = bool(mrr_budget.get("literature_url"))
    add_issue(
        issues,
        dataset,
        "mrr_budget_traceability",
        "pass" if budget_reconciles and source_declared else "warning",
        "Architecture-native 32,768-ring count and literature lock anchor are traceable." if budget_reconciles and source_declared else "MRR sensitivity is not fully traceable; keep it as a scenario rather than a measured claim.",
        value=mrr_budget,
    )

    eval05_dir = input_root / dataset / "eval_05"
    eval05_summary_path = eval05_dir / "summary.json"
    calibration_path = eval05_dir / "adc_scale_calibration.json"
    if eval05_summary_path.exists():
        eval05_summary = load_json(eval05_summary_path)
        if str(eval05_summary.get("adc_scale_mode", "")) == "frozen_calibrated" and calibration_path.exists():
            add_issue(issues, dataset, "eval05_calibration_freshness", "pass", "Robustness result includes the frozen per-layer ADC calibration artifact.")
        else:
            add_issue(issues, dataset, "eval05_calibration_freshness", "warning", "Robustness summary predates or omits the frozen ADC calibration artifact; rerun eval_05 before publication.")
    else:
        add_issue(issues, dataset, "eval05_calibration_freshness", "warning", "No eval_05 summary found; robustness evidence is incomplete.")

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_08: cross-stage validation report")
    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml", type=str)
    parser.add_argument("--device-params", default="configs/device_params.yaml", type=str)
    parser.add_argument("--cifar-config", default="configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml", type=str)
    parser.add_argument("--gesture-config", default="results/dvsgesture/config_dvsgesture_acc88p54.yaml", type=str)
    parser.add_argument("--output-root", default="results/eval_v3/combined/eval_08", type=str)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    device_cfg = load_yaml(args.device_params)
    hardware_cfg = load_yaml(args.hardware)
    config_paths = {
        "cifar10dvs": Path(args.cifar_config),
        "dvsgesture": Path(args.gesture_config),
    }

    issues: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        issues.extend(
            validate_dataset(
                dataset,
                input_root,
                config_paths.get(dataset, Path("")),
                device_cfg,
                hardware_cfg,
            )
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "eval_name": "eval_08",
        "purpose": "cross_stage_validation_report",
        "created_utc": now_utc(),
        "datasets": args.datasets,
        "num_checks": len(issues),
        "num_errors": sum(x["status"] == "error" for x in issues),
        "num_warnings": sum(x["status"] == "warning" for x in issues),
        "issues": issues,
        "scope": "read-only consistency and assumption visibility gate; no architecture or checkpoint changes",
    }
    save_json(summary, output_root / "summary.json")
    save_csv_rows(issues, output_root / "validation_report.csv")

    for row in issues:
        print(f"[{row['status']}] {row['dataset']} {row['check']}: {row['message']}")

    if summary["num_errors"] > 0:
        return 1
    if args.fail_on_warning and summary["num_warnings"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
