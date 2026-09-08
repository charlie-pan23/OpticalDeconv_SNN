"""Reviewer-focused consistency gate for the hybrid A-device/v3-system path.

The gate is deliberately smaller than a silicon sign-off checklist.  It
checks the three points raised by the review: an auditable HAPR bound, a
same-level system-model result (without CPU/GPU software speedups), and
explicit MRR stabilization energy.  Missing checkpoints or PDK/synthesis
artifacts are reported as warnings instead of being filled with invented
numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.result_io import load_csv_rows, load_json, load_yaml, save_csv_rows, save_json


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def issue(rows: List[Dict[str, Any]], dataset: str, check: str, status: str, message: str, value: Any = "") -> None:
    rows.append({"dataset": dataset, "check": check, "status": status, "message": message, "value": value})


def _check_energy(rows: List[Dict[str, Any]], dataset: str, selected: Mapping[str, Any]) -> None:
    metrics = selected.get("metrics", {})
    breakdown = selected.get("energy_breakdown_uJ_per_image", {})
    power = _f(metrics.get("nominal_power_w"), 0.0)
    latency_us = _f(metrics.get("latency_us_per_image"), 0.0)
    expected = power * latency_us + _f(breakdown.get("mrr_configuration")) + _f(breakdown.get("non_mvm_operators"))
    actual = _f(breakdown.get("total"), 0.0)
    rel = abs(expected - actual) / max(abs(expected), 1.0e-12)
    issue(rows, dataset, "nominal_energy_arithmetic", "pass" if rel <= 1.0e-6 else "error", "Nominal energy equals steady-state power × latency plus one-shot/non-MVM terms.", rel)

    lock = _f(metrics.get("mrr_reference_energy_uJ_per_image"), 0.0)
    lock_expected = actual + _f(breakdown.get("mrr_stabilization_reference_additional"), 0.0)
    lock_rel = abs(lock - lock_expected) / max(abs(lock_expected), 1.0e-12)
    issue(rows, dataset, "lock_energy_arithmetic", "pass" if lock > 0.0 and lock_rel <= 1.0e-6 else "error", "Continuous MRR-lock energy is added separately from one-shot configuration energy.", {"relative_error": lock_rel, "lock_energy_uJ": lock})


def validate_dataset(dataset: str, root: Path, device_cfg: Mapping[str, Any], hardware_cfg: Mapping[str, Any], rows: List[Dict[str, Any]]) -> None:
    out = root / dataset / "eval_06"
    selected_path = out / "selected_operating_point.json"
    if not selected_path.exists():
        issue(rows, dataset, "selected_point_present", "error", f"Missing {selected_path}")
        return
    selected = load_json(selected_path)
    issue(rows, dataset, "selected_point_present", "pass", "Hybrid v3 selected operating point is present.")

    analog_path = out / "hapr_analog_validation.csv"
    corner_path = out / "hapr_corner_validation.csv"
    if analog_path.exists():
        analog = load_csv_rows(analog_path, parse_numbers=True)
        g4 = next((r for r in analog if _i(r.get("hapr_group_size"), -1) == 4), None)
        g32 = next((r for r in analog if _i(r.get("hapr_group_size"), -1) == 32), None)
        issue(rows, dataset, "hapr_g4_analog_feasible", "pass" if g4 and _i(g4.get("feasible")) == 1 else "error", "G=4 passes the device-parameter analog screen.", g4 or {})
        issue(rows, dataset, "hapr_g32_analog_upper_bound", "pass" if g32 and _i(g32.get("feasible")) == 0 else "error", "G=32 is also rejected by the analog saturation screen.", g32 or {})
    else:
        issue(rows, dataset, "hapr_analog_validation", "error", f"Missing {analog_path}")

    if corner_path.exists():
        corners = load_csv_rows(corner_path, parse_numbers=True)
        g4_corners = [r for r in corners if _i(r.get("hapr_group_size"), -1) == 4]
        failed = [r.get("corner_name") for r in g4_corners if _i(r.get("feasible")) == 0]
        issue(rows, dataset, "hapr_g4_corner_sweep", "pass" if g4_corners and not failed else "error", "G=4 survives the declared analog corner set.", {"num_corners": len(g4_corners), "failed": failed})
    else:
        issue(rows, dataset, "hapr_corner_sweep", "error", f"Missing {corner_path}")

    bound_path = out / "hapr_bound_derivation.json"
    if bound_path.exists():
        bound = load_json(bound_path)
        equations = bound.get("equations", {})
        physical_limit = _i(bound.get("physical_fanin_limit"), -1)
        physical_reject = _i(bound.get("first_physically_rejected_group_size"), -1)
        bound_ok = len(equations) >= 5 and _i(bound.get("selected_group_size"), -1) == 4 and physical_limit == 4 and physical_reject == 8
        issue(rows, dataset, "hapr_bound_derivation", "pass" if bound_ok else "error", "HAPR selection includes analog bounds and the four-tile spatial-concurrency bound; no temporal analog storage is assumed.", {"equation_count": len(equations), "analog_first_rejected": bound.get("first_rejected_group_size"), "physical_limit": physical_limit, "physical_first_rejected": physical_reject})
    else:
        issue(rows, dataset, "hapr_bound_derivation", "error", f"Missing {bound_path}")

    link = selected.get("link_budget", {})
    issue(rows, dataset, "optical_link_budget", "pass" if _i(link.get("pass")) == 1 and abs(_f(link.get("residual_db"))) <= _f(link.get("target_tolerance_db"), 0.5) else "error", "A's explicit laser case is reconciled by the v3 per-lane link-budget equation.", {"residual_db": link.get("residual_db"), "tolerance_db": link.get("target_tolerance_db"), "derivation_mode": link.get("derivation_mode")})

    evidence = selected.get("device_evidence", {})
    required = {"photodiode", "tia", "mrr_weight_bank"}
    missing = sorted(name for name in required if not isinstance(evidence.get(name), Mapping) or not (evidence[name].get("doi") or evidence[name].get("url")))
    issue(rows, dataset, "device_source_traceability", "pass" if not missing else "error", "Responsivity, TIA-noise and MRR anchors carry source links.", {"missing": missing})

    audit = selected.get("mrr_stabilization", {}).get("budget_audit", {})
    scenarios_path = out / "mrr_stabilization_scenarios.csv"
    scenarios = load_csv_rows(scenarios_path, parse_numbers=True) if scenarios_path.exists() else []
    has_zero = any(abs(_f(r.get("mrr_stabilization_mw"))) <= 1.0e-12 for r in scenarios)
    has_nonzero = any(_f(r.get("mrr_stabilization_mw")) > 0.0 for r in scenarios)
    expected_rings = 2 * _i(hardware_cfg.get("photonic_tiles", {}).get("num_tiles"), 0) * 64 * 64
    system_scaled = _i(audit.get("physical_ring_count"), -1) == expected_rings
    metrics = selected.get("metrics", {})
    input_headline = selected.get("input_encoding", {}).get("headline_case", {})
    headline_nominal = (
        not bool(metrics.get("headline_includes_mrr_reference"))
        and abs(_f(metrics.get("headline_power_w")) - _f(input_headline.get("nominal_power_w"))) <= 1.0e-9
        and abs(_f(metrics.get("headline_energy_uJ_per_image")) - _f(input_headline.get("nominal_energy_uJ_per_image"))) <= 1.0e-9
    )
    mrr_ok = bool(audit.get("literature_url")) and has_zero and has_nonzero and system_scaled and headline_nominal
    issue(rows, dataset, "mrr_stabilization_visibility", "pass" if mrr_ok else "error", "Architecture-native fractional-lock scenarios are auditable; the no-lock point is labeled as a lower bound.", {"scenario_rows": len(scenarios), "expected_physical_rings": expected_rings, "system_scaled": system_scaled, "headline_nominal": headline_nominal})

    adc_sensitivity_path = out / "adc_idle_bias_sensitivity.csv"
    adc_sensitivity = load_csv_rows(adc_sensitivity_path, parse_numbers=True) if adc_sensitivity_path.exists() else []
    fractions = {_f(r.get("adc_idle_bias_fraction"), -1.0) for r in adc_sensitivity}
    issue(
        rows,
        dataset,
        "adc_idle_bias_sensitivity",
        "pass" if {0.0, 0.2, 0.5}.issubset(fractions) else "error",
        "ADC-pool energy exposes the declared static-bias assumption over a 0--50% sensitivity range.",
        {"rows": len(adc_sensitivity), "fractions": sorted(fractions)},
    )

    pulse_path = out / "input_encoding_pulse_sensitivity.csv"
    pulse_rows = load_csv_rows(pulse_path, parse_numbers=True) if pulse_path.exists() else []
    if dataset == "cifar10dvs":
        multipliers = {_i(r.get("first_layer_pulse_multiplier"), -1) for r in pulse_rows}
        pulse_ok = multipliers == {1, 2, 3} and _i(metrics.get("headline_first_layer_pulse_multiplier"), -1) == 3
        issue(rows, dataset, "clipped_count_first_layer", "pass" if pulse_ok else "error", "CIFAR clipped-count input is bounded with 1/2/3-pulse first-layer replay and the conservative 3-pulse case is the headline.", {"multipliers": sorted(multipliers), "headline": metrics.get("headline_first_layer_pulse_multiplier")})
    else:
        pulse_ok = len(pulse_rows) == 1 and str(pulse_rows[0].get("input_encoding")) == "binary"
        issue(rows, dataset, "binary_first_layer", "pass" if pulse_ok else "error", "DVS Gesture retains one-pulse binary input accounting.", {"rows": len(pulse_rows)})

    link_sensitivity_path = out / "link_budget_sensitivity.csv"
    link_sensitivity = load_csv_rows(link_sensitivity_path, parse_numbers=True) if link_sensitivity_path.exists() else []
    link_points = {(_f(r.get("loss_delta_db")), _f(r.get("wall_plug_efficiency"))) for r in link_sensitivity}
    issue(rows, dataset, "forward_link_sensitivity", "pass" if len(link_sensitivity) == 20 and (0.0, 0.2) in link_points else "error", "Laser power is forward-derived from required detector power for loss and wall-plug-efficiency sensitivity points.", {"rows": len(link_sensitivity)})

    transaction = out / "transaction_summary.json"
    queue = out / "adc_queue_summary.json"
    issue(rows, dataset, "v3_system_trace", "pass" if transaction.exists() and queue.exists() else "error", "Transaction and ADC FIFO artifacts are present for the same-level system model.")
    transaction_summary = load_json(transaction) if transaction.exists() else {}
    selected_g = _i(selected.get("selected_design", {}).get("hapr_group_size"), -1)
    spatial_hapr = (
        transaction_summary.get("hapr_execution") == "spatial_same_neuron_same_cycle_no_temporal_analog_storage"
        and selected_g <= _i(transaction_summary.get("hapr_physical_fanin_limit"), -1)
    )
    issue(rows, dataset, "hapr_spatial_concurrency", "pass" if spatial_hapr else "error", "Every HAPR group is bounded by simultaneously available physical tiles.", {"selected_g": selected_g, "physical_limit": transaction_summary.get("hapr_physical_fanin_limit"), "mode": transaction_summary.get("hapr_execution")})

    area_path = out / "area_breakdown.csv"
    area_rows = load_csv_rows(area_path, parse_numbers=True) if area_path.exists() else []
    tia = next((r for r in area_rows if r.get("component") == "tia"), {})
    comparator = next((r for r in area_rows if r.get("component") == "comparators"), {})
    front_end_count_ok = _i(tia.get("count"), -1) == 64 and _i(comparator.get("count"), -1) == 64
    issue(rows, dataset, "post_hapr_frontend_count", "pass" if front_end_count_ok else "error", "G4 spatial reduction provisions 64 post-HAPR TIA/comparator lanes, not 256 always-on lanes.", {"tia_count": tia.get("count"), "comparator_count": comparator.get("count")})
    comparison_disabled = hardware_cfg.get("parameter_policy", {}).get("cpu_gpu_software_speedup_comparison") is False
    serialized = str({
        "selected_design": selected.get("selected_design", {}),
        "metrics": selected.get("metrics", {}),
        "energy_breakdown": selected.get("energy_breakdown_uJ_per_image", {}),
    }).lower()
    no_software_claim = "cpu" not in serialized and "gpu" not in serialized
    issue(rows, dataset, "software_speedup_removed", "pass" if comparison_disabled and no_software_claim else "error", "No modeled-HIPSA-versus-measured-CPU/GPU speedup is used in the paper-facing path.", {"config_disabled": comparison_disabled, "artifact_clean": no_software_claim})

    _check_energy(rows, dataset, selected)
    ablation_path = root / dataset / "eval_07" / "incremental_ablation.csv"
    if ablation_path.exists():
        ablation = load_csv_rows(ablation_path, parse_numbers=True)
        deltas = [
            _f(ablation[i - 1].get("energy_uJ_per_image")) - _f(ablation[i].get("energy_uJ_per_image"))
            for i in range(1, len(ablation))
        ]
        issue(rows, dataset, "four_mechanism_ablation", "pass" if len(ablation) == 5 and all(delta > 0 for delta in deltas) else "error", "The conventional WDM/MRR SNN baseline plus four cumulative mechanisms show visible incremental energy effects.", {"rows": len(ablation), "incremental_energy_savings_uJ": deltas})
    else:
        issue(rows, dataset, "four_mechanism_ablation", "error", f"Missing {ablation_path}")
    publication_replay_dir = PROJECT_ROOT / "results" / "eval_v6_publication" / dataset / "eval_05"
    diagnostic_replay_dir = PROJECT_ROOT / "results" / "eval_v4_checkpoint_replay" / dataset / "eval_05"
    replay_dir = publication_replay_dir if publication_replay_dir.exists() else diagnostic_replay_dir
    replay_summary = replay_dir / "robustness_summary.csv"
    sample_predictions = replay_dir / "per_sample_predictions.csv"
    if replay_summary.exists() and sample_predictions.exists():
        replay_rows = load_csv_rows(replay_summary, parse_numbers=True)
        prediction_rows = load_csv_rows(sample_predictions, parse_numbers=True)
        combined_rows = [r for r in replay_rows if r.get("perturbation_type") == "combined"]
        expected_predictions = sum(_i(r.get("num_samples")) for r in replay_rows)
        replay_meta = load_json(replay_dir / "summary.json") if (replay_dir / "summary.json").exists() else {}
        fixedpoint_path = replay_dir / "fixedpoint_realization.json"
        fixedpoint = load_json(fixedpoint_path) if fixedpoint_path.exists() else {}
        publication_mode = bool(replay_meta.get("publication_mode", False))
        calibration_split = str(replay_meta.get("calibration_split", ""))
        reporting_split = str(replay_meta.get("reporting_split", replay_meta.get("split", "")))
        split_isolated = bool(calibration_split and reporting_split and calibration_split != reporting_split)
        weight = fixedpoint.get("weight", {}) if isinstance(fixedpoint, Mapping) else {}
        adc = fixedpoint.get("adc", {}) if isinstance(fixedpoint, Mapping) else {}
        membrane = fixedpoint.get("membrane", {}) if isinstance(fixedpoint, Mapping) else {}
        fixedpoint_realized = bool(
            weight.get("enabled") and _i(weight.get("bits")) == 6
            and _i(adc.get("bits")) == 6
            and membrane.get("enabled") and _i(membrane.get("bits")) == 16
        )
        structurally_complete = bool(combined_rows) and len(prediction_rows) == expected_predictions
        valid = structurally_complete and publication_mode and split_isolated and fixedpoint_realized
        issue(
            rows,
            dataset,
            "hardware_aware_accuracy",
            "pass" if valid else "warning",
            ("Publication-grade hardware-aware accuracy has paired per-sample replay, independent calibration/reporting splits, and realized W6/ADC6/Vm16." if valid else "Publication-grade hardware-aware accuracy requires paired per-sample replay, independent calibration/reporting splits, and realized W6/ADC6/Vm16."),
            {
                "paired_seeds": len(combined_rows),
                "per_sample_rows": len(prediction_rows),
                "expected_per_sample_rows": expected_predictions,
                "publication_mode": publication_mode,
                "calibration_split": calibration_split,
                "reporting_split": reporting_split,
                "split_isolated": split_isolated,
                "fixedpoint_realized": fixedpoint_realized,
            },
        )
    elif selected.get("hardware_aware_accuracy_percent") is None:
        issue(rows, dataset, "hardware_aware_accuracy", "warning", "Checkpoint/per-sample activation replay is not available; clean accuracy is retained without inventing a hardware-aware accuracy.")
    if str(selected.get("publication_ready")) != "True":
        issue(rows, dataset, "publication_scope", "warning", "Results are architecture-level estimates; photonic extraction, measured thermal control, SRAM macros, and post-route PPA remain required for a silicon claim.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HIPSA hybrid reviewer gate")
    p.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    p.add_argument("--input-root", default="results/eval_v3")
    p.add_argument("--device-params", default="configs/device_params.yaml")
    p.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    p.add_argument("--output-root", default="results/eval_v3/combined/review_gate")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.input_root)
    device_cfg = load_yaml(args.device_params)
    hardware_cfg = load_yaml(args.hardware)
    rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        validate_dataset(dataset, root, device_cfg, hardware_cfg, rows)
    transformer_path = root / "combined" / "eval_11" / "transformer_mapping.json"
    if transformer_path.exists():
        transformer = load_json(transformer_path)
        unsupported = set(transformer.get("unsupported_claims", []))
        mapping_ok = bool(transformer.get("summary", {}).get("mapping_feasible_without_optical_plane_change"))
        issue(rows, "global", "spiking_transformer_mapping", "pass" if mapping_ok and {"latency", "energy", "accuracy"}.issubset(unsupported) else "error", "A recent spiking-Transformer topology is mapped without inventing performance or accuracy.", transformer.get("summary", {}))
    else:
        issue(rows, "global", "spiking_transformer_mapping", "error", f"Missing {transformer_path}")

    breadth_path = root / "combined" / "eval_18" / "advanced_snn_mapping.json"
    if breadth_path.exists():
        breadth = load_json(breadth_path)
        summaries = breadth.get("summaries", [])
        families = {str(row.get("family")) for row in summaries}
        claims_safe = {"accuracy", "latency", "energy"}.issubset(set(breadth.get("unsupported_claims", [])))
        breadth_ok = {"residual_convolutional_snn", "spike_driven_transformer"}.issubset(families) and claims_safe
        issue(rows, "global", "advanced_snn_workload_breadth", "pass" if breadth_ok else "error", "SEW-ResNet and spike-driven attention mappings broaden the workload analysis while preserving mapping-only evidence boundaries.", {"workloads": len(summaries), "families": sorted(families), "mapping_only": claims_safe})
    else:
        issue(rows, "global", "advanced_snn_workload_breadth", "error", f"Missing {breadth_path}")

    lock_main_path = root / "combined" / "main_lock_aware_energy.csv"
    lock_main_rows = load_csv_rows(lock_main_path, parse_numbers=True) if lock_main_path.exists() else []
    lock_fractions = {_f(row.get("lock_fraction"), -1.0) for row in lock_main_rows}
    lock_main_ok = len(lock_main_rows) == 6 and {0.0, 0.01, 0.05}.issubset(lock_fractions)
    issue(rows, "global", "lock_aware_main_evaluation", "pass" if lock_main_ok else "error", "The main evaluation reports no-lock lower bounds beside parameterized 1% and 5% lock-aware energy for both workloads.", {"rows": len(lock_main_rows), "lock_fractions": sorted(lock_fractions)})

    resource_path = root / "combined" / "eval_12" / "mechanism_resource_causality.csv"
    floor_path = root / "combined" / "eval_12" / "energy_floor_summary.csv"
    resource_rows = load_csv_rows(resource_path, parse_numbers=True) if resource_path.exists() else []
    expected_targets = {"reference", "optical_transactions", "adc_conversions", "adc_macros", "sram_accesses"}
    actual_targets = {str(r.get("mechanism_target")) for r in resource_rows}
    issue(rows, "global", "mechanism_resource_causality", "pass" if len(resource_rows) == 10 and expected_targets.issubset(actual_targets) else "error", "The four mechanisms expose their targeted transaction/conversion/macro/state resource counts.", {"rows": len(resource_rows), "targets": sorted(actual_targets)})
    floor_rows = load_csv_rows(floor_path, parse_numbers=True) if floor_path.exists() else []
    floor_ok = len(floor_rows) == 2 and all(_f(r.get("controllable_energy_reduction_percent")) > _f(r.get("end_to_end_energy_reduction_percent")) > 0.0 for r in floor_rows)
    issue(rows, "global", "fixed_variable_energy_floor", "pass" if floor_ok else "error", "The energy-floor analysis separates end-to-end savings from the larger controllable-energy reduction.", {"rows": len(floor_rows)})

    trace_schema = PROJECT_ROOT / "configs" / "per_sample_hardware_trace_schema.yaml"
    anchor_schema = PROJECT_ROOT / "configs" / "external_hardware_anchor_schema.yaml"
    issue(rows, "global", "missing_evidence_interfaces", "pass" if trace_schema.exists() and anchor_schema.exists() else "error", "Schemas define the exact per-sample and external PPA/circuit/layout evidence required; their presence is not treated as completed measurement.", {"trace_schema": trace_schema.exists(), "anchor_schema": anchor_schema.exists()})
    ppa_path = PROJECT_ROOT / "results" / "hardware_validation" / "nangate45" / "ppa_summary.json"
    ppa = load_json(ppa_path) if ppa_path.exists() else {}
    ppa_ok = _i(ppa.get("standard_cell_count"), 0) > 0 and _f(ppa.get("mapped_cell_area_mm2"), 0.0) > 0.0 and bool(ppa.get("abc_target_met")) and "not OpenSTA" in str(ppa.get("timing_status", ""))
    issue(rows, "global", "open45nm_control_synthesis_anchor", "pass" if ppa_ok else "error", "Pipelined integrated control has a reproducible open-45nm mapped-cell and ABC pre-layout timing anchor with limitations explicitly recorded.", ppa)
    overhead_path = root / "combined" / "eval_14" / "control_overhead_anchor.csv"
    overhead_rows = load_csv_rows(overhead_path, parse_numbers=True) if overhead_path.exists() else []
    overhead_ok = len(overhead_rows) == 2 and all(_f(row.get("control_area_fraction_percent"), 100.0) < 1.0 and "Leakage-only" in str(row.get("power_scope", "")) for row in overhead_rows)
    issue(rows, "global", "control_overhead_join", "pass" if overhead_ok else "error", "Control area/leakage are joined to both G4/A8 system points without inventing workload dynamic power.", {"rows": len(overhead_rows)})
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "eval_name": "eval_10_review_gate",
        "purpose": "reviewer_focused_hybrid_A_device_v3_system_consistency_gate",
        "created_utc": now_utc(),
        "datasets": args.datasets,
        "num_checks": len(rows),
        "num_errors": sum(r["status"] == "error" for r in rows),
        "num_warnings": sum(r["status"] == "warning" for r in rows),
        "review_verdict": "conditional_pass" if not any(r["status"] == "error" for r in rows) else "needs_fix",
        "scope": "conference-level analytical evidence; no industrial silicon sign-off claim",
        "issues": rows,
    }
    save_json(summary, output / "summary.json")
    save_csv_rows(rows, output / "validation_report.csv")
    for row in rows:
        print(f"[{row['status']}] {row['dataset']} {row['check']}: {row['message']}")
    return 1 if summary["num_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
