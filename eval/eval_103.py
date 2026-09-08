"""HIPSA queue/partition/state semantic validation (formal eval_103 entry).

This entry composes the request-level ADC queue with the partition-completion
state-commit FSM.  It uses deterministic semantic fixtures rather than claiming
a full dataset/runtime request replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.adc_request_queue import ADCQueueConfig, ADCRequest, simulate_request_queue
from hardware.state_commit_fsm import (
    StateCommitConfig,
    StateCommitSpec,
    simulate_state_commit_fsm,
)
from hardware.adc_evidence_boundary import (
    ADCEvidenceConfig,
    build_adc_evidence_boundary,
    capacity_sensitivity,
)
from hardware.hapr_legality import (
    HAPRLegalityConfig,
    build_hapr_legality_matrix,
    validate_runtime_trace_guard,
)


SELECTED_QUEUE_CONFIG = ADCQueueConfig(
    adc_macros=8,
    samples_per_macro_per_cycle=10,
    fifo_depth=4096,
    post_hapr_lanes=64,
    default_conversion_latency_cycles=1,
    temporal_analog_storage=False,
    drain_tail=True,
)
STATE_CONFIG = StateCommitConfig(
    decay=0.5,
    input_scale=0.5,
    threshold=1.0,
    membrane_bits=16,
    fractional_bits=8,
    reset_mode="zero",
    initial_last_timestep=-1,
)
HAPR_LEGALITY_CONFIG = HAPRLegalityConfig(
    physical_tiles=4,
    outputs_per_tile=64,
    post_hapr_lanes=64,
    adc_service_capacity=80,
    temporal_analog_storage=False,
)
ADC_EVIDENCE_CONFIG = ADCEvidenceConfig(
    adc_macros=8,
    samples_per_macro_per_cycle=10,
    post_hapr_lanes=64,
    architecture_clock_ghz=1.0,
    adc_power_mw_per_macro=14.8,
    temporal_analog_storage=False,
    mux_ratio="64:8",
)


def _logical_workload() -> list[dict[str, Any]]:
    values = {
        (0, 0): (0.7, 0.8, 0.9),
        (0, 1): (0.1, 0.2, 0.3),
        (1, 0): (0.2, 0.3, 0.3),
        (1, 1): (0.5, 0.6, 0.7),
    }
    rows: list[dict[str, Any]] = []
    for neuron in (0, 1):
        for timestep in (0, 1):
            for partition_index, value in enumerate(values[(neuron, timestep)]):
                partition = f"p{partition_index}"
                rows.append(
                    {
                        "request_id": f"n{neuron}:t{timestep}:{partition}",
                        "sample_id": "semantic-sample-0",
                        "timestep_id": timestep,
                        "layer_id": "semantic_lif",
                        "output_tile_id": "tile0",
                        "output_neuron_id": neuron,
                        "partition_id": partition,
                        "hapr_group_id": f"semantic_lif:j{neuron}:{partition}",
                        "arrival_cycle": timestep,
                        "partial_sum_value": value,
                    }
                )
    return rows


def _latency_assignment(variant: str, count: int, seed: int) -> list[int]:
    latencies = list(range(1, count + 1))
    if variant == "canonical":
        return latencies
    if variant == "reverse":
        return list(reversed(latencies))
    if variant == "random":
        random.Random(seed).shuffle(latencies)
        return latencies
    raise ValueError(f"unknown completion-order variant: {variant}")


def build_requests(variant: str, seed: int) -> list[ADCRequest]:
    rows = _logical_workload()
    latencies = _latency_assignment(variant, len(rows), seed)
    return [
        ADCRequest(
            request_id=row["request_id"],
            sample_id=row["sample_id"],
            timestep_id=row["timestep_id"],
            layer_id=row["layer_id"],
            output_tile_id=row["output_tile_id"],
            output_neuron_id=row["output_neuron_id"],
            partition_id=row["partition_id"],
            hapr_group_id=row["hapr_group_id"],
            arrival_cycle=row["arrival_cycle"],
            conversion_latency_cycles=latency,
            partial_sum_value=row["partial_sum_value"],
        )
        for row, latency in zip(rows, latencies)
    ]


def build_specs() -> list[StateCommitSpec]:
    return [
        StateCommitSpec(
            sample_id="semantic-sample-0",
            timestep_id=timestep,
            layer_id="semantic_lif",
            output_neuron_id=neuron,
            output_tile_id="tile0",
            expected_partition_ids=("p0", "p1", "p2"),
        )
        for neuron in (0, 1)
        for timestep in (0, 1)
    ]


def semantic_signature(result: dict[str, Any]) -> dict[str, Any]:
    commit_rows = sorted(
        (
            {
                "sample_id": row["sample_id"],
                "timestep_id": row["timestep_id"],
                "layer_id": row["layer_id"],
                "output_neuron_id": row["output_neuron_id"],
                "membrane_after_commit": row["membrane_after_commit"],
                "spike": row["spike"],
                "reset_applied": row["reset_applied"],
            }
            for row in result["commit_trace"]
        ),
        key=lambda row: (
            row["sample_id"],
            row["layer_id"],
            row["output_neuron_id"],
            row["timestep_id"],
        ),
    )
    final_states = sorted(
        (
            {
                "sample_id": row["sample_id"],
                "layer_id": row["layer_id"],
                "output_neuron_id": row["output_neuron_id"],
                "membrane": row["membrane"],
                "last_committed_timestep": row["last_committed_timestep"],
            }
            for row in result["final_states"]
        ),
        key=lambda row: (row["sample_id"], row["layer_id"], row["output_neuron_id"]),
    )
    return {
        "commits": commit_rows,
        "final_states": final_states,
        "spike_train": sorted(
            result["spike_train"],
            key=lambda row: (
                row["sample_id"],
                row["layer_id"],
                row["output_neuron_id"],
                row["timestep_id"],
            ),
        ),
        "reset_events": sorted(
            result["reset_events"],
            key=lambda row: (
                row["sample_id"],
                row["layer_id"],
                row["output_neuron_id"],
                row["timestep_id"],
            ),
        ),
        "commit_count": result["summary"]["commit_count"],
    }


def _capacity_requests(count: int) -> list[ADCRequest]:
    return [
        ADCRequest(
            request_id=f"capacity-r{index:02d}",
            sample_id="capacity-sample",
            timestep_id=0,
            layer_id="capacity_layer",
            output_tile_id=f"tile{index // 16}",
            output_neuron_id=index,
            partition_id="p0",
            hapr_group_id=f"capacity_layer:j{index}:g0",
            arrival_cycle=0,
            partial_sum_value=0.0,
        )
        for index in range(count)
    ]


def run_semantic_validation(seed: int = 42) -> dict[str, Any]:
    selected_capacity = simulate_request_queue(
        _capacity_requests(64), SELECTED_QUEUE_CONFIG
    )
    a6_capacity = simulate_request_queue(
        _capacity_requests(64),
        ADCQueueConfig(
            adc_macros=6,
            samples_per_macro_per_cycle=10,
            fifo_depth=4096,
            post_hapr_lanes=64,
            default_conversion_latency_cycles=1,
            temporal_analog_storage=False,
            drain_tail=True,
        ),
    )

    variants: dict[str, dict[str, Any]] = {}
    request_trace: list[dict[str, Any]] = []
    receive_trace: list[dict[str, Any]] = []
    transition_trace: list[dict[str, Any]] = []
    commit_trace: list[dict[str, Any]] = []
    for variant in ("canonical", "reverse", "random"):
        queue = simulate_request_queue(
            build_requests(variant, seed), SELECTED_QUEUE_CONFIG
        )
        state = simulate_state_commit_fsm(queue["requests"], build_specs(), STATE_CONFIG)
        variants[variant] = {
            "queue_summary": queue["summary"],
            "state_summary": state["summary"],
            "signature": semantic_signature(state),
        }
        request_trace.extend({"variant": variant, **row} for row in queue["requests"])
        receive_trace.extend({"variant": variant, **row} for row in state["receive_trace"])
        transition_trace.extend(
            {"variant": variant, **row} for row in state["transition_trace"]
        )
        commit_trace.extend({"variant": variant, **row} for row in state["commit_trace"])

    signatures = [variants[name]["signature"] for name in ("canonical", "reverse", "random")]
    order_invariant = signatures[0] == signatures[1] == signatures[2]

    canonical_queue = simulate_request_queue(
        build_requests("canonical", seed), SELECTED_QUEUE_CONFIG
    )
    missing_rows = [
        row
        for row in canonical_queue["requests"]
        if not (
            row["output_neuron_id"] == 0
            and row["timestep_id"] == 0
            and str(row["partition_id"]) == "p2"
        )
    ]
    missing_result = simulate_state_commit_fsm(missing_rows, build_specs(), STATE_CONFIG)
    missing_target = next(
        row
        for row in missing_result["incomplete_contexts"]
        if row["output_neuron_id"] == 0 and row["timestep_id"] == 0
    )
    target_threshold_rows = [
        row
        for row in missing_result["commit_trace"]
        if row["output_neuron_id"] == 0 and row["timestep_id"] == 0
    ]

    duplicate_partition_rejected = False
    duplicate_error = ""
    duplicate_rows = list(canonical_queue["requests"])
    duplicate = dict(duplicate_rows[0])
    duplicate["request_id"] = "duplicate-partition-request"
    duplicate["completion_sequence"] = max(
        int(row["completion_sequence"]) for row in duplicate_rows
    ) + 1
    duplicate["completion_cycle"] = max(
        int(row["completion_cycle"]) for row in duplicate_rows
    ) + 1
    try:
        simulate_state_commit_fsm(duplicate_rows + [duplicate], build_specs(), STATE_CONFIG)
    except ValueError as exc:
        duplicate_error = str(exc)
        duplicate_partition_rejected = "duplicate partition" in duplicate_error

    cross_neuron_hapr_rejected = False
    cross_neuron_error = ""
    illegal_requests = [
        ADCRequest(
            request_id="illegal-0",
            sample_id="s",
            timestep_id=0,
            layer_id="l",
            output_tile_id="tile0",
            output_neuron_id=0,
            partition_id="p0",
            hapr_group_id="shared-group",
            arrival_cycle=0,
        ),
        ADCRequest(
            request_id="illegal-1",
            sample_id="s",
            timestep_id=0,
            layer_id="l",
            output_tile_id="tile0",
            output_neuron_id=1,
            partition_id="p0",
            hapr_group_id="shared-group",
            arrival_cycle=0,
        ),
    ]
    try:
        simulate_request_queue(illegal_requests, SELECTED_QUEUE_CONFIG)
    except ValueError as exc:
        cross_neuron_error = str(exc)
        cross_neuron_hapr_rejected = "crosses output tile/neuron boundary" in cross_neuron_error

    hapr_legality = build_hapr_legality_matrix(HAPR_LEGALITY_CONFIG)
    adc_evidence = build_adc_evidence_boundary(ADC_EVIDENCE_CONFIG)
    adc_capacity_sensitivity = capacity_sensitivity(
        adc_macros=(6, 8),
        samples_per_macro_per_cycle=(8, 10, 12),
        post_hapr_lanes=64,
    )
    aggregate_trace_guard_valid, aggregate_trace_guard_errors = validate_runtime_trace_guard(
        [{"aggregate_cycle": 0, "lane_count": 64}],
        require_real_runtime=True,
    )
    runtime_schema_probe = {
        "request_id": "runtime-probe-0",
        "sample_id": "probe",
        "timestep_id": 0,
        "layer_id": "probe_layer",
        "output_tile_id": "tile0",
        "output_neuron_id": 0,
        "partition_id": "p0",
        "hapr_group_id": "probe:g0",
        "trace_provenance": "real_runtime:schema_probe",
    }
    runtime_schema_guard_valid, runtime_schema_guard_errors = validate_runtime_trace_guard(
        [runtime_schema_probe],
        require_real_runtime=True,
    )
    sensitivity_by_point = {
        (row["adc_macros"], row["samples_per_macro_per_cycle"]): row
        for row in adc_capacity_sensitivity
    }

    validation_checks = {
        "g4_a8_accepts_64_requests": bool(
            selected_capacity["summary"]["accepted"] == 64
            and selected_capacity["summary"]["served"] == 64
            and selected_capacity["summary"]["rejected"] == 0
            and selected_capacity["summary"]["service_capacity_per_cycle"] == 80
            and selected_capacity["summary"]["selected_point_valid"]
        ),
        "a6_exposes_four_request_shortfall": bool(
            a6_capacity["summary"]["accepted"] == 60
            and a6_capacity["summary"]["rejected"] == 4
            and a6_capacity["summary"]["same_window_capacity_shortfall"] == 4
            and not a6_capacity["summary"]["selected_point_valid"]
        ),
        "all_order_variants_queue_valid": all(
            variants[name]["queue_summary"]["selected_point_valid"]
            for name in variants
        ),
        "all_order_variants_state_valid": all(
            variants[name]["state_summary"]["state_commit_valid"]
            for name in variants
        ),
        "completion_order_invariant": order_invariant,
        "missing_partition_prevents_target_threshold": bool(
            missing_target["missing_partition_ids"] == ["p2"]
            and not target_threshold_rows
            and not missing_result["summary"]["state_commit_valid"]
        ),
        "duplicate_partition_rejected": duplicate_partition_rejected,
        "cross_neuron_hapr_group_rejected": cross_neuron_hapr_rejected,
        "no_temporal_analog_storage": not SELECTED_QUEUE_CONFIG.temporal_analog_storage,
        "evidence_boundary_explicit": bool(
            not selected_capacity["summary"]["circuit_timing_closed"]
            and not variants["canonical"]["state_summary"]["commit_engine_bandwidth_modeled"]
        ),
        "hapr_legality_matrix_passed": bool(hapr_legality["matrix_passed"]),
        "g8_g16_rejected_by_explicit_constraint": bool(
            hapr_legality["matrix_checks"]["g8_rejected_by_four_tile_fanin"]
            and hapr_legality["matrix_checks"]["g16_rejected_by_four_tile_fanin"]
        ),
        "non64_output_tail_explicit": bool(
            hapr_legality["matrix_checks"]["non_multiple_output_dimension_is_explicit_tail"]
        ),
        "row_tile_tail_explicit": bool(
            hapr_legality["matrix_checks"]["row_tile_tail_is_explicit"]
            and hapr_legality["matrix_checks"]["small_row_tile_tail_is_explicit"]
        ),
        "aggregate_trace_rejected_by_runtime_guard": bool(
            not aggregate_trace_guard_valid and bool(aggregate_trace_guard_errors)
        ),
        "runtime_schema_guard_accepts_provenance_labeled_trace": bool(
            runtime_schema_guard_valid and not runtime_schema_guard_errors
        ),
        "adc_evidence_boundary_explicit": bool(
            adc_evidence["claim_boundary"]
            and adc_evidence["safe_claim"] == "architecture-level capacity/admission analysis"
        ),
        "no_timing_closure_claim": bool(
            not adc_evidence["timing_closure"]
            and not adc_evidence["physically_closed"]
            and not adc_evidence["silicon_validated"]
            and "timing closure" in adc_evidence["forbidden_claims"]
        ),
        "mux_settling_not_modeled": bool(
            next(item for item in adc_evidence["items"] if item["name"] == "mux_switching_settling")["status"]
            == "not_modeled"
        ),
        "adc_aperture_not_modeled": bool(
            next(item for item in adc_evidence["items"] if item["name"] == "adc_aperture")["status"]
            == "not_modeled"
        ),
        "tia_recovery_not_modeled": bool(
            next(item for item in adc_evidence["items"] if item["name"] == "tia_output_recovery")["status"]
            == "not_modeled"
        ),
        "conversion_latency_pipeline_modeled": bool(
            next(item for item in adc_evidence["items"] if item["name"] == "conversion_latency")["status"]
            == "modeled_pipeline_semantics"
        ),
        "adc_power_conditional": bool(
            next(item for item in adc_evidence["items"] if item["name"] == "adc_power_14_8_mw_per_macro")["status"]
            == "conditional_parameter_anchor"
        ),
        "capacity_sensitivity_g4_a8_is_80": bool(
            sensitivity_by_point[(8, 10)]["nominal_slots_per_window"] == 80
            and sensitivity_by_point[(8, 10)]["same_window_admission_feasible"]
        ),
        "capacity_sensitivity_a6_is_60": bool(
            sensitivity_by_point[(6, 10)]["nominal_slots_per_window"] == 60
            and not sensitivity_by_point[(6, 10)]["same_window_admission_feasible"]
        ),
    }
    return {
        "selected_capacity": selected_capacity,
        "a6_capacity": a6_capacity,
        "variants": variants,
        "missing_partition_case": missing_result,
        "duplicate_partition_error": duplicate_error,
        "cross_neuron_hapr_error": cross_neuron_error,
        "validation_checks": validation_checks,
        "validation_passed": all(validation_checks.values()),
        "request_trace": request_trace,
        "receive_trace": receive_trace,
        "transition_trace": transition_trace,
        "commit_trace": commit_trace,
        "hapr_legality": hapr_legality,
        "adc_evidence": adc_evidence,
        "adc_capacity_sensitivity": adc_capacity_sensitivity,
        "aggregate_trace_guard_errors": aggregate_trace_guard_errors,
        "runtime_schema_guard_errors": runtime_schema_guard_errors,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, tuple, dict))
                        else value
                    )
                    for field, value in row.items()
                }
            )


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def write_outputs(result: dict[str, Any], output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "evaluation": "eval_103_queue_state_hapr_semantic_validation",
        "fixture_scope": "deterministic_semantic_reference_not_dataset_runtime_replay",
        "seed": seed,
        "macroarchitecture": {
            "photonic_tiles": 4,
            "array_per_tile": "64x64",
            "hapr_group_size": 4,
            "post_hapr_lanes": 64,
            "adc_macros": SELECTED_QUEUE_CONFIG.adc_macros,
            "samples_per_macro_per_cycle": SELECTED_QUEUE_CONFIG.samples_per_macro_per_cycle,
            "service_slots_per_window": SELECTED_QUEUE_CONFIG.service_capacity_per_cycle,
            "architecture_clock_ghz": 1,
            "temporal_analog_storage": False,
        },
        "state_commit": STATE_CONFIG.__dict__,
    }
    try:
        import yaml

        (output_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8"
        )
    except ImportError:
        (output_dir / "config_snapshot.yaml").write_text(
            json.dumps(config_snapshot, indent=2) + "\n", encoding="utf-8"
        )

    runtime = {
        "evaluation": "eval_103",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "script": "eval/eval_103.py",
        "repository_relative_output": output_dir.relative_to(REPO_ROOT).as_posix()
        if output_dir.is_relative_to(REPO_ROOT)
        else str(output_dir),
        "evidence_class": "architecture_level_semantic_reference_model",
    }
    _write_json(output_dir / "runtime_provenance.json", runtime)

    metrics = {
        "validation_passed": result["validation_passed"],
        "validation_checks": result["validation_checks"],
        "selected_g4_a8": result["selected_capacity"]["summary"],
        "a6_counterexample": result["a6_capacity"]["summary"],
        "hapr_legality": result["hapr_legality"],
        "adc_evidence": result["adc_evidence"],
        "adc_capacity_sensitivity": result["adc_capacity_sensitivity"],
        "completion_order_variants": {
            name: {
                "queue_summary": data["queue_summary"],
                "state_summary": data["state_summary"],
                "semantic_signature": data["signature"],
            }
            for name, data in result["variants"].items()
        },
        "missing_partition_summary": result["missing_partition_case"]["summary"],
        "claim_boundary": (
            "request_admission_partition_completion_and_state_commit_semantics; "
            "not_dataset_runtime_replay_mux_tia_aperture_commit_bandwidth_or_physical_timing_closure"
        ),
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "validation.json",
        {
            "passed": result["validation_passed"],
            "checks": result["validation_checks"],
            "duplicate_partition_error": result["duplicate_partition_error"],
            "cross_neuron_hapr_error": result["cross_neuron_hapr_error"],
            "missing_contexts": result["missing_partition_case"]["incomplete_contexts"],
            "hapr_matrix_checks": result["hapr_legality"]["matrix_checks"],
            "aggregate_trace_guard_errors": result["aggregate_trace_guard_errors"],
            "runtime_schema_guard_errors": result["runtime_schema_guard_errors"],
            "adc_evidence_claim_boundary": result["adc_evidence"]["claim_boundary"],
        },
    )
    _write_json(output_dir / "hapr_legality.json", result["hapr_legality"])
    _write_json(output_dir / "adc_evidence_boundary.json", result["adc_evidence"])
    _write_csv(output_dir / "hapr_legality_matrix.csv", result["hapr_legality"]["nominal_matrix"])
    _write_csv(output_dir / "hapr_edge_cases.csv", result["hapr_legality"]["edge_cases"])
    _write_csv(output_dir / "adc_capacity_sensitivity.csv", result["adc_capacity_sensitivity"])
    _write_csv(output_dir / "request_trace.csv", result["request_trace"])
    _write_csv(output_dir / "partition_receive_trace.csv", result["receive_trace"])
    _write_csv(output_dir / "state_transition_trace.csv", result["transition_trace"])
    _write_csv(output_dir / "state_commit_trace.csv", result["commit_trace"])
    _write_csv(output_dir / "trace.csv", result["commit_trace"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/eval_v10"),
        help="Output root; relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--dataset",
        default="semantic_reference",
        help="Result namespace. This eval uses a semantic fixture, not dataset replay.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_dir = output_root / args.dataset / "eval_103"
    result = run_semantic_validation(seed=args.seed)
    write_outputs(result, output_dir, args.seed)
    print(json.dumps({
        "evaluation": "eval_103",
        "output_dir": str(output_dir),
        "validation_passed": result["validation_passed"],
        "checks": result["validation_checks"],
    }, indent=2, sort_keys=True))
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
