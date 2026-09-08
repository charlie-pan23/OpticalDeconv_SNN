"""HAPR fan-in legality and edge-case reference checks for HIPSA.

This module separates mapping legality from analog timing closure.  It defines
which HAPR group sizes are representable by the frozen four-tile macroarchitecture,
checks same-neuron/domain invariants, and provides guards for incomplete or
aggregate traces.  It does not prove circuit behavior.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class HAPRLegalityConfig:
    physical_tiles: int = 4
    outputs_per_tile: int = 64
    post_hapr_lanes: int = 64
    adc_service_capacity: int = 80
    temporal_analog_storage: bool = False
    supported_group_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)

    def validate(self) -> None:
        for name in (
            "physical_tiles",
            "outputs_per_tile",
            "post_hapr_lanes",
            "adc_service_capacity",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.supported_group_sizes:
            raise ValueError("supported_group_sizes must not be empty")
        if any(int(group) <= 0 for group in self.supported_group_sizes):
            raise ValueError("supported_group_sizes must be positive")


@dataclass(frozen=True)
class HAPRCase:
    group_size: int
    input_dim: int = 256
    output_dim: int = 64
    row_tiles: int | None = None
    trace_rows: tuple[Mapping[str, Any], ...] = ()
    expected_partition_ids: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class HAPRLegalityResult:
    group_size: int
    input_dim: int
    output_dim: int
    row_tiles: int
    output_tiles: int
    output_active_lanes: int
    output_allocated_lanes: int
    output_tail_lanes: int
    row_tile_tail: bool
    row_tile_tail_fanin: int
    groups_per_row_batch: int
    physical_fanin_limit: int
    status: str
    legal: bool
    same_neuron_grouping: bool
    partition_contract_valid: bool
    runtime_trace_guard_valid: bool
    reasons: tuple[str, ...]
    claim_boundary: str = (
        "architecture_level_mapping_legality_and_admission_constraints; "
        "not_mux_tia_aperture_or_circuit_timing_closure"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _context(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("sample_id", row.get("sample", ""))),
        int(row.get("timestep_id", row.get("timestep", 0))),
        str(row.get("layer_id", row.get("layer", ""))),
        str(row.get("hapr_group_id", "")),
    )


def _domain(row: Mapping[str, Any]) -> tuple[str, int]:
    return (
        str(row.get("output_tile_id", row.get("output_tile", ""))),
        int(row.get("output_neuron_id", row.get("output_neuron", -1))),
    )


def validate_same_neuron_grouping(rows: Iterable[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Check group domains and reject cross-neuron/tile HAPR grouping."""

    group_domains: dict[tuple[str, int, str, str], tuple[str, int]] = {}
    errors: list[str] = []
    for index, row in enumerate(rows):
        context = _context(row)
        domain = _domain(row)
        if not context[0] or not context[2] or not context[3]:
            errors.append(f"row {index}: missing sample/layer/HAPR group identity")
            continue
        if domain[0] == "" or domain[1] < 0:
            errors.append(f"row {index}: invalid output tile/neuron identity")
            continue
        previous = group_domains.setdefault(context, domain)
        if previous != domain:
            errors.append(
                "HAPR group crosses output tile/neuron boundary: "
                f"context={context!r}, first={previous!r}, current={domain!r}"
            )
    return not errors, errors


def validate_partition_contract(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_partition_ids: Iterable[str | int] | None = None,
) -> tuple[bool, list[str]]:
    """Check duplicate/missing/foreign partial sums within each state context."""

    materialized = list(rows)
    errors: list[str] = []
    expected = None if expected_partition_ids is None else {str(item) for item in expected_partition_ids}
    by_context: dict[tuple[str, int, str, str, str, int], list[str]] = {}
    for index, row in enumerate(materialized):
        partition = str(row.get("partition_id", row.get("partial_sum_id", "")))
        if not partition:
            errors.append(f"row {index}: missing partition_id")
            continue
        key = _context(row) + _domain(row)
        by_context.setdefault(key, []).append(partition)
    for key, partitions in by_context.items():
        duplicates = sorted({item for item in partitions if partitions.count(item) > 1})
        if duplicates:
            errors.append(f"context {key!r}: duplicate partitions {duplicates!r}")
        if expected is not None:
            received = set(partitions)
            missing = sorted(expected - received)
            foreign = sorted(received - expected)
            if missing:
                errors.append(f"context {key!r}: missing partitions {missing!r}")
            if foreign:
                errors.append(f"context {key!r}: unexpected partitions {foreign!r}")
    return not errors, errors


def validate_runtime_trace_guard(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_real_runtime: bool = True,
) -> tuple[bool, list[str]]:
    """Guard publication-only analyses against aggregate/proxy traces."""

    materialized = list(rows)
    required = {
        "request_id",
        "sample_id",
        "timestep_id",
        "layer_id",
        "output_tile_id",
        "output_neuron_id",
        "partition_id",
        "hapr_group_id",
    }
    errors: list[str] = []
    if not materialized:
        errors.append("trace is empty")
        return False, errors
    for index, row in enumerate(materialized):
        missing = sorted(field for field in required if field not in row)
        if missing:
            errors.append(f"row {index}: missing request-level fields {missing!r}")
        provenance = str(row.get("trace_provenance", ""))
        if require_real_runtime and "real_runtime" not in provenance:
            errors.append(
                f"row {index}: trace_provenance must contain 'real_runtime', got {provenance!r}"
            )
    return not errors, errors


def evaluate_hapr_case(
    case: HAPRCase,
    config: HAPRLegalityConfig = HAPRLegalityConfig(),
) -> HAPRLegalityResult:
    config.validate()
    group_size = _positive_int(case.group_size, "group_size")
    input_dim = _positive_int(case.input_dim, "input_dim")
    output_dim = _positive_int(case.output_dim, "output_dim")
    row_tiles = (
        _positive_int(case.row_tiles, "row_tiles")
        if case.row_tiles is not None
        else math.ceil(input_dim / config.outputs_per_tile)
    )
    output_tiles = math.ceil(output_dim / config.outputs_per_tile)
    output_active_lanes = output_dim
    output_allocated_lanes = output_tiles * config.outputs_per_tile
    output_tail_lanes = output_allocated_lanes - output_active_lanes
    row_tile_tail_fanin = row_tiles % group_size
    row_tile_tail = row_tile_tail_fanin != 0
    groups_per_row_batch = math.ceil(row_tiles / group_size)
    reasons: list[str] = []

    physical_fanin_valid = (
        group_size <= config.physical_tiles
        and config.physical_tiles % group_size == 0
    )
    if not physical_fanin_valid:
        reasons.append(
            f"group_size_{group_size}_exceeds_or_does_not_divide_{config.physical_tiles}_physical_tiles"
        )

    if output_tail_lanes:
        reasons.append(
            f"output_dimension_tail_{output_tail_lanes}_inactive_lanes"
        )
    if row_tile_tail:
        reasons.append(
            f"row_tile_tail_partial_fanin_{row_tile_tail_fanin}_of_{group_size}"
        )
    if output_active_lanes > config.adc_service_capacity:
        reasons.append(
            f"active_output_lanes_{output_active_lanes}_require_multi_window_admission"
        )
    if config.temporal_analog_storage:
        reasons.append("temporal_analog_storage_enabled_not_selected_point")

    grouping_valid, grouping_errors = validate_same_neuron_grouping(case.trace_rows)
    if case.trace_rows and not grouping_valid:
        reasons.extend(grouping_errors)
    partition_valid, partition_errors = validate_partition_contract(
        case.trace_rows,
        expected_partition_ids=case.expected_partition_ids or None,
    )
    if case.trace_rows and not partition_valid:
        reasons.extend(partition_errors)
    runtime_valid, runtime_errors = validate_runtime_trace_guard(
        case.trace_rows,
        require_real_runtime=True,
    ) if case.trace_rows else (False, ["no runtime trace supplied"])
    if case.trace_rows and not runtime_valid:
        reasons.extend(runtime_errors)

    legal = physical_fanin_valid and grouping_valid and partition_valid
    if group_size == 1 and legal:
        status = "supported_baseline"
    elif group_size == 2 and legal:
        status = "supported_candidate"
    elif group_size == 4 and legal:
        status = "formal_selected_point"
    elif legal:
        status = "supported_with_explicit_edge_conditions"
    else:
        status = "rejected_by_explicit_constraint"

    return HAPRLegalityResult(
        group_size=group_size,
        input_dim=input_dim,
        output_dim=output_dim,
        row_tiles=row_tiles,
        output_tiles=output_tiles,
        output_active_lanes=output_active_lanes,
        output_allocated_lanes=output_allocated_lanes,
        output_tail_lanes=output_tail_lanes,
        row_tile_tail=row_tile_tail,
        row_tile_tail_fanin=row_tile_tail_fanin,
        groups_per_row_batch=groups_per_row_batch,
        physical_fanin_limit=config.physical_tiles,
        status=status,
        legal=legal,
        same_neuron_grouping=grouping_valid,
        partition_contract_valid=partition_valid,
        runtime_trace_guard_valid=runtime_valid,
        reasons=tuple(reasons),
    )


def build_hapr_legality_matrix(
    config: HAPRLegalityConfig = HAPRLegalityConfig(),
    *,
    group_sizes: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Build the formal G1/G2/G4/G8/G16 matrix plus edge-case rows."""

    config.validate()
    selected_groups = tuple(group_sizes or config.supported_group_sizes)
    nominal = [
        evaluate_hapr_case(
            HAPRCase(group_size=group, input_dim=256, output_dim=64),
            config,
        ).to_dict()
        for group in selected_groups
    ]
    edge_cases = [
        evaluate_hapr_case(HAPRCase(group_size=4, input_dim=256, output_dim=65), config).to_dict(),
        evaluate_hapr_case(HAPRCase(group_size=4, input_dim=130, output_dim=64), config).to_dict(),
        evaluate_hapr_case(HAPRCase(group_size=4, input_dim=65, output_dim=127), config).to_dict(),
    ]
    expected = {1: "supported_baseline", 2: "supported_candidate", 4: "formal_selected_point"}
    matrix_checks = {
        "g1_supported_baseline": next((row["status"] == expected[1] and row["legal"] for row in nominal if row["group_size"] == 1), False),
        "g2_supported_candidate": next((row["status"] == expected[2] and row["legal"] for row in nominal if row["group_size"] == 2), False),
        "g4_formal_selected_point": next((row["status"] == expected[4] and row["legal"] for row in nominal if row["group_size"] == 4), False),
        "g8_rejected_by_four_tile_fanin": next((not row["legal"] and any("does_not_divide" in reason or "exceeds" in reason for reason in row["reasons"]) for row in nominal if row["group_size"] == 8), False),
        "g16_rejected_by_four_tile_fanin": next((not row["legal"] and any("does_not_divide" in reason or "exceeds" in reason for reason in row["reasons"]) for row in nominal if row["group_size"] == 16), False),
        "non_multiple_output_dimension_is_explicit_tail": edge_cases[0]["output_tail_lanes"] == 63 and edge_cases[0]["legal"],
        "row_tile_tail_is_explicit": edge_cases[1]["row_tile_tail"] and edge_cases[1]["legal"],
        "small_row_tile_tail_is_explicit": edge_cases[2]["row_tile_tail"] and edge_cases[2]["legal"],
    }
    return {
        "config": asdict(config),
        "nominal_matrix": nominal,
        "edge_cases": edge_cases,
        "matrix_checks": matrix_checks,
        "matrix_passed": all(matrix_checks.values()),
        "claim_boundary": (
            "architecture_level_HAPR_mapping_legality_and_edge_case_analysis; "
            "not_analog_settling_aperture_or_circuit_timing_closure"
        ),
    }


# Descriptive aliases used by eval and downstream audits.
run_hapr_legality_matrix = build_hapr_legality_matrix
