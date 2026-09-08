"""Cycle-based transaction scheduler for the corrected HIPSA model.

The scheduler operates on aggregate bins because the packaged archive contains
eval_01 aggregate counters rather than per-sample event tensors.  Counts are
still scheduled in integer system cycles; every row records the exact number of
opportunities, issued transactions, and the cycle interval used for the
aggregate workload.  The provenance is explicit so these rows are not mistaken
for a silicon trace.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _active_group_probability(activity: float, valid_rows: int, event_gating: bool) -> float:
    if not event_gating:
        return 1.0
    p = _clip(activity)
    return 1.0 - (1.0 - p) ** max(int(valid_rows), 1)


def _layer_map_rows(mapping_rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    result: Dict[str, List[Mapping[str, Any]]] = {}
    for row in mapping_rows:
        result.setdefault(str(row.get("layer", "")), []).append(row)
    return result


def build_transaction_trace(
    activity_summary: Mapping[str, Any],
    mapping_rows: Iterable[Mapping[str, Any]],
    *,
    hapr_group_size: int,
    num_tiles: int,
    array_rows: int = 64,
    array_cols: int = 64,
    event_gating: bool = True,
) -> Dict[str, Any]:
    """Build a deterministic schedule with spatially realizable HAPR groups.

    HAPR combines only same-neuron partial sums that are simultaneously
    produced by different physical tiles.  Without temporal analog storage,
    its fan-in must divide and cannot exceed ``num_tiles``.
    """

    group_size = max(int(hapr_group_size), 1)
    physical_parallelism = max(int(num_tiles), 1)
    if group_size > physical_parallelism or physical_parallelism % group_size != 0:
        raise ValueError(
            f"HAPR G={group_size} is not a spatial divisor of {physical_parallelism} "
            "physical tiles; temporal analog storage is not modeled"
        )

    layers = activity_summary.get("layers", {})
    if not isinstance(layers, Mapping):
        raise TypeError("activity_summary.layers must be a mapping")
    time_steps = max(_int(activity_summary.get("time_steps"), 1), 1)
    mapping_by_layer = _layer_map_rows(mapping_rows)
    trace: List[Dict[str, Any]] = []
    layer_summary: List[Dict[str, Any]] = []
    cursor = 0
    total_opportunities = 0.0
    total_issued = 0.0
    total_valid_sop = 0.0
    total_adc_requests = 0.0
    peak_utilization = 0.0

    for layer_name, info in layers.items():
        if not isinstance(info, Mapping):
            continue
        tile_rows = mapping_by_layer.get(str(layer_name), [])
        if not tile_rows:
            continue
        first = tile_rows[0]
        layer_activity = _float(info.get("mvm_input_activity", info.get("input_activity", 0.0)))
        adc_activity = _clip(_float(info.get("adc_request_activity", info.get("adc_activity_proxy", 0.0))))
        output_dim = _int(first.get("output_dim"), 1)
        row_tiles = _int(first.get("row_tiles"), 1)
        col_tiles = _int(first.get("col_tiles"), 1)
        positions_t = _float(first.get("output_positions_per_timestep"), 1.0)
        layer_opp = 0.0
        layer_issued = 0.0
        layer_requests = 0.0
        layer_cycles = 0
        # For each output-column tile, up to num_tiles input-row tiles are
        # resident concurrently.  The batch is partitioned into spatial HAPR
        # groups, so every analog sum is same-neuron and same-cycle.
        hapr_groups = sum(
            int(math.ceil(min(physical_parallelism, row_tiles - start) / group_size))
            for start in range(0, row_tiles, physical_parallelism)
        )
        for timestep in range(time_steps):
            for col_tile_index in range(col_tiles):
                col_rows = sorted(
                    (t for t in tile_rows if _int(t.get("col_tile")) == col_tile_index),
                    key=lambda r: _int(r.get("row_tile")),
                )
                for batch_start in range(0, len(col_rows), physical_parallelism):
                    batch = col_rows[batch_start:batch_start + physical_parallelism]
                    batch_issued: List[int] = []
                    batch_requests = 0.0
                    batch_valid_sop = 0.0
                    batch_opportunities = 0
                    batch_issue_probability = 0.0
                    tile_ids: List[str] = []
                    tile_issue: List[tuple[Mapping[str, Any], float, int]] = []
                    for tile in batch:
                        row_tile = _int(tile.get("row_tile"))
                        col_tile = _int(tile.get("col_tile"))
                        valid_rows = max(_int(tile.get("valid_rows"), array_rows), 1)
                        valid_cols = max(_int(tile.get("valid_cols"), array_cols), 1)
                        opportunities = max(0, int(round(positions_t)))
                        issue_prob = _active_group_probability(layer_activity, valid_rows, event_gating)
                        issued = opportunities if not event_gating else int(round(opportunities * issue_prob))
                        batch_issued.append(issued)
                        batch_opportunities += opportunities
                        batch_issue_probability = max(batch_issue_probability, issue_prob)
                        batch_valid_sop += issued * valid_rows * valid_cols
                        tile_ids.append(f"r{row_tile}c{col_tile}")
                        tile_issue.append((tile, issue_prob, issued))

                    for local_start in range(0, len(tile_issue), group_size):
                        subgroup = tile_issue[local_start:local_start + group_size]
                        if not any(issued > 0 for _, _, issued in subgroup):
                            continue
                        group_issue_miss = 1.0
                        for _, issue_prob, _ in subgroup:
                            group_issue_miss *= 1.0 - issue_prob
                        group_tx_prob = 1.0 - group_issue_miss
                        group_fanin = len(subgroup)
                        same_neuron_request_prob = 1.0 - (1.0 - adc_activity) ** group_fanin
                        outputs_in_col = max(min(array_cols, output_dim - col_tile_index * array_cols), 0)
                        batch_requests += positions_t * outputs_in_col * group_tx_prob * same_neuron_request_prob

                    duration = max(batch_issued, default=0)
                    start = cursor
                    end = cursor + max(duration - 1, 0)
                    cursor += duration
                    issued_total = sum(batch_issued)
                    layer_opp += batch_opportunities
                    layer_issued += issued_total
                    layer_requests += batch_requests
                    total_adc_requests += batch_requests
                    total_valid_sop += batch_valid_sop
                    util = batch_valid_sop / max(duration * physical_parallelism * array_rows * array_cols, 1)
                    peak_utilization = max(peak_utilization, min(max(util, 0.0), 1.0))
                    trace.append(
                        {
                            "sample": "aggregate_mean",
                            "timestep": timestep,
                            "layer": str(layer_name),
                            "batch_index": batch_start // physical_parallelism,
                            "output_col_tile": col_tile_index,
                            "logical_tiles": ";".join(tile_ids),
                            "logical_tile_count": len(batch),
                            "transaction_opportunities": batch_opportunities,
                            "transaction_issued": issued_total,
                            "transaction_issue_probability_max": batch_issue_probability,
                            "start_cycle": start,
                            "end_cycle": end,
                            "duration_cycles": duration,
                            "adc_requests": batch_requests,
                            "hapr_group_size": group_size,
                            "hapr_groups_per_neuron": hapr_groups,
                            "hapr_execution": "spatial_same_neuron_same_cycle",
                            "temporal_analog_storage": 0,
                            "trace_provenance": "aggregate_eval01_counters_batched_over_physical_tiles",
                        }
                    )

        layer_cycles = sum(int(row.get("duration_cycles", 0)) for row in trace if row.get("layer") == str(layer_name))
        layer_summary.append(
            {
                "layer": str(layer_name),
                "transaction_opportunities_per_image": layer_opp,
                "transactions_issued_per_image": layer_issued,
                "adc_requests_per_image": layer_requests,
                "schedule_cycles_per_image": layer_cycles,
                "derived_tile_utilization": (layer_issued / max(layer_cycles * num_tiles, 1)),
                "mvm_input_activity": layer_activity,
                "adc_request_activity": adc_activity,
                "event_gating": int(event_gating),
                "trace_provenance": "aggregate_eval01_counters_batched_over_physical_tiles",
            }
        )
        total_opportunities += layer_opp
        total_issued += layer_issued

    return {
        "trace": trace,
        "layer_summary": layer_summary,
        "total_cycles_per_image": int(cursor),
        "transaction_opportunities_per_image": total_opportunities,
        "transactions_issued_per_image": total_issued,
        "adc_requests_per_image": total_adc_requests,
        "derived_effective_utilization": total_valid_sop / max(cursor * num_tiles * array_rows * array_cols, 1),
        "peak_bin_utilization": peak_utilization,
        "event_gating": bool(event_gating),
        "schedule_granularity": "aggregate layer/timestep/tile bins",
        "trace_provenance": "derived_from_eval01_aggregate_counters_not_per_sample_event_trace; output-column tiles batch same-neuron row partials over physical tiles",
        "hapr_physical_fanin_limit": physical_parallelism,
        "hapr_execution": "spatial_same_neuron_same_cycle_no_temporal_analog_storage",
    }
