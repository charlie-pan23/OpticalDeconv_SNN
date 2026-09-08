"""Same-neuron partial-sum mapping for HAPR.

HAPR groups partial sums belonging to the *same output neuron*.  It never
groups different output neurons merely because their channel indices are
adjacent.  This distinction is made explicit in the emitted CSV and checked by
``eval_08_validation.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def map_partial_sums(
    mapping_rows: Iterable[Mapping[str, Any]],
    *,
    hapr_group_size: int,
    time_steps: int,
    num_tiles: int = 4,
) -> List[Dict[str, Any]]:
    """Emit one row per layer/timestep/output-neuron/input-tile partial sum."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in mapping_rows:
        grouped.setdefault(str(row.get("layer", "")), []).append(row)

    output: List[Dict[str, Any]] = []
    group_size = max(int(hapr_group_size), 1)
    physical_tiles = max(int(num_tiles), 1)
    if group_size > physical_tiles or physical_tiles % group_size != 0:
        raise ValueError("HAPR fan-in must divide and cannot exceed the physical tile count")
    for layer, rows in grouped.items():
        first = rows[0]
        input_dim = _int(first.get("input_dim"), 1)
        output_dim = _int(first.get("output_dim"), 1)
        row_tiles = _int(first.get("row_tiles"), 1)
        col_tiles = _int(first.get("col_tiles"), 1)
        for timestep in range(max(int(time_steps), 1)):
            for output_neuron in range(output_dim):
                col_tile = output_neuron // 64
                for row_tile in range(row_tiles):
                    tile = next(
                        r for r in rows
                        if _int(r.get("row_tile")) == row_tile and _int(r.get("col_tile")) == col_tile
                    )
                    physical_batch = row_tile // physical_tiles
                    batch_start = physical_batch * physical_tiles
                    local_row = row_tile - batch_start
                    local_group = local_row // group_size
                    group_start = batch_start + local_group * group_size
                    group_fanin = min(group_size, row_tiles - group_start, physical_tiles - local_group * group_size)
                    groups_per_full_batch = physical_tiles // group_size
                    hapr_group_index = physical_batch * groups_per_full_batch + local_group
                    output.append(
                        {
                            "layer": layer,
                            "timestep": timestep,
                            "spatial_position": "aggregate",
                            "output_neuron": output_neuron,
                            "input_tile": f"r{row_tile}",
                            "output_tile": f"c{col_tile}",
                            "partial_sum_id": f"{layer}:j{output_neuron}:r{row_tile}",
                            "hapr_group_id": f"{layer}:j{output_neuron}:g{hapr_group_index}",
                            "hapr_group_index": hapr_group_index,
                            "group_fanin": group_fanin,
                            "physical_tile": _int(tile.get("mapped_physical_tile")),
                            "physical_batch": physical_batch,
                            "concurrent_spatial_group": 1,
                            "temporal_analog_storage": 0,
                            "same_output_neuron": 1,
                            "input_dim": input_dim,
                            "output_dim": output_dim,
                            "row_tiles": row_tiles,
                            "col_tiles": col_tiles,
                            "trace_provenance": "static_mapping_derived_from_eval01",
                        }
                    )
    return output


def validate_no_cross_neuron_sum(rows: Iterable[Mapping[str, Any]]) -> bool:
    """Return True iff each HAPR group contains exactly one output neuron."""

    groups: Dict[str, set[int]] = {}
    for row in rows:
        groups.setdefault(str(row.get("hapr_group_id", "")), set()).add(
            _int(row.get("output_neuron"), -1)
        )
    return all(len(neurons) == 1 for neurons in groups.values())
