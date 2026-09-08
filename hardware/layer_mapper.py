"""Layer-to-tile mapping for the publication-facing HIPSA model.

The legacy evaluator used a configured utilization factor and an active-SOP
proxy for latency.  This module derives the matrix shape of every traced MVM
layer, including row/column tiling and padding utilization.  It deliberately
accepts the frozen ``eval_01`` summaries so the corrected model can be run
without a checkpoint; when a checkpoint is available the same API can be fed
with a freshly generated summary.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _shape(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [_int(x) for x in value]
    return []


def infer_layer_geometry(
    layer_name: str,
    layer: Mapping[str, Any],
    *,
    num_samples: int,
    time_steps: int,
    array_rows: int = 64,
    array_cols: int = 64,
    num_tiles: int = 4,
) -> Dict[str, Any]:
    """Infer a layer's logical matrix and its physical tile geometry.

    ``dense_sop_per_image`` and the output tensor totals are sufficient to
    recover the flattened receptive-field width even when a checkpoint is not
    packaged.  For a convolution this is ``C_in * K_h * K_w``; for a linear
    layer it is the ordinary input feature count.
    """

    in_shape = _shape(layer.get("input_shape_last"))
    out_shape = _shape(layer.get("output_shape_last"))
    # Robustness traces may omit the last-batch shape and Transformer linear
    # layers naturally produce 3-D [B,T,C] tensors.  In both cases an explicit
    # checkpoint-derived output_dim is sufficient because output totals and
    # dense SOPs recover positions and input width below.
    output_dim = _int(layer.get("output_dim"), 0)
    if output_dim <= 0 and len(out_shape) >= 2:
        output_dim = _int(out_shape[-1] if len(out_shape) == 3 else out_shape[1], 0)
    if output_dim <= 0:
        raise ValueError(f"{layer_name}: output channel/feature count is missing; provide output_shape_last or output_dim")

    output_total = _float(layer.get("mvm_output_total", layer.get("output_total", 0.0)))
    output_total_per_image = output_total / max(int(num_samples), 1)
    if output_total_per_image <= 0:
        # Old traces may not have mvm_output_total.  Recover the number of
        # timestep-spatial output elements from the last batch shape.
        if not out_shape:
            raise ValueError(f"{layer_name}: output totals and output shape are both missing")
        elements_last = math.prod(out_shape)
        output_total_per_image = elements_last * max(int(time_steps), 1)

    dense_sop = _float(layer.get("dense_sop_per_image"), 0.0)
    if dense_sop <= 0:
        dense_sop = _float(layer.get("dense_sop_total"), 0.0) / max(int(num_samples), 1)
    input_dim = int(round(dense_sop / max(output_total_per_image, 1.0)))

    if input_dim <= 0 and len(in_shape) >= 2:
        input_dim = _int(in_shape[1], 1)
    input_dim = max(input_dim, 1)

    output_positions_per_image = output_total_per_image / max(output_dim, 1)
    output_positions_per_timestep = output_positions_per_image / max(int(time_steps), 1)

    row_tiles = int(math.ceil(input_dim / max(int(array_rows), 1)))
    col_tiles = int(math.ceil(output_dim / max(int(array_cols), 1)))
    tile_count = row_tiles * col_tiles

    rows: List[Dict[str, Any]] = []
    for row_tile in range(row_tiles):
        valid_rows = min(array_rows, input_dim - row_tile * array_rows)
        for col_tile in range(col_tiles):
            valid_cols = min(array_cols, output_dim - col_tile * array_cols)
            physical_tile = (row_tile * col_tiles + col_tile) % max(int(num_tiles), 1)
            rows.append(
                {
                    "layer": layer_name,
                    "module_type": str(layer.get("module_type", "unknown")),
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "row_tile": row_tile,
                    "col_tile": col_tile,
                    "input_tile": f"r{row_tile}",
                    "output_tile": f"c{col_tile}",
                    "valid_rows": valid_rows,
                    "valid_cols": valid_cols,
                    "tile_utilization": (valid_rows * valid_cols) / float(array_rows * array_cols),
                    "mapped_physical_tile": physical_tile,
                    "weight_block": f"{layer_name}[{row_tile * array_rows}:{row_tile * array_rows + valid_rows},"
                    f"{col_tile * array_cols}:{col_tile * array_cols + valid_cols}]",
                    "output_positions_per_image": output_positions_per_image,
                    "output_positions_per_timestep": output_positions_per_timestep,
                    "time_steps": int(time_steps),
                    "row_tiles": row_tiles,
                    "col_tiles": col_tiles,
                    "weight_tiles": tile_count,
                    "num_tiles": int(num_tiles),
                    "trace_provenance": "derived_from_eval01_layer_summary",
                }
            )

    return {
        "layer": layer_name,
        "module_type": str(layer.get("module_type", "unknown")),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "output_positions_per_image": output_positions_per_image,
        "output_positions_per_timestep": output_positions_per_timestep,
        "row_tiles": row_tiles,
        "col_tiles": col_tiles,
        "weight_tiles": tile_count,
        "tile_rows": rows,
        "mvm_input_activity": _float(layer.get("mvm_input_activity", layer.get("input_activity", 0.0))),
        "adc_request_activity": _float(layer.get("adc_request_activity", layer.get("adc_activity_proxy", 0.0))),
        "lif_spike_activity": _float(layer.get("lif_spike_activity", layer.get("output_activity", 0.0))),
        "dense_sop_per_image": dense_sop,
        "trace_provenance": "derived_from_eval01_layer_summary",
    }


def build_layer_mapping(
    activity_summary: Mapping[str, Any],
    *,
    array_rows: int = 64,
    array_cols: int = 64,
    num_tiles: int = 4,
) -> List[Dict[str, Any]]:
    """Return one mapping row per layer/timestep-independent physical tile."""

    layers = activity_summary.get("layers", {})
    if not isinstance(layers, Mapping):
        raise TypeError("activity summary must contain a mapping-valued 'layers' field")
    num_samples = _int(activity_summary.get("num_samples"), 1)
    time_steps = _int(activity_summary.get("time_steps"), 1)
    rows: List[Dict[str, Any]] = []
    for name, info in layers.items():
        if not isinstance(info, Mapping):
            continue
        geometry = infer_layer_geometry(
            str(name),
            info,
            num_samples=num_samples,
            time_steps=time_steps,
            array_rows=array_rows,
            array_cols=array_cols,
            num_tiles=num_tiles,
        )
        rows.extend(geometry["tile_rows"])
    return rows


def summarize_mapping(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse tile rows to auditable per-layer mapping summaries."""

    by_layer: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        layer = str(row.get("layer", ""))
        item = by_layer.setdefault(
            layer,
            {
                "layer": layer,
                "input_dim": _int(row.get("input_dim")),
                "output_dim": _int(row.get("output_dim")),
                "row_tiles": _int(row.get("row_tiles")),
                "col_tiles": _int(row.get("col_tiles")),
                "weight_tiles": _int(row.get("weight_tiles")),
                "tile_count": 0,
                "tile_utilization_mean": 0.0,
                "output_positions_per_image": _float(row.get("output_positions_per_image")),
                "output_positions_per_timestep": _float(row.get("output_positions_per_timestep")),
                "trace_provenance": str(row.get("trace_provenance", "")),
            },
        )
        item["tile_count"] += 1
        item["tile_utilization_mean"] += _float(row.get("tile_utilization"))
    for item in by_layer.values():
        item["tile_utilization_mean"] /= max(int(item["tile_count"]), 1)
    return list(by_layer.values())
