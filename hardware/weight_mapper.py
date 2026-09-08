"""Weight residency and physical signed-MRR accounting."""

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


def build_weight_map(
    mapping_summary: Iterable[Mapping[str, Any]],
    *,
    num_tiles: int,
    signed_weight_multiplier: int = 2,
    weight_bits: int = 6,
) -> Dict[str, Any]:
    """Calculate logical weights, signed physical rings, and load rounds.

    The selected execution policy is layer/weight-tile major: a weight tile is
    configured once, then reused across all spatial positions and timesteps.
    There is no hidden timestep-level MRR retuning.
    """

    rows: List[Dict[str, Any]] = []
    total_logical = 0
    total_physical = 0
    total_tiles = 0
    total_rounds = 0
    for item in mapping_summary:
        layer = str(item.get("layer", ""))
        input_dim = _int(item.get("input_dim"), 1)
        output_dim = _int(item.get("output_dim"), 1)
        weight_tiles = _int(item.get("weight_tiles"), 1)
        logical = input_dim * output_dim
        physical = logical * max(int(signed_weight_multiplier), 1)
        load_rounds = int(math.ceil(weight_tiles / max(int(num_tiles), 1)))
        row = {
            "layer": layer,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "logical_weight_count": logical,
            "physical_mrr_elements": physical,
            "signed_weight_multiplier": max(int(signed_weight_multiplier), 1),
            "weight_bits": int(weight_bits),
            "row_tiles": _int(item.get("row_tiles"), 1),
            "col_tiles": _int(item.get("col_tiles"), 1),
            "weight_tiles": weight_tiles,
            "physical_tile_load_rounds": load_rounds,
            "tile_utilization_mean": _float(item.get("tile_utilization_mean")),
            "residency_policy": "layer_weight_tile_major",
            "retune_granularity": "layer/weight_tile_boundary",
        }
        rows.append(row)
        total_logical += logical
        total_physical += physical
        total_tiles += weight_tiles
        total_rounds += load_rounds

    return {
        "layers": rows,
        "total_logical_weight_count": total_logical,
        "total_physical_mrr_elements": total_physical,
        "total_weight_tiles": total_tiles,
        "total_physical_tile_load_rounds": total_rounds,
        "num_tiles": int(num_tiles),
        "signed_weight_multiplier": max(int(signed_weight_multiplier), 1),
        "weight_bits": int(weight_bits),
        "residency_policy": "layer_weight_tile_major",
        "mrr_retuned_every_timestep": False,
    }
