"""ADC request proxy model for HIPSA HAPR/TIA output groups."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping


def _shape(info: Mapping[str, Any]) -> list[int]:
    s = info.get("output_shape_last") or info.get("output_shape") or []
    return [int(v) for v in s] if isinstance(s, (list, tuple)) else []


def estimate_layer_adc_requests(info: Mapping[str, Any], hapr_group_size: int = 8) -> Dict[str, Any]:
    """Estimate requests from layer output shape and comparator activity proxy.

    For Conv outputs [B,C,H,W], groups are ceil(C/G) per spatial location.  For
    Linear outputs [B,F], groups are ceil(F/G).  The hook call count already
    contains timestep repetitions, so group opportunities are multiplied by the
    number of calls.
    """
    G = max(int(hapr_group_size), 1)
    shape = _shape(info)
    calls = int(info.get("calls", 0) or 0)
    adc_activity = float(info.get("adc_activity_proxy", info.get("output_activity", 0.0)))
    if len(shape) == 4:
        b, c, h, w = shape
        opportunities = calls * b * h * w * math.ceil(c / G)
    elif len(shape) == 2:
        b, f = shape
        opportunities = calls * b * math.ceil(f / G)
    else:
        total = int(info.get("output_total", 0) or 0)
        opportunities = math.ceil(total / G)
    requests = opportunities * adc_activity
    return {
        "hapr_group_size": G,
        "adc_group_opportunities": float(opportunities),
        "adc_activity_proxy": adc_activity,
        "adc_requests_total": float(requests),
    }


def estimate_adc_requests(activity: Mapping[str, Any], hapr_group_size: int = 8) -> Dict[str, Any]:
    layers = activity.get("layers", {}) if isinstance(activity, Mapping) else {}
    n = int(activity.get("num_samples", 1) or 1)
    by_layer: Dict[str, Any] = {}
    total = 0.0
    opportunities = 0.0
    for name, info in layers.items():
        row = estimate_layer_adc_requests(info, hapr_group_size)
        by_layer[name] = row
        total += float(row["adc_requests_total"])
        opportunities += float(row["adc_group_opportunities"])
    return {
        "num_samples": n,
        "hapr_group_size": int(hapr_group_size),
        "adc_requests_total": total,
        "adc_requests_per_image": total / max(n, 1),
        "adc_group_opportunities_total": opportunities,
        "adc_activity_proxy": total / opportunities if opportunities else 0.0,
        "layers": by_layer,
    }
