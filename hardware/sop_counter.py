"""SOP counting helpers for HIPSA architecture evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


def dense_sop_from_shapes(module: nn.Module, output_shape: tuple[int, ...]) -> float:
    """Compute dense SOPs for one timestep/module output shape."""
    num_outputs = 1
    for v in output_shape:
        num_outputs *= int(v)
    if isinstance(module, nn.Conv2d):
        k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        return float(num_outputs * (module.in_channels // module.groups) * k_h * k_w)
    if isinstance(module, nn.Linear):
        return float(num_outputs * module.in_features)
    return 0.0


def summarize_sops_from_activity(activity: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a SOP summary from eval01 activity JSON."""
    layers = activity.get("layers", {}) if isinstance(activity, Mapping) else {}
    layer_rows = []
    dense_total = 0.0
    active_total = 0.0
    for name, info in layers.items():
        dense = float(info.get("dense_sop_total", 0.0))
        active = float(info.get("active_sop_total", 0.0))
        dense_total += dense
        active_total += active
        layer_rows.append({
            "layer": name,
            "dense_sop_total": dense,
            "active_sop_total": active,
            "dense_sop_per_image": float(info.get("dense_sop_per_image", 0.0)),
            "active_sop_per_image": float(info.get("active_sop_per_image", 0.0)),
            "input_activity": float(info.get("input_activity", 0.0)),
            "output_activity": float(info.get("output_activity", 0.0)),
        })
    n = int(activity.get("num_samples", 1) or 1)
    return {
        "num_samples": n,
        "dense_sop_total": dense_total,
        "active_sop_total": active_total,
        "dense_sop_per_image": dense_total / max(n, 1),
        "active_sop_per_image": active_total / max(n, 1),
        "active_sop_ratio": active_total / dense_total if dense_total else 0.0,
        "layers": layer_rows,
    }


def count_model_parameters_by_mvm_layer(model: nn.Module, include_fc2: bool = False) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name == "fc2" and not include_fc2:
                continue
            out[name] = int(sum(p.numel() for p in module.parameters()))
    return out
