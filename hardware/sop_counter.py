"""SOP counting helpers for HIPSA architecture evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping

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


def _f(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def summarize_sops_from_activity(activity: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a SOP summary from eval01 activity JSON.

    Compatible with both the old schema and the v2 schema. In v2, active SOP is
    driven by `mvm_input_activity`, while `lif_spike_activity` and
    `adc_request_activity` are preserved as separate activity domains.
    """
    layers = activity.get("layers", {}) if isinstance(activity, Mapping) else {}
    layer_rows = []
    dense_total = 0.0
    active_total = 0.0
    lif_active = 0
    lif_total = 0
    adc_active = 0
    adc_total = 0
    mvm_in_active = 0
    mvm_in_total = 0

    for name, info_any in layers.items():
        info = info_any if isinstance(info_any, Mapping) else {}
        dense = _f(info, "dense_sop_total")
        active = _f(info, "active_sop_total")
        dense_total += dense
        active_total += active

        layer_lif_active = int(info.get("lif_spike_active", info.get("output_active", 0)) or 0)
        layer_lif_total = int(info.get("lif_spike_total", info.get("output_total", 0)) or 0)
        layer_adc_active = int(info.get("adc_request_active", 0) or 0)
        layer_adc_total = int(info.get("adc_request_total", 0) or 0)
        layer_mvm_in_active = int(info.get("mvm_input_active", info.get("input_active", 0)) or 0)
        layer_mvm_in_total = int(info.get("mvm_input_total", info.get("input_total", 0)) or 0)
        lif_active += layer_lif_active
        lif_total += layer_lif_total
        adc_active += layer_adc_active
        adc_total += layer_adc_total
        mvm_in_active += layer_mvm_in_active
        mvm_in_total += layer_mvm_in_total

        mvm_input_activity = _f(info, "mvm_input_activity", _f(info, "input_activity"))
        lif_spike_activity = _f(info, "lif_spike_activity", _f(info, "output_activity"))
        adc_request_activity = _f(info, "adc_request_activity", _f(info, "adc_activity_proxy"))
        layer_rows.append({
            "layer": name,
            "dense_sop_total": dense,
            "active_sop_total": active,
            "dense_sop_per_image": _f(info, "dense_sop_per_image"),
            "active_sop_per_image": _f(info, "active_sop_per_image"),
            "mvm_input_activity": mvm_input_activity,
            "lif_spike_activity": lif_spike_activity,
            "adc_request_activity": adc_request_activity,
            "mvm_output_nonzero_activity": _f(info, "mvm_output_nonzero_activity"),
            # Backward-compatible aliases.
            "input_activity": mvm_input_activity,
            "output_activity": lif_spike_activity,
            "adc_activity_proxy": adc_request_activity,
        })

    n = int(activity.get("num_samples", 1) or 1)
    return {
        "num_samples": n,
        "dense_sop_total": dense_total,
        "active_sop_total": active_total,
        "dense_sop_per_image": dense_total / max(n, 1),
        "active_sop_per_image": active_total / max(n, 1),
        "active_sop_ratio": active_total / dense_total if dense_total else 0.0,
        "mvm_input_activity_mean": (mvm_in_active / mvm_in_total) if mvm_in_total else _f(activity, "mvm_input_activity_mean", _f(activity, "active_sop_ratio")),
        "lif_spike_activity": (lif_active / lif_total) if lif_total else _f(activity, "lif_spike_activity"),
        "adc_request_activity": (adc_active / adc_total) if adc_total else _f(activity, "adc_request_activity", _f(activity, "adc_activity_proxy")),
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
