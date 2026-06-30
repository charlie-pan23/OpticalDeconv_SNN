"""Request-driven ADC pool model for HIPSA."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def get_adc_config(hardware_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    backend = hardware_cfg.get("hapr_adc_backend", hardware_cfg.get("hardware", {}).get("hapr_adc_backend", {}))
    if not isinstance(backend, Mapping):
        backend = {}
    return {
        "hapr_group_size": int(backend.get("hapr_group_size", 8)),
        "hapr_output_lanes_total": int(backend.get("hapr_output_lanes_total", 32)),
        "adc_macros": int(backend.get("adc_macros", 16)),
        "comparator_threshold_fs_default": float(backend.get("comparator_threshold_fs_default", 0.02)),
    }


def model_adc_pool(adc_requests: Mapping[str, Any], timing: Mapping[str, Any], hardware_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = get_adc_config(hardware_cfg)
    req_per_img = float(adc_requests.get("adc_requests_per_image", 0.0))
    cycles = float(timing.get("cycles_per_image", timing.get("mvm_cycles_per_image", 1.0)) or 1.0)
    demand_per_cycle = req_per_img / max(cycles, 1.0)
    macros = max(int(cfg["adc_macros"]), 1)
    util = min(1.0, demand_per_cycle / macros)
    service_cycles = req_per_img / macros
    stall_cycles = max(0.0, service_cycles - cycles)
    return {
        **cfg,
        "adc_requests_per_image": req_per_img,
        "cycles_per_image_input": cycles,
        "adc_demand_per_cycle": demand_per_cycle,
        "adc_macro_utilization": util,
        "adc_service_cycles_per_image": service_cycles,
        "adc_stall_cycles_proxy": stall_cycles,
        "adc_is_saturated": bool(demand_per_cycle >= macros),
    }


def sweep_adc_macros(adc_requests: Mapping[str, Any], timing: Mapping[str, Any], hardware_cfg: Mapping[str, Any], macro_values=(4, 8, 16, 32, 64)) -> list[Dict[str, Any]]:
    rows = []
    for m in macro_values:
        cfg = dict(hardware_cfg)
        backend = dict(cfg.get("hapr_adc_backend", {}))
        backend["adc_macros"] = int(m)
        cfg["hapr_adc_backend"] = backend
        row = model_adc_pool(adc_requests, timing, cfg)
        row["sweep_adc_macros"] = int(m)
        rows.append(row)
    return rows
