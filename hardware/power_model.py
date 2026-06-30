"""Device-calibrated power model for HIPSA eval02/eval03.

The v2 model separates activity domains instead of scaling every component by a
single global activity ratio:

- modulator activity      <- MVM input activity / active SOP ratio
- ADC pool activity       <- comparator/HAPR ADC request activity
- SRAM + digital LIF      <- ADC request / membrane update proxy
- NoC spike traffic       <- input spike activity and LIF output spike activity
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _nested(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_device_param(device_cfg: Mapping[str, Any], path: tuple[str, ...], default: float) -> float:
    val = _nested(device_cfg, *path, default=default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    return min(max(v, 0.0), 1.0)


def estimate_power(
    activity: Mapping[str, Any],
    sop_summary: Mapping[str, Any],
    adc_requests: Mapping[str, Any],
    adc_pool: Mapping[str, Any],
    timing: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    mrr_stabilization_mw: float | None = None,
) -> Dict[str, Any]:
    """Estimate HIPSA component power from device params and separated activity factors."""
    input_act = _bounded(activity.get("input_spike_activity", 0.0))
    active_sop_ratio = _bounded(sop_summary.get("active_sop_ratio", activity.get("active_sop_ratio", 0.0)))
    mvm_input_act = _bounded(sop_summary.get("mvm_input_activity_mean", activity.get("mvm_input_activity_mean", active_sop_ratio)))
    lif_spike_act = _bounded(sop_summary.get("lif_spike_activity", activity.get("lif_spike_activity", 0.0)))
    adc_req_act = _bounded(adc_requests.get("adc_group_request_activity", adc_requests.get("adc_activity_proxy", activity.get("adc_request_activity", activity.get("adc_activity_proxy", 0.0)))))
    adc_util = _bounded(adc_pool.get("adc_macro_utilization", 0.0))

    # For membrane SRAM and digital LIF update, the best available architecture
    # proxy is ADC/comparator request activity, because updates occur when analog
    # partial sums are converted and accumulated. Output spike rate is reported
    # separately as NoC/spike-traffic activity.
    membrane_update_act = max(adc_req_act, lif_spike_act)
    noc_activity = max(input_act, lif_spike_act)

    laser_mw = _get_device_param(device_cfg, ("optical_source", "cw_laser", "power_mw_main_case"), 1473.0)
    leakage_mw = _get_device_param(device_cfg, ("memory_digital", "leakage_misc_io", "reference_power_mw"), 90.0)
    if mrr_stabilization_mw is None:
        mrr_stabilization_mw = _get_device_param(device_cfg, ("mrr_stabilization", "main_case_power_mw"), 0.0)

    lanes = int(adc_pool.get("hapr_output_lanes_total", _nested(hardware_cfg, "hapr_adc_backend", "hapr_output_lanes_total", default=32)))
    adc_macros = int(adc_pool.get("adc_macros", _nested(hardware_cfg, "hapr_adc_backend", "adc_macros", default=16)))

    mod_per_lane = _get_device_param(device_cfg, ("modulation", "binary_modulator_driver", "power_per_active_lane_mw"), 2.25)
    tile_outputs = int(_nested(hardware_cfg, "hapr_adc_backend", "tile_outputs", default=64))
    num_tiles = int(_nested(hardware_cfg, "photonic_tiles", "num_tiles", default=4))
    modulator_mw = mod_per_lane * tile_outputs * num_tiles * mvm_input_act

    # Front-end receiver lanes are mostly provisioned/bias costs in this proxy.
    pd_mw = _get_device_param(device_cfg, ("photodetection_frontend", "photodiode", "power_per_output_mw"), 1.1) * lanes
    tia_mw = _get_device_param(device_cfg, ("photodetection_frontend", "tia", "power_per_hapr_lane_mw"), 3.0) * lanes
    comp_mw = _get_device_param(device_cfg, ("photodetection_frontend", "comparator", "power_per_hapr_lane_mw"), 2.2) * lanes
    hapr_ctrl_mw = _get_device_param(device_cfg, ("photodetection_frontend", "hapr_selection_proxy", "power_per_hapr_lane_mw"), 0.1) * lanes
    pd_tia_comparator_mw = pd_mw + tia_mw + comp_mw + hapr_ctrl_mw

    adc_macro_mw = _get_device_param(device_cfg, ("adc", "conventional_adc_macro", "power_per_macro_mw"), 14.8)
    adc_pool_mw = adc_macro_mw * adc_macros * adc_util

    sram_ref = _get_device_param(device_cfg, ("memory_digital", "sram_register_files", "reference_power_mw"), 243.25)
    noc_ref = _get_device_param(device_cfg, ("memory_digital", "noc_bus_controller_clock", "reference_power_mw"), 77.84)
    lif_ref = _get_device_param(device_cfg, ("memory_digital", "digital_lif_update", "reference_power_mw"), 2.43)
    sram_mw = sram_ref * membrane_update_act
    noc_mw = noc_ref * noc_activity
    digital_lif_mw = lif_ref * membrane_update_act

    component_power_mw = {
        "cw_laser": laser_mw,
        "mrr_stabilization": float(mrr_stabilization_mw),
        "leakage_misc_io": leakage_mw,
        "modulator_driver": modulator_mw,
        "pd_tia_comparator_hapr": pd_tia_comparator_mw,
        "adc_pool": adc_pool_mw,
        "sram_register_files": sram_mw,
        "noc_bus_controller_clock": noc_mw,
        "digital_lif_update": digital_lif_mw,
    }
    total_mw = float(sum(component_power_mw.values()))
    total_w = total_mw / 1000.0
    latency_s = float(timing.get("latency_s_per_image", 0.0))
    energy_uJ = total_w * latency_s * 1e6
    dense_sop = float(sop_summary.get("dense_sop_per_image", 0.0))
    active_sop = float(sop_summary.get("active_sop_per_image", 0.0))
    return {
        "component_power_mw": component_power_mw,
        "total_power_mw": total_mw,
        "total_power_w": total_w,
        "latency_us_per_image": float(timing.get("latency_us_per_image", 0.0)),
        "throughput_images_per_s": float(timing.get("throughput_images_per_s", 0.0)),
        "energy_uJ_per_image": energy_uJ,
        "dense_equivalent_GOPS_per_W": (dense_sop / latency_s / 1e9 / total_w) if latency_s > 0 and total_w > 0 else 0.0,
        "active_GOPS_per_W": (active_sop / latency_s / 1e9 / total_w) if latency_s > 0 and total_w > 0 else 0.0,
        "activity_factors": {
            "input_spike_activity": input_act,
            "mvm_input_activity_mean": mvm_input_act,
            "active_sop_ratio": active_sop_ratio,
            "lif_spike_activity": lif_spike_act,
            "adc_request_activity": adc_req_act,
            "adc_macro_utilization": adc_util,
            "membrane_update_activity_proxy": membrane_update_act,
            "noc_activity_proxy": noc_activity,
        },
    }


def sweep_mrr_power(activity: Mapping[str, Any], sop_summary: Mapping[str, Any], adc_requests: Mapping[str, Any], adc_pool: Mapping[str, Any], timing: Mapping[str, Any], hardware_cfg: Mapping[str, Any], device_cfg: Mapping[str, Any]) -> list[Dict[str, Any]]:
    values = _nested(device_cfg, "mrr_stabilization", "stress_cases_mw", default=[0.0, 100.0, 250.0, 350.0, 500.0])
    rows = []
    for v in values:
        p = estimate_power(activity, sop_summary, adc_requests, adc_pool, timing, hardware_cfg, device_cfg, mrr_stabilization_mw=float(v))
        rows.append({"mrr_stabilization_mw": float(v), **p})
    return rows
