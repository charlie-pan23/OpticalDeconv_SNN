"""
eval_02.py

Device-calibrated latency / throughput / power / energy model.

This script does NOT run model inference.
It consumes eval_01 saved activity traces:

  results/eval_v2/<dataset>/eval_01/summary.json

and generates:

  results/eval_v2/<dataset>/eval_02/
    summary.json
    latency_energy_summary.json
    throughput_summary.csv
    power_breakdown.csv
    area_breakdown.csv
    adc_pool_summary.csv
    layer_adc_requests.csv
    derived_counts.json
    config_snapshot.yaml
    run_manifest.json

Design rules:
1. Do not use a fixed total power claim.
2. Do not claim a new ADC circuit.
3. Do not instantiate a full multi-bit DAC for binary spikes.
4. Do not hide continuous per-ring MRR thermal locking in the main case.
5. Derive energy and efficiency from component power and modeled latency.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.result_io import (
    dataset_eval_dir,
    load_json,
    load_yaml,
    save_csv_rows,
    save_json,
    save_run_manifest,
    save_yaml,
)

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("HIPSA")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def section(mapping: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Read top-level section, with fallback to mapping['hardware'][key]."""

    if isinstance(mapping.get(key), Mapping):
        return dict(mapping[key])

    hw = mapping.get("hardware", {})
    if isinstance(hw, Mapping) and isinstance(hw.get(key), Mapping):
        return dict(hw[key])

    return {}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def bounded(value: Any, default: float = 0.0) -> float:
    v = as_float(value, default)
    return min(max(v, 0.0), 1.0)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if float(b) != 0 else float(default)


def power_param(
    device_cfg: Mapping[str, Any],
    path: Tuple[str, ...],
    default: float,
) -> float:
    return as_float(nested_get(device_cfg, *path, default=default), default)


def optional_area_param(
    device_cfg: Mapping[str, Any],
    path: Tuple[str, ...],
) -> Optional[float]:
    value = nested_get(device_cfg, *path, default=None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Architecture / count extraction
# ---------------------------------------------------------------------------


def derive_architecture_counts(
    hardware_cfg: Mapping[str, Any],
    hapr_group_size_override: Optional[int] = None,
    adc_macros_override: Optional[int] = None,
) -> Dict[str, Any]:
    photonic = section(hardware_cfg, "photonic_tiles")
    backend = section(hardware_cfg, "hapr_adc_backend")

    num_tiles = as_int(photonic.get("num_tiles"), 4)
    array_size = photonic.get("logical_array_size", [64, 64])
    if not isinstance(array_size, (list, tuple)) or len(array_size) != 2:
        array_size = [64, 64]

    array_rows = as_int(array_size[0], 64)
    array_cols = as_int(array_size[1], 64)

    tile_outputs = as_int(backend.get("tile_outputs"), array_rows)

    hapr_group_size = as_int(
        hapr_group_size_override
        if hapr_group_size_override is not None
        else backend.get("hapr_group_size"),
        8,
    )
    hapr_group_size = max(hapr_group_size, 1)

    hapr_lanes_per_tile = math.ceil(tile_outputs / hapr_group_size)
    hapr_output_lanes_total = as_int(
        backend.get("hapr_output_lanes_total"),
        num_tiles * hapr_lanes_per_tile,
    )

    # If group-size is overridden, recompute lanes from the override.
    if hapr_group_size_override is not None:
        hapr_output_lanes_total = num_tiles * hapr_lanes_per_tile

    adc_macros = as_int(
        adc_macros_override
        if adc_macros_override is not None
        else backend.get("adc_macros"),
        16,
    )
    adc_macros = max(adc_macros, 1)

    # Before HAPR, each tile still has tile_outputs photodetection outputs.
    photodetector_outputs_total = num_tiles * tile_outputs

    # Use the same count as a conservative binary modulation lane proxy.
    modulator_lanes_total = num_tiles * tile_outputs

    return {
        "num_tiles": num_tiles,
        "array_rows": array_rows,
        "array_cols": array_cols,
        "tile_outputs": tile_outputs,
        "hapr_group_size": hapr_group_size,
        "hapr_lanes_per_tile": hapr_lanes_per_tile,
        "hapr_output_lanes_total": hapr_output_lanes_total,
        "adc_macros": adc_macros,
        "photodetector_outputs_total": photodetector_outputs_total,
        "modulator_lanes_total": modulator_lanes_total,
    }


def derive_timing_config(hardware_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    photonic = section(hardware_cfg, "photonic_tiles")

    peak_sop_per_cycle_total = as_float(
        photonic.get("peak_sop_per_cycle_total"),
        16384.0,
    )
    effective_clock_hz = as_float(
        photonic.get("effective_clock_hz"),
        1.0e9,
    )
    effective_utilization = as_float(
        photonic.get("effective_utilization"),
        0.40,
    )

    realized_active_rate = as_float(
        photonic.get("realized_active_rate_sop_per_s"),
        peak_sop_per_cycle_total * effective_clock_hz * effective_utilization,
    )

    return {
        "peak_sop_per_cycle_total": peak_sop_per_cycle_total,
        "effective_clock_hz": effective_clock_hz,
        "effective_utilization": effective_utilization,
        "realized_active_rate_sop_per_s": realized_active_rate,
        "realized_active_rate_tsop_per_s": realized_active_rate / 1.0e12,
    }


# ---------------------------------------------------------------------------
# ADC request / ADC pool modeling
# ---------------------------------------------------------------------------


def element_to_group_probability(element_activity: float, group_size: int) -> float:
    """Probability that at least one element in a HAPR group requests ADC.

    If each element requests with probability p, a group of G requests with:
      1 - (1 - p)^G

    This is a simple architecture-level proxy.
    """

    p = bounded(element_activity)
    g = max(int(group_size), 1)
    return 1.0 - (1.0 - p) ** g


def output_channels_from_shape(shape: Any) -> Optional[int]:
    if not isinstance(shape, (list, tuple)):
        return None

    if len(shape) == 4:
        # Conv output: [B, C, H, W]
        return as_int(shape[1], 0)

    if len(shape) == 2:
        # Linear output: [B, F]
        return as_int(shape[1], 0)

    return None


def estimate_layer_adc_requests(
    layer_name: str,
    layer: Mapping[str, Any],
    hapr_group_size: int,
) -> Dict[str, Any]:
    output_total = as_float(
        layer.get(
            "adc_request_total",
            layer.get("mvm_output_total", layer.get("output_total", 0.0)),
        ),
        0.0,
    )

    output_shape = layer.get("output_shape_last") or layer.get("output_shape")
    output_channels = output_channels_from_shape(output_shape)

    if output_channels and output_channels > 0:
        groups_per_output_vector = math.ceil(output_channels / hapr_group_size)
        vectors_total = output_total / output_channels
        group_opportunities = vectors_total * groups_per_output_vector
    else:
        group_opportunities = math.ceil(output_total / max(hapr_group_size, 1))

    element_request_activity = bounded(
        layer.get(
            "adc_request_activity",
            layer.get("adc_activity_proxy", layer.get("output_activity", 0.0)),
        )
    )
    group_request_activity = element_to_group_probability(
        element_request_activity,
        hapr_group_size,
    )

    adc_requests_total = group_opportunities * group_request_activity

    return {
        "layer": layer_name,
        "hapr_group_size": int(hapr_group_size),
        "output_shape_last": output_shape,
        "output_channels": output_channels,
        "output_elements_total": output_total,
        "adc_element_request_activity": element_request_activity,
        "adc_group_request_activity": group_request_activity,
        "adc_group_opportunities_total": group_opportunities,
        "adc_requests_total": adc_requests_total,
    }


def estimate_adc_requests(
    activity_summary: Mapping[str, Any],
    hapr_group_size: int,
) -> Dict[str, Any]:
    layers = activity_summary.get("layers", {})
    if not isinstance(layers, Mapping):
        layers = {}

    num_samples = as_int(activity_summary.get("num_samples"), 1)
    layer_rows: List[Dict[str, Any]] = []

    total_requests = 0.0
    total_opportunities = 0.0

    for layer_name, layer_any in layers.items():
        if not isinstance(layer_any, Mapping):
            continue

        row = estimate_layer_adc_requests(
            layer_name=layer_name,
            layer=layer_any,
            hapr_group_size=hapr_group_size,
        )
        layer_rows.append(row)

        total_requests += as_float(row["adc_requests_total"])
        total_opportunities += as_float(row["adc_group_opportunities_total"])

    group_activity = safe_div(total_requests, total_opportunities)

    return {
        "num_samples": num_samples,
        "hapr_group_size": int(hapr_group_size),
        "adc_requests_total": total_requests,
        "adc_requests_per_image": safe_div(total_requests, num_samples),
        "adc_group_opportunities_total": total_opportunities,
        "adc_group_opportunities_per_image": safe_div(total_opportunities, num_samples),
        "adc_group_request_activity": group_activity,
        "adc_element_request_activity": bounded(
            activity_summary.get("adc_request_activity", 0.0)
        ),
        "layers": layer_rows,
    }


def model_adc_pool(
    adc_requests: Mapping[str, Any],
    timing_base: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> Dict[str, Any]:
    adc_macros = max(as_int(counts.get("adc_macros"), 16), 1)
    requests_per_image = as_float(adc_requests.get("adc_requests_per_image"), 0.0)

    base_cycles = as_float(timing_base.get("base_cycles_per_image"), 0.0)
    base_cycles = max(base_cycles, 1.0)

    demand_per_cycle = requests_per_image / base_cycles
    macro_utilization = min(1.0, demand_per_cycle / adc_macros)

    service_cycles = requests_per_image / adc_macros
    stall_cycles = max(0.0, service_cycles - base_cycles)

    return {
        "adc_macros": int(adc_macros),
        "adc_requests_per_image": requests_per_image,
        "base_cycles_per_image": base_cycles,
        "adc_demand_per_cycle": demand_per_cycle,
        "adc_macro_utilization": macro_utilization,
        "adc_service_cycles_per_image": service_cycles,
        "adc_stall_cycles_proxy": stall_cycles,
        "adc_is_saturated": bool(demand_per_cycle >= adc_macros),
    }


# ---------------------------------------------------------------------------
# Timing / power / area
# ---------------------------------------------------------------------------


def estimate_timing(
    activity_summary: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    adc_pool: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    timing_cfg = derive_timing_config(hardware_cfg)

    active_sop_per_image = as_float(
        activity_summary.get("active_sop_per_image"),
        0.0,
    )
    dense_sop_per_image = as_float(
        activity_summary.get("dense_sop_per_image"),
        0.0,
    )

    active_rate = max(as_float(timing_cfg["realized_active_rate_sop_per_s"]), 1.0)
    clock_hz = max(as_float(timing_cfg["effective_clock_hz"]), 1.0)

    base_latency_s = active_sop_per_image / active_rate
    base_cycles = base_latency_s * clock_hz

    stall_cycles = 0.0
    if adc_pool is not None:
        stall_cycles = max(as_float(adc_pool.get("adc_stall_cycles_proxy"), 0.0), 0.0)

    total_cycles = base_cycles + stall_cycles
    latency_s = total_cycles / clock_hz

    return {
        **timing_cfg,
        "dense_sop_per_image": dense_sop_per_image,
        "active_sop_per_image": active_sop_per_image,
        "base_cycles_per_image": base_cycles,
        "adc_stall_cycles_proxy": stall_cycles,
        "cycles_per_image": total_cycles,
        "base_latency_s_per_image": base_latency_s,
        "latency_s_per_image": latency_s,
        "base_latency_us_per_image": base_latency_s * 1.0e6,
        "latency_us_per_image": latency_s * 1.0e6,
        "throughput_images_per_s": (1.0 / latency_s) if latency_s > 0 else 0.0,
    }


def choose_modulator_activity(
    activity_summary: Mapping[str, Any],
    source: str,
) -> float:
    if source == "model_input_activity":
        return bounded(activity_summary.get("model_input_activity", 0.0))

    if source == "mvm_input_activity":
        return bounded(activity_summary.get("mvm_input_activity", 0.0))

    if source == "active_sop_ratio":
        return bounded(activity_summary.get("active_sop_ratio", 0.0))

    if source == "always_on":
        return 1.0

    raise ValueError(
        f"Unknown modulator activity source: {source}. "
        "Use model_input_activity, mvm_input_activity, active_sop_ratio, or always_on."
    )


def add_power_row(
    rows: List[Dict[str, Any]],
    component: str,
    power_mw: float,
    count: Optional[float],
    activity: Optional[float],
    note: str,
) -> None:
    rows.append(
        {
            "component": component,
            "power_mw": float(power_mw),
            "power_w": float(power_mw) / 1000.0,
            "count": count if count is not None else "",
            "activity_factor": activity if activity is not None else "",
            "note": note,
        }
    )


def estimate_power(
    activity_summary: Mapping[str, Any],
    timing: Mapping[str, Any],
    adc_pool: Mapping[str, Any],
    counts: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    *,
    modulator_activity_source: str,
    adc_power_mode: str,
    mrr_stabilization_mw: Optional[float],
) -> Dict[str, Any]:
    model_input_activity = bounded(activity_summary.get("model_input_activity", 0.0))
    mvm_input_activity = bounded(activity_summary.get("mvm_input_activity", 0.0))
    active_sop_ratio = bounded(activity_summary.get("active_sop_ratio", 0.0))
    lif_spike_activity = bounded(activity_summary.get("lif_spike_activity", 0.0))

    adc_group_request_activity = bounded(
        adc_pool.get("adc_macro_utilization", activity_summary.get("adc_request_activity", 0.0))
    )
    adc_macro_utilization = bounded(adc_pool.get("adc_macro_utilization", 0.0))

    modulator_activity = choose_modulator_activity(
        activity_summary,
        source=modulator_activity_source,
    )

    membrane_update_activity = max(adc_group_request_activity, lif_spike_activity)
    noc_activity = max(model_input_activity, lif_spike_activity)

    rows: List[Dict[str, Any]] = []

    laser_mw = power_param(
        device_cfg,
        ("optical_source", "cw_laser", "power_mw_main_case"),
        1473.0,
    )
    add_power_row(
        rows,
        "cw_laser",
        laser_mw,
        count=1,
        activity=1.0,
        note="CW laser link-budget power; not timestep-gated.",
    )

    if mrr_stabilization_mw is None:
        mrr_stabilization_mw = power_param(
            device_cfg,
            ("mrr_stabilization", "main_case_power_mw"),
            0.0,
        )
    add_power_row(
        rows,
        "mrr_stabilization",
        float(mrr_stabilization_mw),
        count="main_case",
        activity=1.0,
        note="Main case excludes continuous per-ring thermal locking.",
    )

    leakage_mw = power_param(
        device_cfg,
        ("memory_digital", "leakage_misc_io", "reference_power_mw"),
        90.0,
    )
    add_power_row(
        rows,
        "leakage_misc_io",
        leakage_mw,
        count=1,
        activity=1.0,
        note="Static leakage / miscellaneous I/O proxy.",
    )

    mod_per_lane_mw = power_param(
        device_cfg,
        ("modulation", "binary_modulator_driver", "power_per_active_lane_mw"),
        2.25,
    )
    mod_lanes = as_int(counts.get("modulator_lanes_total"), 256)
    modulator_mw = mod_per_lane_mw * mod_lanes * modulator_activity
    add_power_row(
        rows,
        "binary_modulator_driver",
        modulator_mw,
        count=mod_lanes,
        activity=modulator_activity,
        note=f"Activity source: {modulator_activity_source}; no full multi-bit DAC.",
    )

    pd_per_output_mw = power_param(
        device_cfg,
        ("photodetection_frontend", "photodiode", "power_per_output_mw"),
        1.1,
    )
    pd_outputs = as_int(counts.get("photodetector_outputs_total"), 256)
    pd_mw = pd_per_output_mw * pd_outputs
    add_power_row(
        rows,
        "photodiodes",
        pd_mw,
        count=pd_outputs,
        activity=1.0,
        note="PDs are counted before HAPR reduction.",
    )

    hapr_lanes = as_int(counts.get("hapr_output_lanes_total"), 32)

    tia_per_lane_mw = power_param(
        device_cfg,
        ("photodetection_frontend", "tia", "power_per_hapr_lane_mw"),
        3.0,
    )
    tia_mw = tia_per_lane_mw * hapr_lanes
    add_power_row(
        rows,
        "tia",
        tia_mw,
        count=hapr_lanes,
        activity=1.0,
        note="TIA lanes after HAPR current summing.",
    )

    comparator_per_lane_mw = power_param(
        device_cfg,
        ("photodetection_frontend", "comparator", "power_per_hapr_lane_mw"),
        2.2,
    )
    comparator_mw = comparator_per_lane_mw * hapr_lanes
    add_power_row(
        rows,
        "comparators",
        comparator_mw,
        count=hapr_lanes,
        activity=1.0,
        note="Comparator lanes after HAPR.",
    )

    selection_per_lane_mw = power_param(
        device_cfg,
        ("photodetection_frontend", "hapr_selection_proxy", "power_per_hapr_lane_mw"),
        0.1,
    )
    selection_mw = selection_per_lane_mw * hapr_lanes
    add_power_row(
        rows,
        "hapr_selection_proxy",
        selection_mw,
        count=hapr_lanes,
        activity=1.0,
        note="Local selection / request proxy after HAPR.",
    )

    adc_per_macro_mw = power_param(
        device_cfg,
        ("adc", "conventional_adc_macro", "power_per_macro_mw"),
        14.8,
    )
    adc_macros = as_int(counts.get("adc_macros"), 16)

    adc_activity_scaled_mw = adc_per_macro_mw * adc_macros * adc_macro_utilization
    adc_all_biased_mw = adc_per_macro_mw * adc_macros

    if adc_power_mode == "activity_scaled":
        adc_mw = adc_activity_scaled_mw
        adc_note = "Activity-scaled conventional ADC macro pool."
    elif adc_power_mode == "all_biased":
        adc_mw = adc_all_biased_mw
        adc_note = "All ADC macros biased upper-bound case."
    else:
        raise ValueError("--adc-power-mode must be activity_scaled or all_biased")

    add_power_row(
        rows,
        "adc_pool",
        adc_mw,
        count=adc_macros,
        activity=adc_macro_utilization if adc_power_mode == "activity_scaled" else 1.0,
        note=adc_note,
    )

    sram_ref_mw = power_param(
        device_cfg,
        ("memory_digital", "sram_register_files", "reference_power_mw"),
        243.25,
    )
    sram_mw = sram_ref_mw * membrane_update_activity
    add_power_row(
        rows,
        "sram_register_files",
        sram_mw,
        count="reference",
        activity=membrane_update_activity,
        note="Scaled by ADC/membrane update activity proxy.",
    )

    noc_ref_mw = power_param(
        device_cfg,
        ("memory_digital", "noc_bus_controller_clock", "reference_power_mw"),
        77.84,
    )
    noc_mw = noc_ref_mw * noc_activity
    add_power_row(
        rows,
        "noc_bus_controller_clock",
        noc_mw,
        count="reference",
        activity=noc_activity,
        note="Scaled by max(input activity, LIF spike activity).",
    )

    lif_ref_mw = power_param(
        device_cfg,
        ("memory_digital", "digital_lif_update", "reference_power_mw"),
        2.43,
    )
    lif_mw = lif_ref_mw * membrane_update_activity
    add_power_row(
        rows,
        "digital_lif_update",
        lif_mw,
        count="reference",
        activity=membrane_update_activity,
        note="Scaled by membrane update activity proxy.",
    )

    total_power_mw = sum(as_float(row["power_mw"]) for row in rows)
    total_power_w = total_power_mw / 1000.0

    latency_s = as_float(timing.get("latency_s_per_image"), 0.0)
    energy_uJ = total_power_w * latency_s * 1.0e6

    active_sop_per_image = as_float(activity_summary.get("active_sop_per_image"), 0.0)
    dense_sop_per_image = as_float(activity_summary.get("dense_sop_per_image"), 0.0)

    active_gops_per_w = (
        active_sop_per_image / latency_s / 1.0e9 / total_power_w
        if latency_s > 0 and total_power_w > 0
        else 0.0
    )
    dense_equiv_gops_per_w = (
        dense_sop_per_image / latency_s / 1.0e9 / total_power_w
        if latency_s > 0 and total_power_w > 0
        else 0.0
    )

    component_power_mw = {
        str(row["component"]): as_float(row["power_mw"])
        for row in rows
    }

    return {
        "component_power_mw": component_power_mw,
        "power_rows": rows,
        "total_power_mw": total_power_mw,
        "total_power_w": total_power_w,
        "latency_us_per_image": as_float(timing.get("latency_us_per_image"), 0.0),
        "throughput_images_per_s": as_float(timing.get("throughput_images_per_s"), 0.0),
        "energy_uJ_per_image": energy_uJ,
        "active_GOPS_per_W": active_gops_per_w,
        "dense_equivalent_GOPS_per_W": dense_equiv_gops_per_w,
        "adc_power_mode": adc_power_mode,
        "adc_activity_scaled_power_mw": adc_activity_scaled_mw,
        "adc_all_biased_power_mw": adc_all_biased_mw,
        "activity_factors": {
            "model_input_activity": model_input_activity,
            "mvm_input_activity": mvm_input_activity,
            "active_sop_ratio": active_sop_ratio,
            "lif_spike_activity": lif_spike_activity,
            "adc_macro_utilization": adc_macro_utilization,
            "adc_group_request_activity": bounded(
                adc_pool.get("adc_macro_utilization", 0.0)
            ),
            "modulator_activity": modulator_activity,
            "membrane_update_activity_proxy": membrane_update_activity,
            "noc_activity_proxy": noc_activity,
        },
    }


def estimate_area(
    counts: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def add(
        component: str,
        count: int,
        area_per_unit_um2: Optional[float],
        note: str,
    ) -> None:
        if area_per_unit_um2 is None:
            total = None
            available = False
        else:
            total = count * area_per_unit_um2
            available = True

        rows.append(
            {
                "component": component,
                "count": count,
                "area_per_unit_um2": area_per_unit_um2 if area_per_unit_um2 is not None else "",
                "area_total_um2": total if total is not None else "",
                "area_total_mm2": (total / 1.0e6) if total is not None else "",
                "available": available,
                "note": note,
            }
        )

    pd_area = optional_area_param(
        device_cfg,
        ("photodetection_frontend", "photodiode", "area_per_output_um2"),
    )
    add(
        "photodiodes",
        as_int(counts.get("photodetector_outputs_total"), 256),
        pd_area,
        "PD area is reported only if provided in device_params.yaml.",
    )

    tia_area = optional_area_param(
        device_cfg,
        ("photodetection_frontend", "tia", "area_per_lane_um2"),
    )
    add(
        "tia",
        as_int(counts.get("hapr_output_lanes_total"), 32),
        tia_area,
        "TIA lanes after HAPR.",
    )

    comp_area = optional_area_param(
        device_cfg,
        ("photodetection_frontend", "comparator", "area_per_hapr_lane_um2"),
    )
    add(
        "comparators",
        as_int(counts.get("hapr_output_lanes_total"), 32),
        comp_area,
        "Comparator area is reported only if provided in device_params.yaml.",
    )

    adc_area = optional_area_param(
        device_cfg,
        ("adc", "conventional_adc_macro", "area_per_macro_um2"),
    )
    add(
        "adc_pool",
        as_int(counts.get("adc_macros"), 16),
        adc_area,
        "Conventional ADC macro pool.",
    )

    available_total_um2 = 0.0
    for row in rows:
        if row["available"]:
            available_total_um2 += as_float(row["area_total_um2"], 0.0)

    return {
        "area_rows": rows,
        "available_area_total_um2": available_total_um2,
        "available_area_total_mm2": available_total_um2 / 1.0e6,
        "note": "Area total includes only components with area parameters available in device_params.yaml.",
    }


# ---------------------------------------------------------------------------
# Per-dataset run
# ---------------------------------------------------------------------------


def run_one_dataset(
    dataset: str,
    args: argparse.Namespace,
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    eval01_dir = dataset_eval_dir(dataset, "eval_01", root=args.input_root)
    eval02_dir = dataset_eval_dir(dataset, "eval_02", root=args.output_root)
    eval02_dir.mkdir(parents=True, exist_ok=True)

    activity_path = eval01_dir / "summary.json"
    if not activity_path.exists():
        raise FileNotFoundError(
            f"eval01 summary not found: {activity_path}. "
            "Run eval_01 first."
        )

    activity = load_json(activity_path)

    counts = derive_architecture_counts(
        hardware_cfg,
        hapr_group_size_override=args.hapr_group_size,
        adc_macros_override=args.adc_macros,
    )

    adc_requests = estimate_adc_requests(
        activity,
        hapr_group_size=as_int(counts["hapr_group_size"], 8),
    )

    timing_base = estimate_timing(
        activity_summary=activity,
        hardware_cfg=hardware_cfg,
        adc_pool=None,
    )

    adc_pool = model_adc_pool(
        adc_requests=adc_requests,
        timing_base=timing_base,
        counts=counts,
    )

    timing = estimate_timing(
        activity_summary=activity,
        hardware_cfg=hardware_cfg,
        adc_pool=adc_pool,
    )

    power = estimate_power(
        activity_summary=activity,
        timing=timing,
        adc_pool=adc_pool,
        counts=counts,
        device_cfg=device_cfg,
        modulator_activity_source=args.modulator_activity_source,
        adc_power_mode=args.adc_power_mode,
        mrr_stabilization_mw=args.mrr_stabilization_mw,
    )

    area = estimate_area(counts, device_cfg)

    latency_energy = {
        "dataset": dataset,
        "num_samples": activity.get("num_samples"),
        "accuracy_percent": activity.get("accuracy_percent"),
        "loss": activity.get("loss"),

        "dense_sop_per_image": activity.get("dense_sop_per_image"),
        "active_sop_per_image": activity.get("active_sop_per_image"),
        "active_sop_ratio": activity.get("active_sop_ratio"),

        "model_input_activity": activity.get("model_input_activity"),
        "mvm_input_activity": activity.get("mvm_input_activity"),
        "lif_spike_activity": activity.get("lif_spike_activity"),
        "adc_element_request_activity": activity.get("adc_request_activity"),
        "adc_group_request_activity": adc_requests.get("adc_group_request_activity"),
        "adc_macro_utilization": adc_pool.get("adc_macro_utilization"),

        "base_latency_us_per_image": timing.get("base_latency_us_per_image"),
        "latency_us_per_image": timing.get("latency_us_per_image"),
        "throughput_images_per_s": timing.get("throughput_images_per_s"),

        "total_power_w": power.get("total_power_w"),
        "total_power_mw": power.get("total_power_mw"),
        "energy_uJ_per_image": power.get("energy_uJ_per_image"),
        "active_GOPS_per_W": power.get("active_GOPS_per_W"),
        "dense_equivalent_GOPS_per_W": power.get("dense_equivalent_GOPS_per_W"),
    }

    throughput_row = {
        "dataset": dataset,
        "throughput_images_per_s": timing.get("throughput_images_per_s"),
        "latency_us_per_image": timing.get("latency_us_per_image"),
        "active_sop_per_image": activity.get("active_sop_per_image"),
        "dense_sop_per_image": activity.get("dense_sop_per_image"),
        "realized_active_rate_sop_per_s": timing.get("realized_active_rate_sop_per_s"),
        "realized_active_rate_tsop_per_s": timing.get("realized_active_rate_tsop_per_s"),
        "effective_clock_hz": timing.get("effective_clock_hz"),
        "effective_utilization": timing.get("effective_utilization"),
        "base_cycles_per_image": timing.get("base_cycles_per_image"),
        "adc_stall_cycles_proxy": timing.get("adc_stall_cycles_proxy"),
        "cycles_per_image": timing.get("cycles_per_image"),
    }

    summary = {
        "eval_name": "eval_02",
        "purpose": "device_calibrated_latency_power_energy_model",
        "created_utc": now_utc(),
        "command": " ".join(sys.argv),

        "dataset": dataset,
        "input_eval01_summary": str(activity_path),
        "hardware_config": str(args.hardware),
        "device_params": str(args.device_params),

        "counts": counts,
        "timing": timing,
        "adc_requests": adc_requests,
        "adc_pool": adc_pool,
        "power": {
            k: v for k, v in power.items()
            if k not in {"power_rows"}
        },
        "area": {
            k: v for k, v in area.items()
            if k not in {"area_rows"}
        },
        "latency_energy_summary": latency_energy,

        "notes": {
            "latency_formula": "latency = (active_sop_per_image / realized_active_rate) plus ADC stall proxy.",
            "energy_formula": "energy = total_component_power * modeled_latency.",
            "pd_count_policy": "Photodiodes are counted before HAPR reduction; TIA/comparator lanes are counted after HAPR.",
            "adc_policy": "ADC pool uses conventional ADC macros; HIPSA contributes request-driven pooling, not a new ADC circuit.",
            "mrr_policy": "Main case does not include continuous per-ring thermal locking unless --mrr-stabilization-mw is passed.",
            "cifar10dvs_note": "If CIFAR10-DVS uses clipped_count max=3, report it as count-coded high-activity stress workload, not strict binary spike input.",
        },
    }

    save_json(summary, eval02_dir / "summary.json")
    save_json(latency_energy, eval02_dir / "latency_energy_summary.json")
    save_json(counts, eval02_dir / "derived_counts.json")
    save_json(adc_pool, eval02_dir / "adc_pool_summary.json")

    save_csv_rows(power["power_rows"], eval02_dir / "power_breakdown.csv")
    save_csv_rows(area["area_rows"], eval02_dir / "area_breakdown.csv")
    save_csv_rows([throughput_row], eval02_dir / "throughput_summary.csv")
    save_csv_rows(adc_requests["layers"], eval02_dir / "layer_adc_requests.csv")

    save_yaml(
        {
            "hardware": dict(hardware_cfg),
            "device_params": dict(device_cfg),
            "eval_02_args": vars(args),
        },
        eval02_dir / "config_snapshot.yaml",
    )

    save_run_manifest(
        eval02_dir,
        eval_name="eval_02",
        command=" ".join(sys.argv),
        inputs={
            "eval01_summary": str(activity_path),
            "hardware": str(args.hardware),
            "device_params": str(args.device_params),
        },
        outputs={
            "summary": "summary.json",
            "latency_energy_summary": "latency_energy_summary.json",
            "power_breakdown": "power_breakdown.csv",
            "area_breakdown": "area_breakdown.csv",
            "throughput_summary": "throughput_summary.csv",
            "adc_pool_summary": "adc_pool_summary.json",
            "layer_adc_requests": "layer_adc_requests.csv",
            "derived_counts": "derived_counts.json",
        },
        extra={
            "dataset": dataset,
            "adc_power_mode": args.adc_power_mode,
            "modulator_activity_source": args.modulator_activity_source,
        },
    )

    print("=" * 80)
    print("[eval_02] device-calibrated power/performance complete")
    print(f"dataset                  : {dataset}")
    print(f"accuracy                 : {as_float(activity.get('accuracy_percent')):.2f}%")
    print(f"active SOP / image       : {as_float(activity.get('active_sop_per_image')):.6e}")
    print(f"active SOP ratio         : {as_float(activity.get('active_sop_ratio')):.4%}")
    print(f"ADC group request act.   : {as_float(adc_requests.get('adc_group_request_activity')):.4%}")
    print(f"ADC macro utilization    : {as_float(adc_pool.get('adc_macro_utilization')):.4%}")
    print(f"latency                  : {as_float(timing.get('latency_us_per_image')):.3f} us/image")
    print(f"throughput               : {as_float(timing.get('throughput_images_per_s')):.2f} img/s")
    print(f"total power              : {as_float(power.get('total_power_w')):.4f} W")
    print(f"energy                   : {as_float(power.get('energy_uJ_per_image')):.3f} uJ/image")
    print(f"active efficiency        : {as_float(power.get('active_GOPS_per_W')):.2f} GOPS/W")
    print(f"output_dir               : {eval02_dir}")
    print("=" * 80)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HIPSA eval_02: device-calibrated latency/power/energy model"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10dvs", "dvsgesture"],
        help="Datasets to process.",
    )
    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="results/eval_v2", type=str)

    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml", type=str)
    parser.add_argument("--device-params", default="configs/device_params.yaml", type=str)

    parser.add_argument("--hapr-group-size", default=None, type=int)
    parser.add_argument("--adc-macros", default=None, type=int)

    parser.add_argument(
        "--modulator-activity-source",
        default="mvm_input_activity",
        choices=[
            "model_input_activity",
            "mvm_input_activity",
            "active_sop_ratio",
            "always_on",
        ],
        help="Activity factor used for binary modulator driver power.",
    )
    parser.add_argument(
        "--adc-power-mode",
        default="activity_scaled",
        choices=["activity_scaled", "all_biased"],
        help="Main ADC macro power accounting mode.",
    )
    parser.add_argument(
        "--mrr-stabilization-mw",
        default=None,
        type=float,
        help="Optional MRR stabilization power stress case. "
             "Default uses main-case value from device_params.yaml.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hardware_cfg = load_yaml(args.hardware)
    device_cfg = load_yaml(args.device_params)

    all_summaries = []
    for dataset in args.datasets:
        summary = run_one_dataset(
            dataset=dataset,
            args=args,
            hardware_cfg=hardware_cfg,
            device_cfg=device_cfg,
        )
        all_summaries.append(summary)

    combined_rows = [
        {
            "dataset": s["dataset"],
            **s["latency_energy_summary"],
        }
        for s in all_summaries
    ]

    combined_dir = Path(args.output_root) / "combined" / "eval_02"
    combined_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "eval_name": "eval_02",
            "created_utc": now_utc(),
            "datasets": args.datasets,
            "summaries": [
                {
                    "dataset": s["dataset"],
                    "latency_energy_summary": s["latency_energy_summary"],
                }
                for s in all_summaries
            ],
        },
        combined_dir / "summary.json",
    )
    save_csv_rows(combined_rows, combined_dir / "latency_energy_summary.csv")

    print(f"[eval_02] combined summary saved to {combined_dir}")


if __name__ == "__main__":
    main()