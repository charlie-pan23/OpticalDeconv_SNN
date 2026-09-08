"""Shared publication accounting for official HIPSA eval-v10 stages."""
from __future__ import annotations
from typing import Any, Dict, List, Mapping
from hardware.analog_validation import derive_mrr_stabilization_budget

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

def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _power_param(device_cfg: Mapping[str, Any], path: tuple[str, ...], default: float) -> float:
    return _float(_get(device_cfg, *path, default=default), default)

    return _float(_get(device_cfg, *path, default=default), default)


def build_mrr_stabilization_scenarios(
    *,
    selected_power_w: float,
    latency_us_per_image: float,
    config_energy_uJ_per_image: float,
    non_mvm_energy_uJ_per_image: float,
    device_cfg: Mapping[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build lock-aware cases without conflating them with MRR programming.

    ``selected_power_w`` is the nominal steady-state design point.  The
    stabilization term is intentionally reported as a separate scenario; the
    one-shot configuration and non-MVM terms remain unchanged across cases.
    The representative value is an explicit system thermal-domain scenario.
    """

    audit = derive_mrr_stabilization_budget(device_cfg)
    cfg = _get(device_cfg, "mrr_stabilization", default={})
    stress = [row["lock_power_mw"] for row in audit.get("locked_fraction_sensitivity", [])]
    values: List[float] = []
    for raw in [0.0, *stress, audit.get("representative_nonzero_lock_budget_mw", 0.0)]:
        try:
            value = float(raw)
        except Exception:
            continue
        if value >= 0.0 and value not in values:
            values.append(value)

    # W * us is numerically equal to uJ (the 1e-6 s/us and 1e6 uJ/J
    # conversion factors cancel).  Keep this explicit to avoid the common
    # extra-1e6 error when adding a continuous-power scenario to an energy
    # breakdown that is already reported in microjoules.
    nominal_energy = (
        float(selected_power_w) * float(latency_us_per_image)
        + float(config_energy_uJ_per_image)
        + float(non_mvm_energy_uJ_per_image)
    )
    rows: List[Dict[str, Any]] = []
    for stabilization_mw in values:
        total_power_w = float(selected_power_w) + stabilization_mw * 1.0e-3
        total_energy = (
            total_power_w * float(latency_us_per_image)
            + float(config_energy_uJ_per_image)
            + float(non_mvm_energy_uJ_per_image)
        )
        rows.append(
            {
                "mrr_stabilization_mw": stabilization_mw,
                "total_power_w": total_power_w,
                "latency_us_per_image": float(latency_us_per_image),
                "energy_uJ_per_image": total_energy,
                "energy_overhead_percent": 100.0 * (total_energy / max(nominal_energy, 1.0e-30) - 1.0),
                "scenario_label": "nominal_no_continuous_lock" if stabilization_mw == 0.0 else "explicit_lock_scenario",
                "status": audit.get("status", "scenario_budget_with_explicit_reserve_not_measured"),
            }
        )
    return audit, rows


def build_area_breakdown(
    counts: Mapping[str, Any],
    weight_map: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def add(component: str, count: float, area_um2: float, status: str, note: str) -> None:
        total = float(count) * float(area_um2)
        rows.append({
            "component": component,
            "count": count,
            "area_per_unit_um2": area_um2,
            "area_total_um2": total,
            "area_total_mm2": total / 1e6,
            "evidence_status": status,
            "note": note,
        })

    add("cw_laser", 1, _power_param(device_cfg, ("optical_source", "cw_laser", "area_um2"), 120000.0), "adopted_parameter", "Laser area anchor; package footprint excluded.")
    add("binary_modulator_driver", _int(counts.get("modulator_lanes_total"), 256), _power_param(device_cfg, ("modulation", "binary_modulator_driver", "area_per_lane_um2"), 5200.0), "adopted_parameter", "One binary driver per physical input lane.")
    add("photodiodes", _int(counts.get("photodetector_outputs_total"), 256), _power_param(device_cfg, ("photodetection_frontend", "photodiode", "area_per_output_um2"), 40.0), "adopted_parameter", "PDs counted before HAPR.")
    physical_hapr_lanes = _int(counts.get("physical_hapr_lanes_total"), 256)
    add("tia", physical_hapr_lanes, _power_param(device_cfg, ("photodetection_frontend", "tia", "area_per_lane_um2"), 50.0), "adopted_parameter", "Same-neuron HAPR lane array.")
    add("comparators", physical_hapr_lanes, _power_param(device_cfg, ("photodetection_frontend", "comparator", "area_per_hapr_lane_um2"), 78.0), "adopted_parameter", "Request comparator per HAPR lane.")
    add("adc_pool", _int(counts.get("adc_macros"), 1), _power_param(device_cfg, ("adc", "conventional_adc_macro", "area_per_macro_um2"), 2850.0), "adopted_macro_parameter", "Conventional ADC macro; no new ADC circuit claim.")
    # Weight tiles are time-multiplexed over the instantiated physical arrays.
    # Keep total model weights in the configuration/SRAM accounting, but count
    # only physical rings that can exist concurrently in the area estimate.
    num_tiles = _int(_get(hardware_cfg, "photonic_tiles", "num_tiles", default=1), 1)
    array_rows = _int(_get(hardware_cfg, "mapping", "array_rows", default=64), 64)
    array_cols = _int(_get(hardware_cfg, "mapping", "array_cols", default=64), 64)
    signed_branches = max(_int(weight_map.get("signed_weight_multiplier"), 2), 1)
    instantiated_rings = num_tiles * array_rows * array_cols * signed_branches
    add(
        "signed_mrr_weight_bank",
        instantiated_rings,
        _power_param(device_cfg, ("mrr_configuration", "area_per_physical_ring_um2"), 0.80),
        "instantiated_tile_lower_bound",
        "Area counts concurrently instantiated rings; total model weights remain in weight-mapping/configuration outputs.",
    )

    sram_area_mm2 = _float(_get(device_cfg, "memory_digital", "sram_area_budget_mm2", default=0.0573), 0.0573)
    add("sram_register_file_estimate", 1, sram_area_mm2 * 1e6, "architecture_estimate", "SRAM/register-file estimate; no compiled memory macro.")
    control_area_mm2 = _float(_get(device_cfg, "memory_digital", "synthesized_control", "area_mm2", default=0.0), 0.0)
    add("synthesized_digital_control", 1, control_area_mm2 * 1e6, "open_45nm_synthesis_anchor", "Yosys/Nangate45 mapped cell area; excludes P&R, clock tree, pads, and SRAM macros.")
    total = sum(_float(r["area_total_mm2"]) for r in rows)
    return {
        "rows": rows,
        "area_mm2": total,
        "physical_rings_instantiated": instantiated_rings,
        "area_status": "architecture_native_component_footprint_plus_open45nm_control_anchor_not_postlayout",
    }


def build_power(
    activity: Mapping[str, Any],
    queue: Mapping[str, Any],
    state: Mapping[str, Any],
    counts: Mapping[str, Any],
    timing: Mapping[str, Any],
    link: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    *,
    mrr_stabilization_mw: float = 0.0,
) -> Dict[str, Any]:
    """Use independent request, ADC-utilization, and state-update activities."""

    mvm_activity = min(max(_float(activity.get("mvm_input_activity"), 0.0), 0.0), 1.0)
    input_activity = min(max(_float(activity.get("model_input_activity"), 0.0), 0.0), 1.0)
    lif_activity = min(max(_float(activity.get("lif_spike_activity"), 0.0), 0.0), 1.0)
    request_activity = min(max(_float(activity.get("adc_request_activity"), 0.0), 0.0), 1.0)
    adc_util = min(max(_float(queue.get("adc_utilization"), 0.0), 0.0), 1.0)
    state_activity = min(max(_float(state.get("state_update_activity"), 0.0), 0.0), 1.0)
    noc_activity = max(input_activity, lif_activity)
    lanes = _int(counts.get("physical_hapr_lanes_total"), 256)
    mod_lanes = _int(counts.get("modulator_lanes_total"), 256)
    macros = _int(counts.get("adc_macros"), 1)
    rows: List[Dict[str, Any]] = []

    def add(name: str, power_mw: float, activity_factor: float, note: str) -> None:
        rows.append({"component": name, "power_mw": power_mw, "activity_factor": activity_factor, "note": note})

    add("cw_laser", _float(link.get("laser_electrical_power_mw"), 0.0), 1.0, "Derived from the explicit optical link budget; CW during inference.")
    add("mrr_stabilization", mrr_stabilization_mw, 1.0, "Nominal case is zero; nonzero cases are reported separately.")
    add("leakage_misc_io", _power_param(device_cfg, ("memory_digital", "leakage_misc_io", "reference_power_mw"), 90.0), 1.0, "Static leakage and I/O reference.")
    add("binary_modulator_driver", _power_param(device_cfg, ("modulation", "binary_modulator_driver", "power_per_active_lane_mw"), 2.25) * mod_lanes * mvm_activity, mvm_activity, "Independent MVM input activity.")
    add("photodiodes", _power_param(device_cfg, ("photodetection_frontend", "photodiode", "power_per_output_mw"), 1.1) * _int(counts.get("photodetector_outputs_total"), 256), 1.0, "Provisioned PD array.")
    add("tia", _power_param(device_cfg, ("photodetection_frontend", "tia", "power_per_hapr_lane_mw"), 3.0) * lanes, 1.0, "Provisioned same-neuron HAPR lanes.")
    add("comparators", _power_param(device_cfg, ("photodetection_frontend", "comparator", "power_per_hapr_lane_mw"), 2.2) * lanes, 1.0, "Comparator front end.")
    add("hapr_selection", _power_param(device_cfg, ("photodetection_frontend", "hapr_selection_proxy", "power_per_hapr_lane_mw"), 0.1) * lanes, 1.0, "Selection/request control.")
    adc_rated_mw = _power_param(device_cfg, ("adc", "conventional_adc_macro", "power_per_macro_mw"), 14.8)
    adc_idle_fraction = min(max(_power_param(device_cfg, ("adc", "conventional_adc_macro", "idle_bias_fraction"), 0.20), 0.0), 1.0)
    adc_effective_activity = adc_idle_fraction + (1.0 - adc_idle_fraction) * adc_util
    add("adc_pool", adc_rated_mw * macros * adc_effective_activity, adc_effective_activity, "Per-macro bias/clock fraction plus utilization-scaled conversion power; split is explicit in device_params.yaml.")
    add("sram_register_files", _power_param(device_cfg, ("memory_digital", "sram_register_files", "reference_power_mw"), 243.25) * state_activity, state_activity, "State-update activity from actual requested state touches.")
    add("noc_bus_controller", _power_param(device_cfg, ("memory_digital", "noc_bus_controller_clock", "reference_power_mw"), 77.84) * noc_activity, noc_activity, "Independent spike/control traffic activity.")
    add("digital_lif_update", _power_param(device_cfg, ("memory_digital", "digital_lif_update", "reference_power_mw"), 2.43) * state_activity, state_activity, "LIF activity independent of ADC macro count.")
    total_mw = sum(_float(r["power_mw"]) for r in rows)
    latency_s = _float(timing.get("latency_s_per_image"), 0.0)
    return {
        "rows": rows,
        "component_power_mw": {str(r["component"]): _float(r["power_mw"]) for r in rows},
        "total_power_mw": total_mw,
        "total_power_w": total_mw / 1000.0,
        "power_floor_mw": _float(link.get("laser_electrical_power_mw"), 0.0) + _power_param(device_cfg, ("memory_digital", "leakage_misc_io", "reference_power_mw"), 90.0),
        "variable_power_mw": max(total_mw - (_float(link.get("laser_electrical_power_mw"), 0.0) + _power_param(device_cfg, ("memory_digital", "leakage_misc_io", "reference_power_mw"), 90.0)), 0.0),
        "latency_us_per_image": latency_s * 1e6,
        "energy_uJ_per_image_before_one_shot_overheads": total_mw / 1000.0 * latency_s * 1e6,
        "activity_factors": {
            "conversion_request_activity": request_activity,
            "adc_macro_utilization": adc_util,
            "adc_idle_bias_fraction": adc_idle_fraction,
            "adc_effective_power_activity": adc_effective_activity,
            "state_update_activity": state_activity,
            "mvm_input_activity": mvm_activity,
            "lif_spike_activity": lif_activity,
        },
    }


def lock_power_mw(hardware_cfg: Mapping[str, Any], fraction: float) -> float:
    """Return parameterized MRR lock power for a declared locked-ring fraction."""

    cfg = _get(hardware_cfg, "mrr_lock_sensitivity", default={}) or {}
    physical_ring_count = _int(cfg.get("physical_ring_count"), 32768)
    power_per_ring_mw = _float(cfg.get("locking_power_mw_per_ring"), 1.2)
    return physical_ring_count * float(fraction) * power_per_ring_mw


def build_lock_fraction_sensitivity(
    *,
    selected_power_w: float,
    latency_us_per_image: float,
    config_energy_uJ_per_image: float,
    non_mvm_energy_uJ_per_image: float,
    hardware_cfg: Mapping[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the official 0/1/5/10% MRR lock-power sensitivity table.

    Fractions and per-ring power are declared parameters, not HIPSA silicon
    measurements.  One-shot programming and non-MVM energy are held constant.
    """

    cfg = _get(hardware_cfg, "mrr_lock_sensitivity", default={}) or {}
    physical_ring_count = _int(cfg.get("physical_ring_count"), 32768)
    power_per_ring_mw = _float(cfg.get("locking_power_mw_per_ring"), 1.2)
    raw_fractions = cfg.get("locked_fractions", [0.0, 0.01, 0.05, 0.10])
    fractions = sorted({float(value) for value in raw_fractions})
    recommended_fraction = _float(cfg.get("recommended_reporting_fraction"), 0.01)
    if recommended_fraction not in fractions:
        fractions.append(recommended_fraction)
        fractions.sort()

    nominal_energy_uJ = (
        float(selected_power_w) * float(latency_us_per_image)
        + float(config_energy_uJ_per_image)
        + float(non_mvm_energy_uJ_per_image)
    )
    rows: List[Dict[str, Any]] = []
    for fraction in fractions:
        added_lock_power_mw = physical_ring_count * fraction * power_per_ring_mw
        total_power_w = float(selected_power_w) + added_lock_power_mw * 1.0e-3
        total_energy_uJ = (
            total_power_w * float(latency_us_per_image)
            + float(config_energy_uJ_per_image)
            + float(non_mvm_energy_uJ_per_image)
        )
        rows.append(
            {
                "locked_fraction": fraction,
                "locked_fraction_percent": 100.0 * fraction,
                "physical_ring_count": physical_ring_count,
                "effective_locked_ring_count": physical_ring_count * fraction,
                "locking_power_mw_per_ring": power_per_ring_mw,
                "added_lock_power_mw": added_lock_power_mw,
                "nominal_power_w_without_lock": float(selected_power_w),
                "total_power_w": total_power_w,
                "latency_us_per_image": float(latency_us_per_image),
                "energy_uJ_per_image": total_energy_uJ,
                "energy_overhead_percent": 100.0
                * (total_energy_uJ / max(nominal_energy_uJ, 1.0e-30) - 1.0),
                "recommended_reporting_case": int(abs(fraction - recommended_fraction) < 1.0e-12),
                "status": cfg.get(
                    "status",
                    "parameterized_literature_anchor_not_hipsa_measurement",
                ),
            }
        )

    summary = {
        "physical_ring_count": physical_ring_count,
        "locking_power_mw_per_ring": power_per_ring_mw,
        "locked_fractions": fractions,
        "recommended_reporting_fraction": recommended_fraction,
        "status": cfg.get(
            "status", "parameterized_literature_anchor_not_hipsa_measurement"
        ),
        "claim_boundary": "parameter_sensitivity_not_measured_lock_power",
    }
    return summary, rows

