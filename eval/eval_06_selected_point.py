"""Publication-candidate HIPSA hybrid evaluation.

This stage rebuilds the paper-facing numbers from:

checkpoint/activity summary -> layer/weight mapping -> transactions -> ADC FIFO
-> state traffic/non-MVM work -> configuration overhead -> PPA/energy.

The architecture metrics are rebuilt from frozen activity summaries. When a
validated eval_105 checkpoint replay is present, its clean and combined
accuracy are joined by provenance; otherwise accuracy remains explicitly
unverified rather than inferred from aggregate activity.

Device values come from the A revision's source-backed configuration; mapping,
cycle scheduling, ADC FIFO, fixed-point state and non-MVM accounting follow the
v3 system model.  Continuous MRR stabilization is emitted as an explicit
scenario alongside (not inside) one-shot configuration energy.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.analog_validation import (
    derive_mrr_stabilization_budget,
    validate_hapr_corner_sweep,
    validate_hapr_fanin,
)
from hardware.adc_pool_model import simulate_adc_queue
from hardware.layer_mapper import build_layer_mapping, summarize_mapping
from hardware.link_budget import derive_link_budget
from hardware.mrr_config_model import model_mrr_configuration
from hardware.non_mvm_model import build_non_mvm_summary
from hardware.pareto import pareto_front, select_balanced
from hardware.partial_sum_mapper import map_partial_sums, validate_no_cross_neuron_sum
from hardware.publication_accounting import build_lock_fraction_sensitivity
from hardware.state_update_model import build_state_update_trace
from hardware.transaction_scheduler import build_transaction_trace
from hardware.weight_mapper import build_weight_map
from utils.result_io import dataset_eval_dir, load_csv_rows, load_json, load_yaml, save_csv_rows, save_json, save_run_manifest
from utils.eval_v10 import file_sha256, validate_eval105_publication_summary, validate_g4_a8_contract


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def _load_eval105_accuracy(
    output_root: str | Path, dataset: str,
) -> tuple[Dict[str, Any] | None, str, Path, Dict[str, Any]]:
    path = Path(output_root) / dataset / "eval_105" / "summary.json"
    if not path.exists():
        validation = {
            "valid": False,
            "status": "not_verified_missing_eval_105",
            "reasons": ["missing_eval_105_summary"],
            "photonic_mvm_layer_names": [],
        }
        return None, validation["status"], path, validation
    summary = load_json(path)
    validation = validate_eval105_publication_summary(summary)
    return summary, validation["status"], path, validation
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



def build_tile_load_sensitivity(
    *,
    dataset: str,
    cycles_per_tile_load_values: List[int],
    base_cycles_without_configuration: float,
    clock_hz: float,
    weight_map: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    activity: Mapping[str, Any],
    queue: Mapping[str, Any],
    state: Mapping[str, Any],
    counts: Mapping[str, Any],
    link: Mapping[str, Any],
    non_mvm_energy_uJ_per_image: float,
    lock_fraction: float,
    lock_power_mw_value: float,
) -> List[Dict[str, Any]]:
    """Recompute latency/energy for each tile-programming latency assumption."""

    rows: List[Dict[str, Any]] = []
    for cycles_per_tile_load in cycles_per_tile_load_values:
        scenario_hardware = copy.deepcopy(hardware_cfg)
        scenario_hardware.setdefault("mrr_configuration", {})[
            "cycles_per_tile_load"
        ] = int(cycles_per_tile_load)
        scenario_mrr = model_mrr_configuration(weight_map, scenario_hardware, device_cfg)
        total_cycles = base_cycles_without_configuration + _float(
            scenario_mrr.get("mrr_reconfiguration_cycles_per_image")
        )
        timing = {
            "cycles_per_image": total_cycles,
            "latency_s_per_image": total_cycles / max(clock_hz, 1.0),
            "latency_us_per_image": total_cycles / max(clock_hz, 1.0) * 1.0e6,
        }
        scenario_power = build_power(
            activity, queue, state, counts, timing, link, device_cfg
        )
        config_energy_uJ = _float(
            scenario_mrr.get("configuration_energy_pj_per_image")
        ) / 1.0e6
        nominal_energy_uJ = (
            _float(
                scenario_power.get(
                    "energy_uJ_per_image_before_one_shot_overheads"
                )
            )
            + config_energy_uJ
            + float(non_mvm_energy_uJ_per_image)
        )
        latency_us = _float(timing.get("latency_us_per_image"))
        rows.append(
            {
                "dataset": dataset,
                "cycles_per_tile_load": int(cycles_per_tile_load),
                "tile_load_events_per_image": _int(
                    scenario_mrr.get("mrr_reconfiguration_events_per_image")
                ),
                "mrr_configuration_cycles_per_image": _float(
                    scenario_mrr.get("mrr_reconfiguration_cycles_per_image")
                ),
                "mrr_configuration_time_us_per_image": _float(
                    scenario_mrr.get("mrr_reconfiguration_time_us_per_image")
                ),
                "total_cycles_per_image": total_cycles,
                "latency_us_per_image": latency_us,
                "throughput_images_per_s": max(clock_hz, 1.0)
                / max(total_cycles, 1.0),
                "nominal_power_w": _float(scenario_power.get("total_power_w")),
                "nominal_energy_uJ_per_image": nominal_energy_uJ,
                "lock_fraction": float(lock_fraction),
                "added_lock_power_mw": float(lock_power_mw_value),
                "lock_aware_power_w": _float(scenario_power.get("total_power_w"))
                + float(lock_power_mw_value) * 1.0e-3,
                "lock_aware_energy_uJ_per_image": nominal_energy_uJ
                + float(lock_power_mw_value) * 1.0e-3 * latency_us,
                "configuration_energy_uJ_per_image": config_energy_uJ,
                "non_mvm_energy_uJ_per_image": float(
                    non_mvm_energy_uJ_per_image
                ),
                "status": "architecture_parameter_sensitivity_not_measured_programming_time",
            }
        )
    return rows

def find_partial_adc_only_accuracy(dataset: str, input_root: Path) -> float | None:
    path = input_root / dataset / "eval_05" / "robustness_summary.csv"
    if not path.exists():
        return None
    for row in load_csv_rows(path, parse_numbers=True):
        if str(row.get("perturbation_type")) == "adc_bits" and _float(row.get("level"), -1) == 6.0 and _int(row.get("seed"), -1) == 0:
            return _float(row.get("accuracy_percent"), 0.0)
    return None


def run_dataset(dataset: str, args: argparse.Namespace, hardware_cfg: Mapping[str, Any], device_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    input_activity_path = Path(args.activity_summary) if getattr(args, "activity_summary", None) else Path(args.input_root) / dataset / "eval_01" / "summary.json"
    if not input_activity_path.exists():
        raise FileNotFoundError(f"Missing required eval_01 artifact: {input_activity_path}")
    activity = load_json(input_activity_path)
    contract = validate_g4_a8_contract(hardware_cfg)
    out_dir = Path(args.output_root) / dataset / "eval_101"
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = _get(hardware_cfg, "photonic_tiles", default={})
    mapping_cfg = _get(hardware_cfg, "mapping", default={})
    num_tiles = _int(_get(tiles, "num_tiles", default=4), 4)
    array_rows = _int(_get(mapping_cfg, "array_rows", default=64), 64)
    array_cols = _int(_get(mapping_cfg, "array_cols", default=64), 64)
    clock_hz = _float(_get(hardware_cfg, "transaction_scheduler", "system_clock_hz", default=_get(tiles, "effective_clock_hz", default=1e9)), 1e9)
    lif_pipeline_flush_cycles = _float(_get(hardware_cfg, "transaction_scheduler", "lif_pipeline_flush_cycles", default=0.0), 0.0)
    adc_queue_pipeline_fill_cycles = _float(_get(hardware_cfg, "transaction_scheduler", "adc_queue_pipeline_fill_cycles", default=0.0), 0.0)
    mapping_rows = build_layer_mapping(activity, array_rows=array_rows, array_cols=array_cols, num_tiles=num_tiles)
    mapping_summary = summarize_mapping(mapping_rows)
    weight_map = build_weight_map(mapping_summary, num_tiles=num_tiles, signed_weight_multiplier=2, weight_bits=_int(_get(hardware_cfg, "precision", "weight_bits", default=6), 6))
    link = derive_link_budget(device_cfg)
    adc_bits = _int(_get(hardware_cfg, "precision", "adc_bits", default=6), 6)
    analog_rows = validate_hapr_fanin(device_cfg, [1, 2, 4, 8, 16, 32], adc_bits=adc_bits)
    analog_corner_rows = validate_hapr_corner_sweep(
        device_cfg,
        [1, 2, 4, 8, 16, 32],
        adc_bits=adc_bits,
    )
    analog_by_g = {_int(row.get("hapr_group_size")): row for row in analog_rows}
    design_rows: List[Dict[str, Any]] = []
    full_details: Dict[str, Any] = {}
    for group_size in args.hapr_group_sizes:
        for adc_macros in args.adc_macro_sizes:
            transaction = build_transaction_trace(activity, mapping_rows, hapr_group_size=group_size, num_tiles=num_tiles, array_rows=array_rows, array_cols=array_cols, event_gating=True)
            queue = simulate_adc_queue(
                transaction["trace"],
                system_clock_hz=clock_hz,
                sample_rate_gsps=_float(_get(device_cfg, "adc", "conventional_adc_macro", "sample_rate_gsps", default=10.0), 10.0),
                adc_macros=adc_macros,
                fifo_depth=_int(_get(hardware_cfg, "hapr_adc_backend", "adc_fifo_depth", default=4096), 4096),
            )
            state = build_state_update_trace(activity, transaction["layer_summary"], lazy_lif=True)
            digital = build_non_mvm_summary(activity, workload_name=dataset, time_steps=_int(activity.get("time_steps"), 1), operator_cfg={**dict(_get(hardware_cfg, "digital_ops", default={}) or {}), "num_classes": 10 if dataset == "cifar10dvs" else 11})
            mrr = model_mrr_configuration(weight_map, hardware_cfg, device_cfg)
            mvm_cycles = _int(transaction.get("total_cycles_per_image"), 0)
            total_cycles = float(mvm_cycles) + _float(queue.get("stall_cycles"), 0.0) + _float(digital.get("total_cycles_per_image"), 0.0) + _float(mrr.get("mrr_reconfiguration_cycles_per_image"), 0.0) + lif_pipeline_flush_cycles + adc_queue_pipeline_fill_cycles
            timing = {"cycles_per_image": total_cycles, "latency_s_per_image": total_cycles / max(clock_hz, 1.0), "latency_us_per_image": total_cycles / max(clock_hz, 1.0) * 1e6}
            timing["throughput_images_per_s"] = 1.0 / max(_float(timing["latency_s_per_image"]), 1e-30)
            physical_hapr_groups = (num_tiles + max(group_size, 1) - 1) // max(group_size, 1)
            post_hapr_lanes = physical_hapr_groups * array_cols
            samples_per_macro_per_cycle = _float(
                _get(
                    hardware_cfg,
                    "hapr_adc_backend",
                    "adc_samples_per_macro_per_architecture_cycle",
                    default=_float(
                        _get(device_cfg, "adc", "conventional_adc_macro", "sample_rate_gsps", default=10.0),
                        10.0,
                    )
                    * 1.0e9
                    / max(clock_hz, 1.0),
                ),
                10.0,
            )
            nominal_slots = adc_macros * samples_per_macro_per_cycle
            counts = {
                "num_tiles": num_tiles,
                "array_rows": array_rows,
                "array_cols": array_cols,
                "tile_outputs": array_cols,
                "hapr_group_size": group_size,
                "adc_macros": adc_macros,
                "photodetector_outputs_total": num_tiles * array_cols,
                "modulator_lanes_total": num_tiles * array_cols,
                "physical_hapr_lanes_total": post_hapr_lanes,
            }
            power = build_power(activity, queue, state, counts, timing, link, device_cfg)
            area = build_area_breakdown(counts, weight_map, hardware_cfg, device_cfg)
            config_energy_uJ = _float(mrr.get("configuration_energy_pj_per_image"), 0.0) / 1e6
            digital_energy_uJ = _float(digital.get("total_energy_pj_per_image"), 0.0) / 1e6
            energy = _float(power.get("energy_uJ_per_image_before_one_shot_overheads"), 0.0) + config_energy_uJ + digital_energy_uJ
            analog = analog_by_g.get(group_size, {})
            row = {
                "dataset": dataset,
                "hapr_group_size": group_size,
                "adc_macros": adc_macros,
                "analog_feasible": _int(analog.get("feasible"), 0),
                "physical_hapr_feasible": int(group_size <= num_tiles and num_tiles % group_size == 0),
                "post_hapr_lanes": post_hapr_lanes,
                "nominal_adc_slots_per_cycle": nominal_slots,
                "same_window_admission_feasible": int(nominal_slots >= post_hapr_lanes),
                "transactions_issued_per_image": _float(transaction.get("transactions_issued_per_image")),
                "transaction_opportunities_per_image": _float(transaction.get("transaction_opportunities_per_image")),
                "derived_effective_utilization": _float(transaction.get("derived_effective_utilization")),
                "adc_requests_per_image": _float(queue.get("requests_per_image")),
                "adc_utilization": _float(queue.get("adc_utilization")),
                "p99_queue": _float(queue.get("queue_p99")),
                "queue_max": _float(queue.get("queue_max")),
                "adc_stall_cycles": _float(queue.get("stall_cycles")),
                "requests_dropped": _float(queue.get("requests_dropped")),
                "sram_reads_per_image": _float(state.get("sram_reads_per_image")),
                "sram_writes_per_image": _float(state.get("sram_writes_per_image")),
                "state_update_activity": _float(state.get("state_update_activity")),
                "digital_cycles_per_image": _float(digital.get("total_cycles_per_image")),
                "mrr_config_cycles_per_image": _float(mrr.get("mrr_reconfiguration_cycles_per_image")),
                "lif_pipeline_flush_cycles_per_image": lif_pipeline_flush_cycles,
                "adc_queue_pipeline_fill_cycles_per_image": adc_queue_pipeline_fill_cycles,
                "config_energy_uJ_per_image": config_energy_uJ,
                "non_mvm_energy_uJ_per_image": digital_energy_uJ,
                "latency_us_per_image": _float(timing.get("latency_us_per_image")),
                "throughput_images_per_s": _float(timing.get("throughput_images_per_s")),
                "total_power_w": _float(power.get("total_power_w")),
                "energy_uJ_per_image": energy,
                "edp_uJ_s_per_image": energy * _float(timing.get("latency_s_per_image")),
                "area_mm2": _float(area.get("area_mm2")),
                "power_floor_mw": _float(power.get("power_floor_mw")),
                "variable_power_mw": _float(power.get("variable_power_mw")),
                "weight_residency": "layer_weight_tile_major",
            }
            design_rows.append(row)
            full_details[(group_size, adc_macros)] = (transaction, queue, state, digital, mrr, counts, power, area)

    feasible = [row for row in design_rows if _int(row.get("analog_feasible"), 0) == 1 and _int(row.get("physical_hapr_feasible"), 0) == 1 and _float(row.get("requests_dropped"), 0.0) == 0.0 and _int(row.get("same_window_admission_feasible"), 0) == 1]
    if not feasible:
        raise RuntimeError(f"No feasible design point for {dataset}")
    pareto_rows = pareto_front(feasible, ["area_mm2", "energy_uJ_per_image", "latency_us_per_image", "p99_queue"])
    target_g = _int(contract["hapr_group_size"])
    target_a = _int(contract["adc_macros"])
    selected_match = [row for row in feasible if _int(row.get("hapr_group_size")) == target_g and _int(row.get("adc_macros")) == target_a]
    if not selected_match:
        raise RuntimeError(f"Configured publication point G{target_g}/A{target_a} is missing or infeasible for {dataset}.")
    selected = dict(selected_match[0])
    for row in design_rows:
        row["pareto_optimal"] = int(any(row is p or (row.get("hapr_group_size") == p.get("hapr_group_size") and row.get("adc_macros") == p.get("adc_macros")) for p in pareto_rows))
        row["selected"] = int(row.get("hapr_group_size") == selected.get("hapr_group_size") and row.get("adc_macros") == selected.get("adc_macros"))
    g = _int(selected["hapr_group_size"])
    a = _int(selected["adc_macros"])
    transaction, queue, state, digital, mrr, counts, power, area = full_details[(g, a)]
    partial_rows = map_partial_sums(mapping_rows, hapr_group_size=g, time_steps=_int(activity.get("time_steps"), 1), num_tiles=num_tiles)
    analog_cfg = _get(device_cfg, "analog_validation", default={})
    hapr_bound_summary = {
        "selected_group_size": g,
        "screened_group_sizes": [_int(row.get("hapr_group_size")) for row in analog_rows],
        "nominal_feasible_group_sizes": [
            _int(row.get("hapr_group_size"))
            for row in analog_rows
            if _int(row.get("feasible"), 0) == 1
        ],
        "first_rejected_group_size": next(
            (
                _int(row.get("hapr_group_size"))
                for row in analog_rows
                if _int(row.get("feasible"), 0) == 0
            ),
            None,
        ),
        "physical_fanin_limit": num_tiles,
        "physical_feasible_group_sizes": [x for x in [1, 2, 4] if x <= num_tiles and num_tiles % x == 0],
        "first_physically_rejected_group_size": next((x for x in [1, 2, 4, 8, 16, 32] if x > num_tiles or num_tiles % x != 0), None),
        "criteria": {
            "min_saturation_margin_db": _float(_get(analog_cfg, "min_saturation_margin_db", default=0.0)),
            "min_snr_margin_db": _float(_get(analog_cfg, "min_snr_margin_db", default=0.0)),
            "settling_time_limit_ns": _float(_get(analog_cfg, "settling_time_limit_ns", default=0.0)),
            "device_error_bounds_applied": bool(_get(analog_cfg, "apply_device_error_bounds", default=False)),
        },
        "equations": {
            "lane_current": "I_lane = R_PD * P_PD",
            "aggregate_current": "I_sum = G * I_lane",
            "tia_output": "V_TIA = Z_TIA * I_sum",
            "differential_shot_noise": "sigma_shot = sqrt(2*q*(2*I_sum + I_dark)*B_eff)",
            "tia_noise": "sigma_tia = i_n*sqrt(B_eff)",
            "snr": "SNR_dB = 20*log10(I_sum/sqrt(sigma_shot^2 + sigma_tia^2))",
            "effective_bandwidth": "B_eff = min(B_declared, max(1/(2*pi*Z_TIA*C_in), B_floor))",
        },
        "physical_rule": "spatial same-neuron partial sums must be simultaneously available; G divides and cannot exceed N_physical_tiles; no temporal analog storage",
        "status": "analytical_architecture_and_spatial_concurrency_screen_not_silicon_measurement",
    }
    save_csv_rows(mapping_rows, out_dir / "mapping_trace.csv")
    save_csv_rows(mapping_summary, out_dir / "mapping_summary.csv")
    save_csv_rows(partial_rows, out_dir / "partial_sum_mapping.csv")
    save_csv_rows(transaction["trace"], out_dir / "transaction_trace.csv")
    save_csv_rows(queue["segments"], out_dir / "adc_queue_trace.csv")
    save_csv_rows([{"dataset": dataset, **{k: v for k, v in queue.items() if k != "segments"}}], out_dir / "adc_queue_summary.csv")
    save_csv_rows(state["rows"], out_dir / "state_update_trace.csv")
    save_csv_rows(digital["rows"], out_dir / "non_mvm_operator_summary.csv")
    save_csv_rows([link], out_dir / "link_budget.csv")
    save_csv_rows(analog_rows, out_dir / "hapr_analog_validation.csv")
    save_csv_rows(analog_corner_rows, out_dir / "hapr_corner_validation.csv")
    save_csv_rows(design_rows, out_dir / "design_space.csv")
    save_csv_rows([r for r in design_rows if r.get("pareto_optimal")], out_dir / "pareto_front.csv")
    save_csv_rows(power["rows"], out_dir / "power_breakdown.csv")
    save_csv_rows(area["rows"], out_dir / "area_breakdown.csv")
    save_csv_rows(
        [
            {
                "layer": layer.get("layer"),
                "input_dim": layer.get("input_dim"),
                "output_dim": layer.get("output_dim"),
                "row_tiles": layer.get("row_tiles"),
                "col_tiles": layer.get("col_tiles"),
                "logical_weights": layer.get("logical_weight_count"),
                "physical_mrr_elements": layer.get("physical_mrr_elements"),
                "weight_bits": layer.get("weight_bits"),
                "residency": layer.get("residency_policy"),
            }
            for layer in weight_map.get("layers", [])
        ],
        out_dir / "weight_mapping.csv",
    )
    save_json(weight_map, out_dir / "weight_mapping.json")
    save_json(hapr_bound_summary, out_dir / "hapr_bound_derivation.json")
    save_json(mrr, out_dir / "mrr_configuration.json")
    save_json(state, out_dir / "state_update_summary.json")
    save_json(digital, out_dir / "non_mvm_operator_summary.json")
    save_json(queue, out_dir / "adc_queue_summary.json")
    save_json(transaction, out_dir / "transaction_summary.json")
    save_json({"selection_policy": "configured_G4_A8_not_posthoc_balanced_selection", "g4_a8_contract": contract, "selected": selected, "pareto_size": len(pareto_rows), "feasible_designs": len(feasible)}, out_dir / "selection.json")

    partial_adc_accuracy = find_partial_adc_only_accuracy(dataset, Path(args.input_root))
    selected_energy = _float(selected.get("energy_uJ_per_image"))
    selected_power_latency_energy = _float(power.get("energy_uJ_per_image_before_one_shot_overheads"))
    selected_config_energy = _float(mrr.get("configuration_energy_pj_per_image")) / 1e6
    selected_non_mvm_energy = _float(digital.get("total_energy_pj_per_image")) / 1e6
    mrr_budget_audit, mrr_device_budget_rows = build_mrr_stabilization_scenarios(
        selected_power_w=_float(selected.get("total_power_w")),
        latency_us_per_image=_float(selected.get("latency_us_per_image")),
        config_energy_uJ_per_image=selected_config_energy,
        non_mvm_energy_uJ_per_image=selected_non_mvm_energy,
        device_cfg=device_cfg,
    )
    mrr_lock_summary, mrr_lock_fraction_rows = build_lock_fraction_sensitivity(
        selected_power_w=_float(selected.get("total_power_w")),
        latency_us_per_image=_float(selected.get("latency_us_per_image")),
        config_energy_uJ_per_image=selected_config_energy,
        non_mvm_energy_uJ_per_image=selected_non_mvm_energy,
        hardware_cfg=hardware_cfg,
    )
    recommended_lock_fraction = _float(
        mrr_lock_summary.get("recommended_reporting_fraction"), 0.01
    )
    mrr_reference = next(
        (
            row
            for row in mrr_lock_fraction_rows
            if abs(
                _float(row.get("locked_fraction"), -1.0)
                - recommended_lock_fraction
            )
            < 1.0e-12
        ),
        mrr_lock_fraction_rows[0] if mrr_lock_fraction_rows else {},
    )
    mrr_reference_mw = _float(mrr_reference.get("added_lock_power_mw"), 0.0)
    tile_load_cycles = [
        _int(value)
        for value in _get(
            hardware_cfg,
            "claim_boundary",
            "tile_load_sensitivity_cycles",
            default=[256, 1000, 10000, 100000],
        )
    ]
    base_cycles_without_configuration = (
        _float(transaction.get("total_cycles_per_image"))
        + _float(queue.get("stall_cycles"))
        + _float(digital.get("total_cycles_per_image"))
        + lif_pipeline_flush_cycles
        + adc_queue_pipeline_fill_cycles
    )
    tile_load_rows = build_tile_load_sensitivity(
        dataset=dataset,
        cycles_per_tile_load_values=tile_load_cycles,
        base_cycles_without_configuration=base_cycles_without_configuration,
        clock_hz=clock_hz,
        weight_map=weight_map,
        hardware_cfg=hardware_cfg,
        device_cfg=device_cfg,
        activity=activity,
        queue=queue,
        state=state,
        counts=counts,
        link=link,
        non_mvm_energy_uJ_per_image=selected_non_mvm_energy,
        lock_fraction=recommended_lock_fraction,
        lock_power_mw_value=mrr_reference_mw,
    )
    lock_power = build_power(
        activity,
        queue,
        state,
        counts,
        timing,
        link,
        device_cfg,
        mrr_stabilization_mw=mrr_reference_mw,
    )
    # CIFAR10-DVS was traced with clipped-count input (max=3). The aggregate
    # archive records only nonzero counts, not their 1/2/3 histogram, so the
    # exact first-layer pulse count cannot be reconstructed. Report a bounded
    # bit-serial replay and use the conservative 3-pulse case as the headline.
    config_name = str(activity.get("config", "")).lower()
    clipped_count_input = dataset.lower() == "cifar10dvs" and "clip" in config_name
    pulse_multipliers = [1, 2, 3] if clipped_count_input else [1]
    first_layer = next(iter(activity.get("layers", {})), "")
    input_encoding_rows: List[Dict[str, Any]] = []
    input_encoding_power_rows: Dict[int, List[Dict[str, Any]]] = {}
    for pulse_multiplier in pulse_multipliers:
        pulse_mapping: List[Dict[str, Any]] = []
        for row in mapping_rows:
            copied = dict(row)
            if str(copied.get("layer")) == first_layer:
                copied["output_positions_per_timestep"] = _float(copied.get("output_positions_per_timestep")) * pulse_multiplier
                copied["output_positions_per_image"] = _float(copied.get("output_positions_per_image")) * pulse_multiplier
            pulse_mapping.append(copied)
        pulse_transaction = build_transaction_trace(
            activity, pulse_mapping, hapr_group_size=g, num_tiles=num_tiles,
            array_rows=array_rows, array_cols=array_cols, event_gating=True,
        )
        pulse_queue = simulate_adc_queue(
            pulse_transaction["trace"], system_clock_hz=clock_hz,
            sample_rate_gsps=_float(_get(device_cfg, "adc", "conventional_adc_macro", "sample_rate_gsps", default=10.0), 10.0),
            adc_macros=a,
            fifo_depth=_int(_get(hardware_cfg, "hapr_adc_backend", "adc_fifo_depth", default=4096), 4096),
        )
        pulse_cycles = (
            _float(pulse_transaction.get("total_cycles_per_image"))
            + _float(pulse_queue.get("stall_cycles"))
            + _float(digital.get("total_cycles_per_image"))
            + _float(mrr.get("mrr_reconfiguration_cycles_per_image"))
            + lif_pipeline_flush_cycles
            + adc_queue_pipeline_fill_cycles
        )
        pulse_timing = {"latency_s_per_image": pulse_cycles / max(clock_hz, 1.0)}
        pulse_power = build_power(activity, pulse_queue, state, counts, pulse_timing, link, device_cfg)
        input_encoding_power_rows[pulse_multiplier] = [dict(row) for row in pulse_power["rows"]]
        pulse_latency_us = pulse_cycles / max(clock_hz, 1.0) * 1e6
        pulse_nominal_energy = (
            _float(pulse_power.get("energy_uJ_per_image_before_one_shot_overheads"))
            + selected_config_energy + selected_non_mvm_energy
        )
        input_encoding_rows.append({
            "dataset": dataset,
            "input_encoding": "clipped_count_bit_serial_bound" if clipped_count_input else "binary",
            "first_layer": first_layer,
            "first_layer_pulse_multiplier": pulse_multiplier,
            "latency_us_per_image": pulse_latency_us,
            "nominal_power_w": _float(pulse_power.get("total_power_w")),
            "nominal_energy_uJ_per_image": pulse_nominal_energy,
            "lock_aware_power_w": _float(pulse_power.get("total_power_w")) + mrr_reference_mw * 1.0e-3,
            "lock_aware_energy_uJ_per_image": pulse_nominal_energy + mrr_reference_mw * 1.0e-3 * pulse_latency_us,
            "adc_requests_per_image": _float(pulse_queue.get("requests_per_image")),
            "adc_queue_p99": _float(pulse_queue.get("queue_p99")),
            "adc_stall_cycles": _float(pulse_queue.get("stall_cycles")),
            "status": "bounded_replay_missing_1_2_3_count_histogram" if clipped_count_input else "binary_input",
        })
    headline_input_case = input_encoding_rows[-1]
    headline_power_rows = input_encoding_power_rows[_int(headline_input_case.get("first_layer_pulse_multiplier"), 1)]
    for row in headline_power_rows:
        if row.get("component") == "mrr_stabilization":
            row["power_mw"] = mrr_reference_mw
            row["note"] = "Continuous lock is unmeasured and excluded from the nominal paper-facing headline."

    link_sensitivity_rows: List[Dict[str, Any]] = []
    base_laser_w = _float(link.get("laser_electrical_power_mw")) * 1.0e-3
    pd_required_mw = _float(link.get("pd_required_optical_power_mw"), 0.25)
    fanout = max(_int(link.get("fanout_channels"), 256), 1)
    base_loss_db = _float(link.get("path_loss_db"), 0.0)
    base_latency_us = _float(headline_input_case.get("latency_us_per_image"))
    for loss_delta_db in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for wall_plug_efficiency in [0.10, 0.20, 0.30, 0.40]:
            path_loss_db = base_loss_db + loss_delta_db
            laser_w = pd_required_mw * fanout * (10.0 ** (path_loss_db / 10.0)) / wall_plug_efficiency * 1.0e-3
            system_power_w = _float(headline_input_case.get("nominal_power_w")) - base_laser_w + laser_w + mrr_reference_mw * 1.0e-3
            system_energy_uJ = system_power_w * base_latency_us + selected_config_energy + selected_non_mvm_energy
            link_sensitivity_rows.append({
                "dataset": dataset,
                "path_loss_db": path_loss_db,
                "loss_delta_db": loss_delta_db,
                "wall_plug_efficiency": wall_plug_efficiency,
                "forward_derived_laser_electrical_w": laser_w,
                "lock_aware_system_power_w": system_power_w,
                "lock_aware_energy_uJ_per_image": system_energy_uJ,
                "pd_required_optical_power_mw": pd_required_mw,
                "status": "forward_link_sensitivity_not_device_measurement",
            })
    adc_bias_rows: List[Dict[str, Any]] = []
    for idle_fraction in [0.0, 0.10, 0.20, 0.35, 0.50]:
        sensitivity_cfg = copy.deepcopy(device_cfg)
        sensitivity_cfg["adc"]["conventional_adc_macro"]["idle_bias_fraction"] = idle_fraction
        sensitivity_power = build_power(
            activity,
            queue,
            state,
            counts,
            timing,
            link,
            sensitivity_cfg,
        )
        nominal_energy = (
            _float(sensitivity_power.get("energy_uJ_per_image_before_one_shot_overheads"))
            + selected_config_energy
            + selected_non_mvm_energy
        )
        adc_bias_rows.append(
            {
                "dataset": dataset,
                "adc_idle_bias_fraction": idle_fraction,
                "nominal_power_w": _float(sensitivity_power.get("total_power_w")),
                "nominal_energy_uJ_per_image": nominal_energy,
                "lock_aware_power_w": _float(sensitivity_power.get("total_power_w")) + mrr_reference_mw * 1.0e-3,
                "lock_aware_energy_uJ_per_image": nominal_energy
                + mrr_reference_mw * 1.0e-3 * _float(selected.get("latency_us_per_image")),
                "status": "declared_parameter_sensitivity_not_measurement",
            }
        )
    save_csv_rows(mrr_lock_fraction_rows, out_dir / "mrr_stabilization_scenarios.csv")
    save_csv_rows(mrr_lock_fraction_rows, out_dir / "mrr_lock_fraction_sensitivity.csv")
    save_csv_rows(mrr_device_budget_rows, out_dir / "mrr_device_budget_scenarios.csv")
    save_csv_rows(tile_load_rows, out_dir / "tile_load_sensitivity.csv")
    save_json(mrr_budget_audit, out_dir / "mrr_budget_audit.json")
    save_json(mrr_lock_summary, out_dir / "mrr_lock_fraction_sensitivity.json")
    save_csv_rows(lock_power["rows"], out_dir / "power_breakdown_lock_aware.csv")
    save_csv_rows(adc_bias_rows, out_dir / "adc_idle_bias_sensitivity.csv")
    save_csv_rows(input_encoding_rows, out_dir / "input_encoding_pulse_sensitivity.csv")
    save_csv_rows(link_sensitivity_rows, out_dir / "link_budget_sensitivity.csv")
    save_csv_rows(headline_power_rows, out_dir / "power_breakdown_headline.csv")
    eval105, eval105_status, eval105_path, eval105_validation = _load_eval105_accuracy(args.output_root, dataset)
    hw_verified = eval105_status == "verified_eval_105_publication_replay"
    clean_accuracy = _float(eval105.get("clean_accuracy_percent")) if eval105 else _float(activity.get("accuracy_percent"), 0.0)
    hardware_accuracy = _float(eval105.get("combined_condition_accuracy_percent")) if hw_verified and eval105 else None
    hardware_status = eval105_status
    selected_operating_point = {
        "eval_name": "eval_101",
        "purpose": "unique_selected_mapping_transaction_queue_operating_point",
        "created_utc": now_utc(),
        "dataset": dataset,
        "g4_a8_contract": contract,
        "clean_accuracy_percent": clean_accuracy,
        "quantized_clean_accuracy_percent": eval105.get("quantized_clean_accuracy_percent") if eval105 else None,
        "hardware_aware_accuracy_percent": hardware_accuracy,
        "hardware_aware_accuracy_status": hardware_status,
        "eval_105_summary": str(eval105_path),
        "eval_105_summary_sha256": file_sha256(eval105_path) if eval105_path.exists() else None,
        "eval_105_validation": eval105_validation,
        "partial_adc_only_accuracy_percent_from_legacy_eval05": partial_adc_accuracy,
        "precision": {
            "weight_bits": _int(_get(hardware_cfg, "precision", "weight_bits", default=6), 6),
            "adc_bits": _int(_get(hardware_cfg, "precision", "adc_bits", default=6), 6),
            "membrane_bits": _int(_get(hardware_cfg, "precision", "membrane_bits", default=16), 16),
            "status": "executed_in_eval_105_publication_replay" if hw_verified else "not_verified_without_valid_eval_105",
        },
        "selected_design": selected,
        "metrics": {
            "latency_us_per_image": selected.get("latency_us_per_image"),
            "energy_uJ_per_image": selected.get("energy_uJ_per_image"),
            "power_w": selected.get("total_power_w"),
            "area_mm2_component_lower_bound": selected.get("area_mm2"),
            "core_area_mm2": None,
            "core_area_status": "not_reportable_without full photonic routing, thermal structures, SRAM macros, and post-layout extraction",
            "throughput_images_per_s": selected.get("throughput_images_per_s"),
            "p99_queue": selected.get("p99_queue"),
            "sram_reads_per_image": selected.get("sram_reads_per_image"),
            "sram_writes_per_image": selected.get("sram_writes_per_image"),
            "control_overhead_cycles": _float(selected.get("mrr_config_cycles_per_image")) + _float(selected.get("digital_cycles_per_image")) + _float(selected.get("lif_pipeline_flush_cycles_per_image")) + _float(selected.get("adc_queue_pipeline_fill_cycles_per_image")),
            "nominal_power_w": selected.get("total_power_w"),
            "nominal_energy_uJ_per_image": selected.get("energy_uJ_per_image"),
            "mrr_reference_power_w": mrr_reference.get("total_power_w"),
            "mrr_reference_energy_uJ_per_image": mrr_reference.get("energy_uJ_per_image"),
            "mrr_reference_energy_overhead_percent": mrr_reference.get("energy_overhead_percent"),
            "headline_includes_mrr_reference": False,
            "recommended_reporting_case": "report nominal lower bound together with lock-aware 1% sensitivity",
            "headline_power_w": headline_input_case.get("nominal_power_w"),
            "headline_latency_us_per_image": headline_input_case.get("latency_us_per_image"),
            "headline_energy_uJ_per_image": headline_input_case.get("nominal_energy_uJ_per_image"),
            "headline_input_encoding_case": headline_input_case.get("input_encoding"),
            "headline_first_layer_pulse_multiplier": headline_input_case.get("first_layer_pulse_multiplier"),
        },
        "energy_breakdown_uJ_per_image": {
            "steady_state_power_x_latency": selected_power_latency_energy,
            "mrr_configuration": selected_config_energy,
            "non_mvm_operators": selected_non_mvm_energy,
            "total": selected_energy,
            "mrr_stabilization_reference_additional": mrr_reference_mw * 1.0e-3 * _float(selected.get("latency_us_per_image")),
            "total_with_mrr_stabilization_reference": mrr_reference.get("energy_uJ_per_image"),
        },
        "link_budget": link,
        "device_evidence": _get(
            device_cfg,
            "analog_validation",
            "literature_evidence",
            default={},
        ),
        "analog_validation": {
            "nominal_rows": analog_rows,
            "corner_rows": analog_corner_rows,
            "bound_derivation": hapr_bound_summary,
            "selected_group_survives_all_corners": int(
                all(
                    _int(row.get("feasible"), 0) == 1
                    for row in analog_corner_rows
                    if _int(row.get("hapr_group_size"), -1) == g
                )
            ),
            "model_status": str(
                _get(
                    device_cfg,
                    "analog_validation",
                    "assumption_level",
                    default="architecture_model_assumption",
                )
            ),
        },
        "mrr_stabilization": {
            "device_budget_audit": mrr_budget_audit,
            "lock_fraction_sensitivity": mrr_lock_summary,
            "reference_case": mrr_reference,
            "distinction": "Continuous stabilization power is separate from one-shot MRR configuration energy.",
        },
        "tile_load_sensitivity": {
            "cycles_per_tile_load": tile_load_cycles,
            "rows": tile_load_rows,
            "status": "architecture_parameter_sensitivity_not_measured_programming_time",
        },
        "input_encoding": {
            "mode": "clipped_count" if clipped_count_input else "binary",
            "first_layer": first_layer,
            "pulse_multiplier_cases": pulse_multipliers,
            "headline_case": headline_input_case,
            "note": "CIFAR clipped-count histogram is absent; the 3-pulse bound is used for the headline rather than treating the first layer as binary.",
        },
        "weight_residency": weight_map,
        "hapr_mapping_same_neuron": validate_no_cross_neuron_sum(partial_rows),
        "publication_ready": False,
        "baseline_definition": {"same_four_tiles": True, "same_laser_and_link_budget": True, "same_precision": "W6/ADC6/Vm16", "same_configuration_and_non_mvm_accounting": True, "area_scope": "component_footprint_lower_bound"},
        "publication_ready_reason": "An open-45nm standard-cell area anchor is present, but hardware-aware accuracy, post-route timing/power, SRAM macros, and photonic extraction remain unavailable; software speedup baselines are intentionally omitted.",
    }
    save_json(selected_operating_point, out_dir / "selected_operating_point.json")
    save_json({"selected_operating_point": selected_operating_point}, out_dir / "summary.json")
    save_json(
        {
            "dataset": dataset,
            "selected_operating_point": selected_operating_point,
            "architecture": {
                "mapping": "64x64 signed differential MRR tiles",
                "hapr": "same-output-neuron, same-cycle spatial partial-sum groups across physical tiles",
                "adc": "10 GS/s conventional ADC macro with FIFO backpressure",
                "state": "16-bit membrane with lazy timestamp decay",
                "weight_residency": "layer_weight_tile_major",
            },
            "trace_provenance": transaction["trace_provenance"],
        },
        out_dir / "system_summary.json",
    )
    save_run_manifest(
        out_dir,
        eval_name="eval_101",
        command=" ".join(sys.argv),
        inputs={"eval01_summary": str(input_activity_path), "hardware": args.hardware, "device_params": args.device_params},
        outputs={
            "selected_operating_point": "selected_operating_point.json",
            "design_space": "design_space.csv",
            "transaction_trace": "transaction_trace.csv",
            "adc_queue_trace": "adc_queue_trace.csv",
            "state_update_trace": "state_update_trace.csv",
            "non_mvm_operator_summary": "non_mvm_operator_summary.csv",
            "hapr_corner_validation": "hapr_corner_validation.csv",
            "hapr_bound_derivation": "hapr_bound_derivation.json",
            "mrr_budget_audit": "mrr_budget_audit.json",
            "mrr_stabilization_scenarios": "mrr_stabilization_scenarios.csv",
            "mrr_lock_fraction_sensitivity": "mrr_lock_fraction_sensitivity.csv",
            "mrr_device_budget_scenarios": "mrr_device_budget_scenarios.csv",
            "tile_load_sensitivity": "tile_load_sensitivity.csv",
            "power_breakdown_lock_aware": "power_breakdown_lock_aware.csv",
            "adc_idle_bias_sensitivity": "adc_idle_bias_sensitivity.csv",
            "input_encoding_pulse_sensitivity": "input_encoding_pulse_sensitivity.csv",
            "link_budget_sensitivity": "link_budget_sensitivity.csv",
            "power_breakdown_headline": "power_breakdown_headline.csv",
        },
        extra={
            "dataset": dataset,
            "selected_hapr": g,
            "selected_adc_macros": a,
            "trace_provenance": transaction["trace_provenance"],
            "mrr_reference_case": mrr_reference,
        },
    )
    return selected_operating_point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HIPSA eval_06 selected point (official eval_101 artifacts)")
    p.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    p.add_argument("--input-root", default="results/eval_v2")
    p.add_argument(
        "--activity-summary",
        default=None,
        help="Use one eval_05 runtime activity summary so accuracy and v3 performance share the exact configuration/trace. Requires one --datasets value.",
    )
    p.add_argument("--output-root", default="results/eval_v10")
    p.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    p.add_argument("--device-params", default="configs/device_params.yaml")
    p.add_argument("--hapr-group-sizes", nargs="+", type=int, default=[1, 2, 4], help="Design-space sweep; selected publication point must be G4.")
    p.add_argument("--adc-macro-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64, 128], help="Design-space sweep; selected publication point must be A8.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.activity_summary and len(args.datasets) != 1:
        raise ValueError("--activity-summary requires exactly one dataset.")
    hardware_cfg = load_yaml(args.hardware)
    device_cfg = load_yaml(args.device_params)
    outputs = [run_dataset(dataset, args, hardware_cfg, device_cfg) for dataset in args.datasets]
    combined = Path(args.output_root) / "combined" / "eval_101"
    combined.mkdir(parents=True, exist_ok=True)
    save_json({"eval_name": "eval_101", "created_utc": now_utc(), "datasets": args.datasets, "selected_operating_points": outputs, "publication_ready": False}, combined / "selected_operating_points.json")
    save_csv_rows([{"dataset": row["dataset"], **row["selected_design"], "hardware_aware_accuracy_percent": row["hardware_aware_accuracy_percent"], "publication_ready": row["publication_ready"]} for row in outputs], combined / "selected_operating_points.csv")
    print(f"[eval_101] G4/A8 artifacts saved to {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










