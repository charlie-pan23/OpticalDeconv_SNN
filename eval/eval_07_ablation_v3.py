"""Official eval_104 mechanism ablation using the G4/A8 contract."""
from __future__ import annotations
import argparse
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.adc_pool_model import simulate_adc_queue
from hardware.layer_mapper import build_layer_mapping, summarize_mapping
from hardware.link_budget import derive_link_budget
from hardware.mrr_config_model import model_mrr_configuration
from hardware.non_mvm_model import build_non_mvm_summary
from hardware.publication_accounting import build_area_breakdown, build_power, lock_power_mw
from hardware.state_update_model import build_state_update_trace
from hardware.transaction_scheduler import build_transaction_trace
from hardware.weight_mapper import build_weight_map
from utils.eval_v10 import file_sha256, validate_eval105_publication_summary, validate_g4_a8_contract
from utils.result_io import load_json, load_yaml, save_csv_rows, save_json, save_run_manifest


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _float(value: Any, default: float = 0.0) -> float:
    try: return float(value)
    except Exception: return float(default)


def _int(value: Any, default: int = 0) -> int:
    try: return int(value)
    except Exception: return int(default)


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur: return default
        cur = cur[key]
    return cur


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
def run_variant(dataset: str, activity: Mapping[str, Any], mapping_rows: List[Mapping[str, Any]], weight_map: Mapping[str, Any], hardware_cfg: Mapping[str, Any], device_cfg: Mapping[str, Any], *, variant_id: str, family: str, event_gating: bool, hapr_group_size: int, adc_macros: int, adc_pooling: bool, lazy_lif: bool, lock_fraction: float, baseline_id: str) -> Dict[str, Any]:
    num_tiles = _int(_get(hardware_cfg, "photonic_tiles", "num_tiles", default=4), 4)
    rows = _int(_get(hardware_cfg, "mapping", "array_rows", default=64), 64)
    cols = _int(_get(hardware_cfg, "mapping", "array_cols", default=64), 64)
    clock = _float(_get(hardware_cfg, "transaction_scheduler", "system_clock_hz", default=1e9), 1e9)
    sample_rate = _float(_get(device_cfg, "adc", "conventional_adc_macro", "sample_rate_gsps", default=10.0), 10.0)
    slots_per_macro = _float(_get(hardware_cfg, "hapr_adc_backend", "adc_samples_per_macro_per_architecture_cycle", default=10.0), 10.0)
    lanes = (num_tiles // max(hapr_group_size, 1)) * cols
    actual_adc_macros = int(adc_macros)
    same_window_feasible = actual_adc_macros * slots_per_macro >= lanes
    transaction = build_transaction_trace(activity, mapping_rows, hapr_group_size=hapr_group_size, num_tiles=num_tiles, array_rows=rows, array_cols=cols, event_gating=event_gating)
    queue = simulate_adc_queue(transaction["trace"], system_clock_hz=clock, sample_rate_gsps=sample_rate, adc_macros=actual_adc_macros, fifo_depth=_int(_get(hardware_cfg, "hapr_adc_backend", "adc_fifo_depth", default=4096), 4096))
    state = build_state_update_trace(activity, transaction["layer_summary"], lazy_lif=lazy_lif)
    digital = build_non_mvm_summary(activity, workload_name=dataset, time_steps=_int(activity.get("time_steps"), 1), operator_cfg={**dict(_get(hardware_cfg, "digital_ops", default={}) or {}), "num_classes": 10 if dataset == "cifar10dvs" else 11})
    link = derive_link_budget(device_cfg)
    mrr = model_mrr_configuration(weight_map, hardware_cfg, device_cfg)
    pipeline = _float(_get(hardware_cfg, "transaction_scheduler", "lif_pipeline_flush_cycles", default=0.0)) + _float(_get(hardware_cfg, "transaction_scheduler", "adc_queue_pipeline_fill_cycles", default=0.0))
    cycles = _float(transaction.get("total_cycles_per_image")) + _float(queue.get("stall_cycles")) + _float(digital.get("total_cycles_per_image")) + _float(mrr.get("mrr_reconfiguration_cycles_per_image")) + pipeline
    timing = {"latency_s_per_image": cycles / max(clock, 1.0), "latency_us_per_image": cycles / max(clock, 1.0) * 1e6, "throughput_images_per_s": max(clock, 1.0) / max(cycles, 1.0)}
    counts = {"num_tiles": num_tiles, "array_rows": rows, "array_cols": cols, "tile_outputs": cols, "hapr_group_size": hapr_group_size, "adc_macros": actual_adc_macros, "photodetector_outputs_total": num_tiles * cols, "modulator_lanes_total": num_tiles * cols, "physical_hapr_lanes_total": lanes}
    lock_mw = lock_power_mw(hardware_cfg, lock_fraction)
    power = build_power(activity, queue, state, counts, timing, link, device_cfg, mrr_stabilization_mw=lock_mw)
    area = build_area_breakdown(counts, weight_map, hardware_cfg, device_cfg)
    energy = _float(power.get("energy_uJ_per_image_before_one_shot_overheads")) + _float(mrr.get("configuration_energy_pj_per_image")) / 1e6 + _float(digital.get("total_energy_pj_per_image")) / 1e6
    return {
        "dataset": dataset, "ablation_family": family, "variant": variant_id, "baseline_id": baseline_id,
        "event_gating": int(event_gating), "hapr_group_size": hapr_group_size, "adc_macros": actual_adc_macros,
        "adc_pooling": int(adc_pooling), "lazy_lif": int(lazy_lif), "lock_fraction": lock_fraction,
        "tile_load_cycles": _int(_get(hardware_cfg, "mrr_configuration", "cycles_per_tile_load", default=256), 256),
        "post_hapr_lanes": lanes, "nominal_adc_slots_per_cycle": actual_adc_macros * slots_per_macro,
        "same_window_admission_feasible": int(same_window_feasible), "requests_dropped": _float(queue.get("requests_dropped")),
        "transactions_issued_per_image": _float(transaction.get("transactions_issued_per_image")), "adc_conversions_per_image": _float(queue.get("requests_per_image")),
        "adc_macro_utilization": _float(queue.get("adc_utilization")), "p99_queue": _float(queue.get("queue_p99")), "adc_stall_cycles": _float(queue.get("stall_cycles")),
        "sram_reads_per_image": _float(state.get("sram_reads_per_image")), "sram_writes_per_image": _float(state.get("sram_writes_per_image")),
        "latency_us_per_image": timing["latency_us_per_image"], "throughput_images_per_s": timing["throughput_images_per_s"],
        "power_w": _float(power.get("total_power_w")), "energy_uJ_per_image": energy, "area_mm2_component_lower_bound": _float(area.get("area_mm2")),
        "lock_power_mw": lock_mw, "hardware_aware_accuracy_percent": None,
        "accuracy_status": "not_inferred_from_aggregate_activity_join_eval_105_for_accuracy",
        "trace_provenance": transaction.get("trace_provenance", "aggregate_layer_timestep_tile_transaction_model"),
        "evidence_class": "architecture_level_estimate", "circuit_timing_closed": False,
    }


def run_dataset(dataset: str, args: argparse.Namespace, hardware_cfg: Mapping[str, Any], device_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    contract = validate_g4_a8_contract(hardware_cfg)
    activity_path = Path(args.input_root) / dataset / "eval_01" / "summary.json"
    if not activity_path.exists(): raise FileNotFoundError(f"Missing activity summary: {activity_path}")
    activity = load_json(activity_path)
    mapping_rows = build_layer_mapping(activity, array_rows=contract["array_rows"], array_cols=contract["array_cols"], num_tiles=contract["num_tiles"])
    mapping_summary = summarize_mapping(mapping_rows)
    weight_map = build_weight_map(mapping_summary, num_tiles=contract["num_tiles"], signed_weight_multiplier=2, weight_bits=_int(_get(hardware_cfg, "precision", "weight_bits", default=6), 6))
    g, a = contract["hapr_group_size"], contract["adc_macros"]
    physical_lanes = contract["num_tiles"] * contract["array_cols"]
    g4_lanes = contract["post_hapr_lanes"]
    min_no_hapr_pool = int(math.ceil(physical_lanes / contract["samples_per_macro_per_cycle"]))
    lock_fraction = float(args.lock_fraction)
    baseline_id = "matched_four_tile_w6_adc6_vm16_lock_and_configuration_accounting"
    incremental_specs = [
        ("B0_dense_interface", False, 1, physical_lanes, False, False),
        ("B1_plus_event_gating", True, 1, physical_lanes, False, False),
        ("B2_plus_HAPR", True, g, g4_lanes, False, False),
        ("B3_plus_ADC_pooling", True, g, a, True, False),
        ("B4_full_HIPSA_plus_lazy_LIF", True, g, a, True, True),
    ]
    leave_specs = [
        ("full_hipsa", True, g, a, True, True),
        ("without_event_gating", False, g, a, True, True),
        ("without_hapr", True, 1, min_no_hapr_pool, True, True),
        ("without_adc_pooling", True, g, g4_lanes, False, True),
        ("without_lazy_lif", True, g, a, True, False),
    ]
    orthogonal_specs = [
        ("adc_pooling_without_hapr", True, 1, min_no_hapr_pool, True, False),
        ("hapr_with_64_adcs", True, g, 64, False, False),
        ("lazy_lif_without_hapr", True, 1, physical_lanes, False, True),
        ("hapr_a8_without_lazy_lif", True, g, a, True, False),
        ("full_hipsa", True, g, a, True, True),
    ]
    def execute(family: str, specs: List[Any]) -> List[Dict[str, Any]]:
        return [run_variant(dataset, activity, mapping_rows, weight_map, hardware_cfg, device_cfg, variant_id=name, family=family, event_gating=eg, hapr_group_size=hg, adc_macros=am, adc_pooling=pool, lazy_lif=lazy, lock_fraction=lock_fraction, baseline_id=baseline_id) for name, eg, hg, am, pool, lazy in specs]
    incremental, leave_one_out, orthogonal = execute("incremental", incremental_specs), execute("leave_one_out", leave_specs), execute("orthogonal", orthogonal_specs)
    out = Path(args.output_root) / dataset / "eval_104"; out.mkdir(parents=True, exist_ok=True)
    eval105, accuracy_status, eval105_path, eval105_validation = _load_eval105_accuracy(args.output_root, dataset)
    combined_accuracy = eval105.get("combined_condition_accuracy_percent") if eval105 and accuracy_status == "verified_eval_105_publication_replay" else None
    for rows in (incremental, leave_one_out, orthogonal):
        for row in rows:
            if row.get("variant") in {"B4_full_HIPSA_plus_lazy_LIF", "full_hipsa"} and combined_accuracy is not None:
                row["hardware_aware_accuracy_percent"] = combined_accuracy
                row["accuracy_status"] = accuracy_status
            else:
                row["accuracy_status"] = "variant_not_replayed; no accuracy inferred from architecture counters"
    baseline = {"baseline_id": baseline_id, "same_four_tiles": True, "same_laser_and_link_budget": True, "same_lock_fraction": lock_fraction, "same_precision": "W6/ADC6/Vm16", "same_configuration_accounting": True, "same_non_mvm_accounting": True, "accuracy_policy": "full HIPSA joins verified eval_105; other variants require dedicated replay"}
    save_csv_rows(incremental, out / "incremental_ablation.csv"); save_csv_rows(leave_one_out, out / "leave_one_out_ablation.csv"); save_csv_rows(orthogonal, out / "orthogonal_ablation.csv")
    save_json(baseline, out / "baseline_definition.json")
    summary = {"eval_name": "eval_104", "created_utc": now_utc(), "dataset": dataset, "g4_a8_contract": contract, "baseline_definition": baseline, "incremental": incremental, "leave_one_out": leave_one_out, "orthogonal": orthogonal, "accuracy": {"clean_percent": eval105.get("clean_accuracy_percent") if eval105 else _float(activity.get("accuracy_percent")), "quantized_clean_percent": eval105.get("quantized_clean_accuracy_percent") if eval105 else None, "hardware_aware_percent": combined_accuracy, "status": accuracy_status, "eval_105_summary": str(eval105_path), "eval_105_validation": eval105_validation}, "hashes": {"input_summary_sha256": file_sha256(activity_path), "eval_105_summary_sha256": file_sha256(eval105_path) if eval105_path.exists() else None, "hardware_config_sha256": file_sha256(args.hardware), "device_params_sha256": file_sha256(args.device_params)}}
    save_json(summary, out / "summary.json")
    save_run_manifest(out, eval_name="eval_104", command=" ".join(sys.argv), inputs={"activity_summary": str(activity_path), "hardware": args.hardware, "device_params": args.device_params}, outputs={"summary": "summary.json", "incremental": "incremental_ablation.csv", "leave_one_out": "leave_one_out_ablation.csv", "orthogonal": "orthogonal_ablation.csv", "baseline": "baseline_definition.json"}, extra={"dataset": dataset, "selected_point": "G4/A8", "lock_fraction": lock_fraction})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_07 ablation (official eval_104 artifacts)")
    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--input-root", default="results/eval_v2")
    parser.add_argument("--output-root", default="results/eval_v10")
    parser.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
    parser.add_argument("--lock-fraction", type=float, default=0.01, choices=[0.0, 0.01, 0.05, 0.10])
    return parser.parse_args()


def main() -> int:
    args = parse_args(); hardware_cfg = load_yaml(args.hardware); device_cfg = load_yaml(args.device_params)
    outputs = [run_dataset(dataset, args, hardware_cfg, device_cfg) for dataset in args.datasets]
    out = Path(args.output_root) / "combined" / "eval_104"; out.mkdir(parents=True, exist_ok=True)
    save_json({"eval_name": "eval_104", "created_utc": now_utc(), "datasets": args.datasets, "selected_point": "G4/A8", "lock_fraction": args.lock_fraction}, out / "summary.json")
    for name in ["incremental", "leave_one_out", "orthogonal"]: save_csv_rows([row for item in outputs for row in item[name]], out / f"{name}_ablation.csv")
    print(f"[eval_104] G4/A8 ablations saved to {out}"); return 0


if __name__ == "__main__": raise SystemExit(main())


