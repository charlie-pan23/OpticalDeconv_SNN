"""
eval_04.py

HAPR / ADC pool / MRR stabilization sensitivity for HIPSA evaluation.

This script does NOT run model inference.
It reads eval_01 activity summaries and reuses eval_02 hardware models.

Inputs:
  results/eval_v2/<dataset>/eval_01/summary.json
  configs/hardware_hipsa.yaml
  configs/device_params.yaml

Outputs:
  results/eval_v2/<dataset>/eval_04/
    summary.json
    adc_pool_sweep.csv
    hapr_adc_sweep.csv
    mrr_sensitivity.csv
    selected_design_points.csv
    config_snapshot.yaml
    run_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_02 import (
    as_float,
    as_int,
    derive_architecture_counts,
    estimate_adc_requests,
    estimate_power,
    estimate_timing,
    model_adc_pool,
    nested_get,
)
from utils.result_io import (
    dataset_eval_dir,
    load_json,
    load_yaml,
    save_csv_rows,
    save_json,
    save_run_manifest,
    save_yaml,
)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def bool_to_int(x: Any) -> int:
    return 1 if bool(x) else 0


def load_eval01_activity(dataset: str, input_root: str | Path) -> Dict[str, Any]:
    path = dataset_eval_dir(dataset, "eval_01", root=input_root) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_01 summary: {path}")
    return load_json(path)


def compute_design_point(
    *,
    dataset: str,
    activity: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    hapr_group_size: int,
    adc_macros: int,
    mrr_stabilization_mw: Optional[float],
    modulator_activity_source: str,
    adc_power_mode: str,
    design_name: str,
) -> Dict[str, Any]:
    counts = derive_architecture_counts(
        hardware_cfg,
        hapr_group_size_override=hapr_group_size,
        adc_macros_override=adc_macros,
    )

    adc_requests = estimate_adc_requests(
        activity,
        hapr_group_size=as_int(counts["hapr_group_size"], hapr_group_size),
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
        modulator_activity_source=modulator_activity_source,
        adc_power_mode=adc_power_mode,
        mrr_stabilization_mw=mrr_stabilization_mw,
    )

    active_sop_per_image = as_float(activity.get("active_sop_per_image"), 0.0)
    dense_sop_per_image = as_float(activity.get("dense_sop_per_image"), 0.0)

    return {
        "dataset": dataset,
        "design_name": design_name,

        "num_samples": as_int(activity.get("num_samples"), 0),
        "accuracy_percent": as_float(activity.get("accuracy_percent"), 0.0),
        "loss": as_float(activity.get("loss"), 0.0),

        "hapr_group_size": as_int(counts.get("hapr_group_size"), hapr_group_size),
        "hapr_output_lanes_total": as_int(counts.get("hapr_output_lanes_total"), 0),
        "adc_macros": as_int(counts.get("adc_macros"), adc_macros),
        "mrr_stabilization_mw": 0.0 if mrr_stabilization_mw is None else float(mrr_stabilization_mw),

        "dense_sop_per_image": dense_sop_per_image,
        "active_sop_per_image": active_sop_per_image,
        "active_sop_ratio": as_float(activity.get("active_sop_ratio"), 0.0),

        "model_input_activity": as_float(activity.get("model_input_activity"), 0.0),
        "mvm_input_activity": as_float(activity.get("mvm_input_activity"), 0.0),
        "lif_spike_activity": as_float(activity.get("lif_spike_activity"), 0.0),
        "adc_element_request_activity": as_float(activity.get("adc_request_activity"), 0.0),

        "adc_group_request_activity": as_float(adc_requests.get("adc_group_request_activity"), 0.0),
        "adc_requests_per_image": as_float(adc_requests.get("adc_requests_per_image"), 0.0),
        "adc_group_opportunities_per_image": as_float(adc_requests.get("adc_group_opportunities_per_image"), 0.0),

        "base_cycles_per_image": as_float(timing.get("base_cycles_per_image"), 0.0),
        "adc_demand_per_cycle": as_float(adc_pool.get("adc_demand_per_cycle"), 0.0),
        "adc_macro_utilization": as_float(adc_pool.get("adc_macro_utilization"), 0.0),
        "adc_stall_cycles_proxy": as_float(adc_pool.get("adc_stall_cycles_proxy"), 0.0),
        "adc_is_saturated": bool_to_int(adc_pool.get("adc_is_saturated", False)),

        "latency_us_per_image": as_float(timing.get("latency_us_per_image"), 0.0),
        "throughput_images_per_s": as_float(timing.get("throughput_images_per_s"), 0.0),

        "total_power_w": as_float(power.get("total_power_w"), 0.0),
        "total_power_mw": as_float(power.get("total_power_mw"), 0.0),
        "energy_uJ_per_image": as_float(power.get("energy_uJ_per_image"), 0.0),
        "active_GOPS_per_W": as_float(power.get("active_GOPS_per_W"), 0.0),
        "dense_equivalent_GOPS_per_W": as_float(power.get("dense_equivalent_GOPS_per_W"), 0.0),

        "adc_power_mode": adc_power_mode,
        "modulator_activity_source": modulator_activity_source,
    }


def default_mrr_stress_cases(device_cfg: Mapping[str, Any], include_thermal_per_ring: bool) -> List[float]:
    cases = nested_get(device_cfg, "mrr_stabilization", "stress_cases_mw", default=[0.0, 100.0, 250.0, 350.0, 500.0])

    if not isinstance(cases, list):
        cases = [0.0, 100.0, 250.0, 350.0, 500.0]

    values = [float(x) for x in cases]

    if include_thermal_per_ring:
        # 4 tiles * 64 * 64 logical weights * 1.2 mW/weight = 19660.8 mW.
        thermal_full_lock_mw = 4 * 64 * 64 * 1.2
        values.append(float(thermal_full_lock_mw))

    unique_sorted = sorted(set(values))
    return unique_sorted


def select_design_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select useful points for Section 4 discussion.

    We select:
    1. default point: HAPR=8, ADC=16 if present
    2. first non-saturated point with minimal energy
    3. minimal-energy point overall
    """

    if not rows:
        return []

    selected: List[Dict[str, Any]] = []

    default_points = [
        r for r in rows
        if as_int(r.get("hapr_group_size")) == 8 and as_int(r.get("adc_macros")) == 16
    ]
    if default_points:
        p = dict(default_points[0])
        p["selection_reason"] = "default_main_case"
        selected.append(p)

    non_sat = [r for r in rows if as_int(r.get("adc_is_saturated")) == 0]
    if non_sat:
        p = dict(min(non_sat, key=lambda x: as_float(x.get("energy_uJ_per_image"))))
        p["selection_reason"] = "lowest_energy_non_saturated"
        selected.append(p)

    p = dict(min(rows, key=lambda x: as_float(x.get("energy_uJ_per_image"))))
    p["selection_reason"] = "lowest_energy_overall"
    selected.append(p)

    # Deduplicate by HAPR/ADC/MRR.
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for p in selected:
        key = (
            p.get("hapr_group_size"),
            p.get("adc_macros"),
            p.get("mrr_stabilization_mw"),
            p.get("selection_reason"),
        )
        if key not in seen:
            seen.add(key)
            dedup.append(p)

    return dedup


def run_dataset(
    *,
    dataset: str,
    args: argparse.Namespace,
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    activity = load_eval01_activity(dataset, args.input_root)

    output_dir = dataset_eval_dir(dataset, "eval_04", root=args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_hapr = int(args.default_hapr_group_size)
    default_adc = int(args.default_adc_macros)

    adc_pool_rows: List[Dict[str, Any]] = []
    for adc_macros in args.adc_pool_sizes:
        row = compute_design_point(
            dataset=dataset,
            activity=activity,
            hardware_cfg=hardware_cfg,
            device_cfg=device_cfg,
            hapr_group_size=default_hapr,
            adc_macros=int(adc_macros),
            mrr_stabilization_mw=0.0,
            modulator_activity_source=args.modulator_activity_source,
            adc_power_mode=args.adc_power_mode,
            design_name=f"adc{int(adc_macros)}_hapr{default_hapr}",
        )
        adc_pool_rows.append(row)

    hapr_adc_rows: List[Dict[str, Any]] = []
    for hapr_g in args.hapr_group_sizes:
        for adc_macros in args.adc_pool_sizes:
            row = compute_design_point(
                dataset=dataset,
                activity=activity,
                hardware_cfg=hardware_cfg,
                device_cfg=device_cfg,
                hapr_group_size=int(hapr_g),
                adc_macros=int(adc_macros),
                mrr_stabilization_mw=0.0,
                modulator_activity_source=args.modulator_activity_source,
                adc_power_mode=args.adc_power_mode,
                design_name=f"adc{int(adc_macros)}_hapr{int(hapr_g)}",
            )
            hapr_adc_rows.append(row)

    mrr_cases = args.mrr_stress_mw
    if not mrr_cases:
        mrr_cases = default_mrr_stress_cases(
            device_cfg,
            include_thermal_per_ring=not args.no_thermal_per_ring_case,
        )

    mrr_rows: List[Dict[str, Any]] = []
    for mrr_mw in mrr_cases:
        row = compute_design_point(
            dataset=dataset,
            activity=activity,
            hardware_cfg=hardware_cfg,
            device_cfg=device_cfg,
            hapr_group_size=default_hapr,
            adc_macros=default_adc,
            mrr_stabilization_mw=float(mrr_mw),
            modulator_activity_source=args.modulator_activity_source,
            adc_power_mode=args.adc_power_mode,
            design_name=f"mrr{float(mrr_mw):.1f}mw_adc{default_adc}_hapr{default_hapr}",
        )
        mrr_rows.append(row)

    selected_rows = select_design_points(hapr_adc_rows)

    summary = {
        "eval_name": "eval_04",
        "purpose": "hapr_adc_pool_mrr_sensitivity",
        "created_utc": now_utc(),
        "command": " ".join(sys.argv),

        "dataset": dataset,
        "input_eval01_summary": str(dataset_eval_dir(dataset, "eval_01", root=args.input_root) / "summary.json"),
        "hardware": str(args.hardware),
        "device_params": str(args.device_params),

        "default_hapr_group_size": default_hapr,
        "default_adc_macros": default_adc,
        "adc_pool_sizes": [int(x) for x in args.adc_pool_sizes],
        "hapr_group_sizes": [int(x) for x in args.hapr_group_sizes],
        "mrr_stress_mw": [float(x) for x in mrr_cases],

        "activity": {
            "num_samples": activity.get("num_samples"),
            "accuracy_percent": activity.get("accuracy_percent"),
            "dense_sop_per_image": activity.get("dense_sop_per_image"),
            "active_sop_per_image": activity.get("active_sop_per_image"),
            "active_sop_ratio": activity.get("active_sop_ratio"),
            "adc_request_activity": activity.get("adc_request_activity"),
            "lif_spike_activity": activity.get("lif_spike_activity"),
        },

        "selected_design_points": selected_rows,

        "notes": {
            "adc_pool_sweep": "Varies ADC macros at the default HAPR group size.",
            "hapr_adc_sweep": "Full cross-product of HAPR group size and ADC pool size.",
            "mrr_sensitivity": "Varies MRR stabilization power while keeping HAPR/ADC at default.",
            "main_case": "MRR stabilization uses 0 mW unless explicitly swept.",
            "thermal_per_ring_case": "Optional stress case: 4*64*64*1.2 mW = 19.6608 W.",
        },
    }

    save_json(summary, output_dir / "summary.json")
    save_csv_rows(adc_pool_rows, output_dir / "adc_pool_sweep.csv")
    save_csv_rows(hapr_adc_rows, output_dir / "hapr_adc_sweep.csv")
    save_csv_rows(mrr_rows, output_dir / "mrr_sensitivity.csv")
    save_csv_rows(selected_rows, output_dir / "selected_design_points.csv")

    save_yaml(
        {
            "eval_04_args": vars(args),
            "hardware": dict(hardware_cfg),
            "device_params": dict(device_cfg),
        },
        output_dir / "config_snapshot.yaml",
    )

    save_run_manifest(
        output_dir,
        eval_name="eval_04",
        command=" ".join(sys.argv),
        inputs={
            "eval01_summary": str(dataset_eval_dir(dataset, "eval_01", root=args.input_root) / "summary.json"),
            "hardware": str(args.hardware),
            "device_params": str(args.device_params),
        },
        outputs={
            "summary": "summary.json",
            "adc_pool_sweep": "adc_pool_sweep.csv",
            "hapr_adc_sweep": "hapr_adc_sweep.csv",
            "mrr_sensitivity": "mrr_sensitivity.csv",
            "selected_design_points": "selected_design_points.csv",
        },
        extra={
            "dataset": dataset,
            "adc_pool_sizes": args.adc_pool_sizes,
            "hapr_group_sizes": args.hapr_group_sizes,
        },
    )

    print("=" * 80)
    print("[eval_04] sensitivity complete")
    print(f"dataset              : {dataset}")
    print(f"default HAPR / ADC   : {default_hapr} / {default_adc}")
    print(f"ADC pool sizes       : {args.adc_pool_sizes}")
    print(f"HAPR group sizes     : {args.hapr_group_sizes}")
    print(f"MRR stress cases mW  : {mrr_cases}")
    print(f"output_dir           : {output_dir}")
    if selected_rows:
        print("selected points:")
        for row in selected_rows:
            print(
                "  {reason}: HAPR={hapr} ADC={adc} sat={sat} latency={lat:.2f}us energy={eng:.2f}uJ".format(
                    reason=row.get("selection_reason"),
                    hapr=row.get("hapr_group_size"),
                    adc=row.get("adc_macros"),
                    sat=row.get("adc_is_saturated"),
                    lat=as_float(row.get("latency_us_per_image")),
                    eng=as_float(row.get("energy_uJ_per_image")),
                )
            )
    print("=" * 80)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HIPSA eval_04: HAPR / ADC pool / MRR sensitivity"
    )

    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--input-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-root", default="results/eval_v2", type=str)

    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml", type=str)
    parser.add_argument("--device-params", default="configs/device_params.yaml", type=str)

    parser.add_argument("--default-hapr-group-size", default=8, type=int)
    parser.add_argument("--default-adc-macros", default=16, type=int)

    parser.add_argument("--adc-pool-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--hapr-group-sizes", nargs="+", type=int, default=[4, 8, 16, 32])

    parser.add_argument("--mrr-stress-mw", nargs="*", type=float, default=None)
    parser.add_argument("--no-thermal-per-ring-case", action="store_true")

    parser.add_argument(
        "--modulator-activity-source",
        default="mvm_input_activity",
        choices=["model_input_activity", "mvm_input_activity", "active_sop_ratio", "always_on"],
    )
    parser.add_argument(
        "--adc-power-mode",
        default="activity_scaled",
        choices=["activity_scaled", "all_biased"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hardware_cfg = load_yaml(args.hardware)
    device_cfg = load_yaml(args.device_params)

    summaries: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        summary = run_dataset(
            dataset=dataset,
            args=args,
            hardware_cfg=hardware_cfg,
            device_cfg=device_cfg,
        )
        summaries.append(summary)

    combined_dir = Path(args.output_root) / "combined" / "eval_04"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_selected = []
    for summary in summaries:
        for row in summary.get("selected_design_points", []):
            combined_selected.append(row)

    save_json(
        {
            "eval_name": "eval_04",
            "created_utc": now_utc(),
            "datasets": args.datasets,
            "summaries": [
                {
                    "dataset": s["dataset"],
                    "selected_design_points": s.get("selected_design_points", []),
                }
                for s in summaries
            ],
        },
        combined_dir / "summary.json",
    )
    save_csv_rows(combined_selected, combined_dir / "selected_design_points.csv")

    print(f"[eval_04] combined summary saved to {combined_dir}")


if __name__ == "__main__":
    main()