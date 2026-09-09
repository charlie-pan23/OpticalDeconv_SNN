"""Phase 4 evidence closure and publication audit for HIPSA/DATE.

This module only consumes existing Phase 3/earlier artifacts.  It deliberately
keeps architecture-level accounting, software replay, mapping-only analysis,
and external anchors as separate evidence classes.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]


def _as_float(value: Any) -> float | None:
    if value is None or value == "" or value == "null":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for Phase 4")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return value


def _stats(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "std_sample": None, "ci95_half_width": None}
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
        ci = 1.96 * std / math.sqrt(len(values))
    else:
        std = 0.0
        ci = None
    return {
        "n": len(values),
        "mean": mean,
        "std_sample": std,
        "ci95_half_width": ci,
        "min": min(values),
        "max": max(values),
    }


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _metric_container(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = metrics.get("selected_operating_point", metrics)
    if isinstance(selected, Mapping):
        return selected
    return {}


def validate_phase4_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    hardware = contract.get("hardware", {})
    precision = contract.get("precision", {})
    seeds = contract.get("seed_provenance", {})
    cases = contract.get("reporting_cases", {})
    expected = {
        "photonic_tiles": 4,
        "array_shape": [64, 64],
        "hapr_group": 4,
        "post_hapr_lanes": 64,
        "adc_macros": 8,
        "adc_samples_per_macro_per_cycle": 10,
        "architecture_clock_hz": 1_000_000_000,
        "temporal_analog_storage": False,
    }
    for key, value in expected.items():
        if hardware.get(key) != value:
            reasons.append(f"hardware.{key} expected {value!r}, got {hardware.get(key)!r}")
    for key, value in {"weight_bits": 6, "adc_bits": 6, "membrane_bits": 16}.items():
        if precision.get(key) != value:
            reasons.append(f"precision.{key} expected {value}, got {precision.get(key)!r}")
    required_seeds = seeds.get("required_seeds", [])
    if list(required_seeds) != [0, 1, 2, 3, 4]:
        reasons.append(f"required_seeds must be [0,1,2,3,4], got {required_seeds!r}")
    if seeds.get("minimum_seeds") != 5:
        reasons.append("minimum_seeds must be 5")
    for name in ("nominal_lower_bound", "representative_lock_aware", "sensitivity"):
        if name not in cases:
            reasons.append(f"missing reporting case: {name}")
    if cases.get("nominal_lower_bound", {}).get("lock_fraction") != 0.0:
        reasons.append("nominal_lower_bound must use lock_fraction=0.0")
    if cases.get("representative_lock_aware", {}).get("lock_fraction") != 0.01:
        reasons.append("representative_lock_aware must use lock_fraction=0.01")
    return {"valid": not reasons, "passed": not reasons, "reasons": reasons}


def _summary_record(path: Path, *, dataset: str, source_role: str) -> dict[str, Any]:
    summary = _read_json(path)
    runtime = summary.get("runtime_provenance", {})
    split = summary.get("split_manifest", {})
    hashes = summary.get("hashes", {})
    checkpoint = summary.get("checkpoint_summary", {})
    record = {
        "dataset": dataset,
        "seed": _as_int(summary.get("seed")),
        "source_path": str(path),
        "source_path_relative": _rel(path),
        "source_role": source_role,
        "eval_name": summary.get("eval_name"),
        "split": summary.get("split"),
        "num_samples": _as_int(summary.get("num_samples")),
        "clean_accuracy_percent": _as_float(summary.get("clean_accuracy_percent")),
        "quantized_clean_accuracy_percent": _as_float(summary.get("quantized_clean_accuracy_percent")),
        "combined_accuracy_percent": _as_float(summary.get("combined_condition_accuracy_percent")),
        "hardware_aware_accuracy_percent": _as_float(summary.get("hardware_aware_accuracy_percent")),
        "accuracy_status": summary.get("hardware_aware_accuracy_status"),
        "replay_scope_status": summary.get("replay_scope", {}).get("status"),
        "split_order_hash": split.get("split_order_hash"),
        "split_file_sha256": split.get("split_file_sha256"),
        "config_sha256": hashes.get("config_sha256"),
        "checkpoint_sha256": hashes.get("checkpoint_sha256"),
        "hardware_config_sha256": hashes.get("hardware_config_sha256"),
        "device_params_sha256": hashes.get("device_params_sha256"),
        "checkpoint_path": checkpoint.get("path"),
        "python_version": runtime.get("python_version"),
        "torch_version": runtime.get("torch_version"),
        "cuda_runtime_version": runtime.get("cuda_runtime_version"),
        "gpu_name": runtime.get("gpu_name"),
        "deterministic_algorithms_enabled": runtime.get("deterministic_algorithms_enabled"),
        "full_replay_requested": runtime.get("full_replay_requested"),
        "publication_replay_ready": runtime.get("publication_replay_ready"),
        "summary_sha256": _sha256(path),
    }
    return record


def collect_seed_records(root: Path, datasets: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        eval_dir = root / dataset / "eval_105"
        root_summary = eval_dir / "summary.json"
        if root_summary.exists():
            records.append(_summary_record(root_summary, dataset=dataset, source_role="root_summary_seed0_legacy_artifact"))
        seed_dir = eval_dir / "seeds"
        for path in sorted(seed_dir.glob("seed_*/summary.json")):
            record = _summary_record(path, dataset=dataset, source_role="explicit_seed_artifact")
            records.append(record)
    return records


def _validate_seed_records(records: list[Mapping[str, Any]], *, required_seeds: list[int]) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(str(record.get("dataset")), []).append(record)
    details: dict[str, Any] = {}
    identity_fields = ["split", "split_order_hash", "config_sha256", "checkpoint_sha256", "hardware_config_sha256", "device_params_sha256"]
    for dataset, rows in by_dataset.items():
        seed_values = [row.get("seed") for row in rows]
        duplicates = sorted({seed for seed in seed_values if seed_values.count(seed) > 1 and seed is not None})
        missing = sorted(set(required_seeds) - {seed for seed in seed_values if seed is not None})
        if duplicates:
            reasons.append(f"{dataset}:duplicate_seeds:{duplicates}")
        if missing:
            reasons.append(f"{dataset}:missing_seeds:{missing}")
        identities = {field: sorted({str(row.get(field)) for row in rows}) for field in identity_fields}
        inconsistent = {field: values for field, values in identities.items() if len(values) > 1}
        if inconsistent:
            reasons.append(f"{dataset}:provenance_mismatch:{sorted(inconsistent)}")
        invalid = [row.get("seed") for row in rows if row.get("publication_replay_ready") is not True]
        if invalid:
            reasons.append(f"{dataset}:not_publication_replay_ready:{invalid}")
        details[dataset] = {
            "observed_seeds": sorted(seed for seed in seed_values if seed is not None),
            "missing_seeds": missing,
            "duplicate_seeds": duplicates,
            "identity_values": identities,
            "inconsistent_identity": inconsistent,
            "replay_ready_failures": invalid,
        }
    return reasons, details


def aggregate_accuracy_statistics(records: list[Mapping[str, Any]], *, required_seeds: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {"statistic_label": "paired_replay_perturbation_statistics", "datasets": {}}
    for dataset in sorted({str(row.get("dataset")) for row in records}):
        subset = sorted((row for row in records if row.get("dataset") == dataset), key=lambda row: int(row.get("seed") or -1))
        seed_map = {int(row["seed"]): row for row in subset if row.get("seed") is not None}
        condition_stats: dict[str, Any] = {}
        for condition, field in (
            ("clean", "clean_accuracy_percent"),
            ("quantized_clean", "quantized_clean_accuracy_percent"),
            ("combined", "combined_accuracy_percent"),
        ):
            values = [float(seed_map[seed][field]) for seed in required_seeds if seed in seed_map and seed_map[seed].get(field) is not None]
            stats = _stats(values)
            condition_stats[condition] = stats
            rows.append({
                "dataset": dataset,
                "condition": condition,
                "statistic_label": "paired_replay_perturbation_statistics",
                "seed_ids": ",".join(str(seed) for seed in required_seeds if seed in seed_map and seed_map[seed].get(field) is not None),
                "n": stats["n"],
                "mean_accuracy_percent": stats["mean"],
                "std_sample_percentage_points": stats["std_sample"],
                "ci95_half_width_percentage_points": stats["ci95_half_width"],
                "min_accuracy_percent": stats["min"],
                "max_accuracy_percent": stats["max"],
                "evidence_class": "software_checkpoint_evaluation" if condition == "clean" else "hardware_aware_software_proxy_replay",
                "claim_boundary": "paired replay perturbation; not independent training seed statistics",
            })
        aggregate["datasets"][dataset] = {
            "seed_records": [dict(seed_map[seed]) for seed in required_seeds if seed in seed_map],
            "conditions": condition_stats,
            "seed_count": len(seed_map),
            "required_seeds": required_seeds,
        }
    return rows, aggregate

def collect_reporting_cases(root: Path, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        eval_dir = root / dataset / "eval_101"
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        selected = _metric_container(_read_json(metrics_path))
        metrics = selected.get("metrics", {})
        lock_csv = _read_csv(eval_dir / "mrr_lock_fraction_sensitivity.csv")
        if lock_csv:
            for row in lock_csv:
                fraction = _as_float(row.get("locked_fraction"))
                rows.append({
                    "dataset": dataset,
                    "case_id": "nominal_lower_bound" if fraction == 0.0 else ("representative_lock_aware" if fraction == 0.01 else "sensitivity"),
                    "lock_fraction": fraction,
                    "lock_fraction_percent": _as_float(row.get("locked_fraction_percent")),
                    "include_lock_power": bool(fraction and fraction > 0.0),
                    "power_w": _as_float(row.get("total_power_w")),
                    "energy_uJ_per_image": _as_float(row.get("energy_uJ_per_image")),
                    "latency_us_per_image": _as_float(row.get("latency_us_per_image")),
                    "energy_overhead_percent": _as_float(row.get("energy_overhead_percent")),
                    "source_file": _rel(eval_dir / "mrr_lock_fraction_sensitivity.csv"),
                    "status": row.get("status"),
                    "evidence_class": "architecture_level_parameterized_sensitivity",
                    "claim_boundary": "lock-power sensitivity anchored to literature parameter; not HIPSA silicon measurement",
                })
        config_csv = _read_csv(eval_dir / "tile_load_sensitivity.csv")
        for row in config_csv:
            rows.append({
                "dataset": dataset,
                "case_id": "configuration_sensitivity",
                "cycles_per_tile_load": _as_int(row.get("cycles_per_tile_load")),
                "tile_load_events_per_image": _as_int(row.get("tile_load_events_per_image")),
                "latency_us_per_image": _as_float(row.get("latency_us_per_image")),
                "nominal_energy_uJ_per_image": _as_float(row.get("nominal_energy_uJ_per_image")),
                "lock_aware_energy_uJ_per_image": _as_float(row.get("lock_aware_energy_uJ_per_image")),
                "configuration_energy_uJ_per_image": _as_float(row.get("configuration_energy_uJ_per_image")),
                "source_file": _rel(eval_dir / "tile_load_sensitivity.csv"),
                "status": row.get("status"),
                "evidence_class": "architecture_parameter_sensitivity",
                "claim_boundary": "cycles/tile is an architecture assumption; programming, stabilization, and calibration are not separately measured",
            })
        rows.append({
            "dataset": dataset,
            "case_id": "selected_nominal_point",
            "lock_fraction": 0.0,
            "include_lock_power": False,
            "latency_us_per_image": _as_float(metrics.get("latency_us_per_image")),
            "energy_uJ_per_image": _as_float(metrics.get("energy_uJ_per_image")),
            "power_w": _as_float(metrics.get("power_w")),
            "area_mm2_component_lower_bound": _as_float(metrics.get("area_mm2_component_lower_bound")),
            "cycles_per_tile_load": _as_int(selected.get("tile_load_sensitivity", {}).get("cycles_per_tile_load", [None])[0]),
            "source_file": _rel(metrics_path),
            "status": "selected_operating_point",
            "evidence_class": "architecture_level_selected_point",
            "claim_boundary": "G4/A8 capacity and system accounting; not circuit timing closure",
        })
    return rows


def build_condition_definitions(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = contract.get("reporting_cases", {})
    rows: list[dict[str, Any]] = []
    for case_id, value in cases.items():
        if case_id == "sensitivity":
            for fraction in value.get("lock_fractions", []):
                rows.append({
                    "case_id": f"lock_{float(fraction):g}",
                    "family": "mrr_lock_fraction",
                    "lock_fraction": float(fraction),
                    "include_lock_power": bool(fraction),
                    "reporting_role": "sensitivity",
                    "evidence_class": "architecture_level_parameterized_sensitivity",
                    "claim_boundary": "literature-anchored parameter sensitivity, not measured lock power",
                })
            continue
        rows.append({
            "case_id": case_id,
            "family": "mrr_lock_fraction",
            "lock_fraction": value.get("lock_fraction"),
            "include_lock_power": value.get("include_lock_power"),
            "reporting_role": case_id,
            "evidence_class": "architecture_level_system_accounting",
            "claim_boundary": "nominal lower bound or parameterized lock-aware accounting",
        })
    rows.extend([
        {"case_id": "paired_replay", "family": "accuracy", "seed_type": "paired_replay_perturbation_seed", "reporting_role": "accuracy aggregate", "claim_boundary": "software checkpoint replay, not independent training statistics"},
        {"case_id": "mapping_only", "family": "representative_workload", "reporting_role": "model coverage", "evidence_class": "mapping_only", "claim_boundary": "no end-to-end hardware latency, energy, throughput, or accuracy claim"},
    ])
    return rows


def collect_ablation_evidence(root: Path, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {"adc_pooling_without_hapr", "hapr_with_64_adcs", "lazy_lif_without_hapr", "hapr_a8_without_lazy_lif", "full_hipsa"}
    for dataset in datasets:
        path = root / dataset / "eval_104" / "orthogonal_ablation.csv"
        for source in _read_csv(path):
            variant = str(source.get("variant", ""))
            row = dict(source)
            row.update({
                "dataset": dataset,
                "replayed": variant == "full_hipsa",
                "accuracy_available": bool(variant == "full_hipsa" and source.get("hardware_aware_accuracy_percent")),
                "accuracy_source": "eval_105 verified paired replay" if variant == "full_hipsa" else "none",
                "counter_source": "eval_101 aggregate counters",
                "evidence_class": "verified_eval_105_replay_plus_architecture_counters" if variant == "full_hipsa" else "architecture_level_counter_estimate",
                "claim_boundary": "only full_hipsa has replay accuracy; other variants are not accuracy-validated",
                "required_variant": variant in required,
            })
            rows.append(row)
    return rows


def collect_configuration_provenance(root: Path, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        eval_dir = root / dataset / "eval_101"
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        selected = _metric_container(_read_json(metrics_path))
        sens = selected.get("tile_load_sensitivity", {})
        source_rows = _read_csv(eval_dir / "tile_load_sensitivity.csv")
        tile_events = sens.get("tile_load_events_per_image")
        if tile_events is None and source_rows:
            tile_events = _as_int(source_rows[0].get("tile_load_events_per_image"))
        row = {
            "dataset": dataset,
            "configuration_protocol": sens.get("configuration_protocol", "layer_weight_tile_boundary"),
            "cycles_per_tile_load": sens.get("cycles_per_tile_load", [256, 1000, 10000, 100000]),
            "tile_load_events_per_image": tile_events,
            "amortize_across_timesteps": sens.get("amortize_across_timesteps", True),
            "cycles_per_tile_source": "architecture assumption from hardware/config model",
            "configuration_energy_source": "eval_101 source table",
            "source_table_keeps_configuration_energy_constant_across_cycle_sensitivity": True,
            "mrr_write_time_included": True,
            "mrr_stabilization_time_included": False,
            "calibration_time_included": False,
            "programming_driver_energy_separately_modeled": False,
            "status": "configuration_assumption_with_explicit_boundary",
            "claim_boundary": "sensitivity is not a measured programming protocol or energy model",
            "source_file": _rel(metrics_path),
        }
        rows.append(row)
    return rows


def collect_mrr_lock_provenance(root: Path, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        path = root / dataset / "eval_101" / "mrr_lock_fraction_sensitivity.csv"
        for source in _read_csv(path):
            row = dict(source)
            fraction = _as_float(source.get("locked_fraction"))
            row.update({
                "dataset": dataset,
                "reporting_case": "nominal_lower_bound" if fraction == 0 else ("representative_lock_aware" if fraction == 0.01 else "sensitivity"),
                "parameter_source": "Zhu et al., Lightening-Transformer, HPCA 2024, Table III",
                "parameter_type": "external_literature_anchor",
                "is_hipsa_measurement": False,
                "controller_modeled": False,
                "thermal_keepout_modeled": False,
                "calibration_overhead_modeled": False,
                "claim_boundary": "parameterized continuous lock-power sensitivity; not measured HIPSA lock power",
                "source_file": _rel(path),
            })
            rows.append(row)
    return rows


def collect_mapping_audit(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = root / "representative_models" / "eval_106"
    mapping_rows = _read_csv(base / "operator_mapping.csv")
    workload_rows = _read_csv(base / "workload_summary.csv")
    summaries = {str(row.get("workload")): row for row in workload_rows}
    audit: list[dict[str, Any]] = []
    for index, source in enumerate(mapping_rows):
        execution = str(source.get("execution", ""))
        optical = "optical" in execution.lower() and "fallback" not in execution.lower()
        workload = str(source.get("workload", ""))
        operator = str(source.get("operator", ""))
        if "transformer" in workload.lower() or "spikformer" in workload.lower():
            role = "qkv_projection" if "qkv" in operator else ("attention_qk" if "qk" in operator else ("attention_av" if "av" in operator else operator))
        else:
            role = "residual_or_elementwise" if "residual" in operator else operator
        unsupported = []
        if not optical:
            unsupported.append("optical_weight_stationary_path")
        audit.append({
            "operator_id": f"{workload}:{index:03d}",
            "workload": workload,
            "operator": operator,
            "operator_role": role,
            "multiplicity": _as_int(source.get("multiplicity")),
            "matrix_k": _as_int(source.get("matrix_k")),
            "matrix_n": _as_int(source.get("matrix_n")),
            "execution_domain": "optical" if optical else "electronic",
            "optical_path": optical,
            "electronic_path": not optical,
            "weight_stationary": optical,
            "dynamic_operand": source.get("dynamic_operand") or ("activation" if not optical else "spike/input"),
            "hapr_eligible": optical,
            "adc_service": "G4/A8 pooled ADC" if optical else "not applicable",
            "state_update": "electronic lazy-LIF/state path" if "residual" in operator or not optical else "downstream state commit",
            "mapping_status": "mapped" if optical or "electronic" in execution.lower() else "unclassified",
            "evidence_level": "mapping_only",
            "unsupported_claims": ",".join(["accuracy", "latency", "energy", "throughput"] + unsupported),
            "source_reference": summaries.get(workload, {}).get("source_url"),
            "notes": "Transformer attention/mask/normalization/residual remain electronic or unsupported unless explicitly represented in source mapping." if "transformer" in workload.lower() or "spikformer" in workload.lower() else "Representative mapping only.",
        })
    evidence = []
    for workload, summary in summaries.items():
        evidence.append({
            "workload": workload,
            "family": summary.get("family"),
            "evidence_level": "mapping_only",
            "operator_instances": summary.get("operator_instances"),
            "optical_operator_coverage_percent": summary.get("optical_operator_coverage_percent"),
            "unsupported_claims": "accuracy,latency,energy,throughput",
            "source_reference": summary.get("source_url"),
            "claim_boundary": "shared-substrate representative mapping; not end-to-end hardware validation",
        })
    return audit, evidence


def collect_area_sensitivity(root: Path, dataset: str = "cifar10dvs") -> list[dict[str, Any]]:
    path = root / dataset / "eval_101" / "area_breakdown.csv"
    source_rows = _read_csv(path)
    total = sum(_as_float(row.get("area_total_mm2")) or 0.0 for row in source_rows)
    rows = []
    for multiplier in (1.0, 1.2, 1.5, 2.0):
        rows.append({
            "dataset": dataset,
            "routing_thermal_overhead_multiplier": multiplier,
            "component_footprint_lower_bound_mm2": total,
            "adjusted_area_mm2": total * multiplier,
            "scope": "component-footprint lower-bound plus parameterized routing/thermal overhead",
            "included_in_source_sum": multiplier == 1.0,
            "evidence_class": "architecture_area_sensitivity",
            "claim_boundary": "not die area, core area, or layout area; no extracted routing/thermal/SRAM macro layout",
            "source_file": _rel(path),
        })
    return rows


def collect_hardware_anchor_provenance(root: Path) -> list[dict[str, Any]]:
    schema_path = REPO_ROOT / "configs" / "external_hardware_anchor_schema.yaml"
    schema = _read_yaml(schema_path) if schema_path.exists() else {}
    return [
        {"anchor_id": "mrr_lock_power", "source": "Zhu et al., Lightening-Transformer, HPCA 2024, Table III", "status": "external_literature_anchor", "measured_for_hipsa": False, "claim_boundary": "parameter anchor only"},
        {"anchor_id": "adc_or_receiver", "source": "device_params.yaml and eval_101 analog audit", "status": "architecture_model_parameter", "measured_for_hipsa": False, "claim_boundary": "architecture-level analog feasibility; no circuit closure"},
        {"anchor_id": "digital_control_area", "source": "NangateOpenCellLibrary 45 nm synthesis anchor", "status": "local_synthesis_pre_layout_anchor", "measured_for_hipsa": False, "claim_boundary": "pre-layout standard-cell area; excludes P&R, pads, SRAM macros"},
        {"anchor_id": "external_hardware_anchor_schema", "source": _rel(schema_path), "status": schema.get("status", "template_only"), "measured_for_hipsa": False, "claim_boundary": "schema/provenance template; no external measurements supplied"},
    ]


def collect_capacity_evidence(root: Path, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        path = root / dataset / "eval_101" / "metrics.json"
        if not path.exists():
            continue
        selected = _metric_container(_read_json(path))
        contract = selected.get("g4_a8_contract", {})
        lanes = _as_float(contract.get("post_hapr_lanes")) or 64
        slots = _as_float(contract.get("nominal_sample_slots_per_cycle")) or 80
        rows.append({
            "dataset": dataset,
            "post_hapr_lanes": lanes,
            "adc_macros": contract.get("adc_macros", 8),
            "samples_per_macro_per_cycle": contract.get("samples_per_macro_per_cycle", 10),
            "nominal_adc_slots_per_cycle": slots,
            "capacity_margin_slots": slots - lanes,
            "capacity_ratio_demand_over_supply": lanes / slots,
            "same_window_admission_feasible": lanes <= slots,
            "evidence_level": "architecture_level_capacity_analysis",
            "circuit_timing_closed": False,
            "mux_settling_included": False,
            "adc_aperture_included": False,
            "tia_recovery_included": False,
            "claim_boundary": "64 post-HAPR lanes <= 80 ADC slots/window; no analog timing closure claim",
            "source_file": _rel(path),
        })
    return rows


def build_publication_tables(root: Path, datasets: Iterable[str], accuracy_rows: list[Mapping[str, Any]], ablation_rows: list[Mapping[str, Any]], mapping_rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    energy: list[dict[str, Any]] = []
    for dataset in datasets:
        selected_path = root / dataset / "eval_101" / "metrics.json"
        if not selected_path.exists():
            continue
        selected = _metric_container(_read_json(selected_path))
        metrics = selected.get("metrics", {})
        energy.append({
            "dataset": dataset,
            "latency_us_per_image": metrics.get("latency_us_per_image"),
            "throughput_images_per_s": metrics.get("throughput_images_per_s"),
            "nominal_power_w": metrics.get("power_w"),
            "nominal_energy_uJ_per_image": metrics.get("energy_uJ_per_image"),
            "lock_aware_1pct_energy_uJ_per_image": next((r.get("energy_uJ_per_image") for r in collect_reporting_cases(root, [dataset]) if r.get("case_id") == "representative_lock_aware"), None),
            "area_mm2_component_lower_bound": metrics.get("area_mm2_component_lower_bound"),
            "evidence_class": "architecture_level_system_accounting",
        })
    mapping = [dict(row) for row in mapping_rows]
    return {"publication_table_accuracy": [dict(row) for row in accuracy_rows], "publication_table_energy": energy, "publication_table_ablation": [dict(row) for row in ablation_rows], "publication_table_mapping": mapping}
