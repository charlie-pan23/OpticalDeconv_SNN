"""Small provenance and G4/A8 contract helpers for the official eval-v10 chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_g4_a8_contract(hardware_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    num_tiles = int(_get(hardware_cfg, "photonic_tiles", "num_tiles", default=0))
    rows = int(_get(hardware_cfg, "mapping", "array_rows", default=0))
    cols = int(_get(hardware_cfg, "mapping", "array_cols", default=0))
    group = int(_get(hardware_cfg, "hapr_adc_backend", "hapr_group_size", default=0))
    lanes = int(_get(hardware_cfg, "hapr_adc_backend", "hapr_output_lanes_total", default=0))
    adc_macros = int(_get(hardware_cfg, "hapr_adc_backend", "adc_macros", default=0))
    slots_per_macro = float(_get(hardware_cfg, "hapr_adc_backend", "adc_samples_per_macro_per_architecture_cycle", default=0.0))
    clock_hz = float(_get(hardware_cfg, "transaction_scheduler", "system_clock_hz", default=0.0))
    temporal_storage = bool(_get(hardware_cfg, "hapr_adc_backend", "temporal_analog_storage", default=True))
    derived_lanes = (num_tiles // max(group, 1)) * cols if group > 0 else 0
    nominal_slots = adc_macros * slots_per_macro
    checks = {
        "four_tiles": num_tiles == 4,
        "logical_array_64x64": rows == 64 and cols == 64,
        "g4": group == 4,
        "reported_lanes_match_derived": lanes == derived_lanes == 64,
        "a8": adc_macros == 8,
        "ten_samples_per_macro_per_cycle": abs(slots_per_macro - 10.0) < 1e-12,
        "one_ghz_architecture_clock": abs(clock_hz - 1.0e9) < 1.0,
        "same_window_capacity": nominal_slots >= lanes,
        "no_temporal_analog_storage": not temporal_storage,
    }
    failed = [name for name, passed in checks.items() if not passed]
    contract = {
        "name": "HIPSA_G4_A8",
        "num_tiles": num_tiles,
        "array_rows": rows,
        "array_cols": cols,
        "hapr_group_size": group,
        "post_hapr_lanes": lanes,
        "derived_post_hapr_lanes": derived_lanes,
        "adc_macros": adc_macros,
        "samples_per_macro_per_cycle": slots_per_macro,
        "nominal_sample_slots_per_cycle": nominal_slots,
        "system_clock_hz": clock_hz,
        "temporal_analog_storage": temporal_storage,
        "checks": checks,
        "valid": not failed,
        "failed_checks": failed,
        "claim_boundary": "architecture_level_capacity_analysis_not_circuit_timing_closure",
    }
    if failed:
        raise ValueError("Official eval-v10 requires the G4/A8 contract; failed: " + ", ".join(failed))
    return contract


def provenance_hashes(**paths: str | Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for name, raw in paths.items():
        path = Path(raw)
        result[f"{name}_sha256"] = file_sha256(path)
    return result

def validate_eval105_publication_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate that an eval_105 artifact is safe to join into paper-facing results."""
    reasons: list[str] = []
    split = summary.get("split_manifest", {})
    calibration = summary.get("adc_calibration", {})
    scope = summary.get("replay_scope", {})
    precision = summary.get("precision_contract", {})
    runtime = summary.get("runtime_provenance", {})

    if summary.get("eval_name") != "eval_105":
        reasons.append("wrong_eval_name")
    replayed = int(summary.get("num_samples") or 0)
    expected = int(split.get("num_samples_expected") or 0)
    if replayed <= 0:
        reasons.append("no_replayed_samples")
    if split.get("status") != "verified_full_split_replay":
        reasons.append("full_reporting_split_not_verified")
    if split.get("full_split_replayed") is not True or expected <= 0 or replayed != expected:
        reasons.append("reporting_split_incomplete")
    if split.get("max_batches") is not None:
        reasons.append("diagnostic_replay_cap_present")

    if calibration.get("test_data_used_for_calibration") is not False:
        reasons.append("test_data_calibration_not_excluded")
    if calibration.get("calibration_split") == calibration.get("reporting_split"):
        reasons.append("calibration_reporting_split_not_isolated")
    if calibration.get("status") != "verified_full_calibration_split":
        reasons.append("full_calibration_split_not_verified")
    if calibration.get("full_calibration_split_used") is not True:
        reasons.append("calibration_split_incomplete")
    if calibration.get("calibration_batches_requested") is not None:
        reasons.append("diagnostic_calibration_cap_present")

    if runtime.get("publication_replay_ready") is not True:
        reasons.append("runtime_not_publication_ready")
    if runtime.get("deterministic_algorithms_enabled") is not True:
        reasons.append("deterministic_algorithms_not_enabled")
    if runtime.get("cudnn_deterministic") is not True or runtime.get("cudnn_benchmark") is not False:
        reasons.append("cudnn_determinism_not_fixed")
    if runtime.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
        reasons.append("cublas_workspace_config_not_fixed")
    if runtime.get("full_replay_requested") is not True:
        reasons.append("runtime_reports_diagnostic_scope")

    if summary.get("combined_condition_accuracy_percent") is None:
        reasons.append("combined_accuracy_missing")
    if precision != {"weight_bits": 6, "adc_bits": 6, "membrane_bits": 16}:
        reasons.append("precision_contract_not_w6_adc6_vm16")

    target_names = list(scope.get("photonic_mvm_layer_names") or [])
    calibrated_names = list((calibration.get("full_scales") or {}).keys())
    if scope.get("status") != "verified_photonic_only":
        reasons.append("photonic_scope_not_verified")
    if scope.get("include_final_classifier") is not False:
        reasons.append("final_classifier_inclusion_not_disabled")
    if scope.get("final_classifier_excluded") is not True or "fc2" in target_names:
        reasons.append("electronic_final_classifier_not_excluded")
    if not target_names:
        reasons.append("photonic_layer_list_missing")
    if target_names != calibrated_names:
        reasons.append("adc_calibration_scope_mismatch")

    return {
        "valid": not reasons,
        "status": "verified_eval_105_publication_replay" if not reasons else "invalid_or_incomplete_eval_105",
        "reasons": reasons,
        "photonic_mvm_layer_names": target_names,
        "num_samples_replayed": replayed,
        "num_samples_expected": expected,
        "runtime_provenance": runtime,
    }
