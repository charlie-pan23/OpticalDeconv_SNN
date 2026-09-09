"""Phase 3 system-evidence aggregation and audit helpers.

This module intentionally stays at the architecture-level evidence boundary.
It joins existing eval_101/eval_104 artifacts with seed-indexed eval_105
software replay results, and never upgrades replay or analytical estimates to
silicon measurements.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.result_io import load_csv_rows, load_json, load_yaml

REQUIRED_VARIANTS = {
    "adc_pooling_without_hapr",
    "hapr_with_64_adcs",
    "lazy_lif_without_hapr",
    "hapr_a8_without_lazy_lif",
    "full_hipsa",
}
REQUIRED_LOCK_FRACTIONS = {"0", "0.0", "0.01", "0.05", "0.1", "0.10"}
REQUIRED_TILE_LOAD_CYCLES = {"256", "1000", "10000", "100000"}
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _normalised_values(values: Iterable[Any]) -> set[str]:
    result: set[str] = set()
    for value in values:
        number = _as_float(value)
        result.add(str(number).rstrip("0").rstrip(".") if number is not None else str(value))
    return result


def validate_phase3_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    hardware = dict(contract.get("hardware", {}))
    precision = dict(contract.get("precision", {}))
    sensitivity = dict(contract.get("sensitivity", {}))
    accuracy = dict(contract.get("accuracy", {}))
    shape = hardware.get("array_shape")
    checks = {
        "phase_is_3": contract.get("phase") == 3,
        "architecture_evidence_scope": contract.get("evidence_scope") == "architecture_level_system_accounting",
        "four_photonic_tiles": hardware.get("photonic_tiles") == 4,
        "array_is_64x64": shape == [64, 64] or shape == (64, 64),
        "g4": hardware.get("hapr_group") == 4,
        "post_hapr_lanes_64": hardware.get("post_hapr_lanes") == 64,
        "a8": hardware.get("adc_macros") == 8,
        "ten_samples_per_macro": hardware.get("adc_samples_per_macro_per_cycle") == 10,
        "one_ghz_clock": hardware.get("architecture_clock_hz") == 1000000000,
        "no_temporal_analog_storage": hardware.get("temporal_analog_storage") is False,
        "w6_adc6_vm16": precision == {"weight_bits": 6, "adc_bits": 6, "membrane_bits": 16},
        "lock_sensitivity_complete": _normalised_values(sensitivity.get("lock_fractions", [])) >= {"0", "0.01", "0.05", "0.1"},
        "configuration_sensitivity_complete": _normalised_values(sensitivity.get("tile_load_cycles", [])) >= REQUIRED_TILE_LOAD_CYCLES,
        "minimum_replay_seeds_at_least_5": int(accuracy.get("minimum_replay_seeds", 0)) >= 5,
        "training_seed_claim_disabled": accuracy.get("independent_training_seed_claim_allowed") is False,
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    return {"valid": not reasons, "checks": checks, "reasons": reasons, "evidence_scope": contract.get("evidence_scope")}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    return load_csv_rows(path, parse_numbers=True) if path.exists() else []


def audit_eval101(eval_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    required = ["summary.json", "metrics.json", "validation.json", "tile_load_sensitivity.csv", "mrr_lock_fraction_sensitivity.csv"]
    missing = [name for name in required if not (eval_dir / name).exists()]
    reasons.extend(f"missing:{name}" for name in missing)
    tile_rows = _read_csv(eval_dir / "tile_load_sensitivity.csv")
    lock_rows = _read_csv(eval_dir / "mrr_lock_fraction_sensitivity.csv")
    tile_field = "cycles_per_tile_load"
    lock_field = "locked_fraction"
    observed_tiles = _normalised_values(row.get(tile_field) for row in tile_rows)
    observed_locks = _normalised_values(row.get(lock_field) for row in lock_rows)
    expected_tiles = _normalised_values(contract.get("sensitivity", {}).get("tile_load_cycles", []))
    expected_locks = _normalised_values(contract.get("sensitivity", {}).get("lock_fractions", []))
    missing_tiles = sorted(expected_tiles - observed_tiles)
    missing_locks = sorted(expected_locks - observed_locks)
    if missing_tiles:
        reasons.append("tile_load_sensitivity_missing:" + ",".join(missing_tiles))
    if missing_locks:
        reasons.append("lock_sensitivity_missing:" + ",".join(missing_locks))
    return {
        "path": str(eval_dir),
        "passed": not reasons,
        "reasons": reasons,
        "tile_load_rows": len(tile_rows),
        "lock_rows": len(lock_rows),
        "observed_tile_load_cycles": sorted(observed_tiles),
        "observed_lock_fractions": sorted(observed_locks),
    }


def audit_eval104(eval_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    required = ["summary.json", "orthogonal_ablation.csv", "matched_accounting_contract.json", "baseline_definition.json"]
    reasons.extend(f"missing:{name}" for name in required if not (eval_dir / name).exists())
    rows = _read_csv(eval_dir / "orthogonal_ablation.csv")
    observed = {str(row.get("variant", "")) for row in rows}
    expected = set(contract.get("orthogonal_ablation", {}).get("required_variants", [])) or REQUIRED_VARIANTS
    missing = sorted(expected - observed)
    if missing:
        reasons.append("orthogonal_variants_missing:" + ",".join(missing))
    matched = load_json(eval_dir / "matched_accounting_contract.json") if (eval_dir / "matched_accounting_contract.json").exists() else {}
    scope = contract.get("matched_scope", {})
    contract_checks = {
        "same_four_photonic_tiles": matched.get("same_four_photonic_tiles") == scope.get("same_four_photonic_tiles"),
        "same_array_shape": matched.get("same_array_shape") == scope.get("same_array_shape"),
        "same_precision": matched.get("same_precision") == scope.get("same_precision"),
        "same_input_activity_trace": matched.get("same_input_activity_trace") == scope.get("same_input_activity_trace"),
        "same_static_power_terms": matched.get("same_static_power_terms") == scope.get("same_static_power_terms"),
        "same_non_mvm_accounting": matched.get("same_non_mvm_accounting") == scope.get("same_non_mvm_accounting"),
        "same_configuration_protocol": matched.get("same_configuration_protocol") == "layer_weight_tile_boundary; amortized across timesteps",
    }
    reasons.extend("matched_contract_mismatch:" + name for name, passed in contract_checks.items() if not passed)
    return {"path": str(eval_dir), "passed": not reasons, "reasons": reasons, "variant_count": len(observed), "variants": sorted(observed), "contract_checks": contract_checks}


def discover_eval105_summaries(eval_dir: Path) -> list[dict[str, Any]]:
    """Load legacy seed-0 and seed-indexed replay summaries together.

    A seed-indexed run does not replace the legacy seed-0 result: the latter is
    the first member of the paired replay set when it was generated from the
    same frozen checkpoint and contract. Keep both, while preventing a future
    duplicate seed-0 path from being counted twice.
    """
    seed_root = eval_dir / "seeds"
    indexed_paths = sorted(seed_root.glob("seed_*/summary.json")) if seed_root.exists() else []
    legacy_path = eval_dir / "summary.json"
    paths = ([legacy_path] if legacy_path.exists() else []) + indexed_paths
    seen_seeds: set[int] = set()
    summaries: list[dict[str, Any]] = []
    for path in paths:
        summary = load_json(path)
        seed_value = summary.get("seed")
        try:
            seed_key = int(seed_value) if seed_value is not None else None
        except (TypeError, ValueError):
            seed_key = None
        if seed_key is not None and seed_key in seen_seeds:
            continue
        if seed_key is not None:
            seen_seeds.add(seed_key)
        summary["_source_path"] = str(path)
        summary["_seed_indexed"] = path.parent.parent.name == "seeds"
        # Seed-0 legacy artifact predates explicit Phase 3 metadata.
        summary.setdefault("seed_type", "paired_replay_perturbation_seed")
        runtime = dict(summary.get("runtime_provenance", {}))
        runtime.setdefault("seed_type", "paired_replay_perturbation_seed")
        summary["runtime_provenance"] = runtime
        summaries.append(summary)
    return summaries
def _stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "ci95_half_width": None}
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    t = T_CRITICAL_95.get(n - 1, 1.96) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "min": min(values), "max": max(values), "ci95_half_width": t * std / math.sqrt(n) if n > 1 else None}


def aggregate_eval105_summaries(summaries: list[Mapping[str, Any]], *, minimum_seeds: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_condition: dict[str, list[float]] = {}
    by_delta: dict[str, list[float]] = {}
    seed_types = set()
    for summary in summaries:
        runtime = summary.get("runtime_provenance", {})
        seed_types.add(runtime.get("seed_type", summary.get("seed_type", "unspecified")))
        for row in summary.get("conditions", []):
            condition = str(row.get("condition", ""))
            value = _as_float(row.get("accuracy_percent"))
            if condition and value is not None:
                by_condition.setdefault(condition, []).append(value)
            delta = _as_float(row.get("paired_accuracy_delta_percent_vs_clean"))
            if condition and delta is not None:
                by_delta.setdefault(condition, []).append(delta)
    rows: list[dict[str, Any]] = []
    for condition in sorted(by_condition):
        stats = _stats(by_condition[condition])
        delta_stats = _stats(by_delta.get(condition, []))
        rows.append({
            "condition": condition,
            "seed_count": stats["n"],
            "accuracy_mean_percent": stats["mean"],
            "accuracy_std_percent": stats["std"],
            "accuracy_min_percent": stats["min"],
            "accuracy_max_percent": stats["max"],
            "accuracy_ci95_half_width_percent": stats["ci95_half_width"],
            "paired_delta_mean_percent_vs_clean": delta_stats["mean"],
            "paired_delta_std_percent_vs_clean": delta_stats["std"],
            "paired_delta_ci95_half_width_percent": delta_stats["ci95_half_width"],
            "statistics_status": "complete" if stats["n"] >= minimum_seeds else "insufficient_replay_seeds",
            "evidence_class": "paired_replay_perturbation_statistics",
        })
    aggregation = {
        "seed_count": len(summaries),
        "minimum_seeds": minimum_seeds,
        "sufficient_seed_count": len(summaries) >= minimum_seeds,
        "seed_type": sorted(seed_types) if seed_types else ["unspecified"],
        "seed_scope": "paired_replay_perturbation_seed_not_independent_training_seed",
        "claim_boundary": "mean/std/CI summarize frozen-checkpoint replay perturbations; they are not independent-training statistics",
    }
    return rows, aggregation


def collect_sensitivity_rows(root: Path, datasets: Iterable[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lock_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        eval_dir = root / dataset / "eval_101"
        for row in _read_csv(eval_dir / "mrr_lock_fraction_sensitivity.csv"):
            lock_rows.append({"dataset": dataset, **row, "evidence_class": "architecture_level_parameterized_lock_power_sensitivity"})
        for row in _read_csv(eval_dir / "tile_load_sensitivity.csv"):
            config_rows.append({"dataset": dataset, **row, "evidence_class": "architecture_level_parameterized_configuration_sensitivity"})
    return lock_rows, config_rows


def collect_ablation_rows(root: Path, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for row in _read_csv(root / dataset / "eval_104" / "orthogonal_ablation.csv"):
            rows.append({"dataset": dataset, **row, "evidence_class": "architecture_level_orthogonal_ablation"})
    return rows
