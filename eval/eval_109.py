"""Official Phase 4 evidence-closure entry point.

The entry point is an audit/aggregation stage: it reads existing eval_101,
eval_104, eval_105, and eval_106 artifacts and never reruns model evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.util.official_schema import repo_path, write_standard_artifacts
from eval.util.phase4 import (
    _read_yaml,
    _validate_seed_records,
    aggregate_accuracy_statistics,
    build_condition_definitions,
    build_publication_tables,
    collect_ablation_evidence,
    collect_area_sensitivity,
    collect_capacity_evidence,
    collect_configuration_provenance,
    collect_hardware_anchor_provenance,
    collect_mapping_audit,
    collect_mrr_lock_provenance,
    collect_reporting_cases,
    collect_seed_records,
    validate_phase4_contract,
)
from utils.result_io import save_csv_rows, save_json, save_run_manifest, save_yaml


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    save_csv_rows(rows, path)


def _required_variant_check(rows: list[dict[str, Any]], required: list[str]) -> dict[str, Any]:
    observed = sorted({str(row.get("variant", "")) for row in rows})
    missing = sorted(set(required) - set(observed))
    return {"required": required, "observed": observed, "missing": missing, "passed": not missing}


def _seed_validation(records: list[dict[str, Any]], required_seeds: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    reasons, details = _validate_seed_records(records, required_seeds=required_seeds)
    return {"passed": not reasons, "reasons": reasons, "datasets": details}, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/eval_v10")
    parser.add_argument("--contract", default="configs/phase4_evidence_contract.yaml")
    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--allow-incomplete", action="store_true", help="write artifacts but return success despite validation failure")
    args = parser.parse_args(argv)

    root = repo_path(args.output_root, "results/eval_v10")
    contract_path = repo_path(args.contract, "configs/phase4_evidence_contract.yaml")
    contract = _read_yaml(contract_path)
    contract_validation = validate_phase4_contract(contract)
    required_seeds = [int(seed) for seed in contract.get("seed_provenance", {}).get("required_seeds", [0, 1, 2, 3, 4])]

    output_dir = root / "combined" / "eval_109"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_records = collect_seed_records(root, args.datasets)
    seed_validation, seed_details = _seed_validation(seed_records, required_seeds)
    accuracy_rows, accuracy_aggregate = aggregate_accuracy_statistics(seed_records, required_seeds=required_seeds)
    reporting_cases = collect_reporting_cases(root, args.datasets)
    condition_rows = build_condition_definitions(contract)
    ablation_rows = collect_ablation_evidence(root, args.datasets)
    required_variants = [str(value) for value in contract.get("orthogonal_ablation", {}).get("required_variants", [])]
    ablation_validation = _required_variant_check(ablation_rows, required_variants)
    configuration_rows = collect_configuration_provenance(root, args.datasets)
    lock_rows = collect_mrr_lock_provenance(root, args.datasets)
    mapping_rows, mapping_evidence = collect_mapping_audit(root)
    area_rows = collect_area_sensitivity(root, args.datasets[0] if args.datasets else "cifar10dvs")
    anchor_rows = collect_hardware_anchor_provenance(root)
    capacity_rows = collect_capacity_evidence(root, args.datasets)
    tables = build_publication_tables(root, args.datasets, accuracy_rows, ablation_rows, mapping_rows)

    checks = {
        "contract_valid": contract_validation["passed"],
        "seed_provenance_valid": seed_validation["passed"],
        "required_orthogonal_variants_present": ablation_validation["passed"],
        "reporting_cases_present": bool(reporting_cases),
        "condition_definitions_present": bool(condition_rows),
        "configuration_provenance_present": bool(configuration_rows),
        "mrr_lock_provenance_present": bool(lock_rows),
        "mapping_audit_present": bool(mapping_rows),
        "area_sensitivity_present": bool(area_rows),
        "hardware_anchor_provenance_present": bool(anchor_rows),
        "capacity_evidence_present": bool(capacity_rows),
        "accuracy_aggregate_present": bool(accuracy_rows),
        "mapping_is_mapping_only": all(row.get("evidence_level") == "mapping_only" for row in mapping_evidence),
        "non_full_hipsa_accuracy_not_inferred": all(
            not bool(row.get("variant") != "full_hipsa" and row.get("accuracy_available")) for row in ablation_rows
        ),
    }
    reasons: list[str] = []
    reasons.extend("contract:" + str(value) for value in contract_validation.get("reasons", []))
    reasons.extend("seed:" + str(value) for value in seed_validation.get("reasons", []))
    if not ablation_validation["passed"]:
        reasons.append("ablation:missing_variants=" + ",".join(ablation_validation["missing"]))
    for key, passed in checks.items():
        if not passed and not key.startswith("contract") and not key.startswith("seed"):
            reasons.append("check_failed:" + key)
    passed = not reasons

    validation = {
        "passed": passed,
        "status": "phase4_evidence_contract_passed" if passed else "phase4_evidence_contract_failed",
        "phase": 4,
        "stage": "A-D",
        "checks": checks,
        "reasons": reasons,
        "contract_validation": contract_validation,
        "seed_validation": seed_validation,
        "ablation_validation": ablation_validation,
        "claim_boundaries": {
            "adc": "architecture-level capacity analysis only; no mux settling, aperture, TIA recovery, or circuit timing closure",
            "accuracy": "paired replay perturbation statistics from frozen checkpoint; not independent training seed statistics",
            "mapping": "representative mapping only; no end-to-end Transformer/ResNet hardware validation",
            "area": "component-footprint lower bound with parameterized overhead; not die/core/layout area",
            "lock_power": "literature-anchored parameterized sensitivity; not HIPSA silicon measurement",
            "configuration": "cycles/tile is an architecture assumption; stabilization, calibration, and driver energy are not separately modeled",
        },
    }

    metrics = {
        "eval_name": "eval_109",
        "purpose": "phase4_evidence_closure_and_publication_audit",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "datasets": list(args.datasets),
        "evidence_scope": contract.get("evidence_scope"),
        "passed": passed,
        "seed_count_by_dataset": {dataset: seed_details.get(dataset, {}).get("observed_seeds", []) for dataset in args.datasets},
        "accuracy_statistics_label": "paired_replay_perturbation_statistics",
        "reporting_case_count": len(reporting_cases),
        "ablation_row_count": len(ablation_rows),
        "mapping_operator_count": len(mapping_rows),
        "capacity_row_count": len(capacity_rows),
        "area_sensitivity_row_count": len(area_rows),
        "output_dir": str(output_dir),
    }

    # Phase 4-A evidence-contract artifacts.
    _write_rows(reporting_cases, output_dir / "reporting_case_matrix.csv")
    save_json({"rows": reporting_cases}, output_dir / "reporting_case_matrix.json")
    _write_rows(seed_records, output_dir / "seed_records.csv")
    save_json(accuracy_aggregate, output_dir / "seed_aggregate.json")
    _write_rows(accuracy_rows, output_dir / "accuracy_statistics.csv")
    save_json({"statistic_label": "paired_replay_perturbation_statistics", "rows": accuracy_rows}, output_dir / "accuracy_statistics.json")
    save_yaml({"conditions": condition_rows}, output_dir / "condition_definitions.yaml")
    _write_rows(ablation_rows, output_dir / "orthogonal_ablation_evidence.csv")
    _write_rows(configuration_rows, output_dir / "configuration_provenance.csv")
    save_json({"rows": configuration_rows}, output_dir / "configuration_provenance.json")
    _write_rows(lock_rows, output_dir / "mrr_lock_provenance.csv")
    save_json({"rows": lock_rows}, output_dir / "mrr_lock_provenance.json")

    # Phase 4-B/C/D artifacts are produced from the same source records.
    _write_rows(mapping_rows, output_dir / "operator_mapping_audit.csv")
    save_json({"rows": mapping_rows}, output_dir / "operator_mapping_audit.json")
    _write_rows(mapping_evidence, output_dir / "evidence_grading.csv")
    claim_rows = []
    for row in mapping_evidence:
        for claim in ("accuracy", "latency", "energy", "throughput"):
            claim_rows.append({"workload": row.get("workload"), "claim": claim, "supported": False, "evidence_level": row.get("evidence_level"), "reason": row.get("claim_boundary")})
    _write_rows(claim_rows, output_dir / "claim_to_evidence.csv")
    _write_rows(area_rows, output_dir / "area_overhead_sensitivity.csv")
    save_json({"rows": area_rows}, output_dir / "area_overhead_sensitivity.json")
    _write_rows(anchor_rows, output_dir / "hardware_anchor_provenance.csv")
    _write_rows(capacity_rows, output_dir / "g4_a8_capacity_evidence.csv")
    save_json({"rows": capacity_rows}, output_dir / "g4_a8_capacity_evidence.json")
    save_json({"rows": configuration_rows, "warning": "configuration_energy_constant_across_cycle_sensitivity_is_source_table_behavior"}, output_dir / "configuration_amortization_audit.json")
    for name, rows in tables.items():
        _write_rows(rows, output_dir / f"{name}.csv")
    save_json({"figures": [], "status": "plot_eval_109_generates_figures_after_audit"}, output_dir / "publication_figures_manifest.json")
    save_json(validation, output_dir / "phase4a_validation.json")

    config_paths = {
        "phase4_contract": contract_path,
        "hardware": repo_path("configs/hardware_hipsa_paper.yaml", "configs/hardware_hipsa_paper.yaml"),
        "device_params": repo_path("configs/device_params.yaml", "configs/device_params.yaml"),
        "external_anchor_schema": repo_path("configs/external_hardware_anchor_schema.yaml", "configs/external_hardware_anchor_schema.yaml"),
    }
    write_standard_artifacts(
        output_dir,
        eval_name="eval_109",
        metrics=metrics,
        validation=validation,
        config_paths=config_paths,
        evidence_class="phase4_evidence_closure",
        extra_provenance={
            "source_eval_stages": ["eval_101", "eval_104", "eval_105", "eval_106"],
            "source_root": str(root),
            "python_version": platform.python_version(),
            "seed_records": len(seed_records),
        },
    )
    save_run_manifest(
        output_dir,
        eval_name="eval_109",
        command=" ".join(sys.argv),
        inputs={"output_root": str(root), "contract": str(contract_path), "datasets": list(args.datasets)},
        outputs={"summary": "combined/eval_109", "validation": "phase4a_validation.json"},
        extra={"audit_only": True, "legacy_eval_scripts_unchanged": True},
    )

    print(json.dumps({"output_dir": str(output_dir), "passed": passed, "reasons": reasons}, indent=2, ensure_ascii=False))
    if not passed and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
