"""Phase 3 system-evidence summary for DATE-facing HIPSA results."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.util.official_schema import repo_path, write_standard_artifacts
from eval.util.phase3 import (
    aggregate_eval105_summaries,
    audit_eval101,
    audit_eval104,
    collect_ablation_rows,
    collect_sensitivity_rows,
    discover_eval105_summaries,
    validate_phase3_contract,
)
from utils.result_io import load_yaml, save_csv_rows, save_json, save_run_manifest, save_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/eval_v10")
    parser.add_argument("--contract", default="configs/phase3_experiment_contract.yaml")
    parser.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    parser.add_argument("--minimum-seeds", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.minimum_seeds is not None and args.minimum_seeds <= 0:
        raise ValueError("--minimum-seeds must be positive")

    root = repo_path(args.output_root, "results/eval_v10")
    contract_path = repo_path(args.contract, "configs/phase3_experiment_contract.yaml")
    contract = load_yaml(contract_path)
    contract_validation = validate_phase3_contract(contract)
    minimum_seeds = int(args.minimum_seeds or contract.get("accuracy", {}).get("minimum_replay_seeds", 5))

    eval101_audits = {dataset: audit_eval101(root / dataset / "eval_101", contract) for dataset in args.datasets}
    eval104_audits = {dataset: audit_eval104(root / dataset / "eval_104", contract) for dataset in args.datasets}
    accuracy_rows: list[dict[str, Any]] = []
    accuracy_aggregation: dict[str, Any] = {}
    for dataset in args.datasets:
        summaries = discover_eval105_summaries(root / dataset / "eval_105")
        rows, aggregation = aggregate_eval105_summaries(summaries, minimum_seeds=minimum_seeds)
        accuracy_aggregation[dataset] = {
            **aggregation,
            "summary_paths": [summary.get("_source_path") for summary in summaries],
            "seed_indexed_artifacts": sum(bool(summary.get("_seed_indexed")) for summary in summaries),
        }
        accuracy_rows.extend({"dataset": dataset, **row} for row in rows)

    lock_rows, config_rows = collect_sensitivity_rows(root, args.datasets)
    ablation_rows = collect_ablation_rows(root, args.datasets)
    expected_variants = set(contract.get("orthogonal_ablation", {}).get("required_variants", []))
    observed_variants = {str(row.get("variant", "")) for row in ablation_rows}
    checks = {
        "contract_valid": contract_validation["valid"],
        "all_eval101_sensitivity_audits_pass": all(row["passed"] for row in eval101_audits.values()),
        "all_eval104_matched_baseline_audits_pass": all(row["passed"] for row in eval104_audits.values()),
        "lock_rows_present": bool(lock_rows),
        "configuration_rows_present": bool(config_rows),
        "all_required_orthogonal_variants_present": expected_variants <= observed_variants,
        "all_datasets_have_minimum_replay_seeds": all(
            bool(accuracy_aggregation[dataset]["sufficient_seed_count"]) for dataset in args.datasets
        ),
        "replay_statistics_are_not_training_seed_claims": all(
            "training" not in " ".join(accuracy_aggregation[dataset]["seed_type"]).lower()
            for dataset in args.datasets
        ),
    }
    reasons: list[str] = []
    reasons.extend("contract:" + reason for reason in contract_validation["reasons"])
    for dataset, audit in eval101_audits.items():
        reasons.extend(f"{dataset}:eval_101:{reason}" for reason in audit["reasons"])
    for dataset, audit in eval104_audits.items():
        reasons.extend(f"{dataset}:eval_104:{reason}" for reason in audit["reasons"])
    if not lock_rows:
        reasons.append("no_lock_sensitivity_rows")
    if not config_rows:
        reasons.append("no_configuration_sensitivity_rows")
    missing_variants = sorted(expected_variants - observed_variants)
    if missing_variants:
        reasons.append("missing_orthogonal_variants:" + ",".join(missing_variants))
    for dataset in args.datasets:
        if not accuracy_aggregation[dataset]["sufficient_seed_count"]:
            reasons.append(
                f"{dataset}:replay_seed_count:{accuracy_aggregation[dataset]['seed_count']}<minimum:{minimum_seeds}"
            )

    metrics = {
        "eval_name": "eval_108",
        "purpose": "phase3_system_evidence_summary",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "datasets": list(args.datasets),
        "contract": contract,
        "contract_validation": contract_validation,
        "eval101_audits": eval101_audits,
        "eval104_audits": eval104_audits,
        "accuracy_aggregation": accuracy_aggregation,
        "claim_boundary": (
            "architecture-level system accounting plus frozen-checkpoint paired replay perturbation statistics; "
            "not silicon measurement, P&R validation, or independent-training statistics"
        ),
    }
    validation = {
        "eval_name": "eval_108",
        "passed": not reasons,
        "release_status": "GO" if not reasons else "INCOMPLETE",
        "checks": checks,
        "reasons": reasons,
        "minimum_replay_seeds": minimum_seeds,
        "evidence_scope": contract.get("evidence_scope"),
        "claim_boundary": metrics["claim_boundary"],
    }

    out = root / "combined" / "eval_108"
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(contract, out / "phase3_contract_snapshot.yaml")
    save_json(metrics, out / "phase3_report.json")
    save_json(validation, out / "phase3_validation.json")
    save_json(metrics, out / "metrics.json")
    save_json(validation, out / "validation.json")
    save_csv_rows(accuracy_rows, out / "accuracy_statistics.csv")
    save_csv_rows(lock_rows, out / "lock_aware_energy_summary.csv")
    save_csv_rows(config_rows, out / "configuration_sensitivity_summary.csv")
    save_csv_rows(ablation_rows, out / "orthogonal_ablation_paper_table.csv")
    write_standard_artifacts(
        out,
        eval_name="eval_108",
        metrics=metrics,
        validation=validation,
        config_paths={
            "phase3_contract": contract_path,
            "hardware": repo_path("configs/hardware_hipsa_paper.yaml", "configs/hardware_hipsa_paper.yaml"),
            "device_params": repo_path("configs/device_params.yaml", "configs/device_params.yaml"),
        },
        evidence_class="architecture_level_phase3_system_evidence_summary",
        extra_provenance={
            "contract_path": str(contract_path),
            "datasets": list(args.datasets),
            "minimum_replay_seeds": minimum_seeds,
            "seed_scope": "paired_replay_perturbation_seed",
        },
    )
    save_run_manifest(
        out,
        eval_name="eval_108",
        command=" ".join(sys.argv),
        inputs={"output_root": args.output_root, "contract": args.contract, "datasets": args.datasets},
        outputs={
            "phase3_report": "phase3_report.json",
            "phase3_validation": "phase3_validation.json",
            "accuracy_statistics": "accuracy_statistics.csv",
            "lock_aware_energy_summary": "lock_aware_energy_summary.csv",
            "configuration_sensitivity_summary": "configuration_sensitivity_summary.csv",
            "orthogonal_ablation_paper_table": "orthogonal_ablation_paper_table.csv",
            "phase3_contract_snapshot": "phase3_contract_snapshot.yaml",
        },
        extra={"checks": checks, "reasons": reasons},
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0 if validation["passed"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
