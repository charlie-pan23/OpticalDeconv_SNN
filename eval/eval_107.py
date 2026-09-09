"""Official publication release gate for HIPSA eval_v10 artifacts."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_v10 import validate_eval105_publication_summary
from utils.result_io import load_json, save_json, save_run_manifest


def _repo_path(value: str) -> Path:
    raw = Path(value)
    return raw if raw.is_absolute() else REPO_ROOT / raw


def _check_file(path: Path, label: str, reasons: list[str]) -> bool:
    if not path.exists():
        reasons.append(f"missing:{label}:{path.as_posix()}")
        return False
    return True


def _check_dir_artifacts(path: Path, label: str, names: list[str], reasons: list[str]) -> None:
    if not _check_file(path, f"{label}_directory", reasons):
        return
    for name in names:
        _check_file(path / name, f"{label}/{name}", reasons)


def _check_sensitivity_csv(
    path: Path,
    *,
    label: str,
    field: str,
    required_values: set[str],
    reasons: list[str],
) -> None:
    if not _check_file(path, label, reasons):
        return
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        observed = {str(row.get(field, "")) for row in rows}
    except Exception as exc:
        reasons.append(f"invalid:{label}:{type(exc).__name__}:{exc}")
        return
    missing = sorted(required_values - observed)
    if missing:
        reasons.append(f"{label}:missing_values:{','.join(missing)}")


def _load_validation(path: Path, label: str, reasons: list[str]) -> dict[str, Any] | None:
    try:
        value = load_json(path)
    except Exception as exc:
        reasons.append(f"invalid:{label}:{type(exc).__name__}:{exc}")
        return None
    if value.get("passed") is False:
        reasons.append(f"{label}:reported_failed")
    return value


def validate_dataset(root: Path, dataset: str) -> dict[str, Any]:
    reasons: list[str] = []
    paths = {
        "eval_101": root / dataset / "eval_101",
        "eval_104": root / dataset / "eval_104",
        "eval_105": root / dataset / "eval_105",
        "eval_102": root / "architecture" / "eval_102",
        "eval_103": root / "semantic_reference" / "eval_103",
        "eval_106": root / "representative_models" / "eval_106",
    }
    requirements = {
        "eval_101": ["selected_operating_point.json", "metrics.json", "validation.json", "config_snapshot.yaml", "runtime_provenance.json"],
        "eval_102": ["metrics.json", "validation.json", "config_snapshot.yaml", "runtime_provenance.json", "hapr_legality_matrix.csv", "adc_capacity_sensitivity.csv"],
        "eval_103": ["metrics.json", "validation.json", "config_snapshot.yaml", "runtime_provenance.json"],
        "eval_104": ["summary.json", "metrics.json", "validation.json", "config_snapshot.yaml", "runtime_provenance.json", "orthogonal_ablation.csv", "baseline_definition.json", "matched_accounting_contract.json"],
        "eval_105": ["summary.json", "metrics.json", "validation.json", "runtime_provenance.json", "config_snapshot.yaml"],
        "eval_106": ["metrics.json", "validation.json", "config_snapshot.yaml", "runtime_provenance.json", "workload_summary.csv", "operator_mapping.csv"],
    }
    for label, names in requirements.items():
        _check_dir_artifacts(paths[label], label, names, reasons)

    # Sensitivity is accepted only when the official table contains all four
    # requested configuration points and all four lock fractions.
    eval101 = paths["eval_101"]
    _check_sensitivity_csv(
        eval101 / "tile_load_sensitivity.csv",
        label="eval_101/tile_load_sensitivity.csv",
        field="cycles_per_tile_load",
        required_values={"256", "1000", "10000", "100000"},
        reasons=reasons,
    )
    _check_sensitivity_csv(
        eval101 / "mrr_lock_fraction_sensitivity.csv",
        label="eval_101/mrr_lock_fraction_sensitivity.csv",
        field="locked_fraction",
        required_values={"0.0", "0.01", "0.05", "0.1"},
        reasons=reasons,
    )

    eval105_valid = False
    eval105_path = paths["eval_105"] / "summary.json"
    if eval105_path.exists():
        try:
            summary = load_json(eval105_path)
            validation = validate_eval105_publication_summary(summary)
            eval105_valid = bool(validation["valid"])
            if not eval105_valid:
                reasons.extend(f"eval_105:{reason}" for reason in validation["reasons"])
        except Exception as exc:
            reasons.append(f"eval_105:unreadable:{type(exc).__name__}:{exc}")

    # The architecture, semantic, and representative mapping reports are
    # shared across datasets; validate their sidecars once but expose the
    # result with each dataset for a compact per-dataset gate report.
    for label in ("eval_102", "eval_103", "eval_106"):
        validation_path = paths[label] / "validation.json"
        if validation_path.exists():
            _load_validation(validation_path, label, reasons)

    # Check that the official ablation contains the five required variants.
    orthogonal_path = paths["eval_104"] / "orthogonal_ablation.csv"
    if orthogonal_path.exists():
        import csv
        with orthogonal_path.open("r", encoding="utf-8", newline="") as handle:
            variants = {row.get("variant", "") for row in csv.DictReader(handle)}
        required_variants = {
            "adc_pooling_without_hapr",
            "hapr_with_64_adcs",
            "lazy_lif_without_hapr",
            "hapr_a8_without_lazy_lif",
            "full_hipsa",
        }
        missing = sorted(required_variants - variants)
        if missing:
            reasons.append(f"eval_104:missing_orthogonal_variants:{','.join(missing)}")

    return {
        "dataset": dataset,
        "passed": not reasons,
        "reasons": reasons,
        "eval_105_valid": eval105_valid,
        "shared_artifacts": {
            "eval_102": str(paths["eval_102"]),
            "eval_103": str(paths["eval_103"]),
            "eval_106": str(paths["eval_106"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="results/eval_v10")
    p.add_argument("--datasets", nargs="+", default=["cifar10dvs", "dvsgesture"])
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args(argv)
    root = _repo_path(args.output_root)
    reports = [validate_dataset(root, dataset) for dataset in args.datasets]
    reasons = [f"{row['dataset']}:{reason}" for row in reports for reason in row["reasons"]]

    # Phase 3 is the system-evidence release gate. Keep this check here so
    # the final publication gate cannot silently pass on only earlier
    # per-dataset artifacts when system-level evidence has not converged.
    phase3_validation_path = root / "combined" / "eval_108" / "phase3_validation.json"
    phase3_validation: dict[str, Any] | None = None
    if not phase3_validation_path.exists():
        reasons.append(f"missing:combined/eval_108/phase3_validation.json:{phase3_validation_path.as_posix()}")
    else:
        try:
            phase3_validation = load_json(phase3_validation_path)
            if phase3_validation.get("passed") is not True or phase3_validation.get("release_status") != "GO":
                reasons.append("combined/eval_108:reported_failed")
        except Exception as exc:
            reasons.append(f"invalid:combined/eval_108/phase3_validation.json:{type(exc).__name__}:{exc}")

    result = {
        "eval_name": "eval_107",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": not reasons,
        "release_status": "GO" if not reasons else "NO_GO",
        "datasets": reports,
        "phase3_evidence": phase3_validation,
        "reasons": reasons,
        "required_evidence": ["G4/A8 contract", "queue/state semantic validation", "lock/configuration sensitivity", "matched baseline", "orthogonal ablation", "full CUDA publication replay", "representative mapping", "Phase 3 system evidence summary"],
        "claim_boundary": "release completeness/provenance gate; does not upgrade architecture-level evidence to silicon measurement",
    }
    out = root / "combined" / "eval_107"
    out.mkdir(parents=True, exist_ok=True)
    save_json(result, out / "metrics.json")
    save_json(result, out / "validation.json")
    save_run_manifest(out, eval_name="eval_107", command=" ".join(sys.argv), inputs={"output_root": args.output_root, "datasets": args.datasets}, outputs={"metrics": "metrics.json", "validation": "validation.json"})
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
