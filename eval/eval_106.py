"""Official eval_106 representative SNN/Transformer mapping report."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_18_advanced_snn_mapping import map_sdt, map_sew
from eval.eval_11_transformer_mapping import build_mapping
from eval.util.official_schema import repo_path, write_standard_artifacts
from utils.eval_v10 import validate_g4_a8_contract
from utils.result_io import load_yaml, save_csv_rows, save_run_manifest


def _aggregate(name: str, family: str, workload: dict[str, Any], ops: list[dict[str, Any]], tiles: int) -> dict[str, Any]:
    optical = [row for row in ops if row.get("execution") == "optical_mrr"]
    total = sum(int(row.get("multiplicity", 1)) for row in ops)
    optical_count = sum(int(row.get("multiplicity", 1)) for row in optical)
    return {
        "workload": name,
        "family": family,
        "operator_instances": total,
        "optical_static_weight_instances": optical_count,
        "electronic_dynamic_operator_instances": total - optical_count,
        "optical_operator_coverage_percent": 100.0 * optical_count / max(total, 1),
        "logical_weight_tiles": sum(int(row.get("logical_weight_tiles", 0)) for row in optical),
        "physical_load_rounds": sum(int(row.get("four_tile_load_rounds", row.get("physical_load_rounds", 0))) for row in optical),
        "status": "mapping_only_no_accuracy_latency_or_energy_claim",
        "unsupported_claims": ["accuracy", "latency", "energy", "throughput"],
        "physical_tiles": tiles,
        "source_url": workload.get("source_url", ""),
    }


def build_report(advanced: dict[str, Any], transformer: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    contract = validate_g4_a8_contract(hardware)
    tiles = int(contract["num_tiles"])
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for workload in advanced.get("workloads", []):
        ops = map_sew(workload, tiles) if workload.get("family") == "residual_convolutional_snn" else map_sdt(workload, tiles)
        summaries.append(_aggregate(workload["name"], workload["family"], workload, ops, tiles))
        details.extend({"workload": workload["name"], **row} for row in ops)
    transformer_ops = transformer.get("operators", [])
    tw = transformer.get("workload", {})
    summaries.append(_aggregate(tw.get("name", "transformer_mapping"), "spiking_transformer", tw, transformer_ops, tiles))
    details.extend({"workload": tw.get("name", "transformer_mapping"), **row} for row in transformer_ops)
    return {
        "eval_name": "eval_106",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence_level": "mapping_only",
        "hardware_contract": contract,
        "summaries": summaries,
        "operator_mapping": details,
        "scope": {
            "optical_components": ["static weighted MVM projections", "HAPR/ADC eligibility by shared substrate"],
            "electronic_components": ["dynamic attention/mask/add", "residual elementwise operations", "state update"],
            "requires_for_end_to_end_claim": ["trained checkpoint", "per-layer activity trace", "same eval_105 replay and state contract"],
        },
        "claim_boundary": "representative workload mapping; not end-to-end model replay or measured hardware result",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", default="configs/advanced_snn_workloads.yaml")
    p.add_argument("--transformer-spec", default="configs/spiking_transformer_mapping.yaml")
    p.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    p.add_argument("--output-root", default="results/eval_v10")
    p.add_argument("--dataset", default="representative_models")
    args = p.parse_args(argv)
    spec_path = repo_path(args.spec, "configs/advanced_snn_workloads.yaml")
    transformer_path = repo_path(args.transformer_spec, "configs/spiking_transformer_mapping.yaml")
    hardware_path = repo_path(args.hardware, "configs/hardware_hipsa_paper.yaml")
    output_root = repo_path(args.output_root, "results/eval_v10")
    advanced, transformer, hardware = load_yaml(spec_path), load_yaml(transformer_path), load_yaml(hardware_path)
    result = build_report(advanced, build_mapping(transformer, hardware), hardware)
    out = output_root / args.dataset / "eval_106"
    validation = {
        "passed": True,
        "evidence_level": result["evidence_level"],
        "unsupported_claims": ["accuracy", "latency", "energy", "throughput"],
        "claim_boundary": result["claim_boundary"],
    }
    write_standard_artifacts(
        out,
        eval_name="eval_106",
        metrics=result,
        validation=validation,
        config_paths={"hardware": hardware_path, "advanced_spec": spec_path, "transformer_spec": transformer_path},
        evidence_class="representative_mapping_only",
    )
    save_csv_rows(result["summaries"], out / "workload_summary.csv")
    save_csv_rows(result["operator_mapping"], out / "operator_mapping.csv")
    save_run_manifest(out, eval_name="eval_106", command=" ".join(sys.argv), inputs={"spec": "configs/advanced_snn_workloads.yaml", "transformer_spec": "configs/spiking_transformer_mapping.yaml", "hardware": "configs/hardware_hipsa_paper.yaml"}, outputs={"metrics": "metrics.json", "validation": "validation.json", "config_snapshot": "config_snapshot.yaml", "workload_summary": "workload_summary.csv", "operator_mapping": "operator_mapping.csv"})
    print(f"[eval_106] representative mapping artifacts saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
