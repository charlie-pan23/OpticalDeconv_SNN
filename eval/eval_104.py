"""Official eval_104 entry point for cumulative and orthogonal ablation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_07_ablation_v3 import main as _legacy_main
from eval.util.official_schema import (
    matched_accounting_contract,
    option_value,
    option_values,
    repo_path,
    write_standard_artifacts,
)
from utils.result_io import load_json, load_yaml, save_json


def main() -> int:
    os.chdir(REPO_ROOT)
    result = int(_legacy_main())
    output_root = repo_path(option_value(sys.argv, "--output-root", "results/eval_v10"), "results/eval_v10")
    datasets = option_values(sys.argv, "--datasets", ("cifar10dvs", "dvsgesture"))
    lock_fraction = float(option_value(sys.argv, "--lock-fraction", "0.01"))
    configs = {
        "hardware": repo_path(
            option_value(sys.argv, "--hardware", "configs/hardware_hipsa_paper.yaml"),
            "configs/hardware_hipsa_paper.yaml",
        ),
        "device_params": repo_path(
            option_value(sys.argv, "--device-params", "configs/device_params.yaml"),
            "configs/device_params.yaml",
        ),
    }
    hardware_cfg = load_yaml(configs["hardware"])
    tile_load_cycles = int(hardware_cfg.get("mrr_configuration", {}).get("cycles_per_tile_load", 256))
    scope = matched_accounting_contract(lock_fraction=lock_fraction, tile_load_cycles=tile_load_cycles)

    for dataset in datasets:
        out = output_root / dataset / "eval_104"
        summary_path = out / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        save_json(scope, out / "matched_accounting_contract.json")
        baseline = dict(summary.get("baseline_definition", {}))
        baseline.update(scope)
        save_json(baseline, out / "baseline_definition.json")
        orthogonal = out / "orthogonal_ablation.csv"
        orthogonal_count = 0
        if orthogonal.exists():
            with orthogonal.open("r", encoding="utf-8") as handle:
                orthogonal_count = max(sum(1 for _ in handle) - 1, 0)
        write_standard_artifacts(
            out,
            eval_name="eval_104",
            metrics=summary,
            validation={
                "passed": bool(summary.get("g4_a8_contract", {}).get("valid", False)),
                "orthogonal_ablation_present": orthogonal.exists(),
                "orthogonal_variant_count": orthogonal_count,
                "accuracy_policy": summary.get("accuracy", {}).get("status"),
                "matched_accounting_contract": scope,
                "configuration_sensitivity_status": (
                    "not_run_in_eval_104; tile_load_cycles is read from the publication hardware config"
                ),
                "tile_load_cycles_source": (
                    "configs/hardware_hipsa_paper.yaml:mrr_configuration.cycles_per_tile_load"
                ),
                "claim_boundary": (
                    "mechanism ablation from architecture-level counters; "
                    "variant accuracy requires dedicated replay"
                ),
            },
            config_paths=configs,
            evidence_class="architecture_level_orthogonal_ablation",
            extra_provenance={
                "matched_accounting_contract": scope,
                "configuration_sensitivity_status": "not_run_in_eval_104",
            },
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())