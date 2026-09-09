"""Official eval_101 entry point for selected-point accounting."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_06_selected_point import main as _legacy_main
from eval.util.official_schema import option_value, option_values, repo_path, write_standard_artifacts
from utils.result_io import load_json


def main() -> int:
    # The delegated implementation has legacy-relative input defaults.  Make
    # the official entry independent of the caller's working directory.
    os.chdir(REPO_ROOT)
    result = int(_legacy_main())
    output_root = repo_path(option_value(sys.argv, "--output-root", "results/eval_v10"), "results/eval_v10")
    datasets = option_values(sys.argv, "--datasets", ("cifar10dvs", "dvsgesture"))
    configs = {
        "hardware": repo_path(option_value(sys.argv, "--hardware", "configs/hardware_hipsa_paper.yaml"), "configs/hardware_hipsa_paper.yaml"),
        "device_params": repo_path(option_value(sys.argv, "--device-params", "configs/device_params.yaml"), "configs/device_params.yaml"),
    }
    for dataset in datasets:
        out = output_root / dataset / "eval_101"
        summary_path = out / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        selected = summary.get("selected_operating_point", summary)
        contract = selected.get("g4_a8_contract", summary.get("g4_a8_contract", {}))
        write_standard_artifacts(
            out,
            eval_name="eval_101",
            metrics=summary,
            validation={
                "passed": bool(contract.get("valid", False)),
                "g4_a8_contract": contract,
                "selected_point": "G4/A8",
                "matched_accounting_present": bool(selected.get("selected_design")),
                "sensitivity_artifacts_present": all(
                    (out / name).exists()
                    for name in ("mrr_lock_fraction_sensitivity.csv", "tile_load_sensitivity.csv")
                ),
                "claim_boundary": "selected-point architecture-level accounting; not timing closure or silicon measurement",
            },
            config_paths=configs,
            evidence_class="architecture_level_selected_point_accounting",
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
