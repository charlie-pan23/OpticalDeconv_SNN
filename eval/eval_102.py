"""Official G4/A8 capacity, HAPR legality, and evidence-boundary report."""
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

from hardware.adc_evidence_boundary import ADCEvidenceConfig, build_adc_evidence_boundary, capacity_sensitivity
from hardware.hapr_legality import HAPRLegalityConfig, build_hapr_legality_matrix
from eval.util.official_schema import repo_path, write_standard_artifacts
from utils.eval_v10 import validate_g4_a8_contract
from utils.result_io import load_yaml, save_csv_rows, save_run_manifest


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _write_case(hardware: dict[str, Any], device: dict[str, Any]) -> dict[str, Any]:
    contract = validate_g4_a8_contract(hardware)
    tiles = _int(hardware.get("photonic_tiles", {}).get("num_tiles"), 4)
    outputs = _int(hardware.get("mapping", {}).get("array_cols"), 64)
    evidence = build_adc_evidence_boundary(ADCEvidenceConfig(
        adc_macros=contract["adc_macros"],
        samples_per_macro_per_cycle=int(contract["samples_per_macro_per_cycle"]),
        post_hapr_lanes=contract["post_hapr_lanes"],
        architecture_clock_ghz=_float(contract["system_clock_hz"], 1e9) / 1e9,
        adc_power_mw_per_macro=_float(device.get("adc", {}).get("conventional_adc_macro", {}).get("power_mw"), 14.8),
        temporal_analog_storage=bool(contract["temporal_analog_storage"]),
        mux_ratio="64:8",
    ))
    hapr = build_hapr_legality_matrix(HAPRLegalityConfig(
        physical_tiles=tiles,
        outputs_per_tile=outputs,
        post_hapr_lanes=contract["post_hapr_lanes"],
        adc_service_capacity=int(contract["nominal_sample_slots_per_cycle"]),
        temporal_analog_storage=bool(contract["temporal_analog_storage"]),
    ))
    sensitivity = capacity_sensitivity(post_hapr_lanes=contract["post_hapr_lanes"])
    selected = {
        "name": "G4_A8",
        "contract": contract,
        "capacity": evidence["nominal_slots_per_window"],
        "capacity_margin_slots": evidence["capacity_margin_slots"],
        "hapr_selected_status": next(row for row in hapr["nominal_matrix"] if row["group_size"] == 4),
        "evidence_class": "architecture_level_capacity_admission_and_legality",
        "claim_boundary": evidence["claim_boundary"],
    }
    return {
        "eval_name": "eval_102",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_point": selected,
        "contract": contract,
        "hapr_legality": hapr,
        "adc_evidence": evidence,
        "capacity_sensitivity": sensitivity,
        "validation": {
            "g4_a8_contract_valid": bool(contract.get("valid")),
            "hapr_matrix_passed": bool(hapr["matrix_passed"]),
            "timing_closure": False,
            "physically_closed": False,
            "silicon_validated": False,
            "claim_boundary": "architecture-level capacity/admission analysis; mux settling, ADC aperture and TIA recovery are not circuit-closed",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="results/eval_v10")
    p.add_argument("--dataset", default="architecture")
    p.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    p.add_argument("--device-params", default="configs/device_params.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hardware_path = repo_path(args.hardware, "configs/hardware_hipsa_paper.yaml")
    device_path = repo_path(args.device_params, "configs/device_params.yaml")
    output_root = repo_path(args.output_root, "results/eval_v10")
    hardware = load_yaml(hardware_path)
    device = load_yaml(device_path)
    result = _write_case(hardware, device)
    out = output_root / args.dataset / "eval_102"
    write_standard_artifacts(
        out,
        eval_name="eval_102",
        metrics=result,
        validation=result["validation"],
        config_paths={"hardware": hardware_path, "device_params": device_path},
        evidence_class="architecture_level_capacity_admission_and_legality",
        extra_provenance={"selected_point": "G4/A8"},
    )
    save_csv_rows(result["hapr_legality"]["nominal_matrix"], out / "hapr_legality_matrix.csv")
    save_csv_rows(result["hapr_legality"]["edge_cases"], out / "hapr_edge_cases.csv")
    save_csv_rows(result["capacity_sensitivity"], out / "adc_capacity_sensitivity.csv")
    save_run_manifest(out, eval_name="eval_102", command=" ".join(sys.argv), inputs={"hardware": "configs/hardware_hipsa_paper.yaml", "device_params": "configs/device_params.yaml"}, outputs={"metrics": "metrics.json", "validation": "validation.json", "config_snapshot": "config_snapshot.yaml", "hapr_matrix": "hapr_legality_matrix.csv", "capacity_sensitivity": "adc_capacity_sensitivity.csv"})
    print(json.dumps({"eval_name": "eval_102", "output_dir": str(out), "valid": result["validation"]["g4_a8_contract_valid"] and result["validation"]["hapr_matrix_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
