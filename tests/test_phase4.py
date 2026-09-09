from __future__ import annotations

import copy
import json
from pathlib import Path

from eval.util.phase4 import (
    _validate_seed_records,
    aggregate_accuracy_statistics,
    collect_ablation_evidence,
    collect_area_sensitivity,
    collect_capacity_evidence,
    collect_configuration_provenance,
    collect_seed_records,
    validate_phase4_contract,
)
from utils.result_io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "eval_v10"


def test_phase4_contract_is_valid_and_g4_a8():
    contract = load_yaml(REPO_ROOT / "configs" / "phase4_evidence_contract.yaml")
    result = validate_phase4_contract(contract)
    assert result["passed"], result["reasons"]
    assert contract["hardware"]["hapr_group"] == 4
    assert contract["hardware"]["adc_macros"] == 8


def test_phase4_seed_provenance_and_statistics_are_explicit():
    records = collect_seed_records(RESULT_ROOT, ["cifar10dvs", "dvsgesture"])
    assert len(records) == 10
    reasons, details = _validate_seed_records(records, required_seeds=[0, 1, 2, 3, 4])
    assert not reasons, reasons
    assert details["cifar10dvs"]["observed_seeds"] == [0, 1, 2, 3, 4]
    rows, aggregate = aggregate_accuracy_statistics(records, required_seeds=[0, 1, 2, 3, 4])
    combined = next(row for row in rows if row["dataset"] == "cifar10dvs" and row["condition"] == "combined")
    assert combined["n"] == 5
    assert abs(combined["mean_accuracy_percent"] - 72.66) < 1e-9
    assert aggregate["statistic_label"] == "paired_replay_perturbation_statistics"


def test_phase4_seed_validation_rejects_missing_duplicate_and_mismatch():
    records = collect_seed_records(RESULT_ROOT, ["cifar10dvs"])
    records = [row for row in records if row["seed"] != 4]
    records.append(copy.deepcopy(records[-1]))
    records[-1]["seed"] = 3
    records[-1]["config_sha256"] = "mismatch"
    reasons, _ = _validate_seed_records(records, required_seeds=[0, 1, 2, 3, 4])
    assert any("missing_seeds" in reason for reason in reasons)
    assert any("duplicate_seeds" in reason for reason in reasons)
    assert any("provenance_mismatch" in reason for reason in reasons)


def test_phase4_ablation_has_no_inferred_accuracy_for_non_full_hipsa():
    rows = collect_ablation_evidence(RESULT_ROOT, ["cifar10dvs", "dvsgesture"])
    assert len(rows) == 10
    assert all(not row["accuracy_available"] for row in rows if row["variant"] != "full_hipsa")
    assert all(row["accuracy_available"] for row in rows if row["variant"] == "full_hipsa")


def test_phase4_capacity_and_area_boundaries_are_explicit():
    capacity = collect_capacity_evidence(RESULT_ROOT, ["cifar10dvs", "dvsgesture"])
    assert all(row["same_window_admission_feasible"] for row in capacity)
    assert all(row["capacity_margin_slots"] == 16 for row in capacity)
    assert all(not row["circuit_timing_closed"] for row in capacity)
    area = collect_area_sensitivity(RESULT_ROOT)
    expected = [4.6150062288, 5.53800747456, 6.9225093432, 9.2300124576]
    assert all(abs(row["adjusted_area_mm2"] - value) < 1e-9 for row, value in zip(area, expected))
    assert all("not die area" in row["claim_boundary"] for row in area)


def test_phase4_configuration_provenance_exposes_constant_energy_warning():
    rows = collect_configuration_provenance(RESULT_ROOT, ["cifar10dvs", "dvsgesture"])
    assert {row["tile_load_events_per_image"] for row in rows} == {75, 39}
    assert all(row["source_table_keeps_configuration_energy_constant_across_cycle_sensitivity"] for row in rows)
    assert all(not row["mrr_stabilization_time_included"] for row in rows)


def test_phase4_generated_validation_is_passed():
    validation_path = RESULT_ROOT / "combined" / "eval_109" / "phase4a_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["passed"] is True
    assert validation["checks"]["seed_provenance_valid"] is True
    assert validation["checks"]["required_orthogonal_variants_present"] is True
