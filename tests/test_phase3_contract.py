from pathlib import Path

from eval.util.phase3 import validate_phase3_contract
from utils.result_io import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_phase3_contract_is_g4_a8_and_complete():
    contract = load_yaml(REPO_ROOT / "configs" / "phase3_experiment_contract.yaml")
    result = validate_phase3_contract(contract)
    assert result["valid"], result["reasons"]
    assert contract["hardware"]["hapr_group"] == 4
    assert contract["hardware"]["adc_macros"] == 8


def test_phase3_contract_rejects_missing_sensitivity_point():
    contract = load_yaml(REPO_ROOT / "configs" / "phase3_experiment_contract.yaml")
    contract["sensitivity"]["tile_load_cycles"] = [256]
    result = validate_phase3_contract(contract)
    assert not result["valid"]
    assert "configuration_sensitivity_complete" in result["reasons"]
