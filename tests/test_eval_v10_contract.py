from __future__ import annotations
import copy
import unittest
from hardware.publication_accounting import build_lock_fraction_sensitivity
from utils.eval_v10 import validate_eval105_publication_summary, validate_g4_a8_contract
from eval.eval_105 import CONDITIONS, HardwareReplayHooks, _adc_quantize, resolve_photonic_mvm_layer_names


class EvalV10ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "photonic_tiles": {"num_tiles": 4},
            "mapping": {"array_rows": 64, "array_cols": 64},
            "transaction_scheduler": {"system_clock_hz": 1_000_000_000},
            "hapr_adc_backend": {
                "hapr_group_size": 4,
                "hapr_output_lanes_total": 64,
                "adc_macros": 8,
                "adc_samples_per_macro_per_architecture_cycle": 10,
                "temporal_analog_storage": False,
            },
        }

    def test_g4_a8_contract_accepts_80_slots_for_64_lanes(self):
        result = validate_g4_a8_contract(self.cfg)
        self.assertTrue(result["valid"])
        self.assertEqual(result["post_hapr_lanes"], 64)
        self.assertEqual(result["nominal_sample_slots_per_cycle"], 80)

    def test_a6_is_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["hapr_adc_backend"]["adc_macros"] = 6
        with self.assertRaises(ValueError):
            validate_g4_a8_contract(cfg)

    def test_temporal_analog_storage_is_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["hapr_adc_backend"]["temporal_analog_storage"] = True
        with self.assertRaises(ValueError):
            validate_g4_a8_contract(cfg)

    def test_adc_quantization_uses_frozen_full_scale(self):
        import torch

        value = torch.tensor([0.5, 2.0], dtype=torch.float32)
        quantized = _adc_quantize(value, bits=6, full_scale=1.0)
        self.assertLessEqual(float(quantized.abs().max()), 1.0)
        self.assertAlmostEqual(float(quantized[-1]), 1.0, places=6)

    def test_adc_quantization_is_not_batch_oracle_scaled(self):
        import torch

        single = _adc_quantize(torch.tensor([0.5]), bits=6, full_scale=1.0)
        mixed = _adc_quantize(torch.tensor([0.5, 100.0]), bits=6, full_scale=1.0)
        self.assertAlmostEqual(float(single[0]), float(mixed[0]), places=6)

    def test_photonic_replay_excludes_electronic_classifier(self):
        import torch
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 2, 1, bias=False)
                self.fc1 = nn.Linear(2, 2, bias=False)
                self.fc2 = nn.Linear(2, 2, bias=False)
                self.photonic_mvm_layer_names = ["conv1", "fc1"]

        model = TinyModel()
        fc2_before = model.fc2.weight.detach().clone()
        names = resolve_photonic_mvm_layer_names(model)
        hooks = HardwareReplayHooks(
            model,
            CONDITIONS["quantized_clean"],
            seed=0,
            hapr_group_size=4,
            adc_full_scales={"conv1": 1.0, "fc1": 1.0},
            target_layer_names=names,
        )
        hooks.quantize_weights()
        hooks.register()
        try:
            self.assertEqual(names, ["conv1", "fc1"])
            self.assertEqual(hooks.mvm_layers, ["conv1", "fc1"])
            self.assertTrue(torch.equal(model.fc2.weight, fc2_before))
        finally:
            hooks.remove()

    def _valid_eval105_summary(self):
        return {
            "eval_name": "eval_105",
            "num_samples": 10,
            "combined_condition_accuracy_percent": 80.0,
            "precision_contract": {"weight_bits": 6, "adc_bits": 6, "membrane_bits": 16},
            "split_manifest": {
                "status": "verified_full_split_replay",
                "num_samples_expected": 10,
                "num_samples_replayed": 10,
                "full_split_replayed": True,
                "max_batches": None,
            },
            "adc_calibration": {
                "calibration_split": "val",
                "reporting_split": "test",
                "test_data_used_for_calibration": False,
                "status": "verified_full_calibration_split",
                "full_calibration_split_used": True,
                "calibration_batches_requested": None,
                "full_scales": {"conv1": 1.0, "fc1": 2.0},
            },
            "runtime_provenance": {
                "publication_replay_ready": True,
                "deterministic_algorithms_enabled": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cublas_workspace_config": ":4096:8",
                "full_replay_requested": True,
            },
            "replay_scope": {
                "status": "verified_photonic_only",
                "include_final_classifier": False,
                "final_classifier_excluded": True,
                "photonic_mvm_layer_names": ["conv1", "fc1"],
            },
        }

    def test_eval105_validator_requires_photonic_only_scope(self):
        summary = self._valid_eval105_summary()
        self.assertTrue(validate_eval105_publication_summary(summary)["valid"])
        summary["replay_scope"]["photonic_mvm_layer_names"].append("fc2")
        summary["replay_scope"]["final_classifier_excluded"] = False
        result = validate_eval105_publication_summary(summary)
        self.assertFalse(result["valid"])
        self.assertIn("electronic_final_classifier_not_excluded", result["reasons"])

    def test_eval105_validator_rejects_partial_probe(self):
        summary = self._valid_eval105_summary()
        summary["num_samples"] = 2
        summary["split_manifest"].update({
            "num_samples_replayed": 2,
            "full_split_replayed": False,
            "max_batches": 1,
            "status": "diagnostic_partial_or_unverified_split",
        })
        summary["runtime_provenance"].update({
            "publication_replay_ready": False,
            "full_replay_requested": False,
        })
        result = validate_eval105_publication_summary(summary)
        self.assertFalse(result["valid"])
        self.assertIn("reporting_split_incomplete", result["reasons"])
        self.assertIn("diagnostic_replay_cap_present", result["reasons"])

    def test_eval105_validator_rejects_partial_calibration(self):
        summary = self._valid_eval105_summary()
        summary["adc_calibration"].update({
            "status": "diagnostic_partial_calibration",
            "full_calibration_split_used": False,
            "calibration_batches_requested": 1,
        })
        result = validate_eval105_publication_summary(summary)
        self.assertFalse(result["valid"])
        self.assertIn("full_calibration_split_not_verified", result["reasons"])
        self.assertIn("diagnostic_calibration_cap_present", result["reasons"])

    def test_eval105_validator_rejects_unfixed_cuda_runtime(self):
        summary = self._valid_eval105_summary()
        summary["runtime_provenance"].update({
            "publication_replay_ready": False,
            "deterministic_algorithms_enabled": False,
            "cudnn_deterministic": False,
            "cudnn_benchmark": True,
            "cublas_workspace_config": None,
        })
        result = validate_eval105_publication_summary(summary)
        self.assertFalse(result["valid"])
        self.assertIn("runtime_not_publication_ready", result["reasons"])
        self.assertIn("cublas_workspace_config_not_fixed", result["reasons"])

    def test_lock_fraction_sensitivity_uses_declared_four_cases(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["mrr_lock_sensitivity"] = {
            "physical_ring_count": 32768,
            "locking_power_mw_per_ring": 1.2,
            "locked_fractions": [0.0, 0.01, 0.05, 0.10],
            "recommended_reporting_fraction": 0.01,
        }
        summary, rows = build_lock_fraction_sensitivity(
            selected_power_w=1.0,
            latency_us_per_image=10.0,
            config_energy_uJ_per_image=0.5,
            non_mvm_energy_uJ_per_image=0.25,
            hardware_cfg=cfg,
        )
        self.assertEqual(summary["locked_fractions"], [0.0, 0.01, 0.05, 0.10])
        self.assertEqual([row["locked_fraction"] for row in rows], [0.0, 0.01, 0.05, 0.10])
        one_percent = rows[1]
        self.assertAlmostEqual(one_percent["added_lock_power_mw"], 393.216)
        self.assertEqual(one_percent["recommended_reporting_case"], 1)


if __name__ == "__main__":
    unittest.main()

