"""Integration coverage for Phase 1-C/D outputs in eval_103."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.eval_103 import run_semantic_validation, write_outputs


class Eval103Phase1CDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_semantic_validation(seed=42)

    def test_phase1_cd_release_checks_pass(self) -> None:
        checks = self.result["validation_checks"]
        required = {
            "hapr_legality_matrix_passed",
            "g8_g16_rejected_by_explicit_constraint",
            "non64_output_tail_explicit",
            "row_tile_tail_explicit",
            "aggregate_trace_rejected_by_runtime_guard",
            "adc_evidence_boundary_explicit",
            "no_timing_closure_claim",
            "conversion_latency_pipeline_modeled",
            "adc_power_conditional",
            "capacity_sensitivity_g4_a8_is_80",
            "capacity_sensitivity_a6_is_60",
        }
        self.assertTrue(required.issubset(checks))
        self.assertTrue(all(checks[name] for name in required))

    def test_phase1_cd_fields_and_files_are_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval_103"
            write_outputs(self.result, output, seed=42)
            hapr = json.loads((output / "hapr_legality.json").read_text(encoding="utf-8"))
            adc = json.loads((output / "adc_evidence_boundary.json").read_text(encoding="utf-8"))
            self.assertTrue(hapr["matrix_passed"])
            self.assertEqual(adc["nominal_slots_per_window"], 80)
            self.assertFalse(adc["timing_closure"])
            self.assertTrue((output / "hapr_legality_matrix.csv").exists())
            self.assertTrue((output / "hapr_edge_cases.csv").exists())
            self.assertTrue((output / "adc_capacity_sensitivity.csv").exists())


if __name__ == "__main__":
    unittest.main()
