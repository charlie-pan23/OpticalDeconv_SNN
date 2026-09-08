"""Integration tests for the formal eval_103 semantic-validation entry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.eval_103 import run_semantic_validation, write_outputs


class Eval103SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_semantic_validation(seed=42)

    def test_all_release_checks_pass(self) -> None:
        self.assertTrue(self.result["validation_passed"])
        self.assertTrue(all(self.result["validation_checks"].values()))
        selected = self.result["selected_capacity"]["summary"]
        a6 = self.result["a6_capacity"]["summary"]
        self.assertEqual((selected["accepted"], selected["rejected"]), (64, 0))
        self.assertEqual((a6["accepted"], a6["rejected"]), (60, 4))

    def test_expected_spike_and_reset_semantics(self) -> None:
        commits = self.result["variants"]["canonical"]["signature"]["commits"]
        observed = {
            (row["output_neuron_id"], row["timestep_id"]): (
                row["spike"], row["reset_applied"], row["membrane_after_commit"]
            )
            for row in commits
        }
        self.assertEqual(observed[(0, 0)], (1, 1, 0.0))
        self.assertEqual(observed[(0, 1)][0:2], (0, 0))
        self.assertEqual(observed[(1, 0)][0:2], (0, 0))
        self.assertEqual(observed[(1, 1)], (1, 1, 0.0))

    def test_output_bundle_contains_required_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval_103"
            write_outputs(self.result, output, seed=42)
            expected = {
                "config_snapshot.yaml",
                "runtime_provenance.json",
                "metrics.json",
                "validation.json",
                "trace.csv",
                "request_trace.csv",
                "partition_receive_trace.csv",
                "state_transition_trace.csv",
                "state_commit_trace.csv",
                "hapr_legality.json",
                "adc_evidence_boundary.json",
                "hapr_legality_matrix.csv",
                "hapr_edge_cases.csv",
                "adc_capacity_sensitivity.csv",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["passed"])
            self.assertTrue(metrics["validation_passed"])
            self.assertIn("claim_boundary", metrics)


if __name__ == "__main__":
    unittest.main()

