"""Focused Phase 1-C HAPR legality and edge-case tests."""

from __future__ import annotations

import unittest

from hardware.hapr_legality import (
    HAPRCase,
    HAPRLegalityConfig,
    build_hapr_legality_matrix,
    evaluate_hapr_case,
    validate_partition_contract,
    validate_runtime_trace_guard,
    validate_same_neuron_grouping,
)


class HAPRLegalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HAPRLegalityConfig()

    def test_nominal_group_statuses_and_explicit_g8_g16_rejection(self) -> None:
        matrix = build_hapr_legality_matrix(self.config)
        self.assertTrue(matrix["matrix_passed"])
        rows = {row["group_size"]: row for row in matrix["nominal_matrix"]}
        self.assertEqual(rows[1]["status"], "supported_baseline")
        self.assertEqual(rows[2]["status"], "supported_candidate")
        self.assertEqual(rows[4]["status"], "formal_selected_point")
        for group in (8, 16):
            self.assertFalse(rows[group]["legal"])
            self.assertTrue(any("physical_tiles" in reason for reason in rows[group]["reasons"]))

    def test_same_neuron_grouping_rejects_cross_neuron_and_cross_tile(self) -> None:
        base = {
            "sample_id": "s0",
            "timestep_id": 0,
            "layer_id": "layer0",
            "hapr_group_id": "g0",
            "output_tile_id": "tile0",
            "output_neuron_id": 0,
        }
        cross_neuron = [base, {**base, "output_neuron_id": 1}]
        cross_tile = [base, {**base, "output_tile_id": "tile1"}]
        self.assertFalse(validate_same_neuron_grouping(cross_neuron)[0])
        self.assertFalse(validate_same_neuron_grouping(cross_tile)[0])

    def test_partition_contract_rejects_duplicate_missing_and_unexpected(self) -> None:
        base = {
            "sample_id": "s0",
            "timestep_id": 0,
            "layer_id": "layer0",
            "hapr_group_id": "g0",
            "output_tile_id": "tile0",
            "output_neuron_id": 0,
        }
        duplicate = [{**base, "partition_id": "p0"}, {**base, "partition_id": "p0"}]
        missing = [{**base, "partition_id": "p0"}]
        unexpected = [{**base, "partition_id": "p0"}, {**base, "partition_id": "px"}]
        self.assertFalse(validate_partition_contract(duplicate, expected_partition_ids=("p0", "p1"))[0])
        self.assertFalse(validate_partition_contract(missing, expected_partition_ids=("p0", "p1"))[0])
        self.assertFalse(validate_partition_contract(unexpected, expected_partition_ids=("p0", "p1"))[0])

    def test_output_tail_and_row_tile_tail_are_explicit_but_legal(self) -> None:
        output_tail = evaluate_hapr_case(HAPRCase(group_size=4, output_dim=65), self.config)
        row_tail = evaluate_hapr_case(HAPRCase(group_size=4, input_dim=130), self.config)
        self.assertTrue(output_tail.legal)
        self.assertEqual(output_tail.output_tail_lanes, 63)
        self.assertTrue(row_tail.legal)
        self.assertTrue(row_tail.row_tile_tail)
        self.assertEqual(row_tail.row_tile_tail_fanin, 3)

    def test_runtime_guard_distinguishes_aggregate_and_runtime_schema(self) -> None:
        aggregate = [{"aggregate_cycle": 0, "lane_count": 64}]
        self.assertFalse(validate_runtime_trace_guard(aggregate)[0])
        runtime = [{
            "request_id": "r0",
            "sample_id": "s0",
            "timestep_id": 0,
            "layer_id": "l0",
            "output_tile_id": "tile0",
            "output_neuron_id": 0,
            "partition_id": "p0",
            "hapr_group_id": "g0",
            "trace_provenance": "real_runtime:test_schema",
        }]
        self.assertTrue(validate_runtime_trace_guard(runtime)[0])


if __name__ == "__main__":
    unittest.main()
