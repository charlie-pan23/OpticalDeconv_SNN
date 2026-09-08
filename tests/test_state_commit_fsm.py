"""Tests for HIPSA partition completion and state-commit semantics."""

from __future__ import annotations

import random
import unittest

from hardware.state_commit_fsm import (
    StateCommitConfig,
    StateCommitSpec,
    simulate_state_commit_fsm,
)


def spec(
    timestep: int,
    *,
    neuron: int = 0,
    sample: str = "s0",
    layer: str = "conv1",
    tile: str = "tile0",
    partitions: tuple[str, ...] = ("p0", "p1", "p2"),
) -> StateCommitSpec:
    return StateCommitSpec(
        sample_id=sample,
        timestep_id=timestep,
        layer_id=layer,
        output_neuron_id=neuron,
        output_tile_id=tile,
        expected_partition_ids=partitions,
    )


def completion(
    request_id: str,
    timestep: int,
    partition: str,
    value: float,
    sequence: int,
    *,
    cycle: int | None = None,
    neuron: int = 0,
    sample: str = "s0",
    layer: str = "conv1",
    tile: str = "tile0",
    status: str = "COMPLETED",
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "sample_id": sample,
        "timestep_id": timestep,
        "layer_id": layer,
        "output_neuron_id": neuron,
        "output_tile_id": tile,
        "partition_id": partition,
        "hapr_group_id": f"{layer}:j{neuron}:{partition}",
        "partial_sum_value": value,
        "completion_cycle": sequence if cycle is None else cycle,
        "completion_sequence": sequence,
        "status": status,
    }


def ordered_fixture(order: list[int]) -> tuple[list[dict[str, object]], list[StateCommitSpec]]:
    logical = [
        ("r0", 0, "p0", 0.7, 0),
        ("r1", 0, "p1", 0.8, 0),
        ("r2", 0, "p2", 0.9, 0),
        ("r3", 1, "p0", 0.1, 0),
        ("r4", 1, "p1", 0.2, 0),
        ("r5", 1, "p2", 0.3, 0),
        ("r6", 0, "p0", 0.2, 1),
        ("r7", 0, "p1", 0.3, 1),
        ("r8", 0, "p2", 0.3, 1),
        ("r9", 1, "p0", 0.5, 1),
        ("r10", 1, "p1", 0.6, 1),
        ("r11", 1, "p2", 0.7, 1),
    ]
    rows = []
    for sequence, logical_index in enumerate(order):
        request_id, timestep, partition, value, neuron = logical[logical_index]
        rows.append(
            completion(
                request_id,
                timestep,
                partition,
                value,
                sequence,
                cycle=sequence // 3,
                neuron=neuron,
            )
        )
    specs = [spec(t, neuron=n) for n in (0, 1) for t in (0, 1)]
    return rows, specs


def semantic_signature(result: dict[str, object]) -> tuple[object, ...]:
    commits = sorted(
        (
            row["sample_id"],
            row["timestep_id"],
            row["layer_id"],
            row["output_neuron_id"],
            row["membrane_after_commit"],
            row["spike"],
            row["reset_applied"],
        )
        for row in result["commit_trace"]
    )
    final_states = sorted(
        (
            row["sample_id"],
            row["layer_id"],
            row["output_neuron_id"],
            row["membrane"],
            row["last_committed_timestep"],
        )
        for row in result["final_states"]
    )
    return tuple(commits), tuple(final_states), result["summary"]["commit_count"]


class StateCommitFSMTests(unittest.TestCase):
    def test_no_threshold_or_commit_before_all_partitions_arrive(self) -> None:
        rows = [
            completion("r0", 0, "p0", 0.7, 0),
            completion("r1", 0, "p1", 0.8, 1),
        ]
        result = simulate_state_commit_fsm(rows, [spec(0)])
        summary = result["summary"]
        self.assertEqual(summary["state_contexts_committed"], 0)
        self.assertEqual(summary["threshold_count"], 0)
        self.assertEqual(summary["decay_count"], 0)
        self.assertEqual(result["incomplete_contexts"][0]["missing_partition_ids"], ["p2"])
        self.assertFalse(summary["state_commit_valid"])

    def test_complete_context_executes_each_commit_stage_once(self) -> None:
        rows = [
            completion("r0", 0, "p0", 0.7, 0),
            completion("r1", 0, "p1", 0.8, 1),
            completion("r2", 0, "p2", 0.9, 2),
        ]
        result = simulate_state_commit_fsm(rows, [spec(0)])
        summary = result["summary"]
        self.assertEqual(summary["commit_count"], 1)
        self.assertEqual(summary["decay_count"], 1)
        self.assertEqual(summary["threshold_count"], 1)
        self.assertEqual(summary["reset_count"], 1)
        self.assertEqual(result["commit_trace"][0]["spike"], 1)
        self.assertEqual(result["commit_trace"][0]["membrane_after_commit"], 0.0)
        self.assertTrue(summary["state_commit_valid"])

    def test_canonical_reverse_and_random_orders_are_semantically_equal(self) -> None:
        canonical = list(range(12))
        reverse = list(reversed(canonical))
        random_order = canonical.copy()
        random.Random(42).shuffle(random_order)
        results = []
        for order in (canonical, reverse, random_order):
            rows, specs = ordered_fixture(order)
            results.append(simulate_state_commit_fsm(rows, specs))
        signatures = [semantic_signature(result) for result in results]
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])
        self.assertTrue(all(result["summary"]["state_commit_valid"] for result in results))

    def test_later_timestep_waits_for_preceding_commit(self) -> None:
        rows = [
            completion("t1p0", 1, "p0", 0.1, 0, cycle=0),
            completion("t1p1", 1, "p1", 0.2, 1, cycle=1),
            completion("t1p2", 1, "p2", 0.3, 2, cycle=2),
            completion("t0p0", 0, "p0", 0.7, 3, cycle=3),
            completion("t0p1", 0, "p1", 0.8, 4, cycle=4),
            completion("t0p2", 0, "p2", 0.9, 5, cycle=5),
        ]
        result = simulate_state_commit_fsm(rows, [spec(0), spec(1)])
        commits = {row["timestep_id"]: row for row in result["commit_trace"]}
        self.assertEqual(commits[0]["commit_cycle"], 5)
        self.assertEqual(commits[1]["commit_cycle"], 5)
        self.assertEqual(commits[0]["commit_sequence"], 0)
        self.assertEqual(commits[1]["commit_sequence"], 1)
        self.assertEqual(result["summary"]["dependency_wait_count"], 1)
        self.assertTrue(result["summary"]["next_timestep_reads_committed_state"])

    def test_elapsed_timestep_gap_is_applied_once(self) -> None:
        rows = [
            completion("r0", 2, "p0", 0.0, 0),
            completion("r1", 2, "p1", 0.0, 1),
            completion("r2", 2, "p2", 0.0, 2),
        ]
        result = simulate_state_commit_fsm(
            rows,
            [spec(2)],
            initial_states={("s0", "conv1", 0): {"membrane": 0.8, "last_committed_timestep": 0}},
        )
        row = result["commit_trace"][0]
        self.assertEqual(row["elapsed_steps"], 2)
        self.assertAlmostEqual(row["membrane_after_decay"], 0.19921875)
        self.assertEqual(row["decay_count"], 1)

    def test_duplicate_partition_is_rejected(self) -> None:
        rows = [
            completion("r0", 0, "p0", 0.1, 0),
            completion("r1", 0, "p0", 0.2, 1),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate partition"):
            simulate_state_commit_fsm(rows, [spec(0)])

    def test_duplicate_completion_request_id_is_rejected(self) -> None:
        rows = [
            completion("same", 0, "p0", 0.1, 0),
            completion("same", 0, "p1", 0.2, 1),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate completion request_id"):
            simulate_state_commit_fsm(rows, [spec(0)])

    def test_unexpected_partition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected partition"):
            simulate_state_commit_fsm(
                [completion("r0", 0, "p9", 0.1, 0)], [spec(0)]
            )

    def test_output_tile_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "output tile mismatch"):
            simulate_state_commit_fsm(
                [completion("r0", 0, "p0", 0.1, 0, tile="tile1")],
                [spec(0, tile="tile0")],
            )

    def test_noncompleted_request_cannot_commit_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not completed"):
            simulate_state_commit_fsm(
                [completion("r0", 0, "p0", 0.1, 0, status="REJECTED")],
                [spec(0)],
            )

    def test_missing_partial_sum_payload_is_rejected(self) -> None:
        row = completion("r0", 0, "p0", 0.1, 0)
        del row["partial_sum_value"]
        with self.assertRaisesRegex(ValueError, "has no partial_sum_value"):
            simulate_state_commit_fsm([row], [spec(0)])

    def test_subtract_reset_mode_preserves_overshoot(self) -> None:
        rows = [
            completion("r0", 0, "p0", 1.0, 0),
            completion("r1", 0, "p1", 1.0, 1),
            completion("r2", 0, "p2", 1.0, 2),
        ]
        result = simulate_state_commit_fsm(
            rows, [spec(0)], StateCommitConfig(reset_mode="subtract")
        )
        row = result["commit_trace"][0]
        self.assertEqual(row["spike"], 1)
        self.assertEqual(row["reset_applied"], 1)
        self.assertEqual(row["membrane_after_commit"], 0.5)

    def test_invalid_or_duplicate_specs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate state spec"):
            simulate_state_commit_fsm([], [spec(0), spec(0)])
        with self.assertRaisesRegex(ValueError, "duplicate expected partitions"):
            simulate_state_commit_fsm([], [spec(0, partitions=("p0", "p0"))])
        with self.assertRaisesRegex(ValueError, "reset_mode"):
            simulate_state_commit_fsm([], [], StateCommitConfig(reset_mode="bad"))


if __name__ == "__main__":
    unittest.main()
