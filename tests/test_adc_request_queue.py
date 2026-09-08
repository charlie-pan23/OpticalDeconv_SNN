"""Tests for the HIPSA request-level ADC queue reference model."""

from __future__ import annotations

import unittest

from hardware.adc_request_queue import ADCQueueConfig, ADCRequest, simulate_request_queue


def make_request(
    index: int,
    *,
    arrival: int = 0,
    sample: str = "s0",
    timestep: int = 0,
    layer: str = "conv1",
    tile: str = "c0",
    neuron: int | None = None,
    partition: int | None = None,
    group: str | None = None,
    latency: int | None = None,
) -> ADCRequest:
    neuron_id = index if neuron is None else neuron
    partition_id = index if partition is None else partition
    return ADCRequest(
        request_id=f"r{index}",
        sample_id=sample,
        timestep_id=timestep,
        layer_id=layer,
        output_tile_id=tile,
        output_neuron_id=neuron_id,
        partition_id=partition_id,
        hapr_group_id=group or f"{layer}:j{neuron_id}:g0",
        arrival_cycle=arrival,
        conversion_latency_cycles=latency,
    )


class ADCRequestQueueTests(unittest.TestCase):
    def test_g4_a8_admits_full_64_lane_window(self) -> None:
        result = simulate_request_queue(
            [make_request(i) for i in range(64)],
            ADCQueueConfig(adc_macros=8, samples_per_macro_per_cycle=10),
        )
        summary = result["summary"]
        self.assertEqual(summary["service_capacity_per_cycle"], 80)
        self.assertTrue(summary["same_window_admission_feasible"])
        self.assertEqual(summary["arrivals"], 64)
        self.assertEqual(summary["accepted"], 64)
        self.assertEqual(summary["served"], 64)
        self.assertEqual(summary["rejected"], 0)
        self.assertEqual(summary["requests_dropped"], 0)
        self.assertEqual(summary["final_fifo_occupancy"], 0)
        self.assertTrue(summary["tail_drain_complete"])
        self.assertTrue(summary["selected_point_valid"])

    def test_a6_exposes_four_request_same_window_shortfall(self) -> None:
        result = simulate_request_queue(
            [make_request(i) for i in range(64)],
            ADCQueueConfig(adc_macros=6, samples_per_macro_per_cycle=10),
        )
        summary = result["summary"]
        self.assertEqual(summary["service_capacity_per_cycle"], 60)
        self.assertFalse(summary["same_window_admission_feasible"])
        self.assertEqual(summary["same_window_capacity_shortfall"], 4)
        self.assertEqual(summary["accepted"], 60)
        self.assertEqual(summary["rejected"], 4)
        self.assertFalse(summary["selected_point_valid"])
        reasons = {
            row["rejection_reason"]
            for row in result["requests"]
            if row["status"] == "REJECTED"
        }
        self.assertEqual(reasons, {"same_window_capacity_exceeded"})

    def test_fifo_overflow_is_explicit_rejection_not_silent_drop(self) -> None:
        result = simulate_request_queue(
            [make_request(i) for i in range(4)],
            ADCQueueConfig(fifo_depth=2),
        )
        summary = result["summary"]
        self.assertEqual(summary["arrivals"], 4)
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["rejected"], 2)
        self.assertEqual(summary["requests_dropped"], 0)
        self.assertTrue(summary["arrival_conservation_valid"])
        self.assertEqual(summary["unaccounted_requests"], 0)

    def test_conversion_latency_is_pipelined_and_tail_is_drained(self) -> None:
        requests = [
            make_request(index + cycle * 80, arrival=cycle, latency=3)
            for cycle in range(2)
            for index in range(80)
        ]
        result = simulate_request_queue(requests, ADCQueueConfig())
        summary = result["summary"]
        self.assertEqual(summary["accepted"], 160)
        self.assertEqual(summary["served"], 160)
        self.assertEqual(summary["rejected"], 0)
        self.assertEqual(summary["last_arrival_cycle"], 1)
        self.assertEqual(summary["last_observed_cycle"], 4)
        self.assertTrue(summary["tail_drain_complete"])
        starts = [row["conversion_starts"] for row in result["cycle_trace"][:2]]
        self.assertEqual(starts, [80, 80])

    def test_per_request_latency_can_reorder_completion(self) -> None:
        result = simulate_request_queue(
            [make_request(0, latency=3), make_request(1, latency=1)],
            ADCQueueConfig(),
        )
        requests = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(requests["r0"]["admission_sequence"], 0)
        self.assertEqual(requests["r1"]["admission_sequence"], 1)
        self.assertEqual(requests["r1"]["completion_sequence"], 0)
        self.assertEqual(requests["r0"]["completion_sequence"], 1)
        self.assertTrue(result["summary"]["completion_reordered_vs_admission"])

    def test_temporal_storage_stress_mode_can_defer_requests(self) -> None:
        result = simulate_request_queue(
            [make_request(i) for i in range(5)],
            ADCQueueConfig(
                adc_macros=1,
                samples_per_macro_per_cycle=2,
                fifo_depth=5,
                post_hapr_lanes=2,
                temporal_analog_storage=True,
            ),
        )
        summary = result["summary"]
        self.assertEqual(summary["accepted"], 5)
        self.assertEqual(summary["served"], 5)
        self.assertEqual(summary["rejected"], 0)
        self.assertEqual(summary["queue_max"], 3)
        self.assertTrue(summary["tail_drain_complete"])
        starts = {
            row["request_id"]: row["conversion_start_cycle"]
            for row in result["requests"]
        }
        self.assertEqual(starts, {"r0": 0, "r1": 0, "r2": 1, "r3": 1, "r4": 2})

    def test_idle_gap_preserves_absolute_cycles(self) -> None:
        result = simulate_request_queue(
            [make_request(0, arrival=0), make_request(1, arrival=5)],
            ADCQueueConfig(),
        )
        requests = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(requests["r0"]["completion_cycle"], 1)
        self.assertEqual(requests["r1"]["conversion_start_cycle"], 5)
        self.assertEqual(requests["r1"]["completion_cycle"], 6)
        self.assertEqual(result["summary"]["last_observed_cycle"], 6)

    def test_no_tail_drain_keeps_inflight_requests_accounted(self) -> None:
        result = simulate_request_queue(
            [make_request(0, latency=3)],
            ADCQueueConfig(drain_tail=False),
        )
        summary = result["summary"]
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["served"], 0)
        self.assertEqual(summary["final_fifo_occupancy"], 1)
        self.assertEqual(summary["final_conversion_inflight"], 1)
        self.assertTrue(summary["accepted_conservation_valid"])
        self.assertFalse(summary["tail_drain_complete"])

    def test_duplicate_request_id_is_rejected(self) -> None:
        requests = [make_request(0), make_request(0, neuron=1, group="conv1:j1:g0")]
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            simulate_request_queue(requests, ADCQueueConfig())

    def test_hapr_group_cannot_cross_neuron_or_output_tile(self) -> None:
        requests = [
            make_request(0, neuron=0, group="shared"),
            make_request(1, neuron=1, group="shared"),
        ]
        with self.assertRaisesRegex(ValueError, "crosses output tile/neuron boundary"):
            simulate_request_queue(requests, ADCQueueConfig())

    def test_same_group_label_may_repeat_in_a_different_timestep_context(self) -> None:
        requests = [
            make_request(0, timestep=0, neuron=0, group="conv1:j0:g0"),
            make_request(1, timestep=1, neuron=0, group="conv1:j0:g0"),
        ]
        result = simulate_request_queue(requests, ADCQueueConfig())
        self.assertEqual(result["summary"]["served"], 2)
        self.assertTrue(result["summary"]["arrival_conservation_valid"])


    def test_output_rows_expose_required_transaction_schema(self) -> None:
        result = simulate_request_queue([make_request(0)], ADCQueueConfig())
        row = result["requests"][0]
        required_fields = {
            "request_id",
            "sample_id",
            "timestep_id",
            "layer_id",
            "output_tile_id",
            "output_neuron_id",
            "partition_id",
            "hapr_group_id",
            "arrival_cycle",
            "admission_cycle",
            "conversion_start_cycle",
            "conversion_done_cycle",
            "completion_cycle",
            "adc_macro_id",
            "adc_slot_id",
            "admission_sequence",
            "completion_sequence",
            "status",
            "rejection_reason",
        }
        self.assertTrue(required_fields.issubset(row))
        self.assertEqual(row["status"], "COMPLETED")

    def test_interleaved_transactions_preserve_identity_fields(self) -> None:
        requests = [
            make_request(
                0,
                arrival=1,
                sample="sample-b",
                timestep=2,
                layer="fc",
                tile="tile-1",
                neuron=11,
                partition=4,
                group="fc:j11:g0",
                latency=1,
            ),
            make_request(
                1,
                arrival=0,
                sample="sample-a",
                timestep=3,
                layer="conv2",
                tile="tile-0",
                neuron=7,
                partition=9,
                group="conv2:j7:g1",
                latency=3,
            ),
            make_request(
                2,
                arrival=0,
                sample="sample-b",
                timestep=2,
                layer="fc",
                tile="tile-1",
                neuron=12,
                partition=5,
                group="fc:j12:g0",
                latency=1,
            ),
        ]
        expected = {
            request.request_id: {
                "sample_id": request.sample_id,
                "timestep_id": request.timestep_id,
                "layer_id": request.layer_id,
                "output_tile_id": request.output_tile_id,
                "output_neuron_id": request.output_neuron_id,
                "partition_id": request.partition_id,
                "hapr_group_id": request.hapr_group_id,
                "arrival_cycle": request.arrival_cycle,
            }
            for request in requests
        }
        result = simulate_request_queue(requests, ADCQueueConfig())
        actual = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(set(actual), set(expected))
        for request_id, identity in expected.items():
            for field, value in identity.items():
                self.assertEqual(actual[request_id][field], value)
            self.assertEqual(actual[request_id]["status"], "COMPLETED")
        self.assertTrue(result["summary"]["completion_reordered_vs_admission"])

    def test_temporal_storage_stress_mode_is_not_selected_point(self) -> None:
        result = simulate_request_queue(
            [make_request(i) for i in range(5)],
            ADCQueueConfig(
                adc_macros=1,
                samples_per_macro_per_cycle=2,
                fifo_depth=5,
                post_hapr_lanes=2,
                temporal_analog_storage=True,
            ),
        )
        summary = result["summary"]
        self.assertTrue(summary["same_window_admission_feasible"])
        self.assertTrue(summary["temporal_analog_storage"])
        self.assertFalse(summary["selected_point_valid"])
        self.assertFalse(summary["circuit_timing_closed"])

    def test_empty_input_and_invalid_config_are_explicit(self) -> None:
        result = simulate_request_queue([], ADCQueueConfig())
        summary = result["summary"]
        self.assertEqual(summary["arrivals"], 0)
        self.assertEqual(summary["served"], 0)
        self.assertEqual(result["requests"], [])
        self.assertEqual(result["cycle_trace"], [])
        self.assertTrue(summary["arrival_conservation_valid"])
        self.assertTrue(summary["accepted_conservation_valid"])
        with self.assertRaisesRegex(ValueError, "adc_macros must be positive"):
            simulate_request_queue([], ADCQueueConfig(adc_macros=0))


if __name__ == "__main__":
    unittest.main()
