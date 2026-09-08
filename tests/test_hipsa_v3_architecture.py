"""Pure-Python regression checks for the corrected HIPSA architecture model."""

from __future__ import annotations

import unittest

from hardware.adc_pool_model import simulate_adc_queue
from hardware.analog_validation import derive_mrr_stabilization_budget
from hardware.partial_sum_mapper import map_partial_sums, validate_no_cross_neuron_sum
from hardware.per_sample_adc_queue import replay_sample


class HipsaV3ArchitectureTests(unittest.TestCase):
    def test_g4_a8_capacity_and_a6_shortfall(self) -> None:
        rows = [{
            "seed": 0,
            "perturbation_type": "clean",
            "level_label": "0",
            "sample_index": 0,
            "timestep": 0,
            "layer": "synthetic",
            "adc_requests": 64,
            "adc_opportunities": 64,
        }]
        a8 = replay_sample(
            rows, adc_macros=8, adc_sample_rate_gsps=10, clock_ghz=1,
            fifo_depth=4096, photonic_tiles=4, outputs_per_tile=64,
            hapr_group_size=4,
        )
        a6 = replay_sample(
            rows, adc_macros=6, adc_sample_rate_gsps=10, clock_ghz=1,
            fifo_depth=4096, photonic_tiles=4, outputs_per_tile=64,
            hapr_group_size=4,
        )
        self.assertEqual(a8["post_hapr_output_lanes_total"], 64)
        self.assertTrue(a8["no_hold_admission_safe"])
        self.assertFalse(a6["no_hold_admission_safe"])
        self.assertEqual(a6["no_hold_capacity_shortfall"], 4)

    def test_hapr_capacity_scales_with_group_size(self) -> None:
        kwargs = dict(
            rows=[], adc_macros=8, adc_sample_rate_gsps=10, clock_ghz=1,
            fifo_depth=4096, photonic_tiles=4, outputs_per_tile=64,
        )
        self.assertEqual(replay_sample(hapr_group_size=4, **kwargs)["post_hapr_output_lanes_total"], 64)
        self.assertEqual(replay_sample(hapr_group_size=8, **kwargs)["post_hapr_output_lanes_total"], 32)
        self.assertEqual(replay_sample(hapr_group_size=16, **kwargs)["post_hapr_output_lanes_total"], 16)

    def test_backpressure_preserves_requests_and_latency(self) -> None:
        result = simulate_adc_queue(
            [{"start_cycle": 0, "end_cycle": 0, "duration_cycles": 1, "adc_requests": 100}],
            system_clock_hz=1e9, sample_rate_gsps=10, adc_macros=8, fifo_depth=16,
        )
        self.assertEqual(result["requests_per_image"], 100)
        self.assertEqual(result["requests_dropped"], 0)
        self.assertGreater(result["stall_cycles"], 0)
        self.assertGreater(result["cycles_observed"], result["nominal_cycles_observed"])
        self.assertGreater(result["service_latency_cycles"], result["requests_per_image"] / 80)

    def test_same_neuron_mapping_and_illegal_fanin(self) -> None:
        mapping = [
            {"layer": "synthetic", "input_dim": 256, "output_dim": 64,
             "row_tiles": 4, "col_tiles": 1, "row_tile": row_tile,
             "col_tile": 0, "mapped_physical_tile": row_tile}
            for row_tile in range(4)
        ]
        mapped = map_partial_sums(mapping, hapr_group_size=4, time_steps=1, num_tiles=4)
        self.assertTrue(validate_no_cross_neuron_sum(mapped))
        with self.assertRaises(ValueError):
            map_partial_sums(mapping, hapr_group_size=8, time_steps=1, num_tiles=4)

    def test_lock_power_reference_points(self) -> None:
        device = {
            "mrr_stabilization": {
                "literature_anchor": {"locking_power_mw_per_ring": 1.2},
                "reference_budget_audit": {"physical_ring_count": 32768},
                "stress_locked_fractions": [0.0, 0.01, 0.05, 0.10],
            }
        }
        rows = derive_mrr_stabilization_budget(device)["locked_fraction_sensitivity"]
        powers = [row["lock_power_mw"] for row in rows]
        self.assertEqual(powers, [0.0, 393.216, 1966.08, 3932.16])
        self.assertEqual(powers, sorted(powers))


if __name__ == "__main__":
    unittest.main()
