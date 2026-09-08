"""Focused Phase 1-D ADC evidence-boundary tests."""

from __future__ import annotations

import unittest

from hardware.adc_evidence_boundary import (
    ADCEvidenceConfig,
    build_adc_evidence_boundary,
    capacity_sensitivity,
)


class ADCEvidenceBoundaryTests(unittest.TestCase):
    def test_selected_point_capacity_and_claim_boundary(self) -> None:
        evidence = build_adc_evidence_boundary(ADCEvidenceConfig())
        self.assertEqual(evidence["nominal_slots_per_window"], 80)
        self.assertEqual(evidence["capacity_margin_slots"], 16)
        self.assertFalse(evidence["timing_closure"])
        self.assertFalse(evidence["physically_closed"])
        self.assertFalse(evidence["silicon_validated"])
        self.assertIn("timing closure", evidence["forbidden_claims"])

    def test_unclosed_physical_items_are_explicit(self) -> None:
        evidence = build_adc_evidence_boundary()
        items = {item["name"]: item for item in evidence["items"]}
        for name in ("mux_switching_settling", "adc_aperture", "tia_output_recovery"):
            self.assertEqual(items[name]["status"], "not_modeled")
            self.assertFalse(items[name]["modeled"])
        self.assertEqual(items["conversion_latency"]["status"], "modeled_pipeline_semantics")
        self.assertEqual(items["adc_power_14_8_mw_per_macro"]["status"], "conditional_parameter_anchor")

    def test_capacity_sensitivity_contains_g4_a8_and_a6_points(self) -> None:
        rows = capacity_sensitivity()
        points = {(row["adc_macros"], row["samples_per_macro_per_cycle"]): row for row in rows}
        self.assertEqual(points[(8, 10)]["nominal_slots_per_window"], 80)
        self.assertTrue(points[(8, 10)]["same_window_admission_feasible"])
        self.assertEqual(points[(6, 10)]["nominal_slots_per_window"], 60)
        self.assertFalse(points[(6, 10)]["same_window_admission_feasible"])


if __name__ == "__main__":
    unittest.main()
