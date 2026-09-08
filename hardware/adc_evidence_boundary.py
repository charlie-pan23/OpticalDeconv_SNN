"""Evidence boundary for the frozen 10 GS/s ADC architecture model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ADCEvidenceConfig:
    adc_macros: int = 8
    samples_per_macro_per_cycle: int = 10
    post_hapr_lanes: int = 64
    architecture_clock_ghz: float = 1.0
    adc_power_mw_per_macro: float = 14.8
    temporal_analog_storage: bool = False
    mux_ratio: str = "64:8"

    @property
    def nominal_slots_per_window(self) -> int:
        return int(self.adc_macros) * int(self.samples_per_macro_per_cycle)

    def validate(self) -> None:
        if int(self.adc_macros) <= 0:
            raise ValueError("adc_macros must be positive")
        if int(self.samples_per_macro_per_cycle) <= 0:
            raise ValueError("samples_per_macro_per_cycle must be positive")
        if int(self.post_hapr_lanes) <= 0:
            raise ValueError("post_hapr_lanes must be positive")
        if float(self.architecture_clock_ghz) <= 0:
            raise ValueError("architecture_clock_ghz must be positive")
        if float(self.adc_power_mw_per_macro) < 0:
            raise ValueError("adc_power_mw_per_macro must be non-negative")


@dataclass(frozen=True)
class EvidenceItem:
    name: str
    status: str
    modeled: bool
    affects_admission_capacity: bool
    affects_physical_timing: bool
    affects_power_applicability: bool
    declared_input: str
    model_treatment: str
    required_evidence_to_close: str
    safe_claim: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_adc_evidence_boundary(
    config: ADCEvidenceConfig = ADCEvidenceConfig(),
) -> dict[str, Any]:
    """Return a machine-readable claim boundary for the 10 GS/s model."""

    config.validate()
    items = [
        EvidenceItem(
            name="post_hapr_lane_count",
            status="modeled_architecture_capacity",
            modeled=True,
            affects_admission_capacity=True,
            affects_physical_timing=False,
            affects_power_applicability=False,
            declared_input=f"{config.post_hapr_lanes} post-HAPR lanes",
            model_treatment="Used as the number of analog requests that must be admitted per service window.",
            required_evidence_to_close="None for the logical count; provenance of HAPR mapping must remain explicit.",
            safe_claim="architecture-level lane/admission accounting",
        ),
        EvidenceItem(
            name="input_multiplexing_64_to_8",
            status="capacity_assumption_only",
            modeled=False,
            affects_admission_capacity=True,
            affects_physical_timing=True,
            affects_power_applicability=True,
            declared_input=config.mux_ratio,
            model_treatment="The queue uses the resulting nominal ADC slot capacity; mux fan-in, switching, and settling are not simulated.",
            required_evidence_to_close="Transient mux/TIA simulation or measured settling and recovery under the declared 1 ns window.",
            safe_claim="capacity assumption; not mux timing closure",
        ),
        EvidenceItem(
            name="mux_switching_settling",
            status="not_modeled",
            modeled=False,
            affects_admission_capacity=True,
            affects_physical_timing=True,
            affects_power_applicability=False,
            declared_input="not supplied",
            model_treatment="No settling penalty or usable-aperture reduction is applied.",
            required_evidence_to_close="Circuit-level settling characterization with worst-case load and multiplexing schedule.",
            safe_claim="not demonstrated",
        ),
        EvidenceItem(
            name="adc_aperture",
            status="not_modeled",
            modeled=False,
            affects_admission_capacity=True,
            affects_physical_timing=True,
            affects_power_applicability=False,
            declared_input="not supplied",
            model_treatment="Conversion latency is modeled as a pipeline completion delay, not as an aperture/track-and-hold waveform.",
            required_evidence_to_close="ADC sampling aperture and track/hold timing under the muxed input waveform.",
            safe_claim="not demonstrated",
        ),
        EvidenceItem(
            name="tia_output_recovery",
            status="not_modeled",
            modeled=False,
            affects_admission_capacity=True,
            affects_physical_timing=True,
            affects_power_applicability=False,
            declared_input="not supplied",
            model_treatment="No TIA recovery or inter-request interference penalty is included.",
            required_evidence_to_close="TIA transient/recovery simulation or measurement at the declared request rate.",
            safe_claim="not demonstrated",
        ),
        EvidenceItem(
            name="hapr_analog_hold_behavior",
            status="no_temporal_storage_selected_point",
            modeled=True,
            affects_admission_capacity=True,
            affects_physical_timing=True,
            affects_power_applicability=False,
            declared_input="temporal_analog_storage=false",
            model_treatment="Requests that cannot start conversion in the arrival window are rejected; a digital FIFO does not rescue the analog sample.",
            required_evidence_to_close="Front-end hold/settling evidence if the claim is strengthened beyond architecture-level admission.",
            safe_claim="no temporal analog storage in the selected-point model",
        ),
        EvidenceItem(
            name="conversion_latency",
            status="modeled_pipeline_semantics",
            modeled=True,
            affects_admission_capacity=True,
            affects_physical_timing=False,
            affects_power_applicability=False,
            declared_input="per-request/default conversion latency cycles",
            model_treatment="Latency delays completion while declared pipelined issue capacity remains unchanged.",
            required_evidence_to_close="ADC pipeline implementation or a measured latency/throughput pair if making silicon claims.",
            safe_claim="architecture-level pipelined conversion semantics",
        ),
        EvidenceItem(
            name="adc_power_14_8_mw_per_macro",
            status="conditional_parameter_anchor",
            modeled=True,
            affects_admission_capacity=False,
            affects_physical_timing=False,
            affects_power_applicability=True,
            declared_input=f"{config.adc_power_mw_per_macro:g} mW per ADC macro",
            model_treatment="Available as a declared power parameter; applicability to the exact muxed operating mode is not established by this model.",
            required_evidence_to_close="Operating-mode-matched ADC power data including muxing, aperture, and throughput conditions.",
            safe_claim="parameterized power anchor, not demonstrated operating-mode measurement",
        ),
    ]
    return {
        "config": asdict(config),
        "nominal_slots_per_window": config.nominal_slots_per_window,
        "capacity_margin_slots": config.nominal_slots_per_window - config.post_hapr_lanes,
        "items": [item.to_dict() for item in items],
        "model_scope": "architecture_level_capacity_admission_and_transaction_semantics",
        "timing_closure": False,
        "physically_closed": False,
        "silicon_validated": False,
        "safe_claim": "architecture-level capacity/admission analysis",
        "forbidden_claims": ["timing closure", "physically closed", "silicon validated"],
        "sensitivity_axes": {
            "adc_macros": [6, 8],
            "samples_per_macro_per_cycle": [8, 10, 12],
            "post_hapr_lanes": [64],
            "adc_power_mw_per_macro": [14.8],
        },
        "claim_boundary": (
            "mux_settling_adc_aperture_tia_recovery_and_operating_mode_power_are_not_closed"
        ),
    }


def capacity_sensitivity(
    *,
    adc_macros: Iterable[int] = (6, 8),
    samples_per_macro_per_cycle: Iterable[int] = (8, 10, 12),
    post_hapr_lanes: int = 64,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for macros in adc_macros:
        for samples in samples_per_macro_per_cycle:
            capacity = int(macros) * int(samples)
            rows.append(
                {
                    "adc_macros": int(macros),
                    "samples_per_macro_per_cycle": int(samples),
                    "post_hapr_lanes": int(post_hapr_lanes),
                    "nominal_slots_per_window": capacity,
                    "capacity_margin_slots": capacity - int(post_hapr_lanes),
                    "same_window_admission_feasible": capacity >= int(post_hapr_lanes),
                    "evidence_class": "architecture_level_capacity_only",
                }
            )
    return rows
