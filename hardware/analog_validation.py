"""Analytical HAPR/TIA feasibility checks used by the HIPSA evaluation.

The HIPSA architecture is unchanged by this module.  It provides a transparent
front-end screening model so that a selected HAPR fan-in is supported by an
explicit dynamic-range/noise/settling-time budget rather than by an energy
sweep alone.  The numerical inputs live in ``configs/device_params.yaml`` and
are intentionally reported as assumptions; they are not silicon measurements.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping

from hardware.link_budget import derive_link_budget


_E_CHARGE_C = 1.602176634e-19


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _required(cfg: Mapping[str, Any], key: str) -> float:
    value = _get(cfg, key, default=None)
    if value is None:
        raise ValueError(
            "Missing device_params.yaml analog_validation field: "
            f"analog_validation.{key}"
        )
    return float(value)


def _db20(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1.0e-30))


def _db10(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1.0e-30))


def derive_optical_link_budget(device_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive received optical power from the configured source/link budget.

    The previous screen accepted ``optical_power_per_pd_mw`` as an isolated
    number.  That made the HAPR bound internally reproducible but did not show
    how the number relates to the laser budget.  This helper keeps the same
    architecture and makes the bookkeeping explicit:

    ``P_pd = P_laser,electrical * WPE / N_lanes * 10^(-L_total/10)``.

    All loss terms remain configurable assumptions.  The result is therefore
    an auditable architecture-level link budget, not a silicon measurement.
    """

    cfg = _get(device_cfg, "analog_validation", default={})
    if not isinstance(cfg, Mapping):
        raise ValueError("device_params.yaml must contain analog_validation mapping")

    link = _get(cfg, "link_budget", default={})
    if not isinstance(link, Mapping):
        link = {}

    enabled = bool(_get(link, "enabled", default=False))
    fallback_power = float(_get(cfg, "optical_power_per_pd_mw", default=0.0) or 0.0)
    if not enabled:
        return {
            "enabled": False,
            "source": "configured_optical_power_per_pd_mw",
            "received_power_per_lane_mw": fallback_power,
            "target_power_per_pd_mw": fallback_power,
            "residual_db": 0.0,
            "tolerance_db": 0.0,
            "pass": 1,
            "note": "Link budget disabled; using the legacy per-PD assumption.",
        }

    # Reuse the canonical v3/A link-budget implementation so the optical
    # power used by the HAPR screen and by eval_06 cannot drift apart.
    canonical = derive_link_budget(device_cfg)
    laser_electrical_mw = float(canonical.get("laser_electrical_power_mw", 0.0))
    wall_plug_efficiency = float(canonical.get("laser_wall_plug_efficiency", 0.2))
    optical_lanes = max(int(canonical.get("fanout_channels", 256)), 1)
    total_loss_db = float(canonical.get("path_loss_db", 0.0))
    launched_optical_mw = optical_at_source_mw = float(canonical.get("optical_power_at_source_mw", 0.0))
    pre_loss_lane_mw = launched_optical_mw / optical_lanes
    received_power_mw = float(canonical.get("received_power_per_lane_mw", 0.0))
    target_power_mw = float(canonical.get("pd_required_optical_power_mw", fallback_power))
    tolerance_db = float(canonical.get("target_tolerance_db", _get(link, "target_tolerance_db", default=0.5)))
    residual_db = float(canonical.get("residual_db", _db10(received_power_mw / max(target_power_mw, 1.0e-30))))
    normalized_losses = {
        "splitter_loss_db": float(canonical.get("splitter_loss_db", 0.0)),
        "waveguide_loss_db": float(canonical.get("waveguide_loss_db", 0.0)),
        "modulator_insertion_loss_db": float(canonical.get("modulator_insertion_loss_db", 0.0)),
        "mrr_insertion_loss_db": float(canonical.get("mrr_insertion_loss_db", 0.0)),
        "coupling_loss_db": float(canonical.get("coupling_loss_db", 0.0)),
    }

    return {
        "enabled": True,
        "source": str(canonical.get("source", "derived_from_laser_wpe_and_loss_terms")),
        "laser_electrical_power_mw": laser_electrical_mw,
        "laser_wall_plug_efficiency": wall_plug_efficiency,
        "launched_optical_power_mw": optical_at_source_mw,
        "optical_lanes_total": optical_lanes,
        "pre_loss_power_per_lane_mw": pre_loss_lane_mw,
        "loss_terms_db": normalized_losses,
        "total_loss_db": total_loss_db,
        "received_power_per_lane_mw": received_power_mw,
        "target_power_per_pd_mw": target_power_mw,
        "residual_db": residual_db,
        "tolerance_db": tolerance_db,
        "pass": int(abs(residual_db) <= tolerance_db),
        "note": str(
            _get(
                link,
                "note",
                default="Equal per-lane WDM power allocation; replace loss terms with measured/circuit link budget before publication.",
            )
        ),
    }


def derive_mrr_stabilization_budget(device_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return architecture-native fractional-lock sensitivity for HIPSA."""

    cfg = _get(device_cfg, "mrr_stabilization", default={})
    if not isinstance(cfg, Mapping):
        cfg = {}
    anchor = _get(cfg, "literature_anchor", default={})
    if not isinstance(anchor, Mapping):
        anchor = {}
    reference = _get(
        cfg,
        "reference_budget_audit",
        default=_get(cfg, "reference_budget", default={}),
    )
    if not isinstance(reference, Mapping):
        reference = {}

    ring_count = int(reference.get("physical_ring_count", 32768) or 32768)
    per_ring_mw = float(anchor.get("locking_power_mw_per_ring", 0.0) or 0.0)
    fractions = [float(value) for value in cfg.get("stress_locked_fractions", [0.0, 1.0])]
    sensitivity = [
        {"locked_fraction": fraction, "locked_rings": ring_count * fraction,
         "lock_power_mw": ring_count * fraction * per_ring_mw}
        for fraction in fractions
    ]
    representative_mw = float(reference.get("representative_nonzero_lock_budget_mw", 0.0) or 0.0)
    return {
        "physical_ring_count": ring_count,
        "locking_power_mw_per_ring": per_ring_mw,
        "locked_fraction_sensitivity": sensitivity,
        "representative_nonzero_lock_budget_mw": representative_mw,
        "headline_includes_continuous_lock": False,
        "formula": str(
            reference.get(
                "formula",
                "P_lock(f) = N_physical_rings * f_locked * P_lock_per_ring",
            )
        ),
        "status": str(
            reference.get(
                "status",
                "architecture_native_fractional_lock_sensitivity_not_measured",
            )
        ),
        "literature_source": str(anchor.get("source", "")),
        "literature_doi": str(anchor.get("doi", "")),
        "literature_url": str(anchor.get("url", "")),
        "note": str(
            reference.get(
                "note",
                "The independent 9x1 layout is not used in this derivation; replace the sensitivity with extracted thermal domains and controller measurements.",
            )
        ),
    }


_DEFAULT_CORNER_CASES = [
    {"name": "nominal", "multipliers": {}},
    {"name": "optical_high", "multipliers": {"optical_power": 1.10}},
    {"name": "responsivity_high", "multipliers": {"responsivity": 1.10}},
    {"name": "tia_gain_high", "multipliers": {"tia_transimpedance": 1.10}},
    {"name": "noise_high", "multipliers": {"tia_noise_density": 1.50, "noise_bandwidth": 1.20, "dark_current": 2.0}},
    {"name": "settling_slow", "multipliers": {"settling_time": 1.25}},
    {
        "name": "combined_worst_case",
        "multipliers": {
            "optical_power": 1.10,
            "responsivity": 1.10,
            "tia_transimpedance": 1.10,
            "tia_noise_density": 1.50,
            "noise_bandwidth": 1.20,
            "dark_current": 2.0,
            "settling_time": 1.25,
        },
    },
]


def validate_hapr_fanin(
    device_cfg: Mapping[str, Any],
    group_sizes: Iterable[int],
    *,
    adc_bits: int = 6,
    corner_multipliers: Mapping[str, float] | None = None,
    corner_name: str = "nominal",
) -> list[Dict[str, Any]]:
    """Return one feasibility row for each HAPR group size.

    The check uses a worst-case full-scale aggregate current.  A row is marked
    feasible only when all configured requirements pass:

    * TIA/ADC output remains below full scale with the requested margin;
    * analog SNR supports the requested effective ADC resolution;
    * the configured settling time fits the allowed front-end window.

    ``optical_power_per_pd_mw`` and the circuit noise numbers are explicit
    architecture-model assumptions.  The resulting report must therefore be
    described as an analytical feasibility screen until replaced with measured
    or circuit-simulated values.
    """

    cfg = _get(device_cfg, "analog_validation", default={})
    if not isinstance(cfg, Mapping):
        raise ValueError("device_params.yaml must contain analog_validation mapping")

    multipliers = {str(k): float(v) for k, v in (corner_multipliers or {}).items()}
    responsivity = _required(cfg, "photodiode_responsivity_a_per_w") * multipliers.get("responsivity", 1.0)
    link_budget = derive_optical_link_budget(device_cfg)
    optical_power_mw = float(link_budget["received_power_per_lane_mw"]) * multipliers.get("optical_power", 1.0)
    tia_z_ohm = _required(cfg, "tia_transimpedance_ohm") * multipliers.get("tia_transimpedance", 1.0)
    tia_linear_v = _required(cfg, "tia_linear_output_v") * multipliers.get("tia_linear_output", 1.0)
    adc_full_scale_v = _required(cfg, "adc_full_scale_v") * multipliers.get("adc_full_scale", 1.0)
    noise_bw_hz = _required(cfg, "effective_noise_bandwidth_hz") * multipliers.get("noise_bandwidth", 1.0)
    tia_noise_density_pa = _required(cfg, "tia_current_noise_density_pa_per_sqrt_hz") * multipliers.get("tia_noise_density", 1.0)
    dark_current_ua = _required(cfg, "photodiode_dark_current_ua") * multipliers.get("dark_current", 1.0)
    settling_time_ns = _required(cfg, "settling_time_ns") * multipliers.get("settling_time", 1.0)
    settling_limit_ns = _required(cfg, "settling_time_limit_ns")
    target_enob = _required(cfg, "target_enob_bits")
    min_margin_db = _required(cfg, "min_saturation_margin_db")
    min_snr_margin_db = _required(cfg, "min_snr_margin_db")

    # The v3 screen makes the front-end loading explicit.  These are still
    # architecture-level placeholders until a PDK extraction is available, but
    # they prevent HAPR fan-in from being justified by current alone.
    pd_cap_pf = float(_get(cfg, "pd_capacitance_pf", default=0.02))
    wire_cap_pf = float(_get(cfg, "wire_capacitance_pf", default=0.40))
    tia_input_cap_pf = float(_get(cfg, "tia_input_capacitance_pf", default=0.20))
    tia_bandwidth_floor_hz = float(_get(cfg, "tia_bandwidth_floor_hz", default=1.0e8))
    laser_error = float(_get(cfg, "worst_case_laser_error", default=0.0))
    pd_error = float(_get(cfg, "worst_case_pd_error", default=0.0))
    mrr_error = float(_get(cfg, "worst_case_mrr_error", default=0.0))
    differential_branch_factor = max(
        float(_get(cfg, "differential_branch_factor", default=2.0)),
        1.0,
    )
    apply_device_error_bounds = bool(
        _get(cfg, "apply_device_error_bounds", default=False)
    )

    optical_power_w = optical_power_mw * 1.0e-3
    dark_current_a = dark_current_ua * 1.0e-6
    tia_noise_density_a = tia_noise_density_pa * 1.0e-12
    adc_lsb_v = adc_full_scale_v / max(2 ** max(int(adc_bits), 1), 1)
    adc_target_snr_db = 6.02 * float(target_enob) + 1.76

    rows: list[Dict[str, Any]] = []
    for raw_group in group_sizes:
        group = max(int(raw_group), 1)
        lane_current_a = responsivity * optical_power_w
        aggregate_current_a = lane_current_a * group
        tia_output_v = aggregate_current_a * tia_z_ohm

        c_in_f = (group * pd_cap_pf + wire_cap_pf + tia_input_cap_pf) * 1.0e-12
        bandwidth_hz = max(
            1.0 / max(2.0 * math.pi * tia_z_ohm * c_in_f, 1.0e-30),
            tia_bandwidth_floor_hz,
        )
        effective_noise_bw_hz = min(noise_bw_hz, bandwidth_hz)
        shot_noise_a = math.sqrt(
            max(
                2.0
                * _E_CHARGE_C
                * (differential_branch_factor * aggregate_current_a + dark_current_a)
                * effective_noise_bw_hz,
                0.0,
            )
        )
        tia_noise_a = tia_noise_density_a * math.sqrt(max(effective_noise_bw_hz, 0.0))
        total_noise_a = math.sqrt(shot_noise_a**2 + tia_noise_a**2)
        snr_db = _db20(aggregate_current_a / max(total_noise_a, 1.0e-30))
        effective_enob = (snr_db - 1.76) / 6.02

        limiting_voltage_v = min(float(tia_linear_v), float(adc_full_scale_v))
        saturation_margin_db = _db20(limiting_voltage_v / max(tia_output_v, 1.0e-30))
        wc_factor = (
            (1.0 + laser_error)
            * (1.0 + pd_error)
            * (1.0 + mrr_error)
            if apply_device_error_bounds
            else 1.0
        )
        wc_current_a = aggregate_current_a * wc_factor
        wc_output_v = wc_current_a * tia_z_ohm
        saturation_margin_wc_db = _db20(
            limiting_voltage_v / max(wc_output_v, 1.0e-30)
        )
        lsb_to_noise = adc_lsb_v / max(total_noise_a * tia_z_ohm, 1.0e-30)
        snr_margin_db = snr_db - adc_target_snr_db
        settling_time_from_bw_ns = (
            2.2 / max(2.0 * math.pi * bandwidth_hz, 1.0e-30) * 1.0e9
        )
        settling_time_ns_g = max(settling_time_ns, settling_time_from_bw_ns)
        settling_margin_ns = settling_limit_ns - settling_time_ns_g

        dynamic_range_pass = saturation_margin_wc_db >= min_margin_db
        snr_pass = snr_margin_db >= min_snr_margin_db
        settling_pass = settling_margin_ns >= 0.0

        rows.append(
            {
                "hapr_group_size": group,
                "corner_name": corner_name,
                "adc_bits_checked": int(adc_bits),
                "lane_optical_power_mw": optical_power_mw,
                "lane_current_uA": lane_current_a * 1.0e6,
                "aggregate_current_mA": aggregate_current_a * 1.0e3,
                "tia_output_v": tia_output_v,
                "tia_linear_output_v": tia_linear_v,
                "adc_full_scale_v": adc_full_scale_v,
                "saturation_margin_db": saturation_margin_db,
                "worst_case_laser_error": laser_error,
                "worst_case_pd_error": pd_error,
                "worst_case_mrr_error": mrr_error,
                "worst_case_aggregate_current_mA": wc_current_a * 1.0e3,
                "worst_case_tia_output_v": wc_output_v,
                "saturation_margin_wc_db": saturation_margin_wc_db,
                "shot_noise_uA_rms": shot_noise_a * 1.0e6,
                "tia_noise_uA_rms": tia_noise_a * 1.0e6,
                "total_noise_uA_rms": total_noise_a * 1.0e6,
                "differential_branch_factor": differential_branch_factor,
                "snr_db": snr_db,
                "target_snr_db": adc_target_snr_db,
                "snr_margin_db": snr_margin_db,
                "effective_enob_bits": effective_enob,
                "adc_lsb_to_noise_ratio": lsb_to_noise,
                "input_capacitance_pf": c_in_f * 1.0e12,
                "bandwidth_hz": bandwidth_hz,
                "effective_noise_bandwidth_hz": effective_noise_bw_hz,
                "settling_time_ns": settling_time_ns_g,
                "settling_time_limit_ns": settling_limit_ns,
                "settling_margin_ns": settling_margin_ns,
                "device_error_bounds_applied": int(apply_device_error_bounds),
                "dynamic_range_pass": int(dynamic_range_pass),
                "snr_pass": int(snr_pass),
                "settling_pass": int(settling_pass),
                "feasible": int(dynamic_range_pass and snr_pass and settling_pass),
                "link_budget_source": link_budget.get("source", ""),
                "link_budget_received_power_per_lane_mw": link_budget.get("received_power_per_lane_mw", ""),
                "link_budget_residual_db": link_budget.get("residual_db", ""),
                "link_budget_pass": link_budget.get("pass", ""),
                "assumption_level": str(
                    _get(cfg, "assumption_level", default="architecture_model_assumption")
                ),
            }
        )

    return rows


def validate_hapr_corner_sweep(
    device_cfg: Mapping[str, Any],
    group_sizes: Iterable[int],
    *,
    adc_bits: int = 6,
) -> list[Dict[str, Any]]:
    """Evaluate nominal and configured worst-case analog corners.

    The sweep deliberately reports failures rather than forcing G=16 to pass.
    This makes the selected point defensible only when it survives the declared
    corner set; otherwise the paper must report the conditional limitation.
    """

    configured = _get(device_cfg, "analog_validation", "corner_cases", default=None)
    cases = configured if isinstance(configured, list) else _DEFAULT_CORNER_CASES
    rows: list[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        name = str(case.get("name", "unnamed"))
        multipliers = case.get("multipliers", {})
        if not isinstance(multipliers, Mapping):
            multipliers = {}
        rows.extend(
            validate_hapr_fanin(
                device_cfg,
                group_sizes,
                adc_bits=adc_bits,
                corner_multipliers=multipliers,
                corner_name=name,
            )
        )
    return rows
