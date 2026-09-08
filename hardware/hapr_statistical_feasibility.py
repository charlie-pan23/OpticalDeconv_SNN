"""Trace-driven, parasitic-aware statistical HAPR feasibility model.

This is an architecture-level physical screen. It is not transistor-level
SPICE, foundry Monte Carlo, or a HIPSA silicon measurement.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

E_CHARGE = 1.602176634e-19


def run_joint_monte_carlo(
    trace: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    *,
    samples: int = 10_000,
    seed: int = 42,
    capacitance_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Sample workload operations and declared independent variation sources."""
    rng = np.random.default_rng(seed)
    n_trace = len(trace["fan_in"])
    if n_trace == 0:
        raise ValueError("HAPR trace is empty")
    pick = rng.integers(0, n_trace, size=int(samples))
    fan_in = np.asarray(trace["fan_in"], dtype=float)[pick]
    i_pos_ideal = np.asarray(trace["i_pos_a"], dtype=float)[pick]
    i_neg_ideal = np.asarray(trace["i_neg_a"], dtype=float)[pick]

    def normal(name: str, default: float) -> np.ndarray:
        return rng.normal(1.0, float(cfg.get(name, default)), size=samples)

    optical_common = normal("laser_sigma", 0.01)
    mrr_pos = normal("mrr_sigma", 0.02)
    mrr_neg = normal("mrr_sigma", 0.02)
    responsivity_pos = normal("pd_responsivity_sigma", 0.03)
    responsivity_neg = normal("pd_responsivity_sigma", 0.03)
    branch_pos = normal("branch_mismatch_sigma", 0.01)
    branch_neg = normal("branch_mismatch_sigma", 0.01)
    tia_gain = float(cfg["tia_transimpedance_ohm"]) * normal("tia_gain_sigma", 0.01)
    adc_fs = float(cfg["adc_full_scale_v"]) * normal("adc_full_scale_sigma", 0.01)

    i_pos = i_pos_ideal * optical_common * mrr_pos * responsivity_pos * branch_pos
    i_neg = i_neg_ideal * optical_common * mrr_neg * responsivity_neg * branch_neg
    ideal = i_pos_ideal - i_neg_ideal
    nonideal = i_pos - i_neg
    vout = nonideal * tia_gain

    bandwidth = float(cfg["effective_noise_bandwidth_hz"])
    dark = float(cfg["photodiode_dark_current_ua"]) * 1e-6
    tia_density = float(cfg["tia_current_noise_density_pa_per_sqrt_hz"]) * 1e-12
    shot_rms = np.sqrt(2.0 * E_CHARGE * (np.abs(i_pos) + np.abs(i_neg) + fan_in * dark) * bandwidth)
    tia_rms = tia_density * math.sqrt(bandwidth)
    noise_v_rms = np.sqrt(shot_rms**2 + tia_rms**2) * tia_gain
    snr_db_all = 20.0 * np.log10(np.maximum(np.abs(vout), 1e-18) / np.maximum(noise_v_rms, 1e-18))
    adc6_required_db = 6.02 * 6 + 1.76

    c_total = (
        float(cfg["pd_capacitance_pf"]) * fan_in
        + float(cfg["wire_capacitance_pf"])
        + float(cfg["tia_input_capacitance_pf"])
    ) * 1e-12 * float(capacitance_multiplier)
    r_eq = float(cfg.get("settling_equivalent_resistance_ohm", 200.0))
    epsilon = float(cfg.get("settling_error_fraction", 1.0 / 128.0))
    settling_s = -r_eq * c_total * math.log(epsilon)

    signal_floor = float(cfg.get("mismatch_denominator_floor_a", 1e-9))
    signal_mask = np.abs(ideal) > signal_floor
    # Relative error and SNR are undefined for zero/near-zero ideal differential
    # sums. Report their fraction explicitly and calculate signal-quality
    # percentiles only on operations with a resolvable ideal signal.
    if not np.any(signal_mask):
        raise ValueError("No nonzero differential partial sums above the declared signal floor")
    snr_db = snr_db_all[signal_mask]
    normalized_error = np.abs(nonideal[signal_mask] - ideal[signal_mask]) / np.abs(ideal[signal_mask])
    saturated = np.abs(vout) >= np.abs(adc_fs)
    settle_limit_s = float(cfg["settling_time_limit_ns"]) * 1e-9
    return {
        "samples": int(samples),
        "seed": int(seed),
        "capacitance_multiplier": float(capacitance_multiplier),
        "saturation_count": int(saturated.sum()),
        "saturation_probability": float(saturated.mean()),
        "voltage_abs_p50_v": float(np.percentile(np.abs(vout), 50)),
        "voltage_abs_p95_v": float(np.percentile(np.abs(vout), 95)),
        "voltage_abs_p99_v": float(np.percentile(np.abs(vout), 99)),
        "voltage_abs_max_v": float(np.max(np.abs(vout))),
        "signal_floor_a": signal_floor,
        "nonzero_signal_count": int(signal_mask.sum()),
        "nonzero_signal_fraction": float(signal_mask.mean()),
        "zero_or_near_zero_signal_fraction": float((~signal_mask).mean()),
        "zero_signal_noise_v_rms_p95": float(np.percentile(noise_v_rms[~signal_mask], 95)) if np.any(~signal_mask) else 0.0,
        "snr_db_p01": float(np.percentile(snr_db, 1)),
        "snr_db_p05": float(np.percentile(snr_db, 5)),
        "snr_db_p50": float(np.percentile(snr_db, 50)),
        "frontend_precision_margin_db_p01": float(np.percentile(snr_db - adc6_required_db, 1)),
        "frontend_precision_margin_db_p50": float(np.percentile(snr_db - adc6_required_db, 50)),
        "adc6_snr_pass_fraction_nonzero": float(np.mean(snr_db >= adc6_required_db)),
        "settling_ns_p99": float(np.percentile(settling_s, 99) * 1e9),
        "settling_ns_max": float(np.max(settling_s) * 1e9),
        "settling_violation_probability": float(np.mean(settling_s >= settle_limit_s)),
        "normalized_accumulation_error_p50": float(np.percentile(normalized_error, 50)),
        "normalized_accumulation_error_p95": float(np.percentile(normalized_error, 95)),
        "normalized_accumulation_error_p99": float(np.percentile(normalized_error, 99)),
        "relative_metrics_population": "nonzero_ideal_differential_partial_sums_only",
        "evidence_scope": "trace_driven_parasitic_aware_statistical_feasibility_not_spice_not_foundry_mc",
    }
