"""Derive electrical laser power from an explicit optical link budget."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def derive_link_budget(device_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    # v3 exposes the link budget at the device-parameter root, while the
    # earlier HIPSA configuration keeps it under ``analog_validation``.  Read
    # both layouts so the system model can consume the source-backed A values
    # without duplicating a second, potentially inconsistent budget.
    cfg = _get(device_cfg, "link_budget", default=None)
    if not isinstance(cfg, Mapping):
        cfg = _get(device_cfg, "analog_validation", "link_budget", default={})
    if not isinstance(cfg, Mapping):
        raise ValueError("device_params.yaml must contain link_budget or analog_validation.link_budget")
    loss_terms = cfg.get("loss_terms_db")
    if isinstance(loss_terms, Mapping):
        # Accept both the A audit names and the v3 canonical component names.
        def loss(*names: str) -> float:
            for name in names:
                if name in loss_terms:
                    return _float(loss_terms.get(name))
            return 0.0

        components = {
            "splitter_loss_db": loss("splitter_fanout", "splitter_loss_db"),
            "waveguide_loss_db": loss("waveguide_routing", "waveguide_loss_db"),
            "modulator_insertion_loss_db": loss("modulator_insertion", "modulator_insertion_loss_db"),
            "mrr_insertion_loss_db": loss("mrr_weight_bank", "mrr_insertion_loss_db"),
            "coupling_loss_db": loss("photodiode_coupling", "coupling_loss_db"),
        }
    else:
        components = {
            "splitter_loss_db": _float(cfg.get("splitter_loss_db")),
            "waveguide_loss_db": _float(cfg.get("waveguide_loss_db")),
            "modulator_insertion_loss_db": _float(cfg.get("modulator_insertion_loss_db")),
            "mrr_insertion_loss_db": _float(cfg.get("mrr_insertion_loss_db")),
            "coupling_loss_db": _float(cfg.get("coupling_loss_db")),
        }
    path_loss_db = sum(components.values())
    pd_power_mw = _float(
        cfg.get("pd_required_optical_power_mw", cfg.get("target_power_per_pd_mw", 0.25)),
        0.25,
    )
    channels = max(int(cfg.get("fanout_channels", cfg.get("optical_lanes_total", 1))), 1)
    wpe = _float(cfg.get("laser_wall_plug_efficiency"), 0.2)
    if not 0.0 < wpe <= 1.0:
        raise ValueError("laser_wall_plug_efficiency must be in (0,1]")

    # The A revision already fixes the main-case laser at 1473 mW and checks
    # that it delivers approximately 0.25 mW per photodetector.  Preserve
    # that nominal operating point in the paper-facing run; use the v3
    # inverse formula as a consistency audit only when no explicit source
    # power is supplied (the native v3 configuration layout).
    explicit_laser_mw = cfg.get("laser_electrical_power_mw")
    use_forward_main_case = (
        explicit_laser_mw is not None
        and cfg.get("pd_required_optical_power_mw") is None
    )
    if use_forward_main_case:
        electrical_mw = _float(explicit_laser_mw)
        optical_at_source_mw = electrical_mw * wpe
        received_power_mw = optical_at_source_mw / channels * (10.0 ** (-path_loss_db / 10.0))
        residual_db = 10.0 * math.log10(max(received_power_mw, 1.0e-30) / max(pd_power_mw, 1.0e-30))
        derivation_mode = "forward_from_configured_laser_with_inverse_audit"
        formula = "P_PD = P_laser_elec * WPE / N_channels * 10^(-L_path/10)"
    else:
        optical_at_source_mw = pd_power_mw * channels * (10.0 ** (path_loss_db / 10.0))
        electrical_mw = optical_at_source_mw / wpe
        received_power_mw = pd_power_mw
        residual_db = 0.0
        derivation_mode = "inverse_from_required_pd_power"
        formula = "P_laser_elec = P_PD_req * 10^(L_path/10) * N_channels / WPE"
    tolerance_db = _float(cfg.get("target_tolerance_db"), 0.5)
    return {
        "pd_required_optical_power_mw": pd_power_mw,
        "fanout_channels": channels,
        "laser_wall_plug_efficiency": wpe,
        **components,
        "path_loss_db": path_loss_db,
        "optical_power_at_source_mw": optical_at_source_mw,
        "received_power_per_lane_mw": received_power_mw,
        "laser_electrical_power_mw": electrical_mw,
        "target_tolerance_db": tolerance_db,
        "residual_db": residual_db,
        "pass": int(abs(residual_db) <= tolerance_db),
        "source": "A_device_params_with_v3_link_budget_audit" if use_forward_main_case else "v3_link_budget_model",
        "derivation_mode": derivation_mode,
        "derived": True,
        "formula": formula,
    }
