"""Explicit MRR programming latency, energy, and SRAM traffic model."""

from __future__ import annotations

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


def model_mrr_configuration(
    weight_map: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return per-inference configuration overhead under tile-major residency."""

    num_tiles = max(int(weight_map.get("num_tiles", 1)), 1)
    rounds = int(weight_map.get("total_physical_tile_load_rounds", 0))
    physical = int(weight_map.get("total_physical_mrr_elements", 0))
    bits = int(weight_map.get("weight_bits", 6))
    clock_hz = _float(_get(hardware_cfg, "photonic_tiles", "effective_clock_hz", default=1e9), 1e9)
    cycles_per_round = _float(
        _get(hardware_cfg, "mrr_configuration", "cycles_per_tile_load", default=256),
        256,
    )
    energy_per_ring_pj = _float(
        _get(device_cfg, "mrr_configuration", "energy_per_physical_ring_pj", default=0.25),
        0.25,
    )
    sram_read_energy_pj_per_bit = _float(
        _get(device_cfg, "mrr_configuration", "sram_read_energy_pj_per_bit", default=0.02),
        0.02,
    )
    cycles = rounds * cycles_per_round
    config_energy_pj = physical * energy_per_ring_pj
    sram_bits = physical * bits
    config_energy_pj += sram_bits * sram_read_energy_pj_per_bit
    return {
        "weight_residency_mode": "layer_weight_tile_major",
        "mrr_reconfiguration_events_per_image": rounds,
        "mrr_reconfiguration_cycles_per_image": cycles,
        "mrr_reconfiguration_time_us_per_image": cycles / max(clock_hz, 1.0) * 1e6,
        "physical_mrr_elements_configured_per_image": physical,
        "weight_sram_read_bits_per_image": sram_bits,
        "configuration_energy_pj_per_image": config_energy_pj,
        "configuration_energy_nj_per_image": config_energy_pj / 1e3,
        "cycles_per_tile_load": cycles_per_round,
        "mrr_retuned_every_timestep": False,
        "note": "Weights are configured at layer/weight-tile boundaries and reused across all positions and timesteps.",
    }
