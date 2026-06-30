"""HIPSA architecture-level timing model."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def get_photonic_tile_config(hardware_cfg: Mapping[str, Any]) -> Dict[str, float]:
    tiles = hardware_cfg.get("photonic_tiles", hardware_cfg.get("hardware", {}).get("photonic_tiles", {}))
    if not isinstance(tiles, Mapping):
        tiles = {}
    peak = float(tiles.get("peak_sop_per_cycle_total", 16384.0))
    freq = float(tiles.get("effective_clock_hz", 1e9))
    util = float(tiles.get("effective_utilization", 0.40))
    realized = float(tiles.get("realized_active_rate_sop_per_s", peak * freq * util))
    return {
        "peak_sop_per_cycle_total": peak,
        "effective_clock_hz": freq,
        "effective_utilization": util,
        "realized_active_rate_sop_per_s": realized,
    }


def estimate_timing(sop_summary: Mapping[str, Any], hardware_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = get_photonic_tile_config(hardware_cfg)
    active_sop = float(sop_summary.get("active_sop_per_image", 0.0))
    dense_sop = float(sop_summary.get("dense_sop_per_image", 0.0))
    active_rate = max(float(cfg["realized_active_rate_sop_per_s"]), 1.0)
    latency_s = active_sop / active_rate if active_sop > 0 else 0.0
    cycles = latency_s * float(cfg["effective_clock_hz"])
    throughput = 1.0 / latency_s if latency_s > 0 else 0.0
    return {
        **cfg,
        "dense_sop_per_image": dense_sop,
        "active_sop_per_image": active_sop,
        "mvm_cycles_per_image": cycles,
        "cycles_per_image": cycles,
        "latency_s_per_image": latency_s,
        "latency_us_per_image": latency_s * 1e6,
        "throughput_images_per_s": throughput,
    }


def apply_adc_stall(timing: Mapping[str, Any], adc_pool: Mapping[str, Any]) -> Dict[str, Any]:
    freq = float(timing.get("effective_clock_hz", 1e9))
    base_cycles = float(timing.get("cycles_per_image", 0.0))
    stall = float(adc_pool.get("adc_stall_cycles_proxy", 0.0))
    total_cycles = base_cycles + max(0.0, stall)
    latency_s = total_cycles / max(freq, 1.0)
    out = dict(timing)
    out.update({
        "base_cycles_per_image": base_cycles,
        "adc_stall_cycles_proxy": stall,
        "cycles_per_image": total_cycles,
        "latency_s_per_image": latency_s,
        "latency_us_per_image": latency_s * 1e6,
        "throughput_images_per_s": 1.0 / latency_s if latency_s > 0 else 0.0,
    })
    return out
