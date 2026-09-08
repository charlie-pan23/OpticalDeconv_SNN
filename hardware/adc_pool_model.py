"""Request-driven ADC pool model for HIPSA."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping


def get_adc_config(hardware_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    backend = hardware_cfg.get("hapr_adc_backend", hardware_cfg.get("hardware", {}).get("hapr_adc_backend", {}))
    if not isinstance(backend, Mapping):
        backend = {}
    return {
        "hapr_group_size": int(backend.get("hapr_group_size", 8)),
        "hapr_output_lanes_total": int(backend.get("hapr_output_lanes_total", 32)),
        "adc_macros": int(backend.get("adc_macros", 16)),
        "comparator_threshold_fs_default": float(backend.get("comparator_threshold_fs_default", 0.02)),
    }


def model_adc_pool(adc_requests: Mapping[str, Any], timing: Mapping[str, Any], hardware_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = get_adc_config(hardware_cfg)
    req_per_img = float(adc_requests.get("adc_requests_per_image", 0.0))
    cycles = float(timing.get("cycles_per_image", timing.get("mvm_cycles_per_image", 1.0)) or 1.0)
    demand_per_cycle = req_per_img / max(cycles, 1.0)
    macros = max(int(cfg["adc_macros"]), 1)
    util = min(1.0, demand_per_cycle / macros)
    service_cycles = req_per_img / macros
    stall_cycles = max(0.0, service_cycles - cycles)
    return {
        **cfg,
        "adc_requests_per_image": req_per_img,
        "cycles_per_image_input": cycles,
        "adc_demand_per_cycle": demand_per_cycle,
        "adc_macro_utilization": util,
        "adc_service_cycles_per_image": service_cycles,
        "adc_stall_cycles_proxy": stall_cycles,
        "adc_is_saturated": bool(demand_per_cycle >= macros),
    }


def sweep_adc_macros(adc_requests: Mapping[str, Any], timing: Mapping[str, Any], hardware_cfg: Mapping[str, Any], macro_values=(4, 8, 16, 32, 64)) -> list[Dict[str, Any]]:
    rows = []
    for m in macro_values:
        cfg = dict(hardware_cfg)
        backend = dict(cfg.get("hapr_adc_backend", {}))
        backend["adc_macros"] = int(m)
        cfg["hapr_adc_backend"] = backend
        row = model_adc_pool(adc_requests, timing, cfg)
        row["sweep_adc_macros"] = int(m)
        rows.append(row)
    return rows


def simulate_adc_queue(
    transaction_trace: Iterable[Mapping[str, Any]],
    *,
    system_clock_hz: float,
    sample_rate_gsps: float,
    adc_macros: int,
    fifo_depth: int = 4096,
) -> Dict[str, Any]:
    """Simulate an ADC request/service queue from scheduled transaction bins.

    ``sample_rate_gsps`` is converted to request tokens per system-clock cycle.
    FIFO overflow is handled as scheduler backpressure, not silent request loss:
    the affected transaction segment is followed by an explicit drain interval.
    Consequently the reported observed cycles and service latency include every
    backpressure stall.
    """

    clock = max(float(system_clock_hz), 1.0)
    service_per_cycle = max(int(adc_macros), 1) * max(float(sample_rate_gsps), 0.0) * 1.0e9 / clock
    depth = max(int(fifo_depth), 1)
    queue = 0.0
    max_queue = 0.0
    dropped = 0.0
    stall_cycles_total = 0.0
    busy_tokens = 0.0
    total_requests = 0.0
    nominal_cycles = 0
    observed_cycles = 0
    cumulative_stall = 0
    segments: list[Dict[str, Any]] = []
    weighted_queue: list[tuple[float, int]] = []

    for row in transaction_trace:
        duration = max(int(row.get("duration_cycles", 0) or 0), 0)
        arrivals = max(float(row.get("adc_requests", 0.0) or 0.0), 0.0)
        if duration <= 0:
            continue
        start = int(row.get("start_cycle", nominal_cycles) or nominal_cycles)
        end = int(row.get("end_cycle", start + duration - 1) or (start + duration - 1))
        duration = max(end - start + 1, 1)
        arrival_per_cycle = arrivals / duration
        before = queue
        net = arrival_per_cycle - service_per_cycle
        if net >= 0:
            after = queue + net * duration
            segment_max = after
        else:
            after = max(0.0, queue + net * duration)
            segment_max = queue
        overflow = max(0.0, segment_max - depth)
        # Backpressure pauses new issue until enough queued samples drain.  No
        # requests are dropped; the extra interval is part of observed latency.
        backpressure_stall = math.ceil(overflow / max(service_per_cycle, 1.0)) if overflow > 0 else 0
        stall_cycles_total += backpressure_stall
        cumulative_stall += backpressure_stall
        queue = max(0.0, after - backpressure_stall * service_per_cycle)
        visible_max = min(segment_max, float(depth))
        max_queue = max(max_queue, visible_max)
        total_requests += arrivals
        busy_tokens += min(arrivals, service_per_cycle * duration)
        nominal_start = start
        nominal_end = end
        observed_start = nominal_start + (cumulative_stall - backpressure_stall)
        observed_end = nominal_end + cumulative_stall
        nominal_cycles = max(nominal_cycles, nominal_end + 1)
        observed_cycles = max(observed_cycles, observed_end + 1)
        weighted_queue.append((visible_max, duration))
        segments.append(
            {
                "start_cycle": observed_start,
                "end_cycle": observed_end,
                "nominal_start_cycle": nominal_start,
                "nominal_end_cycle": nominal_end,
                "duration_cycles": duration,
                "layer": row.get("layer", ""),
                "timestep": row.get("timestep", ""),
                "row_tile": row.get("row_tile", ""),
                "col_tile": row.get("col_tile", ""),
                "arrival_requests": arrivals,
                "arrival_requests_per_cycle": arrival_per_cycle,
                "service_capacity_requests_per_cycle": service_per_cycle,
                "queue_start": before,
                "queue_end": queue,
                "queue_max": visible_max,
                "fifo_depth": depth,
                "overflow_requests": overflow,
                "backpressure_stall_cycles": backpressure_stall,
                "cumulative_backpressure_stall_cycles": cumulative_stall,
                "trace_granularity": "constant_rate_cycle_segment",
            }
        )

    def weighted_quantile(q: float) -> float:
        if not weighted_queue:
            return 0.0
        target = max(min(float(q), 1.0), 0.0) * sum(w for _, w in weighted_queue)
        acc = 0
        for value, weight in sorted(weighted_queue, key=lambda x: x[0]):
            acc += weight
            if acc >= target:
                return float(value)
        return float(weighted_queue[-1][0])

    utilization = busy_tokens / max(service_per_cycle * max(observed_cycles, 1), 1.0)
    service_latency = total_requests / max(service_per_cycle, 1.0) + stall_cycles_total
    return {
        "adc_macros": int(adc_macros),
        "adc_sample_rate_gsps_per_macro": float(sample_rate_gsps),
        "system_clock_hz": clock,
        "service_capacity_requests_per_cycle": service_per_cycle,
        "fifo_depth": depth,
        "requests_per_image": total_requests,
        "nominal_cycles_observed": nominal_cycles,
        "cycles_observed": observed_cycles,
        "queue_mean": sum(v * w for v, w in weighted_queue) / max(sum(w for _, w in weighted_queue), 1),
        "queue_p95": weighted_quantile(0.95),
        "queue_p99": weighted_quantile(0.99),
        "queue_max": max_queue,
        "adc_utilization": min(max(utilization, 0.0), 1.0),
        "service_latency_cycles": service_latency,
        "stall_cycles": stall_cycles_total,
        "requests_dropped": dropped,
        "segments": segments,
        "note": "Queue is simulated from transaction/HAPR arrivals; 10 GS/s is included in service capacity and backpressure is included in observed latency.",
    }
