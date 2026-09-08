"""Integer-cycle ADC queue replay from per-sample/layer/timestep request bins."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Iterable, Mapping


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(q)
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - position) + ordered[hi] * (position - lo))


def distribute_integer(total: int, cycles: int) -> list[int]:
    """Even deterministic placement; preserves the exact integer request total."""
    cycles = max(int(cycles), 1)
    total = max(int(total), 0)
    base, remainder = divmod(total, cycles)
    return [base + (1 if i < remainder else 0) for i in range(cycles)]


def replay_sample(
    rows: Iterable[Mapping[str, Any]],
    *,
    adc_macros: int,
    adc_sample_rate_gsps: float,
    clock_ghz: float,
    fifo_depth: int,
    photonic_tiles: int,
    outputs_per_tile: int,
    hapr_group_size: int = 4,
) -> dict[str, Any]:
    service_per_cycle = max(int(math.floor(adc_macros * adc_sample_rate_gsps / clock_ghz)), 1)
    # HIPSA does not provide a multi-cycle analog request queue. The TIA/HAPR
    # outputs may be held only inside the current architecture service window
    # while the ADC input mux consumes them. Admission must therefore be
    # provisioned for the full post-HAPR lane burst, not just trace-average
    # queue depth.
    physical_tiles = max(int(photonic_tiles), 1)
    tile_outputs = max(int(outputs_per_tile), 1)
    group_size = max(int(hapr_group_size), 1)
    post_hapr_lanes = max(int(math.ceil(physical_tiles * tile_outputs / group_size)), 1)
    no_hold_admission_safe = service_per_cycle >= post_hapr_lanes
    queue = 0
    depth_runs: list[tuple[float, int]] = []
    stalls = 0
    dropped = 0
    cycles = 0
    arrivals_total = 0

    def record_depth(value: float, count: int = 1) -> None:
        if count <= 0:
            return
        if depth_runs and depth_runs[-1][0] == value:
            depth_runs[-1] = (value, depth_runs[-1][1] + count)
        else:
            depth_runs.append((value, count))

    def advance_constant(arrivals: int, count: int) -> None:
        """Apply an exact constant-arrival queue recurrence."""
        nonlocal queue, stalls, dropped, cycles, arrivals_total
        while count > 0:
            if queue <= service_per_cycle and arrivals <= service_per_cycle:
                arrivals_total += arrivals * count
                queue = arrivals
                record_depth(float(queue), count)
                cycles += count
                return
            if queue == fifo_depth and arrivals > service_per_cycle:
                arrivals_total += arrivals * count
                dropped += (arrivals - service_per_cycle) * count
                stalls += count
                record_depth(float(fifo_depth), count)
                cycles += count
                return
            arrivals_total += arrivals
            served = min(queue, service_per_cycle)
            residual = queue - served + arrivals
            overflow = max(residual - fifo_depth, 0)
            dropped += overflow
            queue = min(residual, fifo_depth)
            if queue >= max(fifo_depth - 64, 0) or overflow > 0:
                stalls += 1
            record_depth(float(queue))
            cycles += 1
            count -= 1

    for row in sorted(rows, key=lambda r: (int(r.get("timestep", 0)), str(r.get("layer", "")))):
        requests = max(int(row.get("adc_requests", 0)), 0)
        opportunities = max(int(row.get("adc_opportunities", requests)), 1)
        # For G4, four 64-output tiles expose 64 post-HAPR lanes. For a sweep,
        # the issue window scales with the same derived lane count.
        issue_cycles = max(int(math.ceil(opportunities / post_hapr_lanes)), 1)
        base, remainder = divmod(requests, issue_cycles)
        advance_constant(base + 1, remainder)
        advance_constant(base, issue_cycles - remainder)

    while queue > 0:
        queue = max(queue - service_per_cycle, 0)
        record_depth(float(queue))
        cycles += 1

    depth_count = sum(count for _, count in depth_runs)
    depth_sum = sum(value * count for value, count in depth_runs)

    def weighted_percentile(q: float) -> float:
        if depth_count == 0:
            return 0.0
        ordered = sorted(depth_runs, key=lambda item: item[0])
        rank = q * (depth_count - 1)

        def value_at(index: int) -> float:
            seen = 0
            for value, count in ordered:
                if index < seen + count:
                    return value
                seen += count
            return ordered[-1][0]

        lo, hi = int(math.floor(rank)), int(math.ceil(rank))
        if lo == hi:
            return value_at(lo)
        return value_at(lo) * (hi - rank) + value_at(hi) * (rank - lo)

    return {
        "adc_macros": int(adc_macros),
        "photonic_tiles": physical_tiles,
        "outputs_per_tile": tile_outputs,
        "hapr_group_size": group_size,
        "post_hapr_output_lanes_total": post_hapr_lanes,
        "service_requests_per_cycle": service_per_cycle,
        "no_hold_required_capacity_per_cycle": post_hapr_lanes,
        "no_hold_admission_safe": no_hold_admission_safe,
        "no_hold_capacity_shortfall": max(post_hapr_lanes - service_per_cycle, 0),
        "requests": arrivals_total,
        "queue_mean": depth_sum / depth_count if depth_count else 0.0,
        "queue_p95": weighted_percentile(0.95),
        "queue_p99": weighted_percentile(0.99),
        "queue_max": max((value for value, _ in depth_runs), default=0.0),
        "stall_cycles": stalls,
        "dropped_requests": dropped,
        "fifo_overflow": bool(dropped),
        "latency_cycles": cycles,
        "latency_ns": cycles / float(clock_ghz),
        "trace_model": "exact request counts; deterministic even placement inside each measured layer/timestep bin; post-HAPR lane count = ceil(photonic_tiles * outputs_per_tile / hapr_group_size); same-window ADC admission without multi-cycle analog storage; mathematically compressed recurrence",
    }


def summarize_samples(rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row.get("seed"), row.get("perturbation_type"), row.get("level_label"), row.get("sample_index"))
        groups.setdefault(key, []).append(row)
    samples: list[dict[str, Any]] = []
    for key, group in groups.items():
        result = replay_sample(group, **kwargs)
        result.update({"seed": key[0], "perturbation_type": key[1], "level_label": key[2], "sample_index": key[3]})
        samples.append(result)
    latency = [float(row["latency_ns"]) for row in samples]
    queue_max = [float(row["queue_max"]) for row in samples]
    return samples, {
        "num_samples": len(samples),
        "latency_ns_mean": mean(latency) if latency else 0.0,
        "latency_ns_p99": percentile(latency, 0.99),
        "queue_max_mean": mean(queue_max) if queue_max else 0.0,
        "queue_max_p99": percentile(queue_max, 0.99),
        "queue_max_global": max(queue_max, default=0.0),
        "stall_cycles_total": sum(int(row["stall_cycles"]) for row in samples),
        "dropped_requests_total": sum(int(row["dropped_requests"]) for row in samples),
        "fifo_overflow_any": any(bool(row["fifo_overflow"]) for row in samples),
        "no_hold_admission_safe_all": all(bool(row["no_hold_admission_safe"]) for row in samples),
        "no_hold_capacity_shortfall_max": max((int(row["no_hold_capacity_shortfall"]) for row in samples), default=0),
    }
