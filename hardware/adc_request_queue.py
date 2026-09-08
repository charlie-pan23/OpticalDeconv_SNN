"""Request-level ADC queue reference model for HIPSA.

This module models architecture-level request admission, pipelined ADC issue,
conversion completion, and tail drain.  It deliberately separates the ADC
throughput contract (samples accepted per architecture cycle) from conversion
latency (cycles until a result completes).

The selected HIPSA design has no temporal analog storage.  In that mode a
request that cannot start conversion in its arrival window is explicitly
rejected; a digital metadata FIFO must not be used to invent an analog hold
mechanism.  Enabling ``temporal_analog_storage`` is therefore a generic stress
mode, not the selected G4/A8 operating point.

Nothing in this model establishes mux/TIA settling, aperture timing,
transistor-level timing closure, post-layout feasibility, or silicon behavior.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Optional


@dataclass
class ADCRequest:
    """One HAPR-output request and its transaction identity/lifecycle."""

    request_id: str
    sample_id: str | int
    timestep_id: int
    layer_id: str
    output_tile_id: str | int
    output_neuron_id: int
    partition_id: str | int
    hapr_group_id: str
    arrival_cycle: int
    conversion_latency_cycles: Optional[int] = None
    partial_sum_value: float = 0.0

    admission_cycle: Optional[int] = None
    conversion_start_cycle: Optional[int] = None
    conversion_done_cycle: Optional[int] = None
    completion_cycle: Optional[int] = None
    adc_macro_id: Optional[int] = None
    adc_slot_id: Optional[int] = None
    admission_sequence: Optional[int] = None
    completion_sequence: Optional[int] = None
    status: str = "CREATED"
    rejection_reason: Optional[str] = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ADCRequest":
        """Build a request from explicit-schema rows with a few trace aliases."""

        return cls(
            request_id=str(row["request_id"]),
            sample_id=row.get("sample_id", row.get("sample", "")),
            timestep_id=int(row.get("timestep_id", row.get("timestep", 0))),
            layer_id=str(row.get("layer_id", row.get("layer", ""))),
            output_tile_id=row.get("output_tile_id", row.get("output_tile", "")),
            output_neuron_id=int(row.get("output_neuron_id", row.get("output_neuron", -1))),
            partition_id=row.get("partition_id", row.get("partial_sum_id", "")),
            hapr_group_id=str(row.get("hapr_group_id", "")),
            arrival_cycle=int(row.get("arrival_cycle", 0)),
            conversion_latency_cycles=(
                None
                if row.get("conversion_latency_cycles") in (None, "")
                else int(row["conversion_latency_cycles"])
            ),
            partial_sum_value=float(row.get("partial_sum_value", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ADCQueueConfig:
    """Architecture parameters for request admission and pipelined conversion."""

    adc_macros: int = 8
    samples_per_macro_per_cycle: int = 10
    fifo_depth: int = 4096
    post_hapr_lanes: int = 64
    default_conversion_latency_cycles: int = 1
    temporal_analog_storage: bool = False
    drain_tail: bool = True
    max_simulation_cycles: int = 10_000_000

    @property
    def service_capacity_per_cycle(self) -> int:
        return self.adc_macros * self.samples_per_macro_per_cycle

    def validate(self) -> None:
        for name in (
            "adc_macros",
            "samples_per_macro_per_cycle",
            "fifo_depth",
            "post_hapr_lanes",
            "default_conversion_latency_cycles",
            "max_simulation_cycles",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


_REQUIRED_ID_FIELDS = (
    "request_id",
    "sample_id",
    "layer_id",
    "output_tile_id",
    "partition_id",
    "hapr_group_id",
)


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(max(float(q), 0.0), 1.0)
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - position) + ordered[hi] * (position - lo))


def validate_request_domains(requests: Iterable[ADCRequest]) -> None:
    """Reject duplicate IDs and HAPR groups that cross transaction domains."""

    seen_ids: set[str] = set()
    group_domains: dict[tuple[str, int, str, str], tuple[str, int]] = {}
    for request in requests:
        for field in _REQUIRED_ID_FIELDS:
            value = getattr(request, field)
            if value is None or str(value) == "":
                raise ValueError(f"request {request.request_id!r} has empty {field}")
        if request.request_id in seen_ids:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        seen_ids.add(request.request_id)
        if request.arrival_cycle < 0:
            raise ValueError(f"request {request.request_id!r} has negative arrival_cycle")
        if request.output_neuron_id < 0:
            raise ValueError(f"request {request.request_id!r} has invalid output_neuron_id")
        if request.conversion_latency_cycles is not None and request.conversion_latency_cycles <= 0:
            raise ValueError(
                f"request {request.request_id!r} has non-positive conversion latency"
            )

        context = (
            str(request.sample_id),
            int(request.timestep_id),
            str(request.layer_id),
            str(request.hapr_group_id),
        )
        domain = (str(request.output_tile_id), int(request.output_neuron_id))
        previous = group_domains.setdefault(context, domain)
        if previous != domain:
            raise ValueError(
                "HAPR group crosses output tile/neuron boundary: "
                f"context={context!r}, first={previous!r}, current={domain!r}"
            )


def _clone_and_sort(requests: Iterable[ADCRequest | Mapping[str, Any]]) -> list[ADCRequest]:
    cloned: list[ADCRequest] = []
    for item in requests:
        request = item if isinstance(item, ADCRequest) else ADCRequest.from_mapping(item)
        cloned.append(
            replace(
                request,
                admission_cycle=None,
                conversion_start_cycle=None,
                conversion_done_cycle=None,
                completion_cycle=None,
                adc_macro_id=None,
                adc_slot_id=None,
                admission_sequence=None,
                completion_sequence=None,
                status="CREATED",
                rejection_reason=None,
            )
        )
    validate_request_domains(cloned)
    return sorted(cloned, key=lambda r: (r.arrival_cycle, r.request_id))


def simulate_request_queue(
    requests: Iterable[ADCRequest | Mapping[str, Any]],
    config: ADCQueueConfig,
) -> dict[str, Any]:
    """Simulate explicit request admission, ADC issue, completion, and drain.

    ADC macros are treated as pipelined: conversion latency delays completion
    but does not reduce the declared per-cycle sample acceptance rate.
    """

    config.validate()
    work = _clone_and_sort(requests)
    if not work:
        return _empty_result(config)

    arrivals_by_cycle: dict[int, list[ADCRequest]] = defaultdict(list)
    for request in work:
        arrivals_by_cycle[request.arrival_cycle].append(request)

    pending: deque[ADCRequest] = deque()
    inflight: dict[int, list[ADCRequest]] = defaultdict(list)
    cycle_trace: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0
    served_count = 0
    admission_sequence = 0
    completion_sequence = 0
    inflight_count = 0
    queue_depths: list[int] = []
    first_cycle = min(arrivals_by_cycle)
    last_arrival_cycle = max(arrivals_by_cycle)
    cycle = first_cycle

    def reject(request: ADCRequest, reason: str) -> None:
        nonlocal rejected_count
        request.status = "REJECTED"
        request.rejection_reason = reason
        rejected_count += 1

    while True:
        if cycle - first_cycle > config.max_simulation_cycles:
            raise RuntimeError("ADC request queue exceeded max_simulation_cycles")

        completed_now = sorted(
            inflight.pop(cycle, []),
            key=lambda r: (
                r.conversion_start_cycle if r.conversion_start_cycle is not None else -1,
                r.admission_sequence if r.admission_sequence is not None else -1,
            ),
        )
        inflight_count -= len(completed_now)
        for request in completed_now:
            request.status = "COMPLETED"
            request.conversion_done_cycle = cycle
            request.completion_cycle = cycle
            request.completion_sequence = completion_sequence
            completion_sequence += 1
            served_count += 1

        arrivals_now = arrivals_by_cycle.get(cycle, [])
        available_fifo = max(config.fifo_depth - len(pending), 0)
        admitted_to_fifo = arrivals_now[:available_fifo]
        fifo_overflow = arrivals_now[available_fifo:]
        for request in fifo_overflow:
            reject(request, "fifo_capacity_exceeded")

        if config.temporal_analog_storage:
            for request in admitted_to_fifo:
                request.status = "ACCEPTED"
                request.admission_cycle = cycle
                request.admission_sequence = admission_sequence
                admission_sequence += 1
                accepted_count += 1
                pending.append(request)
        else:
            pending.extend(admitted_to_fifo)

        starts_now: list[ADCRequest] = []
        for slot_index in range(min(config.service_capacity_per_cycle, len(pending))):
            request = pending.popleft()
            if not config.temporal_analog_storage:
                request.status = "ACCEPTED"
                request.admission_cycle = cycle
                request.admission_sequence = admission_sequence
                admission_sequence += 1
                accepted_count += 1
            request.status = "CONVERTING"
            request.conversion_start_cycle = cycle
            request.adc_macro_id = slot_index // config.samples_per_macro_per_cycle
            request.adc_slot_id = slot_index % config.samples_per_macro_per_cycle
            latency = (
                request.conversion_latency_cycles
                if request.conversion_latency_cycles is not None
                else config.default_conversion_latency_cycles
            )
            done_cycle = cycle + latency
            request.conversion_done_cycle = done_cycle
            inflight[done_cycle].append(request)
            inflight_count += 1
            starts_now.append(request)

        same_window_rejected = 0
        if not config.temporal_analog_storage:
            while pending:
                reject(pending.popleft(), "same_window_capacity_exceeded")
                same_window_rejected += 1

        pending_depth = len(pending)
        inflight_depth = inflight_count
        queue_depths.append(pending_depth)
        cycle_trace.append(
            {
                "cycle": cycle,
                "arrivals": len(arrivals_now),
                "accepted": (
                    len(admitted_to_fifo)
                    if config.temporal_analog_storage
                    else len(starts_now)
                ),
                "rejected": len(fifo_overflow) + same_window_rejected,
                "conversion_starts": len(starts_now),
                "completions": len(completed_now),
                "fifo_occupancy": pending_depth,
                "conversion_inflight": inflight_depth,
                "outstanding_requests": pending_depth + inflight_depth,
            }
        )

        if cycle >= last_arrival_cycle:
            if not config.drain_tail:
                break
            if not pending and not inflight:
                break
        cycle += 1

    final_pending = len(pending)
    final_inflight = inflight_count
    final_outstanding = final_pending + final_inflight
    arrivals_count = len(work)
    unaccounted_arrivals = arrivals_count - accepted_count - rejected_count
    unaccounted_accepted = accepted_count - served_count - final_outstanding
    arrival_conservation = unaccounted_arrivals == 0
    accepted_conservation = unaccounted_accepted == 0
    same_window_capacity = config.service_capacity_per_cycle >= config.post_hapr_lanes
    served_requests = [request for request in work if request.completion_sequence is not None]
    completion_order = [
        request.request_id
        for request in sorted(
            served_requests, key=lambda request: int(request.completion_sequence or 0)
        )
    ]
    admission_order = [
        request.request_id
        for request in sorted(
            served_requests, key=lambda request: int(request.admission_sequence or 0)
        )
    ]

    summary = {
        "evidence_class": "architecture_level_transaction_reference_model",
        "circuit_timing_closed": False,
        "adc_macros": config.adc_macros,
        "samples_per_macro_per_cycle": config.samples_per_macro_per_cycle,
        "service_capacity_per_cycle": config.service_capacity_per_cycle,
        "post_hapr_lanes": config.post_hapr_lanes,
        "same_window_admission_feasible": same_window_capacity,
        "same_window_capacity_shortfall": max(
            config.post_hapr_lanes - config.service_capacity_per_cycle, 0
        ),
        "temporal_analog_storage": config.temporal_analog_storage,
        "fifo_depth": config.fifo_depth,
        "default_conversion_latency_cycles": config.default_conversion_latency_cycles,
        "arrivals": arrivals_count,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "served": served_count,
        "requests_dropped": 0,
        "final_fifo_occupancy": final_outstanding,
        "final_pending_fifo_occupancy": final_pending,
        "final_conversion_inflight": final_inflight,
        "unaccounted_requests": unaccounted_arrivals,
        "unaccounted_accepted_requests": unaccounted_accepted,
        "arrival_conservation_valid": arrival_conservation,
        "accepted_conservation_valid": accepted_conservation,
        "tail_drain_requested": config.drain_tail,
        "tail_drain_complete": final_outstanding == 0,
        "queue_mean": sum(queue_depths) / max(len(queue_depths), 1),
        "queue_p95": _percentile(queue_depths, 0.95),
        "queue_p99": _percentile(queue_depths, 0.99),
        "queue_max": max(queue_depths, default=0),
        "first_cycle": first_cycle,
        "last_arrival_cycle": last_arrival_cycle,
        "last_observed_cycle": cycle,
        "cycles_observed": cycle - first_cycle + 1,
        "completion_reordered_vs_admission": completion_order != admission_order,
        "selected_point_valid": bool(
            same_window_capacity
            and not config.temporal_analog_storage
            and rejected_count == 0
            and arrival_conservation
            and accepted_conservation
            and (not config.drain_tail or final_outstanding == 0)
        ),
        "claim_boundary": (
            "architecture_level_admission_and_transaction_conservation; "
            "not_mux_tia_aperture_or_circuit_timing_closure"
        ),
    }
    return {
        "config": asdict(config),
        "summary": summary,
        "requests": [request.to_dict() for request in work],
        "cycle_trace": cycle_trace,
    }


def _empty_result(config: ADCQueueConfig) -> dict[str, Any]:
    same_window_capacity = config.service_capacity_per_cycle >= config.post_hapr_lanes
    return {
        "config": asdict(config),
        "summary": {
            "evidence_class": "architecture_level_transaction_reference_model",
            "circuit_timing_closed": False,
            "adc_macros": config.adc_macros,
            "samples_per_macro_per_cycle": config.samples_per_macro_per_cycle,
            "service_capacity_per_cycle": config.service_capacity_per_cycle,
            "post_hapr_lanes": config.post_hapr_lanes,
            "same_window_admission_feasible": same_window_capacity,
            "same_window_capacity_shortfall": max(
                config.post_hapr_lanes - config.service_capacity_per_cycle, 0
            ),
            "temporal_analog_storage": config.temporal_analog_storage,
            "fifo_depth": config.fifo_depth,
            "default_conversion_latency_cycles": config.default_conversion_latency_cycles,
            "arrivals": 0,
            "accepted": 0,
            "rejected": 0,
            "served": 0,
            "requests_dropped": 0,
            "final_fifo_occupancy": 0,
            "final_pending_fifo_occupancy": 0,
            "final_conversion_inflight": 0,
            "unaccounted_requests": 0,
            "unaccounted_accepted_requests": 0,
            "arrival_conservation_valid": True,
            "accepted_conservation_valid": True,
            "tail_drain_requested": config.drain_tail,
            "tail_drain_complete": True,
            "queue_mean": 0.0,
            "queue_p95": 0.0,
            "queue_p99": 0.0,
            "queue_max": 0,
            "first_cycle": None,
            "last_arrival_cycle": None,
            "last_observed_cycle": None,
            "cycles_observed": 0,
            "completion_reordered_vs_admission": False,
            "selected_point_valid": bool(
                same_window_capacity and not config.temporal_analog_storage
            ),
            "claim_boundary": (
                "architecture_level_admission_and_transaction_conservation; "
                "not_mux_tia_aperture_or_circuit_timing_closure"
            ),
        },
        "requests": [],
        "cycle_trace": [],
    }


# A descriptive alias for callers that treat the model as trace replay.
replay_adc_requests = simulate_request_queue


