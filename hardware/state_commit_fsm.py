"""Partition-completion and state-commit reference FSM for HIPSA.

The model consumes request-level ADC completions.  It does not threshold a
neuron until every explicitly expected digital partition has arrived, and it
serializes timesteps for each persistent neuron state.  Partial sums are
accumulated in a deterministic partition order so ADC completion reordering
cannot change the numerical result.

This is an architecture-level semantic reference model.  Commit-engine
bandwidth, SRAM-port timing, mux/TIA settling, ADC aperture timing, post-layout
closure, and silicon behavior are outside its evidence boundary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

from hardware.membrane_fixedpoint import quantize_membrane


StateKey = tuple[str, int, str, int]
StateChainKey = tuple[str, str, int]


@dataclass(frozen=True)
class StateCommitSpec:
    """Expected partitions for one neuron state at one timestep."""

    sample_id: str | int
    timestep_id: int
    layer_id: str
    output_neuron_id: int
    output_tile_id: str | int
    expected_partition_ids: tuple[str | int, ...]

    @property
    def key(self) -> StateKey:
        return (
            str(self.sample_id),
            int(self.timestep_id),
            str(self.layer_id),
            int(self.output_neuron_id),
        )

    @property
    def chain_key(self) -> StateChainKey:
        return (
            str(self.sample_id),
            str(self.layer_id),
            int(self.output_neuron_id),
        )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "StateCommitSpec":
        partitions = row.get("expected_partition_ids", row.get("partitions", ()))
        if isinstance(partitions, str):
            partitions = tuple(item.strip() for item in partitions.split(",") if item.strip())
        return cls(
            sample_id=row.get("sample_id", row.get("sample", "")),
            timestep_id=int(row.get("timestep_id", row.get("timestep", 0))),
            layer_id=str(row.get("layer_id", row.get("layer", ""))),
            output_neuron_id=int(
                row.get("output_neuron_id", row.get("output_neuron", -1))
            ),
            output_tile_id=row.get("output_tile_id", row.get("output_tile", "")),
            expected_partition_ids=tuple(partitions),
        )


@dataclass(frozen=True)
class PartitionCompletion:
    """One completed ADC request carrying a digital partial-sum value."""

    request_id: str
    sample_id: str | int
    timestep_id: int
    layer_id: str
    output_neuron_id: int
    output_tile_id: str | int
    partition_id: str | int
    hapr_group_id: str
    partial_sum_value: float
    completion_cycle: int
    completion_sequence: int
    status: str = "COMPLETED"

    @property
    def key(self) -> StateKey:
        return (
            str(self.sample_id),
            int(self.timestep_id),
            str(self.layer_id),
            int(self.output_neuron_id),
        )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PartitionCompletion":
        completion_cycle = row.get("completion_cycle")
        completion_sequence = row.get("completion_sequence")
        if completion_cycle is None:
            raise ValueError(f"request {row.get('request_id')!r} has no completion_cycle")
        if completion_sequence is None:
            raise ValueError(f"request {row.get('request_id')!r} has no completion_sequence")
        if "partial_sum_value" not in row:
            raise ValueError(f"request {row.get('request_id')!r} has no partial_sum_value")
        return cls(
            request_id=str(row.get("request_id", "")),
            sample_id=row.get("sample_id", row.get("sample", "")),
            timestep_id=int(row.get("timestep_id", row.get("timestep", 0))),
            layer_id=str(row.get("layer_id", row.get("layer", ""))),
            output_neuron_id=int(
                row.get("output_neuron_id", row.get("output_neuron", -1))
            ),
            output_tile_id=row.get("output_tile_id", row.get("output_tile", "")),
            partition_id=row.get("partition_id", row.get("partial_sum_id", "")),
            hapr_group_id=str(row.get("hapr_group_id", "")),
            partial_sum_value=float(row["partial_sum_value"]),
            completion_cycle=int(completion_cycle),
            completion_sequence=int(completion_sequence),
            status=str(row.get("status", "COMPLETED")),
        )


@dataclass(frozen=True)
class StateCommitConfig:
    """Numerical contract for one lazy-LIF state commit."""

    decay: float = 0.5
    input_scale: float = 0.5
    threshold: float = 1.0
    membrane_bits: int = 16
    fractional_bits: int = 8
    reset_mode: str = "zero"
    initial_last_timestep: int = -1

    def validate(self) -> None:
        if not 0.0 <= float(self.decay) <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if float(self.input_scale) < 0.0:
            raise ValueError("input_scale must be non-negative")
        if int(self.membrane_bits) < 2:
            raise ValueError("membrane_bits must be at least 2")
        if not 0 <= int(self.fractional_bits) < int(self.membrane_bits):
            raise ValueError("fractional_bits must be in [0, membrane_bits)")
        if self.reset_mode not in {"zero", "subtract"}:
            raise ValueError("reset_mode must be 'zero' or 'subtract'")


@dataclass
class _Context:
    spec: StateCommitSpec
    state: str = "WAITING"
    received: dict[str, PartitionCompletion] = field(default_factory=dict)
    all_received_cycle: Optional[int] = None
    commit_cycle: Optional[int] = None
    commit_sequence: Optional[int] = None
    decay_count: int = 0
    threshold_count: int = 0
    reset_count: int = 0


def _q(value: float, config: StateCommitConfig) -> float:
    quantized = quantize_membrane(
        float(value),
        bits=config.membrane_bits,
        fractional_bits=config.fractional_bits,
    )
    return float(quantized)


def _partition_id(value: str | int) -> str:
    return str(value)


def _key_to_dict(key: StateKey) -> dict[str, Any]:
    return {
        "sample_id": key[0],
        "timestep_id": key[1],
        "layer_id": key[2],
        "output_neuron_id": key[3],
    }


def _validate_specs(specs: list[StateCommitSpec]) -> None:
    seen: set[StateKey] = set()
    previous_timestep: dict[StateChainKey, int] = {}
    for spec in sorted(specs, key=lambda item: (item.chain_key, item.timestep_id)):
        if not str(spec.sample_id):
            raise ValueError("state spec has empty sample_id")
        if not str(spec.layer_id):
            raise ValueError("state spec has empty layer_id")
        if not str(spec.output_tile_id):
            raise ValueError("state spec has empty output_tile_id")
        if spec.timestep_id < 0:
            raise ValueError("state spec has negative timestep_id")
        if spec.output_neuron_id < 0:
            raise ValueError("state spec has invalid output_neuron_id")
        if spec.key in seen:
            raise ValueError(f"duplicate state spec: {spec.key!r}")
        seen.add(spec.key)
        normalized = [_partition_id(item) for item in spec.expected_partition_ids]
        if not normalized:
            raise ValueError(f"state spec {spec.key!r} has no expected partitions")
        if any(not item for item in normalized):
            raise ValueError(f"state spec {spec.key!r} has an empty partition ID")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"state spec {spec.key!r} has duplicate expected partitions")
        previous = previous_timestep.get(spec.chain_key)
        if previous is not None and spec.timestep_id <= previous:
            raise ValueError(f"non-increasing timestep chain: {spec.chain_key!r}")
        previous_timestep[spec.chain_key] = spec.timestep_id


def simulate_state_commit_fsm(
    completions: Iterable[PartitionCompletion | Mapping[str, Any]],
    specs: Iterable[StateCommitSpec | Mapping[str, Any]],
    config: StateCommitConfig = StateCommitConfig(),
    *,
    initial_states: Optional[Mapping[StateChainKey, Mapping[str, Any] | float]] = None,
) -> dict[str, Any]:
    """Replay partition completions through the deterministic state-commit FSM."""

    config.validate()
    spec_list = [
        item if isinstance(item, StateCommitSpec) else StateCommitSpec.from_mapping(item)
        for item in specs
    ]
    _validate_specs(spec_list)
    contexts = {spec.key: _Context(spec=spec) for spec in spec_list}

    chains: dict[StateChainKey, list[StateKey]] = {}
    for spec in spec_list:
        chains.setdefault(spec.chain_key, []).append(spec.key)
    for keys in chains.values():
        keys.sort(key=lambda key: key[1])

    chain_position = {chain: 0 for chain in chains}
    chain_state: dict[StateChainKey, dict[str, Any]] = {}
    for chain in chains:
        raw = (initial_states or {}).get(chain, 0.0)
        if isinstance(raw, Mapping):
            membrane = float(raw.get("membrane", 0.0))
            last_timestep = int(raw.get("last_committed_timestep", config.initial_last_timestep))
        else:
            membrane = float(raw)
            last_timestep = int(config.initial_last_timestep)
        chain_state[chain] = {
            "membrane": _q(membrane, config),
            "last_committed_timestep": last_timestep,
            "last_commit_cycle": None,
        }

    completion_rows = [
        item if isinstance(item, PartitionCompletion) else PartitionCompletion.from_mapping(item)
        for item in completions
    ]
    completion_rows.sort(
        key=lambda item: (item.completion_cycle, item.completion_sequence, item.request_id)
    )

    transitions: list[dict[str, Any]] = []
    commit_trace: list[dict[str, Any]] = []
    receive_trace: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    commit_sequence = 0
    premature_threshold_count = 0
    duplicate_commit_count = 0
    dependency_wait_count = 0

    def transition(
        context: _Context,
        next_state: str,
        cycle: int,
        *,
        request_id: Optional[str],
        reason: str,
    ) -> None:
        previous = context.state
        context.state = next_state
        transitions.append(
            {
                **_key_to_dict(context.spec.key),
                "output_tile_id": context.spec.output_tile_id,
                "cycle": int(cycle),
                "request_id": request_id,
                "from_state": previous,
                "to_state": next_state,
                "reason": reason,
            }
        )

    def commit_ready_contexts(chain: StateChainKey, release_cycle: int) -> None:
        nonlocal commit_sequence, premature_threshold_count, duplicate_commit_count
        keys = chains[chain]
        while chain_position[chain] < len(keys):
            key = keys[chain_position[chain]]
            context = contexts[key]
            if context.state != "ALL_PARTITIONS_RECEIVED":
                break
            if context.commit_sequence is not None:
                duplicate_commit_count += 1
                raise RuntimeError(f"state context committed twice: {key!r}")

            expected = sorted(
                (_partition_id(item) for item in context.spec.expected_partition_ids)
            )
            if set(expected) != set(context.received):
                premature_threshold_count += 1
                raise RuntimeError(f"premature commit attempted for {key!r}")

            state = chain_state[chain]
            previous_timestep = int(state["last_committed_timestep"])
            elapsed_steps = context.spec.timestep_id - previous_timestep
            if elapsed_steps <= 0:
                raise RuntimeError(f"state timestep dependency violated for {key!r}")

            transition(
                context,
                "APPLY_DECAY",
                release_cycle,
                request_id=None,
                reason="preceding_timestep_committed_and_all_partitions_received",
            )
            membrane_before = float(state["membrane"])
            membrane_after_decay = _q(
                membrane_before * (float(config.decay) ** elapsed_steps), config
            )
            context.decay_count += 1

            transition(
                context,
                "ACCUMULATE",
                release_cycle,
                request_id=None,
                reason="decay_applied_once",
            )
            input_current = math.fsum(
                context.received[partition].partial_sum_value for partition in expected
            )
            membrane_pre_threshold = _q(
                membrane_after_decay + input_current * float(config.input_scale), config
            )

            transition(
                context,
                "THRESHOLD",
                release_cycle,
                request_id=None,
                reason="deterministic_partition_accumulation_complete",
            )
            context.threshold_count += 1
            spike = membrane_pre_threshold >= float(config.threshold)
            reset_applied = False
            if spike:
                transition(
                    context,
                    "SPIKE_AND_RESET",
                    release_cycle,
                    request_id=None,
                    reason="threshold_reached",
                )
                reset_applied = True
                context.reset_count += 1
                if config.reset_mode == "zero":
                    membrane_after_commit = _q(0.0, config)
                else:
                    membrane_after_commit = _q(
                        membrane_pre_threshold - float(config.threshold), config
                    )
            else:
                transition(
                    context,
                    "NO_SPIKE",
                    release_cycle,
                    request_id=None,
                    reason="threshold_not_reached",
                )
                membrane_after_commit = membrane_pre_threshold

            transition(
                context,
                "COMMITTED",
                release_cycle,
                request_id=None,
                reason="state_write_visible",
            )
            context.commit_cycle = int(release_cycle)
            context.commit_sequence = commit_sequence
            state["membrane"] = membrane_after_commit
            state["last_committed_timestep"] = context.spec.timestep_id
            state["last_commit_cycle"] = int(release_cycle)
            commit_trace.append(
                {
                    **_key_to_dict(key),
                    "output_tile_id": context.spec.output_tile_id,
                    "expected_partition_ids": expected,
                    "received_partition_ids": expected,
                    "partition_count": len(expected),
                    "all_partitions_received_cycle": context.all_received_cycle,
                    "commit_cycle": int(release_cycle),
                    "commit_sequence": commit_sequence,
                    "elapsed_steps": elapsed_steps,
                    "membrane_before": membrane_before,
                    "membrane_after_decay": membrane_after_decay,
                    "input_current": input_current,
                    "input_scale": float(config.input_scale),
                    "membrane_pre_threshold": membrane_pre_threshold,
                    "threshold": float(config.threshold),
                    "spike": int(spike),
                    "reset_applied": int(reset_applied),
                    "reset_mode": config.reset_mode,
                    "membrane_after_commit": membrane_after_commit,
                    "decay_count": context.decay_count,
                    "threshold_count": context.threshold_count,
                    "reset_count": context.reset_count,
                }
            )
            commit_sequence += 1
            chain_position[chain] += 1

    for completion in completion_rows:
        if completion.status != "COMPLETED":
            raise ValueError(
                f"request {completion.request_id!r} is not completed: {completion.status!r}"
            )
        if not completion.request_id:
            raise ValueError("completion has empty request_id")
        if completion.request_id in seen_requests:
            raise ValueError(f"duplicate completion request_id: {completion.request_id}")
        seen_requests.add(completion.request_id)
        if completion.completion_cycle < 0 or completion.completion_sequence < 0:
            raise ValueError(f"request {completion.request_id!r} has invalid completion order")
        if completion.key not in contexts:
            raise ValueError(f"completion has no state spec: {completion.key!r}")

        context = contexts[completion.key]
        if str(completion.output_tile_id) != str(context.spec.output_tile_id):
            raise ValueError(
                f"completion output tile mismatch for {completion.key!r}: "
                f"expected {context.spec.output_tile_id!r}, got {completion.output_tile_id!r}"
            )
        partition = _partition_id(completion.partition_id)
        expected = {_partition_id(item) for item in context.spec.expected_partition_ids}
        if partition not in expected:
            raise ValueError(
                f"unexpected partition {partition!r} for state context {completion.key!r}"
            )
        if partition in context.received:
            raise ValueError(
                f"duplicate partition {partition!r} for state context {completion.key!r}"
            )
        if context.state == "COMMITTED":
            duplicate_commit_count += 1
            raise ValueError(f"partition arrived after commit for {completion.key!r}")

        if context.state == "WAITING":
            transition(
                context,
                "RECEIVING_PARTIALS",
                completion.completion_cycle,
                request_id=completion.request_id,
                reason="first_partition_completed",
            )
        context.received[partition] = completion
        receive_trace.append(
            {
                **_key_to_dict(completion.key),
                "output_tile_id": completion.output_tile_id,
                "request_id": completion.request_id,
                "partition_id": partition,
                "hapr_group_id": completion.hapr_group_id,
                "partial_sum_value": completion.partial_sum_value,
                "completion_cycle": completion.completion_cycle,
                "completion_sequence": completion.completion_sequence,
                "received_partition_count": len(context.received),
                "expected_partition_count": len(expected),
            }
        )

        if set(context.received) == expected:
            context.all_received_cycle = completion.completion_cycle
            transition(
                context,
                "ALL_PARTITIONS_RECEIVED",
                completion.completion_cycle,
                request_id=completion.request_id,
                reason="last_expected_partition_completed",
            )
            chain = context.spec.chain_key
            if chains[chain][chain_position[chain]] != context.spec.key:
                dependency_wait_count += 1
            commit_ready_contexts(chain, completion.completion_cycle)

    incomplete_contexts: list[dict[str, Any]] = []
    for key, context in sorted(contexts.items()):
        expected = {_partition_id(item) for item in context.spec.expected_partition_ids}
        received = set(context.received)
        if context.state != "COMMITTED":
            incomplete_contexts.append(
                {
                    **_key_to_dict(key),
                    "output_tile_id": context.spec.output_tile_id,
                    "state": context.state,
                    "expected_partition_ids": sorted(expected),
                    "received_partition_ids": sorted(received),
                    "missing_partition_ids": sorted(expected - received),
                }
            )

    final_states = [
        {
            "sample_id": chain[0],
            "layer_id": chain[1],
            "output_neuron_id": chain[2],
            "membrane": float(state["membrane"]),
            "last_committed_timestep": int(state["last_committed_timestep"]),
            "last_commit_cycle": state["last_commit_cycle"],
        }
        for chain, state in sorted(chain_state.items())
    ]
    spike_train = [
        {
            "sample_id": row["sample_id"],
            "timestep_id": row["timestep_id"],
            "layer_id": row["layer_id"],
            "output_neuron_id": row["output_neuron_id"],
            "spike": row["spike"],
        }
        for row in commit_trace
    ]
    reset_events = [
        {
            "sample_id": row["sample_id"],
            "timestep_id": row["timestep_id"],
            "layer_id": row["layer_id"],
            "output_neuron_id": row["output_neuron_id"],
            "reset_applied": row["reset_applied"],
        }
        for row in commit_trace
    ]

    commit_count = len(commit_trace)
    decay_count = sum(context.decay_count for context in contexts.values())
    threshold_count = sum(context.threshold_count for context in contexts.values())
    reset_count = sum(context.reset_count for context in contexts.values())
    committed_once = all(
        context.commit_sequence is None or context.state == "COMMITTED"
        for context in contexts.values()
    )
    all_complete = not incomplete_contexts
    summary = {
        "evidence_class": "architecture_level_state_commit_reference_model",
        "commit_engine_bandwidth_modeled": False,
        "state_contexts_expected": len(contexts),
        "completion_events": len(completion_rows),
        "unique_completion_requests": len(seen_requests),
        "state_contexts_committed": commit_count,
        "state_contexts_incomplete": len(incomplete_contexts),
        "commit_count": commit_count,
        "decay_count": decay_count,
        "threshold_count": threshold_count,
        "reset_count": reset_count,
        "spike_count": sum(row["spike"] for row in commit_trace),
        "premature_threshold_count": premature_threshold_count,
        "duplicate_commit_count": duplicate_commit_count,
        "dependency_wait_count": dependency_wait_count,
        "all_expected_partitions_received": all_complete,
        "one_decay_per_commit": decay_count == commit_count,
        "one_threshold_per_commit": threshold_count == commit_count,
        "one_commit_per_context": committed_once and commit_count <= len(contexts),
        "reset_iff_spike": reset_count == sum(row["spike"] for row in commit_trace),
        "next_timestep_reads_committed_state": True,
        "deterministic_partition_accumulation": True,
        "state_commit_valid": bool(
            all_complete
            and decay_count == commit_count
            and threshold_count == commit_count
            and committed_once
            and commit_count == len(contexts)
            and reset_count == sum(row["spike"] for row in commit_trace)
            and premature_threshold_count == 0
            and duplicate_commit_count == 0
        ),
        "claim_boundary": (
            "architecture_level_partition_completion_and_state_commit_semantics; "
            "not_commit_bandwidth_sram_timing_or_circuit_timing_closure"
        ),
    }
    return {
        "config": asdict(config),
        "summary": summary,
        "receive_trace": receive_trace,
        "transition_trace": transitions,
        "commit_trace": commit_trace,
        "incomplete_contexts": incomplete_contexts,
        "final_states": final_states,
        "spike_train": spike_train,
        "reset_events": reset_events,
    }


replay_partition_completions = simulate_state_commit_fsm
