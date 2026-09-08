"""Event-driven membrane/SRAM traffic accounting."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def build_state_update_trace(
    activity_summary: Mapping[str, Any],
    transaction_layer_summary: List[Mapping[str, Any]],
    *,
    lazy_lif: bool = True,
) -> Dict[str, Any]:
    """Return per-layer actual state updates and SRAM read/write counts.

    A state is touched when the layer has a requested converted partial sum or
    emits a spike.  This keeps state activity independent of the number of ADC
    macros: ADC count affects queue service, not the number of states that need
    updating.
    """

    layers = activity_summary.get("layers", {})
    rows: List[Dict[str, Any]] = []
    total_dense = total_updates = total_reads = total_writes = 0.0
    for item in transaction_layer_summary:
        name = str(item.get("layer", ""))
        info = layers.get(name, {}) if isinstance(layers, Mapping) else {}
        dense_updates = _float(info.get("mvm_output_total"), 0.0) / max(_float(activity_summary.get("num_samples"), 1.0), 1.0)
        if dense_updates <= 0:
            dense_updates = _float(info.get("output_elements_per_image"), 0.0)
        request_activity = _clip(_float(info.get("adc_request_activity", info.get("adc_activity_proxy", 0.0))))
        spike_activity = _clip(_float(info.get("lif_spike_activity", info.get("output_activity", 0.0))))
        state_activity = max(request_activity, spike_activity) if lazy_lif else 1.0
        updates = dense_updates * state_activity
        reads = updates
        writes = updates
        row = {
            "layer": name,
            "dense_state_updates_per_image": dense_updates,
            "actual_lif_updates_per_image": updates,
            "sram_reads_per_image": reads,
            "sram_writes_per_image": writes,
            "state_update_activity": state_activity,
            "adc_request_activity_input": request_activity,
            "lif_spike_activity_input": spike_activity,
            "lazy_lif_enabled": int(lazy_lif),
            "activity_independent_of_adc_macro_count": 1,
            "trace_provenance": "derived_from_eval01_requests_and_lif_activity",
        }
        rows.append(row)
        total_dense += dense_updates
        total_updates += updates
        total_reads += reads
        total_writes += writes

    return {
        "rows": rows,
        "dense_state_updates_per_image": total_dense,
        "actual_lif_updates_per_image": total_updates,
        "sram_reads_per_image": total_reads,
        "sram_writes_per_image": total_writes,
        "state_update_activity": total_updates / max(total_dense, 1.0),
        "lazy_lif_enabled": bool(lazy_lif),
        "activity_independent_of_adc_macro_count": True,
    }
