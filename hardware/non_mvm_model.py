"""Transparent accounting for operators outside the photonic MVM path."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def build_non_mvm_summary(
    activity_summary: Mapping[str, Any],
    *,
    workload_name: str,
    time_steps: int,
    operator_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Estimate digital operator work without hiding it in MVM throughput."""

    cfg = operator_cfg or {}
    layers = activity_summary.get("layers", {})
    conv_shapes: List[List[int]] = []
    for info in layers.values() if isinstance(layers, Mapping) else []:
        shape = info.get("output_shape_last", []) if isinstance(info, Mapping) else []
        if isinstance(shape, list) and len(shape) == 4:
            conv_shapes.append([_int(v) for v in shape])
    # The archived trace stores only the last batch shape.  Use C/H/W and the
    # frozen timestep count; the leading batch is intentionally ignored.
    pool_kernel = _int(cfg.get("pool_kernel", 2), 2)
    rows: List[Dict[str, Any]] = []

    def add(operator: str, operations: float, note: str, included: bool = True) -> None:
        digital_parallelism = max(_float(cfg.get("digital_parallelism", 4096.0), 4096.0), 1.0)
        cycles_per_op = _float(cfg.get("cycles_per_digital_op", 1.0), 1.0) / digital_parallelism
        energy_pj = _float(cfg.get("energy_per_digital_op_pj", 0.25), 0.25)
        rows.append(
            {
                "workload": workload_name,
                "operator": operator,
                "operations_per_image": operations,
                "cycles_per_image": operations * cycles_per_op if included else 0.0,
                "energy_pj_per_image": operations * energy_pj if included else 0.0,
                "digital_parallelism": digital_parallelism,
                "included": int(included),
                "note": note,
            }
        )

    if workload_name.lower() == "cifar10dvs":
        pool_indices = [0, 1, 3, 5]
    elif workload_name.lower() == "dvsgesture":
        pool_indices = [0, 1, 2, 4]
    else:
        # A conservative fallback for an unseen CNN workload: one pool after
        # every convolution except the final one.
        pool_indices = list(range(max(len(conv_shapes) - 1, 0)))
    for index in pool_indices:
        if index >= len(conv_shapes):
            continue
        shape = conv_shapes[index]
        _, channels, height, width = shape
        pooled = max(height // pool_kernel, 1) * max(width // pool_kernel, 1)
        add(
            f"maxpool{index + 1}",
            float(time_steps * channels * pooled * (pool_kernel * pool_kernel - 1)),
            "Digital max-pool comparisons after the preceding LIF.",
        )

    last_conv = conv_shapes[-1] if conv_shapes else [1, 1, 1, 1]
    _, channels, height, width = last_conv
    add("global_average_pool", float(time_steps * channels * height * width), "Digital accumulation and divide; explicitly counted.")

    # The current models use a 256-wide digital FC2 classifier after fc1/LIF.
    num_classes = _int(cfg.get("num_classes", 10), 10)
    hidden_dim = _int(cfg.get("hidden_dim", 256), 256)
    add("digital_fc2", float(time_steps * hidden_dim * num_classes), "Final classifier is digital in the packaged models.")
    add("time_output_aggregation", float(time_steps * num_classes), "Mean/sum over timestep logits.")
    add("batchnorm_runtime", 0.0, "BatchNorm is folded into weights/thresholds for inference.")

    total_cycles = sum(_float(r["cycles_per_image"]) for r in rows)
    total_energy = sum(_float(r["energy_pj_per_image"]) for r in rows)
    return {
        "rows": rows,
        "total_cycles_per_image": total_cycles,
        "total_energy_pj_per_image": total_energy,
        "total_energy_nj_per_image": total_energy / 1e3,
        "operators_accounted": [r["operator"] for r in rows],
        "end_to_end_scope": "photonic MVM + ADC + digital operators + LIF/SRAM + configuration",
    }
