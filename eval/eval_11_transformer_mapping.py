"""Mapping-only coverage analysis for a recent spiking Transformer.

The supplied archive has no trained Transformer checkpoint or activation
trace.  This stage therefore reports operator coverage, tile counts, and
configuration rounds only.  It deliberately refuses to emit latency, energy,
or accuracy, which would require per-layer activity and dynamic-attention
traces under the same evaluation model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

from utils.result_io import load_yaml, save_csv_rows, save_json


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def map_matrix(name: str, k: int, n: int, *, rows: int, cols: int, tiles: int, multiplicity: int = 1, execution: str = "optical_mrr") -> Dict[str, Any]:
    row_tiles = math.ceil(k / rows)
    col_tiles = math.ceil(n / cols)
    logical_tiles = row_tiles * col_tiles * multiplicity
    return {
        "operator": name,
        "execution": execution,
        "multiplicity": multiplicity,
        "matrix_k": k,
        "matrix_n": n,
        "row_tiles_per_instance": row_tiles,
        "col_tiles_per_instance": col_tiles,
        "logical_weight_tiles": logical_tiles if execution == "optical_mrr" else 0,
        "physical_load_rounds": math.ceil(logical_tiles / tiles) if execution == "optical_mrr" else 0,
        "logical_parameters": k * n * multiplicity if execution == "optical_mrr" else 0,
        "dynamic_operand": int(execution != "optical_mrr"),
    }


def build_mapping(spec: Mapping[str, Any], hardware: Mapping[str, Any]) -> Dict[str, Any]:
    w = spec["workload"]
    rows = _i(hardware.get("mapping", {}).get("array_rows"), 64)
    cols = _i(hardware.get("mapping", {}).get("array_cols"), 64)
    tiles = _i(hardware.get("photonic_tiles", {}).get("num_tiles"), 4)
    d = _i(w.get("embedding_dim"), 256)
    blocks = _i(w.get("blocks"), 2)
    ratio = _i(w.get("mlp_ratio"), 4)
    patch_k = _i(w.get("input_channels"), 2) * _i(w.get("patch_size"), 16) ** 2
    classes = _i(w.get("num_classes"), 10)

    ops: List[Dict[str, Any]] = [
        map_matrix("patch_embedding", patch_k, d, rows=rows, cols=cols, tiles=tiles),
        map_matrix("qkv_projections", d, d, rows=rows, cols=cols, tiles=tiles, multiplicity=3 * blocks),
        map_matrix("attention_qk", d, d, rows=rows, cols=cols, tiles=tiles, multiplicity=blocks, execution="electronic_exact_fallback"),
        map_matrix("attention_av", d, d, rows=rows, cols=cols, tiles=tiles, multiplicity=blocks, execution="electronic_exact_fallback"),
        map_matrix("attention_output_projection", d, d, rows=rows, cols=cols, tiles=tiles, multiplicity=blocks),
        map_matrix("mlp_expand", d, ratio * d, rows=rows, cols=cols, tiles=tiles, multiplicity=blocks),
        map_matrix("mlp_contract", ratio * d, d, rows=rows, cols=cols, tiles=tiles, multiplicity=blocks),
        map_matrix("classifier", d, classes, rows=rows, cols=cols, tiles=tiles),
    ]
    optical = [x for x in ops if x["execution"] == "optical_mrr"]
    total_ops = sum(x["multiplicity"] for x in ops)
    optical_instances = sum(x["multiplicity"] for x in optical)
    return {
        "eval_name": "eval_11_transformer_mapping",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload": dict(w),
        "hardware": {"array_rows": rows, "array_cols": cols, "physical_tiles": tiles, "signed_weight_branches": 2},
        "operators": ops,
        "summary": {
            "operator_instances": total_ops,
            "optical_static_weight_instances": optical_instances,
            "electronic_dynamic_attention_instances": total_ops - optical_instances,
            "optical_operator_coverage_percent": 100.0 * optical_instances / max(total_ops, 1),
            "logical_weight_tiles": sum(x["logical_weight_tiles"] for x in optical),
            "physical_load_rounds": sum(x["physical_load_rounds"] for x in optical),
            "logical_parameters": sum(x["logical_parameters"] for x in optical),
            "mapping_feasible_without_optical_plane_change": True,
        },
        "unsupported_claims": ["latency", "energy", "accuracy", "throughput"],
        "status": "mapping_only_requires_checkpoint_and_per_layer_activity_trace_for_performance_claims",
        "comparison_to_picosnn": "HIPSA keeps dynamic QK/AV attention in exact electronics; PICoSNN adds optical LIF/KV-SSA and dynamically configurable optical operands. The two architectures therefore should not be compared using peak optical throughput alone.",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="configs/spiking_transformer_mapping.yaml")
    p.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    p.add_argument("--output", default="results/eval_v3/combined/eval_11")
    args = p.parse_args()
    result = build_mapping(load_yaml(args.spec), load_yaml(args.hardware))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(result, out / "transformer_mapping.json")
    save_csv_rows(result["operators"], out / "transformer_operator_mapping.csv")
    print(f"[eval_11] mapping-only artifacts saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
