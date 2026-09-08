"""Auditable mapping-only breadth analysis for advanced SNN families."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.result_io import load_yaml, save_csv_rows, save_json


def matrix(name: str, k: int, n: int, multiplicity: int, tiles: int, execution: str) -> Dict[str, Any]:
    logical_tiles = math.ceil(k / 64) * math.ceil(n / 64) * multiplicity if execution == "optical_mrr" else 0
    return {
        "operator": name,
        "multiplicity": multiplicity,
        "matrix_k": k,
        "matrix_n": n,
        "execution": execution,
        "logical_weight_tiles": logical_tiles,
        "four_tile_load_rounds": math.ceil(logical_tiles / tiles) if logical_tiles else 0,
    }


def map_sew(w: Dict[str, Any], tiles: int) -> List[Dict[str, Any]]:
    channels = [int(x) for x in w["stage_channels"]]
    blocks = [int(x) for x in w["blocks_per_stage"]]
    rows = [matrix("stem_conv", int(w["input_channels"]) * int(w["stem_kernel"]) ** 2, channels[0], 1, tiles, "optical_mrr")]
    cin = channels[0]
    for stage, (cout, count) in enumerate(zip(channels, blocks), start=1):
        rows.append(matrix(f"stage{stage}_conv3x3", cin * 9, cout, 1, tiles, "optical_mrr"))
        rows.append(matrix(f"stage{stage}_conv3x3", cout * 9, cout, 2 * count - 1, tiles, "optical_mrr"))
        if cin != cout:
            rows.append(matrix(f"stage{stage}_projection_shortcut", cin, cout, 1, tiles, "optical_mrr"))
        rows.append(matrix(f"stage{stage}_sew_residual", cout, cout, count, tiles, "electronic_spike_elementwise"))
        cin = cout
    rows.append(matrix("classifier", cin, int(w["classes"]), 1, tiles, "optical_mrr"))
    return rows


def map_sdt(w: Dict[str, Any], tiles: int) -> List[Dict[str, Any]]:
    d, b, ratio = int(w["embedding_dim"]), int(w["blocks"]), int(w["mlp_ratio"])
    patch_k = int(w["input_channels"]) * int(w["patch_size"]) ** 2
    return [
        matrix("patch_embedding", patch_k, d, 1, tiles, "optical_mrr"),
        matrix("qkv_projections", d, d, 3 * b, tiles, "optical_mrr"),
        matrix("sdsa_mask_add", d, d, b, tiles, "electronic_spike_mask_add"),
        matrix("output_projection", d, d, b, tiles, "optical_mrr"),
        matrix("mlp_expand", d, ratio * d, b, tiles, "optical_mrr"),
        matrix("mlp_contract", ratio * d, d, b, tiles, "optical_mrr"),
        matrix("classifier", d, int(w["classes"]), 1, tiles, "optical_mrr"),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="configs/advanced_snn_workloads.yaml")
    p.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    p.add_argument("--output", default="results/eval_v6/combined/eval_18")
    args = p.parse_args()
    spec, hw = load_yaml(args.spec), load_yaml(args.hardware)
    tiles = int(hw["photonic_tiles"]["num_tiles"])
    summaries, detail = [], []
    for workload in spec["workloads"]:
        ops = map_sew(workload, tiles) if workload["family"] == "residual_convolutional_snn" else map_sdt(workload, tiles)
        optical = [x for x in ops if x["execution"] == "optical_mrr"]
        total_instances = sum(int(x["multiplicity"]) for x in ops)
        optical_instances = sum(int(x["multiplicity"]) for x in optical)
        summary = {
            "workload": workload["name"],
            "family": workload["family"],
            "source_url": workload["source_url"],
            "operator_instances": total_instances,
            "optical_static_weight_instances": optical_instances,
            "optical_operator_coverage_percent": 100.0 * optical_instances / total_instances,
            "logical_weight_tiles": sum(int(x["logical_weight_tiles"]) for x in optical),
            "four_tile_load_rounds": sum(int(x["four_tile_load_rounds"]) for x in optical),
            "status": "mapping_only_no_accuracy_latency_or_energy_claim",
        }
        summaries.append(summary)
        detail.extend({"workload": workload["name"], **row} for row in ops)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json({"evidence_level": "mapping_only", "summaries": summaries, "unsupported_claims": ["accuracy", "latency", "energy", "throughput"]}, out / "advanced_snn_mapping.json")
    save_csv_rows(summaries, out / "workload_summary.csv")
    save_csv_rows(detail, out / "operator_mapping.csv")
    print(f"[eval_18] saved {len(summaries)} mapping-only workloads to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
