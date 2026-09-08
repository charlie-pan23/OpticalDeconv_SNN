"""Sample real Conv/Linear partial sums in HIPSA's 64-row tile organization."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sample_hapr_operations(
    module: nn.Module,
    x: torch.Tensor,
    *,
    group_size: int = 4,
    array_rows: int = 64,
    max_operations: int = 2048,
    current_scale_a_per_accum_unit: float = 1e-6,
) -> list[dict[str, Any]]:
    """Return workload-derived signed branch currents for sampled output sites.

    Exact fan-in counts are determined by the layer shape. Output sites are
    deterministically subsampled to bound artifact size; no magnitude-based
    selection is used.
    """
    if isinstance(module, nn.Conv2d):
        patches = F.unfold(x, module.kernel_size, dilation=module.dilation, padding=module.padding, stride=module.stride)
        # [B,K,L] -> [B*L,K]
        vectors = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        weights = module.weight.reshape(module.out_channels, -1)
    elif isinstance(module, nn.Linear):
        vectors = x.reshape(-1, x.shape[-1])
        weights = module.weight
    else:
        return []
    total_sites = int(vectors.shape[0]) * int(weights.shape[0])
    stride = max(int(math.ceil(total_sites / max(int(max_operations), 1))), 1)
    rows: list[dict[str, Any]] = []
    kdim = int(vectors.shape[1])
    row_tiles = int(math.ceil(kdim / int(array_rows)))
    for flat_index in range(0, total_sites, stride):
        site = flat_index // int(weights.shape[0])
        out_channel = flat_index % int(weights.shape[0])
        vector = vectors[site]
        weight = weights[out_channel]
        partials: list[tuple[float, float]] = []
        for start in range(0, kdim, int(array_rows)):
            xv = vector[start:start + int(array_rows)]
            wv = weight[start:start + int(array_rows)]
            positive = torch.sum(xv * torch.clamp(wv, min=0)).item()
            negative = torch.sum(xv * torch.clamp(-wv, min=0)).item()
            partials.append((float(positive), float(negative)))
        for group_start in range(0, row_tiles, int(group_size)):
            group = partials[group_start:group_start + int(group_size)]
            rows.append({
                "fan_in": len(group),
                "i_pos_a": sum(max(v[0], 0.0) for v in group) * float(current_scale_a_per_accum_unit),
                "i_neg_a": sum(max(v[1], 0.0) for v in group) * float(current_scale_a_per_accum_unit),
                "site_index": int(site),
                "output_channel": int(out_channel),
                "row_group_index": int(group_start // int(group_size)),
                "row_tiles_total": row_tiles,
                "sampling_stride": stride,
                "trace_provenance": "real_runtime_input_and_frozen_weight_partial_sum_deterministic_subsample",
            })
        if len(rows) >= int(max_operations):
            break
    return rows[: int(max_operations)]
