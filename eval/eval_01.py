"""
eval_01.py

Activity trace and active SOP statistics for HIPSA evaluation.

This script collects data only. It does not generate figures.

Outputs:
  results/eval_v2/<dataset>/eval_01/
    summary.json
    sop_summary.json
    layer_activity.csv
    timestep_activity.csv
    config_snapshot.yaml

Key definitions:
  model_input_activity:
    Non-zero activity of the prepared model input [T,B,C,H,W].

  mvm_input_activity:
    Non-zero activity at each Conv/Linear input.
    This is the primary signal used for active SOP counting.

  mvm_output_nonzero_activity:
    Non-zero ratio of raw Conv/Linear output.
    Debug only. Do not interpret as spike activity.

  lif_spike_activity:
    Non-zero activity of the following LIF output.
    Used as digital spike traffic / NoC proxy.

  adc_request_activity:
    Comparator-style request proxy from raw Conv/Linear output.
    Used for request-driven ADC modeling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from eval.eval_utils import (
    build_eval_model,
    build_loader,
    load_eval_context,
    save_csv_rows,
)
from utils.config_utils import (
    copy_config_snapshot,
    get_by_path,
    save_json,
    time_steps,
)
from utils.data_utils import (
    accuracy_from_logits,
    aggregate_time_logits,
    logits_aggregation_from_config,
    prepare_snn_batch,
    reset_snn_state,
)
from utils.seed_utils import set_seed

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("HIPSA")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _as_tensor(obj: Any) -> Optional[torch.Tensor]:
    """Extract a tensor from a forward hook input/output payload."""

    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (tuple, list)):
        for item in obj:
            if torch.is_tensor(item):
                return item

    return None


def _count_nonzero(x: torch.Tensor) -> int:
    return int(torch.count_nonzero(x).item())


def _safe_numel(x: Optional[torch.Tensor]) -> int:
    return int(x.numel()) if x is not None else 0


def _ratio(active: int | float, total: int | float) -> float:
    return float(active) / float(total) if float(total) > 0 else 0.0


def _module_type(module: nn.Module) -> str:
    return module.__class__.__name__


def _is_mvm_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Linear))


def _is_lif_module(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return "lif" in name


def _dense_sop_from_module(module: nn.Module, output: torch.Tensor) -> float:
    """Dense SOP count for one Conv/Linear call.

    Conv2d:
      output elements * input channels per output * kernel_h * kernel_w

    Linear:
      output elements * input features
    """

    if isinstance(module, nn.Conv2d):
        k_h, k_w = module.kernel_size
        in_per_out = (module.in_channels // module.groups) * k_h * k_w
        return float(output.numel() * in_per_out)

    if isinstance(module, nn.Linear):
        return float(output.numel() * module.in_features)

    return 0.0


def _default_photonic_layer_names(model: nn.Module, include_fc2: bool = False) -> List[str]:
    """Resolve HIPSA photonic MVM layers.

    Prefer model.photonic_mvm_layer_names.
    Fallback to all Conv2d/Linear modules.
    """

    if hasattr(model, "photonic_mvm_layer_names"):
        names = list(getattr(model, "photonic_mvm_layer_names"))
    else:
        names = [
            name
            for name, module in model.named_modules()
            if _is_mvm_module(module)
        ]

    if include_fc2 and "fc2" in dict(model.named_modules()) and "fc2" not in names:
        names.append("fc2")

    if not include_fc2:
        names = [name for name in names if name != "fc2"]

    return names


def _default_lif_names(model: nn.Module) -> List[str]:
    return [
        name
        for name, module in model.named_modules()
        if _is_lif_module(module)
    ]


def _map_mvm_to_lif(
    target_layer_names: Sequence[str],
    lif_names: Sequence[str],
) -> Dict[str, Optional[str]]:
    """Map each MVM layer to the following LIF layer by order.

    The current models are sequential:
      conv1 -> lif1, conv2 -> lif2, ..., fc1 -> lifN.

    fc2 has no following LIF and is mapped to None.
    """

    mapping: Dict[str, Optional[str]] = {}

    lif_idx = 0
    for layer in target_layer_names:
        if layer == "fc2":
            mapping[layer] = None
            continue

        if lif_idx < len(lif_names):
            mapping[layer] = lif_names[lif_idx]
            lif_idx += 1
        else:
            mapping[layer] = None

    return mapping


def _nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def default_adc_threshold_fs(hardware_cfg: Mapping[str, Any]) -> float:
    return float(
        _nested_get(
            hardware_cfg,
            "hapr_adc_backend",
            "comparator_threshold_fs_default",
            default=0.02,
        )
        or 0.02
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class TimestepStats:
    dense_sop_total: float = 0.0
    active_sop_total: float = 0.0

    mvm_input_active: int = 0
    mvm_input_total: int = 0

    mvm_output_nonzero_active: int = 0
    mvm_output_total: int = 0

    adc_request_active: int = 0
    adc_request_total: int = 0

    lif_spike_active: int = 0
    lif_spike_total: int = 0

    calls: int = 0
    lif_calls: int = 0

    def add_mvm(
        self,
        dense_sop: float,
        active_sop: float,
        input_active: int,
        input_total: int,
        output_nonzero_active: int,
        output_total: int,
        adc_active: int,
        adc_total: int,
    ) -> None:
        self.dense_sop_total += float(dense_sop)
        self.active_sop_total += float(active_sop)
        self.mvm_input_active += int(input_active)
        self.mvm_input_total += int(input_total)
        self.mvm_output_nonzero_active += int(output_nonzero_active)
        self.mvm_output_total += int(output_total)
        self.adc_request_active += int(adc_active)
        self.adc_request_total += int(adc_total)
        self.calls += 1

    def add_lif(self, spike_active: int, spike_total: int) -> None:
        self.lif_spike_active += int(spike_active)
        self.lif_spike_total += int(spike_total)
        self.lif_calls += 1

    def to_dict(self, num_samples: int) -> Dict[str, Any]:
        return {
            "calls": int(self.calls),
            "lif_calls": int(self.lif_calls),

            "dense_sop_total": float(self.dense_sop_total),
            "active_sop_total": float(self.active_sop_total),
            "dense_sop_per_image": float(self.dense_sop_total / max(num_samples, 1)),
            "active_sop_per_image": float(self.active_sop_total / max(num_samples, 1)),
            "active_sop_ratio": _ratio(self.active_sop_total, self.dense_sop_total),

            "mvm_input_active": int(self.mvm_input_active),
            "mvm_input_total": int(self.mvm_input_total),
            "mvm_input_activity": _ratio(self.mvm_input_active, self.mvm_input_total),

            "mvm_output_nonzero_active": int(self.mvm_output_nonzero_active),
            "mvm_output_total": int(self.mvm_output_total),
            "mvm_output_nonzero_activity": _ratio(
                self.mvm_output_nonzero_active,
                self.mvm_output_total,
            ),

            "adc_request_active": int(self.adc_request_active),
            "adc_request_total": int(self.adc_request_total),
            "adc_request_activity": _ratio(
                self.adc_request_active,
                self.adc_request_total,
            ),

            "lif_spike_active": int(self.lif_spike_active),
            "lif_spike_total": int(self.lif_spike_total),
            "lif_spike_activity": _ratio(
                self.lif_spike_active,
                self.lif_spike_total,
            ),
        }


@dataclass
class LayerStats:
    name: str
    module_type: str
    mapped_lif_name: Optional[str] = None

    input_shape_last: Optional[Tuple[int, ...]] = None
    output_shape_last: Optional[Tuple[int, ...]] = None
    lif_output_shape_last: Optional[Tuple[int, ...]] = None

    calls: int = 0
    lif_calls: int = 0

    dense_sop_total: float = 0.0
    active_sop_total: float = 0.0

    mvm_input_active: int = 0
    mvm_input_total: int = 0

    mvm_output_nonzero_active: int = 0
    mvm_output_total: int = 0

    adc_request_active: int = 0
    adc_request_total: int = 0

    lif_spike_active: int = 0
    lif_spike_total: int = 0

    output_abs_sum: float = 0.0
    output_sq_sum: float = 0.0
    output_max_abs: float = 0.0

    by_timestep: Dict[int, TimestepStats] = field(default_factory=dict)

    def _ts(self, timestep: int) -> TimestepStats:
        if timestep not in self.by_timestep:
            self.by_timestep[timestep] = TimestepStats()
        return self.by_timestep[timestep]

    def add_mvm_call(
        self,
        timestep: int,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        module: nn.Module,
        adc_threshold_abs: float,
        adc_threshold_fs: float,
    ) -> None:
        input_active = _count_nonzero(input_tensor)
        input_total = int(input_tensor.numel())

        out_abs = output_tensor.detach().abs()
        output_total = int(out_abs.numel())
        output_nonzero_active = int((out_abs > 0).sum().item()) if output_total else 0

        max_abs = float(out_abs.max().item()) if output_total else 0.0
        threshold = max(float(adc_threshold_abs), float(adc_threshold_fs) * max_abs)

        if output_total == 0:
            adc_active = 0
        elif threshold <= 0:
            adc_active = output_nonzero_active
        else:
            adc_active = int((out_abs > threshold).sum().item())

        dense_sop = _dense_sop_from_module(module, output_tensor)
        mvm_activity = _ratio(input_active, input_total)
        active_sop = dense_sop * mvm_activity

        self.calls += 1
        self.input_shape_last = tuple(int(v) for v in input_tensor.shape)
        self.output_shape_last = tuple(int(v) for v in output_tensor.shape)

        self.dense_sop_total += float(dense_sop)
        self.active_sop_total += float(active_sop)

        self.mvm_input_active += int(input_active)
        self.mvm_input_total += int(input_total)

        self.mvm_output_nonzero_active += int(output_nonzero_active)
        self.mvm_output_total += int(output_total)

        self.adc_request_active += int(adc_active)
        self.adc_request_total += int(output_total)

        self.output_abs_sum += float(out_abs.sum().item()) if output_total else 0.0
        self.output_sq_sum += float((out_abs * out_abs).sum().item()) if output_total else 0.0
        self.output_max_abs = max(self.output_max_abs, max_abs)

        self._ts(timestep).add_mvm(
            dense_sop=dense_sop,
            active_sop=active_sop,
            input_active=input_active,
            input_total=input_total,
            output_nonzero_active=output_nonzero_active,
            output_total=output_total,
            adc_active=adc_active,
            adc_total=output_total,
        )

    def add_lif_call(
        self,
        timestep: int,
        lif_output: torch.Tensor,
    ) -> None:
        spike_active = _count_nonzero(lif_output)
        spike_total = int(lif_output.numel())

        self.lif_calls += 1
        self.lif_output_shape_last = tuple(int(v) for v in lif_output.shape)

        self.lif_spike_active += int(spike_active)
        self.lif_spike_total += int(spike_total)

        self._ts(timestep).add_lif(spike_active, spike_total)

    def to_dict(self, num_samples: int) -> Dict[str, Any]:
        mean_abs = (
            self.output_abs_sum / self.mvm_output_total
            if self.mvm_output_total
            else 0.0
        )
        rms = (
            math.sqrt(self.output_sq_sum / self.mvm_output_total)
            if self.mvm_output_total
            else 0.0
        )

        return {
            "name": self.name,
            "module_type": self.module_type,
            "mapped_lif_name": self.mapped_lif_name,

            "calls": int(self.calls),
            "lif_calls": int(self.lif_calls),

            "input_shape_last": list(self.input_shape_last) if self.input_shape_last else None,
            "output_shape_last": list(self.output_shape_last) if self.output_shape_last else None,
            "lif_output_shape_last": (
                list(self.lif_output_shape_last)
                if self.lif_output_shape_last
                else None
            ),

            "dense_sop_total": float(self.dense_sop_total),
            "active_sop_total": float(self.active_sop_total),
            "dense_sop_per_image": float(self.dense_sop_total / max(num_samples, 1)),
            "active_sop_per_image": float(self.active_sop_total / max(num_samples, 1)),
            "active_sop_ratio": _ratio(self.active_sop_total, self.dense_sop_total),

            "mvm_input_active": int(self.mvm_input_active),
            "mvm_input_total": int(self.mvm_input_total),
            "mvm_input_activity": _ratio(self.mvm_input_active, self.mvm_input_total),

            "mvm_output_nonzero_active": int(self.mvm_output_nonzero_active),
            "mvm_output_total": int(self.mvm_output_total),
            "mvm_output_nonzero_activity": _ratio(
                self.mvm_output_nonzero_active,
                self.mvm_output_total,
            ),

            "adc_request_active": int(self.adc_request_active),
            "adc_request_total": int(self.adc_request_total),
            "adc_request_activity": _ratio(
                self.adc_request_active,
                self.adc_request_total,
            ),

            "lif_spike_active": int(self.lif_spike_active),
            "lif_spike_total": int(self.lif_spike_total),
            "lif_spike_activity": _ratio(
                self.lif_spike_active,
                self.lif_spike_total,
            ),

            "output_mean_abs": float(mean_abs),
            "output_rms": float(rms),
            "output_max_abs": float(self.output_max_abs),

            # Backward-compatible aliases for old analysis notebooks.
            "input_activity": _ratio(self.mvm_input_active, self.mvm_input_total),
            "output_activity": _ratio(self.lif_spike_active, self.lif_spike_total),
            "adc_activity_proxy": _ratio(
                self.adc_request_active,
                self.adc_request_total,
            ),
        }


class ActivityCollector:
    """Forward-hook based activity collector for eval_01."""

    def __init__(
        self,
        model: nn.Module,
        target_layer_names: Sequence[str],
        time_steps: int,
        adc_threshold_abs: float,
        adc_threshold_fs: float,
        trace_lif: bool = True,
    ) -> None:
        self.model = model
        self.target_layer_names = list(target_layer_names)
        self.time_steps = int(time_steps)
        self.adc_threshold_abs = float(adc_threshold_abs)
        self.adc_threshold_fs = float(adc_threshold_fs)
        self.trace_lif = bool(trace_lif)

        self.modules = dict(model.named_modules())
        self.lif_names = _default_lif_names(model)
        self.mvm_to_lif = _map_mvm_to_lif(self.target_layer_names, self.lif_names)

        self.layer_stats: Dict[str, LayerStats] = {}
        for name in self.target_layer_names:
            if name not in self.modules:
                raise KeyError(f"Target layer not found in model: {name}")

            module = self.modules[name]
            if not _is_mvm_module(module):
                raise TypeError(
                    f"Target layer {name} is not Conv2d/Linear: {_module_type(module)}"
                )

            self.layer_stats[name] = LayerStats(
                name=name,
                module_type=_module_type(module),
                mapped_lif_name=self.mvm_to_lif.get(name),
            )

        self.handles: List[RemovableHandle] = []

        self.num_batches = 0
        self.num_samples = 0

        self.model_input_active = 0
        self.model_input_total = 0
        self.model_input_by_timestep: Dict[int, Dict[str, int]] = {}

    def register(self) -> None:
        """Register MVM and LIF hooks."""

        for name in self.target_layer_names:
            module = self.modules[name]

            def mvm_hook(
                mod: nn.Module,
                inputs: Tuple[Any, ...],
                output: Any,
                layer_name: str = name,
            ) -> None:
                self._on_mvm(layer_name, mod, inputs, output)

            self.handles.append(module.register_forward_hook(mvm_hook))

        if self.trace_lif:
            lif_to_mvm = {
                lif_name: mvm_name
                for mvm_name, lif_name in self.mvm_to_lif.items()
                if lif_name is not None and lif_name in self.modules
            }

            for lif_name, mvm_name in lif_to_mvm.items():
                module = self.modules[lif_name]

                def lif_hook(
                    mod: nn.Module,
                    inputs: Tuple[Any, ...],
                    output: Any,
                    layer_name: str = mvm_name,
                ) -> None:
                    self._on_lif(layer_name, output)

                self.handles.append(module.register_forward_hook(lif_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def observe_model_input(self, x: torch.Tensor) -> None:
        """Record prepared model input activity.

        Expected shape: [T,B,C,H,W].
        """

        self.num_batches += 1

        if x.dim() >= 2:
            # [T,B,...]
            self.num_samples += int(x.shape[1])
        else:
            self.num_samples += int(x.shape[0])

        active = _count_nonzero(x)
        total = int(x.numel())

        self.model_input_active += active
        self.model_input_total += total

        if x.dim() >= 2 and x.shape[0] == self.time_steps:
            for t in range(self.time_steps):
                x_t = x[t]
                ts = self.model_input_by_timestep.setdefault(
                    t,
                    {"active": 0, "total": 0},
                )
                ts["active"] += _count_nonzero(x_t)
                ts["total"] += int(x_t.numel())

    def _on_mvm(
        self,
        layer_name: str,
        module: nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> None:
        x = _as_tensor(inputs)
        y = _as_tensor(output)

        if x is None or y is None:
            return

        stats = self.layer_stats[layer_name]
        timestep = stats.calls % max(self.time_steps, 1)

        stats.add_mvm_call(
            timestep=timestep,
            input_tensor=x.detach(),
            output_tensor=y.detach(),
            module=module,
            adc_threshold_abs=self.adc_threshold_abs,
            adc_threshold_fs=self.adc_threshold_fs,
        )

    def _on_lif(self, layer_name: str, output: Any) -> None:
        y = _as_tensor(output)
        if y is None:
            return

        stats = self.layer_stats[layer_name]
        timestep = stats.lif_calls % max(self.time_steps, 1)
        stats.add_lif_call(timestep=timestep, lif_output=y.detach())

    def _aggregate_timestep_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for layer_name, stats in self.layer_stats.items():
            for timestep in range(self.time_steps):
                ts = stats.by_timestep.get(timestep, TimestepStats())
                row = {
                    "dataset": "",
                    "layer": layer_name,
                    "timestep": int(timestep),
                    **ts.to_dict(self.num_samples),
                }
                rows.append(row)

        return rows

    def layer_rows(self, dataset: str) -> List[Dict[str, Any]]:
        rows = []
        for name, stats in self.layer_stats.items():
            rows.append(
                {
                    "dataset": dataset,
                    **stats.to_dict(self.num_samples),
                }
            )
        return rows

    def timestep_rows(self, dataset: str) -> List[Dict[str, Any]]:
        rows = self._aggregate_timestep_rows()
        for row in rows:
            row["dataset"] = dataset
        return rows

    def summary(
        self,
        *,
        dataset: str,
        config: str,
        checkpoint: str,
        split: str,
        accuracy_percent: Optional[float],
        loss: Optional[float],
        command: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        layers = {
            name: stats.to_dict(self.num_samples)
            for name, stats in self.layer_stats.items()
        }

        dense_sop_total = sum(v["dense_sop_total"] for v in layers.values())
        active_sop_total = sum(v["active_sop_total"] for v in layers.values())

        mvm_input_active = sum(v["mvm_input_active"] for v in layers.values())
        mvm_input_total = sum(v["mvm_input_total"] for v in layers.values())

        lif_spike_active = sum(v["lif_spike_active"] for v in layers.values())
        lif_spike_total = sum(v["lif_spike_total"] for v in layers.values())

        adc_request_active = sum(v["adc_request_active"] for v in layers.values())
        adc_request_total = sum(v["adc_request_total"] for v in layers.values())

        mvm_output_nonzero_active = sum(
            v["mvm_output_nonzero_active"] for v in layers.values()
        )
        mvm_output_total = sum(v["mvm_output_total"] for v in layers.values())

        model_input_ts = []
        for t in range(self.time_steps):
            item = self.model_input_by_timestep.get(t, {"active": 0, "total": 0})
            model_input_ts.append(
                {
                    "timestep": int(t),
                    "active": int(item["active"]),
                    "total": int(item["total"]),
                    "activity": _ratio(item["active"], item["total"]),
                }
            )

        summary = {
            "eval_name": "eval_01",
            "purpose": "activity_trace_and_active_sop_statistics",
            "created_utc": _now_utc(),
            "command": command,

            "dataset": dataset,
            "split": split,
            "config": config,
            "checkpoint": checkpoint,

            "num_batches": int(self.num_batches),
            "num_samples": int(self.num_samples),
            "time_steps": int(self.time_steps),
            "target_layers": list(self.target_layer_names),
            "mapped_lif_layers": dict(self.mvm_to_lif),

            "accuracy_percent": accuracy_percent,
            "loss": loss,

            "adc_threshold_abs": float(self.adc_threshold_abs),
            "adc_threshold_fs": float(self.adc_threshold_fs),

            "model_input_active": int(self.model_input_active),
            "model_input_total": int(self.model_input_total),
            "model_input_activity": _ratio(
                self.model_input_active,
                self.model_input_total,
            ),
            "model_input_by_timestep": model_input_ts,

            "dense_sop_total": float(dense_sop_total),
            "active_sop_total": float(active_sop_total),
            "dense_sop_per_image": float(dense_sop_total / max(self.num_samples, 1)),
            "active_sop_per_image": float(active_sop_total / max(self.num_samples, 1)),
            "active_sop_ratio": _ratio(active_sop_total, dense_sop_total),

            "mvm_input_active": int(mvm_input_active),
            "mvm_input_total": int(mvm_input_total),
            "mvm_input_activity": _ratio(mvm_input_active, mvm_input_total),

            "mvm_output_nonzero_active": int(mvm_output_nonzero_active),
            "mvm_output_total": int(mvm_output_total),
            "mvm_output_nonzero_activity": _ratio(
                mvm_output_nonzero_active,
                mvm_output_total,
            ),

            "lif_spike_active": int(lif_spike_active),
            "lif_spike_total": int(lif_spike_total),
            "lif_spike_activity": _ratio(lif_spike_active, lif_spike_total),

            "adc_request_active": int(adc_request_active),
            "adc_request_total": int(adc_request_total),
            "adc_request_activity": _ratio(adc_request_active, adc_request_total),

            "layers": layers,

            "notes": {
                "mvm_input_activity": "Primary signal used for active SOP counting.",
                "mvm_output_nonzero_activity": "Debug only; do not interpret as spike rate.",
                "lif_spike_activity": "Post-LIF spike activity used for digital/NoC proxy.",
                "adc_request_activity": "Comparator threshold proxy used for request-driven ADC modeling.",
                "adc_threshold_rule": "threshold = max(adc_threshold_abs, adc_threshold_fs * per-call max(abs(MVM output))).",
                "fc2_default": "fc2 is excluded by default unless --include-fc2 is passed.",
            },
        }

        if extra:
            summary["extra"] = dict(extra)

        return summary


def make_sop_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    layers = summary.get("layers", {})
    layer_rows = []

    if isinstance(layers, Mapping):
        for name, info in layers.items():
            if not isinstance(info, Mapping):
                continue
            layer_rows.append(
                {
                    "layer": name,
                    "dense_sop_total": float(info.get("dense_sop_total", 0.0)),
                    "active_sop_total": float(info.get("active_sop_total", 0.0)),
                    "dense_sop_per_image": float(info.get("dense_sop_per_image", 0.0)),
                    "active_sop_per_image": float(info.get("active_sop_per_image", 0.0)),
                    "active_sop_ratio": float(info.get("active_sop_ratio", 0.0)),
                    "mvm_input_activity": float(info.get("mvm_input_activity", 0.0)),
                    "lif_spike_activity": float(info.get("lif_spike_activity", 0.0)),
                    "adc_request_activity": float(info.get("adc_request_activity", 0.0)),
                }
            )

    return {
        "dataset": summary.get("dataset"),
        "num_samples": summary.get("num_samples"),
        "time_steps": summary.get("time_steps"),
        "dense_sop_total": summary.get("dense_sop_total"),
        "active_sop_total": summary.get("active_sop_total"),
        "dense_sop_per_image": summary.get("dense_sop_per_image"),
        "active_sop_per_image": summary.get("active_sop_per_image"),
        "active_sop_ratio": summary.get("active_sop_ratio"),
        "mvm_input_activity": summary.get("mvm_input_activity"),
        "lif_spike_activity": summary.get("lif_spike_activity"),
        "adc_request_activity": summary.get("adc_request_activity"),
        "layers": layer_rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_01: activity trace")

    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--run-dir", default=None, type=str)

    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml", type=str)
    parser.add_argument("--device-params", default="configs/device_params.yaml", type=str)

    parser.add_argument("--output-root", default="results/eval_v2", type=str)
    parser.add_argument("--output-dir", default=None, type=str)

    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--max-batches", default=None, type=int)

    parser.add_argument("--adc-threshold-abs", default=0.0, type=float)
    parser.add_argument(
        "--adc-threshold-fs",
        default=None,
        type=float,
        help="Comparator threshold as a fraction of per-call full scale. "
             "Defaults to hardware config value or 0.02.",
    )

    parser.add_argument("--include-fc2", action="store_true")
    parser.add_argument("--no-lif-trace", action="store_true")
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--non-strict", action="store_true")

    return parser.parse_args()


@torch.inference_mode()
def run_eval_01(args: argparse.Namespace) -> None:
    ctx = load_eval_context(
        config_path=args.config,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        hardware_config=args.hardware,
        device_params=args.device_params,
        output_dir=args.output_dir,
        output_root=args.output_root,
        eval_name="eval_01",
        device=args.device,
    )

    seed = int(
        ctx.config.get("experiment", {}).get(
            "seed",
            ctx.config.get("split", {}).get("seed", 42),
        )
    )
    set_seed(seed, deterministic=True, benchmark=False)

    model = build_eval_model(ctx, strict=not args.non_strict)

    loader = build_loader(
        ctx.config,
        split_name=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_no_split=args.allow_no_split,
        shuffle=False,
    )

    target_layer_names = _default_photonic_layer_names(
        model,
        include_fc2=args.include_fc2,
    )

    threshold_fs = (
        float(args.adc_threshold_fs)
        if args.adc_threshold_fs is not None
        else default_adc_threshold_fs(ctx.hardware_cfg)
    )

    collector = ActivityCollector(
        model=model,
        target_layer_names=target_layer_names,
        time_steps=time_steps(ctx.config),
        adc_threshold_abs=args.adc_threshold_abs,
        adc_threshold_fs=threshold_fs,
        trace_lif=not args.no_lif_trace,
    )

    collector.register()

    agg_mode = logits_aggregation_from_config(ctx.config)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    correct = 0
    total = 0
    loss_sum = 0.0

    logger.info("=== eval_01 activity trace ===")
    logger.info("Dataset: %s", ctx.dataset)
    logger.info("Target MVM layers: %s", target_layer_names)
    logger.info("LIF layers: %s", _default_lif_names(model))
    logger.info("ADC threshold: abs=%s, fs=%s", args.adc_threshold_abs, threshold_fs)

    try:
        for batch_idx, (data, target) in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            reset_snn_state(model)
            data, target = prepare_snn_batch(data, target, ctx.config, ctx.device)

            collector.observe_model_input(data)

            logits_t = model(data)
            logits = aggregate_time_logits(logits_t, agg_mode)

            loss = criterion(logits, target)
            c, n = accuracy_from_logits(logits, target)

            correct += int(c)
            total += int(n)
            loss_sum += float(loss.item())
    finally:
        collector.remove()

    accuracy_percent = 100.0 * correct / max(total, 1)
    loss_value = loss_sum / max(total, 1)

    summary = collector.summary(
        dataset=ctx.dataset,
        config=str(Path(args.config)),
        checkpoint=str(ctx.checkpoint),
        split=args.split,
        accuracy_percent=accuracy_percent,
        loss=loss_value,
        command=" ".join(sys.argv),
        extra={
            "device": str(ctx.device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "include_fc2": bool(args.include_fc2),
            "trace_lif": bool(not args.no_lif_trace),
        },
    )

    sop_summary = make_sop_summary(summary)
    layer_rows = collector.layer_rows(ctx.dataset)
    timestep_rows = collector.timestep_rows(ctx.dataset)

    save_json(summary, ctx.output_dir / "summary.json")
    save_json(sop_summary, ctx.output_dir / "sop_summary.json")
    save_csv_rows(layer_rows, ctx.output_dir / "layer_activity.csv")
    save_csv_rows(timestep_rows, ctx.output_dir / "timestep_activity.csv")

    config_paths = [Path(args.config)]
    if Path(args.hardware).exists():
        config_paths.append(Path(args.hardware))
    if Path(args.device_params).exists():
        config_paths.append(Path(args.device_params))

    copy_config_snapshot(
        config_paths=config_paths,
        output_dir=ctx.output_dir,
        snapshot_name="config_snapshot.yaml",
        merged_config=ctx.config,
    )

    print("=" * 80)
    print("[eval_01] activity trace complete")
    print(f"dataset              : {ctx.dataset}")
    print(f"split                : {args.split}")
    print(f"samples              : {summary['num_samples']}")
    print(f"accuracy             : {accuracy_percent:.2f}%")
    print(f"loss                 : {loss_value:.6f}")
    print(f"dense SOP / image    : {summary['dense_sop_per_image']:.6e}")
    print(f"active SOP / image   : {summary['active_sop_per_image']:.6e}")
    print(f"active SOP ratio     : {summary['active_sop_ratio']:.4%}")
    print(f"model input activity : {summary['model_input_activity']:.4%}")
    print(f"MVM input activity   : {summary['mvm_input_activity']:.4%}")
    print(f"LIF spike activity   : {summary['lif_spike_activity']:.4%}")
    print(f"ADC request activity : {summary['adc_request_activity']:.4%}")
    print(f"output_dir           : {ctx.output_dir}")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    run_eval_01(args)


if __name__ == "__main__":
    main()