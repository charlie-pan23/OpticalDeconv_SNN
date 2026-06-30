"""Layer-wise SNN activity tracing for HIPSA evaluation.

This module records activity at architecture-relevant MVM layers.  It is used by
``eval/eval_01_activity_trace.py`` and intentionally avoids dataset-specific
logic.  The trace is a proxy suitable for architecture-level evaluation, not a
fabricated-chip measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


@dataclass
class LayerActivityStats:
    name: str
    module_type: str
    input_active: int = 0
    input_total: int = 0
    output_active: int = 0
    output_total: int = 0
    adc_request_active: int = 0
    adc_request_total: int = 0
    dense_sop: float = 0.0
    output_abs_sum: float = 0.0
    output_sq_sum: float = 0.0
    output_max_abs: float = 0.0
    calls: int = 0
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None

    def to_dict(self, num_samples: int) -> Dict[str, Any]:
        input_activity = self.input_active / self.input_total if self.input_total else 0.0
        output_activity = self.output_active / self.output_total if self.output_total else 0.0
        adc_activity = self.adc_request_active / self.adc_request_total if self.adc_request_total else 0.0
        active_sop = float(self.dense_sop) * float(input_activity)
        mean_abs = self.output_abs_sum / self.output_total if self.output_total else 0.0
        rms = (self.output_sq_sum / self.output_total) ** 0.5 if self.output_total else 0.0
        return {
            "name": self.name,
            "module_type": self.module_type,
            "calls": int(self.calls),
            "input_shape_last": list(self.input_shape) if self.input_shape is not None else None,
            "output_shape_last": list(self.output_shape) if self.output_shape is not None else None,
            "input_active": int(self.input_active),
            "input_total": int(self.input_total),
            "input_activity": float(input_activity),
            "output_active": int(self.output_active),
            "output_total": int(self.output_total),
            "output_activity": float(output_activity),
            "adc_request_active": int(self.adc_request_active),
            "adc_request_total": int(self.adc_request_total),
            "adc_activity_proxy": float(adc_activity),
            "dense_sop_total": float(self.dense_sop),
            "active_sop_total": float(active_sop),
            "dense_sop_per_image": float(self.dense_sop / max(num_samples, 1)),
            "active_sop_per_image": float(active_sop / max(num_samples, 1)),
            "output_mean_abs": float(mean_abs),
            "output_rms": float(rms),
            "output_max_abs": float(self.output_max_abs),
        }


class ActivityTracer:
    """Forward-hook based activity tracer for Conv2d/Linear MVM layers."""

    def __init__(
        self,
        model: nn.Module,
        target_layer_names: Optional[Sequence[str]] = None,
        include_linear: bool = True,
        include_fc2: bool = False,
        adc_threshold_abs: float = 0.0,
    ):
        self.model = model
        self.target_layer_names = set(target_layer_names) if target_layer_names else None
        self.include_linear = bool(include_linear)
        self.include_fc2 = bool(include_fc2)
        self.adc_threshold_abs = float(adc_threshold_abs)
        self.stats: Dict[str, LayerActivityStats] = {}
        self.handles: List[Any] = []
        self.input_active: int = 0
        self.input_total: int = 0
        self.num_samples: int = 0
        self.num_batches: int = 0

    def _is_target(self, name: str, module: nn.Module) -> bool:
        if self.target_layer_names is not None:
            return name in self.target_layer_names
        if isinstance(module, nn.Conv2d):
            return True
        if isinstance(module, nn.Linear) and self.include_linear:
            if name == "fc2" and not self.include_fc2:
                return False
            return True
        return False

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if self._is_target(name, module):
                self.stats[name] = LayerActivityStats(name=name, module_type=module.__class__.__name__)
                self.handles.append(module.register_forward_hook(self._make_hook(name, module)))
                logger.debug("Registered activity hook on %s", name)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self) -> None:
        for s in self.stats.values():
            s.input_active = s.input_total = 0
            s.output_active = s.output_total = 0
            s.adc_request_active = s.adc_request_total = 0
            s.dense_sop = 0.0
            s.output_abs_sum = s.output_sq_sum = 0.0
            s.output_max_abs = 0.0
            s.calls = 0
            s.input_shape = s.output_shape = None
        self.input_active = 0
        self.input_total = 0
        self.num_samples = 0
        self.num_batches = 0

    def observe_batch_input(self, data_tbc: torch.Tensor) -> None:
        """Record activity of a prepared model input [T,B,C,H,W]."""
        self.num_batches += 1
        if data_tbc.dim() >= 2:
            self.num_samples += int(data_tbc.shape[1])
        self.input_active += int(torch.count_nonzero(data_tbc).item())
        self.input_total += int(data_tbc.numel())

    def _make_hook(self, name: str, module: nn.Module):
        def hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if not inputs:
                return
            x = inputs[0]
            y = output
            if not torch.is_tensor(x) or not torch.is_tensor(y):
                return
            st = self.stats[name]
            st.calls += 1
            st.input_shape = tuple(int(v) for v in x.shape)
            st.output_shape = tuple(int(v) for v in y.shape)
            st.input_active += int(torch.count_nonzero(x).item())
            st.input_total += int(x.numel())
            st.output_active += int(torch.count_nonzero(y).item())
            st.output_total += int(y.numel())
            abs_y = y.detach().abs()
            if self.adc_threshold_abs > 0:
                adc_mask = abs_y > self.adc_threshold_abs
            else:
                adc_mask = y.detach() != 0
            st.adc_request_active += int(adc_mask.count_nonzero().item())
            st.adc_request_total += int(y.numel())
            st.output_abs_sum += float(abs_y.sum().item())
            st.output_sq_sum += float((y.detach().float() ** 2).sum().item())
            if abs_y.numel() > 0:
                st.output_max_abs = max(st.output_max_abs, float(abs_y.max().item()))
            st.dense_sop += float(dense_sop_for_call(mod, x, y))
        return hook

    def summary(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        layers = {name: stat.to_dict(self.num_samples) for name, stat in self.stats.items()}
        dense_total = float(sum(v["dense_sop_total"] for v in layers.values()))
        active_total = float(sum(v["active_sop_total"] for v in layers.values()))
        adc_active = int(sum(v["adc_request_active"] for v in layers.values()))
        adc_total = int(sum(v["adc_request_total"] for v in layers.values()))
        out: Dict[str, Any] = {
            "num_samples": int(self.num_samples),
            "num_batches": int(self.num_batches),
            "input_active_count": int(self.input_active),
            "input_total_count": int(self.input_total),
            "input_spike_activity": float(self.input_active / self.input_total) if self.input_total else 0.0,
            "dense_sop_total": dense_total,
            "active_sop_total": active_total,
            "dense_sop_per_image": dense_total / max(self.num_samples, 1),
            "active_sop_per_image": active_total / max(self.num_samples, 1),
            "active_sop_ratio": active_total / dense_total if dense_total else 0.0,
            "adc_request_active": adc_active,
            "adc_request_total": adc_total,
            "adc_activity_proxy": adc_active / adc_total if adc_total else 0.0,
            "layers": layers,
        }
        if extra:
            out.update(dict(extra))
        return out


def dense_sop_for_call(module: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Dense SOPs for one module call.

    Hooks are called once per timestep in the current models, so summing calls over
    a full test pass gives total dense SOPs over all timesteps/samples.
    """
    if isinstance(module, nn.Conv2d):
        k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        in_per_group = module.in_channels // module.groups
        return float(y.numel() * in_per_group * k_h * k_w)
    if isinstance(module, nn.Linear):
        return float(y.numel() * module.in_features)
    return 0.0


def default_photonic_layer_names(model: nn.Module, include_fc2: bool = False) -> List[str]:
    if hasattr(model, "photonic_mvm_layer_names"):
        names = [str(n) for n in getattr(model, "photonic_mvm_layer_names")]
    else:
        names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                names.append(name)
            elif isinstance(module, nn.Linear) and (include_fc2 or name != "fc2"):
                names.append(name)
    if not include_fc2:
        names = [n for n in names if n != "fc2"]
    return names
