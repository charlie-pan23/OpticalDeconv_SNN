"""Layer-wise SNN activity tracing for HIPSA evaluation.

Version 2 fixes the main ambiguity in the earlier trace: Conv/Linear output
non-zero ratio is *not* the same as LIF spike activity.  This tracer therefore
records three separate architecture signals for each photonic MVM layer:

1. mvm_input_activity
   Non-zero activity at the Conv/Linear input. Used for active SOP counting.
2. mvm_output_nonzero_activity
   Non-zero activity of raw Conv/Linear analog output. Debug only.
3. lif_spike_activity
   Non-zero activity of the following LIFNode output. Used as spike traffic / digital proxy.
4. adc_request_activity
   Comparator-style request proxy from raw Conv/Linear output after thresholding.

The module remains an architecture-level proxy, not a fabricated-chip
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

    # MVM input activity: used for active SOP.
    mvm_input_active: int = 0
    mvm_input_total: int = 0

    # Raw Conv/Linear output nonzero activity: debug only, not LIF spike rate.
    mvm_output_nonzero_active: int = 0
    mvm_output_total: int = 0

    # Comparator/ADC request proxy from raw analog MVM output.
    adc_request_active: int = 0
    adc_request_total: int = 0

    # Following LIF output spike activity.
    lif_spike_active: int = 0
    lif_spike_total: int = 0

    dense_sop: float = 0.0
    output_abs_sum: float = 0.0
    output_sq_sum: float = 0.0
    output_max_abs: float = 0.0
    calls: int = 0
    lif_calls: int = 0
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    lif_output_shape: Optional[Tuple[int, ...]] = None
    mapped_lif_name: Optional[str] = None

    def _ratio(self, active: int, total: int) -> float:
        return float(active / total) if total else 0.0

    def to_dict(self, num_samples: int) -> Dict[str, Any]:
        mvm_input_activity = self._ratio(self.mvm_input_active, self.mvm_input_total)
        mvm_output_nonzero_activity = self._ratio(self.mvm_output_nonzero_active, self.mvm_output_total)
        adc_request_activity = self._ratio(self.adc_request_active, self.adc_request_total)
        lif_spike_activity = self._ratio(self.lif_spike_active, self.lif_spike_total)
        active_sop = float(self.dense_sop) * float(mvm_input_activity)
        mean_abs = self.output_abs_sum / self.mvm_output_total if self.mvm_output_total else 0.0
        rms = (self.output_sq_sum / self.mvm_output_total) ** 0.5 if self.mvm_output_total else 0.0
        return {
            "name": self.name,
            "module_type": self.module_type,
            "mapped_lif_name": self.mapped_lif_name,
            "calls": int(self.calls),
            "lif_calls": int(self.lif_calls),
            "input_shape_last": list(self.input_shape) if self.input_shape is not None else None,
            "output_shape_last": list(self.output_shape) if self.output_shape is not None else None,
            "lif_output_shape_last": list(self.lif_output_shape) if self.lif_output_shape is not None else None,

            # Preferred field names.
            "mvm_input_active": int(self.mvm_input_active),
            "mvm_input_total": int(self.mvm_input_total),
            "mvm_input_activity": float(mvm_input_activity),
            "mvm_output_nonzero_active": int(self.mvm_output_nonzero_active),
            "mvm_output_total": int(self.mvm_output_total),
            "mvm_output_nonzero_activity": float(mvm_output_nonzero_activity),
            "adc_request_active": int(self.adc_request_active),
            "adc_request_total": int(self.adc_request_total),
            "adc_request_activity": float(adc_request_activity),
            "lif_spike_active": int(self.lif_spike_active),
            "lif_spike_total": int(self.lif_spike_total),
            "lif_spike_activity": float(lif_spike_activity),

            # Backward-compatible aliases. In v2 output_activity means LIF spike activity.
            "input_active": int(self.mvm_input_active),
            "input_total": int(self.mvm_input_total),
            "input_activity": float(mvm_input_activity),
            "output_active": int(self.lif_spike_active),
            "output_total": int(self.lif_spike_total),
            "output_activity": float(lif_spike_activity),
            "adc_activity_proxy": float(adc_request_activity),

            "dense_sop_total": float(self.dense_sop),
            "active_sop_total": float(active_sop),
            "dense_sop_per_image": float(self.dense_sop / max(num_samples, 1)),
            "active_sop_per_image": float(active_sop / max(num_samples, 1)),
            "output_mean_abs": float(mean_abs),
            "output_rms": float(rms),
            "output_max_abs": float(self.output_max_abs),
        }


class ActivityTracer:
    """Forward-hook based activity tracer for HIPSA photonic MVM layers.

    Parameters
    ----------
    adc_threshold_abs:
        Absolute analog threshold. Takes priority if > 0.
    adc_threshold_fs:
        Full-scale threshold ratio. If > 0 and adc_threshold_abs == 0, each
        hook call uses threshold = adc_threshold_fs * max(abs(output)) for that
        layer/timestep batch.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_names: Optional[Sequence[str]] = None,
        include_linear: bool = True,
        include_fc2: bool = False,
        adc_threshold_abs: float = 0.0,
        adc_threshold_fs: float = 0.02,
        trace_lif: bool = True,
    ):
        self.model = model
        self.target_layer_names = set(target_layer_names) if target_layer_names else None
        self.include_linear = bool(include_linear)
        self.include_fc2 = bool(include_fc2)
        self.adc_threshold_abs = float(adc_threshold_abs or 0.0)
        self.adc_threshold_fs = float(adc_threshold_fs or 0.0)
        self.trace_lif = bool(trace_lif)
        self.stats: Dict[str, LayerActivityStats] = {}
        self.handles: List[Any] = []
        self.input_active: int = 0
        self.input_total: int = 0
        self.num_samples: int = 0
        self.num_batches: int = 0
        self.layer_to_lif: Dict[str, str] = {}
        self.lif_to_layer: Dict[str, str] = {}

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

    def _is_lif_module(self, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        return "lif" in cls_name and "node" in cls_name

    def _infer_lif_mapping(self, target_names: Sequence[str]) -> None:
        """Map photonic MVM layers to following LIF nodes by model order.

        This matches the current models:
          CIFAR: conv1->lif1, ..., conv6->lif6, fc1->lif7
          DVS:   conv1->lif1, ..., conv5->lif5, fc1->lif6
        """
        lif_names = [name for name, module in self.model.named_modules() if self._is_lif_module(module)]
        self.layer_to_lif = {}
        self.lif_to_layer = {}
        for layer_name, lif_name in zip(target_names, lif_names):
            self.layer_to_lif[str(layer_name)] = str(lif_name)
            self.lif_to_layer[str(lif_name)] = str(layer_name)
        if len(lif_names) < len(target_names):
            logger.warning(
                "Only mapped %d LIF nodes for %d target MVM layers. Missing layers will have lif_spike_activity=0.",
                len(lif_names), len(target_names),
            )

    def register(self) -> None:
        target_names: List[str] = []
        for name, module in self.model.named_modules():
            if self._is_target(name, module):
                target_names.append(name)
                self.stats[name] = LayerActivityStats(name=name, module_type=module.__class__.__name__)
                self.handles.append(module.register_forward_hook(self._make_mvm_hook(name, module)))
                logger.debug("Registered MVM activity hook on %s", name)

        if self.trace_lif:
            self._infer_lif_mapping(target_names)
            for layer_name, lif_name in self.layer_to_lif.items():
                if layer_name in self.stats:
                    self.stats[layer_name].mapped_lif_name = lif_name
            for name, module in self.model.named_modules():
                if name in self.lif_to_layer and self._is_lif_module(module):
                    self.handles.append(module.register_forward_hook(self._make_lif_hook(name)))
                    logger.debug("Registered LIF spike hook on %s -> %s", name, self.lif_to_layer[name])

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self) -> None:
        for s in self.stats.values():
            s.mvm_input_active = s.mvm_input_total = 0
            s.mvm_output_nonzero_active = s.mvm_output_total = 0
            s.adc_request_active = s.adc_request_total = 0
            s.lif_spike_active = s.lif_spike_total = 0
            s.dense_sop = 0.0
            s.output_abs_sum = s.output_sq_sum = 0.0
            s.output_max_abs = 0.0
            s.calls = 0
            s.lif_calls = 0
            s.input_shape = s.output_shape = s.lif_output_shape = None
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

    def _make_mvm_hook(self, name: str, module: nn.Module):
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

            st.mvm_input_active += int(torch.count_nonzero(x).item())
            st.mvm_input_total += int(x.numel())
            st.mvm_output_nonzero_active += int(torch.count_nonzero(y).item())
            st.mvm_output_total += int(y.numel())

            y_detached = y.detach()
            abs_y = y_detached.abs()
            if self.adc_threshold_abs > 0:
                threshold = self.adc_threshold_abs
            elif self.adc_threshold_fs > 0 and abs_y.numel() > 0:
                threshold = float(abs_y.amax().clamp_min(1e-12).item()) * self.adc_threshold_fs
            else:
                threshold = 0.0
            if threshold > 0:
                adc_mask = abs_y > threshold
            else:
                adc_mask = y_detached != 0
            st.adc_request_active += int(adc_mask.count_nonzero().item())
            st.adc_request_total += int(y.numel())

            st.output_abs_sum += float(abs_y.sum().item())
            st.output_sq_sum += float((y_detached.float() ** 2).sum().item())
            if abs_y.numel() > 0:
                st.output_max_abs = max(st.output_max_abs, float(abs_y.max().item()))
            st.dense_sop += float(dense_sop_for_call(mod, x, y))
        return hook

    def _make_lif_hook(self, lif_name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            layer_name = self.lif_to_layer.get(lif_name)
            if layer_name is None or layer_name not in self.stats:
                return
            if not torch.is_tensor(output):
                return
            st = self.stats[layer_name]
            st.lif_calls += 1
            st.lif_output_shape = tuple(int(v) for v in output.shape)
            st.lif_spike_active += int(torch.count_nonzero(output).item())
            st.lif_spike_total += int(output.numel())
        return hook

    def summary(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        layers = {name: stat.to_dict(self.num_samples) for name, stat in self.stats.items()}
        dense_total = float(sum(v["dense_sop_total"] for v in layers.values()))
        active_total = float(sum(v["active_sop_total"] for v in layers.values()))

        adc_active = int(sum(v["adc_request_active"] for v in layers.values()))
        adc_total = int(sum(v["adc_request_total"] for v in layers.values()))
        lif_active = int(sum(v["lif_spike_active"] for v in layers.values()))
        lif_total = int(sum(v["lif_spike_total"] for v in layers.values()))
        mvm_in_active = int(sum(v["mvm_input_active"] for v in layers.values()))
        mvm_in_total = int(sum(v["mvm_input_total"] for v in layers.values()))
        mvm_out_active = int(sum(v["mvm_output_nonzero_active"] for v in layers.values()))
        mvm_out_total = int(sum(v["mvm_output_total"] for v in layers.values()))

        out: Dict[str, Any] = {
            "activity_schema_version": "v2_lif_adc",
            "num_samples": int(self.num_samples),
            "num_batches": int(self.num_batches),
            "input_active_count": int(self.input_active),
            "input_total_count": int(self.input_total),
            "input_spike_activity": float(self.input_active / self.input_total) if self.input_total else 0.0,
            "mvm_input_active": mvm_in_active,
            "mvm_input_total": mvm_in_total,
            "mvm_input_activity_mean": float(mvm_in_active / mvm_in_total) if mvm_in_total else 0.0,
            "mvm_output_nonzero_active": mvm_out_active,
            "mvm_output_total": mvm_out_total,
            "mvm_output_nonzero_activity_mean": float(mvm_out_active / mvm_out_total) if mvm_out_total else 0.0,
            "lif_spike_active": lif_active,
            "lif_spike_total": lif_total,
            "lif_spike_activity": float(lif_active / lif_total) if lif_total else 0.0,
            "adc_request_active": adc_active,
            "adc_request_total": adc_total,
            "adc_request_activity": float(adc_active / adc_total) if adc_total else 0.0,
            "adc_activity_proxy": float(adc_active / adc_total) if adc_total else 0.0,
            "dense_sop_total": dense_total,
            "active_sop_total": active_total,
            "dense_sop_per_image": dense_total / max(self.num_samples, 1),
            "active_sop_per_image": active_total / max(self.num_samples, 1),
            "active_sop_ratio": active_total / dense_total if dense_total else 0.0,
            "layer_to_lif": dict(self.layer_to_lif),
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
