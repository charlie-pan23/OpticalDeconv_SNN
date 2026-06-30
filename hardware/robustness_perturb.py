"""Hardware non-ideality injection for HIPSA robustness evaluation."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


def photonic_target_names(model: nn.Module, include_fc2: bool = False) -> List[str]:
    if hasattr(model, "photonic_mvm_layer_names"):
        names = [str(x) for x in getattr(model, "photonic_mvm_layer_names")]
    else:
        names = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    if not include_fc2:
        names = [n for n in names if n != "fc2"]
    return names


class PerturbationContext(AbstractContextManager):
    """Apply temporary hardware perturbations through weights and hooks.

    Supported keys in perturbation dict:
      mrr_pct: static multiplicative weight-bank perturbation percentage.
      laser_pct: multiplicative input/carrier fluctuation percentage.
      wdm_xtalk_db: adjacent-channel crosstalk in dB, e.g. -20.
      tia_noise_pct: Gaussian output noise as percent of output dynamic scale.
      adc_bits: post-TIA quantization bits.
      comparator_threshold_fs: zero small analog outputs below threshold * max(abs(x)).
    """

    def __init__(self, model: nn.Module, perturbation: Optional[Mapping[str, Any]] = None, target_names: Optional[Sequence[str]] = None, seed: int = 42, include_fc2: bool = False):
        self.model = model
        self.perturbation = dict(perturbation or {})
        self.seed = int(seed)
        self.target_names = set(target_names or photonic_target_names(model, include_fc2=include_fc2))
        self.handles: List[Any] = []
        self.weight_backups: Dict[str, torch.Tensor] = {}
        self.generator: Optional[torch.Generator] = None

    def __enter__(self):
        device = next(self.model.parameters()).device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)
        self._apply_mrr_weight_perturbation()
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._restore_weights()
        return False

    def _target_modules(self):
        for name, module in self.model.named_modules():
            if name in self.target_names and isinstance(module, (nn.Conv2d, nn.Linear)):
                yield name, module

    def _apply_mrr_weight_perturbation(self) -> None:
        pct = float(self.perturbation.get("mrr_pct", 0.0) or 0.0)
        if pct == 0.0:
            return
        sigma = pct / 100.0
        for name, module in self._target_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            self.weight_backups[name] = module.weight.detach().clone()
            noise = torch.randn(module.weight.shape, device=module.weight.device, dtype=module.weight.dtype, generator=self.generator) * sigma
            module.weight.data.mul_(1.0 + noise)

    def _restore_weights(self) -> None:
        modules = dict(self.model.named_modules())
        for name, weight in self.weight_backups.items():
            mod = modules.get(name)
            if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                mod.weight.data.copy_(weight.to(mod.weight.device))
        self.weight_backups = {}

    def _register_hooks(self) -> None:
        for name, module in self._target_modules():
            if float(self.perturbation.get("laser_pct", 0.0) or 0.0) != 0.0:
                self.handles.append(module.register_forward_pre_hook(self._laser_pre_hook()))
            if any(k in self.perturbation for k in ["wdm_xtalk_db", "tia_noise_pct", "adc_bits", "comparator_threshold_fs"]):
                self.handles.append(module.register_forward_hook(self._output_hook()))

    def _laser_pre_hook(self):
        pct = float(self.perturbation.get("laser_pct", 0.0) or 0.0) / 100.0
        def hook(module: nn.Module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return inputs
            x = inputs[0]
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self.generator) * pct
            return (x * (1.0 + noise), *inputs[1:])
        return hook

    def _output_hook(self):
        def hook(module: nn.Module, inputs, output):
            if not torch.is_tensor(output):
                return output
            y = output
            if "wdm_xtalk_db" in self.perturbation:
                db = float(self.perturbation.get("wdm_xtalk_db"))
                alpha = 10.0 ** (db / 20.0)
                if y.dim() >= 2 and y.shape[1] > 1:
                    y = y + alpha * 0.5 * (torch.roll(y, shifts=1, dims=1) + torch.roll(y, shifts=-1, dims=1))
            if "tia_noise_pct" in self.perturbation:
                pct = float(self.perturbation.get("tia_noise_pct") or 0.0) / 100.0
                scale = y.detach().abs().amax().clamp_min(1e-6)
                y = y + torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=self.generator) * (pct * scale)
            if "comparator_threshold_fs" in self.perturbation:
                thr = float(self.perturbation.get("comparator_threshold_fs") or 0.0)
                if thr > 0:
                    scale = y.detach().abs().amax().clamp_min(1e-6)
                    y = torch.where(y.detach().abs() >= thr * scale, y, torch.zeros_like(y))
            if "adc_bits" in self.perturbation:
                bits = int(self.perturbation.get("adc_bits") or 0)
                if bits > 0:
                    levels = max(2 ** bits - 1, 1)
                    scale = y.detach().abs().amax().clamp_min(1e-6)
                    y_norm = torch.clamp(y / scale, -1.0, 1.0)
                    y_q = torch.round((y_norm + 1.0) * 0.5 * levels) / levels
                    y = (y_q * 2.0 - 1.0) * scale
            return y
        return hook


def default_robustness_sweeps() -> Dict[str, list[Dict[str, Any]]]:
    return {
        "clean": [{"name": "clean", "perturbation": {}}],
        "mrr": [{"name": f"mrr_{v}pct", "perturbation": {"mrr_pct": v}} for v in [1.0, 2.0, 3.0, 5.0]],
        "laser": [{"name": f"laser_{v}pct", "perturbation": {"laser_pct": v}} for v in [1.0, 2.0, 3.0, 5.0]],
        "wdm": [{"name": f"wdm_{abs(v)}dB", "perturbation": {"wdm_xtalk_db": v}} for v in [-30, -25, -20, -15]],
        "adc_bits": [{"name": f"adc_{v}bit", "perturbation": {"adc_bits": v}} for v in [8, 6, 5, 4]],
        "threshold": [{"name": f"thr_{v}", "perturbation": {"comparator_threshold_fs": v}} for v in [0.01, 0.02, 0.05, 0.10]],
        "tia_noise": [{"name": f"tia_{v}pct", "perturbation": {"tia_noise_pct": v}} for v in [0.5, 1.0, 2.0, 3.0]],
    }
