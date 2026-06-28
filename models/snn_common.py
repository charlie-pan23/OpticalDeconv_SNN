"""Common utilities for HIPSA SpikingJelly SNN models."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


def build_surrogate(name: str = "sigmoid", alpha: float = 4.0):
    """Create a SpikingJelly surrogate function from a config-friendly name."""
    name = (name or "sigmoid").lower()
    if name in {"sigmoid", "sigmoid_surrogate"}:
        return surrogate.Sigmoid(alpha=alpha)
    if name in {"atan", "arctan"}:
        return surrogate.ATan(alpha=alpha)
    if name in {"fast_sigmoid", "fastsigmoid"}:
        return surrogate.Sigmoid(alpha=alpha)
    raise ValueError(f"Unsupported surrogate function: {name}")


def build_lif(
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: float | None = 0.0,
    surrogate_name: str = "sigmoid",
    surrogate_alpha: float = 4.0,
) -> neuron.LIFNode:
    """Create a LIFNode with parameters used by both CIFAR10-DVS and DVS Gesture."""
    sg = build_surrogate(surrogate_name, surrogate_alpha)
    return neuron.LIFNode(
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        surrogate_function=sg,
    )


def aggregate_time_logits(logits_t: torch.Tensor, mode: str = "mean_time") -> torch.Tensor:
    """Aggregate model output from [T, B, C] to [B, C].

    This helper is intentionally placed in ``models`` so train/eval scripts use the
    same convention. The model forward itself still returns [T, B, C] for activity
    and hardware-timestep analysis.
    """
    if logits_t.dim() != 3:
        raise ValueError(f"Expected logits with shape [T,B,C], got {tuple(logits_t.shape)}")
    mode = (mode or "mean_time").lower()
    if mode == "mean_time":
        return logits_t.mean(dim=0)
    if mode == "sum_time":
        return logits_t.sum(dim=0)
    if mode == "last_time":
        return logits_t[-1]
    raise ValueError(f"Unsupported logits aggregation mode: {mode}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def named_modules_by_names(model: nn.Module, names: Iterable[str]) -> List[Tuple[str, nn.Module]]:
    module_dict = dict(model.named_modules())
    return [(name, module_dict[name]) for name in names if name in module_dict]


def neuron_kwargs_from_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ncfg = model_cfg.get("neuron", {}) if isinstance(model_cfg, dict) else {}
    return {
        "tau": float(ncfg.get("tau", model_cfg.get("tau", 2.0) if isinstance(model_cfg, dict) else 2.0)),
        "v_threshold": float(ncfg.get("v_threshold", model_cfg.get("v_threshold", 1.0) if isinstance(model_cfg, dict) else 1.0)),
        "v_reset": ncfg.get("v_reset", model_cfg.get("v_reset", 0.0) if isinstance(model_cfg, dict) else 0.0),
        "surrogate_name": str(ncfg.get("surrogate", "sigmoid")),
        "surrogate_alpha": float(ncfg.get("surrogate_alpha", 4.0)),
    }


def regularization_kwargs_from_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rcfg = model_cfg.get("regularization", {}) if isinstance(model_cfg, dict) else {}
    return {
        "dropout": float(rcfg.get("dropout", model_cfg.get("dropout", 0.2) if isinstance(model_cfg, dict) else 0.2)),
        "use_batchnorm": bool(rcfg.get("use_batchnorm", True)),
    }
