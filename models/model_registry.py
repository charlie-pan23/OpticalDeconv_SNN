"""Config-driven model registry for HIPSA SNN workloads."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Type

import torch.nn as nn

from .snn_common import count_parameters, neuron_kwargs_from_config, regularization_kwargs_from_config


_MODEL_ALIASES = {
    "snn_vgg": ("models.snn_vgg", "SpikingVGG"),
    "spiking_vgg": ("models.snn_vgg", "SpikingVGG"),
    "snn_vgg_gap": ("models.snn_vgg_gap", "SpikingVGGGAP"),
    "spiking_vgg_gap": ("models.snn_vgg_gap", "SpikingVGGGAP"),
    "snn_gesture_cnn": ("models.snn_gesture_cnn", "SpikingGestureCNN"),
    "spiking_gesture_cnn": ("models.snn_gesture_cnn", "SpikingGestureCNN"),
}


def get_model_class(model_cfg: Dict[str, Any]) -> Type[nn.Module]:
    """Resolve a model class from a config ``model`` section.

    Resolution order:
      1. Explicit ``module`` + ``class_name`` in YAML.
      2. Registry alias from ``type``.
      3. Optional fallback_module + fallback_class_name.
    """
    if not isinstance(model_cfg, dict):
        raise TypeError("model_cfg must be a dictionary")

    module_name = model_cfg.get("module")
    class_name = model_cfg.get("class_name")

    if not module_name or not class_name:
        alias = str(model_cfg.get("type", "")).lower()
        if alias in _MODEL_ALIASES:
            module_name, class_name = _MODEL_ALIASES[alias]

    if not module_name or not class_name:
        module_name = model_cfg.get("fallback_module")
        class_name = model_cfg.get("fallback_class_name")

    if not module_name or not class_name:
        raise ValueError(f"Cannot resolve model from config: {model_cfg}")

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as primary_error:
        fallback_module = model_cfg.get("fallback_module")
        fallback_class = model_cfg.get("fallback_class_name")
        if fallback_module and fallback_class:
            module = importlib.import_module(fallback_module)
            return getattr(module, fallback_class)
        raise primary_error


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build a model from full YAML config or directly from a model section."""
    model_cfg = config.get("model", config) if isinstance(config, dict) else {}
    cls = get_model_class(model_cfg)

    kwargs: Dict[str, Any] = {
        "num_classes": int(model_cfg.get("num_classes", 10)),
        "input_channels": int(model_cfg.get("input_channels", 2)),
        **neuron_kwargs_from_config(model_cfg),
        **regularization_kwargs_from_config(model_cfg),
    }

    # Optional architecture knobs. Only pass if present to keep legacy models compatible.
    for key in ["channels", "hidden_dim"]:
        if key in model_cfg:
            kwargs[key] = model_cfg[key]

    try:
        return cls(**kwargs)
    except TypeError:
        # Some older model constructors may not support every config key.
        fallback_kwargs = {
            "num_classes": kwargs["num_classes"],
            "input_channels": kwargs["input_channels"],
            "tau": kwargs["tau"],
            "v_threshold": kwargs["v_threshold"],
            "v_reset": kwargs["v_reset"],
        }
        return cls(**fallback_kwargs)


def describe_model(model: nn.Module) -> Dict[str, Any]:
    """Return a lightweight model summary for logs/checkpoints."""
    photonic_names = []
    if hasattr(model, "photonic_mvm_layer_names"):
        photonic_names = list(getattr(model, "photonic_mvm_layer_names"))
    return {
        "class_name": model.__class__.__name__,
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
        "photonic_mvm_layer_names": photonic_names,
    }
