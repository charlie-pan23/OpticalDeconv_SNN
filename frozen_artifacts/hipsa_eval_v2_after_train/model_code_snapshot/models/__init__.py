"""HIPSA SNN model package.

The training scripts should build models through ``models.model_registry.build_model``
so that CIFAR10-DVS and DVS Gesture can share one config-driven interface.
"""

from .snn_vgg import SpikingVGG
from .snn_vgg_gap import SpikingVGGGAP
from .snn_gesture_cnn import SpikingGestureCNN
from .model_registry import build_model, get_model_class, count_parameters

__all__ = [
    "SpikingVGG",
    "SpikingVGGGAP",
    "SpikingGestureCNN",
    "build_model",
    "get_model_class",
    "count_parameters",
]
