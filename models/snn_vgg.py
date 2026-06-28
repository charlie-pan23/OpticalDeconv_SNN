"""Legacy Spiking VGG model for CIFAR10-DVS.

This file keeps the original high-capacity VGG-style model available for backward
compatibility with old checkpoints. For new paper experiments, prefer
``models.snn_vgg_gap.SpikingVGGGAP`` because the legacy fc1 layer has many more
parameters and can overfit CIFAR10-DVS.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .snn_common import build_lif, named_modules_by_names


class SpikingVGG(nn.Module):
    """Spiking VGG for event-frame inputs.

    Input:  ``[T, B, C, H, W]``
    Output: ``[T, B, num_classes]``
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 2,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_name: str = "sigmoid",
        surrogate_alpha: float = 4.0,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.return_time_logits = True
        self.photonic_mvm_layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "fc1"]

        def bn(ch: int) -> nn.Module:
            return nn.BatchNorm2d(ch) if use_batchnorm else nn.Identity()

        lif_kwargs = dict(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1, bias=False)
        self.bn1 = bn(32)
        self.lif1 = build_lif(**lif_kwargs)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = bn(64)
        self.lif2 = build_lif(**lif_kwargs)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = bn(128)
        self.lif3 = build_lif(**lif_kwargs)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = bn(128)
        self.lif4 = build_lif(**lif_kwargs)
        self.pool3 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = bn(256)
        self.lif5 = build_lif(**lif_kwargs)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn6 = bn(256)
        self.lif6 = build_lif(**lif_kwargs)
        self.pool4 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 8 * 8, 512, bias=False)
        self.lif7 = build_lif(**lif_kwargs)
        self.fc2 = nn.Linear(512, num_classes, bias=False)

    def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
        x_t = self.pool1(self.lif1(self.bn1(self.conv1(x_t))))
        x_t = self.pool2(self.lif2(self.bn2(self.conv2(x_t))))
        x_t = self.lif3(self.bn3(self.conv3(x_t)))
        x_t = self.pool3(self.lif4(self.bn4(self.conv4(x_t))))
        x_t = self.lif5(self.bn5(self.conv5(x_t)))
        x_t = self.pool4(self.lif6(self.bn6(self.conv6(x_t))))
        x_t = self.flatten(x_t)
        x_t = self.dropout(x_t)
        x_t = self.lif7(self.fc1(x_t))
        return self.fc2(x_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected input [T,B,C,H,W], got {tuple(x.shape)}")
        outputs = [self.forward_step(x[t]) for t in range(x.shape[0])]
        return torch.stack(outputs, dim=0)

    def named_photonic_modules(self) -> List[Tuple[str, nn.Module]]:
        return named_modules_by_names(self, self.photonic_mvm_layer_names)
