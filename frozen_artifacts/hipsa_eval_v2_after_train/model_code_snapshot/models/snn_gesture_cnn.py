"""Compact SNN CNN for IBM DVS Gesture / DVS128Gesture.

DVS Gesture is less class-dense than CIFAR10-DVS but has strong motion patterns.
This model keeps the photonic MVM layers explicit while using GAP to avoid a large
fully connected classifier. It returns timestep logits for TET loss and hardware
activity tracing.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .snn_common import build_lif, named_modules_by_names


class SpikingGestureCNN(nn.Module):
    """SNN CNN for DVS Gesture.

    Input:  ``[T, B, C, H, W]``
    Output: ``[T, B, num_classes]``
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 2,
        channels: Sequence[int] = (32, 64, 128, 128, 256),
        hidden_dim: int = 256,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_name: str = "sigmoid",
        surrogate_alpha: float = 4.0,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("SpikingGestureCNN expects exactly 5 convolution channel values")
        c1, c2, c3, c4, c5 = [int(c) for c in channels]
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.photonic_mvm_layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1"]

        def bn(ch: int) -> nn.Module:
            return nn.BatchNorm2d(ch) if use_batchnorm else nn.Identity()

        lif_kwargs = dict(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        self.conv1 = nn.Conv2d(input_channels, c1, 3, padding=1, bias=False)
        self.bn1 = bn(c1)
        self.lif1 = build_lif(**lif_kwargs)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn2 = bn(c2)
        self.lif2 = build_lif(**lif_kwargs)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False)
        self.bn3 = bn(c3)
        self.lif3 = build_lif(**lif_kwargs)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1, bias=False)
        self.bn4 = bn(c4)
        self.lif4 = build_lif(**lif_kwargs)

        self.conv5 = nn.Conv2d(c4, c5, 3, padding=1, bias=False)
        self.bn5 = bn(c5)
        self.lif5 = build_lif(**lif_kwargs)
        self.pool4 = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c5, hidden_dim, bias=False)
        self.lif6 = build_lif(**lif_kwargs)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward_features_step(self, x_t: torch.Tensor) -> torch.Tensor:
        x_t = self.pool1(self.lif1(self.bn1(self.conv1(x_t))))
        x_t = self.pool2(self.lif2(self.bn2(self.conv2(x_t))))
        x_t = self.pool3(self.lif3(self.bn3(self.conv3(x_t))))
        x_t = self.lif4(self.bn4(self.conv4(x_t)))
        x_t = self.pool4(self.lif5(self.bn5(self.conv5(x_t))))
        x_t = self.gap(x_t)
        x_t = self.flatten(x_t)
        return x_t

    def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
        x_t = self.forward_features_step(x_t)
        x_t = self.dropout(x_t)
        x_t = self.lif6(self.fc1(x_t))
        return self.fc2(x_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected input [T,B,C,H,W], got {tuple(x.shape)}")
        outputs = [self.forward_step(x[t]) for t in range(x.shape[0])]
        return torch.stack(outputs, dim=0)

    def named_photonic_modules(self) -> List[Tuple[str, nn.Module]]:
        return named_modules_by_names(self, self.photonic_mvm_layer_names)
