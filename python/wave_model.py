"""
Neural-network definitions for WaveAccel.

The compact WaveRegressor is trained from scratch on numerical fields
generated through the public cuWaves C++ API.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

import torch
from torch import nn


class WaveRegressor(nn.Module):
    """Compact MLP: 64 spatial wave samples -> [amplitude, wavelength]."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicalOutputModel(nn.Module):
    """Wrap a normalized-output model so exported ONNX returns physical units."""

    def __init__(
        self,
        model: WaveRegressor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.model(x)
        return normalized * self.target_std + self.target_mean
