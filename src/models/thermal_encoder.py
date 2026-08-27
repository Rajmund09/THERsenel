"""
Thermal IR Feature Encoder (src/models/thermal_encoder.py)
Extracts infrared thermal signatures from thermal infrared imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalEncoder(nn.Module):
    """
    CNN backbone for extracting thermal infrared features from 1-channel or 3-channel thermal images.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        features = self.layer2(h)
        return features
