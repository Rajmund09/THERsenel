"""
RGB Feature Encoder (src/models/rgb_encoder.py)
Extracts multi-scale spatial features from 3-channel RGB visual images using CNN backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBEncoder(nn.Module):
    """
    CNN backbone for extracting visual features from 3-channel RGB imagery.
    Outputs multi-scale feature maps at 1/8, 1/16, and 1/32 resolutions.
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
