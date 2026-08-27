"""
RGB + Thermal Dual-Stream Fusion Detector (src/models/fusion_model.py)
Fuses visual (RGB) and infrared (Thermal) feature maps using Spatial Attention Fusion (SAF)
and outputs bounding box predictions and class scores.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rgb_encoder import RGBEncoder
from src.models.thermal_encoder import ThermalEncoder


class SpatialAttentionFusion(nn.Module):
    """
    Learns per-pixel attention weighting to dynamically weight RGB vs Thermal modalities based on lighting/heat contrast.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv_att = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, f_rgb: torch.Tensor, f_thermal: torch.Tensor) -> torch.Tensor:
        cat_feat = torch.cat([f_rgb, f_thermal], dim=1)
        att_weights = self.conv_att(cat_feat)
        
        w_rgb = att_weights[:, 0:1, :, :]
        w_thermal = att_weights[:, 1:2, :, :]

        f_fused = w_rgb * f_rgb + w_thermal * f_thermal
        return self.fusion_conv(f_fused)


class RGBThermalFusionDetector(nn.Module):
    """
    Dual-stream architecture fusing RGB and Thermal inputs for border intrusion object detection.
    """

    def __init__(self, num_classes: int = 7, feature_dim: int = 256):
        super().__init__()
        self.rgb_encoder = RGBEncoder(in_channels=3, feature_dim=feature_dim)
        self.thermal_encoder = ThermalEncoder(in_channels=3, feature_dim=feature_dim)
        self.fusion_module = SpatialAttentionFusion(channels=feature_dim)

        # Detection Head (Bounding Box Regression + Class Classification)
        self.head_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(128, 4, kernel_size=1)

    def forward(self, x_rgb: torch.Tensor, x_thermal: torch.Tensor) -> dict:
        f_rgb = self.rgb_encoder(x_rgb)
        f_thermal = self.thermal_encoder(x_thermal)

        f_fused = self.fusion_module(f_rgb, f_thermal)
        h = self.head_conv(f_fused)

        cls_logits = self.cls_head(h)
        box_preds = self.box_head(h)

        return {
            "cls_logits": cls_logits,
            "box_preds": box_preds,
            "fused_features": f_fused
        }
