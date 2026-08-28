"""
YOLOv8 Dual-Stream Spatial Attention Fusion Module (src/models/fusion_yolov8.py)
Integrates dual-stream feature fusion directly with Ultralytics YOLOv8 architecture backbones.
"""

from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO
from src.models.fusion_model import SpatialAttentionFusion


class FusionYOLOv8(nn.Module):
    """
    Combines YOLOv8 pretrained backbones with Spatial Attention Fusion for RGB + Thermal detection.
    """

    def __init__(self, model_name: str = "yolov8n.pt", num_classes: int = 7):
        super().__init__()
        # Load baseline YOLOv8 model architecture
        yolo_base = YOLO(model_name)
        self.backbone = yolo_base.model

        # Spatial fusion module for neck features
        self.fusion = SpatialAttentionFusion(channels=256)

    def forward(self, x_rgb: torch.Tensor, x_thermal: torch.Tensor):
        # Extract features from RGB and Thermal through backbone
        feat_rgb = self.backbone(x_rgb)
        feat_thermal = self.backbone(x_thermal)

        if isinstance(feat_rgb, list) and isinstance(feat_thermal, list):
            fused_feats = [self.fusion(r, t) for r, t in zip(feat_rgb, feat_thermal)]
            return fused_feats
        elif isinstance(feat_rgb, torch.Tensor) and isinstance(feat_thermal, torch.Tensor):
            return self.fusion(feat_rgb, feat_thermal)

        return feat_rgb
