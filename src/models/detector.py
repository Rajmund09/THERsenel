"""
Object Detection Head Interface (src/models/detector.py)
Unified Object Detector head wrapper for anchor-free bounding box regression and object classification.
"""

import torch
import torch.nn as nn


class ObjectDetectionHead(nn.Module):
    """
    Decoupled Object Detection Head for predicting class probabilities and bounding box coordinates.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes

        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

        # Bounding box regression branch (x_center, y_center, width, height)
        self.box_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor of shape [batch_size, num_classes + 4, H, W]
        """
        cls_score = self.cls_conv(x)
        box_loc = self.box_conv(x)
        return torch.cat([cls_score, box_loc], dim=1)
