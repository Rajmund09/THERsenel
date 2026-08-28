"""
Training scripts and loss functions module for RGB baseline, Thermal baseline, and Fusion model.
"""

from src.training.losses import MultimodalDetectionLoss, bbox_ciou

__all__ = ["MultimodalDetectionLoss", "bbox_ciou"]
