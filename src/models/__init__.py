"""
Neural Network Models & Architectures Module
Includes RGB Encoder, Thermal Encoder, Spatial Attention Fusion, and Decoupled Detection Heads.
"""

from src.models.rgb_encoder import RGBEncoder
from src.models.thermal_encoder import ThermalEncoder
from src.models.fusion_model import SpatialAttentionFusion, RGBThermalFusionDetector
from src.models.detector import ObjectDetectionHead
from src.models.fusion_yolov8 import FusionYOLOv8

__all__ = [
    "RGBEncoder",
    "ThermalEncoder",
    "SpatialAttentionFusion",
    "RGBThermalFusionDetector",
    "ObjectDetectionHead",
    "FusionYOLOv8",
]
