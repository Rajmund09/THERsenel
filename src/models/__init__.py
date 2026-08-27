"""
Models Module (src/models/)
Encapsulates RGB encoder, thermal encoder, and dual-stream fusion architecture for YOLOv8 object detection.
"""

from src.models.rgb_encoder import RGBEncoder
from src.models.thermal_encoder import ThermalEncoder
from src.models.fusion_model import RGBThermalFusionDetector

__all__ = ["RGBEncoder", "ThermalEncoder", "RGBThermalFusionDetector"]
