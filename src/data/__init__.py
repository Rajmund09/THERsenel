"""
Data management module for FLIR dataset, preprocessing, loading, registration, and augmentation.
"""

from src.data.dataset import RGBThermalDataset
from src.data.alignment import RGBIRAligner
from src.data.augmentation import MultimodalAugmenter

__all__ = ["RGBThermalDataset", "RGBIRAligner", "MultimodalAugmenter"]
