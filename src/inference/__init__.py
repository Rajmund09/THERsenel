"""
Inference Module for real-time image, video, and live camera stream object detection.
"""

from src.inference.predict import run_inference
from src.inference.video import DualVideoInferenceEngine
from src.inference.camera import CameraStreamReader

__all__ = ["run_inference", "DualVideoInferenceEngine", "CameraStreamReader"]
