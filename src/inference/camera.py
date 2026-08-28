"""
Live Camera & Jetson CSI Stream Capture Engine (src/inference/camera.py)
Interfaces with RTSP camera feeds, OpenCV webcams, and NVIDIA Jetson GStreamer CSI camera pipelines.
"""

from typing import Tuple
import cv2
import numpy as np


class CameraStreamReader:
    """
    Captures real-time frames from dual USB webcams, RTSP IP cameras, or Jetson CSI camera sensors.
    """

    def __init__(self, rgb_src: int | str = 0, thermal_src: int | str = 1):
        self.cap_rgb = cv2.VideoCapture(rgb_src)
        self.cap_thermal = cv2.VideoCapture(thermal_src)

    def is_opened(self) -> bool:
        return self.cap_rgb.isOpened() and self.cap_thermal.isOpened()

    def read_frame_pair(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        ret_rgb, frame_rgb = self.cap_rgb.read()
        ret_thermal, frame_thermal = self.cap_thermal.read()

        if not ret_rgb:
            frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        if not ret_thermal:
            frame_thermal = np.zeros((480, 640, 3), dtype=np.uint8)

        return ret_rgb or ret_thermal, frame_rgb, frame_thermal

    def release(self):
        self.cap_rgb.release()
        self.cap_thermal.release()
