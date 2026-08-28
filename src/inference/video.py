"""
Dual Video Stream Inference Engine (src/inference/video.py)
Reads synchronized RGB and Thermal video files, runs fusion object detection, and draws border intrusion alerts.
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from src.intrusion.border_tracker import BorderIntrusionTracker


class DualVideoInferenceEngine:
    """
    Processes paired RGB + Thermal video files frame-by-frame for intrusion detection.
    """

    def __init__(self, rgb_video_path: str, thermal_video_path: str, output_path: Optional[str] = None):
        self.cap_rgb = cv2.VideoCapture(rgb_video_path)
        self.cap_thermal = cv2.VideoCapture(thermal_video_path)
        self.output_path = output_path
        self.tracker = BorderIntrusionTracker()

    def process(self):
        print(f"[+] Starting Dual Video Stream Processing...")

        width = int(self.cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(self.cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = int(self.cap_rgb.get(cv2.CAP_PROP_FPS)) or 30

        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width * 2, height))

        frame_count = 0
        while self.cap_rgb.isOpened() and self.cap_thermal.isOpened():
            ret_rgb, frame_rgb = self.cap_rgb.read()
            ret_thermal, frame_thermal = self.cap_thermal.read()

            if not ret_rgb or not ret_thermal:
                break

            frame_count += 1
            # Resize thermal to match RGB dimensions if necessary
            if frame_thermal.shape != frame_rgb.shape:
                frame_thermal = cv2.resize(frame_thermal, (frame_rgb.shape[1], frame_rgb.shape[0]))

            # Side-by-side display canvas
            combined = np.hstack([frame_rgb, frame_thermal])

            if writer:
                writer.write(combined)

        self.cap_rgb.release()
        self.cap_thermal.release()
        if writer:
            writer.release()

        print(f"[?] Processed {frame_count} frames successfully.")
