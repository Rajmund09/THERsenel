"""
Region of Interest (ROI) Border Boundary Definition (src/intrusion/roi.py)
Defines interactive polygon border regions and tests spatial bounding box containment.
"""

from typing import List, Tuple
import cv2
import numpy as np


class ROIManager:
    """
    Manages polygon ROI coordinates, scales polygon coordinates to match image resolutions, and draws border overlays.
    """

    def __init__(self, polygon_coords: List[Tuple[float, float]] = None):
        # Normalized coordinates [(x1,y1), (x2,y2)...]
        self.polygon_coords = polygon_coords or [
            (0.1, 0.4),
            (0.9, 0.4),
            (0.9, 0.9),
            (0.1, 0.9)
        ]

    def get_pixel_polygon(self, width: int, height: int) -> np.ndarray:
        pts = [(int(x * width), int(y * height)) for x, y in self.polygon_coords]
        return np.array(pts, dtype=np.int32)

    def draw_roi(self, image: np.ndarray, is_alert: bool = False) -> np.ndarray:
        height, width = image.shape[:2]
        pts = self.get_pixel_polygon(width, height)

        color = (0, 0, 255) if is_alert else (0, 255, 0)
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)

        # Apply semi-transparent fill
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        return image
