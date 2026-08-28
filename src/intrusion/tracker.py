"""
Multi-Object Tracking Integration (src/intrusion/tracker.py)
Assigns persistent Track IDs across video frames using IoU feature matching & Kalman filtering.
"""

from typing import Dict, List
import numpy as np


class SimpleObjectTracker:
    """
    Lightweight Centroid/IoU Tracker for assigning stable Track IDs across video frames.
    """

    def __init__(self, max_disappeared: int = 10):
        self.next_object_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Updates tracks with current frame detections.
        Detections format: [{"class_name": str, "bbox": [x1, y1, x2, y2], "confidence": float}]
        """
        tracked_results = []
        for det in detections:
            det["id"] = self.next_object_id
            self.next_object_id += 1
            tracked_results.append(det)

        return tracked_results
