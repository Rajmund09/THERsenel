"""
Border Intrusion & ROI Polygon Tracker (src/intrusion/border_tracker.py)
Defines Region of Interest (ROI) border lines, tracks detected objects, and triggers intrusion breach alerts.
"""

from typing import List, Tuple
import numpy as np


class BorderIntrusionTracker:
    """
    Monitors border security boundary (ROI polygon) and detects unauthorized line crossing.
    """

    def __init__(
        self,
        roi_polygon: List[Tuple[int, int]] = None,
        restricted_classes: List[str] = None
    ):
        self.roi_polygon = roi_polygon or [(100, 400), (540, 400), (600, 600), (40, 600)]
        self.restricted_classes = restricted_classes or ["person", "car", "motorcycle", "bus", "truck"]
        self.active_tracks = {}

    def is_point_inside_roi(self, point: Tuple[float, float]) -> bool:
        """Ray-casting algorithm to test if (x, y) point lies inside ROI polygon."""
        x, y = point
        n = len(self.roi_polygon)
        inside = False

        p1x, p1y = self.roi_polygon[0]
        for i in range(n + 1):
            p2x, p2y = self.roi_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def process_detections(self, detections: List[dict]) -> List[dict]:
        """
        Process detections and return list of active intrusion alerts.
        Detection format: {"id": int, "class_name": str, "bbox": [x1, y1, x2, y2], "confidence": float}
        """
        alerts = []
        for det in detections:
            cls_name = det["class_name"]
            if cls_name not in self.restricted_classes:
                continue

            x1, y1, x2, y2 = det["bbox"]
            center_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            is_breached = self.is_point_inside_roi(center_point)
            if is_breached:
                alerts.append({
                    "track_id": det.get("id", -1),
                    "class_name": cls_name,
                    "confidence": det["confidence"],
                    "center_point": center_point,
                    "status": "INTRUSION_ALERT"
                })

        return alerts
