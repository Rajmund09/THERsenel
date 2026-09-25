"""
Border Intrusion & ROI Polygon Tracker (src/intrusion/border_tracker.py)
Defines Region of Interest (ROI) border lines, tracks detected objects across frames,
and accurately detects unauthorized perimeter intrusions with persistent track IDs.
"""

from typing import List, Tuple, Dict, Any, Set
import numpy as np
import cv2


class BorderIntrusionTracker:
    """
    Monitors border security boundary (ROI polygon) and detects unauthorized line crossing
    with persistent multi-frame centroid tracking.
    """

    def __init__(
        self,
        roi_polygon: List[Tuple[int, int]] = None,
        restricted_classes: List[str] = None,
        max_distance: float = 110.0,
        max_lost_frames: int = 24
    ):
        self.roi_polygon = roi_polygon or [(100, 400), (540, 400), (600, 600), (40, 600)]
        self.restricted_classes = set(restricted_classes) if restricted_classes else {
            "person", "car", "motorcycle", "bus", "truck", "rider", "bicycle"
        }
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames

        # Active tracks: track_id -> {"center": (cx, cy), "base": (cx, y2), "lost": int, "history": [(cx, cy), ...], "breached": bool}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id: int = 1
        self.unique_breached_ids: Set[int] = set()

    def is_point_inside_roi(self, point: Tuple[float, float]) -> bool:
        """Ray-casting algorithm & OpenCV point polygon test to check if (x, y) point is inside ROI."""
        if not self.roi_polygon or len(self.roi_polygon) < 3:
            return False
        
        pts = np.array(self.roi_polygon, np.int32)
        # cv2.pointPolygonTest returns >= 0 if point is inside or on the contour
        res = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return res >= 0

    def check_bbox_breached(self, bbox: List[float]) -> bool:
        """
        Check if any grounded or central anchor of the bounding box breaches the ROI:
        1. Base ground point: (center_x, bottom_y)
        2. Centroid point: (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        base_point = (cx, y2)
        center_point = (cx, cy)

        return self.is_point_inside_roi(base_point) or self.is_point_inside_roi(center_point)

    def process_detections(self, detections: List[dict]) -> List[dict]:
        """
        Process detections with persistent centroid matching, assign stable track IDs,
        and return list of active intrusion alerts for this frame.
        Detection format: {"id": int, "class_name": str, "bbox": [x1, y1, x2, y2], "confidence": float}
        """
        alerts = []
        if not detections:
            # Age existing tracks
            to_remove = []
            for tid, trk in self.active_tracks.items():
                trk["lost"] += 1
                if trk["lost"] > self.max_lost_frames:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.active_tracks[tid]
            return alerts

        # Filter candidates by restricted classes
        candidate_items = []
        for det in detections:
            cls_name = det.get("class_name", "").lower()
            if self.restricted_classes and cls_name not in self.restricted_classes:
                continue
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            candidate_items.append((cx, cy, det))

        # Centroid distance matching across frames
        matched_track_ids = set()
        for cx, cy, det in candidate_items:
            best_id = None
            best_dist = self.max_distance

            for tid, trk in self.active_tracks.items():
                if tid in matched_track_ids:
                    continue
                last_cx, last_cy = trk["center"]
                dist = np.hypot(cx - last_cx, cy - last_cy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[best_id] = {
                    "center": (cx, cy),
                    "lost": 0,
                    "history": [(cx, cy)],
                    "breached": False,
                    "class_name": det.get("class_name", "intruder")
                }
            else:
                trk = self.active_tracks[best_id]
                trk["center"] = (cx, cy)
                trk["lost"] = 0
                trk["history"].append((cx, cy))
                if len(trk["history"]) > 30:
                    trk["history"].pop(0)

            matched_track_ids.add(best_id)
            det["track_id"] = best_id

            # Check if this object breaches the restricted ROI perimeter
            is_breached = self.check_bbox_breached(det["bbox"])
            det["is_breached"] = is_breached

            if is_breached:
                self.active_tracks[best_id]["breached"] = True
                self.unique_breached_ids.add(best_id)
                alerts.append({
                    "track_id": best_id,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "center_point": (cx, cy),
                    "status": "INTRUSION_ALERT",
                    "history": list(self.active_tracks[best_id]["history"])
                })

        # Age and remove inactive tracks
        to_remove = []
        for tid, trk in self.active_tracks.items():
            if tid not in matched_track_ids:
                trk["lost"] += 1
                if trk["lost"] > self.max_lost_frames:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.active_tracks[tid]

        return alerts

    @property
    def unique_breached_count(self) -> int:
        """Returns the total number of distinct intruders that breached the perimeter."""
        return len(self.unique_breached_ids)
