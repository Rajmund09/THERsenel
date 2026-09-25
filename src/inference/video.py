"""
Video Stream Inference Engine (src/inference/video.py)
Processes video streams (single RGB/Thermal or synchronized pairs) with YOLOv8 and BorderIntrusionTracker.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import cv2
import numpy as np
from ultralytics import YOLO
from src.intrusion.border_tracker import BorderIntrusionTracker


class VideoInferenceProcessor:
    """
    Processes single video streams (MP4/AVI/RTSP/Webcam) with YOLO detection and ROI tracking.
    """

    def __init__(
        self,
        video_source: str | int,
        output_path: Optional[str] = None,
        model: Optional[YOLO] = None,
        conf_threshold: float = 0.5,
        roi_polygon: Optional[List[tuple]] = None
    ):
        src = int(video_source) if str(video_source).isdigit() else str(video_source)
        self.cap = cv2.VideoCapture(src)
        self.output_path = output_path
        self.conf_threshold = conf_threshold

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Default or adaptive ROI polygon
        if roi_polygon:
            self.roi_polygon = roi_polygon
        else:
            self.roi_polygon = [
                (int(self.width * 0.08), int(self.height * 0.25)),
                (int(self.width * 0.92), int(self.height * 0.25)),
                (int(self.width * 0.92), int(self.height * 0.85)),
                (int(self.width * 0.08), int(self.height * 0.85))
            ]

        self.tracker = BorderIntrusionTracker(roi_polygon=self.roi_polygon)
        self.model = model or YOLO("yolov8n.pt")

    def process(self) -> Dict[str, Any]:
        """Process entire video file and export annotated output."""
        if not self.cap.isOpened():
            return {"error": "Could not open video source"}

        writer = None
        if self.output_path:
            out_p = Path(self.output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        total_alerts = 0
        t0 = time.perf_counter()

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                t_frame = time.perf_counter()

                # YOLO Inference
                results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
                detections = []
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    detections.append({
                        "id": i,
                        "class_name": cls_name,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf
                    })

                alerts = self.tracker.process_detections(detections)
                has_alert = len(alerts) > 0
                if has_alert:
                    total_alerts += len(alerts)

                # Annotate Frame
                annotated = self.annotate_frame(frame, detections, alerts, frame_count, t_frame)
                if writer:
                    writer.write(annotated)
        finally:
            self.cap.release()
            if writer:
                writer.release()

        elapsed = time.perf_counter() - t0
        return {
            "total_frames": frame_count,
            "total_alerts": total_alerts,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(frame_count / max(elapsed, 0.001), 1),
            "output_path": self.output_path
        }

    def annotate_frame(self, frame: np.ndarray, detections: list, alerts: list, frame_idx: int, t_frame: float) -> np.ndarray:
        """Render tactical ROI, bounding boxes, and HUD."""
        has_alert = len(alerts) > 0

        # ROI Polygon
        overlay = frame.copy()
        pts = np.array(self.roi_polygon, np.int32).reshape((-1, 1, 2))
        roi_color = (0, 0, 255) if has_alert else (0, 165, 255)
        cv2.fillPoly(overlay, [pts], roi_color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=roi_color, thickness=2)

        # Detections
        alert_ids = {a["track_id"]: a for a in alerts}
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            is_alert = det["id"] in alert_ids
            box_color = (0, 0, 255) if is_alert else (0, 230, 118)
            label = f"{det['class_name'].upper()} {det['confidence']:.2f}"
            if is_alert:
                label = f"[ALERT] {label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + len(label) * 10 + 10, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 4, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0) if not is_alert else (255, 255, 255), 1, cv2.LINE_AA)

        # HUD
        latency = (time.perf_counter() - t_frame) * 1000.0
        fps = 1000.0 / max(latency, 0.01)
        cv2.rectangle(frame, (10, 10), (450, 50), (15, 15, 18), -1)
        cv2.rectangle(frame, (10, 10), (450, 50), (255, 255, 255), 1)
        hud_text = f"FRAME {frame_idx:04d} | {fps:.1f} FPS | ALERTS: {len(alerts)}"
        status_color = (0, 0, 255) if has_alert else (0, 230, 118)
        cv2.putText(frame, hud_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)

        return frame


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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width * 2, height))

        frame_count = 0
        while self.cap_rgb.isOpened() and self.cap_thermal.isOpened():
            ret_rgb, frame_rgb = self.cap_rgb.read()
            ret_thermal, frame_thermal = self.cap_thermal.read()

            if not ret_rgb or not ret_thermal:
                break

            frame_count += 1
            if frame_thermal.shape != frame_rgb.shape:
                frame_thermal = cv2.resize(frame_thermal, (frame_rgb.shape[1], frame_rgb.shape[0]))

            combined = np.hstack([frame_rgb, frame_thermal])
            if writer:
                writer.write(combined)

        self.cap_rgb.release()
        self.cap_thermal.release()
        if writer:
            writer.release()

        print(f"[?] Processed {frame_count} frames successfully.")
