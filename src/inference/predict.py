"""
Real-Time Inference Engine (src/inference/predict.py)
Performs real-time YOLOv8 object detection and intrusion monitoring.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO
from src.intrusion.border_tracker import BorderIntrusionTracker


def run_inference(image_path: str = None):
    print("==========================================================================")
    print("           REAL-TIME RGB + THERMAL INTRUSION DETECTION ENGINE             ")
    print("==========================================================================")

    # Initialize Tracker with a custom ROI polygon (adjust points based on image)
    # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    roi_polygon = [(50, 150), (600, 150), (600, 450), (50, 450)]
    tracker = BorderIntrusionTracker(roi_polygon=roi_polygon)
    print("[+] Initialized Border Intrusion ROI Tracker")

    # Load trained RGB baseline model (or default YOLOv8n if not found)
    model_path = PROJECT_ROOT / "runs" / "rgb_baseline" / "rgb_yolov8" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"[!] Custom weights not found at {model_path}. Using yolov8n.pt instead.")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(str(model_path))
    
    # Load Image
    if not image_path:
        # Generate a dummy test image with a person if none provided
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # Draw a fake "person" in the middle of the ROI
        cv2.rectangle(frame, (300, 200), (380, 400), (200, 200, 200), -1)
        print("[!] No input image provided. Using simulated test frame.")
    else:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[!] Could not read image at {image_path}")
            return

    # Run YOLOv8 Inference
    t0 = time.perf_counter()
    results = model(frame, verbose=False)[0]
    
    detections = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        
        detections.append({
            "id": i,
            "class_name": cls_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })

    # Process Intrusions
    alerts = tracker.process_detections(detections)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[+] Inference Latency: {latency_ms:.2f} ms | FPS: {1000.0 / max(latency_ms, 0.01):.1f}")
    print(f"[+] Total Intrusion Alerts Triggered: {len(alerts)}")

    # Visualization
    # 1. Draw ROI Polygon (Semi-transparent overlay)
    overlay = frame.copy()
    pts = np.array(roi_polygon, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], (0, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "RESTRICTED BORDER ROI", (roi_polygon[0][0], roi_polygon[0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 2. Draw Detections & Alerts
    alert_ids = {a["track_id"]: a for a in alerts}
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        is_alert = det["id"] in alert_ids
        
        color = (0, 0, 255) if is_alert else (0, 255, 0)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if is_alert:
            label = f"[ALERT] {label}"
            print(f"    🚨 [ALERT] Track #{det['id']} ({det['class_name'].upper()}) breached border ROI! Conf: {det['confidence']:.2f}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print("\n[?] Real-Time Application Pipeline Functional.")
    
    # Save Result
    output_path = str(PROJECT_ROOT / "data" / "samples" / "intrusion_output.jpg")
    cv2.imwrite(output_path, frame)
    
    print("\n" + "="*74)
    print(" 📸 SUCCESS! VISUALIZATION SAVED ")
    print("="*74)
    print(f"I have saved the output image with the bounding boxes and ROI overlay here:")
    print(f"👉 file:///{output_path.replace(chr(92), '/')}")
    print("Click the link above in VS Code to see how it is detecting!")
    print("==========================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Intrusion Predictor")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    args = parser.parse_args()
    
    run_inference(image_path=args.image)


if __name__ == "__main__":
    main()

