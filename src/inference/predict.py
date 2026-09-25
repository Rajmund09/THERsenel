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


def process_video(
    video_source: str,
    output_path: str = None,
    conf_threshold: float = 0.5,
    model_path: str = None,
    display: bool = False
):
    print("==========================================================================")
    print("           REAL-TIME VIDEO STREAM INTRUSION DETECTION ENGINE              ")
    print("==========================================================================")

    # Resolve Video Capture Source (File Path or Webcam Index)
    src = int(video_source) if video_source.isdigit() else str(video_source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source '{video_source}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"[+] Video Source: {video_source}")
    print(f"[+] Resolution: {width}x{height} | Base FPS: {fps:.1f} | Total Frames: {total_frames if total_frames > 0 else 'Live Stream'}")

    # Initialize Tracker with Adaptive ROI Polygon scaled to video resolution
    roi_polygon = [
        (int(width * 0.08), int(height * 0.25)),
        (int(width * 0.92), int(height * 0.25)),
        (int(width * 0.92), int(height * 0.85)),
        (int(width * 0.08), int(height * 0.85))
    ]
    tracker = BorderIntrusionTracker(roi_polygon=roi_polygon)
    print(f"[+] Initialized ROI Polygon Boundary: {roi_polygon}")

    # Load Model
    if not model_path:
        default_pt = PROJECT_ROOT / "runs" / "rgb_baseline" / "rgb_yolov8" / "weights" / "best.pt"
        model_path = str(default_pt) if default_pt.exists() else "yolov8n.pt"

    print(f"[+] Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    # Initialize Video Writer if output path requested
    writer = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
        print(f"[+] Output will be saved to: {output_path}")

    frame_idx = 0
    total_alerts_count = 0
    start_time = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_frame_start = time.perf_counter()

            # YOLOv8 Inference
            results = model(frame, conf=conf_threshold, verbose=False)[0]

            detections = []
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "id": i,
                    "class_name": cls_name,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf
                })

            # Intrusion Tracker
            alerts = tracker.process_detections(detections)
            has_alert = len(alerts) > 0
            if has_alert:
                total_alerts_count += len(alerts)

            frame_latency_ms = (time.perf_counter() - t_frame_start) * 1000.0
            curr_fps = 1000.0 / max(frame_latency_ms, 0.01)

            # --- ANNOTATIONS ---
            # 1. Semi-transparent Restricted ROI
            overlay = frame.copy()
            pts = np.array(roi_polygon, np.int32).reshape((-1, 1, 2))
            roi_color = (0, 0, 255) if has_alert else (0, 165, 255)
            cv2.fillPoly(overlay, [pts], roi_color)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=roi_color, thickness=2)

            # 2. Detections & Alert Boxes
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

            # 3. Tactical Heads-Up Display (HUD)
            hud_bg = (15, 15, 18)
            cv2.rectangle(frame, (10, 10), (450, 50), hud_bg, -1)
            cv2.rectangle(frame, (10, 10), (450, 50), (255, 255, 255), 1)

            status_color = (0, 0, 255) if has_alert else (0, 230, 118)
            hud_text = f"FRAME {frame_idx:04d} | {curr_fps:.1f} FPS | ALERTS: {len(alerts)}"
            cv2.putText(frame, hud_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)

            if display:
                cv2.imshow("THERsenel - Video Stream Intrusion Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[!] Early termination requested by user.")
                    break

            if total_frames > 0 and frame_idx % 25 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"  --> Processed {frame_idx}/{total_frames} frames ({pct:.1f}%) | Current FPS: {curr_fps:.1f}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_time
    avg_fps = frame_idx / max(total_time, 0.001)

    print("\n" + "=" * 74)
    print(" 🎬 VIDEO PROCESSING COMPLETED ")
    print("=" * 74)
    print(f"  • Total Frames Processed: {frame_idx}")
    print(f"  • Total Intrusion Events: {total_alerts_count}")
    print(f"  • Total Processing Time: {total_time:.2f} s")
    print(f"  • Average Processing FPS: {avg_fps:.1f}")
    if output_path:
        print(f"  • Annotated Output Video: 👉 file:///{str(Path(output_path).resolve()).replace(chr(92), '/')}")
    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Intrusion Predictor (Image & Video)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file or webcam index (0, 1)")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video/image")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold [0.01 - 1.00]")
    parser.add_argument("--model", type=str, default=None, help="Custom YOLO model weights path")
    parser.add_argument("--display", action="store_true", help="Display real-time preview window")
    args = parser.parse_args()

    if args.video:
        process_video(
            video_source=args.video,
            output_path=args.output,
            conf_threshold=args.conf,
            model_path=args.model,
            display=args.display
        )
    else:
        run_inference(image_path=args.image)


if __name__ == "__main__":
    main()

