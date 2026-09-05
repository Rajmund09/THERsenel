"""
Web UI Frontend (src/inference/app.py)
Streamlit dashboard for the Thermal Border Intrusion Detection system.
"""

import sys
from pathlib import Path
import time
import numpy as np
import cv2
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.intrusion.border_tracker import BorderIntrusionTracker

# Streamlit Page Config
st.set_page_config(
    page_title="Intrusion Detection UI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #E0E0E0; font-family: 'Inter', sans-serif; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .css-1d391kg { padding-top: 1rem; }
    .alert-box { padding: 10px; border-radius: 8px; margin-top: 10px; font-weight: bold; }
    .alert-danger { background-color: #ff4b4b20; color: #ff4b4b; border: 1px solid #ff4b4b; }
    .alert-success { background-color: #00cc4420; color: #00cc44; border: 1px solid #00cc44; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "runs" / "rgb_baseline" / "rgb_yolov8" / "weights" / "best.pt"
    if not model_path.exists():
        st.warning(f"Custom weights not found at {model_path}. Using yolov8n.pt instead.")
        return YOLO("yolov8n.pt")
    return YOLO(str(model_path))

def main():
    st.title("🛡️ Thermal-Border-Intrusion Dashboard")
    st.markdown("Upload a camera feed frame (RGB or Thermal) to test the Dual-Stream Intrusion Detection Engine.")
    
    # Sidebar config
    st.sidebar.title("⚙️ Engine Parameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    # ROI Configuration (Hardcoded for demo, but can be dynamic)
    roi_polygon = [(50, 150), (600, 150), (600, 450), (50, 450)]
    tracker = BorderIntrusionTracker(roi_polygon=roi_polygon)
    
    model = load_model()
    
    uploaded_file = st.sidebar.file_uploader("Upload Image Frame", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("👈 Please upload an image frame from the sidebar to begin detection.")
        # Load the sample image to show as placeholder
        sample_path = PROJECT_ROOT / "data" / "samples" / "rgb_thermal_preview.png"
        if sample_path.exists():
            st.image(Image.open(sample_path), caption="Awaiting Live Input...", use_column_width=True)
        return

    # Process Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)
    # Convert RGB to BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    st.write("### Live Engine Output")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.spinner("Processing Frame..."):
            t0 = time.perf_counter()
            results = model(frame_bgr, conf=conf_threshold, verbose=False)[0]
            
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
            fps = 1000.0 / max(latency_ms, 0.01)

            # Draw visualization
            overlay = frame_bgr.copy()
            pts = np.array(roi_polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.2, frame_bgr, 0.8, 0, frame_bgr)
            cv2.polylines(frame_bgr, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame_bgr, "RESTRICTED BORDER ROI", (roi_polygon[0][0], roi_polygon[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            alert_ids = {a["track_id"]: a for a in alerts}
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                is_alert = det["id"] in alert_ids
                
                color = (0, 0, 255) if is_alert else (0, 255, 0)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                if is_alert:
                    label = f"[ALERT] {label}"

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert back to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Detection Results | FPS: {fps:.1f}", use_column_width=True)

    with col2:
        st.write("### System Status")
        st.metric("Inference Latency", f"{latency_ms:.1f} ms")
        st.metric("Total Detections", len(detections))
        st.metric("Intrusion Alerts", len(alerts))
        
        st.write("### Event Log")
        if len(alerts) > 0:
            for alert in alerts:
                st.markdown(f'<div class="alert-box alert-danger">🚨 INTRUSION: {alert["class_name"].upper()} [Conf: {alert["confidence"]:.2f}]</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-success">✅ BORDER SECURE</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
