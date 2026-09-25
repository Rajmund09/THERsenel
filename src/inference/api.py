import sys
import time
import logging
import asyncio
from collections import defaultdict
from pathlib import Path

import os
import subprocess
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import Response, FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("thermal_api")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intrusion.border_tracker import BorderIntrusionTracker
from src.intrusion.audit_logger import ForensicIncidentLogger
from src.intrusion.alert_dispatcher import WebhookAlertDispatcher

incident_logger = ForensicIncidentLogger()
webhook_dispatcher = WebhookAlertDispatcher()

# ---------------------------------------------------------------------------
# CONSTANTS & SECURITY CONFIGURATION
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_VIDEO_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB for video files
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
    "application/octet-stream"  # Browser generic fallback
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
ALLOWED_VIDEO_MIME_TYPES = {
    "video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska",
    "video/webm", "application/octet-stream"
}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_CONCURRENT_INFERENCES = 4
RATE_LIMIT_REQUESTS = 60  # Max requests per window per IP
RATE_LIMIT_WINDOW = 60.0  # Window size in seconds

SERVER_START_TIME = time.time()
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
ip_request_history = defaultdict(list)
ip_lock = asyncio.Lock()

app = FastAPI(
    title="Thermal Border Intrusion API",
    description="Military-spec edge AI fusion engine for real-time intrusion monitoring.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# ---------------------------------------------------------------------------
# CORS POLICY (Configured for security)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_credentials=False,
    max_age=600,
)

# ---------------------------------------------------------------------------
# DEFENSE-IN-DEPTH HTTP SECURITY HEADERS MIDDLEWARE
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_and_cache_middleware(request: Request, call_next):
    response = await call_next(request)

    # Standard security defense headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    # Cache control policies
    if request.url.path.endswith((".css", ".js", ".html")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

# ---------------------------------------------------------------------------
# GLOBAL EXCEPTION HANDLERS (Information Hiding & Clean Status Codes)
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": str(exc.detail), "status_code": exc.status_code},
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        errors.append(f"{field}: {msg}")
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input parameter", "detail": "; ".join(errors), "details": errors}
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred during execution. Diagnostic data logged securely."
        }
    )

# ---------------------------------------------------------------------------
# IN-MEMORY SLIDING-WINDOW RATE LIMITER
# ---------------------------------------------------------------------------
async def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    async with ip_lock:
        history = [t for t in ip_request_history[client_ip] if now - t < RATE_LIMIT_WINDOW]
        if len(history) >= RATE_LIMIT_REQUESTS:
            ip_request_history[client_ip] = history
            return False
        history.append(now)
        ip_request_history[client_ip] = history
        return True

# ---------------------------------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------------------------------
model_path = PROJECT_ROOT / "runs" / "rgb_baseline" / "rgb_yolov8" / "weights" / "best.pt"
try:
    if not model_path.exists():
        logger.warning(f"Custom weights not found at {model_path}. Loading default yolov8n.pt fallback.")
        model = YOLO("yolov8n.pt")
    else:
        logger.info(f"Loading trained weights from {model_path}...")
        model = YOLO(str(model_path))
except Exception as e:
    logger.critical(f"Failed to initialize YOLO model: {e}", exc_info=True)
    model = None

def safe_model_predict(frame, conf: float):
    """
    Executes YOLO model prediction with automatic graceful fallback from GPU to CPU
    if device-specific memory or driver exceptions occur.
    """
    global model
    if model is None:
        raise RuntimeError("YOLO model is not initialized.")
    try:
        return model(frame, conf=conf, verbose=False)[0]
    except Exception as e:
        err_msg = str(e).lower()
        if "cuda" in err_msg or "device" in err_msg or "out of memory" in err_msg:
            logger.warning(f"Hardware execution exception ({e}). Gracefully migrating model tensor pipeline to CPU.")
            try:
                model.to("cpu")
                return model(frame, conf=conf, verbose=False)[0]
            except Exception as e2:
                logger.error(f"CPU fallback prediction failed: {e2}")
                raise
        raise

# Default Restricted Border Zone Polygon
ROI_POLYGON = [(50, 150), (600, 150), (600, 450), (50, 450)]

# ---------------------------------------------------------------------------
# SYSTEM HEALTH & READINESS ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health", summary="Service Liveness & Health Probe")
async def health_check():
    device_info = "uninitialized"
    if model is not None:
        try:
            device_info = str(getattr(model.model, "device", "cpu"))
        except Exception:
            device_info = "cpu"

    return {
        "status": "healthy" if model is not None else "degraded",
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 2),
        "model_loaded": bool(model is not None),
        "device": device_info,
        "max_upload_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
        "active_concurrency_limit": MAX_CONCURRENT_INFERENCES,
        "service": "Thermal Border Intrusion Detection Edge API",
        "version": "1.0.0"
    }

# ---------------------------------------------------------------------------
# PREDICTION INFERENCE ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/predict", summary="Execute Multimodal Border Intrusion Detection")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP, TIFF)"),
    conf_threshold: float = Form(0.5, ge=0.01, le=1.0, description="Detection confidence threshold [0.01 - 1.00]")
):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural network inference engine is currently offline."
        )

    # 1. Rate Limiting Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not await check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per minute.",
            headers={"Retry-After": "60"}
        )

    # 2. Content-Length Header Guard
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded payload exceeds maximum allowed limit of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
        )

    # 3. MIME Type & File Extension Validation
    ext = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{content_type or ext}'. Allowed: JPEG, PNG, WEBP, BMP, TIFF."
        )

    # 4. Streamed Read with Memory Cap (Prevents Memory Exhaustion / OOM DoS)
    chunks = []
    bytes_read = 0
    chunk_size = 64 * 1024  # 64 KB chunks
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        bytes_read += len(chunk)
        if bytes_read > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
            )
        chunks.append(chunk)

    if bytes_read == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    # 5. Image Decoding & Integrity Validation
    raw_buffer = b"".join(chunks)
    nparr = np.frombuffer(raw_buffer, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0 or len(frame.shape) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted, unreadable, or invalid image binary."
        )

    # 6. Concurrency Control (Guards GPU/CPU against concurrency starvation)
    try:
        await asyncio.wait_for(inference_semaphore.acquire(), timeout=8.0)
    except asyncio.TimeoutError:
        logger.warning("Inference semaphore timeout — queue saturated.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server inference queue is currently saturated. Please retry shortly.",
            headers={"Retry-After": "3"}
        )

    try:
        t0 = time.perf_counter()
        
        # Model Prediction (Runs YOLO with bounded confidence threshold)
        results = safe_model_predict(frame, conf=conf_threshold)

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

        # Thread-safe Per-Request Intrusion Tracker
        req_tracker = BorderIntrusionTracker(roi_polygon=ROI_POLYGON)
        alerts = req_tracker.process_detections(detections)
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(latency_ms, 0.01)

        # Draw Tactical ROI & Detection Visualizations
        overlay = frame.copy()
        pts = np.array(ROI_POLYGON, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(
            frame, "RESTRICTED BORDER ROI", 
            (ROI_POLYGON[0][0], ROI_POLYGON[0][1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

        alert_ids = {a["track_id"]: a for a in alerts}
        has_alert = False
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            is_alert = det["id"] in alert_ids

            color = (0, 0, 255) if is_alert else (0, 255, 0)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            if is_alert:
                label = f"[ALERT] {label}"
                has_alert = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Add Telemetry Diagnostic HUD Bar
        status_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Alerts: {len(alerts)}"
        box_color = (0, 0, 255) if has_alert else (0, 180, 0)
        cv2.rectangle(frame, (10, 10), (550, 50), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

        # Encode resulting frame to JPEG
        encode_success, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not encode_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode annotated output frame."
            )

        return Response(
            content=encoded_img.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Alerts-Count": str(len(alerts)),
                "X-Inference-Latency-Ms": f"{latency_ms:.1f}",
                "X-Inference-FPS": f"{fps:.1f}",
                "X-Detections-Count": str(len(detections))
            }
        )
    finally:
        inference_semaphore.release()

# ---------------------------------------------------------------------------
# REAL-TIME MJPEG VIDEO STREAMING PIPELINE
# ---------------------------------------------------------------------------
async def stream_frames_generator(source: str = "sample", conf_threshold: float = 0.5):
    """
    Asynchronously captures frames from webcam, RTSP stream, or test clip,
    runs YOLOv8 detection + Border Tracker, and yields multipart MJPEG stream.
    """
    sample_file = PROJECT_ROOT / "Create_a_high_end_cinematic_ac.mp4"
    if source == "sample" and sample_file.exists():
        cap_src = str(sample_file)
    elif str(source).isdigit():
        cap_src = int(source)
    else:
        cap_src = source if source != "sample" else 0

    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened() and sample_file.exists():
        cap = cv2.VideoCapture(str(sample_file))

    tracker = BorderIntrusionTracker(roi_polygon=ROI_POLYGON)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # If stream/file reached end, loop for continuous live monitoring
                if isinstance(cap_src, str) and Path(cap_src).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.04)
                        continue
                else:
                    await asyncio.sleep(0.04)
                    continue

            frame_idx += 1
            t0 = time.perf_counter()

            # Downscale high-res video for smooth web streaming
            h, w = frame.shape[:2]
            if w > 854:
                target_w = 854
                target_h = int(h * (854 / w))
                frame = cv2.resize(frame, (target_w, target_h))
                h, w = target_h, target_w

            # Tactical corridor surveillance perimeter
            stream_roi = [
                (int(w * 0.05), int(h * 0.30)),
                (int(w * 0.95), int(h * 0.30)),
                (int(w * 0.95), int(h * 0.95)),
                (int(w * 0.05), int(h * 0.95))
            ]
            tracker.roi_polygon = stream_roi

            # YOLOv8 Inference
            detections = []
            alerts = []
            if model is not None:
                results = safe_model_predict(frame, conf=conf_threshold)
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
                alerts = tracker.process_detections(detections)

            has_alert = len(alerts) > 0
            latency_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / max(latency_ms, 0.01)

            # Draw Tactical Overlay
            overlay = frame.copy()
            pts = np.array(stream_roi, np.int32).reshape((-1, 1, 2))
            roi_color = (0, 0, 255) if has_alert else (0, 165, 255)
            cv2.fillPoly(overlay, [pts], roi_color)
            cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=roi_color, thickness=2)
            cv2.putText(
                frame, "RESTRICTED PERIMETER // ACTIVE",
                (stream_roi[0][0] + 8, stream_roi[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, roi_color, 2, cv2.LINE_AA
            )

            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                is_alert = det.get("is_breached", False)
                trk_id = det.get("track_id", 1)
                box_color = (0, 0, 255) if is_alert else (0, 230, 118)
                label = f"[ALERT #{trk_id}] {det['class_name'].upper()} {det['confidence']:.2f}" if is_alert else f"[ID #{trk_id}] {det['class_name'].upper()} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2, cv2.LINE_AA)

            # Live HUD
            cv2.rectangle(frame, (10, 10), (480, 48), (15, 15, 18), -1)
            cv2.rectangle(frame, (10, 10), (480, 48), (255, 255, 255), 1)
            hud_text = f"LIVE STREAM | {fps:.1f} FPS | INTRUDERS: {tracker.unique_breached_count} | CURRENT: {len(alerts)}"
            status_color = (0, 0, 255) if has_alert else (0, 230, 118)
            cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2, cv2.LINE_AA)

            # Encode Frame to JPEG
            encode_ok, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if encode_ok:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_img.tobytes() + b'\r\n')

            # Cooperative yield for smooth async streaming
            await asyncio.sleep(0.015)
    finally:
        cap.release()


@app.get("/video_feed", summary="Live Real-Time MJPEG Intrusion Video Stream")
async def video_feed(source: str = "sample", conf: float = 0.45):
    return StreamingResponse(
        stream_frames_generator(source=source, conf_threshold=conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# VIDEO FILE UPLOAD & INFERENCE PIPELINE
# ---------------------------------------------------------------------------
def remove_temp_file(path: str):
    """Cleanup temporary video files."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Could not remove temp file {path}: {e}")


def transcode_to_h264(input_path: Path, output_path: Path) -> bool:
    """
    Transcodes raw OpenCV video to browser-standard H.264 (avc1/yuv420p) with faststart
    so HTML5 video players in Chrome, Edge, and Safari can render and play it smoothly.
    """
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        return res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.error(f"FFmpeg transcoding failed or timed out: {e}")
        return False


@app.post("/predict_video", summary="Execute Video File Intrusion Detection & Annotation")
async def predict_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV, WEBM)"),
    conf_threshold: float = Form(0.5, ge=0.01, le=1.0, description="Detection confidence threshold")
):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural network inference engine is currently offline."
        )

    # 1. Rate Limiting Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not await check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per minute.",
            headers={"Retry-After": "60"}
        )

    # 2. Content-Length & Format Check
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_VIDEO_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded video exceeds maximum allowed limit of {MAX_VIDEO_UPLOAD_SIZE // (1024 * 1024)} MB."
        )

    ext = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_VIDEO_MIME_TYPES and ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{content_type or ext}'. Allowed: MP4, AVI, MOV, MKV, WEBM."
        )

    # 3. Stream upload to a temporary file
    temp_dir = Path(tempfile.gettempdir())
    unique_suffix = f"{time.time()}_{os.getpid()}"
    temp_in_path = temp_dir / f"upload_in_{unique_suffix}{ext}"
    temp_raw_path = temp_dir / f"upload_raw_{unique_suffix}.mp4"
    temp_out_path = temp_dir / f"upload_out_{unique_suffix}.mp4"

    bytes_read = 0
    with open(temp_in_path, "wb") as f_out:
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > MAX_VIDEO_UPLOAD_SIZE:
                temp_in_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds maximum allowed size of {MAX_VIDEO_UPLOAD_SIZE // (1024 * 1024)} MB."
                )
            f_out.write(chunk)

    if bytes_read == 0:
        temp_in_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    # 4. Process Video with Concurrency Control
    try:
        await asyncio.wait_for(inference_semaphore.acquire(), timeout=12.0)
    except asyncio.TimeoutError:
        temp_in_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server inference queue is saturated. Please retry shortly.",
            headers={"Retry-After": "5"}
        )

    try:
        cap = cv2.VideoCapture(str(temp_in_path))
        if not cap.isOpened():
            temp_in_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Corrupted, unreadable, or invalid video file.")

        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        v_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        # Downscale if > 1280 to keep processing fast
        if v_width > 1280:
            scale = 1280 / v_width
            v_width = 1280
            v_height = int(v_height * scale)

        video_roi = [
            (int(v_width * 0.08), int(v_height * 0.25)),
            (int(v_width * 0.92), int(v_height * 0.25)),
            (int(v_width * 0.92), int(v_height * 0.85)),
            (int(v_width * 0.08), int(v_height * 0.85))
        ]
        video_tracker = BorderIntrusionTracker(roi_polygon=video_roi)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_raw_path), fourcc, v_fps, (v_width, v_height))

        frame_count = 0
        MAX_FRAMES_TO_PROCESS = 300  # Caps processing at first 300 frames (~12.5s) for instant turnaround

        t_start = time.perf_counter()
        while cap.isOpened() and frame_count < MAX_FRAMES_TO_PROCESS:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame.shape[1] != v_width or frame.shape[0] != v_height:
                frame = cv2.resize(frame, (v_width, v_height))

            # YOLO Inference
            results = safe_model_predict(frame, conf=conf_threshold)
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

            alerts = video_tracker.process_detections(detections)
            has_alert = len(alerts) > 0

            # Draw Tactical Perimeter Zone
            overlay = frame.copy()
            pts = np.array(video_roi, np.int32).reshape((-1, 1, 2))
            roi_color = (0, 0, 255) if has_alert else (0, 165, 255)
            cv2.fillPoly(overlay, [pts], roi_color)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=roi_color, thickness=2)
            cv2.putText(
                frame, "RESTRICTED PERIMETER ZONE",
                (video_roi[0][0] + 8, video_roi[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, roi_color, 2, cv2.LINE_AA
            )

            # Draw Historical Motion Trails
            for tid, trk in video_tracker.active_tracks.items():
                pts_hist = trk.get("history", [])
                if len(pts_hist) > 1:
                    trail_color = (0, 0, 255) if trk.get("breached", False) else (0, 230, 118)
                    for j in range(1, len(pts_hist)):
                        thickness = max(1, int(np.sqrt(48 / float(len(pts_hist) - j + 1))))
                        pt1 = (int(pts_hist[j - 1][0]), int(pts_hist[j - 1][1]))
                        pt2 = (int(pts_hist[j][0]), int(pts_hist[j][1]))
                        cv2.line(frame, pt1, pt2, trail_color, thickness, cv2.LINE_AA)

            # Draw Bounding Boxes & Tracking Badges
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                is_breached = det.get("is_breached", False)
                trk_id = det.get("track_id", 1)
                box_color = (0, 0, 255) if is_breached else (0, 230, 118)
                label = f"[ALERT #{trk_id}] {det['class_name'].upper()} {det['confidence']:.2f}" if is_breached else f"[ID #{trk_id}] {det['class_name'].upper()} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2, cv2.LINE_AA)

            # Draw Tactical HUD Telemetry Header
            hud_text = f"FRAME {frame_count:04d} | INTRUDERS: {video_tracker.unique_breached_count} | ACTIVE: {len(alerts)}"
            cv2.rectangle(frame, (10, 10), (450, 45), (15, 15, 18), -1)
            cv2.rectangle(frame, (10, 10), (450, 45), (255, 255, 255), 1)
            cv2.putText(frame, hud_text, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255) if has_alert else (0, 230, 118), 2, cv2.LINE_AA)

            writer.write(frame)

        cap.release()
        writer.release()
        total_time = time.perf_counter() - t_start
        avg_fps = frame_count / max(total_time, 0.001)

        # Remove input file
        temp_in_path.unlink(missing_ok=True)

        # Transcode raw OpenCV output to web-native H.264
        h264_success = transcode_to_h264(temp_raw_path, temp_out_path)
        if h264_success:
            temp_raw_path.unlink(missing_ok=True)
            target_response_path = temp_out_path
        else:
            # Fallback if transcoding is unavailable
            logger.warning("Falling back to raw OpenCV video (ffmpeg transcoding bypassed)")
            target_response_path = temp_raw_path

        # Cleanup output files after response is sent
        background_tasks.add_task(remove_temp_file, str(target_response_path))
        if target_response_path != temp_out_path:
            background_tasks.add_task(remove_temp_file, str(temp_out_path))

        unique_intruders = video_tracker.unique_breached_count
        return FileResponse(
            str(target_response_path),
            media_type="video/mp4",
            headers={
                "X-Total-Frames": str(frame_count),
                "X-Total-Alerts": str(unique_intruders),
                "X-Total-Intruders": str(unique_intruders),
                "X-Average-FPS": f"{avg_fps:.1f}",
                "X-Processing-Time-Sec": f"{total_time:.2f}"
            }
        )
    finally:
        inference_semaphore.release()


# ---------------------------------------------------------------------------
# STATIC FRONTEND ROUTES & FAVICONS
# ---------------------------------------------------------------------------
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    ico_file = static_path / "favicon.ico"
    if ico_file.exists():
        return FileResponse(ico_file, media_type="image/x-icon")
    return Response(status_code=404)

@app.get("/favicon.png", include_in_schema=False)
async def favicon_png():
    png_file = static_path / "favicon.png"
    if png_file.exists():
        return FileResponse(png_file, media_type="image/png")
    return Response(status_code=404)

# ---------------------------------------------------------------------------
# FORENSIC AUDIT TRAIL & INCIDENT LOGGING ROUTES
# ---------------------------------------------------------------------------
@app.get("/incidents/recent", summary="Retrieve Recent Perimeter Breach Audit Incidents")
async def get_recent_incidents(limit: int = 50):
    return JSONResponse(content={"incidents": incident_logger.get_recent_incidents(limit=limit)})


@app.get("/incidents/export", summary="Export Forensic Incident Audit Trail as CSV")
async def export_incident_audit_trail():
    csv_content = incident_logger.export_csv_string()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=breach_incidents_audit.csv"}
    )


# Mount Frontend Static Assets
app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
