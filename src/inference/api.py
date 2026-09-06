import sys
import time
import logging
import asyncio
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import Response, FileResponse, JSONResponse
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

# ---------------------------------------------------------------------------
# CONSTANTS & SECURITY CONFIGURATION
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
    "application/octet-stream"  # Browser generic fallback
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
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

# Mount Frontend Static Assets
app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
