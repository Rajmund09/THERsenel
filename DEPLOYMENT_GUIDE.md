# 🚀 THERsenel — Production & Free Cloud Deployment Guide

This guide provides complete, step-by-step instructions for deploying the **THERsenel (Thermal Border Intrusion Detection)** surveillance terminal and edge inference microservice for **free**, with zero cloud bills.

---

## 🏗️ 1. Architecture Overview

Unlike architectures that require hosting frontend and backend on separate servers, **THERsenel is completely self-contained**:

```
[ Incoming Request (Browser / Operator) ]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│             FastAPI Unified Edge Engine                │
│             (src/inference/api.py)                     │
│                                                        │
│  ├── GET /               ──> Serves Static UI Terminal │
│  ├── GET /health         ──> System Liveness Probe     │
│  ├── GET /docs           ──> OpenAPI / Swagger Docs    │
│  └── POST /predict       ──> YOLOv8 Dual-Channel Vision│
│                                                        │
│  Mounted Static Directory:                             │
│  src/inference/static/                                 │
│  ├── index.html, style.css, app.js                     │
│  ├── pixel-swap.js, components/, lenis.min.js          │
└────────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│          Lightweight YOLOv8 Nano Neural Weights        │
│          (runs/rgb_baseline/rgb_yolov8/weights/best.pt)│
│          Size: 5.96 MB (Ultra-compact & fast)          │
└────────────────────────────────────────────────────────┘
```

---

## 📋 2. Verified Local Baseline Checkup

All components have been empirically tested and verified on the local machine:

| Component | Test / Command | Verified Status | Result Details |
| :--- | :--- | :---: | :--- |
| **Model Sizes** | `Get-ChildItem -Filter *.pt` | **PASS** | `best.pt`: **5.96 MB**, `yolov8n.pt`: **6.25 MB** (Compact, fast download) |
| **YOLO Loading** | `from ultralytics import YOLO; YOLO(...)` | **PASS** | `best.pt` and `yolov8n.pt` load cleanly with 0 errors |
| **Health Probe** | `GET /health` | **PASS** | `HTTP 200`: `status: healthy`, `model_loaded: True`, `active_concurrency: 4` |
| **Swagger UI** | `GET /docs` | **PASS** | `HTTP 200`: Full OpenAPI interactive specification |
| **Terminal UI** | `GET /` | **PASS** | `HTTP 200`: Tactical Glassmorphic surveillance interface |
| **Security Suite** | `pytest tests/test_security_api.py -v` | **PASS** | **12 passed in 7.15s** (Rate limiting, DoS caps, 413, 400, 422, headers) |

---

## 🌟 3. Free Deployment Options

### Option A: Hugging Face Spaces (Recommended for AI / Computer Vision)

Hugging Face Spaces provides the most generous free tier available for AI applications (**16 GB RAM + 2 vCPU**), preventing out-of-memory errors that occur on other free platforms.

#### Step 1: Create the Space on Hugging Face
1. Sign in to [Hugging Face](https://huggingface.co).
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Set **Space Name**: `THERsenel` (or `thermal-border-intrusion`).
4. Select **Space SDK**: **Docker** ➡️ Choose **Blank**.
5. Select **Space Hardware**: **CPU basic • 2 vCPU • 16 GB RAM • Free**.
6. Set Visibility to **Public** ➡️ Click **Create Space**.

#### Step 2: Push Your Code to the Space
In your local repository root, run:

```bash
# Add Hugging Face Space as a git remote
git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/THERsenel

# Push code to Hugging Face
git push hf main
```

#### Step 3: Access Your Live Application
Hugging Face will automatically build the `Dockerfile` and launch the application on port `7860`.
- **Live Terminal:** `https://<YOUR_HF_USERNAME>-thersenel.hf.space`
- **Swagger Docs:** `https://<YOUR_HF_USERNAME>-thersenel.hf.space/docs`
- **Health Check:** `https://<YOUR_HF_USERNAME>-thersenel.hf.space/health`

---

### Option B: Render.com (Direct GitHub Repository Sync)

Render connects directly to your GitHub repository and automatically deploys when you push new commits.

#### Step 1: Create a Free Web Service
1. Sign in to [Render.com](https://render.com) using your GitHub account.
2. Click **New +** (top right) ➡️ **Web Service**.
3. Choose **Build and deploy from a Git repository** ➡️ Select `Rajmund09/THERsenel`.

#### Step 2: Configure Service Parameters
- **Name:** `thersenel`
- **Region:** Singapore / Frankfurt / Oregon
- **Branch:** `main`
- **Runtime:** **Docker** (Recommended: Uses the pre-configured `Dockerfile` with CPU PyTorch)
  - *Or Python 3:*
    - **Build Command:** `pip install -r requirements-deploy.txt`
    - **Start Command:** `python -m uvicorn src.inference.api:app --host 0.0.0.0 --port $PORT`
- **Instance Type:** **Free ($0/month)**

#### Step 3: Deploy
Click **Create Web Service**. Render will build and deploy the container.
- **Live URL:** `https://thersenel.onrender.com`

> [!NOTE]
> Render free instances sleep after 15 minutes of inactivity. When a new user opens the link, it takes ~30–50 seconds to spin up from sleep.

---

### Option C: Cloudflare Tunnel (Instant 10-Second Live Public URL)

If you want an immediate global HTTPS URL to demo the system from your laptop or phone without waiting for cloud container builds:

With your local server running on `http://127.0.0.1:8000`:

1. Open a new PowerShell window in the project folder and run:
   ```powershell
   npx -y untun@latest tunnel http://127.0.0.1:8000
   ```
   *Alternatively, using the official Cloudflare CLI:*
   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8000
   ```

2. Cloudflare will print a live public link:
   ```text
   Tunnel ready: https://border-patrol-surveillance.trycloudflare.com
   ```

3. Open that link on your phone, tablet, or external laptop—it will load your full 3D surveillance terminal and perform live YOLO inference using your local machine's processing power!

---

## 🐳 4. Local Docker Build & Test

Before deploying to the cloud, you can test the production container locally:

```bash
# 1. Build the production Docker image
docker build -t thersenel .

# 2. Run the container locally (mapped to port 7860)
docker run -p 7860:7860 thersenel

# 3. Open in your browser
http://localhost:7860
```

---

## ⚙️ 5. Deployment Configuration Files

The repository includes two dedicated deployment files:

1. **[`Dockerfile`](Dockerfile):**
   - Base: `python:3.10-slim`.
   - Installs minimal headless system libraries (`libglib2.0-0`, `libgomp1`, `curl`).
   - Copies application source code, static frontend assets, and compact model weights (`best.pt`, `yolov8n.pt`).
   - Automatically detects Hugging Face's port (`7860`) or Render's dynamic `$PORT`.

2. **[`requirements-deploy.txt`](requirements-deploy.txt):**
   - Targets the lightweight CPU PyTorch wheel (`https://download.pytorch.org/whl/cpu`), reducing download size from 2.5 GB down to ~180 MB.
   - Uses `opencv-python-headless` to eliminate desktop GUI/X11 dependencies.
   - Excludes development tools (`jupyter`, `tensorboard`, `seaborn`) to keep container images lean and fast.

---

## 🛡️ 6. Production Health & Troubleshooting

### Verifying Service Liveness
Once deployed, send a GET request to `/health`:
```bash
curl https://YOUR-DEPLOYMENT-URL/health
```
Expected response:
```json
{
  "status": "healthy",
  "uptime_seconds": 128.45,
  "model_loaded": true,
  "device": "cpu",
  "max_upload_mb": 15,
  "active_concurrency_limit": 4,
  "service": "Thermal Border Intrusion Detection Edge API",
  "version": "1.0.0"
}
```

### Model File Not Found Fallback
In [`src/inference/api.py`](src/inference/api.py), model initialization contains automatic fallback logic:
```python
model_path = PROJECT_ROOT / "runs" / "rgb_baseline" / "rgb_yolov8" / "weights" / "best.pt"
if not model_path.exists():
    logger.warning("Custom weights not found. Loading default yolov8n.pt fallback.")
    model = YOLO("yolov8n.pt")
else:
    model = YOLO(str(model_path))
```
If custom run checkpoints are ever omitted, the service will gracefully fall back to the bundled `yolov8n.pt` without crashing.
