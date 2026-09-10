# ===========================================================================
# THERsenel - Production Dockerfile for Free Cloud Deployment
# Compatible with Hugging Face Spaces (Port 7860) & Render.com ($PORT)
# ===========================================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install minimal OS dependencies for headless OpenCV and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight deployment requirements (CPU-only PyTorch)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

# Copy application source, weights, configs, and frontend
COPY src/ ./src/
COPY configs/ ./configs/
COPY weights/ ./weights/
COPY yolov8n.pt ./yolov8n.pt
COPY runs/ ./runs/
COPY README.md ./README.md

# Default port for Hugging Face Spaces is 7860
EXPOSE 7860

# Shell form CMD ensures $PORT is dynamically expanded on Render or defaults to 7860 on Hugging Face
CMD sh -c "python -m uvicorn src.inference.api:app --host 0.0.0.0 --port ${PORT:-7860}"
