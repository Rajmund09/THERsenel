# 🛡️ Thermal-Border-Intrusion: Project Status Dashboard

---

## 📌 Quick Overview

| Property | Value | Status |
|---|---|---|
| **Implementation Plan** | `Automated Border Intrusion Detection Using Thermal–Visible Fusion.pdf` | ✅ **CONFIRMED & ACTIVE** |
| **Current Phase** | **Phase 4 / Milestone M4** (Thermal Baseline Model Training) | 🔄 **IN PROGRESS (Next Step)** |
| **M3 Deliverable** | **RGB YOLOv8 Baseline Model Trained & Validated** | ✅ **100% COMPLETED** |
| **M3 Performance** | **mAP@50: 53.9% \| Person mAP@50: 70.9% \| Car mAP@50: 80.0%** | 🚀 **BENCHMARK SAVED** |
| **M3 Training Time** | **Start: Aug 29 18:41 \| Finish: Aug 30 00:41 (Duration: 6.013 hrs)** | ⏱️ **6.013 HOURS ON GPU** |
| **Hardware Used** | **NVIDIA GeForce RTX 3050 4GB Laptop GPU** (`CUDA 12.1 + AMP`) | ⚡ **AMP ACCELERATED** |

---

## 📊 Milestone M3: RGB Baseline Training Execution Log & Benchmark Results

### 1. Timing & Execution Details
- **Training Start Time**: `Saturday, Aug 29, 2026 @ 18:41:40 IST`
- **Training Finish Time**: `Sunday, Aug 30, 2026 @ 00:41:50 IST`
- **Total Elapsed Duration**: **`6.013 Hours`** (50 Epochs completed)
- **Hardware Platform**: NVIDIA GeForce RTX 3050 4GB Laptop GPU (CUDA 12.1, Automatic Mixed Precision `AMP: Passed`)
- **Dataset Size Used**: 21,060 Training Images / 2,229 Validation Images
- **Model Checkpoint Saved**: [`runs/rgb_baseline/rgb_yolov8/weights/best.pt`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/runs/rgb_baseline/rgb_yolov8/weights/best.pt) *(Size: 6.2 MB)*

---

### 2. Validation Metrics Benchmark (mAP, Precision, Recall)

| Object Class | Images | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| **ALL CLASSES (OVERALL)** | **2,229** | **23,056** | **0.636 (63.6%)** | **0.494 (49.4%)** | **0.539 (53.9%)** | **0.337 (33.7%)** |
| 🚷 **Person (Intruder)** | 1,472 | 7,693 | **0.770 (77.0%)** | **0.631 (63.1%)** | **0.709 (70.9%)** | **0.389 (38.9%)** |
| 🚗 **Car** | 2,007 | 14,413 | **0.828 (82.8%)** | **0.721 (72.1%)** | **0.800 (80.0%)** | **0.561 (56.1%)** |
| 🚌 **Bus** | 273 | 362 | **0.787 (78.7%)** | **0.469 (46.9%)** | **0.625 (62.5%)** | **0.441 (44.1%)** |
| 🏍️ **Motorcycle** | 107 | 132 | **0.714 (71.4%)** | **0.492 (49.2%)** | **0.558 (55.8%)** | **0.299 (29.9%)** |
| 🚲 **Bicycle** | 260 | 363 | 0.518 (51.8%) | 0.444 (44.4%) | 0.434 (43.4%) | 0.260 (26.0%) |
| 🚚 **Truck** | 87 | 93 | 0.200 (20.0%) | 0.204 (20.4%) | 0.110 (11.0%) | 0.071 (7.1%) |

---

### 3. Real-Time Inference Latency Benchmarks
- **Preprocess Speed**: `0.3 ms` per image
- **Inference Latency**: **`3.5 ms` per image** ($\approx$ **285 FPS**)
- **Postprocess Speed**: `1.0 ms` per image
- **Total Frame Processing Time**: **`4.8 ms`** (Ultra-real-time throughput for border security cameras!)

---

## 💻 Dual-GPU System Architecture & Graphics Switch Report

### 1. How & Why We Switched to NVIDIA Graphics
- **How**: We replaced CPU PyTorch with **CUDA-enabled PyTorch (`torch-2.5.1+cu121`)**. PyTorch connects to NVIDIA CUDA drivers and targets `GPU 1 (NVIDIA GeForce RTX 3050)`.
- **Why**: Deep neural networks require millions of matrix math operations per second. CPUs run calculations sequentially, whereas GPUs process thousands of calculations simultaneously across parallel CUDA cores (the RTX 3050 has **2,048 CUDA cores**).

---

### 2. Are We Currently Using Both Graphics Chips? (YES! 🟢)

Your laptop is operating in **Dual-GPU Hybrid Mode (NVIDIA Optimus)**:

| Graphics Hardware | Role & Assigned Task | Status |
|---|---|---|
| **Intel(R) UHD Graphics** *(Integrated)* | Renders Windows OS, VS Code UI, display output, & screen drawing | 🟢 **Active (Desktop GUI)** |
| **NVIDIA GeForce RTX 3050** *(Dedicated)* | Computes PyTorch tensor matrices, loss gradients, & YOLOv8 model training | ⚡ **Active (100% AI Computation)** |

---

## 🚦 Milestone Progress Tracker

| Milestone | Deliverable | Status | Details |
|---|---|---|---|
| **M1** | Environment & Repository Setup | ✅ **COMPLETED** | Packages, setup.py, environment.yml, & CUDA PyTorch configured |
| **M2** | FLIR Dataset & Preprocessing | ✅ **COMPLETED** | 30.7k images preprocessed & `labels.cache` generated |
| **M3** | RGB YOLOv8 Baseline Model | ✅ **COMPLETED** | **mAP50: 53.9%, Person mAP50: 70.9% (Trained in 6.013h)** |
| **M4** | Thermal YOLOv8 Baseline Model | ⚡ **NEXT STEP** | Run in terminal: `python src/training/train_thermal.py` |
| **M5** | Spatial Attention Fusion CNN | ✅ **COMPLETED** | Encoders & Spatial Attention Fusion modules built |
| **M6** | YOLOv8 + Fusion Integration | ✅ **COMPLETED** | Dual-stream architecture & CIoU/BCE losses built |
| **M7** | Scientific Day vs. Night Evaluation | ⚡ **READY ON GPU** | Metric suite (mAP50, IoU, PR) & benchmark plotters built |
| **M8** | Intrusion & Border Tracking | ✅ **COMPLETED** | Polygon ROI breach engine & Multi-Object tracker built |
| **M9** | Real-Time Application Engines | ✅ **COMPLETED** | Predictor, video stream, & live camera readers built |
| **M10**| Jetson Xavier Edge Deployment | ⏳ **PENDING** | ONNX export & TensorRT FP16 optimization (post-training) |

---

## ⚡ Execution Command Pipeline & Next Steps

Run these commands in order in your PowerShell / CMD terminal (inside `C:\Users\prabh\Downloads\thermal-border-intrusion\thermal-border-intrusion`):

### 1️⃣ Train RGB Baseline Model (Milestone M3 — Completed)
```powershell
python src/training/train_rgb.py --epochs 50 --batch 16
```
> *Status: ✅ Completed! 50 epochs trained in 6.013 hrs on RTX 3050. Checkpoint: `runs/rgb_baseline/rgb_yolov8/weights/best.pt`.*

### 2️⃣ Train Thermal Baseline Model (Milestone M4 — NEXT STEP)
```powershell
python src/training/train_thermal.py --epochs 50 --batch 16
```
> *Trains baseline YOLOv8 on infrared thermal imagery on GPU. Checkpoint saved to `runs/thermal_baseline/thermal_yolov8/weights/best.pt`.*

### 3️⃣ Train Dual-Stream Spatial Attention Fusion Model (Milestones M5 & M6)
```powershell
python src/training/train_fusion.py --epochs 50 --batch 16 --lr 0.001
```
> *Trains Spatial Attention RGB+Thermal Fusion Detector on GPU. Checkpoint saved to `weights/fusion_best.pt`.*

### 4️⃣ Execute Scientific Evaluation & Day vs. Night Benchmarking (Milestone M7)
```powershell
python src/evaluation/evaluate_models.py
```
> *Calculates mAP@50, mAP@50-95, Precision, Recall, and FPS across Day and Night conditions.*

### 5️⃣ Test Real-Time Intrusion Detection Engine (Milestones M8 & M9)
```powershell
python src/inference/predict.py
```
> *Runs real-time fusion detection, object tracking, polygon ROI border crossing tests, and triggers alert overlays.*

---

## 📁 Repository Structure & File Audit

### Core System Modules (`src/`)

| Subsystem | File Path | Description | Status |
|---|---|---|---|
| **Data** | [`src/data/explore_dataset.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/data/explore_dataset.py) | Scans raw FLIR dataset & generates preview plots | ✅ Complete |
| **Data** | [`src/data/preprocessing.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/data/preprocessing.py) | COCO JSON to YOLO normalized `.txt` converter | ✅ Complete |
| **Data** | [`src/data/dataset.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/data/dataset.py) | PyTorch `RGBThermalDataset` loader | ✅ Complete |
| **Data** | [`src/data/alignment.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/data/alignment.py) | ORB Homography registration (`RGBIRAligner`) | ✅ Complete |
| **Data** | [`src/data/augmentation.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/data/augmentation.py) | Synchronized Albumentations spatial transforms | ✅ Complete |
| **Models**| [`src/models/rgb_encoder.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/models/rgb_encoder.py) | ResNet-style visual RGB feature CNN encoder | ✅ Complete |
| **Models**| [`src/models/thermal_encoder.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/models/thermal_encoder.py) | ResNet-style infrared thermal feature CNN encoder | ✅ Complete |
| **Models**| [`src/models/fusion_model.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/models/fusion_model.py) | `SpatialAttentionFusion` & Dual-Stream Detector | ✅ Complete |
| **Models**| [`src/models/detector.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/models/detector.py) | Decoupled classification & box regression head | ✅ Complete |
| **Models**| [`src/models/fusion_yolov8.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/models/fusion_yolov8.py) | Dual-stream YOLOv8 spatial fusion wrapper | ✅ Complete |
| **Training**| [`src/training/train_rgb.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/training/train_rgb.py) | RGB-only YOLOv8 baseline training script (GPU auto) | ✅ Complete |
| **Training**| [`src/training/train_thermal.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/training/train_thermal.py) | Thermal-only YOLOv8 baseline training script (GPU auto) | ✅ Complete |
| **Training**| [`src/training/train_fusion.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/training/train_fusion.py) | End-to-end PyTorch Fusion training script | ✅ Complete |
| **Training**| [`src/training/losses.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/training/losses.py) | Complete IoU (`bbox_ciou`) & BCE loss module | ✅ Complete |
| **Evaluation**| [`src/evaluation/evaluate_models.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/evaluation/evaluate_models.py) | Day vs. Night model comparison benchmark runner | ✅ Complete |
| **Evaluation**| [`src/evaluation/metrics.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/evaluation/metrics.py) | IoU, 11-point AP, mAP@50, Precision, & Recall | ✅ Complete |
| **Evaluation**| [`src/evaluation/visualization.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/evaluation/visualization.py) | Comparative performance chart generator | ✅ Complete |
| **Inference**| [`src/inference/predict.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/inference/predict.py) | Real-time fusion predictor & ROI tracker launcher | ✅ Complete |
| **Inference**| [`src/inference/video.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/inference/video.py) | Dual video file stream processing engine | ✅ Complete |
| **Inference**| [`src/inference/camera.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/inference/camera.py) | Live Webcam, RTSP, & Jetson CSI camera reader | ✅ Complete |
| **Intrusion**| [`src/intrusion/border_tracker.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/intrusion/border_tracker.py) | Ray-Casting ROI polygon border breach detector | ✅ Complete |
| **Intrusion**| [`src/intrusion/roi.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/intrusion/roi.py) | ROI polygon manager & semi-transparent painter | ✅ Complete |
| **Intrusion**| [`src/intrusion/tracker.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/intrusion/tracker.py) | Multi-Object Tracker ID assigner | ✅ Complete |
| **Intrusion**| [`src/intrusion/alert.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/intrusion/alert.py) | Event logger & visual alarm banner dispatcher | ✅ Complete |
