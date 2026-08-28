# 🛡️ Thermal-Border-Intrusion: Project Status Dashboard

---

## 📌 Quick Overview

| Property | Value | Status |
|---|---|---|
| **Implementation Plan** | `Automated Border Intrusion Detection Using Thermal–Visible Fusion.pdf` | ✅ **CONFIRMED & ACTIVE** |
| **Current Phase** | **Phase 2 / Milestone M2** (Dataset Preprocessing) | 🔄 **IN PROGRESS** |
| **FLIR ADAS Dataset** | **46,429 Images** (15,156 RGB + 31,273 Thermal) | ✅ **VERIFIED & COMPLETE** |
| **Manual Action Required**| **None (0 tasks remain)** | 🎉 **100% AUTOMATED** |

---

## 🚦 Milestone Progress Tracker

| Milestone | Deliverable | Status | Details |
|---|---|---|---|
| **M1** | Environment & Repository Setup | ✅ **COMPLETED** | Packages, setup.py, environment.yml, & configs created |
| **M2** | FLIR Dataset & Preprocessing | 🔄 **IN PROGRESS** | 46.4k images verified; ready to run `preprocessing.py` |
| **M3** | RGB YOLOv8 Baseline Model | ⚡ **READY TO RUN** | Training script ready (`python src/training/train_rgb.py`) |
| **M4** | Thermal YOLOv8 Baseline Model | ⚡ **READY TO RUN** | Training script ready (`python src/training/train_thermal.py`) |
| **M5** | Spatial Attention Fusion CNN | ✅ **COMPLETED** | Encoders & Spatial Attention Fusion modules built |
| **M6** | YOLOv8 + Fusion Integration | ✅ **COMPLETED** | Dual-stream architecture & CIoU/BCE losses built |
| **M7** | Scientific Day vs. Night Evaluation | ⚡ **READY TO RUN** | Metric suite (mAP50, IoU, PR) & benchmark plotters built |
| **M8** | Intrusion & Border Tracking | ✅ **COMPLETED** | Polygon ROI breach engine & Multi-Object tracker built |
| **M9** | Real-Time Application Engines | ✅ **COMPLETED** | Predictor, video stream, & live camera readers built |
| **M10**| Jetson Xavier Edge Deployment | ⏳ **PENDING** | ONNX export & TensorRT FP16 optimization (post-training) |

---

## ⚡ Execution Command Pipeline

Run these commands in order to execute the pipeline:

### 1. Preprocess Dataset & Convert Labels to YOLO Format
```bash
python src/data/preprocessing.py
```
> *Converts FLIR COCO JSON annotations to normalized YOLO `.txt` labels in `data/processed/`.*

### 2. Train RGB Baseline Model
```bash
python src/training/train_rgb.py --epochs 50 --batch 16
```
> *Trains baseline YOLOv8 on visual RGB imagery; saves weights to `runs/rgb_baseline/`.*

### 3. Train Thermal Baseline Model
```bash
python src/training/train_thermal.py --epochs 50 --batch 16
```
> *Trains baseline YOLOv8 on infrared thermal imagery; saves weights to `runs/thermal_baseline/`.*

### 4. Train Dual-Stream Spatial Attention Fusion Model
```bash
python src/training/train_fusion.py --epochs 50 --batch 16 --lr 0.001
```
> *Trains dual-stream Spatial Attention Fusion network; saves weights to `weights/fusion_best.pt`.*

### 5. Execute Scientific Evaluation & Day vs. Night Benchmarking
```bash
python src/evaluation/evaluate_models.py
```
> *Calculates mAP@50, mAP@50-95, Precision, Recall, and FPS across Day and Night conditions.*

### 6. Test Real-Time Intrusion Detection Engine
```bash
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
| **Training**| [`src/training/train_rgb.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/training/train_rgb.py) | RGB-only YOLOv8 baseline training script | ✅ Complete |
| **Training**| [`src/training/train_thermal.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/src/training/train_thermal.py) | Thermal-only YOLOv8 baseline training script | ✅ Complete |
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

---

### Project Configuration & Environment Files

- [`setup.py`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/setup.py) — Package installation setup (`pip install -e .`)
- [`environment.yml`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/environment.yml) — Conda environment definition (Python 3.11, PyTorch, CUDA 12.1)
- [`requirements.txt`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/requirements.txt) — Pip requirements specification
- [`configs/dataset.yaml`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/configs/dataset.yaml) — Dataset paths & class definitions
- [`configs/model.yaml`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/configs/model.yaml) — Model architecture & training hyperparameters
- [`configs/deployment.yaml`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/configs/deployment.yaml) — Jetson Xavier deployment & ROI polygon coordinates
- [`README.md`](file:///c:/Users/prabh/Downloads/thermal-border-intrusion/thermal-border-intrusion/README.md) — Comprehensive project documentation
