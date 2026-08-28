# Thermal-Border-Intrusion

Automated Border Intrusion Detection Using Thermal-Visible (RGB) Image Fusion and YOLOv8, Deployed on NVIDIA Jetson Xavier.

---

## 🎯 Research Objectives & Questions

1. **RGB vs. Thermal**: Does thermal infrared imagery improve object detection accuracy compared to RGB imagery alone?
2. **Multimodal Fusion**: Does Spatial Attention RGB + Thermal fusion outperform single-modality baselines?
3. **Environmental Visibility**: Does fusion provide a larger performance gain under low-light/night conditions compared to daylight?
4. **Edge Deployment**: Can the dual-stream fusion detector achieve real-time inference (>30 FPS) when deployed on NVIDIA Jetson Xavier?

---

## 🚀 Development Milestones

| Milestone | Deliverable Description | Status |
|---|---|---|
| **M1** | Environment setup, Git repository, and professional project structure | ✅ Completed |
| **M2** | FLIR ADAS dataset exploration, homography alignment, & YOLO preprocessing | ✅ Modules Ready |
| **M3** | RGB-only YOLOv8 baseline model training & metrics | ⏳ Pending Dataset |
| **M4** | Thermal-only YOLOv8 baseline model training & metrics | ⏳ Pending Dataset |
| **M5** | PyTorch Dual-Stream Spatial Attention Fusion CNN network implementation | ✅ Completed |
| **M6** | End-to-end YOLOv8 + Fusion integration & loss functions | ✅ Completed |
| **M7** | Scientific evaluation & Day vs. Night comparative mAP benchmarking | ✅ Modules Ready |
| **M8** | Polygon ROI definition & ByteTrack multi-object tracking integration | ✅ Completed |
| **M9** | Real-time dual-video/camera intrusion inference application | ✅ Completed |
| **M10** | ONNX export, TensorRT FP16 optimization, & Jetson Xavier benchmarking | ⏳ Pending Deployment |

---

## 🏗️ Professional Project Architecture

```
thermal-border-intrusion/
├── configs/                  # YAML configurations
│   ├── dataset.yaml          # YOLO dataset paths & class labels
│   ├── model.yaml            # ResNet-18 dual encoders, Spatial Attention Fusion params
│   └── deployment.yaml       # Jetson Xavier TensorRT params & ROI border polygon
├── data/                     # Dataset storage
│   ├── raw/FLIR/             # Original untouched FLIR ADAS dataset
│   ├── processed/            # Preprocessed train/val/test splits (YOLO format)
│   └── samples/              # Preview image grids for sanity checks
├── notebooks/                # Jupyter exploration & experiment notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_rgb_baseline.ipynb
│   ├── 04_thermal_baseline.ipynb
│   ├── 05_fusion_experiments.ipynb
│   └── 06_evaluation.ipynb
├── src/                      # Core Source Code Package
│   ├── data/                 # Dataset loader, alignment, augmentation, & exploration
│   │   ├── dataset.py        # PyTorch Multimodal RGBThermalDataset loader
│   │   ├── alignment.py      # RGB-Thermal homography registration
│   │   ├── augmentation.py   # Synchronized Albumentations pipeline
│   │   ├── explore_dataset.py# Raw FLIR scanning & summary generator
│   │   └── preprocessing.py  # COCO JSON to YOLO format converter
│   ├── models/               # Neural network architectures
│   │   ├── rgb_encoder.py    # ResNet-style RGB feature encoder
│   │   ├── thermal_encoder.py# ResNet-style Thermal feature encoder
│   │   ├── fusion_model.py   # SpatialAttentionFusion & RGBThermalFusionDetector
│   │   ├── detector.py       # Decoupled Object Detection Head
│   │   └── fusion_yolov8.py  # Dual-stream YOLOv8 spatial fusion wrapper
│   ├── training/             # Model training pipelines & loss functions
│   │   ├── train_rgb.py      # Baseline RGB YOLOv8 training script
│   │   ├── train_thermal.py  # Baseline Thermal YOLOv8 training script
│   │   ├── train_fusion.py   # End-to-end PyTorch Fusion training script
│   │   └── losses.py         # CIoU loss & BCE classification loss
│   ├── evaluation/           # Evaluation metrics & visualization
│   │   ├── evaluate_models.py# Model evaluation & comparative benchmark suite
│   │   ├── metrics.py        # IoU, AP, mAP@50, Precision, & Recall calculation
│   │   └── visualization.py  # Plotting comparative charts & detection overlays
│   ├── inference/            # Real-time inference engines
│   │   ├── predict.py        # Single image / frame real-time inference launcher
│   │   ├── video.py          # Dual video file stream processing engine
│   │   └── camera.py         # Live Webcam / RTSP / Jetson CSI camera reader
│   └── intrusion/            # Security border intrusion logic
│       ├── border_tracker.py # Ray-Casting algorithm for polygon border breach test
│       ├── roi.py            # ROI polygon manager & overlay painter
│       ├── tracker.py        # Multi-Object Tracker ID assigner
│       └── alert.py          # Event logging, audio/visual alarm dispatcher
├── weights/                  # Saved checkpoint weights (.pt, .onnx, .engine)
│   ├── rgb/
│   ├── thermal/
│   └── fusion/
├── logs/                     # TensorBoard logs & intrusion event logs
├── PROJECT_STATUS.md         # Full project status & phase tracking document
├── setup.py                  # Package installation configuration
├── environment.yml           # Conda environment definition file
├── requirements.txt          # Pip dependencies specification file
└── README.md                 # Project documentation
```

---

## ⚙️ Environment Setup & Installation

### Option A: Conda Environment (Recommended)

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate thermal-border

# 2. Install editable package
pip install -e .
```

### Option B: Pip Virtual Environment

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install PyTorch with CUDA support (Example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install requirements and package
pip install -r requirements.txt
pip install -e .
```

---

## 🏃 Usage Commands

### 1. Dataset Exploration & Preprocessing (Milestone M2)
Place original FLIR ADAS files into `data/raw/FLIR/` and run:

```bash
# Explore dataset counts & resolutions
python src/data/explore_dataset.py

# Convert COCO JSON annotations to YOLO format under data/processed/
python src/data/preprocessing.py
```

### 2. Single-Modality Baseline Model Training (Milestones M3 & M4)

```bash
# Train RGB YOLOv8 baseline
python src/training/train_rgb.py --epochs 50 --batch 16

# Train Thermal YOLOv8 baseline
python src/training/train_thermal.py --epochs 50 --batch 16
```

### 3. Dual-Stream Spatial Attention Fusion Training (Milestones M5 & M6)

```bash
# Train Spatial Attention RGB+Thermal Fusion Detector
python src/training/train_fusion.py --epochs 50 --batch 16 --lr 0.001
```

### 4. Scientific Evaluation & Benchmarking (Milestone M7)

```bash
# Run comparative evaluation suite across Day vs. Night scenarios
python src/evaluation/evaluate_models.py
```

### 5. Real-Time Intrusion Detection & Application (Milestones M8 & M9)

```bash
# Run real-time detection & ROI border tracking engine
python src/inference/predict.py
```

---

## 📌 Citation & References

- Teledyne FLIR ADAS Thermal Dataset
- Ultralytics YOLOv8 Architecture
- PyTorch Deep Learning Framework
- NVIDIA TensorRT & Jetson Xavier Platform
