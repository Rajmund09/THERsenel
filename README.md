# Thermal-Border-Intrusion

Automated border intrusion detection using RGB + thermal image fusion and YOLOv8,
deployed on NVIDIA Jetson Xavier.

## Project story (research questions)

1. Does thermal imagery improve detection compared with RGB alone?
2. Does RGB + thermal fusion outperform either modality alone?
3. Does fusion provide greater benefit under low-light conditions?
4. Can the fusion detector hit practical inference speed on Jetson Xavier?

## Milestones

| # | Deliverable |
|---|---|
| M1 | Environment + Git project |
| M2 | FLIR dataset prepared |
| M3 | RGB YOLOv8 baseline |
| M4 | Thermal YOLOv8 baseline |
| M5 | Fusion CNN |
| M6 | Fusion + YOLOv8 |
| M7 | Evaluation + comparison |
| M8 | Intrusion / ROI / tracking |
| M9 | Real-time application |
| M10 | Jetson Xavier deployment |

## Setup

```bash
# 1. Create environment
conda create -n thermal-border python=3.11 -y
conda activate thermal-border

# 2. Install PyTorch with CUDA (check pytorch.org for your CUDA version, example below is CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt

# 4. Verify
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Getting the dataset

1. Download the FLIR ADAS / thermal dataset (Teledyne FLIR site, or Kaggle mirror).
2. Extract it, unmodified, into `data/raw/FLIR/`.
3. Never edit files in `data/raw/` directly — all processing writes to `data/processed/`.

## Folder structure

```
thermal-border-intrusion/
├── configs/            # dataset.yaml, model.yaml, deployment.yaml
├── data/
│   ├── raw/FLIR/        # untouched original dataset
│   ├── processed/       # train/val/test after preprocessing
│   └── samples/         # a few images for quick sanity checks
├── notebooks/           # exploration & experiment notebooks
├── src/
│   ├── data/            # dataset loading, preprocessing, alignment, augmentation
│   ├── models/           # rgb_encoder, thermal_encoder, fusion, detector
│   ├── training/         # train_rgb.py, train_thermal.py, train_fusion.py
│   ├── evaluation/       # metrics, comparison plots
│   ├── inference/        # image/video/camera inference
│   └── intrusion/        # ROI, tracker, alert logic
└── weights/              # saved model checkpoints (rgb / thermal / fusion)
```

## Next step

Run `src/data/explore_dataset.py` after placing the dataset in `data/raw/FLIR/`
to get counts, class distribution, and RGB/thermal sample pairs (M2).
