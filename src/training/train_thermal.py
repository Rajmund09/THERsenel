"""
Thermal YOLOv8 Baseline Training Script (src/training/train_thermal.py)
Trains YOLOv8 baseline model exclusively on Thermal Infrared imagery.
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"


def main():
    default_device = "0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Train Thermal YOLOv8 Baseline")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=default_device, help="CUDA device ID (e.g. 0) or cpu")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO weights")
    args = parser.parse_args()

    print("==========================================================================")
    print(f"     TRAINING THERMAL YOLOV8 BASELINE MODEL (Device: {args.device})       ")
    print("==========================================================================")

    model = YOLO(args.model)
    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(PROJECT_ROOT / "runs" / "thermal_baseline"),
        name="thermal_yolov8",
        exist_ok=True
    )
    print("[?] Thermal Baseline Training Completed.")


if __name__ == "__main__":
    main()
