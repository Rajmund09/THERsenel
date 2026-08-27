"""
RGB YOLOv8 Baseline Training Script (src/training/train_rgb.py)
Trains YOLOv8 baseline model exclusively on RGB visual imagery.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"


def main():
    parser = argparse.ArgumentParser(description="Train RGB YOLOv8 Baseline")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO weights")
    args = parser.parse_args()

    print("==========================================================================")
    print("                TRAINING RGB YOLOV8 BASELINE MODEL                        ")
    print("==========================================================================")

    model = YOLO(args.model)
    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(PROJECT_ROOT / "runs" / "rgb_baseline"),
        name="rgb_yolov8",
        exist_ok=True
    )
    print("[?] RGB Baseline Training Completed.")


if __name__ == "__main__":
    main()
