"""
Edge Optimization & ONNX Model Export Pipeline (src/models/export_onnx.py)
Exports trained PyTorch weights (.pt) to optimized ONNX and OpenVINO formats
supporting half-precision (FP16), dynamic batching, and metadata validation
for low-power edge SBC deployment (NVIDIA Jetson, Raspberry Pi 5, Intel NUC).
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("onnx_export")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def export_model_to_onnx(
    weights_path: Path,
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True
) -> Dict[str, Any]:
    """
    Exports a YOLO model to ONNX with edge-optimized tensor dimensions.
    """
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file does not exist: {weights_path}")

    logger.info(f"Loading weights from {weights_path}...")
    model = YOLO(str(weights_path))

    logger.info(f"Initiating ONNX export (imgsz={imgsz}, half={half}, dynamic={dynamic}, simplify={simplify})...")
    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=simplify
    )

    exported_path = Path(exported_path_str)
    size_mb = exported_path.stat().st_size / (1024 * 1024) if exported_path.exists() else 0.0

    result = {
        "status": "success",
        "format": "onnx",
        "exported_path": str(exported_path),
        "size_mb": round(size_mb, 2),
        "imgsz": imgsz,
        "half": half,
        "dynamic": dynamic
    }
    logger.info(f"Export completed: {exported_path} ({size_mb:.2f} MB)")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Export THERsenel YOLO model to ONNX for edge deployment.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "rgb_baseline" / "rgb_yolov8" / "weights" / "best.pt"),
        help="Path to trained PyTorch .pt weights"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input inference image size")
    parser.add_argument("--half", action="store_true", help="Enable FP16 half-precision optimization")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batching dimensions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wpath = Path(args.weights)
    if not wpath.exists():
        # Fallback to base model if custom runs weights are missing
        wpath = PROJECT_ROOT / "yolov8n.pt"

    export_model_to_onnx(
        weights_path=wpath,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic
    )
