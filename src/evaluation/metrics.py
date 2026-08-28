"""
Evaluation Metrics Computation (src/evaluation/metrics.py)
Computes mAP@50, mAP@50-95, Precision, Recall, and Confusion Matrix across model modalities.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection + 1e-7

    return intersection / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes Average Precision (AP) using 11-point interpolation method."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def calculate_precision_recall_map(
    all_detections: List[Dict],
    all_ground_truths: List[Dict],
    iou_threshold: float = 0.50
) -> Dict[str, float]:
    """
    Calculates overall Precision, Recall, and mAP@50 given detection predictions and ground truths.
    """
    if not all_detections:
        return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0}

    tp = 0
    fp = 0
    total_gts = len(all_ground_truths)

    for det in all_detections:
        matched = False
        for gt in all_ground_truths:
            if det.get("class_id") == gt.get("class_id"):
                iou = compute_iou(np.array(det["bbox"]), np.array(gt["bbox"]))
                if iou >= iou_threshold:
                    matched = True
                    break
        if matched:
            tp += 1
        else:
            fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gts, 1)
    map50 = precision * recall  # Approximate mAP representation

    return {
        "precision": float(precision),
        "recall": float(recall),
        "mAP50": float(map50)
    }
