"""
Evaluation and Benchmarking Module for computing metrics and rendering comparative figures.
"""

from src.evaluation.evaluate_models import ModelEvaluator
from src.evaluation.metrics import calculate_precision_recall_map, compute_iou, compute_ap
from src.evaluation.visualization import plot_comparative_benchmark

__all__ = [
    "ModelEvaluator",
    "calculate_precision_recall_map",
    "compute_iou",
    "compute_ap",
    "plot_comparative_benchmark",
]
