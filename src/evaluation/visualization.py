"""
Evaluation Visualization & Chart Generator (src/evaluation/visualization.py)
Generates Precision-Recall curves, mAP bar charts, Day vs. Night comparative plots, and detection overlay grids.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def plot_comparative_benchmark(results: List[Dict], save_path: Path):
    """
    Plots a grouped bar chart comparing RGB, Thermal, and Fusion mAP@50 performance across Day and Night conditions.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    modalities = ["RGB", "THERMAL", "FUSION"]
    day_maps = [res["mAP50"] for res in results if res["condition"] == "day"]
    night_maps = [res["mAP50"] for res in results if res["condition"] == "night"]

    if len(day_maps) < 3 or len(night_maps) < 3:
        return

    x = np.arange(len(modalities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width / 2, day_maps, width, label="Daylight", color="#4C72B0")
    rects2 = ax.bar(x + width / 2, night_maps, width, label="Night / Low-Light", color="#DD8452")

    ax.set_ylabel("mAP@50 Score")
    ax.set_title("Thermal-Visible Fusion vs. Single Modality Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[+] Saved comparative benchmark plot to {save_path}")
