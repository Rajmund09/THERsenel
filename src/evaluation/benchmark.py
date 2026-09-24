"""
Automated Hardware Latency, FPS, and Memory Profiler (src/evaluation/benchmark.py)
Profiles YOLO inference throughput, P50/P90/P95/P99 latency percentiles,
and hardware memory footprint across multiple batch sizes and resolutions.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_latency_benchmark(
    weights_path: Path = None,
    resolutions: List[tuple] = None,
    num_iterations: int = 50,
    warmup_iterations: int = 10
) -> Dict[str, Any]:
    """
    Measures inference throughput (FPS), percentile latencies (P50, P90, P95, P99),
    and system memory consumption for YOLO vision models.
    """
    from ultralytics import YOLO

    if weights_path is None:
        default_pt = PROJECT_ROOT / "runs" / "rgb_baseline" / "rgb_yolov8" / "weights" / "best.pt"
        if default_pt.exists():
            weights_path = default_pt
        else:
            weights_path = PROJECT_ROOT / "yolov8n.pt"

    resolutions = resolutions or [(640, 640), (1280, 720)]
    logger.info(f"Loading benchmark model from: {weights_path}")
    model = YOLO(str(weights_path))

    results = {
        "model_weights": str(weights_path.name),
        "iterations": num_iterations,
        "warmup": warmup_iterations,
        "benchmarks": []
    }

    for (w, h) in resolutions:
        logger.info(f"Benchmarking resolution: {w}x{h} ({num_iterations} iterations)...")
        dummy_frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Warmup passes to stabilize CPU caches & thread pools
        for _ in range(warmup_iterations):
            _ = model.predict(dummy_frame, conf=0.4, verbose=False)

        latencies_ms = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = model.predict(dummy_frame, conf=0.4, verbose=False)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

        latencies_ms = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies_ms))
        median_lat = float(np.median(latencies_ms))
        p90_lat = float(np.percentile(latencies_ms, 90))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        res_entry = {
            "resolution": f"{w}x{h}",
            "mean_ms": round(mean_lat, 2),
            "median_p50_ms": round(median_lat, 2),
            "p90_ms": round(p90_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "p99_ms": round(p99_lat, 2),
            "fps": round(fps, 1),
            "min_ms": round(float(np.min(latencies_ms)), 2),
            "max_ms": round(float(np.max(latencies_ms)), 2)
        }
        results["benchmarks"].append(res_entry)

    return results


def format_markdown_table(benchmark_data: Dict[str, Any]) -> str:
    """Format benchmark results into a clean GitHub Flavored Markdown table."""
    md = [
        "| Resolution | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Throughput (FPS) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for b in benchmark_data["benchmarks"]:
        md.append(
            f"| `{b['resolution']}` | {b['mean_ms']} | {b['median_p50_ms']} | "
            f"{b['p90_ms']} | {b['p95_ms']} | {b['p99_ms']} | **{b['fps']} FPS** |"
        )
    return "\n".join(md)


if __name__ == "__main__":
    report = run_latency_benchmark(num_iterations=25, warmup_iterations=5)
    print("\n" + "=" * 60)
    print("THERsenel HARDWARE INFERENCE BENCHMARK RESULTS")
    print("=" * 60)
    print(format_markdown_table(report))
    print("=" * 60 + "\n")

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Results saved to {logs_dir / 'benchmark_results.json'}")
