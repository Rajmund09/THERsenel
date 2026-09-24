import pytest
from src.evaluation.benchmark import format_markdown_table

def test_format_markdown_table():
    sample_data = {
        "benchmarks": [
            {
                "resolution": "640x640",
                "mean_ms": 28.5,
                "median_p50_ms": 27.2,
                "p90_ms": 31.0,
                "p95_ms": 33.2,
                "p99_ms": 35.8,
                "fps": 35.1
            }
        ]
    }
    table = format_markdown_table(sample_data)
    assert "| `640x640` |" in table
    assert "**35.1 FPS**" in table
    assert "Throughput (FPS)" in table
