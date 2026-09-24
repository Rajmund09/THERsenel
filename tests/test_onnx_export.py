import pytest
from pathlib import Path
from src.models.export_onnx import export_model_to_onnx

def test_export_model_to_onnx_missing_file():
    with pytest.raises(FileNotFoundError):
        export_model_to_onnx(Path("non_existent_weights_12345.pt"))
