"""
PyTorch Multimodal Dataset (src/data/dataset.py)
Loads synchronized RGB and Thermal image pairs with associated YOLO format bounding box annotations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RGBThermalDataset(Dataset):
    """
    PyTorch Dataset for loading synchronized RGB visual and Thermal infrared image pairs.
    """

    def __init__(
        self,
        img_dir: Path,
        label_dir: Path,
        img_size: Tuple[int, int] = (640, 640),
        transform=None
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transform = transform

        self.rgb_files = sorted(list(self.img_dir.glob("rgb_*.jpg")) + list(self.img_dir.glob("rgb_*.png")))
        self.thermal_files = sorted(list(self.img_dir.glob("thermal_*.jpg")) + list(self.img_dir.glob("thermal_*.png")))

        # If files are not explicitly prefixed, group by matching stem
        if not self.rgb_files:
            all_imgs = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
            self.rgb_files = [p for p in all_imgs if "rgb" in p.name.lower()]
            self.thermal_files = [p for p in all_imgs if "thermal" in p.name.lower() or "ir" in p.name.lower()]

        self.samples = self._pair_images()

    def _pair_images(self) -> List[Dict[str, Path]]:
        pairs = []
        thermal_dict = {p.stem.replace("thermal_", ""): p for p in self.thermal_files}

        for rgb_path in self.rgb_files:
            key = rgb_path.stem.replace("rgb_", "")
            if key in thermal_dict:
                lbl_path = self.label_dir / f"{rgb_path.stem}.txt"
                if not lbl_path.exists():
                    lbl_path = self.label_dir / f"{key}.txt"
                pairs.append({
                    "rgb": rgb_path,
                    "thermal": thermal_dict[key],
                    "label": lbl_path if lbl_path.exists() else None
                })
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def _load_labels(self, label_path: Optional[Path]) -> torch.Tensor:
        if label_path is None or not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = float(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    labels.append([cls_id, xc, yc, w, h])

        return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        rgb_img = cv2.imread(str(sample["rgb"]))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, self.img_size)

        thermal_img = cv2.imread(str(sample["thermal"]))
        if thermal_img is None:
            thermal_img = np.zeros_like(rgb_img)
        else:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
            thermal_img = cv2.resize(thermal_img, self.img_size)

        labels = self._load_labels(sample["label"])

        # Normalize to [0, 1] tensor
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        thermal_tensor = torch.from_numpy(thermal_img).permute(2, 0, 1).float() / 255.0

        return rgb_tensor, thermal_tensor, labels
