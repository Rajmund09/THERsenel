"""
FLIR Dataset Preprocessing & YOLO Converter (src/data/preprocessing.py)
Parses FLIR COCO JSON annotations, converts bounding boxes to normalized YOLO format,
organizes train/val/test splits under data/processed/, and updates configs/dataset.yaml.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FLIR_DIR = PROJECT_ROOT / "data" / "raw" / "FLIR"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_FILE = PROJECT_ROOT / "configs" / "dataset.yaml"

# Target class mapping for Border Intrusion Detection
CLASS_MAPPING = {
    1: 0,   # person -> 0
    2: 1,   # bike / bicycle -> 1
    3: 2,   # car -> 2
    4: 3,   # motor -> 3
    6: 4,   # bus -> 4
    8: 5,   # truck -> 5
    74: 6,  # rider -> 6
}

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "rider"
}


def convert_coco_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [x_min, y_min, width, height] to normalized YOLO [x_center, y_center, w, h]."""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_width
    y_center = (y_min + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, w_norm)),
        max(0.0, min(1.0, h_norm))
    )


def resolve_image_path(split_dir: Path, file_name: str) -> Path | None:
    """Find the exact source image path across potential FLIR layout conventions."""
    candidates = [
        split_dir / file_name,
        split_dir / "data" / Path(file_name).name,
        split_dir / Path(file_name).name,
        split_dir / "data" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_coco_split(split_dir: Path, output_split: str, modality: str = "rgb"):
    """
    Process a single FLIR COCO split directory (e.g. images_rgb_train) into data/processed/.
    """
    coco_json_path = split_dir / "coco.json"

    if not coco_json_path.exists():
        print(f"[!] Skipping {split_dir.name}: missing coco.json")
        return

    print(f"[+] Processing {split_dir.name} ({modality}) -> {output_split}...")
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Build image id lookup
    images_info = {img["id"]: img for img in coco_data.get("images", [])}

    # Group annotations by image_id
    img_to_anns = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id in CLASS_MAPPING:
            img_to_anns.setdefault(img_id, []).append(ann)

    # Output directory layout
    out_img_dir = PROCESSED_DIR / output_split / "images"
    out_lbl_dir = PROCESSED_DIR / output_split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for img_id, img_meta in tqdm(images_info.items(), desc=f"Converting {split_dir.name}"):
        file_name = img_meta["file_name"]
        src_img_path = resolve_image_path(split_dir, file_name)

        if src_img_path is None:
            continue

        width = img_meta.get("width", 640)
        height = img_meta.get("height", 480)

        # Prefix filename with modality and split for uniqueness
        new_base_name = f"{modality}_{split_dir.name}_{Path(file_name).stem}"
        dst_img_path = out_img_dir / f"{new_base_name}{Path(file_name).suffix}"
        dst_label_path = out_lbl_dir / f"{new_base_name}.txt"

        # Copy image file
        shutil.copy2(src_img_path, dst_img_path)

        # Write YOLO label file
        anns = img_to_anns.get(img_id, [])
        with open(dst_label_path, "w") as lf:
            for ann in anns:
                yolo_cls = CLASS_MAPPING[ann["category_id"]]
                xc, yc, w_norm, h_norm = convert_coco_bbox_to_yolo(ann["bbox"], width, height)
                lf.write(f"{yolo_cls} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        processed_count += 1

    print(f"[?] Completed {output_split} ({modality}): {processed_count} images & labels written.")


def update_dataset_config():
    """Update configs/dataset.yaml with processed dataset paths and class labels."""
    config = {
        "path": str(PROCESSED_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": CLASS_NAMES
    }
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[+] Updated dataset configuration in {CONFIG_FILE}")


def main():
    print("==========================================================================")
    print("        FLIR DATASET PREPROCESSING & YOLO CONVERSION PIPELINE             ")
    print("==========================================================================")

    splits = [
        (RAW_FLIR_DIR / "images_rgb_train", "train", "rgb"),
        (RAW_FLIR_DIR / "images_thermal_train", "train", "thermal"),
        (RAW_FLIR_DIR / "images_rgb_val", "val", "rgb"),
        (RAW_FLIR_DIR / "images_thermal_val", "val", "thermal"),
        (RAW_FLIR_DIR / "video_rgb_test", "test", "rgb"),
        (RAW_FLIR_DIR / "video_thermal_test", "test", "thermal"),
    ]

    for split_path, output_split, modality in splits:
        if split_path.exists():
            process_coco_split(split_path, output_split, modality)

    update_dataset_config()
    print("\n[?] Preprocessing complete. Data ready for YOLOv8 training.")


if __name__ == "__main__":
    main()
