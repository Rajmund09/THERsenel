"""
M2 — Dataset exploration.

Run this AFTER you've placed the FLIR dataset (untouched) into data/raw/FLIR/.

What it does:
  - Walks data/raw/FLIR/ and counts RGB images, thermal images, and annotation files
  - Tries to detect the annotation format (YOLO .txt / COCO .json)
  - Reports image resolutions found
  - Saves a handful of side-by-side RGB/thermal sample plots to data/samples/

This script deliberately makes very few assumptions about FLIR's exact folder
layout, because that varies by which FLIR release you download. Read the printed
summary and adjust src/data/preprocessing.py accordingly — don't hardcode paths
here beyond what's found.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "FLIR"
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ANNOTATION_EXTS = {".txt", ".json", ".xml"}


def find_files(root: Path):
    images, annotations = [], []
    for p in root.rglob("*"):
        if p.is_file():
            suffix = p.suffix.lower()
            if suffix in IMAGE_EXTS:
                images.append(p)
            elif suffix in ANNOTATION_EXTS:
                annotations.append(p)
    return images, annotations


def guess_modality(path: Path) -> str:
    """Best-effort guess of RGB vs thermal from filename/folder conventions."""
    try:
        lowered = str(path.relative_to(RAW_DIR)).lower()
    except ValueError:
        lowered = str(path).lower()
    if "thermal" in lowered or "ir" in lowered.split("/") or "_ir" in lowered:
        return "thermal"
    if "rgb" in lowered or "visible" in lowered:
        return "rgb"
    return "unknown"


def image_resolution(path: Path):
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def main():
    if not RAW_DIR.exists():
        print(f"[!] {RAW_DIR} does not exist yet.")
        print("    Download the FLIR dataset and extract it there before running this script.")
        return

    images, annotations = find_files(RAW_DIR)
    print(f"Found {len(images)} image files and {len(annotations)} annotation files under {RAW_DIR}\n")

    if not images:
        print("[!] No images found. Check that the dataset actually extracted into data/raw/FLIR/.")
        return

    modality_counts = Counter(guess_modality(p) for p in images)
    print("Modality breakdown (best-effort filename guess — verify manually):")
    for modality, count in modality_counts.items():
        print(f"  {modality}: {count}")

    ext_counts = Counter(p.suffix.lower() for p in annotations)
    print("\nAnnotation file types found:")
    for ext, count in ext_counts.items():
        print(f"  {ext}: {count}")

    # Sample resolutions from up to 20 images
    print("\nSampling resolutions from up to 20 images...")
    resolutions = Counter()
    for p in images[:20]:
        res = image_resolution(p)
        if res:
            resolutions[res] += 1
    for res, count in resolutions.items():
        print(f"  {res}: {count} image(s) sampled")

    # Peek at one annotation file's structure
    json_anns = [p for p in annotations if p.suffix.lower() == ".json"]
    if json_anns:
        print(f"\nPeeking into {json_anns[0].name}:")
        with open(json_anns[0]) as f:
            data = json.load(f)
        if isinstance(data, dict):
            print("  Top-level keys:", list(data.keys())[:10])
        elif isinstance(data, list):
            print(f"  List with {len(data)} entries, first entry:", data[0] if data else None)

    # Save a few sample images (RGB + thermal side by side, if pairs can be guessed)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rgb_imgs = [p for p in images if guess_modality(p) == "rgb"][:3]
    thermal_imgs = [p for p in images if guess_modality(p) == "thermal"][:3]

    if rgb_imgs and thermal_imgs:
        n = min(len(rgb_imgs), len(thermal_imgs))
        fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
        if n == 1:
            axes = [axes]
        for i in range(n):
            for ax, path, label in zip(axes[i], [rgb_imgs[i], thermal_imgs[i]], ["RGB", "Thermal"]):
                img = Image.open(path)
                ax.imshow(img, cmap="gray" if label == "Thermal" else None)
                ax.set_title(f"{label}: {path.name}")
                ax.axis("off")
        out_path = SAMPLES_DIR / "rgb_thermal_preview.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        print(f"\nSaved preview grid to {out_path}")
    else:
        print("\n[!] Could not confidently pair RGB and thermal images by filename.")
        print("    Open a few files manually and check FLIR's actual naming convention,")
        print("    then update guess_modality() above accordingly.")

    print("\nDone. Use this summary to fill in configs/dataset.yaml and write src/data/preprocessing.py.")


if __name__ == "__main__":
    main()
