"""
Multimodal Data Augmentation Pipeline (src/data/augmentation.py)
Applies synchronized spatial and intensity data augmentations to paired RGB and Thermal images.
"""

from typing import Dict, Tuple
import albumentations as A
import numpy as np


class MultimodalAugmenter:
    """
    Applies identical spatial transformations (Crop, Flip, Scale) to RGB & Thermal image pairs simultaneously.
    """

    def __init__(self, img_size: Tuple[int, int] = (640, 640), is_train: bool = True):
        if is_train:
            self.spatial_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.8, 1.0), p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
                ],
                additional_targets={"thermal": "image"}
            )
            self.rgb_color_transform = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(p=0.3)
            ])
            self.thermal_intensity_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
            ])
        else:
            self.spatial_transform = A.Compose(
                [A.Resize(height=img_size[0], width=img_size[1])],
                additional_targets={"thermal": "image"}
            )
            self.rgb_color_transform = None
            self.thermal_intensity_transform = None

    def __call__(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> Dict[str, np.ndarray]:
        res = self.spatial_transform(image=rgb_image, thermal=thermal_image)
        rgb_aug = res["image"]
        thermal_aug = res["thermal"]

        if self.rgb_color_transform is not None:
            rgb_aug = self.rgb_color_transform(image=rgb_aug)["image"]

        if self.thermal_intensity_transform is not None:
            thermal_aug = self.thermal_intensity_transform(image=thermal_aug)["image"]

        return {"rgb": rgb_aug, "thermal": thermal_aug}
