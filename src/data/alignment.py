"""
RGB-Thermal Image Alignment & Registration (src/data/alignment.py)
Computes homography warp matrices to align RGB and Thermal camera feeds for pixel-level fusion.
"""

from typing import Tuple
import cv2
import numpy as np


class RGBIRAligner:
    """
    Feature matching (ORB/SIFT) & Homography alignment module for registering Thermal onto RGB coordinate system.
    """

    def __init__(self, nfeatures: int = 1000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align_images(self, img_rgb: np.ndarray, img_thermal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warps img_thermal to match the spatial perspective of img_rgb.
        Returns: (img_rgb, aligned_thermal)
        """
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if len(img_rgb.shape) == 3 else img_rgb
        gray_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_RGB2GRAY) if len(img_thermal.shape) == 3 else img_thermal

        kp1, des1 = self.orb.detectAndCompute(gray_rgb, None)
        kp2, des2 = self.orb.detectAndCompute(gray_thermal, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            # Fallback to direct resize if insufficient feature points
            aligned_thermal = cv2.resize(img_thermal, (img_rgb.shape[1], img_rgb.shape[0]))
            return img_rgb, aligned_thermal

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        pts_rgb = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_thermal = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_thermal, pts_rgb, cv2.RANSAC, 5.0)

        if H is None:
            aligned_thermal = cv2.resize(img_thermal, (img_rgb.shape[1], img_rgb.shape[0]))
            return img_rgb, aligned_thermal

        height, width = img_rgb.shape[:2]
        aligned_thermal = cv2.warpPerspective(img_thermal, H, (width, height))

        return img_rgb, aligned_thermal
