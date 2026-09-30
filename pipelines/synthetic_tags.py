"""
vidsim-curator/pipelines/synthetic_tags.py
==================================
Automated action dynamics and camera motion classification (Panning, Zoom, Static, Complex)
for multimodal world model dataset tagging.
"""

from enum import Enum
from typing import Dict, List, Union

import numpy as np
import torch


class CameraMotionClass(str, Enum):
    STATIC = "static"
    PANNING = "panning"
    ZOOM = "zoom"
    COMPLEX_DYNAMIC = "complex_dynamic"


class SyntheticTaggingPipeline:
    """
    Analyzes optical motion vectors across spatial quadrants to categorize camera and scene motion.
    """

    def __init__(
        self,
        static_threshold: float = 2.5,
        dynamic_threshold: float = 18.0,
    ):
        self.static_threshold = static_threshold
        self.dynamic_threshold = dynamic_threshold

    def classify_camera_motion(
        self, frames: torch.Tensor
    ) -> Dict[str, Union[str, float]]:
        """
        Estimates camera trajectory by comparing frame quarter diffs (TL, TR, BL, BR).

        Args:
            frames: PyTorch tensor of shape (T, H, W, C).

        Returns:
            Dict containing motion class label and spatial motion symmetry scores.
        """
        if frames.shape[0] < 2:
            return {
                "motion_class": CameraMotionClass.STATIC.value,
                "quadrant_variance": 0.0,
            }

        # Convert to float and compute spatial frame differences
        diffs = torch.diff(frames.float(), dim=0).abs()  # (T-1, H, W, C)
        h, w = diffs.shape[1], diffs.shape[2]
        half_h, half_w = h // 2, w // 2

        # Extract 4 spatial quadrants across time
        tl = diffs[:, :half_h, :half_w, :].mean().item()
        tr = diffs[:, :half_h, half_w:, :].mean().item()
        bl = diffs[:, half_h:, :half_w, :].mean().item()
        br = diffs[:, half_h:, half_w:, :].mean().item()

        quadrants = [tl, tr, bl, br]
        mean_motion = float(np.mean(quadrants))
        quadrant_variance = float(np.var(quadrants))

        # Classification heuristics
        if mean_motion < self.static_threshold:
            motion_class = CameraMotionClass.STATIC
        elif mean_motion > self.dynamic_threshold and quadrant_variance > 10.0:
            motion_class = CameraMotionClass.COMPLEX_DYNAMIC
        elif abs((tl + bl) - (tr + br)) > 5.0:  # Horizontal directional imbalance
            motion_class = CameraMotionClass.PANNING
        else:
            motion_class = CameraMotionClass.ZOOM

        return {
            "motion_class": motion_class.value,
            "mean_spatial_motion": round(mean_motion, 4),
            "quadrant_variance": round(quadrant_variance, 4),
        }
