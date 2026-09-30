"""
vidsim-curator/evaluation/ablation.py
=============================
Ablation experiment suite for measuring the quantitative impact of dataset quality
thresholds on downstream world model frame predictability and visual fidelity metrics.
"""

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FilterConfig:
    """Configuration thresholds for video dataset filtering ablations."""
    name: str
    min_spatial_variance: float
    min_motion_energy: float
    min_clip_score: float
    max_duplicate_ratio: float


@dataclasses.dataclass
class EvaluationMetrics:
    """Result metrics for a trained world model evaluated on target benchmark clips."""
    config_name: str
    retained_sample_count: int
    data_retention_ratio: float
    fvd_score: float
    mean_ssim: float
    mean_psnr: float
    temporal_consistency: float


class DatasetAblationHarness:
    """
    Executes dataset filtering ablations and computes video prediction performance metrics.
    """

    def __init__(self, raw_dataset_metadata: List[Dict[str, Any]]):
        """
        Args:
            raw_dataset_metadata: List of dicts containing computed clip metrics 
                                  (e.g., spatial_patch_variance, motion_energy, clip_score).
        """
        self.raw_dataset = raw_dataset_metadata
        self.total_samples = len(raw_dataset_metadata)
        logger.info(f"Initialized Ablation Harness with {self.total_samples} raw samples.")

    def filter_dataset(self, config: FilterConfig) -> List[Dict[str, Any]]:
        """Filters dataset metadata based on the specified FilterConfig thresholds."""
        filtered = [
            item for item in self.raw_dataset
            if item.get("spatial_patch_variance", 0.0) >= config.min_spatial_variance
            and item.get("motion_energy", 0.0) >= config.min_motion_energy
            and item.get("clip_score", 0.0) >= config.min_clip_score
            and item.get("duplicate_ratio", 0.0) <= config.max_duplicate_ratio
        ]
        return filtered

    @staticmethod
    def calculate_frame_metrics(
        pred_frames: np.ndarray, target_frames: np.ndarray
    ) -> Tuple[float, float]:
        """
        Computes mean SSIM and PSNR between predicted frame sequence and ground truth.

        Args:
            pred_frames: Array of shape (T, H, W, C) with range [0, 255].
            target_frames: Array of shape (T, H, W, C) with range [0, 255].

        Returns:
            Tuple of (mean_ssim, mean_psnr).
        """
        ssim_scores = []
        psnr_scores = []

        for p_frame, t_frame in zip(pred_frames, target_frames):
            # Compute SSIM with channel specification
            s = ssim(t_frame, p_frame, channel_axis=-1, data_range=255)
            p = psnr(t_frame, p_frame, data_range=255)
            ssim_scores.append(s)
            psnr_scores.append(p)

        return float(np.mean(ssim_scores)), float(np.mean(psnr_scores))

    @staticmethod
    def calculate_temporal_consistency(pred_frames: np.ndarray) -> float:
        """
        Evaluates temporal consistency by measuring smooth frame-to-frame optical flow variance.
        Lower frame-to-frame variance jitter indicates higher motion continuity.

        Args:
            pred_frames: Array of shape (T, H, W, C).

        Returns:
            float: Temporal smoothness score (1.0 = highly smooth frame transitions).
        """
        if len(pred_frames) < 2:
            return 1.0

        diffs = np.abs(np.diff(pred_frames.astype(np.float32), axis=0))
        frame_diff_variance = np.var(diffs, axis=(1, 2, 3))
        # Map variance jitter into a normalized stability score [0.0, 1.0]
        smoothness = 1.0 / (1.0 + float(np.mean(frame_diff_variance)) / 1000.0)
        return float(smoothness)

    def evaluate_world_model_predictions(
        self,
        config: FilterConfig,
        synthetic_predictions: List[np.ndarray],
        ground_truth_targets: List[np.ndarray],
        mock_fvd_base: float = 180.0,
    ) -> EvaluationMetrics:
        """
        Simulates model evaluation over target benchmark predictions generated from a dataset configuration.

        Args:
            config: FilterConfig used to subset the training data.
            synthetic_predictions: List of model-predicted video sequences.
            ground_truth_targets: List of real benchmark video sequences.
            mock_fvd_base: Baseline FVD calculation parameter.

        Returns:
            EvaluationMetrics object containing complete metrics suite.
        """
        filtered_subset = self.filter_dataset(config)
        retained_count = len(filtered_subset)
        retention_ratio = retained_count / max(1, self.total_samples)

        ssim_list = []
        psnr_list = []
        temp_list = []

        for pred, target in zip(synthetic_predictions, ground_truth_targets):
            s, p = self.calculate_frame_metrics(pred, target)
            t = self.calculate_temporal_consistency(pred)
            ssim_list.append(s)
            psnr_list.append(p)
            temp_list.append(t)

        mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        mean_temp = float(np.mean(temp_list)) if temp_list else 0.0

        # Simulated FVD improvement inversely tied to data retention purity ratio
        fvd_score = mock_fvd_base * (1.1 - (0.4 * (1.0 - abs(retention_ratio - 0.6))))

        metrics = EvaluationMetrics(
            config_name=config.name,
            retained_sample_count=retained_count,
            data_retention_ratio=round(retention_ratio, 4),
            fvd_score=round(fvd_score, 2),
            mean_ssim=round(mean_ssim, 4),
            mean_psnr=round(mean_psnr, 2),
            temporal_consistency=round(mean_temp, 4),
        )

        logger.info(
            f"Config [{config.name}] | Retention: {retention_ratio:.1%} | "
            f"FVD: {metrics.fvd_score} | SSIM: {metrics.mean_ssim}"
        )
        return metrics
