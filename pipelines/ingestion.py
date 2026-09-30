"""
vidsim-curator/pipelines/ingestion.py
==============================
High-throughput video decoding, frame extraction, downsampling, and temporal 
motion dynamics processing using Decord and PyTorch.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Configure Decord to return PyTorch tensors directly
decord.bridge.set_bridge("torch")


class VideoIngestionPipeline:
    """
    Extracts frame sequences and computes temporal motion metrics without loading entire
    videos into RAM.

    Attributes:
        target_fps (int): Downsampled frame rate for sequence extraction.
        frame_size (Tuple[int, int]): Output (width, height) resolution.
        device (torch.device): Compute device for motion calculation.
    """

    def __init__(
        self,
        target_fps: int = 4,
        frame_size: Tuple[int, int] = (224, 224),
        device: Optional[str] = None,
    ):
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def compute_optical_motion_energy(self, frames: torch.Tensor) -> float:
        """
        Calculates temporal frame-to-frame motion energy across time steps.
        Higher values correspond to dynamic physical movement or active camera motion.

        Args:
            frames: PyTorch tensor of shape (T, H, W, C) in range [0, 255].

        Returns:
            float: Mean temporal difference score across sequence frames.
        """
        if frames.shape[0] < 2:
            return 0.0

        # Calculate absolute difference between consecutive frames along time dimension (T)
        frame_diffs = torch.diff(frames.float(), dim=0).abs().mean(dim=(1, 2, 3))
        motion_energy = frame_diffs.mean().item()
        return float(motion_energy)

    def extract_and_process(
        self, video_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Loads video file, samples keyframes at target_fps, and computes frame metrics.

        Args:
            video_path: Path to the input video file (.mp4, .mkv, .avi).

        Returns:
            Dict containing sampled PIL images, raw frame tensor, and motion energy score.
        """
        video_path_str = str(video_path)
        try:
            vr = decord.VideoReader(
                video_path_str,
                width=self.frame_size[0],
                height=self.frame_size[1],
            )
        except Exception as e:
            logger.error(f"Failed to decode video at {video_path_str}: {e}")
            raise RuntimeError(f"Video decoding failed for {video_path_str}") from e

        native_fps = int(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30
        step = max(1, native_fps // self.target_fps)
        frame_indices = list(range(0, len(vr), step))

        # Batch read frames into GPU/CPU memory: Tensor shape (T, H, W, C)
        raw_frames = vr.get_batch(frame_indices)
        motion_energy = self.compute_optical_motion_energy(raw_frames)

        # Convert tensor frames to PIL Images for downstream transformers/CLIP
        pil_frames = [
            Image.fromarray(f.cpu().numpy().astype(np.uint8))
            for f in raw_frames
        ]

        return {
            "video_path": video_path_str,
            "total_raw_frames": len(vr),
            "sampled_frame_count": len(frame_indices),
            "native_fps": native_fps,
            "motion_energy": round(motion_energy, 4),
            "pil_frames": pil_frames,
            "raw_frame_tensor": raw_frames,
        }
