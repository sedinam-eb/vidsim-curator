"""
vidsim-curator/pipelines/spatial.py
===========================
Spatial and semantic quality scoring pipeline for video frame sequences using
DINOv2 patch representation variance and OpenCLIP text-image embedding similarity.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import open_clip

logger = logging.getLogger(__name__)


class SpatialQualityScorer:
    """
    Evaluates spatial clarity, visual richness, and semantic alignment for video frames.
    
    Attributes:
        device (torch.device): Device to execute model inference on.
        dinov2_processor: Feature processor for DINOv2 vision transformer.
        dinov2_model: DINOv2 backbone model for spatial representation extraction.
        clip_model: OpenCLIP vision-language model.
        clip_preprocess: OpenCLIP preprocessing transform.
        clip_tokenizer: OpenCLIP text tokenizer.
    """

    def __init__(
        self,
        dinov2_model_name: str = "facebook/dinov2-small",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Initializing SpatialQualityScorer on device: {self.device}")

        # Initialize DINOv2 for spatial variance / visual density analysis
        self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        self.dinov2_model = (
            AutoModel.from_pretrained(dinov2_model_name).to(self.device).eval()
        )

        # Initialize OpenCLIP for semantic visual-text score verification
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=self.device
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    @torch.no_grad()
    def compute_patch_variance(
        self, frames: Union[torch.Tensor, List[Image.Image]]
    ) -> float:
        """
        Computes the spatial variance across DINOv2 patch embeddings.
        High spatial variance correlates with rich physical detail and visual texture,
        whereas low variance indicates monochrome, out-of-focus, or blank frames.

        Args:
            frames: Batch of frames as PIL Images or PyTorch tensor (N, C, H, W).

        Returns:
            float: Normalized spatial patch variance score across the frame sequence.
        """
        if isinstance(frames, torch.Tensor):
            if frames.ndim == 3:  # Single frame (C, H, W)
                frames = frames.unsqueeze(0)
            # Convert tensor batch to PIL Images for processor compatibility
            pil_frames = [
                Image.fromarray(
                    f.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                for f in frames
            ]
        else:
            pil_frames = frames

        inputs = self.dinov2_processor(images=pil_frames, return_tensors="pt").to(
            self.device
        )
        outputs = self.dinov2_model(**inputs)
        
        # Extract patch-level embeddings excluding the [CLS] token (last_hidden_state: [B, N_patches, D])
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        
        # Calculate cross-patch variance per frame and take batch mean
        patch_var = torch.var(patch_embeddings, dim=1).mean().item()
        return float(patch_var)

    @torch.no_grad()
    def compute_clip_alignment(
        self, frames: List[Image.Image], prompt: str
    ) -> float:
        """
        Calculates cosine similarity between sequence frames and a descriptive text prompt.

        Args:
            frames: List of PIL Image frames extracted from a video segment.
            prompt: Text prompt describing expected physical/visual content.

        Returns:
            float: Average cosine similarity score across frames (range: -1.0 to 1.0).
        """
        if not frames:
            return 0.0

        # Preprocess images and tokenize text
        image_tensors = (
            torch.stack([self.clip_preprocess(f) for f in frames]).to(self.device)
        )
        text_tokens = self.clip_tokenizer([prompt]).to(self.device)

        # Encode image and text features
        image_features = self.clip_model.encode_image(image_tensors)
        text_features = self.clip_model.encode_text(text_tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute mean similarity across all sampled sequence frames
        similarity = (image_features @ text_features.T).squeeze(-1).mean().item()
        return float(similarity)

    def evaluate_sequence(
        self,
        frames: List[Image.Image],
        prompt: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Runs comprehensive spatial quality analysis on a video frame sequence.

        Args:
            frames: List of sequence frames as PIL Images.
            prompt: Optional text prompt for semantic alignment check.

        Returns:
            Dict containing patch variance, clip score, and combined quality flag.
        """
        patch_variance = self.compute_patch_variance(frames)
        clip_score = (
            self.compute_clip_alignment(frames, prompt) if prompt else 0.0
        )

        return {
            "spatial_patch_variance": round(patch_variance, 4),
            "clip_alignment_score": round(clip_score, 4),
            "is_high_quality": patch_variance >= 0.45 and (clip_score >= 0.22 if prompt else True),
        }
