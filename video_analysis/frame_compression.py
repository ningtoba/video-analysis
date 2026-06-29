"""
LongVU-Style DINOv2 Frame Compression — adaptive temporal redundancy removal.

Inspired by *LongVU: Spatiotemporal Adaptive Compression for Long Video
Understanding* (ICML 2025), this module reduces redundant frames across a
video's timeline by computing per-frame DINOv2 features and dropping frames
whose feature similarity to the last kept frame exceeds a configurable
threshold.

The core insight: adjacent video frames are often nearly identical.
Rather than sampling uniformly or only at scene boundaries, we use a
lightweight ViT-based self-supervised encoder (DINOv2) to measure
*perceptual* similarity, keeping only the most informative frames
for downstream processing (CLIP, YOLO, OCR, etc.).

Usage::

    from video_analysis.frame_compression import DINOv2FrameCompressor

    compressor = DINOv2FrameCompressor()
    kept_indices = compressor.compress(frames, threshold=0.85)

Dependencies::
    - transformers >= 4.45.0 (DINOv2 via HuggingFace)
    - torch (for GPU inference)

Zero additional VRAM footprint when integrated into the pipeline's
sequential model loading pattern: the DINOv2 model is loaded, used for
compression, then unloaded before other pipeline stages run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# dinov2-small output dimension (for the zero-feature fallback array)
_DINOV2_SMALL_DIM = 384


class DINOv2FrameCompressor:
    """Adaptive frame compression using DINOv2 perceptual similarity.

    Loads a compact DINOv2 model (``dinov2-small`` — 21M params, ~85 MB
    VRAM), computes feature vectors for each frame, and drops frames that
    are too similar (cosine similarity above *threshold*) to the last kept
    frame.

    The model is loaded lazily on the first ``compress()`` call and can be
    explicitly unloaded via ``unload()`` to free VRAM.

    Args:
        model_name: HuggingFace model name for DINOv2 variant.
            ``facebook/dinov2-small`` (21M) is the default — compact and
            fast.  ``facebook/dinov2-base`` (86M) for higher accuracy.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        threshold: Cosine similarity threshold [0,1].  Frames with
            similarity above this to the last kept frame are dropped.
            Lower = more aggressive compression.
        batch_size: Frames to process per batch (GPU memory trade-off).

    Attributes:
        available: True if DINOv2/transformers can be imported.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str = "cuda",
        threshold: float = 0.88,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._available = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if DINOv2 can be imported and the model is loadable."""
        if self._available is None:
            try:
                from transformers import AutoImageProcessor, AutoModel  # noqa: F401

                self._available = True
            except Exception:
                self._available = False
        return self._available

    def compress(
        self,
        frames: List[Path | str],
        threshold: Optional[float] = None,
    ) -> List[int]:
        """Return indices of frames to *keep* after redundancy removal.

        Args:
            frames: List of paths to frame images.
            threshold: Override the instance threshold for this call.

        Returns:
            Sorted list of indices into *frames* that are perceptually
            distinct enough to keep.

        Raises:
            RuntimeError: If DINOv2 is not available.
        """
        if not self.available:
            raise RuntimeError(
                "DINOv2 is not available. Install transformers >= 4.45.0"
            )

        if len(frames) <= 1:
            return list(range(len(frames)))

        threshold = threshold if threshold is not None else self.threshold
        self._load()

        # Compute features in batches
        features = self._compute_features(frames)

        # Greedy redundancy removal
        kept: List[int] = [0]
        last_feat = self._normalise(features[0])

        for i in range(1, len(features)):
            feat = self._normalise(features[i])
            sim = float(np.dot(last_feat, feat))
            if sim < threshold:
                kept.append(i)
                last_feat = feat

        logger.info(
            "DINOv2 compression: %d → %d frames (%.0f%% reduction, threshold=%.2f)",
            len(frames),
            len(kept),
            (1 - len(kept) / len(frames)) * 100,
            threshold,
        )
        return kept

    def unload(self):
        """Unload the DINOv2 model from GPU memory."""
        if self._model is not None:
            import torch

            del self._model
            del self._processor
            self._model = None
            self._processor = None
            torch.cuda.empty_cache()
            logger.info("DINOv2 model unloaded from GPU")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        """Lazily load DINOv2 model and processor."""
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel

            logger.info("Loading DINOv2 model: %s on %s", self.model_name, self.device)
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self.device)
            self._model.eval()
            logger.info("DINOv2 model loaded")
        except Exception as e:
            logger.error("Failed to load DINOv2 model: %s", e)
            raise RuntimeError(f"DINOv2 loading failed: {e}") from e

    def _compute_features(self, frames: List[Path | str]) -> np.ndarray:
        """Compute DINOv2 [CLS] token features for all frames.

        Processes frames in batches and returns a (N, D) numpy array.
        """
        import torch

        all_features: List[np.ndarray] = []
        for batch_start in range(0, len(frames), self.batch_size):
            batch_paths = frames[batch_start : batch_start + self.batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning("Failed to open frame %s: %s", p, e)

            if not images:
                continue

            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            # [CLS] token is the first token for DINOv2 (pooler_output)
            cls_tokens = outputs.pooler_output  # (B, D)
            all_features.append(cls_tokens.cpu().numpy())

        if not all_features:
            return np.zeros((0, _DINOV2_SMALL_DIM))

        return np.concatenate(all_features, axis=0)

    @staticmethod
    def _normalise(feature: np.ndarray) -> np.ndarray:
        """L2-normalise a single feature vector."""
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
