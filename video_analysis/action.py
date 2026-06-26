"""
Action recognition module for video analysis pipeline.

Wraps X-CLIP (Microsoft) for zero-shot open-vocabulary action recognition
on extracted video frames.  Uses ``transformers`` (already a project
dependency) and requires no additional packages.

Model: ``microsoft/xclip-base-patch16-zero-shot`` (200M params, Apache 2.0)
VRAM:  ~4 GB when loaded, 0 GB when unloaded.

Usage::

    from video_analysis.action import ActionRecognizer

    recognizer = ActionRecognizer()
    results = recognizer.classify(frames, candidate_actions)
    recognizer.unload()  # free VRAM
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from video_analysis.models import FrameInfo

logger = logging.getLogger(__name__)

# Default action categories covering common video scenarios.
# These are open-vocabulary — X-CLIP scores each frame against each label.
DEFAULT_ACTION_CATEGORIES = [
    "a person walking",
    "a person running",
    "a person sitting",
    "a person standing",
    "people talking",
    "a person speaking",
    "a person cooking",
    "a person eating",
    "a person typing",
    "a person reading",
    "a person dancing",
    "a person exercising",
    "a person driving",
    "a person riding a bicycle",
    "a person playing an instrument",
    "a person using a phone",
    "a person shaking hands",
    "a person clapping",
    "a person jumping",
    "a person fighting",
    "a person throwing",
    "a person lifting",
    "a person sleeping",
    "a person writing",
    "a person pointing",
    "no person visible",
]


class ActionRecognizer:
    """Zero-shot open-vocabulary action recognizer using X-CLIP.

    Loads the model lazily on first call to :meth:`classify`.  Call
    :meth:`unload` after use to free GPU VRAM.

    Args:
        model_name: HuggingFace model ID for X-CLIP.
        device: Torch device (``"cuda"`` or ``"cpu"``).
        categories: List of action descriptions to score.
    """

    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        device: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = categories or DEFAULT_ACTION_CATEGORIES
        self._model = None
        self._processor = None

    def _load(self):
        """Lazily load the X-CLIP model."""
        if self._model is not None:
            return
        try:
            from transformers import XCLIPProcessor, XCLIPModel
            import torch

            logger.info(
                f"Loading X-CLIP action recognizer: {self.model_name} "
                f"(categories: {len(self.categories)})"
            )
            self._processor = XCLIPProcessor.from_pretrained(self.model_name)
            self._model = XCLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            logger.info("X-CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load X-CLIP model: {e}")
            raise

    def classify(
        self, frames: List[FrameInfo]
    ) -> List[Tuple[FrameInfo, Optional[str], Optional[float]]]:
        """Classify actions in a list of frames.

        For each frame, returns the top-scoring action category and its
        confidence.  Frames that fail to load are returned with ``None``
        action.

        Args:
            frames: List of FrameInfo objects (must have valid ``filepath``).

        Returns:
            List of ``(frame, best_action, best_confidence)`` tuples in the
            same order as the input.
        """
        if not frames:
            return []

        self._load()

        results: List[Tuple[FrameInfo, Optional[str], Optional[float]]] = []

        # Process frames in batch for GPU efficiency
        batch_size = 16
        for batch_start in range(0, len(frames), batch_size):
            batch = frames[batch_start : batch_start + batch_size]
            batch_images = []
            batch_indices = []

            for i, frame in enumerate(batch):
                try:
                    img = Image.open(frame.filepath).convert("RGB")
                    batch_images.append(img)
                    batch_indices.append(i)
                except Exception:
                    results.append((frame, None, None))

            if not batch_images:
                continue

            try:
                # X-CLIP: encode text once, encode images per batch
                text_inputs = self._processor(
                    text=self.categories,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                image_inputs = self._processor(
                    images=batch_images,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    text_features = self._model.get_text_features(**text_inputs)
                    image_features = self._model.get_image_features(**image_inputs)

                # Normalize and compute similarity
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

                # (batch, num_categories)
                similarity = image_features @ text_features.T

                best_scores, best_indices = similarity.max(dim=-1)

                for j, local_idx in enumerate(batch_indices):
                    frame = batch[local_idx]
                    best_idx = best_indices[j].item()
                    best_conf = best_scores[j].item()
                    best_action = self.categories[best_idx]
                    results.append((frame, best_action, best_conf))

            except Exception as e:
                logger.debug(f"X-CLIP batch classification error: {e}")
                for frame in batch:
                    if frame not in [r[0] for r in results]:
                        results.append((frame, None, None))

        return results

    def classify_scenes(
        self,
        scenes: "List[SceneInfo]",
        frames_per_scene: int = 4,
    ) -> dict:
        """Classify actions per scene, aggregating per-scene results.

        Args:
            scenes: List of SceneInfo objects with key_frames.
            frames_per_scene: Max frames to sample per scene.

        Returns:
            Dict ``{scene_id: [(action, confidence), ...]}``.
        """
        from video_analysis.models import SceneInfo

        self._load()
        scene_results: dict = {}

        for scene in scenes:
            if not scene.key_frames:
                continue

            sampled = scene.key_frames[:frames_per_scene]
            classifications = self.classify(sampled)

            # Aggregate: de-duplicate by action, keep max confidence
            action_map: dict = {}
            for frame, action, conf in classifications:
                if action is None:
                    continue
                if action not in action_map or (conf and conf > action_map[action]):
                    action_map[action] = conf or 0.0

            scene_results[scene.scene_id] = sorted(
                action_map.items(), key=lambda x: x[1], reverse=True
            )

        return scene_results

    def unload(self):
        """Free GPU VRAM by releasing the model reference."""
        self._model = None
        self._processor = None
        import gc

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("X-CLIP model unloaded from GPU")
