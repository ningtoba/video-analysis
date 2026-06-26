"""
Video MLLM (Multimodal Large Language Model) module.

Wraps **VideoChat-Flash** (OpenGVLab, ICLR 2026) — a hierarchical compression
video MLLM that supports long-context video understanding with only 16
tokens per frame.  The 2B variant (``OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448``)
fits in ~5.4 GB BF16 on a 12 GB RTX 4070 with comfortable headroom.

Provides:
- ``describe_scene(frames)`` — richer scene descriptions than OpenCLIP labels
- ``summarize_video(video_path, num_frames=32)`` — global video summary
- ``answer(query, context_frames)`` — video-native Q&A without text context

Usage:
    mllm = VideoMLLM()
    if mllm.available:
        desc = mllm.describe_scene([frame1, frame2, frame3])

All methods gracefully return None when the model is unavailable
(dependencies missing, out of memory, etc.).
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class VideoMLLM:
    """Wrapper around VideoChat-Flash for video-native understanding.

    Lazy-loads the model on first use and provides GPU memory management
    (load / unload) compatible with the pipeline's sequential model loading
    pattern for 12 GB VRAM.
    """

    def __init__(self, model_name: str = "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._available = None  # None = unchecked, True/False after check

    # ------------------------------------------------------------------
    # Availability check & lazy loading
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Check whether the Video MLLM model can be loaded.

        Performs a soft import check first (no model download), then
        loads the model on first use.  Returns True only if both the
        dependencies and the model can be found.
        """
        if self._available is not None:
            return self._available
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            # VideoChat-Flash uses AutoModel with trust_remote_code
            self._available = True
        except ImportError as e:
            logger.warning(f"Video MLLM dependencies not available: {e}")
            self._available = False
        return self._available

    def load(self) -> bool:
        """Load the VideoChat-Flash model onto GPU.

        Returns True on success, False if dependencies are missing or
        the model cannot be found on disk / Hugging Face.
        """
        if self._model is not None:
            return True

        if not self.available:
            return False

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            logger.info(f"Loading Video MLLM model: {self.model_name}")
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._model.eval()

            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            logger.info("Video MLLM model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Video MLLM model: {e}")
            self._model = None
            self._processor = None
            self._available = False
            return False

    def unload(self):
        """Unload the model from GPU memory.

        Call this after using the Video MLLM to free ~5.4 GB VRAM for
        other pipeline stages.
        """
        if self._model is not None:
            import gc
            import torch

            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Video MLLM model unloaded from GPU")

    # ------------------------------------------------------------------
    # Video-native understanding methods
    # ------------------------------------------------------------------

    def describe_scene(self, frames: List[str], prompt: str = None) -> Optional[str]:
        """Describe a scene from its key frames using VideoChat-Flash.

        Args:
            frames: List of paths to frame images.
            prompt: Optional custom prompt.  Defaults to a rich scene
                description prompt.

        Returns:
            Natural language scene description, or None on failure.
        """
        if not frames:
            return None
        if not self.load():
            return None

        try:
            import torch

            prompt = prompt or (
                "Describe this video scene in detail. "
                "Include: setting (indoor/outdoor), visible objects, "
                "people present, their actions, on-screen text, and "
                "the overall mood or tone."
            )

            # Prepare inputs: VideoChat-Flash accepts image paths + text
            inputs = self._processor(
                images=frames,
                text=prompt,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=256)

            description = self._processor.decode(output[0], skip_special_tokens=True)
            return description.strip()
        except Exception as e:
            logger.warning(f"Video MLLM scene description failed: {e}")
            return None

    def summarize_video(
        self,
        video_path: str,
        num_frames: int = 32,
        prompt: str = None,
    ) -> Optional[str]:
        """Generate a comprehensive summary of the entire video.

        The model internally samples frames from the video file using
        VideoChat-Flash's native video handling, which uses hierarchical
        compression to represent long videos with few tokens.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to sample (VideoChat-Flash handles
                the sampling internally).
            prompt: Optional custom summary prompt.

        Returns:
            Text summary of the video content, or None on failure.
        """
        video = Path(video_path)
        if not video.exists():
            logger.warning(f"Video file not found: {video_path}")
            return None
        if not self.load():
            return None

        try:
            import torch

            prompt = prompt or (
                "Provide a comprehensive summary of this video. "
                "Cover: the setting, key events, people present, "
                "actions that occur, and any important visual or "
                "textual information visible on screen."
            )

            # VideoChat-Flash can accept a video path directly
            inputs = self._processor(
                videos=[str(video)],
                text=prompt,
                return_tensors="pt",
                num_frames=num_frames,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=512)

            summary = self._processor.decode(output[0], skip_special_tokens=True)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Video MLLM summarization failed: {e}")
            return None

    def answer(
        self,
        query: str,
        frames: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Optional[str]:
        """Answer a question about video content using VideoChat-Flash.

        Args:
            query: Natural language question.
            frames: Optional list of frame image paths for visual context.
            video_path: Optional video file path (VideoChat-Flash handles
                its own frame sampling when a video path is given).

        Returns:
            Answer text, or None on failure.
        """
        if not frames and not video_path:
            logger.warning("No frames or video path provided for VLM QA")
            return None
        if not self.load():
            return None

        try:
            import torch

            if video_path:
                inputs = self._processor(
                    videos=[video_path],
                    text=query,
                    return_tensors="pt",
                    num_frames=16,
                )
            elif frames:
                inputs = self._processor(
                    images=frames,
                    text=query,
                    return_tensors="pt",
                )
            else:
                return None

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=512)

            answer = self._processor.decode(output[0], skip_special_tokens=True)
            return answer.strip()
        except Exception as e:
            logger.warning(f"Video MLLM QA failed: {e}")
            return None
