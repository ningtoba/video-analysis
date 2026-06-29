"""
Video MLLM (Multimodal Large Language Model) module.

|Supports four backends:
|- **VideoChat-Flash** (OpenGVLab, ICLR 2026) — a hierarchical compression
|  video MLLM that supports long-context video understanding with only 16
|  tokens per frame.  The 2B variant (``OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448``)
|  fits in ~5.4 GB BF16 on a 12 GB RTX 4070 with comfortable headroom.
|- **SmolVLM2** (HuggingFaceTB) — a family of compact vision-language models
|  (2.2B, 500M, 256M) using standard ``AutoModelForImageTextToText`` from
|  transformers (no trust_remote_code).  Uses ``decord`` for video decoding.
|- **Qwen3-VL-30B-A3B** (Qwen Team, Apache 2.0) — Mixture-of-Experts VLM with
|  30B total / 3B active params, FP8 quantization, 128K context, hybrid
|  thinking/non-thinking mode.  Deployed via vLLM server (recommended),
|  vLLM offline inference, or transformers fallback.
|- **InternVideo3-8B** (OpenGVLab, June 2026, arXiv:2606.12195) — the
|  strongest open-weight video MLLM as of mid-2026, with Multimodal
|  Contextual Reasoning (MCR) for iterative evidence accumulation and
|  M^2LA KV-cache compression (1.84× faster decode). Best Video-MME
|  score (73.8) among open-weight 8B-class models.  Fits ~10GB at FP8
|  or ~6GB at INT4 on 12GB RTX 4070.

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
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

logger = logging.getLogger(__name__)

# SmolVLM2 model paths
SMOLVLM2_MODEL_PATHS = {
    "2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "500M": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "256M": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
}

BackendType = Literal["auto", "videochat_flash", "smolvlm2", "qwen3_vl", "internvideo3"]
ModelSizeType = Literal["2.2B", "500M", "256M"]


class VideoMLLM:
    """Wrapper around VideoChat-Flash or SmolVLM2 for video-native understanding.

    Lazy-loads the model on first use and provides GPU memory management
    (load / unload) compatible with the pipeline's sequential model loading
    pattern for 12 GB VRAM.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448",
        backend: BackendType = "auto",
        model_size: ModelSizeType = "2.2B",
    ):
        self.model_name = model_name
        self.backend = backend
        self.model_size = model_size
        self._model = None
        self._processor = None
        self._available = None  # None = unchecked, True/False after check
        self._resolved_backend = None  # which backend was actually loaded

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

            self._available = True
        except ImportError as e:
            logger.warning(f"Video MLLM dependencies not available: {e}")
            self._available = False
        return self._available

    def load(self) -> bool:
        """Load the Video MLLM model onto GPU.

        Returns True on success, False if dependencies are missing or
        the model cannot be found on disk / Hugging Face.
        """
        if self._model is not None:
            return True

        if not self.available:
            return False

        # Determine which backend to use
        backend = self._resolve_backend()

        if backend == "smolvlm2":
            return self._load_smolvlm2()
        elif backend == "qwen3_vl":
            return self._load_qwen3_vl()
        elif backend == "internvideo3":
            return self._load_internvideo3()
        else:
            return self._load_videochat_flash()

    def _resolve_backend(self) -> str:
        """Resolve the backend to use, handling 'auto' fallback."""
        if self.backend == "videochat_flash":
            return "videochat_flash"
        elif self.backend == "smolvlm2":
            return "smolvlm2"
        elif self.backend == "qwen3_vl":
            return "qwen3_vl"
        elif self.backend == "internvideo3":
            return "internvideo3"
        elif self.backend == "auto":
            # Try InternVideo3 first (SOTA open-weight model)
            try:
                from video_analysis.backends.internvideo3 import InternVideo3Backend

                ib = InternVideo3Backend()
                if ib._check_vllm_server():
                    logger.info("Auto backend: using InternVideo3 (vLLM server)")
                    return "internvideo3"
            except Exception:
                pass
            # Try Qwen3-VL next (check if vLLM server is available)
            try:
                from video_analysis.backends.qwen3_vl import Qwen3VLBackend

                qb = Qwen3VLBackend(backend="auto")
                if qb._check_vllm_server():
                    logger.info("Auto backend: using Qwen3-VL (vLLM server)")
                    return "qwen3_vl"
            except Exception:
                pass
            # Try SmolVLM2 next (standard API, no trust_remote_code)
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                import decord  # noqa: F401

                # Check if the smolvlm2 model path exists
                model_path = SMOLVLM2_MODEL_PATHS.get(self.model_size)
                if model_path:
                    logger.info(f"Auto backend: trying SmolVLM2 ({model_path}) first")
                    return "smolvlm2"
            except ImportError:
                pass
            # Fall back to VideoChat-Flash
            logger.info("Auto backend: falling back to VideoChat-Flash")
            return "videochat_flash"
        else:
            logger.warning(
                f"Unknown backend '{self.backend}', falling back to videochat_flash"
            )
            return "videochat_flash"

    def _load_videochat_flash(self) -> bool:
        """Load VideoChat-Flash model."""
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            logger.info(f"Loading VideoChat-Flash model: {self.model_name}")
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
            self._resolved_backend = "videochat_flash"
            logger.info("VideoChat-Flash model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load VideoChat-Flash model: {e}")
            self._model = None
            self._processor = None
            self._available = False
            return False

    def _load_smolvlm2(self) -> bool:
        """Load SmolVLM2 model using standard AutoModelForImageTextToText."""
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
            )

            model_path = SMOLVLM2_MODEL_PATHS.get(self.model_size)
            if model_path is None:
                logger.warning(
                    f"Unknown SmolVLM2 model size '{self.model_size}'. "
                    f"Valid options: {list(SMOLVLM2_MODEL_PATHS.keys())}"
                )
                self._available = False
                return False

            logger.info(
                f"Loading SmolVLM2 model: {model_path} (size={self.model_size})"
            )

            # Load processor first (needed for chat template processing)
            self._processor = AutoProcessor.from_pretrained(model_path)

            # Load model without trust_remote_code (standard HF model)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._model.eval()
            self._resolved_backend = "smolvlm2"

            # Update model_name to the actual path used
            self.model_name = model_path

            logger.info("SmolVLM2 model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load SmolVLM2 model: {e}")
            self._model = None
            self._processor = None
            self._available = False
            return False

    def _load_qwen3_vl(self) -> bool:
        """Load Qwen3-VL-30B-A3B backend.

        Delegates to Qwen3VLBackend from video_analysis.backends.qwen3_vl,
        which handles vLLM server, vLLM offline, and transformers backends.
        """
        try:
            from video_analysis.backends.qwen3_vl import Qwen3VLBackend

            # Determine the Qwen3-VL backend type based on config
            qwen_backend = "auto"
            qwen_vllm_url = os.environ.get("QWEN3_VL_VLLM_URL")

            logger.info(f"Loading Qwen3-VL backend (model={self.model_name})")
            self._qwen3_vl = Qwen3VLBackend(
                model_name=self.model_name,
                backend=qwen_backend,
                vllm_server_url=qwen_vllm_url,
                use_fp8=True,
                max_frames=32,
            )
            loaded = self._qwen3_vl.load()
            if loaded:
                self._resolved_backend = "qwen3_vl"
                logger.info("Qwen3-VL backend loaded successfully")
            else:
                logger.warning("Qwen3-VL backend failed to load")
                self._available = False
            return loaded
        except ImportError as e:
            logger.warning(
                f"Qwen3-VL backend not available: {e}. "
                "Install with: pip install vllm  # for vLLM mode"
            )
            self._available = False
            return False
        except Exception as e:
            logger.warning(f"Failed to load Qwen3-VL backend: {e}")
            self._available = False
            return False

    def _load_internvideo3(self) -> bool:
        """Load InternVideo3-8B backend.

        Delegates to InternVideo3Backend from video_analysis.backends.internvideo3,
        which handles vLLM server, vLLM offline, and transformers backends.
        InternVideo3 (arXiv:2606.12195, June 2026) is the strongest open-weight
        video MLLM with MCR reasoning and M^2LA KV-cache compression.
        """
        try:
            from video_analysis.backends.internvideo3 import InternVideo3Backend

            # Determine FP8 mode from config
            use_fp8 = os.environ.get("INTERNVIDEO3_FP8", "true").lower() in (
                "true",
                "1",
                "yes",
            )
            thinking_mode = os.environ.get(
                "INTERNVIDEO3_THINKING", "false"
            ).lower() in (
                "true",
                "1",
                "yes",
            )
            vllm_url = os.environ.get("INTERNVIDEO3_VLLM_URL")

            logger.info(
                "Loading InternVideo3 backend " "(model=%s, fp8=%s, thinking=%s)",
                self.model_name,
                use_fp8,
                thinking_mode,
            )
            self._internvideo3 = InternVideo3Backend(
                model_name=self.model_name,
                use_fp8=use_fp8,
                thinking_mode=thinking_mode,
                vllm_server_url=vllm_url,
            )
            loaded = self._internvideo3.load()
            if loaded:
                self._resolved_backend = "internvideo3"
                logger.info("InternVideo3 backend loaded successfully")
            else:
                logger.warning("InternVideo3 backend failed to load")
                self._available = False
            return loaded
        except ImportError as e:
            logger.warning(
                f"InternVideo3 backend not available: {e}. "
                "Install with: pip install vllm  # for vLLM mode"
            )
            self._available = False
            return False
        except Exception as e:
            logger.warning(f"Failed to load InternVideo3 backend: {e}")
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
            self._resolved_backend = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Video MLLM model unloaded from GPU")

        if hasattr(self, "_qwen3_vl") and self._qwen3_vl is not None:
            self._qwen3_vl.unload()
            self._qwen3_vl = None
            self._resolved_backend = None

        if hasattr(self, "_internvideo3") and self._internvideo3 is not None:
            self._internvideo3.unload()
            self._internvideo3 = None
            self._resolved_backend = None

    # ------------------------------------------------------------------
    # Video-native understanding methods
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded, dispatching to the right backend."""
        return self.load()

    def _decode_video_frames(
        self, video_path: str, num_frames: int = 32
    ) -> Optional[List[str]]:
        """Decode video frames using decord (for SmolVLM2 backend).

        Returns a list of temporary frame file paths, or None on failure.
        """
        try:
            import decord
            import tempfile
            from PIL import Image

            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            if total_frames == 0:
                return None

            # Sample evenly spaced frame indices
            indices = [
                int(i * total_frames / num_frames)
                for i in range(min(num_frames, total_frames))
            ]
            frames = vr.get_batch(indices)

            temp_dir = tempfile.mkdtemp(prefix="smolvlm2_frames_")
            paths = []
            for i, frame_tensor in enumerate(frames):
                img = Image.fromarray(frame_tensor.numpy())
                fp = Path(temp_dir) / f"frame_{i:04d}.jpg"
                img.save(fp)
                paths.append(str(fp))
            return paths
        except ImportError:
            logger.warning(
                "decord not installed; cannot decode video frames for SmolVLM2"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to decode video frames with decord: {e}")
            return None

    def describe_scene(self, frames: List[str], prompt: str = None) -> Optional[str]:
        """Describe a scene from its key frames.

        Args:
            frames: List of paths to frame images.
            prompt: Optional custom prompt.  Defaults to a rich scene
                description prompt.

        Returns:
            Natural language scene description, or None on failure.
        """
        if not frames:
            return None
        if not self._ensure_loaded():
            return None

        try:
            import torch

            prompt = prompt or (
                "Describe this video scene in detail. "
                "Include: setting (indoor/outdoor), visible objects, "
                "people present, their actions, on-screen text, and "
                "the overall mood or tone."
            )

            if self._resolved_backend == "smolvlm2":
                return self._smolvlm2_generate(prompt, frames=frames)
            elif self._resolved_backend == "qwen3_vl":
                return self._qwen3_vl.describe_scene(frames, prompt=prompt)
            elif self._resolved_backend == "internvideo3":
                return self._internvideo3.describe_scene(frames, prompt=prompt)
            else:
                # VideoChat-Flash path
                inputs = self._processor(
                    images=frames,
                    text=prompt,
                    return_tensors="pt",
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    output = self._model.generate(**inputs, max_new_tokens=256)

                description = self._processor.decode(
                    output[0], skip_special_tokens=True
                )
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

        The model internally samples frames from the video file.  SmolVLM2
        uses decord for frame decoding and chat templates; VideoChat-Flash
        handles sampling natively via its processor.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to sample.
            prompt: Optional custom summary prompt.

        Returns:
            Text summary of the video content, or None on failure.
        """
        video = Path(video_path)
        if not video.exists():
            logger.warning(f"Video file not found: {video_path}")
            return None
        if not self._ensure_loaded():
            return None

        try:
            import torch

            prompt = prompt or (
                "Provide a comprehensive summary of this video. "
                "Cover: the setting, key events, people present, "
                "actions that occur, and any important visual or "
                "textual information visible on screen."
            )

            if self._resolved_backend == "smolvlm2":
                # Decode frames with decord, then pass to model
                frame_paths = self._decode_video_frames(
                    str(video), num_frames=num_frames
                )
                if frame_paths is None:
                    return None
                try:
                    result = self._smolvlm2_generate(prompt, frames=frame_paths)
                    return result
                finally:
                    # Clean up temp frames
                    import shutil

                    temp_dir = Path(frame_paths[0]).parent if frame_paths else None
                    if temp_dir and temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
            elif self._resolved_backend == "qwen3_vl":
                return self._qwen3_vl.summarize_video(str(video), num_frames=num_frames)
            elif self._resolved_backend == "internvideo3":
                iv3_frame_paths = self._decode_video_frames(
                    str(video), num_frames=num_frames
                )
                if iv3_frame_paths is None:
                    return None
                try:
                    return self._internvideo3.summarize_video(
                        iv3_frame_paths, max_tokens=512
                    )
                finally:
                    import shutil

                    temp_dir = (
                        Path(iv3_frame_paths[0]).parent if iv3_frame_paths else None
                    )
                    if temp_dir and temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # VideoChat-Flash path
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
        """Answer a question about video content.

        Args:
            query: Natural language question.
            frames: Optional list of frame image paths for visual context.
            video_path: Optional video file path (the model handles
                its own frame sampling when a video path is given).

        Returns:
            Answer text, or None on failure.
        """
        if not frames and not video_path:
            logger.warning("No frames or video path provided for VLM QA")
            return None
        if not self._ensure_loaded():
            return None

        try:
            import torch

            if self._resolved_backend == "smolvlm2":
                if video_path:
                    frame_paths = self._decode_video_frames(video_path, num_frames=16)
                    if frame_paths is None:
                        return None
                    try:
                        result = self._smolvlm2_generate(query, frames=frame_paths)
                        return result
                    finally:
                        import shutil

                        temp_dir = Path(frame_paths[0]).parent if frame_paths else None
                        if temp_dir and temp_dir.exists():
                            shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    return self._smolvlm2_generate(query, frames=frames)
            elif self._resolved_backend == "qwen3_vl":
                return self._qwen3_vl.answer(
                    query, frames=frames, video_path=video_path
                )
            elif self._resolved_backend == "internvideo3":
                return self._internvideo3.answer(
                    query, frame_paths=frames or [], max_tokens=512
                )
            else:
                # VideoChat-Flash path
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

                answer_text = self._processor.decode(
                    output[0], skip_special_tokens=True
                )
                return answer_text.strip()
        except Exception as e:
            logger.warning(f"Video MLLM QA failed: {e}")
            return None

    def _smolvlm2_generate(
        self,
        prompt: str,
        frames: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Optional[str]:
        """Generate text using SmolVLM2 with chat templates.

        SmolVLM2 uses ``{"type": "video", "path": "..."}`` or
        ``{"type": "image", "path": "..."}`` content blocks in
        the chat template.
        """
        try:
            import torch

            content = []
            if frames:
                for fp in frames:
                    content.append({"type": "image", "path": fp})
            if video_path:
                content.append({"type": "video", "path": video_path})
            content.append({"type": "text", "text": prompt})

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            # Apply chat template to build inputs
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            generated_text = self._processor.decode(output[0], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            logger.warning(f"SmolVLM2 generation failed: {e}")
            return None
