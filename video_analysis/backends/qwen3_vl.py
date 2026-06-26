"""
Qwen3-VL-30B-A3B MoE Video MLLM Backend.

Qwen3-VL-30B-A3B is a Mixture-of-Experts vision-language model with:
- 30B total parameters, 3B active per token (MoE, 128 experts / 8 active)
- FP8 quantization via vLLM or torchao for reduced memory
- 128K context length with sliding window attention
- Hybrid thinking/non-thinking mode (--thinking / --no-thinking)
- Apache 2.0 license

Two deployment modes:
1. **vLLM server** (recommended for production) — run as a separate
   OpenAI-compatible API server, connect via HTTP.  Best for RTX 4070
   where the model is too large for in-process loading alongside other
   pipeline stages.
2. **vLLM offline inference** — direct in-process inference using vLLM's
   LLM class (no separate server needed).  Loads/unloads GPU memory
   compatible with the pipeline's sequential loading pattern.
3. **Transformers fallback** — use when vLLM is unavailable.

References:
    - Model: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
    - Blog:  https://qwenlm.github.io/blog/qwen3/
    - Paper: arXiv:2505.09388 (Qwen3 Technical Report)
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union

logger = logging.getLogger(__name__)

# Model identifiers
QWEN3_VL_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
QWEN3_VL_FP8_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

# Environment variable for vLLM server URL
VLLM_SERVER_URL_ENV = "QWEN3_VL_VLLM_URL"
VLLM_SERVER_DEFAULT = "http://localhost:8000"

# Default vLLM server arguments for Qwen3-VL
VLLM_SERVER_ARGS = [
    "--port",
    "8000",
    "--host",
    "0.0.0.0",
    "--max-model-len",
    "32768",
    "--gpu-memory-utilization",
    "0.9",
    "--enforce-eager",  # Avoid CUDA graph OOM on 12GB
    "--trust-remote-code",
]

Qwen3VLBackendType = Literal["vllm_server", "vllm_offline", "transformers", "auto"]


class Qwen3VLBackend:
    """Backend for Qwen3-VL-30B-A3B video understanding.

    Supports describe_scene, summarize_video, and answer with frames
    or video paths.  Gracefully returns None when unavailable.

    Two deployment modes:
    - vllm_server: Connect to a pre-existing vLLM OpenAI-compatible server
    - vllm_offline: Load vLLM in-process (heavy, ~18 GB VRAM for BF16)
    - transformers: Use HF transformers (heaviest, not recommended on 12GB)
    - auto: Try vLLM server first, then transformers

    For 12 GB RTX 4070, the recommended mode is **vllm_server** with FP8
    quantization on a separate GPU or via CPU offloading.
    """

    def __init__(
        self,
        model_name: str = QWEN3_VL_FP8_MODEL_NAME,
        backend: Qwen3VLBackendType = "auto",
        vllm_server_url: Optional[str] = None,
        use_fp8: bool = True,
        max_frames: int = 32,
        thinking_mode: bool = False,
    ):
        self.model_name = model_name
        self.backend = backend
        self._vllm_server_url = vllm_server_url or os.environ.get(
            VLLM_SERVER_URL_ENV, VLLM_SERVER_DEFAULT
        )
        self.use_fp8 = use_fp8
        self.max_frames = max_frames
        self.thinking_mode = thinking_mode
        self._model = None
        self._llm = None  # vLLM offline LLM instance
        self._sampling_params = None
        self._available = None
        self._resolved_backend = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        # Quick check: can we import dependencies?
        try:
            import torch  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _resolve_backend(self) -> str:
        """Resolve which backend to use."""
        if self.backend == "vllm_server":
            return "vllm_server"
        elif self.backend == "vllm_offline":
            return "vllm_offline"
        elif self.backend == "transformers":
            return "transformers"
        elif self.backend == "auto":
            # Try vLLM server first (fastest, least resource usage in-process)
            if self._check_vllm_server():
                logger.info("Auto backend: using vLLM server")
                return "vllm_server"
            # Try vLLM offline next
            try:
                import vllm  # noqa: F401

                logger.info("Auto backend: using vLLM offline inference")
                return "vllm_offline"
            except ImportError:
                pass
            # Fall back to transformers
            logger.info("Auto backend: falling back to transformers")
            return "transformers"
        return "vllm_server"

    def load(self) -> bool:
        """Load the model for the selected backend.

        For vllm_server, this only checks connectivity.
        For offline backends, this loads the model into GPU memory.
        """
        if self._model is not None or self._llm is not None:
            return True
        if not self.available:
            return False

        backend = self._resolve_backend()
        self._resolved_backend = backend

        if backend == "vllm_server":
            return self._check_vllm_server()
        elif backend == "vllm_offline":
            return self._load_vllm_offline()
        else:
            return self._load_transformers()

    def unload(self):
        """Unload the model from GPU memory."""
        import gc
        import torch

        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._model is not None:
            del self._model
            self._model = None
        self._sampling_params = None
        self._resolved_backend = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("Qwen3-VL model unloaded from GPU")

    # ------------------------------------------------------------------
    # vLLM Server backend
    # ------------------------------------------------------------------

    def _check_vllm_server(self) -> bool:
        """Check if a vLLM server is reachable.

        Sends a lightweight GET to /v1/models to verify connectivity.
        """
        import urllib.request
        import urllib.error

        url = f"{self._vllm_server_url.rstrip('/')}/v1/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("data", [])
                if not models:
                    logger.warning(
                        f"vLLM server at {self._vllm_server_url} "
                        "returned empty model list"
                    )
                    return False
                logger.info(
                    f"Connected to vLLM server at {self._vllm_server_url} "
                    f"(models: {[m['id'] for m in models[:3]]})"
                )
                self._available = True
                return True
        except Exception as e:
            logger.info(f"vLLM server not available at {self._vllm_server_url}: {e}")
            return False

    def _vllm_server_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """Send a chat request to the vLLM server (OpenAI-compatible API)."""
        import urllib.request
        import urllib.error

        url = f"{self._vllm_server_url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.thinking_mode:
            body["extra_body"] = {"thinking": True}

        try:
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"vLLM server chat request failed: {e}")
            return None

    # ------------------------------------------------------------------
    # vLLM Offline Inference backend
    # ------------------------------------------------------------------

    def _load_vllm_offline(self) -> bool:
        """Load vLLM offline inference engine in-process."""
        try:
            from vllm import LLM as VLLM
            from vllm import SamplingParams
        except ImportError:
            logger.warning("vLLM not installed. Install with: pip install vllm")
            self._available = False
            return False

        try:
            import torch

            logger.info(f"Loading Qwen3-VL via vLLM offline: {self.model_name}")

            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "trust_remote_code": True,
                "max_model_len": 32768,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
            }

            if self.use_fp8:
                kwargs["quantization"] = "fp8"
                logger.info("Using FP8 quantization for Qwen3-VL")

            # Try to use FlashAttention-3 if available (Hopper GPUs)
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap >= (9, 0):  # Hopper (H100) or later
                    kwargs["enable_flash_attn"] = True
                    logger.info("Enabling FlashAttention-3 (Hopper GPU)")

            self._llm = VLLM(**kwargs)
            self._sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=512,
            )
            if self.thinking_mode:
                self._sampling_params = SamplingParams(
                    temperature=0.6,
                    max_tokens=1024,
                    top_p=0.95,
                )
            logger.info("vLLM offline inference loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Qwen3-VL via vLLM offline: {e}")
            self._llm = None
            self._available = False
            return False

    def _vllm_offline_generate(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Generate with vLLM offline inference.

        Qwen3-VL expects images embedded in the chat template via
        ``<|vision_start|><|image_pad|><|vision_end|>`` tokens.
        vLLM handles multimodal inputs via ``multi_modal_data`` in the
        SamplingParams or via the newer VLM/MMLU API.

        For vLLM >= 0.8, multimodal inputs use ``/v1/chat/completions``
        with content blocks (similar to OpenAI vision API).
        """
        if self._llm is None:
            return None

        try:
            # Build messages for vLLM's chat interface
            messages = self._build_vllm_messages(prompt, image_paths)

            # vLLM's chat template handles vision tokens automatically
            output = self._llm.chat(
                messages=messages,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )
            return output[0].outputs[0].text.strip()
        except Exception as e:
            logger.warning(f"vLLM offline generation failed: {e}")
            return None

    def _build_vllm_messages(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages for vLLM chat API (OpenAI-compatible format).

        For multimodal input, we use the OpenAI vision format with
        ``image_url`` content blocks, which vLLM translates to the
        model's internal representations.
        """
        content: List[Dict[str, Any]] = []
        if image_paths:
            for img_path in image_paths:
                # Use data URL for local files
                if Path(img_path).exists():
                    import base64

                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    ext = Path(img_path).suffix.lstrip(".") or "jpeg"
                    data_url = f"data:image/{ext};base64,{b64}"
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_path},
                        }
                    )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Transformers backend fallback
    # ------------------------------------------------------------------

    def _load_transformers(self) -> bool:
        """Load Qwen3-VL via HuggingFace transformers.

        Note: On 12GB VRAM, the 30B model is very tight even with FP8.
        Use this only for testing or on larger GPUs.
        """
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            logger.warning(
                "transformers not installed; cannot use transformers backend"
            )
            self._available = False
            return False

        try:
            model_id = self.model_name
            logger.info(
                f"Loading Qwen3-VL via transformers: {model_id} "
                f"(FP8={self.use_fp8})"
            )

            dtype = torch.float8_e4m3fn if self.use_fp8 else torch.bfloat16

            # Load processor first (needed for chat template + vision tokens)
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            # Load model with optional FP8
            kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": dtype,
            }
            if self.use_fp8:
                try:
                    from torchao.quantization import quantize_
                    from torchao.quantization.quant_api import (
                        Int8WeightOnlyConfig,
                    )

                    # Load in BF16 first then quantize weights to FP8
                    kwargs["torch_dtype"] = torch.bfloat16
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        model_id, **kwargs
                    )
                    # Apply FP8 weight quantization via torchao
                    quantize_(self._model, Int8WeightOnlyConfig())
                    logger.info("Applied torchao Int8 weight quantization")
                except ImportError:
                    logger.warning("torchao not available; loading with torch_dtype")
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        model_id, **kwargs
                    )
            else:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_id, **kwargs
                )

            self._model.eval()
            logger.info("Qwen3-VL transformers model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Qwen3-VL via transformers: {e}")
            self._model = None
            self._processor = None
            self._available = False
            return False

    def _transformers_generate(
        self,
        prompt: str,
        frames: Optional[List[str]] = None,
        max_new_tokens: int = 512,
    ) -> Optional[str]:
        """Generate with transformers backend."""
        if self._model is None:
            return None
        try:
            import torch

            messages = [
                {
                    "role": "user",
                    "content": [
                        *([{"type": "image", "path": fp} for fp in (frames or [])]),
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template — Qwen3-VL uses
            # <|vision_start|><|image_pad|><|vision_end|> for images
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=not self.thinking_mode,
                    temperature=0.3,
                )

            generated = self._processor.decode(output[0], skip_special_tokens=True)
            return generated.strip()
        except Exception as e:
            logger.warning(f"Qwen3-VL transformers generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Frame decoding (shared across backends)
    # ------------------------------------------------------------------

    def _decode_video_frames(
        self, video_path: str, num_frames: Optional[int] = None
    ) -> Optional[List[str]]:
        """Decode evenly-spaced frames from a video file.

        Returns a list of temporary JPEG file paths.
        """
        num_frames = num_frames or self.max_frames
        try:
            import decord
            import tempfile
            from PIL import Image

            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(video_path)
            total = len(vr)
            if total == 0:
                return None

            indices = [
                int(i * total / num_frames) for i in range(min(num_frames, total))
            ]
            frames = vr.get_batch(indices)

            temp_dir = tempfile.mkdtemp(prefix="qwen3vl_frames_")
            paths = []
            for i, ft in enumerate(frames):
                img = Image.fromarray(ft.numpy())
                fp = Path(temp_dir) / f"frame_{i:04d}.jpg"
                img.save(fp)
                paths.append(str(fp))
            return paths
        except ImportError:
            logger.warning("decord not installed; cannot decode video frames")
            return None
        except Exception as e:
            logger.warning(f"Failed to decode video frames: {e}")
            return None

    @staticmethod
    def _cleanup_temp_frames(frame_paths: Optional[List[str]]):
        """Clean up temporary frame files."""
        if not frame_paths:
            return
        import shutil

        temp_dir = Path(frame_paths[0]).parent if frame_paths else None
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        return self.load()

    def _generate(
        self,
        prompt: str,
        frames: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Optional[str]:
        """Dispatch generation to the resolved backend."""
        if self._resolved_backend is None:
            if not self._ensure_loaded():
                return None

        backend = self._resolved_backend

        # Decode video frames if video path is given
        frame_paths = frames
        if video_path and not frames:
            frame_paths = self._decode_video_frames(video_path)

        try:
            if backend == "vllm_server":
                messages = self._build_vllm_messages(prompt, frame_paths)
                return self._vllm_server_chat(messages, max_tokens=max_new_tokens)
            elif backend == "vllm_offline":
                return self._vllm_offline_generate(prompt, frame_paths)
            else:
                return self._transformers_generate(
                    prompt, frame_paths, max_new_tokens=max_new_tokens
                )
        finally:
            # Only clean up temp frames we created
            if video_path and not frames and frame_paths:
                self._cleanup_temp_frames(frame_paths)

    def describe_scene(
        self, frames: List[str], prompt: Optional[str] = None
    ) -> Optional[str]:
        """Describe a scene from its key frames."""
        if not frames:
            return None
        prompt = prompt or (
            "Describe this video scene in detail. "
            "Include: setting (indoor/outdoor), visible objects, "
            "people present, their actions, on-screen text, and "
            "the overall mood or tone."
        )
        return self._generate(prompt, frames=frames, max_new_tokens=256)

    def summarize_video(
        self, video_path: str, num_frames: Optional[int] = None
    ) -> Optional[str]:
        """Generate a comprehensive summary of a video."""
        path = Path(video_path)
        if not path.exists():
            logger.warning(f"Video not found: {video_path}")
            return None
        prompt = (
            "Provide a comprehensive summary of this video. "
            "Cover: the setting, key events, people present, "
            "actions that occur, and any important visual or "
            "textual information visible on screen."
        )
        return self._generate(
            prompt,
            video_path=str(path),
            max_new_tokens=512,
            max_frames=num_frames,
        )

    def answer(
        self,
        query: str,
        frames: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Optional[str]:
        """Answer a question about video content."""
        if not frames and not video_path:
            logger.warning("No frames or video path provided")
            return None

        prompt = query or "What do you see in this video?"
        return self._generate(
            prompt,
            frames=frames,
            video_path=video_path,
            max_new_tokens=512,
        )

    # ------------------------------------------------------------------
    # vLLM Server Management (start/stop)
    # ------------------------------------------------------------------

    @staticmethod
    def start_vllm_server(
        model_name: str = QWEN3_VL_FP8_MODEL_NAME,
        port: int = 8000,
        host: str = "0.0.0.0",
        use_fp8: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 32768,
        extra_args: Optional[List[str]] = None,
    ) -> Optional[subprocess.Popen]:
        """Start a vLLM server for Qwen3-VL as a background process.

        This is useful when you want to run the model as a standalone
        service that the video analysis platform connects to via HTTP.

        Returns the Popen handle, or None on failure.
        The caller is responsible for managing the process lifecycle.
        """
        try:
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_name,
                "--port",
                str(port),
                "--host",
                host,
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--max-model-len",
                str(max_model_len),
                "--enforce-eager",
                "--trust-remote-code",
            ]
            if use_fp8:
                cmd.extend(["--quantization", "fp8"])

            if extra_args:
                cmd.extend(extra_args)

            logger.info(f"Starting vLLM server: {' '.join(cmd[:8])}...")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return proc
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return None
