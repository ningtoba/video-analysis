"""
InternVideo3 Video MLLM Backend.

InternVideo3 (June 2026, arXiv:2606.12195) is the strongest open-weight
video MLLM, built on Qwen3-VL-8B with two key innovations:

1. **Multimodal Contextual Reasoning (MCR):** closed-loop long-video
   understanding as iterative evidence accumulation — the model watches,
   reasons, accumulates evidence, and re-watches selectively.

2. **M^2LA (Multi-Modal Memory-Latency Adapter):** token-preserving
   KV-cache compression that achieves 1.84× faster decode at 32K tokens
   without degrading quality.

Key metrics (8B open-weight class):
- Video-MME: 73.8 (best; Qwen3-VL-8B: 71.4, Eagle2.5: 72.4)
- MLVU: 77.3 (best; Qwen3-VL-8B: 57.6)
- EgoSchema: 76.6 (best; Qwen3-VL-8B: 69.8)
- VRBench: 69.4 (best; Qwen3-VL-8B: 59.4)

Three deployment modes:
1. **vLLM server** (recommended) — OpenAI-compatible API server, best for
   RTX 4070 where the 8B model shares VRAM with other pipeline stages
2. **vLLM offline inference** — in-process via vLLM LLM class, loads/unloads
   GPU memory compatible with pipeline's sequential loading pattern
3. **Transformers fallback** — direct HuggingFace transformers inference

References:
    - Paper: https://arxiv.org/abs/2606.12195
    - Code:  https://github.com/OpenGVLab/InternVideo3 (expected public Q3 2026)
    - Model: (pending HF release — Qwen3-VL-8B-MCR variant)
"""

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# Model identifiers
INTERNVIDEO3_MODEL_NAME = "OpenGVLab/InternVideo3-8B-Instruct"

# Environment variable for vLLM server URL
VLLM_SERVER_URL_ENV = "INTERNVIDEO3_VLLM_URL"
VLLM_SERVER_DEFAULT = "http://localhost:8001"

# Default vLLM server arguments for InternVideo3
VLLM_SERVER_ARGS = [
    "--port",
    "8001",
    "--host",
    "0.0.0.0",
    "--model",
    INTERNVIDEO3_MODEL_NAME,
    "--trust-remote-code",
    "--max-model-len",
    "32768",
    "--gpu-memory-utilization",
    "0.9",
    "--enforce-eager",
    "--dtype",
    "bfloat16",
]

# When using Qwen3-VL-8B as base (pre-InternVideo3 weights)
INTERNVIDEO3_BASE_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


class InternVideo3Backend:
    """InternVideo3 video MLLM backend.

    Supports three modes of operation, tried in order:
    1. vLLM server (OpenAI-compatible API, via URL)
    2. vLLM offline inference (in-process)
    3. Transformers fallback (direct HuggingFace)

    Args:
        model_name: Model identifier (default: InternVideo3-8B)
        use_fp8: Enable FP8 quantization (reduces VRAM ~50%)
        thinking_mode: Enable MCR thinking mode for complex reasoning
        vllm_server_url: Override vLLM server URL
    """

    def __init__(
        self,
        model_name: str = INTERNVIDEO3_MODEL_NAME,
        use_fp8: bool = False,
        thinking_mode: bool = False,
        vllm_server_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_fp8 = use_fp8
        self.thinking_mode = thinking_mode
        self._vllm_server_url = (
            vllm_server_url or os.environ.get(VLLM_SERVER_URL_ENV) or VLLM_SERVER_DEFAULT
        )

        # Runtime state
        self._available: Optional[bool] = None  # None = unchecked
        self._mode: Optional[Literal["vllm_server", "vllm_offline", "transformers"]] = None
        self._llm = None  # vLLM offline engine
        self._sampling_params = None
        self._model = None  # transformers model
        self._processor = None  # transformers processor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Check whether the backend can be used."""
        if self._available is not None:
            return self._available
        self._detect()
        return bool(self._available)

    def load(self) -> bool:
        """Load the backend (try vLLM server first, then offline, then transformers)."""
        if self._available:
            return True

        # 1. Try vLLM server
        if self._check_vllm_server():
            self._mode = "vllm_server"
            self._available = True
            logger.info(
                "InternVideo3: using vLLM server at %s",
                self._vllm_server_url,
            )
            return True

        # 2. Try vLLM offline
        if self._load_vllm_offline():
            self._mode = "vllm_offline"
            self._available = True
            logger.info("InternVideo3: using vLLM offline inference")
            return True

        # 3. Try transformers fallback
        if self._load_transformers():
            self._mode = "transformers"
            self._available = True
            logger.info("InternVideo3: using transformers fallback")
            return True

        self._available = False
        logger.warning("InternVideo3: no backend available. Install vLLM or transformers.")
        return False

    def unload(self) -> None:
        """Unload model from GPU memory."""
        self._llm = None
        self._model = None
        self._processor = None
        self._available = None
        self._mode = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("InternVideo3: unloaded from GPU memory")

    def describe_scene(
        self,
        frame_paths: List[str],
        prompt: Optional[str] = None,
        max_tokens: int = 256,
    ) -> Optional[str]:
        """Describe a scene from its key frames.

        Args:
            frame_paths: Paths to frame images (JPEG/PNG)
            prompt: Optional custom prompt; defaults to a detailed scene description prompt
            max_tokens: Maximum tokens in response

        Returns:
            Scene description text, or None on failure
        """
        prompt = prompt or (
            "Describe this video scene in detail. Include:\n"
            "1. What objects, people, and actions are visible\n"
            "2. The setting and environment\n"
            "3. Any text or signs visible\n"
            "4. The overall mood and visual style\n"
            "Be specific and thorough."
        )
        return self._generate(prompt, frame_paths, max_tokens=max_tokens)

    def answer(
        self,
        query: str,
        frame_paths: List[str],
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Answer a question about video content using frames as context.

        Args:
            query: The user's question about the video
            frame_paths: Paths to relevant frame images
            max_tokens: Maximum tokens in response

        Returns:
            Answer text, or None on failure
        """
        return self._generate(query, frame_paths, max_tokens=max_tokens)

    def summarize_video(
        self,
        frame_paths: List[str],
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Generate a global video summary from sampled frames.

        Uses InternVideo3's MCR reasoning for comprehensive summarization.

        Args:
            frame_paths: Sampled frame paths (typically 16-32 evenly spaced)
            max_tokens: Maximum tokens in summary

        Returns:
            Video summary text, or None on failure
        """
        prompt = (
            "Provide a comprehensive summary of this video based on the sampled frames. "
            "Cover:\n"
            "1. The main topic or narrative\n"
            "2. Key scenes, objects, and people\n"
            "3. Any actions or events that unfold\n"
            "4. The overall message or purpose\n"
            "Be detailed and well-structured."
        )
        return self._generate(prompt, frame_paths, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _detect(self) -> None:
        """Detect which backend mode is available without full loading."""
        if self._check_vllm_server():
            self._mode = "vllm_server"
            self._available = True
            return
        try:
            import vllm  # noqa: F401

            self._mode = "vllm_offline"
            self._available = True
            return
        except ImportError:
            pass
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            self._mode = "transformers"
            self._available = True
            return
        except ImportError:
            pass
        self._available = False

    # ------------------------------------------------------------------
    # vLLM Server backend
    # ------------------------------------------------------------------

    def _check_vllm_server(self) -> bool:
        """Check if a vLLM server is already running and responsive."""
        url = f"{self._vllm_server_url.rstrip('/')}/v1/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            models = data.get("data", [])
            if not models:
                logger.warning(
                    "vLLM server at %s returned empty model list",
                    self._vllm_server_url,
                )
                return False
            logger.info(
                "Connected to vLLM server at %s (models: %s)",
                self._vllm_server_url,
                [m["id"] for m in models[:3]],
            )
            return True
        except Exception as e:
            logger.info("vLLM server not available at %s: %s", self._vllm_server_url, e)
            return False

    def _vllm_server_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """Send a chat request to the vLLM server (OpenAI-compatible API)."""
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
            logger.warning("InternVideo3 vLLM server chat failed: %s", e)
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
            return False

        try:
            import torch

            logger.info("Loading InternVideo3 via vLLM offline: %s", self.model_name)

            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "trust_remote_code": True,
                "max_model_len": 32768,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
            }

            if self.use_fp8:
                kwargs["quantization"] = "fp8"
                logger.info("Using FP8 quantization for InternVideo3")

            # FlashAttention-3 for Hopper GPUs
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap >= (9, 0):
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
            logger.info("InternVideo3 vLLM offline loaded successfully")
            return True
        except Exception as e:
            logger.warning("Failed to load InternVideo3 via vLLM offline: %s", e)
            self._llm = None
            self._available = False
            return False

    def _vllm_offline_generate(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Generate with vLLM offline inference."""
        if self._llm is None:
            return None
        try:
            messages = self._build_multimodal_messages(prompt, image_paths)
            output = self._llm.chat(
                messages=messages,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )
            return output[0].outputs[0].text.strip()
        except Exception as e:
            logger.warning("InternVideo3 vLLM offline generation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Transformers backend fallback
    # ------------------------------------------------------------------

    def _load_transformers(self) -> bool:
        """Load InternVideo3 via HuggingFace transformers."""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            logger.warning("transformers not installed; cannot use transformers backend")
            return False

        try:
            import torch

            logger.info("Loading InternVideo3 via transformers: %s", self.model_name)

            # Use FP8 if requested (requires torchao or bitsandbytes)
            kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
            }

            if self.use_fp8 and torch.cuda.is_available():
                try:
                    kwargs["quantization_config"] = {
                        "quant_method": "fp8",
                        "activation_scheme": "static",
                    }
                    logger.info("Using FP8 quantization")
                except Exception:
                    logger.warning("FP8 quantization not available, using BF16")

            self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, **kwargs)
            self._processor = AutoProcessor.from_pretrained(self.model_name)

            # Move to GPU
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

            self._model.eval()
            logger.info("InternVideo3 transformers backend loaded successfully")
            return True
        except Exception as e:
            logger.warning(
                "Failed to load InternVideo3 via transformers: %s. "
                "The model may not yet be released on HF — "
                "try using the Qwen3-VL-8B base model instead.",
                e,
            )
            self._model = None
            self._processor = None
            return False

    def _transformers_generate(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Generate with transformers."""
        if self._model is None or self._processor is None:
            return None
        try:
            import torch
            from PIL import Image

            images = []
            if image_paths:
                for p in image_paths:
                    if Path(p).exists():
                        images.append(Image.open(p).convert("RGB"))

            # Build conversation
            messages = self._build_multimodal_messages(prompt, image_paths)
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self._processor(
                text=[text],
                images=images if images else None,
                return_tensors="pt",
                padding=True,
            )

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                )

            response = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            return response.strip()
        except Exception as e:
            logger.warning("InternVideo3 transformers generation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_multimodal_messages(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages in OpenAI vision format for multimodal input."""
        content: List[Dict[str, Any]] = []
        if image_paths:
            for img_path in image_paths:
                if Path(img_path).exists():
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

    def _generate(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Route generation to the active backend mode."""
        if not self._available and not self.load():
            return None

        if self._mode == "vllm_server":
            messages = self._build_multimodal_messages(prompt, image_paths)
            return self._vllm_server_chat(messages, max_tokens=max_tokens)
        elif self._mode == "vllm_offline":
            return self._vllm_offline_generate(prompt, image_paths)
        elif self._mode == "transformers":
            return self._transformers_generate(prompt, image_paths)
        else:
            logger.warning("InternVideo3: no active backend mode")
            return None

    @property
    def mode(self) -> Optional[str]:
        """Return the active backend mode."""
        return self._mode
