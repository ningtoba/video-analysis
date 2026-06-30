"""
Auto model selection and download manager.

Detects available hardware (GPU VRAM) and selects the optimal
Whisper model size for the current environment. Downloads models on first use.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Whisper model configs
WHISPER_MODELS = {
    "tiny": {"params": "39M", "vram_mb": 200, "speed": 32, "wer": 7.7},
    "base": {"params": "74M", "vram_mb": 300, "speed": 16, "wer": 6.4},
    "small": {"params": "244M", "vram_mb": 460, "speed": 6, "wer": 5.1},
    "medium": {"params": "769M", "vram_mb": 1000, "speed": 2, "wer": 4.7},
    "large-v3": {"params": "1550M", "vram_mb": 2200, "speed": 1, "wer": 4.2},
    "large-v3-turbo": {"params": "809M", "vram_mb": 1500, "speed": 8, "wer": 4.4},
    "distil-large-v3": {"params": "756M", "vram_mb": 1100, "speed": 9, "wer": 4.7},
}

# VRAM tiers: (min_vram_mb, model_name)
VRAM_TIERS = [
    (12000, "large-v3"),
    (8000, "large-v3-turbo"),
    (6000, "distil-large-v3"),
    (4000, "medium"),
    (2000, "small"),
    (1000, "base"),
]


def detect_cuda() -> Tuple[bool, int]:
    """Detect CUDA availability and VRAM in MB.

    Returns:
        Tuple of (cuda_available, vram_mb)
    """
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            vram_mb = int(result.stdout.strip().split("\n")[0].strip())
            logger.info("Detected CUDA GPU: %d MB VRAM", vram_mb)
            return True, vram_mb
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass

    # Try torch
    try:
        import torch

        if torch.cuda.is_available():
            vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            logger.info("Detected CUDA via torch: %d MB VRAM", vram_mb)
            return True, vram_mb
    except (ImportError, RuntimeError):
        pass

    logger.info("No CUDA GPU detected, using CPU")
    return False, 0


def select_whisper_model(vram_mb: int = 0) -> str:
    """Select the best Whisper model for available VRAM."""
    if vram_mb <= 0:
        logger.info("No GPU — selecting whisper 'tiny' for CPU")
        return "tiny"

    for tier_vram, model_name in VRAM_TIERS:
        if vram_mb >= tier_vram:
            logger.info(
                "Selected whisper '%s' (%d MB VRAM available, need %d MB)",
                model_name, vram_mb, WHISPER_MODELS[model_name]["vram_mb"],
            )
            return model_name

    logger.info("Limited VRAM (%d MB) — selecting whisper 'tiny'", vram_mb)
    return "tiny"


def select_compute_type(vram_mb: int, model_name: str) -> str:
    """Select optimal compute type based on VRAM."""
    if vram_mb <= 0:
        return "int8"
    model_vram = WHISPER_MODELS.get(model_name, {}).get("vram_mb", 1000)
    return "float16" if vram_mb >= model_vram * 2 else "int8_float16"


def ensure_whisper_model(
    model_name: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Auto-select Whisper model and return (model_name, device, compute_type)."""
    cuda_available, vram_mb = detect_cuda()

    if model_name is None or model_name == "auto":
        model_name = select_whisper_model(vram_mb)
    elif model_name not in WHISPER_MODELS:
        logger.warning("Unknown model '%s', auto-selecting", model_name)
        model_name = select_whisper_model(vram_mb)

    device = "cuda" if cuda_available else "cpu"
    compute_type = select_compute_type(vram_mb, model_name)

    logger.info("Whisper: model=%s device=%s compute=%s", model_name, device, compute_type)
    return model_name, device, compute_type


def download_whisper_model(model_name: str) -> bool:
    """Pre-download a Whisper model. Returns True if ready."""
    try:
        from faster_whisper import download_model

        logger.info("Downloading whisper model '%s'...", model_name)
        download_model(model_name)
        logger.info("Whisper model '%s' ready", model_name)
        return True
    except ImportError:
        logger.warning("faster-whisper not installed")
        return False
    except Exception as e:
        logger.warning("Failed to download '%s': %s", model_name, e)
        return False
