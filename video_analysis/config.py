"""
Configuration for the video analysis platform.

All settings can be provided via environment variables or a `.env` file.
Only LLM_API_KEY is required (or an Anthropic/Gemini equivalent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


# Load .env file if present
load_dotenv()




def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


@dataclass
class Config:
    """Central configuration — simplified to essential settings only."""

    # ── Paths ───────────────────────────────────────────────────────────
    data_dir: Path = Path(_env_str("VIDEO_ANALYSIS_DATA", "data"))
    videos_dir: Path = Path(os.environ.get("VIDEOS_DIR", ""))  # default: data_dir / "videos"
    frames_dir: Path = Path(os.environ.get("FRAMES_DIR", ""))
    audio_dir: Path = Path(os.environ.get("AUDIO_DIR", ""))

    # ── Server ──────────────────────────────────────────────────────────
    host: str = _env_str("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 7860)

    # ── LLM Provider (BYO key) ──────────────────────────────────────────
    # Supported providers: openai, anthropic, gemini, deepseek, ollama
    llm_provider: str = _env_str("LLM_PROVIDER", "openai")
    llm_api_key: str = _env_str("LLM_API_KEY", "")
    llm_api_base: str = _env_str("LLM_API_BASE", "")
    llm_model: str = _env_str("LLM_MODEL", "")
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.3)
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 4096)

    # Anthropic specific
    anthropic_api_key: str = _env_str("ANTHROPIC_API_KEY", "")

    # Gemini specific
    gemini_api_key: str = _env_str("GEMINI_API_KEY", "")

    # ── ASR (faster-whisper) ────────────────────────────────────────────
    # Auto-selected based on VRAM if set to "auto"
    whisper_model: str = _env_str("WHISPER_MODEL", "auto")
    whisper_device: str = _env_str("WHISPER_DEVICE", "auto")  # "auto", "cuda", "cpu"
    whisper_compute_type: str = _env_str("WHISPER_COMPUTE_TYPE", "auto")

    # ── Frame extraction ────────────────────────────────────────────────
    # Frames per second to extract for LLM Vision analysis
    frame_rate: float = _env_float("FRAME_RATE", 0.2)  # 1 frame every 5s by default
    # Maximum number of frames to send to LLM Vision per video
    max_frames_for_llm: int = _env_int("MAX_FRAMES_FOR_LLM", 30)

    # ── Scene detection ─────────────────────────────────────────────────
    scene_threshold: float = _env_float("SCENE_THRESHOLD", 0.3)
    scene_detector: str = _env_str("SCENE_DETECTOR", "adaptive")

    # ── YouTube import ──────────────────────────────────────────────────
    yt_dlp_enabled: bool = _env_bool("YT_DLP_ENABLED", True)

    # ── Processing ──────────────────────────────────────────────────────
    # "video_full" or "audio_only"
    processing_mode: str = _env_str("PROCESSING_MODE", "video_full")

    # ── Video quality screening ─────────────────────────────────────────
    quality_screening_enabled: bool = _env_bool("QUALITY_SCREENING_ENABLED", True)

    # ── Webhook ─────────────────────────────────────────────────────────
    webhook_urls: list[str] | None = None  # comma-separated in env var
    webhook_timeout: float = _env_float("WEBHOOK_TIMEOUT", 5.0)

    # ── Hugging Face token (for gated whisper models) ──────────────────
    hf_token: str = _env_str("HF_TOKEN", "")

    def __post_init__(self) -> None:
        """Resolve relative paths and set defaults."""
        # Resolve data paths
        if not self.videos_dir.name:
            self.videos_dir = self.data_dir / "videos"
        if not self.frames_dir.name:
            self.frames_dir = self.data_dir / "frames"
        if not self.audio_dir.name:
            self.audio_dir = self.data_dir / "audio"

        # Resolve webhook URLs
        webhook_str = os.environ.get("WEBHOOK_URLS", "")
        if webhook_str:
            self.webhook_urls = [u.strip() for u in webhook_str.split(",") if u.strip()]

        # Auto-detect API key based on provider
        if not self.llm_api_key and self.llm_provider == "anthropic":
            self.llm_api_key = self.anthropic_api_key
        elif not self.llm_api_key and self.llm_provider == "gemini":
            self.llm_api_key = self.gemini_api_key

        # Default model for each provider
        if not self.llm_model:
            defaults = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-5-sonnet-20241022",
                "gemini": "gemini-2.0-flash-001",
                "deepseek": "deepseek-chat",
                "ollama": "llama3.2-vision",
            }
            self.llm_model = defaults.get(self.llm_provider, "gpt-4o")

        # Default API base for DeepSeek
        if not self.llm_api_base and self.llm_provider == "deepseek":
            self.llm_api_base = "https://api.deepseek.com"

        # Create data directories
        for d in [self.data_dir, self.videos_dir, self.frames_dir, self.audio_dir]:
            d.mkdir(parents=True, exist_ok=True)
