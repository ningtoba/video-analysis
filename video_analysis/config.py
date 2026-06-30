"""
Configuration for the video analysis platform.

All settings are configured through the web UI and persisted to
``data/settings.json``. No environment variables or .env files needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration — all defaults, no env var lookups."""

    # ── Paths ───────────────────────────────────────────────────────────
    data_dir: Path = Path("data")

    # ── Server ──────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 7860

    # ── LLM Provider (configured via UI) ────────────────────────────────
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # ── ASR (faster-whisper) ────────────────────────────────────────────
    whisper_model: str = "auto"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    # ── Frame extraction ────────────────────────────────────────────────
    frame_rate: float = 0.2
    max_frames_for_llm: int = 30

    # ── Scene detection ─────────────────────────────────────────────────
    scene_threshold: float = 0.3
    scene_detector: str = "adaptive"

    # ── YouTube import ──────────────────────────────────────────────────
    yt_dlp_enabled: bool = True

    # ── Processing ──────────────────────────────────────────────────────
    processing_mode: str = "video_full"

    # ── Quality screening ───────────────────────────────────────────────
    quality_screening_enabled: bool = True

    # ── Webhook ─────────────────────────────────────────────────────────
    webhook_urls: list[str] | None = None
    webhook_timeout: float = 5.0

    # ── Hugging Face token ──────────────────────────────────────────────
    hf_token: str = ""

    # Derived paths (computed in __post_init__)
    videos_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.videos_dir = self.data_dir / "videos"
        self.frames_dir = self.data_dir / "frames"
        self.audio_dir = self.data_dir / "audio"

        # Auto-fill API key from provider-specific field
        if not self.llm_api_key and self.llm_provider == "anthropic":
            self.llm_api_key = self.anthropic_api_key
        elif not self.llm_api_key and self.llm_provider == "gemini":
            self.llm_api_key = self.gemini_api_key

        # Default model per provider
        if not self.llm_model:
            defaults = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-5-sonnet-20241022",
                "gemini": "gemini-2.0-flash-001",
                "deepseek": "deepseek-chat",
                "ollama": "llama3.2-vision",
            }
            self.llm_model = defaults.get(self.llm_provider, "gpt-4o")

        if not self.llm_api_base and self.llm_provider == "deepseek":
            self.llm_api_base = "https://api.deepseek.com"

        for d in [self.data_dir, self.videos_dir, self.frames_dir, self.audio_dir]:
            d.mkdir(parents=True, exist_ok=True)


SETTINGS_KEYS = [
    "llm_provider", "llm_api_key", "llm_api_base", "llm_model",
    "llm_temperature", "llm_max_tokens",
    "whisper_model", "whisper_device", "whisper_compute_type",
    "frame_rate", "max_frames_for_llm", "scene_threshold",
    "scene_detector", "processing_mode",
    "host", "port",
]


def _default_settings(config: Optional[Config] = None) -> dict:
    """Return default settings dict from a Config instance or hardcoded defaults."""
    if config is not None:
        return {k: getattr(config, k) for k in SETTINGS_KEYS if hasattr(config, k)}
    return {
        "llm_provider": "openai",
        "llm_api_key": "",
        "llm_api_base": "",
        "llm_model": "gpt-4o",
        "llm_temperature": 0.3,
        "llm_max_tokens": 4096,
        "whisper_model": "auto",
        "whisper_device": "auto",
        "whisper_compute_type": "auto",
        "frame_rate": 0.2,
        "max_frames_for_llm": 30,
        "scene_threshold": 0.3,
        "scene_detector": "adaptive",
        "processing_mode": "video_full",
        "host": "0.0.0.0",
        "port": 7860,
    }


def load_settings(data_dir: Path) -> dict:
    """Load settings from ``data_dir/settings.json``, returning defaults if missing."""
    settings_path = data_dir / "settings.json"
    if settings_path.exists():
        try:
            return json.loads(settings_path.read_text())
        except Exception as e:
            logger.warning("Failed to load settings from %s: %s", settings_path, e)
    return _default_settings()


def save_settings(data_dir: Path, settings: dict) -> None:
    """Persist settings dict to ``data_dir/settings.json``."""
    settings_path = data_dir / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2, default=str))
