"""
Health check endpoints for the video analysis platform.

Simplified: basic liveness, readiness, and model info.
Uses lazy CUDA detection to avoid crash at startup if NVIDIA driver is flaky.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from pydantic import BaseModel

from video_analysis.config import Config
from video_analysis.model_manager import detect_cuda

logger = logging.getLogger(__name__)

_start_time = time.time()


class HealthStatus(BaseModel):
    status: str
    uptime_seconds: float
    version: str = "0.61.0"
    gpu_available: bool = False
    gpu_vram_mb: int = 0
    llm_configured: bool = False
    whisper_model: str = "auto"


async def _gpu_info():
    """Lazy CUDA detection — called per-request, not at import time."""
    try:
        return detect_cuda()
    except Exception:
        return False, 0


def add_health_endpoints(app: FastAPI, config: Config):
    """Add health check endpoints to the FastAPI app."""

    @app.get("/health/live")
    async def health_live():
        """Liveness probe — always returns 200 if the server is running."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def health_ready():
        """Readiness probe — returns 200 when ready to serve."""
        gpu_available, gpu_vram_mb = await _gpu_info()
        return HealthStatus(
            status="ready",
            uptime_seconds=time.time() - _start_time,
            gpu_available=gpu_available,
            gpu_vram_mb=gpu_vram_mb,
            llm_configured=bool(config.llm_api_key),
            whisper_model=config.whisper_model,
        )

    @app.get("/health")
    async def health():
        """Combined health check."""
        gpu_available, gpu_vram_mb = await _gpu_info()
        return HealthStatus(
            status="ok",
            uptime_seconds=time.time() - _start_time,
            gpu_available=gpu_available,
            gpu_vram_mb=gpu_vram_mb,
            llm_configured=bool(config.llm_api_key),
            whisper_model=config.whisper_model,
        )
