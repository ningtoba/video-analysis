"""
Health check endpoints for the video analysis platform.

Simplified: basic liveness, readiness, and model info.
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


def add_health_endpoints(app: FastAPI, config: Config):
    """Add health check endpoints to the FastAPI app."""

    cuda_available, vram_mb = detect_cuda()
    llm_configured = bool(config.llm_api_key)

    @app.get("/health/live")
    async def health_live():
        """Liveness probe — always returns 200 if the server is running."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def health_ready():
        """Readiness probe — returns 200 when ready to serve."""
        return HealthStatus(
            status="ready",
            uptime_seconds=time.time() - _start_time,
            gpu_available=cuda_available,
            gpu_vram_mb=vram_mb,
            llm_configured=llm_configured,
            whisper_model=config.whisper_model,
        )

    @app.get("/health")
    async def health():
        """Combined health check."""
        return HealthStatus(
            status="ok",
            uptime_seconds=time.time() - _start_time,
            gpu_available=cuda_available,
            gpu_vram_mb=vram_mb,
            llm_configured=llm_configured,
            whisper_model=config.whisper_model,
        )
