# syntax=docker/dockerfile:1
# Multi-stage build for video-analysis platform
# GPU-enabled with CUDA 12.4 runtime

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source
COPY video_analysis/ video_analysis/
COPY ui/ ui/
COPY pyproject.toml .

# ── Runtime stage ──
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="Video Analysis Platform"
LABEL org.opencontainers.image.description="Self-hosted video analysis with AI chatbot — scene understanding, RAG, clip export"
LABEL org.opencontainers.image.version="0.4.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIDEO_ANALYSIS_DATA=/app/data
ENV CUDA_VISIBLE_DEVICES=0

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

WORKDIR /app

# Make sure local bin is on PATH
ENV PATH=/root/.local/bin:$PATH

# Create data directories
RUN mkdir -p /app/data/videos /app/data/frames /app/data/audio \
    /app/data/thumbnails /app/data/chroma /app/data/clips

# Expose Gradio UI port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

ENTRYPOINT ["python3", "-m", "video_analysis"]
