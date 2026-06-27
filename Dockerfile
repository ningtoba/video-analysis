# syntax=docker/dockerfile:1
# ── Multi-stage build for video-analysis platform ──
# GPU-enabled: CUDA 12.8 runtime with torch 2.12+ wheels (cu128)
#   Host driver: NVIDIA 610.43.02  |  nvidia-container-toolkit required
#   VRAM budget (12 GB RTX 4070): models loaded sequentially in pipeline
#
# Build:
#   docker build -t video-analysis:latest .
#
# Run:
#   docker run --gpus all -p 7860:7860 -v ./data:/app/data video-analysis:latest
#
# ── Builder stage: compile deps & install Python packages ──
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build & media dependencies
#   - ffmpeg: audio extraction, sprite sheets, clip export
#   - gcc/g++: C extensions (onnxruntime, tokenizers, open-clip-torch)
#   - libgomp1: OpenMP parallel loops (numpy, torch)
#   - rustc: tokenizers build dependency (sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps into /install
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --target=/install \
    -r requirements.txt

# Copy application source
COPY video_analysis/ video_analysis/
COPY ui/ ui/
COPY pyproject.toml .

# ── Runtime stage ──
# nvidia/cuda:12.8.0-runtime-ubuntu22.04 ships CUDA 12.8 runtime + cuBLAS + cuFFT + NCCL
#   - Compatible with torch 2.6+ cu128 wheels from pytorch.org
#   - No nvcc/devel headers needed after pip installs compiled against CUDA stubs
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="Video Analysis Platform"
LABEL org.opencontainers.image.description="Self-hosted video analysis with AI chatbot — scene understanding, RAG, clip export, YouTube import, batch processing"
LABEL org.opencontainers.image.version="0.48.0"
LABEL org.opencontainers.image.vendor="Nous Research"
LABEL org.opencontainers.image.documentation="https://github.com/.../video-analysis"

# ── Runtime environment ──
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Application paths ──
ENV VIDEO_ANALYSIS_DATA=/app/data
ENV CUDA_VISIBLE_DEVICES=0

# Install runtime system dependencies
#   - python3 / pip: must match builder's Python (3.11) — we install python3.11
#   - ffmpeg: audio extraction, sprite sheet, clip export
#   - libgl1, libglib: OpenCV GUI modules (imshow etc.)
#   - libsm/libxext/libxrender: X11 SHM (used by some CV libs)
#   - libgomp1: OpenMP threading (numpy, torch inference)
#   - libnss3, libnspr4: Chromium-based components in some Gradio plugins
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libnss3 \
    libnspr4 \
    && rm -rf /var/lib/apt/lists/*

# ── Symlink python3 → python3.11 if needed ──
# Ensures `python3` resolves correctly; the slim builder used 3.11
RUN python3 --version

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
# Copy application source
COPY --from=builder /app /app

WORKDIR /app

# ── Data directories (bind-mount override via -v) ──
RUN mkdir -p /app/data/videos /app/data/frames /app/data/audio \
    /app/data/thumbnails /app/data/chroma /app/data/clips

# ── Create non-root user ──
# Security best practice: run as non-root inside container
RUN groupadd -r video-analysis && useradd -r -g video-analysis -d /app -s /sbin/nologin video-analysis \
    && chown -R video-analysis:video-analysis /app

# ── Expose ports ──
EXPOSE 7860

# ── Health check ──
# Probes /health endpoint on the FastAPI server.
# start-period accounts for model loading (faster-whisper, CLIP, ChromaDB).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "
import urllib.request
try:
    resp = urllib.request.urlopen('http://localhost:7860/health', timeout=5)
    assert resp.status == 200
except Exception:
    raise SystemExit(1)
" || exit 1

# ── Drop privileges ──
USER video-analysis

# ── Launch ──
ENTRYPOINT ["python3", "-m", "video_analysis"]
