# ── Single-stage Docker image for video-analysis ──
# Uses python:3.11-slim + ffmpeg + faster-whisper for ASR
# All vision tasks delegated to LLM Vision API (BYO key)
#
# Build:
#   docker build -t video-analysis:latest .
#
# Run:
#   docker run -p 7860:7860 -v ./data:/app/data \
#     -e LLM_API_KEY=sk-... \
#     video-analysis:latest

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Video Analysis Platform"
LABEL org.opencontainers.image.description="Self-hosted video analysis — ASR + LLM Vision for scene understanding, search, and Q&A"
LABEL org.opencontainers.image.version="0.61.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VIDEO_ANALYSIS_DATA=/app/data
ENV WHISPER_MODEL=auto
ENV HOST=0.0.0.0
ENV PORT=7860

WORKDIR /app

# Install system deps: ffmpeg, build tools for pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY pyproject.toml .
COPY video_analysis/ video_analysis/
COPY ui/ ui/
COPY scripts/ scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /app/data/videos /app/data/frames /app/data/audio /app/data/thumbnails /app/data/stream_frames

# Expose web UI port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; r=urllib.request.urlopen('http://localhost:7860/health', timeout=5); exit(0) if r.status==200 else exit(1)" || exit 1

ENTRYPOINT ["/app/scripts/init.sh"]
