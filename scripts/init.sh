#!/bin/bash
set -e

# ── Video Analysis — Docker Entrypoint ──────────────────────────────
# Auto-detects hardware and downloads the optimal Whisper model.
# ─────────────────────────────────────────────────────────────────────

echo "=== Video Analysis Platform ==="
echo "Starting initialization..."

# Ensure data directories exist
mkdir -p /app/data/videos /app/data/frames /app/data/audio /app/data/thumbnails

# Check for CUDA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "GPU detected: $GPU_INFO"
else
    echo "No NVIDIA GPU detected — will use CPU"
fi

# Auto-select and download Whisper model (unless user specified a fixed model)
if [ "${WHISPER_MODEL}" = "auto" ] || [ -z "${WHISPER_MODEL}" ]; then
    echo "Auto-selecting Whisper model based on hardware..."
    # Let the Python app select the model at startup
    export WHISPER_MODEL="auto"
else
    echo "Using configured Whisper model: ${WHISPER_MODEL}"
fi

echo "Initialization complete. Starting server..."

# Start the application
exec python3 -m video_analysis "$@"
