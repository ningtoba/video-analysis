#!/bin/bash
set -e
echo "=== Video Analysis Platform ==="
echo "Starting initialization..."
mkdir -p /app/data/videos /app/data/frames /app/data/audio /app/data/thumbnails

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "GPU detected: $GPU_INFO"
else
    echo "No NVIDIA GPU detected — will use CPU"
fi

echo "Initialization complete. Starting server..."
exec python3 -m video_analysis "$@"
