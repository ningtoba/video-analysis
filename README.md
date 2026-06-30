## Quick Start

### Prerequisites

- Docker (with NVIDIA Container Toolkit for GPU support)
- An LLM API key (OpenAI, Anthropic, Gemini, or any OpenAI-compatible provider)

### Run

```bash
# Start the container (no .env file needed)
docker compose up -d

# Open the web UI
open http://localhost:7860
```

### First-Time Setup

All configuration is done through the web UI:

1. Open **Settings** (`/settings`) — configure your LLM provider, API key, model, and processing parameters. Changes save immediately to `data/settings.json`.
2. Open **Models** (`/models`) — see available Whisper models, download the one you need, or let it auto-select based on your GPU.
3. Open **Home** (`/`) — upload a video or paste a YouTube URL to start analyzing.
4. Use the **Chat** panel to ask questions about the processed video.

GPU acceleration (NVIDIA): uncomment the `deploy` section in `docker-compose.yml`.

### CLI Mode

```bash
# Process a video and ask a question
docker compose exec video-analysis python -m video_analysis --cli --video /app/data/videos/my_video.mp4 --query "What happens in this video?"

# Download from YouTube
docker compose exec video-analysis python -m video_analysis --url "https://youtube.com/watch?v=..."
```

### Real-Time Stream Analysis (CCTV / Live)

```bash
# Watch an RTSP camera feed (1 FPS, periodic analysis every 30s)
docker compose exec video-analysis python -m video_analysis --watch rtsp://camera:554/stream --fps 1.0 --interval 30

# Watch a webcam
docker compose exec video-analysis python -m video_analysis --watch 0 --source webcam --fps 2.0

# Process an uploaded video in streaming mode
docker compose exec video-analysis python -m video_analysis --watch /app/data/videos/my_video.mp4 --source file --fps 5.0
```

---

## Configuration

All settings are configured through the web UI at `/settings`. No environment variables or `.env` files needed.

| Setting | Default | Description |
|---------|---------|-------------|
| LLM Provider | `openai` | `openai`, `anthropic`, `gemini`, `deepseek`, `ollama` |
| LLM API Key | *(empty)* | Your API key for the LLM provider |
| LLM Model | `gpt-4o` | Model name (auto-defaults per provider) |
| Whisper Model | `auto` | Auto-selects based on GPU VRAM |
| Frame Rate | `0.2` | Frames per second for LLM Vision analysis |
| Max Frames | `30` | Maximum frames sent to LLM per video |
| Scene Threshold | `0.3` | Sensitivity for scene detection |
| Processing Mode | `video_full` | `video_full` or `audio_only` |

Settings are persisted to `data/settings.json` in the mounted volume and survive container restarts.

### Settings API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/settings` | GET | Current configuration |
| `/api/settings` | PUT | Update configuration |
| `/api/models` | GET | List Whisper models with download status |
| `/api/models/download` | POST | Download a Whisper model |
| `/api/models/status` | GET | Download progress |

---

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/videos` | GET | List processed videos |
| `/api/videos/{id}` | GET | Get video analysis |
| `/api/videos/process` | POST | Process a video |
| `/api/videos/{id}/query` | POST | Ask a question |
| `/api/upload` | POST | Upload a video file |
| `/api/chat` | POST | Chat about a video |
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness check |
