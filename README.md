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

1. Open **Settings** (`/settings`) — configure your LLM provider, API key, model, and processing parameters.
2. Open **Models** (`/models`) — see available Whisper models, download the one you need, or let it auto-select based on your GPU.
3. Open **Live Stream** (`/stream`) — enter an RTSP URL, set FPS, and click Start to analyze a live feed.

## Features

### Offline Video Analysis (Upload)

Upload any video or paste a YouTube URL. The pipeline transcribes audio via Whisper, extracts keyframes, then uses an LLM Vision API to analyze scenes, detect objects, read text, and answer questions.

- **ASR** — faster-whisper with auto-selected model based on GPU VRAM
- **Scene Detection** — PySceneDetect (AdaptiveDetector)
- **LLM Vision Analysis** — send frames to GPT-4o/Claude/Gemini/DeepSeek/Ollama
- **Q&A Chat** — ask questions about video content with LLM-powered answers

### Real-Time Stream Analysis (CCTV / Live)

Open the **Live Stream** page at `/stream` to analyze real-time RTSP, webcam, or file feeds:

1. Enter the source URL (RTSP URL, webcam index like `0`, or file path)
2. Set FPS, analysis interval, and motion threshold
3. Click **Start Stream** — the engine samples frames, detects motion, and runs LLM Vision analysis
4. Watch events appear in real-time in the timeline
5. Use the chat panel to ask questions about stream events

```bash
# CLI equivalent
docker compose exec video-analysis python -m video_analysis --watch rtsp://camera:554/stream --fps 1.0 --interval 30
```

---

## Configuration

All settings configured through the web UI at `/settings`. No environment variables or `.env` files needed. Settings are persisted to `data/settings.json`.

| Setting | Default | Description |
|---------|---------|-------------|
| LLM Provider | `openai` | `openai`, `anthropic`, `gemini`, `deepseek`, `ollama` |
| LLM API Key | *(empty)* | Your API key |
| LLM Model | `gpt-4o` | Override model name |
| Whisper Model | `auto` | Auto-selects based on GPU VRAM |
| Frame Rate | `0.2` | Frames per second for LLM Vision |
| Max Frames | `30` | Maximum frames sent to LLM per video |
| Scene Threshold | `0.3` | Sensitivity for scene detection |
| Processing Mode | `video_full` | `video_full` or `audio_only` |

### Settings API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/settings` | GET | Current configuration |
| `/api/settings` | PUT | Update configuration |
| `/api/models` | GET | List Whisper models with download status |
| `/api/models/download` | POST | Download a Whisper model |
| `/api/models/status` | GET | Download progress |

### Stream API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stream/start` | POST | Start a real-time stream |
| `/api/stream/stop` | POST | Stop all streams |
| `/api/stream/status` | GET | Stream status and stats |
| `/api/stream/events` | GET | Recent events (JSON) |

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
