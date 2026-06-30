# Video Analysis Platform

**Self-hosted video understanding with LLM-powered Q&A.** Upload any video or paste a YouTube URL — the pipeline transcribes audio via Whisper, extracts keyframes, then uses an LLM Vision API (GPT-4o, Claude, Gemini, etc.) to analyze scenes, detect objects, read text, and answer questions. No local vision models needed. BYO LLM API key.

```
┌──────────────┐    ┌──────────────────────────┐    ┌───────────────────┐
│  Upload      │───▶│  Analysis Pipeline        │───▶│  JSON Analysis    │
│  (drag-drop) │    │  FFmpeg → Whisper ASR    │    │  + LLM Summary    │
│  or YouTube  │    │  → Scene Detect → Frames  │    └────────┬──────────┘
└──────────────┘    │  → LLM Vision API*       │             │
                    └──────────────────────────┘             │
┌──────────────┐    ┌──────────────────────────┐             │
│  Ask Q&A     │◀───│  LLM API* + Context      │◀────────────┘
│  + Citations │    │  (transcript + analysis)  │
└──────────────┘    └──────────────────────────┘
```

*\*LLM Vision API handles scene understanding, object detection, OCR, people detection, and action recognition — replacing YOLO+CLIP+OCR+Face+Paddle+etc.*

---

## Features

### Core Pipeline

- **ASR** — faster-whisper with auto-selected model based on GPU VRAM (large-v3 on 12GB+, distil-large-v3 on 6GB+, tiny on CPU)
- **Scene Detection** — PySceneDetect (AdaptiveDetector) for boundary detection
- **Keyframe Extraction** — extract frames at scene midpoints or regular intervals
- **LLM Vision Analysis** — send frames to GPT-4o/Claude/Gemini/DeepSeek/Ollama for scene description, object detection, OCR, people count, and action recognition
- **YouTube Import** — download via yt-dlp
- **Auto GPU Detection** — selects optimal Whisper model and compute type based on available VRAM at startup
### Real-Time Stream Analysis (CCTV / Live)

- **Frame Sampling** — capture frames at configurable FPS from RTSP, webcam, or file sources
- **Motion Detection** — lightweight CPU-based change detection (frame diff, histogram, or MOG2)
- **LLM Vision Scheduling** — dual-mode: periodic (every N seconds) AND motion-triggered analysis
- **Event Timeline** — SQLite-backed persistent log of LLM-analyzed events with timestamps
- **Circular Buffer** — keeps the last N seconds/months of frames for context
- **Stream Chat** — ask questions about events in the stream timeline

### Real-Time Stream Analysis

```bash
# Watch an RTSP camera feed (1 FPS, periodic analysis every 30s)
docker compose exec video-analysis python -m video_analysis \
  --watch rtsp://camera:554/stream \
  --fps 1.0 \
  --interval 30

# Watch a webcam
docker compose exec video-analysis python -m video_analysis \
  --watch 0 --source webcam \
  --fps 2.0

# Process an uploaded video in streaming mode
docker compose exec video-analysis python -m video_analysis \
  --watch /app/data/videos/my_video.mp4 --source file \
  --fps 5.0 --interval 15 \
  --motion-threshold 0.01
```

The stream engine creates an event timeline in `data/stream_events.db`. Events are
stored with timestamps, LLM descriptions, motion scores, and frame references.

### LLM Providers (BYO Key)

| Provider | Env Value | Default Model |
|----------|-----------|---------------|
| OpenAI | `openai` | `gpt-4o` |
| Anthropic | `anthropic` | `claude-3-5-sonnet-20241022` |
| Google Gemini | `gemini` | `gemini-2.0-flash-001` |
| DeepSeek | `deepseek` | `deepseek-chat` |
| Ollama (local) | `ollama` | `llama3.2-vision` |

### UI

- FastAPI + Jinja2 web UI with video upload, library management, and chat
- REST API with OpenAPI docs at `/docs`
- Health endpoints at `/health` and `/health/ready`

---

## Quick Start

### Prerequisites

- Docker (with NVIDIA Container Toolkit for GPU support)
- An LLM API key (OpenAI, Anthropic, Gemini, or any OpenAI-compatible provider)

### Run

```bash
# 1. Set your API key
echo "LLM_API_KEY=sk-..." >> .env
# or for Anthropic:
# echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
# echo "LLM_PROVIDER=anthropic" >> .env

# 2. Start
docker compose up -d

# 3. Open http://localhost:7860
```

GPU acceleration (NVIDIA): uncomment the `deploy` section in `docker-compose.yml`.

### CLI Mode

```bash
# Process a video and ask a question
docker compose exec video-analysis python -m video_analysis --cli --video /app/data/videos/my_video.mp4 --query "What happens in this video?"

# Download from YouTube
docker compose exec video-analysis python -m video_analysis --url "https://youtube.com/watch?v=..."
```

---

## Configuration

All settings via environment variables (`.env` file or docker-compose `environment`).

### Required

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | OpenAI-compatible API key |
| `ANTHROPIC_API_KEY` | Anthropic API key (when `LLM_PROVIDER=anthropic`) |
| `GEMINI_API_KEY` | Gemini API key (when `LLM_PROVIDER=gemini`) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `gemini`, `deepseek`, `ollama` |
| `LLM_API_BASE` | *(provider default)* | Custom API base URL |
| `LLM_MODEL` | *(provider default)* | Override model name |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature (0.0–1.0) |
| `WHISPER_MODEL` | `auto` | `auto`, `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3` |
| `FRAME_RATE` | `0.2` | Frames per second for extraction |
| `MAX_FRAMES_FOR_LLM` | `30` | Maximum frames sent to LLM Vision per video |
| `HOST` | `0.0.0.0` | Web UI host |
| `PORT` | `7860` | Web UI port |
| `HF_TOKEN` | *(unset)* | Hugging Face token for gated models |

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/videos` | GET | List processed videos |
| `/api/videos/{id}` | GET | Get video analysis |
| `/api/videos/process` | POST | Process a video |
| `/api/videos/{id}/query` | POST | Ask a question |
| `/api/videos/{id}/frames/{file}` | GET | Get a frame image |
| `/api/upload` | POST | Upload a video file |
| `/api/chat` | POST | Chat about a video |
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness check |

---

## Architecture

The simplified pipeline uses only one local ML model (Whisper for ASR). All vision tasks are delegated to the configured LLM Vision API. This eliminates the need for:

- YOLO (object detection)
- OpenCLIP (scene classification)
- PaddleOCR / EasyOCR (text extraction)
- InsightFace (face detection)
- BGE-VL / Nomic-Embed (text embeddings)
- Cross-encoder / ColBERT (rerankers)
- ChromaDB (vector store)
- Multiple MLLM backends

Pipeline stages:
1. **Audio Extraction** — FFmpeg to 16kHz mono WAV
2. **Transcription** — faster-whisper (auto-selected model size)
3. **Scene Detection** — PySceneDetect ContentDetector
4. **Keyframe Extraction** — FFmpeg at scene midpoints or regular intervals
5. **LLM Vision Analysis** — Send frames to LLM API for scene description, object detection, OCR, people count
6. **Summary** — LLM generates video summary from transcript + visual analysis
7. **Storage** — Analysis saved as JSON (no vector database needed)

---

## Performance

| Operation | Typical Time |
|-----------|-------------|
| Transcription (10 min video, GPU) | ~30s (large-v3) / ~10s (distil-large-v3) |
| Scene Detection (10 min) | ~5s |
| LLM Vision (10-30 frames) | ~5-15s per video |
| Q&A Response | ~2-5s per question |

---

## Project Structure

```
video_analysis/
  __init__.py     — Package version
  __main__.py     — Entry point (CLI + web UI)
  config.py       — Configuration (~15 settings)
  model_manager.py — GPU detection + Whisper model auto-select
  pipeline.py     — Video processing pipeline (ASR + frames + LLM Vision)
  llm_provider.py — LLM provider abstraction (OpenAI, Anthropic, Gemini, etc.)
  chat.py         — Q&A over video analysis
  api.py          — REST API
  models.py       — Data models
  storage.py      — Frame storage utilities
  quality.py      — Frame quality screening
  classifier.py   — Video type classification
  webhook.py      — Event-driven HTTP callbacks
  job_queue.py    — Background job queue
  cache.py        — Pipeline caching
  rate_limiter.py — API rate limiting
  error_handlers.py — Error response formatting
  logging_setup.py — Logging configuration
  client.py       — Python API client
ui/
  server.py       — FastAPI web app
  health.py       — Health check endpoints
  templates/      — Jinja2 HTML templates
  static/         — CSS, JS, images
```

---

## License

MIT
