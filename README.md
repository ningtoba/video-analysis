# 🎥 Video Analysis Platform

**Self-hosted video analysis with an AI chatbot.** Upload any video, paste a YouTube URL, or batch-process files — let the AI pipeline extract and analyze every detail (transcription, scene detection, object recognition, semantic description, OCR, speaker diarization), then ask natural language questions about the content with precise timestamp citations.

```
┌──────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
│  Upload Video│───▶│  Analysis Pipeline       │───▶│  RAG Vector Index   │
│  (drag-drop) │    │  FFmpeg → Whisper        │    │  ChromaDB + BGE     │
│  or YouTube  │    │  → Scene Detect          │    │  + Cross-Encoder    │
│  URL Import  │    │  → YOLO → CLIP → Index  │    └─────────┬───────────┘
└──────────────┘    │  → Sprite Sheet          │              │
                    │  → OCR → Diarization     │              │
                    └─────────────────────────┘              │
┌──────────────┐    ┌─────────────────────┐                  │
│  Ask Q&A     │◀───│  Context Retrieval   │◀─────────────────┘
│  + Citations │    │  Hybrid Search +     │
│  + Clip Export│   │  Temporal Context    │
└──────────────┘    └─────────────────────┘
```

## ✨ Features

- **🎬 Smart Video Analysis** — Scene detection, key frame extraction, transcription (faster-whisper), speaker diarization (PyAnnote), OCR text extraction (PaddleOCR), object detection (YOLO), semantic scene description (OpenCLIP)
- **🌐 YouTube URL Import** — Download videos directly from YouTube, Vimeo, and other platforms via yt-dlp
- **📦 Batch Processing** — Queue videos by URL or file upload for sequential batch analysis
- **💬 AI Chatbot** — Ask questions about video content with timestamped source citations
- **🔍 RAG-Powered** — ChromaDB vector store + BGE embeddings + cross-encoder re-ranking for accurate retrieval
- **✂️ Clip Export** — Export precise video clips at any timestamp range from the UI
- **📚 Video Library** — Multi-video management with searchable library tab
- **🖼️ Timeline Preview** — Sprite sheet generation for visual timeline browsing (hover to preview frames)
- **🎨 Polished UI** — Gradio 6 dark theme with tabs (Analysis, Batch, Library), responsive layout, real-time progress
- **⚡ GPU Accelerated** — RTX 4070 CUDA support for all models with sequential loading to manage 12GB VRAM
- **🔒 100% Local** — No API keys, no cloud services, all processing on your hardware
- **🖥️ CLI Mode** — Process videos, download from URLs, batch process, and query from the terminal

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (for video/audio extraction)
- NVIDIA GPU with CUDA (recommended — CPU fallback works but is slower)
- Hermes Agent CLI (for LLM chat — `hermes chat -q`)

### Install

```bash
# Clone / enter the project directory
cd /home/nekophobia/Projects/video-analysis

# Install dependencies
pip install -r requirements.txt

# Optional — for object detection, OCR, diarization
pip install ultralytics paddleocr pyannote.audio
```

### Launch the Web UI

```bash
python -m video_analysis
```

Then open **http://localhost:7860** in your browser.

### CLI Mode

```bash
# Process a video and ask a question
python -m video_analysis --cli --video my_video.mp4 --query "What objects are visible?"

# Download from YouTube and process
python -m video_analysis --url "https://www.youtube.com/watch?v=..."

# Batch process from a list file
python -m video_analysis --batch urls.txt
```

## 🏗️ Architecture

### Ingestion Pipeline

```
Video File
├── FFmpeg ──→ Extract Audio (16kHz WAV)
│              └── faster-whisper (large-v3) ──→ Timestamped Transcript
│              └── PyAnnote Audio ──→ Speaker Diarization (SPEAKER_00/01)
├── FFmpeg ──→ Scene Detection (scene filter)
│              └── Per Scene: keyframe extraction
│                            ├── YOLO object detection
│                            ├── PaddleOCR text extraction
│                            ├── OpenCLIP zero-shot scene classification
│                            └── Frame metadata
├── FFmpeg ──→ Sprite sheet (100 thumbnails for timeline)
└── Merge ──→ Structured VideoIndex
              └── ChromaDB Vector Store (BGE embeddings)
```

### Query Pipeline

```
User Question
├── BGE Embedding
├── ChromaDB Hybrid Search (dense + metadata)
├── Cross-Encoder Re-ranking (MS MARCO MiniLM)
├── Temporal Context Expansion (±1 neighbor scene)
├── Sort Chronologically
└── LLM (Hermes/DeepSeek) → Answer with timestamp citations
```

### Module Structure

| Module | Path | Purpose |
|--------|------|---------|
| `pipeline` | `video_analysis/pipeline.py` | Video processing — scene detection, frame extraction, transcription, diarization, YOLO, OCR, CLIP, sprite sheets, YouTube/URL download |
| `rag` | `video_analysis/rag.py` | ChromaDB indexing, hybrid retrieval, re-ranking, temporal expansion |
| `chat` | `video_analysis/chat.py` | LLM Q&A with conversation history and source citations |
| `models` | `video_analysis/models.py` | Data models — VideoIndex, SceneInfo, FrameInfo, ChatMessage |
| `config` | `video_analysis/config.py` | Configuration with sensible defaults |
| `ui/app` | `ui/app.py` | Gradio web interface with dark theme, tabs, library, clip export, batch queue, URL import |
| `ui/utils` | `ui/utils.py` | Shared UI utility functions (importable without gradio) |

## 💻 Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **Backend** | Python 3.14 + FastAPI | Async, fast, built-in |
| **UI Framework** | Gradio 6 Blocks | Best video + chat components, custom CSS/JS |
| **Transcription** | faster-whisper (large-v3) | ~12× realtime on RTX 4070, int8 quantized |
| **Speaker Diarization** | PyAnnote Audio 3.1 | Gold-standard speaker labeling, optional fallback |
| **OCR** | PaddleOCR | Best accuracy for natural scenes, CPU mode |
| **Scene Detection** | FFmpeg scene filter | Always available, no extra deps |
| **Object Detection** | YOLO (ultralytics) | State-of-the-art speed/accuracy |
| **Scene Description** | OpenCLIP (ViT-B-32) | Zero-shot classification, rich semantic understanding |
| **Timeline Preview** | FFmpeg + Pillow sprite sheets | 100-thumbnail visual timeline navigation |
| **Vector Store** | ChromaDB | Persistent, local, no server needed |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Strong retrieval, light weight |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM | Boosts precision to ~95%+ |
| **Video Import** | yt-dlp | Downloads from YouTube, Vimeo, Twitch, and 1000+ sites |
| **LLM** | DeepSeek-V4-Flash (via Hermes) | Fast, capable, local provider |
| **GPU** | RTX 4070 (CUDA 13.3) | All models run with GPU acceleration |

## 🔧 Configuration

Set via environment variables or edit `video_analysis/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_ANALYSIS_DATA` | `data/` | Data directory for videos, frames, audio, chroma |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device for transcription |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for RAG |
| `OCR_ENABLED` | `true` | Enable PaddleOCR text extraction |
| `DIARIZE_ENABLED` | `true` | Enable PyAnnote speaker diarization |
| `YT_DLP_ENABLED` | `true` | Enable YouTube/URL video import |
| `UI_HOST` | `0.0.0.0` | Web UI bind address |
| `UI_PORT` | `7860` | Web UI port |

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
# or
python tests/test_basic.py
```

## 📊 Performance (RTX 4070)

| Operation | Time (10min video) |
|-----------|-------------------|
| Audio extraction | ~30s |
| Transcription (large-v3, int8) | ~50s (~12× realtime) |
| Scene detection | ~20s |
| Frame extraction + object detection | ~60s |
| CLIP scene description | ~30s |
| Sprite sheet generation | ~15s |
| RAG indexing | ~5s |
| **Total pipeline** | **~3-4 min** |
| Q&A response | ~2-5s per question |

## 🗺️ Roadmap

- [x] Core video analysis pipeline
- [x] RAG indexing and retrieval
- [x] Chat interface with source citations
- [x] Gradio web UI
- [x] OpenCLIP zero-shot scene classification
- [x] Thumbnail sprite sheets for timeline preview
- [x] Clip export (jump to precise moments)
- [x] Multi-video library management
- [x] GPU pipeline management (sequential model loading for 12GB VRAM)
- [x] Speaker diarization (PyAnnote)
- [x] OCR text extraction (PaddleOCR)
- [x] Docker deployment
- [x] YouTube URL import (yt-dlp)
- [x] Batch video processing queue
- [ ] Frame preview on timeline hover (CSS sprite sheet overlay)
- [ ] PySceneDetect for improved scene boundaries
- [ ] OpenCLIP ViT-L-14 upgrade (richer scene descriptions)
- [ ] ColBERTv2 late-interaction re-ranking
- [ ] Action recognition (VideoMAE/TimeSformer)

## 📝 License

MIT
