# 🎥 Video Analysis Platform

**Self-hosted video analysis with an AI chatbot.** Upload any video, paste a YouTube URL, or batch-process files — let the AI pipeline extract and analyze every detail (transcription, scene detection, object recognition, semantic description, OCR, speaker diarization), then ask natural language questions about the content with precise timestamp citations.

```
┌──────────────┐    ┌─────────────────────────┐    ┌──────────────────────────┐
│  Upload Video│───▶│  Analysis Pipeline       │───▶│  RAG Vector Index        │
│  (drag-drop) │    │  PySceneDetect 0.7        │    │  ChromaDB + BGE-VL       │
│  or YouTube  │    │  → Scene Detect          │    │  + Multi-Granularity     │
│  URL Import  │    │  → YOLO → CLIP → Index  │    │  + Temporal Weighting    │
└──────────────┘    │  → Sprite Sheet          │    └─────────┬───────────────┘
                    │  → OCR → Diarization     │              │
                    └─────────────────────────┘              │
┌──────────────┐    ┌─────────────────────────────┐          │
│  Ask Q&A     │◀───│  Context Retrieval           │◀─────────┘
│  + Citations │    │  BGE-VL + Cross-Encoder      │
│  + Clip Export│   │  + TV-RAG Temporal Decay     │
└──────────────┘    └─────────────────────────────┘
```

## ✨ Features

- **🤖 Agentic RAG** — iterative retrieval loop with confidence-based early stopping across 3 rounds (standard → multi-hop → scene-graph expansion), inspired by Self-RAG and FLARE
- **🎬 Smart Video Analysis** — Scene detection, key frame extraction, transcription (faster-whisper), speaker diarization (PyAnnote), OCR text extraction (PaddleOCR), object detection (YOLO), semantic scene description (OpenCLIP), **zero-shot action recognition (X-CLIP)**
- **🧠 Dual-Backend Video MLLM** — SmolVLM2 (Apache 2.0, transformers-native, 2.2B/500M/256M) or VideoChat-Flash 2B (MIT, ICLR 2026) for video-native scene description, summarization, and Q&A
- **🌐 YouTube URL Import** — Download videos directly from YouTube, Vimeo, and other platforms via yt-dlp
- **📦 Batch Processing** — Queue videos by URL or file upload for sequential batch analysis
- **💬 AI Chatbot** — Ask questions about video content with timestamped source citations
- **🔍 RAG-Powered** — ChromaDB vector store + **BGE-VL-base multimodal embedding** (MIT, 150M params, ~0.8 GB VRAM) + embedding prefix normalization + cross-encoder re-ranking for state-of-the-art retrieval
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
├── PySceneDetect 0.7 ──→ Scene Detection
│   ├── AdaptiveDetector (default) — rolling HSV average
│   ├── ContentDetector — fixed-threshold HSV changes
│   ├── HistogramDetector — Y-channel histogram diffs
│   ├── HashDetector — perceptual hashing for similarity
│   └── FFmpeg fallback (gt(scene,...))
│              └── Per Scene: keyframe extraction
│                            ├── YOLO object detection
│                            ├── PaddleOCR text extraction
│                            ├── OpenCLIP zero-shot scene classification
│                            ├── X-CLIP zero-shot action recognition (optional)
│                            └── Frame metadata
├── FFmpeg ──→ Sprite sheet (100 thumbnails for timeline)
└── Merge ──→ Structured VideoIndex
              └── ChromaDB Vector Store (BGE embeddings)
```

### Query Pipeline

```
User Question
├── BGE-VL Multimodal Embedding (or SentenceTransformer fallback)
│   └── Query prefix normalization for text-only models
├── ChromaDB Hybrid Search (dense + metadata + chunk_type)
│   └── TV-RAG Temporal Decay (optional, score × exp(-λ·Δt))
├── Cross-Encoder Re-ranking (MS MARCO MiniLM)
├── Optional ColBERTv2 Late-Interaction Re-ranking
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
| `config` | `video_analysis/config.py` | Configuration with sensible defaults (auth, frame sampling, CLIP dedup) |
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
| **Scene Detection** | PySceneDetect 0.7+ (Adaptive/Content/Histogram/Hash) — or FFmpeg fallback |
| **Object Detection** | YOLO (ultralytics) | State-of-the-art speed/accuracy |
| **Scene Description** | OpenCLIP (ViT-B-32 / ViT-L-14) | Configurable model size, zero-shot classification |
| **Timeline Preview** | FFmpeg + Pillow sprite sheets | 100-thumbnail visual timeline navigation |
| **Vector Store** | ChromaDB | Persistent, local, no server needed |
| **Embeddings** | **BAAI/BGE-VL-base** (default, 150M, MIT, multimodal) + Nomic Embed v1.5 (fallback, text-only) | Single unified model for text/image/composed, ~0.8 GB VRAM |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM (default) + optional ColBERTv2 (RAGatouille) | Dual re-ranking for precision |
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
| `EMBEDDING_MODEL` | `BAAI/BGE-VL-base` | Primary embedding model (BGE-VL, MIT, multimodal) |
| `TEXT_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Fallback text-only embedding model |
| `TEMPORAL_DECAY_RATE` | `0.1` | TV-RAG temporal decay rate (0 = disabled) |
| `CLIP_MODEL` | `ViT-B-32` | OpenCLIP model size (ViT-B-32 or ViT-L-14) |
| `CLIP_PRETRAINED` | `laion2b_s34b_b79k` | OpenCLIP pretrained dataset |
| `SCENE_DETECTOR` | `adaptive` | Scene detection mode (adaptive/content/histogram/hash/ffmpeg) |
| `COLBERT_RERANKER_ENABLED` | `false` | Enable ColBERTv2 late-interaction re-ranking (requires ragatouille) |
| `OCR_ENABLED` | `true` | Enable PaddleOCR text extraction |
| `DIARIZE_ENABLED` | `true` | Enable PyAnnote speaker diarization |
| `YT_DLP_ENABLED` | `true` | Enable YouTube/URL video import |
| `UI_HOST` | `0.0.0.0` | Web UI bind address |
| `UI_PORT` | `7860` | Web UI port |
| `GRADIO_USER` | `admin` | UI auth username |
| `GRADIO_PASSWORD` | (unset) | UI auth password — set to enable authentication |
| `ADAPTIVE_FRAME_SAMPLING` | `false` | Enable motion-based adaptive frame sampling |
| `ADAPTIVE_FRAME_SAMPLING_SENSITIVITY` | `0.3` | Sampling density near scene boundaries |
| `CLIP_FRAME_DEDUP` | `false` | Enable CLIP-similarity frame deduplication |
| `CLIP_FRAME_DEDUP_THRESHOLD` | `0.92` | Similarity threshold for frame deduplication |
| `MULTIMODAL_EMBEDDING` | `false` | Enable Qwen3-VL-Embedding multimodal search (Apache 2.0) |
| `ACTION_RECOGNITION_ENABLED` | `false` | Enable X-CLIP zero-shot action recognition (requires transformers) |
| `ACTION_MODEL_NAME` | `microsoft/xclip-base-patch16-zero-shot` | X-CLIP model for action recognition |
| `VIDEO_MLLM_ENABLED` | `false` | Enable VideoChat-Flash 2B video MLLM (~5.4 GB VRAM) |
| `VIDEO_MLLM_MODEL` | `OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448` | Video MLLM model name |
| `VIDEO_MLLM_BACKEND` | `auto` | Video MLLM backend (auto/videochat_flash/smolvlm2) |
| `VIDEO_MLLM_MODEL_SIZE` | `2.2B` | SmolVLM2 model size (2.2B/500M/256M) |
| `VIDEO_MLLM_AS_DESCRIBER` | `false` | Use MLLM for scene descriptions (replaces OpenCLIP) |
| `VIDEO_MLLM_AS_CHAT_BACKEND` | `false` | Use MLLM as video-native Q&A backend |
| `AGENTIC_RETRIEVAL_ENABLED` | `false` | Enable agentic iterative retrieval loop |
| `AGENTIC_MAX_ROUNDS` | `3` | Max retrieval rounds in agentic loop |
| `AGENTIC_MIN_CONFIDENCE` | `0.5` | Min avg score of top-3 chunks to stop early |

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
- [x] PySceneDetect for improved scene boundaries (Adaptive + Content + Histogram + Hash)
- [x] OpenCLIP ViT-L-14 upgrade (richer scene descriptions)
- [x] FastAPI health endpoint and API
- [x] Embedding model upgrade (Nomic Embed v1.5)
- [x] Docker production hardening (CUDA 12.8, torch 2.6)
- [x] Frame preview on timeline hover (CSS sprite sheet overlay)
- [x] ColBERTv2 late-interaction re-ranking
- [x] Semantic video search (cross-video, multimodal — Qwen3-VL-Embedding + Video Search tab)
- [x] Gradio auth via env vars
- [x] Motion-based adaptive frame sampling
- [x] CLIP-similarity frame deduplication
- [x] Action recognition (X-CLIP — zero-shot open-vocabulary action detection, ~4GB VRAM)
- [x] **BGE-VL multimodal embedding** (replaces dual-model approach, MIT, ~0.8 GB VRAM)
- [x] **TV-RAG temporal-aware retrieval** (time-decay weighting, ACM Multimedia 2025)
- [x] **Multi-granularity chunking** (fixed-window 60s + sliding-window 30s + scene + frame)
- [x] **Systematic GPU memory management** (per-stage model unloading, 12 GB VRAM friendly)
- [x] **Graceful SIGTERM/SIGINT shutdown** (clean partial saves on termination)
- [x] **Production deployment** (DCGM GPU monitoring, Caddy reverse proxy)
- [x] Video MLLM integration (VideoChat-Flash 2B — optional scene describer + long-video Q&A + video-native chat backend)
- [x] Graph-based video RAG (VGent/ViG-RAG inspired — scene-graph retrieval + K-hop expansion)
- [x] Query classification & routing (text/visual/temporal modality dispatch)
- [x] Multi-hop query decomposition (sub-question → retrieve → reason)
- [x] **SmolVLM2 dual-backend** (Apache 2.0 — 2.2B, 500M, 256M video MLLM via transformers-native API)
- [x] **Agentic RAG** (iterative retrieval loop with confidence-based early stopping, 3-round strategy)
- [x] **CI/CD + pre-commit hooks** (GitHub Actions matrix build, ruff, mypy, benchmark infrastructure)
- [x] [RESEARCH] Entity tracking — ByteTrack/BoxMOT confirmed for persistent person/object IDs across scenes (~500 MB, integrates with YOLO)
- [x] [RESEARCH] Cross-video scene graph edges — scene_graph.py adjacency structure already supports cross-video keys; add entity-based + BGE-VL cross-video edges
- [x] [RESEARCH] Gradio 6 Workflow subgraphs — Gradio 6.19+ exposes composable subgraph API endpoints; FastAPI hybrid approach complements it
- [x] [RESEARCH] Sparse-frame optical flow — FFmpeg motion vectors (zero-cost) recommended over deep flow models for 12GB VRAM pipeline
- [ ] Entity-level tracking across scenes (ByteTrack/BoxMOT + persistent person/object IDs)
- [ ] Cross-video scene graph edges (multi-video semantic retrieval via entity + BGE-VL)
- [ ] Gradio 6 Workflow integration (expose pipeline stages as composable API subgraphs)
- [ ] Sparse-frame optical flow for motion-based adaptive frame sampling

## 📝 License

MIT
