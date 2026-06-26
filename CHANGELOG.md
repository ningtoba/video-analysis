# Changelog

## 0.6.0 (2026-06-26)

### 🧠 New Feature: Optional ColBERTv2 Late-Interaction Re-Ranker

- **ColBERTv2 Integration**: New optional `ColBERTReranker` module at `video_analysis/colbert_reranker.py` wraps RAGatouille (AnswerDotAI) for token-level late-interaction re-ranking. Improves retrieval precision for complex queries by matching individual tokens rather than whole vectors.
- **Config toggle**: `colbert_reranker_enabled: bool = False` in `video_analysis/config.py` — set to `True` to enable. Falls back gracefully to the cross-encoder if RAGatouille is not installed.
- **VRAM efficient**: Lazy-loads ColBERTv2 (~2-3 GB VRAM), runs re-ranking, then unloads to free GPU memory. Compatible with 12 GB RTX 4070 sequential model loading.
- **Optional dependency**: `ragatouille>=1.0.0` commented out in `requirements.txt` — install when needed with `pip install ragatouille`.

### 🎬 Timeline Hover Preview: Gradio 6 Shadow DOM Fix

- **Shadow DOM penetration**: Rewrote the JavaScript timeline preview (`ui/app.py:1381-1571`) to use a recursive shadow DOM traversal (`findVideoElements()`) instead of the broken `.gradio-video video` CSS selector. Works with Gradio 6's LitElement-based Web Components where `<video>` lives inside a Shadow DOM.
- **Visibility-aware detection**: `scanForVideo()` filters hidden video elements by checking `getBoundingClientRect()` — only attaches to the visible video player.
- **Graceful tab switching**: Periodic 2-second polling re-scans when Gradio lazy-renders tabs, ensuring the preview attaches to newly loaded video players without manual refresh.
- **Cleaner code**: Removed brittle `video.closest('gradio-video')` call that failed when the video was nested in a shadow root. Now walks up from the `<video>` element to find the first non-video container.

### 🔧 Improvements

- **.dockerignore fix**: Removed the blanket `*.md` exclusion pattern that was accidentally excluding `README.md` and `CHANGELOG.md` from the Docker build context. Now explicitly lists only research documents for exclusion, preserving README and CHANGELOG in the image.
- **Health model check fix**: The `health.py` module was trying to `import whisper` instead of `faster_whisper` in the model check — now correctly reflects the actual dependency.

### 📦 Dependencies

- **Optional**: `ragatouille>=1.0.0` — ColBERTv2 late-interaction re-ranking (commented out, install on demand)

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py              # v0.6.0
│   ├── colbert_reranker.py      # NEW — ColBERTv2 late-interaction re-ranker
│   ├── config.py                # +colbert_reranker_enabled
│   └── rag.py                   # +_rerank_colbert() method
├── ui/
│   └── app.py                   # Shadow DOM JS for timeline preview
├── .dockerignore                # Fixed: no longer excludes README/CHANGELOG
├── requirements.txt             # +ragatouille optional dep (commented)
├── pyproject.toml               # v0.6.0
├── README.md                    # Updated roadmap
└── CHANGELOG.md
```

## 0.5.0 (2026-06-26)

### 🎬 New Features

- **🧠 OpenCLIP ViT-L-14 Support**: Configurable CLIP model size — switch between ViT-B-32 (default, fast) and ViT-L-14 (richer scene descriptions, +3% accuracy). New config fields: `clip_model`, `clip_pretrained_dataset`, `clip_embed_dim`. ViT-L-14 uses `laion2b_s32b_b82k` pretrained weights.
- **🔍 Enhanced Scene Detection**: New `"histogram"` and `"hash"` detector modes in addition to `"adaptive"`, `"content"`, and `"ffmpeg"`. HistogramDetector uses Y-channel histogram differences for fast cuts; HashDetector uses perceptual hashing for similarity-based scene boundary detection.
- **🏥 Health Endpoint & API**: FastAPI `/health` endpoint with GPU availability, model status, version, and uptime. API endpoints at `/api/library` and `/api/video/{video_id}` for programmatic access. Gradio app mounts on FastAPI using `gr.mount_gradio_app()`.
- **⬆️ Embedding Model Upgrade**: Default embedding model changed to `nomic-ai/nomic-embed-text-v1.5` (768-dim, Apache 2.0, MTEB ~64) — significantly better retrieval quality vs previous BGE-small (384-dim, MTEB ~50).

### 🔧 Improvements

- **Docker Production Ready**: Updated to CUDA 12.8 runtime with torch 2.6 wheels. HEALTHCHECK now uses proper `/health` endpoint. Docker Compose exposes port 7861 for health API.
- **Pipeline Cleanup**: Improved model loading with configurable CLIP model size, pretrained dataset selection, and batch inference. The `_describe_scenes_clip` method now reads `clip_model` and `clip_pretrained_dataset` from config instead of hardcoded values.
- **Test Suite**: Added 7 new tests covering CLIP config fields, scene detector options, embedding model defaults, health module import.

### 📦 Dependencies

- **Updated**: `open-clip-torch>=2.24.0` (supports ViT-L-14 via pretrained flag)
- **Updated**: `sentence-transformers>=3.0.0` (recommended for nomic-embed)
- **Updated**: `scenedetect>=0.7.0` (now explicitly uncommented in requirements.txt)
- **Updated**: CUDA stacks upgraded from 12.4 to 12.8, torch from 2.1 to 2.6

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py        # v0.5.0
│   ├── config.py          # +clip_model, clip_pretrained_dataset, clip_embed_dim
│   └── pipeline.py        # +histogram/hash scene detectors, configurable CLIP model
├── ui/
│   ├── health.py          # NEW — FastAPI health/API endpoint
│   └── app.py             # +FastAPI mounting, /health endpoint wiring
├── tests/
│   └── test_basic.py      # +7 tests for new config fields
├── Dockerfile             # CUDA 12.8, torch 2.6, /health healthcheck
├── docker-compose.yml     # +7861 port, updated healthcheck
├── requirements.txt       # scenedetect uncommented
├── pyproject.toml         # v0.5.0
├── README.md              # Updated with new features
└── CHANGELOG.md
```

### 🎬 New Features

- **🌐 YouTube URL Import**: Download and analyze videos directly from YouTube, Vimeo, and other platforms via yt-dlp integration. Paste any URL in the UI or use `--url` in CLI mode.
- **📦 Batch Processing Queue**: New batch processing tab allows queuing multiple videos (by URL or file upload) for sequential analysis. Batch mode also available via `--batch urls.txt` in CLI.
- **🗂️ UI Utils Module**: Extracted `parse_yt_url()` and `queue_html()` into `ui/utils.py` — importable without gradio dependency, enabling proper unit testing of UI logic.

### 🔧 Improvements

- **Timeline Hover Preview JS Fix**: Enhanced the JavaScript timeline preview with proper CSS positioning, multiple sprite URL fallback paths, and fixed floating-point hover card rendering. Preview now shows thumbnail + timestamp on timeline hover.
- **CLI Enhancements**: Added `--url` flag for YouTube downloads, `--batch` flag for processing from a file list, and improved error handling.
- **Config**: New `yt_dlp_enabled`, `yt_dlp_format`, `yt_dlp_output_template`, and `batch_concurrent` configuration fields.

### 📦 Dependencies

- **New**: `yt-dlp>=2024.0.0` — YouTube/URL video import and batch processing

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py        # v0.4.0
│   ├── config.py          # +yt_dlp_enabled, yt_dlp_format, batch_concurrent
│   ├── pipeline.py        # +download_from_url() static method
│   └── ...                # (models, rag, chat — unchanged)
├── ui/
│   ├── app.py             # +YouTube import, batch tab, enhanced timeline JS
│   ├── utils.py           # NEW — importable utility functions
│   └── ...
├── tests/
│   └── test_basic.py      # +5 tests: yt-dlp import, download fallback, URL parsing, queue HTML, config fields
├── Dockerfile             # v0.4.0 label
├── requirements.txt       # +yt-dlp
├── pyproject.toml         # v0.4.0
├── README.md              # Updated with new features
└── CHANGELOG.md
```

## 0.3.0 (2026-06-26)

### 🎬 New Features

- **🗣️ Speaker Diarization**: Automatic speaker labeling via PyAnnote Audio (`pyannote/speaker-diarization-3.1`). Each transcript segment now gets a `SPEAKER_00`, `SPEAKER_01`, etc. label, enabling speaker-aware Q&A. Configurable via `diarize_enabled`. Graceful fallback if PyAnnote is not installed.
- **🔤 OCR Text Extraction**: On-screen text detection via PaddleOCR (CPU mode). Extracts text from key frames and stores in `FrameInfo.ocr_text`. Visible in RAG context and Q&A responses. Configurable via `ocr_enabled` and `ocr_confidence`.
- **🐳 Docker Deployment**: Complete Dockerfile (multi-stage, CUDA 12.4 runtime) and docker-compose.yml with GPU passthrough, health checks, persistent volumes, and Nvidia container toolkit support.
- **📚 Library Tab Video Player**: Library cards are now clickable — clicking a video in the library loads it in a video player with metadata display. JS bridge (`window.__selectVideo`) connects Gradio UI to the library backend.

### 🔧 Improvements

- **Timeline Hover Preview Fix**: Rewrote the JavaScript timeline hover detection to work with Gradio 6's `<gradio-video>` web component. Now detects hover on the video container's bottom area rather than relying on the non-existent `<input type="range">` element.
- **Config Flags**: New `ocr_enabled`, `diarize_enabled`, `ocr_confidence` config fields for fine-grained pipeline control.
- **Pipeline Step Count**: 12 pipeline steps (up from 9) — added OCR extraction and speaker diarization.

### 📦 Dependencies

- **New optional**: `paddleocr>=2.8.0` — OCR text extraction
- **New optional**: `pyannote.audio>=3.1.0` — Speaker diarization
- Both are optional with graceful fallbacks if not installed.

## 0.2.0 (2026-06-26)

### 🎬 New Features

- **Clip Export**: Export video clips at precise timestamps directly from the UI — select start/end times and export a trimmed MP4
- **📚 Video Library**: Multi-video management with library tab, refresh, and video info display
- **🖼️ Sprite Sheet Timeline Preview**: Automatic generation of 100-thumbnail sprite sheets for visual timeline browsing
- **🧠 OpenCLIP Zero-shot Classification**: Rich semantic scene descriptions (indoor/outdoor, interview, lecture, etc.) using OpenCLIP ViT-B-32 embeddings on each key frame — improves RAG context quality
- **🎛️ GPU Pipeline Management**: Sequential model loading/unloading to respect 12GB VRAM limits

## 0.1.0 (2026-06-26)

### Initial Release

- **Core pipeline**: FFmpeg-based scene detection, frame extraction, faster-whisper transcription, YOLO object detection
- **RAG engine**: ChromaDB vector store with hybrid BM25/dense retrieval, cross-encoder re-ranking, temporal context expansion
- **Chat interface**: Video Q&A with source citations (clickable timestamps), conversation history
- **Web UI**: Gradio Blocks with dark theme, video upload, real-time analysis progress, streaming chat
- **CLI mode**: Batch processing and Q&A from the terminal
- **GPU acceleration**: Full CUDA support for RTX 4070
- **All local**: No API keys required — runs entirely on self-hosted hardware
