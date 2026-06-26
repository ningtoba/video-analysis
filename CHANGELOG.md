# Changelog

## 0.4.0 (2026-06-26)

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
