# Changelog

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

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py        # v0.3.0
│   ├── config.py          # +ocr_enabled, diarize_enabled, ocr_confidence
│   ├── pipeline.py        # +_extract_ocr(), _diarize() methods
│   └── ...                # (models, rag, chat — enhanced for diarization/OCR)
├── ui/
│   └── app.py             # Library tab wired, timeline JS fixed, JS bridge
├── tests/
│   └── test_basic.py      # +tests for OCR config and diarization config
├── Dockerfile             # NEW — multi-stage CUDA Dockerfile
├── docker-compose.yml     # NEW — GPU passthrough compose
├── .dockerignore          # NEW
├── requirements.txt       # +paddleocr, pyannote.audio (optional)
├── pyproject.toml
├── README.md              # Updated
└── CHANGELOG.md
```

## 0.2.0 (2026-06-26)

### 🎬 New Features

- **Clip Export**: Export video clips at precise timestamps directly from the UI — select start/end times and export a trimmed MP4
- **📚 Video Library**: Multi-video management with library tab, refresh, and video info display
- **🖼️ Sprite Sheet Timeline Preview**: Automatic generation of 100-thumbnail sprite sheets for visual timeline browsing (pending UI integration)
- **🧠 OpenCLIP Zero-shot Classification**: Rich semantic scene descriptions (indoor/outdoor, interview, lecture, etc.) using OpenCLIP ViT-B-32 embeddings on each key frame — improves RAG context quality
- **🎛️ GPU Pipeline Management**: Sequential model loading/unloading to respect 12GB VRAM limits (Whisper → YOLO → CLIP models are loaded one at a time)

### 🔧 Improvements

- **Enhanced Data Models**: VideoIndex now tracks sprite sheet path and metadata; FrameInfo supports descriptions from CLIP classification
- **Better Progress Tracking**: Pipeline now reports 9 steps instead of 7 (added frame description + sprite sheet generation)
- **Expanded Test Suite**: 13 tests (up from 6) covering new models, config paths, and import sanity checks
- **Refined UI**: Tabs for Analysis and Library, clip export controls, progress panel reliable visibility toggling

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py         # +clip_export_dir, library_max_videos
│   ├── models.py         # +sprite_sheet, sprite_metadata on VideoIndex
│   ├── pipeline.py       # +sprite_sheet, export_clip, CLIP description
│   ├── rag.py            # (unchanged)
│   └── chat.py           # (unchanged)
├── ui/
│   └── app.py            # Tabs, clip export, library, enhanced progress
├── tests/
│   └── test_basic.py     # 13 tests
├── requirements.txt      # +open-clip-torch, pillow
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

## 0.1.0 (2026-06-26)

### Initial Release

- **Core pipeline**: FFmpeg-based scene detection, frame extraction, faster-whisper transcription, YOLO object detection
- **RAG engine**: ChromaDB vector store with hybrid BM25/dense retrieval, cross-encoder re-ranking, temporal context expansion
- **Chat interface**: Video Q&A with source citations (clickable timestamps), conversation history
- **Web UI**: Gradio Blocks with dark theme, video upload, real-time analysis progress, streaming chat
- **CLI mode**: Batch processing and Q&A from the terminal
- **GPU acceleration**: Full CUDA support for RTX 4070 (faster-whisper, sentence-transformers, torch)
- **All local**: No API keys required — runs entirely on self-hosted hardware
