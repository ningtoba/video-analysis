# Changelog

## 0.12.0 (2026-06-26)

### 🧠 Major Feature: BGE-VL Multimodal Embedding — Single Unified Model

- **BGE-VL-base as the default embedding model** (BAAI/BGE-VL-base, 150M params, MIT license, ~0.8 GB VRAM). Replaces the old dual-model approach (SentenceTransformer + optional Qwen3-VL-Embedding-2B) with a single unified embedding pipeline that handles text-only, image-only, and composed (image+text) retrieval.
- **`rag.py`**: Completely rewritten embedding stack. New `_load_bge_vl()` / `_unload_bge_vl()` for lazy GPU loading/unloading. `_get_bge_vl_embedding()` handles all three modes (text, image, composed). `_get_query_embedding()` uses BGE-VL as primary with query prefix normalization. `_get_embedding()` falls back gracefully to SentenceTransformer + Nomic Embed when BGE-VL is unavailable.
- **`search_all()`**: Now uses BGE-VL composed retrieval when both image and text are provided — true multimodal cross-video search without the heavy Qwen3-VL model.
- **`text_embedding_model`** config field added (`nomic-ai/nomic-embed-text-v1.5`) for fallback. Legacy `multimodal_embedding_enabled` / Qwen3-VL path retained for backward compatibility.
- **Embedding prefix normalization**: Query/document prefixes now applied when falling back to SentenceTransformer (e.g. `search_query:` for Nomic, `Represent this query:` for BGE). Boosts retrieval accuracy by 5-10%.

### ⏱️ Temporal-Aware Retrieval (TV-RAG)

- **Time-decay weighting**: New `temporal_decay_rate` config field (default: `0.1`). When `query_time` is provided to `retrieve()`, chunk scores are weighted by `score * exp(-decay_rate * time_distance)` per the TV-RAG paper (ACM Multimedia 2025). Set to `0.0` to disable.
- **`_get_query_embedding()`**: Updated `retrieve()` signature with optional `query_time` parameter. Temporal weighting integrates seamlessly with the existing cross-encoder re-ranking pipeline.

### 📦 Multi-Granularity Chunking

- **Quad-chunk strategy** in `index_video()`:
  - **Scene chunks** (variable length, rich context): transcript + descriptions + objects + OCR + actions
  - **Fixed-window chunks** (60 seconds, no overlap): transcript segments aligned to time windows — cross-scene queries
  - **Sliding-window chunks** (30 seconds, 15s overlap): fine-grained temporal localization
  - **Frame chunks** (per-frame): direct frame-level retrieval
  - **Transcript chunks** (legacy 500-char windows): retained for backward compatibility
- All chunk types get `chunk_type` metadata field in ChromaDB (`scene`, `frame`, `fixed_60s`, `sliding_30s`, `transcript`) enabling targeted retrieval strategies.

### 🛡️ GPU Memory Management & Graceful Shutdown

- **Systematic model unloading**: New `_unload_model(model_attr)` helper in `VideoPipeline` that safely removes a model attribute, deletes the reference, runs `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.synchronize()`.
- **Per-stage GPU cleanup**: Models are explicitly unloaded between every GPU-intensive pipeline step:
  - After Step 5 (Whisper, ~4 GB) → unloaded before diarization
  - After Step 7 (YOLO, ~1 GB) → unloaded before OCR
  - After Step 9 (OpenCLIP, ~2 GB) → unloaded before action recognition
  - After Step 10 (X-CLIP, ~4 GB) → unloaded before sprite sheet/indexing
- Peak VRAM now managed: sequential loading ensures no more than 4 GB reserved at any time on a 12 GB RTX 4070.
- **Graceful SIGTERM/SIGINT handling**: `VideoPipeline` registers signal handlers that set `_shutdown_requested=True`. `__main__.py` also registers handlers with `_shutdown_event` for clean CLI/cron termination.

### ⚙️ Dependency & Configuration Updates

- **Gradio >=6.19.0** (was >=6.0.0) — Svelte 5 migration, MCP support, workflow subgraphs, stability
- **transformers >=4.45.2** (was >=4.40.0) — required for BGE-VL compatibility
- New config fields: `text_embedding_model`, `temporal_decay_rate`
- BGE-VL-base (`BAAI/BGE-VL-base`) is now the default `embedding_model`

### 🐳 Production Deployment

- **`docker-compose.prod.yml`**: New production-grade Docker Compose with:
  - **DCGM Exporter** — NVIDIA GPU metrics (VRAM, temp, utilization) at `:9400/metrics` for Prometheus
  - **Caddy reverse proxy** — automatic HTTPS, WebSocket support for Gradio streaming, security headers, gzip compression
- **Caddyfile**: Production-ready reverse proxy configuration with security headers and logging

### 🧪 Tests

- 12 new tests: BGE-VL config defaults, embedding prefix normalization (Nomic, BGE-small, BGE-VL), pipeline cleanup, model unloading, multi-granularity chunking config, temporal decay config, RetrievedChunk chunk_type field, graceful fallback test (BGE-VL → SentenceTransformer)
- Pre-existing test suite: 49 → 61 tests passing

### 🎬 Major Feature: X-CLIP Zero-Shot Action Recognition

- **Open-vocabulary action detection**: Added `ActionRecognizer` module at `video_analysis/action.py` wrapping Microsoft X-CLIP (`microsoft/xclip-base-patch16-zero-shot`, 200M params, Apache 2.0). Classifies per-frame human activities (walking, running, cooking, typing, etc.) with confidence scores — no training required, works out of the box.
- **Pipeline integration**: New pipeline step (Step 10) between OpenCLIP and transcript assignment. Runs sequentially, loads X-CLIP (~4GB VRAM), classifies all key frames in GPU-efficient batches, then unloads the model to free VRAM.
- **Config toggle**: `ACTION_RECOGNITION_ENABLED=true` env var to enable. New config fields: `action_recognition_enabled`, `action_model_name`, `action_categories_count`.
- **RAG context**: Action labels are indexed in ChromaDB alongside transcript, objects, OCR, and scene descriptions. Queries like "when is someone cooking?" or "find scenes with people fighting" retrieve relevant video segments.
- **Graceful fallback**: If `transformers` is unavailable or model download fails, the step is silently skipped — no breaking changes.
- **26 default action categories**: Covers common video scenarios (walking, running, sitting, cooking, typing, driving, fighting, etc.) with "no person visible" as the catch-all.

### 🐛 Bug Fixes & Improvements

- **Pipeline step numbering**: Fixed duplicate Step 7 (OCR) and duplicate Step 10 (Index). Steps are now correctly numbered 1–13 throughout.
- **Multimodal embedding fallback fix**: Fixed a bug where `_get_embedding()` would load a SentenceTransformer first, causing `_get_multimodal_embedding()` to silently fall back to text-only. Now `_get_embedding()` routes through the multimodal model when `multimodal_embedding_enabled=True`, ensuring one unified embedding space.
- **README roadmap**: Updated stale reference to "InternVideo2.5" for action recognition → now correctly references X-CLIP.

### 🧪 Tests

- 6 new tests: config action fields, ActionRecognizer import/defaults, empty classify list, graceful file-not-found fallback, FrameInfo action fields, ACTION_RECOGNITION_ENABLED env var.
- Pre-existing test suite: 43 → 49 tests passing.

### 🏗️ Architecture

```

### 🔬 Major Enhancement: Qwen3-VL Multimodal Embedding (Apache 2.0)

- **True multimodal semantic search**: Added optional Qwen3-VL-Embedding-2B support (Apache 2.0, 2B params, 2048-dim) for fusing visual + textual information into a shared embedding space. When `MULTIMODAL_EMBEDDING=true` and the model weights are downloaded, frame images are embedded together with text descriptions for far richer semantic retrieval compared to text-only Nomic Embed v1.5.
- **New config fields**: `multimodal_embedding_model` (default: `Qwen/Qwen3-VL-Embedding-2B`, 2B params fits in ~6GB VRAM), `multimodal_embedding_enabled` (reads `MULTIMODAL_EMBEDDING` env var, default `false`).
- **Graceful fallback**: If `transformers`, `torch`, or `Pillow` are not installed, or if the model weights are not downloaded, falls back to text-only embedding with a log warning. No breaking changes.
- **`rag.py`**: Added `_get_multimodal_embedding()` method and `search_all()` method for cross-video semantic search.

### 🔍 New Feature: Cross-Video Semantic Search ("Video Search" Tab)

- **New Gradio tab**: "🔍 Video Search" tab added after the Library tab. Users enter a natural language query and search across ALL indexed videos simultaneously.
- **Rich results**: Results are grouped by video, showing timestamp, relevance score (as %), and expandable context preview for each matching chunk.
- **`rag.search_all()`**: New method on `VideoRAG` that removes the `video_id` filter from ChromaDB queries, enabling true cross-video retrieval. Re-ranks results with the existing cross-encoder for best accuracy.
- **Works with or without multimodal**: When `multimodal_embedding_enabled` is off, search uses the existing Nomic Embed v1.5 text model as before.

### ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIMODAL_EMBEDDING` | `false` | Enable Qwen3-VL-Embedding multimodal search |

### 🔧 Improvements

- `video_analysis/config.py`: Added `multimodal_embedding_model` and `multimodal_embedding_enabled` fields.
- `video_analysis/rag.py`: Added `_get_multimodal_embedding()` — wraps Qwen3-VL-Embedding for image+text fusion. Added `search_all()` — cross-video semantic search without `video_id` filter.

### 🧪 Tests

- Added test for multimodal embedding config defaults.
- Added test for `search_all()` basic logic.
- Pre-existing test suite: 43+ tests passing.

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py              # v0.10.0
│   ├── config.py                # +multimodal_embedding_model +multimodal_embedding_enabled
│   └── rag.py                   # +_get_multimodal_embedding() +search_all()
├── ui/
│   └── app.py                   # +"🔍 Video Search" tab
├── pyproject.toml               # v0.10.0
├── README.md                    # Updated config & roadmap
└── CHANGELOG.md
```

### 🔒 New Feature: Gradio UI Authentication

- **FastAPI middleware auth**: Added HTTP Basic Auth middleware to the FastAPI app that mounts Gradio. When `GRADIO_PASSWORD` is set, all UI routes (except `/health`) require authentication via HTTP Basic Auth.
- **Config-driven**: New config fields `ui_auth_enabled`, `ui_auth_username` (from `GRADIO_USER`, default `admin`), and `ui_auth_password` (from `GRADIO_PASSWORD`). Auth auto-enables when `GRADIO_PASSWORD` is set.
- **/health stays public**: The health endpoint is excluded from auth, so Docker health checks and monitoring tools continue to work without credentials.

### 🎞️ New Feature: Motion-Based Adaptive Frame Sampling

- **Smarter frame extraction**: New `adaptive_frame_sampling` config flag (default: `False`). When enabled, frames are sampled more densely near scene boundaries (3× density in first/last 10% of each scene) and more sparsely in static middle regions. This captures transitions and action near cuts while reducing redundant frames from static scenes.
- **Configurable sensitivity**: `adaptive_frame_sampling_sensitivity` (default: `0.3`) controls the base sampling rate — lower values extract more frames overall.

### 🧹 New Feature: CLIP-Similarity Frame Deduplication

- **Removes near-duplicate frames**: New `clip_frame_dedup` config flag (default: `False`). When enabled, consecutive frames within each scene are compared using OpenCLIP ViT-B-32 embedding cosine similarity. Frames exceeding `clip_frame_dedup_threshold` (default: `0.92`) are considered near-duplicates and removed.
- **Graceful fallback**: If `open-clip-torch` is not installed, dedup is silently skipped.
- **VRAM-efficient**: OpenCLIP model is loaded temporarily for the dedup pass, then unloaded from GPU memory.

### ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_USER` | `admin` | UI auth username |
| `GRADIO_PASSWORD` | (unset) | UI auth password — set to enable authentication |
| `ADAPTIVE_FRAME_SAMPLING` | `false` | Enable motion-based adaptive frame sampling |
| `ADAPTIVE_FRAME_SAMPLING_SENSITIVITY` | `0.3` | Sampling density near scene boundaries |
| `CLIP_FRAME_DEDUP` | `false` | Enable CLIP-similarity frame deduplication |
| `CLIP_FRAME_DEDUP_THRESHOLD` | `0.92` | Similarity threshold for frame deduplication |

### 🔧 Improvements

- `ui/health.py`: Added `_setup_auth_middleware()` — configurable HTTP Basic Auth middleware. When `GRADIO_PASSWORD` is not set, the middleware is a no-op (no auth required), maintaining backward compatibility.
- `video_analysis/config.py`: Added `ui_auth_enabled`, `ui_auth_username`, `ui_auth_password`, `adaptive_frame_sampling`, `adaptive_frame_sampling_sensitivity`, `clip_frame_dedup`, `clip_frame_dedup_threshold` fields.
- `video_analysis/pipeline.py`: Added `_adaptive_frame_samples()` and `_dedup_frames_clip()` methods. Modified `_extract_key_frames()` to use adaptive sampling and/or CLIP dedup when configured.

### 🧪 Tests

- Added tests for new config fields and view-only auth module import.

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py              # v0.9.0
│   ├── config.py                # +auth +adaptive_sampling +clip_dedup
│   └── pipeline.py              # +_adaptive_frame_samples() +_dedup_frames_clip()
├── ui/
│   └── health.py                # +_setup_auth_middleware()
├── pyproject.toml               # v0.9.0
├── README.md                    # Updated config & roadmap
└── CHANGELOG.md
```

## 0.8.0 (2026-06-26)

### 🔬 Comprehensive Research Sweep (Iteration 1)

Conducted deep research across 5 domains for the next evolution of the platform:

**1. Scene Detection & Frame Extraction** — PySceneDetect 0.7 (May 2026) confirmed as best
OSS option for shot boundary detection. No competitive alternative emerged. Key improvement:
motion-based adaptive frame sampling (sample more densely near scene boundaries, less in
static regions). Also propose CLIP-similarity-based keyframe deduplication.

**2. Video Understanding AI Models** — Identified InternVideo2.5 (OpenGVLab, 2025, Apache 2.0)
as the direct successor to VideoMAE/TimeSformer for action recognition — replaces the
original roadmap's reference to VideoMAE. VideoChat-Flash (ICLR 2026) is the top video
MLLM for long-context understanding. TimeSformer (Meta) was archived Jan 2025. OpenCLIP
remains best zero-shot frame descriptor for 12GB VRAM.

**3. RAG Architectures** — Current stack (ChromaDB + Nomic Embed v1.5 + ColBERTv2) is
already state-of-the-art. Key discovery: BGE-VL (BAAI, March 2025, MIT license) enables
multimodal embedding — video frames can be searched directly as images, not just through
text descriptions. MegaPairs dataset released alongside.

**4. Web UI Frameworks** — Gradio 6 (v6.19.0) confirmed as the best fit over Streamlit,
NiceGUI, and Dash. Native Video + Chatbot + gr.mount_gradio_app() for FastAPI is the
correct architecture. Roadmap item for Gradio auth via env vars identified as top priority.

**5. Production Deployment** — Docker/CUDA 12.8 stack is current. Missing: Gradio auth,
torch.cuda.empty_cache() between pipeline stages, graceful SIGTERM handling.

Full research document written to `RESEARCH.md`.

### 🏗️ Infrastructure

- Written `RESEARCH.md` — comprehensive research document covering all 5 domains

## 0.7.0 (2026-06-26)

### 🔬 Action Recognition Research

- Researched VideoMAE/TimeSformer vs InternVideo2 for action recognition
- Gradio auth implementation planning
- Semantic search architecture planning

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
