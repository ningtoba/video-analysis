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

- **🌐 Full REST API** (v0.41.0) — comprehensive HTTP API with 10+ endpoints for video processing, Q&A with SSE streaming, transcript/chapter retrieval, frame extraction, and cross-video search; auto-generated OpenAPI docs at `/docs`
|- **📷 Webcam Capture** (v0.41.0) — real-time webcam capture and frame analysis tab in the Gradio UI; supports live preview, capture & analyze, and continuous monitoring mode
|- **🧠 MLLM Streaming Q&A** (v0.41.0) — token-by-token SSE streaming for LLM responses from both Hermes CLI and OpenAI-compatible backends; enables real-time chat updates in the Gradio UI and REST API
|- **📡 Live Stream Analysis** (v0.40.0) — capture and analyze live RTMP/RTSP/HLS streams in real-time with auto-reconnect, sliding window context, and incremental indexing; connect OBS, IP cameras, and streaming platforms directly to the analysis pipeline
|- **🤖 Agentic RAG** — iterative retrieval loop with confidence-based early stopping across 4 rounds (standard → multi-hop → scene-graph → LLM self-check verification with re-retrieval), inspired by Self-RAG, FLARE, and CRAG
|- **🤖 Agentic Video Agent** (v0.36.0) — multi-tool video understanding agent with 7 specialized tools (analyze_frames, detect_objects, OCR, search_transcript, search_rag, temporal_grounding, summarize_video) that dynamically routes questions to the right tools
|- **📖 Video Chaptering** (v0.37.0) — automatic topic segmentation of transcripts into chapters using NLTK TextTiling, with LLM-generated chapter titles and summaries; generates structured chapter reports and agent-chapter context
- **🎯 MMR Diversity Re-Ranking** — Maximal Marginal Relevance (Carbonell & Goldstein, SIGIR'98) reduces context redundancy by 30-50% over pure relevance-sorted retrieval; configurable via `MMR_DIVERSITY_ENABLED`, `MMR_LAMBDA`, and `MMR_TOP_K`
- **🎬 Smart Video Analysis** — Scene detection, key frame extraction, transcription (faster-whisper), speaker diarization (PyAnnote), OCR text extraction (PaddleOCR PP-OCRv6 — +4.6% detection, +5.1% recognition over v5), object detection (YOLO), semantic scene description (OpenCLIP), **zero-shot action recognition (X-CLIP)**, **DINOv2 perceptual frame compression (LongVU-style)**
|- **🧠 Dual-Backend Video MLLM** — SmolVLM2 (Apache 2.0, transformers-native, 2.2B/500M/256M) or VideoChat-Flash 2B (MIT, ICLR 2026) or **Qwen3-VL-30B-A3B (Apache 2.0, MoE 30B/3B active, FP8, 128K context via vLLM/production server)** for video-native scene description, summarization, and Q&A
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
- **🧑‍🤝‍🧑 Face Recognition** — InsightFace (SCRFD-10G + ArcFace W50) for face detection, 512-d embeddings, and cross-video person identity matching (optional, ~1.1 GB VRAM)
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

# Stream a video in chunks (low-latency first results)
python -m video_analysis --stream --video my_video.mp4 --chunk-duration 30

# Analyze a live RTMP stream (e.g. OBS, Twitch)
python -m video_analysis --live-stream rtmp://example.com/live/stream --chunk-duration 30

# Analyze an RTSP camera feed (security camera, NVR)
python -m video_analysis --live-stream rtsp://192.168.1.100:554/stream1 --source-type rtsp

# Analyze an HLS stream with max chunk limit
python -m video_analysis --live-stream https://cdn.example.com/live/stream.m3u8 --source-type hls --max-chunks 10
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
│                            ├── InsightFace face detection (optional)
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
| `face_recognition` | `video_analysis/face.py` | InsightFace face detection & recognition — DetectedFace, FaceRecognizer, clustering |
| `ui/app` | `ui/app.py` | Gradio web interface with dark theme, tabs, library, clip export, batch queue, URL import |
| `ui/utils` | `ui/utils.py` | Shared UI utility functions (importable without gradio) |
| `ui/workflow` | `ui/workflow.py` | Gradio 6 Workflow visual pipeline builder (gr.Workflow canvas) |
| `streaming` | `video_analysis/streaming.py` | Real-time streaming/chunked video analysis (StreamingVLM-inspired) |
| `federation` | `video_analysis/federation.py` | Federated MCP-based cross-instance video search |
| `backends` | `video_analysis/backends/` | MLLM backend implementations (Qwen3-VL-30B-A3B with vLLM + FP8) |
| `agent` | `video_analysis/agent.py` | Agentic Video Understanding Agent — multi-tool video analysis agent |
| `chapters` | `video_analysis/chapters.py` | Video content chaptering — topic segmentation & LLM chapter title generation |

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
| **Face Recognition** | InsightFace (SCRFD-10G + ArcFace W50) | Cross-video person identity, 512-d embeddings, ~1.1 GB VRAM (optional) |
| **Scene Description** | OpenCLIP (ViT-B-32 / ViT-L-14) | Configurable model size, zero-shot classification |
| **Timeline Preview** | FFmpeg + Pillow sprite sheets | 100-thumbnail visual timeline navigation |
| **Vector Store** | ChromaDB | Persistent, local, no server needed |
| **Embeddings** | **BAAI/BGE-VL-base** (default, 150M, MIT, multimodal) + Nomic Embed v1.5 (fallback, text-only) | Single unified model for text/image/composed, ~0.8 GB VRAM |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM (default) + optional ColBERTv2 (RAGatouille) | Dual re-ranking for precision |
| **Video Import** | yt-dlp | Downloads from YouTube, Vimeo, Twitch, and 1000+ sites |
| **LLM** | DeepSeek-V4-Flash (via Hermes CLI) or any OpenAI-compatible API (vLLM, Ollama, llama.cpp, TGI) via `LLM_PROVIDER=openai`
| **GPU** | RTX 4070 (CUDA 13.3) | All models run with GPU acceleration |
| **Live Stream** | FFmpeg `-re` capture | RTMP/RTSP/HLS with auto-reconnect and sliding window (v0.40.0) |

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
| `VIDEO_MLLM_ENABLED` | `false` | Enable VideoChat-Flash 2B / Qwen3-VL-30B-A3B video MLLM (~5.4 GB VRAM) |
| `VIDEO_MLLM_MODEL` | `OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448` | Video MLLM model name (overridable, e.g. Qwen/Qwen3-VL-30B-A3B-Instruct-FP8) |
| `VIDEO_MLLM_BACKEND` | `auto` | Video MLLM backend (auto/videochat_flash/smolvlm2/qwen3_vl) |
| `VIDEO_MLLM_MODEL_SIZE` | `2.2B` | SmolVLM2 model size (2.2B/500M/256M) |
| `VIDEO_MLLM_AS_DESCRIBER` | `false` | Use MLLM for scene descriptions (replaces OpenCLIP) |
| `VIDEO_MLLM_AS_CHAT_BACKEND` | `false` | Use MLLM as video-native Q&A backend |
| `LLM_PROVIDER` | `hermes` | LLM backend (hermes, openai, auto) — v0.39.0 |
| `OPENAI_API_BASE` | `http://localhost:11434/v1` | OpenAI-compatible API URL |
| `OPENAI_API_KEY` | (empty) | API key (can be empty for local servers) |
| `OPENAI_MODEL` | `qwen2.5` | Model name for the OpenAI-compatible API |
| `AGENTIC_RETRIEVAL_ENABLED` | `false` | Enable agentic iterative retrieval loop |
| `AGENTIC_MAX_ROUNDS` | `4` | Max retrieval rounds in agentic loop |
| `AGENTIC_MIN_CONFIDENCE` | `0.5` | Min avg score of top-3 chunks to stop early |
| `PROCESSING_MODE` | `video_full` | Processing mode: video_full or audio_only |
| `CONVERSATION_MEMORY_ENABLED` | `true` | Enable ChromaDB-backed conversation memory |
| `CONVERSATION_MEMORY_MAX_ENTRIES` | `50` | Max conversation memory entries |
| `CONVERSATION_MEMORY_TTL_DAYS` | `30` | Entry TTL in days |
| `STRUCTURED_LOGGING_ENABLED` | `true` | Enable structlog-based structured logging |
| `STRUCTURED_LOGGING_FORMAT` | `auto` | Output format: auto, console, json |
| `STRUCTURED_LOGGING_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |
| `FACE_RECOGNITION_ENABLED` | `false` | Enable InsightFace face detection & recognition (requires insightface + onnxruntime-gpu) |
| `DINO_FRAME_COMPRESSION` | `false` | Enable DINOv2 perceptual frame compression (LongVU-style, ~85 MB VRAM) |
| `DINO_FRAME_COMPRESSION_THRESHOLD` | `0.88` | Cosine sim threshold [0,1]; lower = more aggressive compression |
| `DINO_FRAME_COMPRESSION_MODEL` | `facebook/dinov2-small` | DINOv2 variant (small=21M, base=86M) |
| `COLBERT_ATT_RERANKER_ENABLED` | `false` | Enable ColBERT-Att attention-weighted re-ranking (arXiv:2603.25248, +1-3% recall) |
| `FACE_DETECTION_MODEL` | `buffalo_l` | InsightFace model pack for detection/recognition |
|| `FACE_MATCH_THRESHOLD` | `0.45` | Cosine similarity threshold for face identity matching |
|| `PROMETHEUS_ENABLED` | `true` | Enable Prometheus /metrics endpoint with pipeline/retrieval/GPU metrics |
||| `FEDERATION_ENABLED` | `false` | Enable federated video search REST endpoint (v0.33.0) |
||| `FEDERATION_PEERS` | (empty) | Comma-separated peer MCP server URLs |
||| `FEDERATION_TIMEOUT` | `30.0` | HTTP request timeout per peer (seconds) |
||| `FEDERATION_INCLUDE_LOCAL` | `true` | Include local index in federated results |
||| `MMR_DIVERSITY_ENABLED` | `false` | Enable MMR diversity re-ranking (v0.34.0) |
||| `MMR_LAMBDA` | `0.5` | MMR lambda [0,1]; 0 = pure diversity, 1 = pure relevance |
||| `MMR_TOP_K` | `15` | Number of chunks to re-rank with MMR |
|||| `OCR_MODEL_VERSION` | `PP-OCRv6` | OCR model version (PP-OCRv6 or PP-OCRv5) |
|||| `OCR_MODEL_TIER` | `medium` | OCR model tier (tiny/small/medium) |
||| `AGENT_ENABLED` | `false` | Enable Agentic Video Understanding Agent (v0.36.0) |
||| `AGENT_MAX_TOOLS` | `5` | Max tool invocations per agent query |
||| `LIVE_STREAM_ENABLED` | `false` | Enable live stream analysis (v0.40.0 — RTMP/RTSP/HLS) |
||| `LIVE_STREAM_URL` | (empty) | RTMP/RTSP/HLS stream URL |
||| `LIVE_STREAM_SOURCE` | `rtmp` | Stream type: rtmp, rtsp, hls |
||| `LIVE_STREAM_CHUNK_DURATION` | `30.0` | Chunk duration in seconds |
||| `LIVE_STREAM_SLIDING_WINDOW` | `300` | Sliding context window in seconds |
||| `LIVE_STREAM_AUTO_RECONNECT` | `true` | Auto-reconnect on stream loss |
||| `LIVE_STREAM_MAX_RETRIES` | `3` | Max reconnection attempts |
||| `LIVE_STREAM_RETRY_DELAY` | `5.0` | Delay between retries (seconds) |

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
|- [x] [RESEARCH] Cross-video scene graph edges — scene_graph.py adjacency structure already supports cross-video keys; add entity-based + BGE-VL cross-video edges
|- [x] [RESEARCH] Gradio 6 Workflow subgraphs — Gradio 6.19+ exposes composable subgraph API endpoints; FastAPI hybrid approach complements it
|- [x] [RESEARCH] Sparse-frame optical flow — FFmpeg motion vectors (zero-cost) recommended over deep flow models for 12GB VRAM pipeline
|- [x] [RESEARCH v0.18] Qwen3-VL-30B-A3B (Apache 2.0, 3B active, MoE, FP8) — new optimal MLLM backend displacing VideoChat-Flash 2B
|- [x] [RESEARCH v0.18] PaddleOCR v5 upgrade — +13% accuracy, 109 languages, PP-StructureV3
|- [x] [RESEARCH v0.18] Dependency modernization — torch 2.12.1, transformers 5.12.1, sentence-transformers 5.6.0
||- [x] [RESEARCH v0.18] Qwen3.5-0.8B (Apache 2.0, 800M multimodal) — lightweight video classifier for PipelineOrchestrator
||- [x] [RESEARCH v0.18] ChromaDB confirmed (stay) — LanceDB only if >5M vectors
- [x] **Entity-level tracking across scenes** (ByteTrack via Ultralytics built-in — MIT, ~500 MB shared with YOLO)
- [x] **Cross-video scene graph edges** (track_id entity matching enables cross-video scene retrieval)
|- [x] [RESEARCH v0.20] Modular actor pipeline — PipelineStage ABC with DAG orchestration (Stage-as-a-Service: FastAPI + Gradio Workflow + MCP)
|- [x] [RESEARCH v0.20] Content-addressable pipeline cache — SHA-256 per-stage caching, 70-90% faster re-runs
|- [x] [RESEARCH v0.20] MCP tool server — Python SDK server design (process_video, search_videos, ask_question, extract_scenes)
|- [x] [RESEARCH v0.20] InsightFace integration — RetinaFace + ArcFace person identity blueprint
|- [x] [RESEARCH v0.20] PipelineOrchestrator — heuristic + Qwen3.5-0.8B ML video type classifier
|- [x] [RESEARCH v0.20] FFmpeg motion vector extractor — zero-GPU sparse optical flow
|- [x] Qwen3-VL-30B-A3B FP8 backend (vLLM FP8, FlashAttention-3, 128K context, backends/qwen3_vl.py)
||- [x] [RESEARCH v0.22] Audio-only processing mode — config-driven stage filtering, 50-75% faster for podcasts/lectures
||- [x] [RESEARCH v0.22] Conversation memory — ChromaDB-backed persistent chat history, cross-video Q&A continuity
||- [x] [RESEARCH v0.22] Structured JSON logging — structlog integration across pipeline stages
||- [x] [RESEARCH v0.22] Dependency modernization — transformers 5.12.1, torch 2.12+, sentence-transformers 5.6+
||- [x] [RESEARCH v0.22] Pipeline caching blueprint — SHA-256 content-addressable per-stage cache design
||- [x] [RESEARCH v0.22] PipelineOrchestrator blueprint — file-type heuristic + optional MLLM classifier design
|- [x] Audio-only processing mode — `processing_mode` config, stage filtering in pipeline.py
|- [x] Conversation memory — `video_analysis/memory.py`, ChromaDB-backed persistent chat history
|- [x] Structured JSON logging — structlog integration across pipeline stages
||- [x] Dependency modernization — pyproject.toml bounds updated (torch>=2.12, transformers>=5.12, sentence-transformers>=5.6)
||- [x] Pipeline caching + incremental re-indexing — `video_analysis/cache.py`, content-addressable SHA-256 per-stage cache with persistent index, config-aware invalidation, TTL expiry
||- [x] PipelineOrchestrator heuristic — `video_analysis/orchestrator.py`, file-type + ffprobe + heuristic classification into 7 video types with stage overrides
|- [x] Pipeline benchmarking infra — pynvml per-stage VRAM tracking, pytest-benchmark suite
|- [x] MCP tool server (expose stages as MCP tools for Hermes/agentic workflows) — 7 tools, stdio + SSE
||- [x] Sparse-frame optical flow for motion-based adaptive frame sampling (FFmpeg MVs, zero GPU, video_analysis/flow.py)
||- [x] DINOv2 perceptual frame compression (LongVU-style, ICML 2025, 21M params, ~85 MB VRAM)
|||- [x] PP-OCRv6 upgrade — PP-OCRv6 with configurable model tiers (tiny/small/medium, +4.6% det, +5.1% rec over v5)
|- [x] InsightFace face recognition (SCRFD-10G + ArcFace, cross-video person identity)
|- [x] Agentic self-check + re-retrieval (LLM-verified answer-evidence alignment)
|- [x] **Prometheus metrics endpoint + Grafana dashboards** — 20+ counters/histograms/gauges for pipeline runs, retrieval, GPU memory, ChromaDB size, and question answering; graceful fallback when prometheus_client absent; config toggle via `PROMETHEUS_ENABLED`
|- [x] **Dependency modernization** — all pyproject.toml & requirements.txt bounds updated to latest stable (torch 2.12, transformers 5.12, sentence-transformers 5.6, fastapi 0.138, etc.)
|- [x] **Gradio 6 Workflow integration** — `ui/workflow.py` with `gr.Workflow` visual pipeline builder canvas (Gradio 6.17+ API: `bind`, `edges`, `graph`)
|||- [x] ColBERT-Att attention-weighted re-ranking (drop-in ColBERTv2 upgrade, +1-3% recall)
|- [x] ColBERT-Att attention-weighted re-ranking (drop-in ColBERTv2 upgrade, +1-3% recall)
||- [x] Real-time streaming video analysis (chunked processing, watch/stream modes)
||- [x] **Federated video search (MCP-based cross-instance query)**
||- [x] **PP-OCRv6 upgrade (config + model tier for tiny/small/medium)**
||- [x] **Scene graph face-entity enrichment (cross-video person-based edges)**
||- [x] **MMR diversity re-ranking (30-50% context redundancy reduction)**
||- [x] Qwen3-VL-30B-A3B FP8 backend (torchao FP8, FlashAttention-3, 256K context)
||- [x] **Video Content Chaptering** — NLTK TextTiling-based topic segmentation with LLM/fallback title generation, chapter report generation, agent chapter context integration |
|- [x] **Live Stream Analysis (RTMP/RTSP/HLS)** — real-time capture via FFmpeg `-re` with auto-reconnect, sliding window, and URL-based auto-detection (v0.40.0)
|
|
MIT
