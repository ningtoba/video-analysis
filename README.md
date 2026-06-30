# 🎥 Video Analysis Platform

**Self-hosted video understanding with RAG-powered AI chatbot.** Upload any video, paste a YouTube URL, or batch-process files — the AI pipeline extracts and analyzes every detail (transcription, scene detection, object recognition, semantic description, OCR, speaker diarization, face recognition, action recognition), indexes it into a multi-granularity vector store, and lets you ask natural language questions with precise timestamp citations.

```
┌──────────────┐    ┌─────────────────────────────┐    ┌──────────────────────────┐
│  Upload      │───▶│  Analysis Pipeline            │───▶│  RAG Vector Index        │
│  (drag-drop) │    │  FFmpeg → SceneDetect → YOLO │    │  ChromaDB + BGE-VL       │
│  or YouTube  │    │  → Whisper → CLIP → OCR      │    │  + Multi-Granularity     │
│  URL Import  │    │  → Diarization → Face → XCLIP │    │  + Temporal Weighting    │
└──────────────┘    │  → Sprite Sheet → Entity Tk   │    └─────────┬───────────────┘
                    └─────────────────────────────┘              │
┌──────────────┐    ┌─────────────────────────────┐              │
│  Ask Q&A     │◀───│  Context Retrieval           │◀─────────────┘
│  + Citations │    │  BGE-VL + Cross-Encoder      │
│  + Clip Export│   │  + TV-RAG Temporal Decay     │
│  + Events    │    │  + Scene Graph + MMR         │
└──────────────┘    │  + Agentic RAG + Self-Check   │
                    └─────────────────────────────┘
```

---

## ✨ Features

### Core Pipeline

- **🎬 Smart Video Analysis** — scene detection (PySceneDetect 0.7+ with 5 detector modes: Adaptive, Content, Histogram, Hash, FFmpeg), keyframe extraction, transcription (faster-whisper large-v3, int8), speaker diarization (PyAnnote 3.1), OCR text extraction (PaddleOCR PP-OCRv6 — +4.6% detection, +5.1% recognition over v5), object detection (YOLO via ultralytics), semantic scene description (OpenCLIP ViT-L-14), entity tracking (ByteTrack via Ultralytics), zero-shot action recognition (X-CLIP)
- **🌐 YouTube / URL Import** — download videos directly from YouTube, Vimeo, Twitch, and 1000+ sites via yt-dlp
- **📦 Batch Processing** — queue videos by URL list or file upload for sequential analysis
- **🎨 Adaptive Pipeline Scaling** (v0.60.0) — intelligent per-video quality/resource auto-tuning; 3 explicit policies (conservative/balanced/performance) + auto mode that selects based on duration and resolution; VRAM-aware auto-downgrade prevents OOM on low-memory GPUs; long videos get aggressive DINOv2 compression, short high-res videos get maximum detail
- **⚡ GPU Accelerated** — RTX 4070 CUDA 12.8 support for all models with sequential loading to manage 12 GB VRAM; graceful CPU fallback
- **🔒 100% Local** — no API keys, no cloud services, all processing on your hardware

### Retrieval & Q&A

- **🔍 RAG-Powered** — ChromaDB vector store with **BGE-VL-base** multimodal embedding (MIT, 150M params, ~0.8 GB VRAM) + Nomic Embed v1.5 fallback + embedding prefix normalization + cross-encoder re-ranking
- **⏱ TV-RAG Temporal Decay** — time-decay weighting for temporally aware retrieval (ACM Multimedia 2025)
- **📐 Multi-Granularity Chunking** — fixed-window 60s + sliding-window 30s + scene + frame chunk types for precise retrieval
- **🧠 Quad-Backend Video MLLM** — SmolVLM2 (Apache 2.0, transformers-native, 2.2B/500M/256M), VideoChat-Flash 2B (MIT, ICLR 2026), **Qwen3-VL-30B-A3B** (Apache 2.0, MoE 30B/3B active, FP8, 128K context via vLLM), or **InternVideo3-8B** (OpenGVLab, June 2026, SOTA — 73.8 Video-MME, MCR reasoning, M^2LA KV-cache compression)
- **🤖 Agentic RAG** — iterative retrieval loop with confidence-based early stopping across 4 rounds (standard → multi-hop → scene-graph → LLM self-check verification with re-retrieval), inspired by Self-RAG, FLARE, and CRAG
- **🧠 Self-Check + Re-Retrieval** — LLM-verified answer-evidence alignment with up to 2 rounds of verification
- **📊 Scene Graph Retrieval** — VGent/ViG-RAG inspired graph-based video retrieval with K-hop expansion, temporal/semantic/entity edges, and cross-video scene connections
- **🔀 Query Classifier & Router** — automatic text/visual/temporal/multimodal modality dispatch
- **🧩 Multi-Hop Query Decomposition** — sub-question generation → per-sub-query retrieval → reasoning synthesis
- **🎯 MMR Diversity Re-Ranking** — Maximal Marginal Relevance for 30–50% context redundancy reduction
- **📖 ColBERTv2 / ColBERT-Att** — optional late-interaction re-ranking; ColBERT-Att (arXiv:2603.25248) adds +1–3% recall
- **🖼️ Qwen3-VL-Embedding** — optional multimodal embedding for composed text+image search
- **📚 Conversation Memory** — ChromaDB-backed persistent chat history with TTL expiry for cross-video Q&A continuity

### UI & Interaction

- **🎨 Production Web UI** (v0.61.0) — FastAPI + Jinja2 + HTMX + Alpine.js dark theme with 10+ tabs, lazy-loading, SSE streaming, WebSocket progress (~30 KB JS, no Node.js build step)
- **💬 AI Chatbot** — ask questions about video content with timestamped source citations and clip export
- **📚 Video Library** — multi-video management with searchable library, delete, re-index
- **🖼️ Timeline Preview** — sprite sheet generation (100 thumbnails) for visual timeline browsing with hover preview
- **✂️ Clip Export** — export precise video clips at any timestamp range from the UI
- **📖 Video Chaptering** (v0.37.0) — automatic topic segmentation of transcripts into chapters using NLTK TextTiling, with LLM-generated titles and summaries
- **📡 SSE Streaming Chat** — token-by-token response streaming for real-time chat UX
- **🩺 Pipeline Health Dashboard** (v0.52.0) — automated anomaly detection, drift tracking, alerting, and composite health scoring; 4 REST API endpoints for programmatic access
- **📊 Evaluation Dashboard** (v0.46.0) — real-time system metrics: live pipeline run counters, GPU memory usage, job queue status, interactive evaluation runner
- **📈 Eval Comparison** (v0.48.0) — cross-report comparison dashboard for regression/improvement tracking across pipeline versions; 3 REST API endpoints
- **🔗 Knowledge Graph Explorer** (v0.53.0) — visual entity browsing, timeline exploration, entity type filtering, relationship visualization, and LLM context injection
- **⏱ Events Timeline Tab** (v0.58.0) — event timeline visualization with causal chain rendering
- **📷 Webcam Capture** (v0.41.0) — real-time webcam capture and frame analysis tab with live preview, capture & analyze, and continuous monitoring mode

### Streaming & Live

- **📡 Live Stream Analysis** (v0.40.0) — capture and analyze real-time RTMP/RTSP/HLS streams with auto-reconnect, sliding window context (300s default), and incremental indexing; connect OBS, IP cameras, and streaming platforms
- **💭 Streaming Thinking** (v0.57.0) — amortized reasoning during real-time video streaming (arXiv:2603.12262); per-chunk entity accumulation, causal prediction (forward thinking), causal explanation (backward thinking), question generation, and incremental answer engine for live streams
- **📺 Chunked Streaming Pipeline** — process videos in configurable chunks for low-latency first results; watch mode for files being written

### Advanced AI

- **🧩 Event-Causal RAG** (v0.57.0) — semantic event-level video segmentation with State-Event-State (SES) graphs and bidirectional causal-topological retrieval (arXiv:2605.06185); 3-tier segmentation (LLM → transcript-coherence → temporal-grid), DualStoreMemory for fused semantic + causal retrieval, forward/backward causal path analysis
- **🤖 Agentic Video Understanding Agent** (v0.36.0) — multi-tool video understanding agent with 7 specialized tools (analyze_frames, detect_objects, OCR, search_transcript, search_rag, temporal_grounding, summarize_video) that dynamically routes questions to the right tools
- **🧑‍🤝‍🧑 Face Recognition** (optional) — InsightFace (SCRFD-10G + ArcFace W50) for face detection, 512-d embeddings, and cross-video person identity matching (~1.1 GB VRAM)
- **🕵️ Autonomous Video Curator** (v0.45.0) — closed-loop MCR video exploration agent that proactively watches videos, discovers entities/objects/scenes, builds structured knowledge, and generates comprehensive curation reports; inspired by InternVideo3 MCR and HKUDS VideoAgent
- **🧩 Hierarchical Multi-Agent Orchestrator** (v0.51.0) — HiCrew-inspired multi-agent architecture with RouterAgent planning layer, 7 specialist sub-agents (Visual/RAG/Transcript/Object/OCR/Confidence/Summarizer), parallel dispatch, and EvidenceSynthesizer with tiered weighting
- **🛡️ Robust Agent Confidence** (v0.50.0) — Robust-TO inspired per-frame trustworthiness scoring, per-source evidence confidence adjustment, three-tier evidence weighting with weighted combination and consensus, untrustworthy frame filtering
- **📋 Structured Video Reports** (v0.50.0) — comprehensive JSON schema report generator with VideoMetadata, TimelineSummary, SceneReport, TranscriptReport, ObjectCatalog, ActionSummary, RAGStats
- **🎯 Zero-Shot Action Recognition** (optional) — X-CLIP open-vocabulary action detection (~4 GB VRAM)
- **🖼️ DINOv2 Perceptual Frame Compression** (optional, v0.30.0) — LongVU-style perceptual similarity compression; DINOv2-small (21M params, ~85 MB VRAM) drops redundant frames based on cosine similarity

### Observability & Operations

- **📈 Prometheus Metrics** — 20+ counters/histograms/gauges for pipeline runs, retrieval latency, GPU memory, ChromaDB size, and Q&A performance; `/metrics` endpoint at `:9400`
- **📊 Grafana Dashboard** — production-ready dashboard (`deploy/grafana-dashboard.json`) with panels for pipeline throughput, retrieval latency, GPU resources, system health, Q&A quality
- **🔔 Webhook Notifications** (v0.59.0) — event-driven HTTP POST callbacks on `pipeline.complete`, `eval.complete`, `health.alert`, and `health.critical` events; configurable via `WEBHOOK_URL` (comma-separated)
- **🧪 Evaluation Suite** (v0.44.0) — 5 evaluation tasks (retrieval_precision, scene_boundary_accuracy, ocr_accuracy, action_recognition_quality, frame_compression_efficiency) with 44+ harness tests; auto-persisted reports with Prometheus gauge export
- **📡 OpenTelemetry Tracing** (v0.49.0) — distributed tracing for pipeline, RAG, and API spans with OTLP export
- **🩺 Health Endpoints** — `/health`, `/api/health/live`, `/api/health/ready` for Kubernetes liveness/readiness probes
- **🔄 Graceful Shutdown** — SIGTERM/SIGINT handling with clean partial saves on termination
- **🔐 Rate Limiting** (v0.49.0) — in-memory token bucket rate limiter for REST API (100 requests/minute per client, configurable)

### REST API

- **🌐 Full REST API** (v0.41.0+) — 30+ endpoints with auto-generated OpenAPI docs at `/docs`
  - `POST /api/videos/process` — enqueue video for async processing (returns job_id)
  - `GET /api/jobs/{job_id}` — poll job status
  - `POST /api/videos/{video_id}/query` — ask a question about a video
  - `GET /api/videos/search` — cross-video semantic search
  - `GET /api/videos/{video_id}/transcript` — get transcript
  - `GET /api/videos/{video_id}/chapters` — get chapters
  - `GET /api/videos/{video_id}/frames/{timestamp}` — get frame image
  - `DELETE /api/videos/{video_id}` — delete from index
  - `GET /api/sse/chat` — SSE streaming chat
  - `GET /api/evaluations` — list evaluation reports
  - `GET /api/evaluations/{run_id}` — full evaluation report
  - `GET /api/evaluations/compare` — compare two reports
  - Knowledge Graph endpoints (v0.53.0): stats, entities, timeline, relationships, video entities, LLM context
  - Pipeline Health endpoints (v0.53.0): runs report, summary, alerts, alert acknowledge
- **🐍 Python Client SDK** (v0.49.0) — `video_analysis/client.py` for programmatic API access
- **🔌 Model Context Protocol (MCP) Server** — expose pipeline stages as MCP tools for Hermes/agentic workflows (stdio + SSE modes); 7 tools: `process_video`, `search_videos`, `ask_question`, `extract_scenes`, `get_transcript`, `get_video_info`, `list_videos`

### Cross-Video Intelligence

- **🧠 Persistent Video Knowledge Graph** (v0.52.0) — SQLite-backed cross-video entity & relationship store; tracks people, objects, actions, and concepts with frequency counters, typed relationships, cross-video search, and chronological timeline; injectable as LLM context
- **🔗 Cross-Video Scene Graph** — entity-aware edges (track_id matching) connecting scenes across videos for cross-video scene retrieval
- **🌍 Federated Video Search** (v0.33.0) — MCP-based cross-instance video search across multiple federated peers

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10–3.12** (3.11 recommended)
- **FFmpeg** (for video/audio extraction, sprite sheets, clip export)
- **NVIDIA GPU with CUDA** (recommended — RTX 4070 12 GB verified; CPU fallback works but is slower)
- **NVIDIA Container Toolkit** (for Docker GPU access)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd video-analysis

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install with default extras (API + UI + streaming)
pip install --upgrade pip
pip install -e ".[default]"

# Or install everything (all extras + dev tools)
pip install -e ".[all,dev]"

# Optional: GPU-optimized PyTorch
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[all,dev]"
```

### Launch the Web UI

```bash
python -m video_analysis
```

Then open **<http://localhost:7860>** in your browser.

### CLI Mode

```bash
# Process a single video and ask a question
python -m video_analysis --cli --video my_video.mp4 --query "What objects are visible?"

# Download from YouTube and process
python -m video_analysis --url "https://www.youtube.com/watch?v=..."

# Batch process from a URL list file
python -m video_analysis --batch urls.txt
```

### Docker

```bash
# Build and run with GPU
make docker-run

# Or manually:
docker build -t video-analysis:latest .
docker run --gpus all -p 7860:7860 -v ./data:/app/data video-analysis:latest

# Docker Compose (development)
docker compose up -d

# Docker Compose (production with DCGM + Caddy)
docker compose -f docker-compose.prod.yml up -d
```

---

## 🏗️ Architecture

### Ingestion Pipeline

```
Video File / YouTube URL / Live Stream / RTSP Camera
│
├── 1. FFprobe — extract duration, resolution, FPS, codec
├── 2. Adaptive Pipeline Scaler — select quality/resource policy
├── 3. FFmpeg — extract audio (16 kHz WAV)
│   └── faster-whisper (large-v3, int8) → Timestamped Transcript
│   └── PyAnnote Audio → Speaker Diarization (SPEAKER_00/01/...)
├── 4. PySceneDetect 0.7+ — scene boundary detection
│   ├── AdaptiveDetector (default) — rolling HSV average
│   ├── ContentDetector — fixed-threshold HSV changes
│   ├── HistogramDetector — Y-channel histogram diffs
│   ├── HashDetector — perceptual hashing
│   └── FFmpeg fallback (gt(scene, ...))
├── 5. Per scene: keyframe extraction
│   ├── DINOv2 perceptual compression (optional) → deduplicated frames
│   ├── YOLO object detection + ByteTrack entity tracking
│   ├── InsightFace face detection & recognition (optional)
│   ├── PaddleOCR PP-OCRv6 text extraction
│   ├── OpenCLIP zero-shot scene classification (ViT-L-14)
│   ├── X-CLIP zero-shot action recognition (optional)
│   ├── Video MLLM scene description (optional, replaces CLIP)
│   └── Frame quality screening (blur, brightness, static detection)
├── 6. FFmpeg → Sprite sheet (100 thumbnails for timeline)
├── 7. NLTK TextTiling → Chapter segmentation (optional)
├── 8. Merge → Structured VideoIndex
└── 9. ChromaDB Vector Store (BGE-VL embeddings)
    └── Multi-granularity chunks: scene, frame, transcript, fixed-60s, sliding-30s
    └── TV-RAG temporal decay weighting
    └── Event-Causal RAG segmentation (optional) → SES graph + DualStoreMemory
    └── Knowledge Graph entity/relationship persistence
```

### Query Pipeline

```
User Question
│
├── 1. Query Classifier & Router — text/visual/temporal/multimodal
├── 2. Multi-Hop Decomposition (optional) → sub-questions
├── 3. Query Embedding — BGE-VL (or Nomic Embed fallback)
├── 4. ChromaDB Hybrid Search — dense + metadata + chunk_type
│   └── TV-RAG Temporal Decay — score × exp(-λ·Δt)
├── 5. Cross-Encoder Re-Ranking (ms-marco-MiniLM)
├── 6. Optional: ColBERTv2 / ColBERT-Att late-interaction re-rank
├── 7. MMR Diversity Re-Ranking (optional)
├── 8. Scene Graph K-Hop Expansion (optional)
├── 9. Agentic RAG Loop (optional) — up to 4 rounds
│   ├── Round 1: Standard retrieval
│   ├── Round 2: Multi-hop sub-queries
│   ├── Round 3: Scene-graph expansion
│   └── Round 4: LLM self-check + re-retrieval
├── 10. Event-Causal Retrieval (optional) — forward/backward causal paths
├── 11. Temporal Context Expansion — ±1 neighbor scene
├── 12. Sort Chronologically
└── 13. LLM (DeepSeek-V4-Flash / OpenAI-compatible) → Answer
    └── Timestamp citations + source evidence
```

### Audio-Only Mode

When `PROCESSING_MODE=audio_only`, the pipeline skips all visual stages (frame extraction, scene detection, YOLO, CLIP, OCR, face recognition, sprite sheets) and processes only audio transcription + diarization. Ideal for podcasts, lectures, and audio recordings — **50–75% faster** for content that has no visual information.

### Hierarchical Multi-Agent Orchestrator

```
RouterAgent (Planning Layer)
├── Question analysis → modality / complexity detection
├── RoutePlan generation (sub-questions, dependencies)
└── Specialist Agent dispatch (parallel where possible)
    ├── VisualAnalyst → analyze_frames (Video MLLM)
    ├── RAGSearcher → search_rag (query refinement)
    ├── TranscriptAnalyst → search_transcript (temporal grounding)
    ├── ObjectDetectorAgent → detect_objects (YOLO + tracking)
    ├── OCRAgent → extract_text (OCR)
    ├── ConfidenceAuditor → cross-validate (agent_confidence)
    └── SummarizerAgent → summarize_video
EvidenceSynthesizer (final combination)
├── EvidenceWeighter (tiered/continuous)
├── Source attribution with citations
└── Weighted combination → OrchestratorResult
```

### Agentic Video Understanding Agent

7 specialized tools that dynamically route questions:

| Tool | Purpose |
|------|---------|
| `analyze_frames(keywords)` | Multi-frame analysis with Video MLLM |
| `detect_objects(object_name)` | YOLO object detection with temporal filtering |
| `OCR()` | Text extraction from frames |
| `search_transcript(query)` | Full-text transcript search |
| `search_rag(query)` | Semantic RAG retrieval |
| `temporal_grounding(query)` | Pin events to timestamps via transcript alignment |
| `summarize_video()` | Structured video summary |

---

## 📖 Module Reference

| Module | File | Purpose |
|--------|------|---------|
| `pipeline` | `video_analysis/pipeline.py` | Core video ingestion: scene detection, frame extraction, transcription, diarization, YOLO, OCR, CLIP, sprite sheets, YouTube/URL download (~1900 lines) |
| `rag` | `video_analysis/rag.py` | ChromaDB indexing, hybrid retrieval, multi-granularity chunks, cross-encoder + ColBERT re-ranking, TV-RAG temporal decay, MMR diversity, scene graph retrieval, agentic RAG loop (~1700 lines) |
| `chat` | `video_analysis/chat.py` | LLM Q&A with conversation history, source citations, multi-video context, event-causal retrieval integration |
| `models` | `video_analysis/models.py` | Data models: VideoIndex, SceneInfo, FrameInfo, ChatMessage, TranscriptSegment |
| `config` | `video_analysis/config.py` | Central Config dataclass — 80+ fields with env var overrides, path auto-creation (~800 lines) |
| `llm_provider` | `video_analysis/llm_provider.py` | LLM abstraction layer — Hermes CLI, OpenAI-compatible API, auto-detect |
| `stream_chat` | `video_analysis/stream_chat.py` | Token-by-token SSE streaming for LLM responses |
| `api` | `video_analysis/api.py` | Full REST API with 30+ endpoints, OpenAPI docs, error handlers, rate limiting (~1600 lines) |
| `client` | `video_analysis/client.py` | Python API client SDK for the REST API |
| `job_queue` | `video_analysis/job_queue.py` | In-process async job queue — background video processing with status polling |
| `agent` | `video_analysis/agent.py` | Multi-tool Video Understanding Agent — 7 tools, dynamic dispatch |
| `agent_confidence` | `video_analysis/agent_confidence.py` | Robust-TO inspired per-frame trustworthiness scoring, evidence weighting |
| `orchestra` | `video_analysis/orchestra.py` | Hierarchical multi-agent orchestrator — HiCrew-inspired planning, 7 specialist agents, evidence synthesis (~1450 lines) |
| `knowledge_graph` | `video_analysis/knowledge_graph.py` | SQLite-backed cross-video entity & relationship store |
| `pipeline_health` | `video_analysis/pipeline_health.py` | Automated anomaly detection, drift tracking, alerting, composite health scoring |
| `event_rag` | `video_analysis/event_rag.py` | Event-Causal RAG — SES graph segmentation, DualStoreMemory, causal-topological retrieval |
| `streaming_think` | `video_analysis/streaming_think.py` | Amortized reasoning for live video streaming |
| `evaluation` | `video_analysis/evaluation.py` | Benchmark-driven evaluation harness — 5 tasks, synthetic fixtures, auto-persisted reports |
| `curator` | `video_analysis/curator.py` | Autonomous MCR video exploration agent — closed-loop discovery |
| `adaptive_scaler` | `video_analysis/adaptive_scaler.py` | Per-video quality/resource auto-tuning with VRAM awareness |
| `webhook` | `video_analysis/webhook.py` | Event-driven HTTP POST callbacks — pure stdlib |
| `frame_compression` | `video_analysis/frame_compression.py` | DINOv2 perceptual frame compression (LongVU-style) |
| `face` | `video_analysis/face.py` | InsightFace face detection & recognition |
| `scene_graph` | `video_analysis/scene_graph.py` | VGent/ViG-RAG inspired graph-based video retrieval |
| `query_router` | `video_analysis/query_router.py` | Query classification & routing — text/visual/temporal/multimodal dispatch |
| `chapters` | `video_analysis/chapters.py` | NLTK TextTiling topic segmentation + LLM chapter titles |
| `quality` | `video_analysis/quality.py` | Frame quality pre-screening — blur, brightness, static detection |
| `memory` | `video_analysis/memory.py` | ChromaDB-backed conversation memory with TTL |
| `telemetry` | `video_analysis/telemetry.py` | OpenTelemetry distributed tracing — pipeline/RAG/API spans with OTLP export |
| `rate_limiter` | `video_analysis/rate_limiter.py` | In-memory token bucket rate limiter for REST API |
| `error_handlers` | `video_analysis/error_handlers.py` | Structured JSON error responses for REST API |
| `report` | `video_analysis/report.py` | Structured video report generation — comprehensive JSON schema |
| `benchmark` | `video_analysis/benchmark.py` | Pipeline benchmarking — per-stage VRAM tracking, pytest-benchmark suite |
| `flow` | `video_analysis/flow.py` | Sparse-frame optical flow via FFmpeg motion vectors (zero GPU) |
| `metrics` | `video_analysis/metrics.py` | Prometheus metrics — counters, histograms, gauges |
| `classifier` | `video_analysis/classifier.py` | Video type classification |
| `cache` | `video_analysis/cache.py` | Content-addressable SHA-256 per-stage pipeline cache |
| `orchestrator` | `video_analysis/orchestrator.py` | PipelineOrchestrator — file-type heuristic + optional MLLM classifier |
| `storage` | `video_analysis/storage.py` | Tiered frame storage — full/tiered/compressed modes |
| `streaming` | `video_analysis/streaming.py` | Real-time streaming/chunked video analysis pipeline |
| `video_mllm` | `video_analysis/video_mllm.py` | Quad-backend Video MLLM manager — auto-selects from 4 backends |
| `federation` | `video_analysis/federation.py` | Federated MCP-based cross-instance video search |
| `mcp_server` | `video_analysis/mcp_server.py` | MCP tool server — stdio + HTTP SSE modes |
| `action` | `video_analysis/action.py` | X-CLIP zero-shot action recognition |
| `self_check` | `video_analysis/self_check.py` | LLM-based self-check verification + re-retrieval |
| `logging_setup` | `video_analysis/logging_setup.py` | structlog-structured logging configuration |
| `config_store` | `video_analysis/config_store.py` | Persistent config store |
| `backends/qwen3_vl` | `video_analysis/backends/qwen3_vl.py` | Qwen3-VL-30B-A3B FP8 backend — vLLM, FlashAttention-3, 128K context |
| `backends/internvideo3` | `video_analysis/backends/internvideo3.py` | InternVideo3-8B backend — vLLM, MCR reasoning, M^2LA compression |
| `ui/server` | `ui/server.py` | FastAPI app factory — Jinja2 templates, HTMX partials, WebSocket progress, static files |
| `ui/routes/` | `ui/routes/*.py` | Route handlers per tab (analysis, import, batch, search, library, camera, monitor, comparison, knowledge_graph, event_timeline, settings) |
| `ui/app` | `ui/app.py` | Legacy Gradio UI application |

---

## ⚙️ Configuration

Configuration is managed via the `Config` dataclass in `video_analysis/config.py`. All fields can be overridden through environment variables.

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_ANALYSIS_DATA` | `data/` | Data directory for videos, frames, audio, ChromaDB |
| `PROCESSING_MODE` | `video_full` | `video_full`, `audio_only`, or `auto` |
| `HF_TOKEN` | (unset) | Hugging Face token for gated models & higher rate limits |
| `STRUCTURED_LOGGING_ENABLED` | `true` | Enable structlog-based structured logging |
| `STRUCTURED_LOGGING_FORMAT` | `auto` | Output format: `auto`, `console`, `json` |
| `STRUCTURED_LOGGING_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Transcription & Audio

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device for transcription: `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8_float16` | Compute type: `int8_float16`, `float16`, `int8` |
| `DIARIZE_ENABLED` | `true` | Enable PyAnnote speaker diarization |
| `ASR_BACKEND` | `faster-whisper` | ASR backend (only faster-whisper implemented) |

### Scene Detection & Frame Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `SCENE_DETECTOR` | `adaptive` | Detector mode: `adaptive`, `content`, `histogram`, `hash`, `ffmpeg` |
| `ADAPTIVE_FRAME_SAMPLING` | `false` | Enable motion-based adaptive frame sampling |
| `ADAPTIVE_FRAME_SAMPLING_SENSITIVITY` | `0.3` | Sampling density near scene boundaries |
| `CLIP_FRAME_DEDUP` | `false` | Enable CLIP-similarity frame deduplication |
| `CLIP_FRAME_DEDUP_THRESHOLD` | `0.92` | Similarity threshold for frame deduplication |
| `FRAME_STORAGE_MODE` | `tiered` | Storage mode: `full`, `tiered`, `compressed` |
| `VIDEO_ANALYSIS_PORT` | `7860` | Web UI port |

### Vision Models

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_MODEL` | `ViT-L-14-quickgelu` | OpenCLIP model: `ViT-B-32`, `ViT-L-14-quickgelu`, `ViT-H-14-quickgelu`, `ViT-SO400M-14-SigLIP-384` |
| `CLIP_PRETRAINED` | `dfn5b` | OpenCLIP pretrained dataset |
| `OCR_ENABLED` | `true` | Enable PaddleOCR text extraction |
| `OCR_MODEL_VERSION` | `PP-OCRv6` | OCR version: `PP-OCRv6` or `PP-OCRv5` |
| `OCR_MODEL_TIER` | `medium` | OCR tier: `tiny`, `small`, `medium` |
| `FACE_RECOGNITION_ENABLED` | `false` | Enable InsightFace face detection & recognition |
| `ACTION_RECOGNITION_ENABLED` | `false` | Enable X-CLIP zero-shot action recognition |
| `DINO_FRAME_COMPRESSION` | `false` | Enable DINOv2 perceptual frame compression |
| `DINO_FRAME_COMPRESSION_THRESHOLD` | `0.88` | Cosine sim threshold; lower = more aggressive |

### Embedding & Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/BGE-VL-base` | Primary multimodal embedding model |
| `TEXT_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v2` | Fallback text-only embedding |
| `TEMPORAL_DECAY_RATE` | `0.1` | TV-RAG temporal decay; `0` = disabled |
| `COLBERT_RERANKER_ENABLED` | `false` | Enable ColBERTv2 late-interaction re-ranking |
| `COLBERT_ATT_RERANKER_ENABLED` | `false` | Enable ColBERT-Att attention-weighted re-ranking |
| `MMR_DIVERSITY_ENABLED` | `false` | Enable MMR diversity re-ranking |
| `MMR_LAMBDA` | `0.5` | MMR diversity/relevance balance |
| `MULTIMODAL_EMBEDDING` | `false` | Enable Qwen3-VL-Embedding multimodal search |
| `AGENTIC_RETRIEVAL_ENABLED` | `true` | Enable iterative agentic retrieval loop |
| `AGENTIC_MAX_ROUNDS` | `4` | Max retrieval rounds |
| `SELF_CHECK_ENABLED` | `true` | Enable LLM self-check verification |

### Video MLLM Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_MLLM_ENABLED` | `false` | Enable video MLLM |
| `VIDEO_MLLM_BACKEND` | `auto` | Backend: `auto`, `videochat_flash`, `smolvlm2`, `qwen3_vl`, `internvideo3` |
| `VIDEO_MLLM_MODEL` | `OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448` | MLLM model name |
| `VIDEO_MLLM_MODEL_SIZE` | `2.2B` | SmolVLM2 size: `2.2B`, `500M`, `256M` |
| `VIDEO_MLLM_AS_DESCRIBER` | `false` | Use MLLM for scene descriptions (replaces CLIP) |
| `VIDEO_MLLM_AS_CHAT_BACKEND` | `false` | Use MLLM as video-native Q&A backend |
| `INTERNVIDEO3_VLLM_URL` | `http://localhost:8001` | InternVideo3 vLLM server URL |
| `INTERNVIDEO3_FP8` | `true` | Enable FP8 quantization for InternVideo3 |
| `INTERNVIDEO3_THINKING` | `false` | Enable MCR thinking mode |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `hermes` | Backend: `hermes`, `openai`, `auto` |
| `OPENAI_API_BASE` | `http://localhost:11434/v1` | OpenAI-compatible API URL |
| `OPENAI_API_KEY` | (empty) | API key (can be empty for local servers) |
| `OPENAI_MODEL` | `deepseek-ai/DeepSeek-V4-Flash` | Model name for OpenAI-compatible API |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per response |

### Streaming & Live

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVE_STREAM_ENABLED` | `false` | Enable live stream analysis |
| `LIVE_STREAM_URL` | (empty) | RTMP/RTSP/HLS stream URL |
| `LIVE_STREAM_SOURCE` | `rtmp` | Stream type: `rtmp`, `rtsp`, `hls` |
| `LIVE_STREAM_CHUNK_DURATION` | `30.0` | Chunk duration in seconds |
| `LIVE_STREAM_SLIDING_WINDOW` | `300` | Sliding context window in seconds |
| `LIVE_STREAM_AUTO_RECONNECT` | `true` | Auto-reconnect on stream loss |
| `STREAMING_CHUNK_DURATION` | `30.0` | Streaming chunk duration |
| `STREAMING_OVERLAP` | `2.0` | Overlap between chunks (seconds) |
| `STREAMING_INCREMENTAL_INDEX` | `true` | Incremental ChromaDB indexing per chunk |
| `STREAMING_THINKING_ENABLED` | `false` | Enable amortized reasoning during streaming |

### Event-Causal RAG & Knowledge Graph

| Variable | Default | Description |
|----------|---------|-------------|
| `EVENT_CAUSAL_RAG_ENABLED` | `false` | Enable Event-Causal RAG |
| `EVENT_CAUSAL_RAG_INDEX_ON_PROCESS` | `true` | Auto-index events during pipeline processing |
| `EVENT_CAUSAL_RAG_IN_CHAT` | `false` | Use event-causal retrieval in chat |
| `EVENT_SEGMENTATION_STRATEGY` | `auto` | Strategy: `auto`, `llm`, `transcript`, `temporal` |

### Advanced / Agent

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_ENABLED` | `false` | Enable Agentic Video Understanding Agent |
| `ORCHESTRA_ENABLED` | `false` | Enable hierarchical multi-agent orchestrator |
| `CURATOR_ENABLED` | `false` | Enable autonomous MCR video curator |
| `CURATOR_CURIOSITY` | `0.5` | Exploration aggressiveness (0.0–1.0) |
| `CURATOR_MAX_ITERATIONS` | `15` | Max closed-loop iterations |
| `AGENT_CONFIDENCE_ENABLED` | `false` | Enable Robust-TO confidence scoring |
| `PROMETHEUS_ENABLED` | `true` | Enable Prometheus `/metrics` endpoint |
| `TELEMETRY_ENABLED` | `true` | Enable OpenTelemetry tracing |
| `RATE_LIMIT_ENABLED` | `true` | Enable API rate limiting |
| `RATE_LIMIT_CAPACITY` | `100` | Max burst requests per client |
| `RATE_LIMIT_RATE` | `1.6667` | Token refill rate per second (100/min) |
| `FEDERATION_ENABLED` | `false` | Enable federated video search |
| `FEDERATION_PEERS` | (empty) | Comma-separated peer MCP server URLs |
| `ADAPTIVE_SCALING_ENABLED` | `true` | Enable adaptive pipeline scaling |
| `ADAPTIVE_SCALING_POLICY` | `auto` | Policy: `conservative`, `balanced`, `performance`, `auto` |
| `WEBHOOK_URL` | (empty) | Comma-separated webhook callback URLs |
| `WEBHOOK_TIMEOUT` | `5.0` | Webhook POST timeout in seconds |
| `CONVERSATION_MEMORY_ENABLED` | `true` | Enable ChromaDB-backed conversation memory |
| `CONVERSATION_MEMORY_MAX_ENTRIES` | `50` | Max memory entries |
| `CONVERSATION_MEMORY_TTL_DAYS` | `30` | Entry TTL in days |

---

## 📊 Performance (RTX 4070 12 GB)

| Operation | Time (10 min video) | VRAM |
|-----------|-------------------|------|
| Audio extraction (FFmpeg) | ~30s | 0 MB |
| Transcription (large-v3, int8) | ~50s (~12× realtime) | ~4 GB |
| Scene detection | ~20s | ~500 MB |
| Frame extraction + YOLO object detection | ~60s | ~2 GB |
| CLIP scene description (ViT-L-14) | ~30s | ~2 GB |
| OCR (PP-OCRv6) | ~20s | ~500 MB |
| Sprite sheet generation | ~15s | 0 MB |
| Speaker diarization | ~45s | ~1 GB |
| RAG indexing | ~5s | ~1 GB |
| **Total pipeline** | **~3–4 min** | **~12 GB** (sequential) |
| Q&A response | ~2–5s per question | ~1 GB |

---

## 📈 Production Monitoring

The platform exports 20+ **Prometheus metrics** covering pipeline runs, retrieval latency, GPU memory, ChromaDB size, and Q&A performance. A **production-ready Grafana dashboard** is provided at `deploy/grafana-dashboard.json`:

```bash
Grafana → Dashboards → Import → Upload deploy/grafana-dashboard.json
```

Dashboard panels:

- **Pipeline throughput** — runs/s, duration P50/P95/P99, success rate
- **Retrieval latency** — embedding, search, rerank, temporal expansion per stage
- **GPU resources** — VRAM usage, utilization, temperature (via DCGM exporter)
- **System health** — disk usage, error rate, job queue depth
- **Q&A quality** — response latency, tokens/s, requests/s, evaluation scores

### Production Docker Stack

```bash
# Full production stack with DCGM + Caddy + auto-HTTPS
docker compose -f docker-compose.prod.yml up -d

# Set your domain for automatic TLS
export DOMAIN=video.example.com
docker compose -f docker-compose.prod.yml up -d
```

The Caddy reverse proxy provides automatic HTTPS, security headers, compression, and structured access logs.

---

## 🧪 Development

### Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all extras + dev tools
pip install -e ".[all,dev]"

# Install pre-commit hooks
make pre-commit
```

### Available Commands

| Command | Description |
|---------|-------------|
| `make run` | Launch web UI |
| `make run-cli VIDEO=path.mp4 QUERY="question"` | CLI mode |
| `make run-api` | FastAPI server with hot-reload |
| `make run-stream VIDEO=path.mp4` | Stream-process a video |
| `make run-live VIDEO=recording.mp4` | Watch file and process live |
| `make run-mcp` | MCP server (stdio mode) |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |
| `make typecheck` | Run mypy type checker |
| `make check` | All checks (format + lint + typecheck) |
| `make test` | Run all tests (excl. GPU/benchmark/slow) |
| `make test-fast` | Run fast unit tests |
| `make test-cov` | Tests with coverage report |
| `make test-all` | All tests (incl. slow) |
| `make test-bench` | Benchmark tests |
| `make docker-build` | Build Docker image |
| `make clean` | Clean caches and build artifacts |

### Testing

```bash
# Run all standard tests
python -m pytest tests/ -v -x --timeout=120

# Run with coverage
python -m pytest tests/ --cov=video_analysis --cov-report=term

# Run specific test file
python -m pytest tests/test_adaptive_scaler.py -v

# Run evaluation harness
python -m video_analysis --eval

# List evaluation tasks
python -m video_analysis --eval-list

# Run specific evaluation tasks
python -m video_analysis --eval-tasks retrieval,scene
```

The test suite includes **~1,100+ tests** across 40+ test files covering pipeline, RAG, agent, API, streaming, evaluation, event-causal RAG, knowledge graph, webhook, rate limiter, frame compression, and more.

### Evaluation Tasks

| Task | Description |
|------|-------------|
| `retrieval_precision` | Top-k retrieval precision on curated synthetic QA pairs |
| `scene_boundary_accuracy` | Scene detection precision/recall/F1 against ground truth |
| `ocr_accuracy` | OCR character accuracy (CER + word accuracy) on synthetic text images |
| `action_recognition_quality` | Action recognition top-1/top-5 accuracy on synthetic motion video |
| `frame_compression_efficiency` | DINOv2 perceptual frame compression ratios + quality proxy |

Each run produces a timestamped JSON report with pass/fail thresholds and Prometheus gauge export (`va_evaluation_score`). Reports are auto-persisted in `data/eval_reports/`.

---

## 🧪 Optional Capabilities

These features require additional dependencies and GPU VRAM. Enable them via environment variables.

| Capability | Env Var | VRAM | Dependencies |
|-----------|---------|------|--------------|
| **Face Recognition** | `FACE_RECOGNITION_ENABLED=true` | ~1.1 GB | `insightface`, `onnxruntime-gpu` |
| **Action Recognition** | `ACTION_RECOGNITION_ENABLED=true` | ~4 GB | `transformers` (X-CLIP) |
| **Video MLLM** | `VIDEO_MLLM_ENABLED=true` | ~5.4 GB | `vllm`, `decord`, `qwen-vl-utils` |
| **DINOv2 Frame Compression** | `DINO_FRAME_COMPRESSION=true` | ~85 MB | transformers (DINOv2-small) |
| **ColBERTv2 Re-Ranker** | `COLBERT_RERANKER_ENABLED=true` | ~2 GB | `ragatouille` |
| **ColBERT-Att Re-Ranker** | `COLBERT_ATT_RERANKER_ENABLED=true` | ~2 GB | `ragatouille` |
| **Qwen3-VL-Embedding** | `MULTIMODAL_EMBEDDING=true` | ~4 GB | Qwen3-VL models |
| **Event-Causal RAG** | `EVENT_CAUSAL_RAG_ENABLED=true` | ~500 MB | nltk, transformers |
| **Agentic Video Agent** | `AGENT_ENABLED=true` | — | — |
| **Multi-Agent Orchestrator** | `ORCHESTRA_ENABLED=true` | — | — |
| **Video Curator** | `CURATOR_ENABLED=true` | — | — |
| **Live Stream Analysis** | `LIVE_STREAM_ENABLED=true` | — | FFmpeg |
| **Streaming Thinking** | `STREAMING_THINKING_ENABLED=true` | — | — |
| **Federated Search** | `FEDERATION_ENABLED=true` | — | — |
| **Agent Confidence** | `AGENT_CONFIDENCE_ENABLED=true` | — | — |
| **OpenTelemetry** | `TELEMETRY_ENABLED=true` | — | `opentelemetry-api`, `opentelemetry-sdk` |
| **Webcam Capture** | `CAMERA_ENABLED=true` | — | OpenCV |

### Installing Optional Dependencies

```bash
# Install all optional capabilities at once
pip install -e ".[all]"

# Or install specific groups
pip install -e ".[face]"            # Face recognition only
pip install -e ".[mllm]"            # Video MLLM backends
pip install -e ".[ocr]"             # PaddleOCR
pip install -e ".[diarize]"         # PyAnnote diarization
pip install -e ".[telemetry]"       # OpenTelemetry
pip install -e ".[metrics]"         # Prometheus
```

---

## 🗺️ Tech Stack

| Component | Choice | Why |
|-----------|--------|------|
| **Language** | Python 3.11 | Rich ML ecosystem, asyncio, type hints |
| **Backend** | FastAPI | Async, high-performance, auto OpenAPI docs |
| **Web UI** | FastAPI + Jinja2 + HTMX + Alpine.js | Production stack, ~30 KB JS, no Node.js build step |
| **Legacy UI** | Gradio 6+ | Rapid prototyping, Workflow builder |
| **Transcription** | faster-whisper large-v3 (CTranslate2) | ~12× realtime on RTX 4070, int8 quantized |
| **Speaker Diarization** | PyAnnote Audio 3.1 | Gold-standard speaker labeling |
| **Scene Detection** | PySceneDetect 0.7+ | 5 detector modes + FFmpeg fallback |
| **Object Detection** | YOLO (ultralytics) | SOTA speed/accuracy, ByteTrack built-in |
| **Face Recognition** | InsightFace (SCRFD-10G + ArcFace) | Cross-video identity, 512-d embeddings |
| **OCR** | PaddleOCR PP-OCRv6 | Best natural-scene accuracy, configurable tiers |
| **Scene Description** | OpenCLIP ViT-L-14 | Zero-shot classification, configurable model size |
| **Action Recognition** | X-CLIP (microsoft/xclip-base-patch16) | Zero-shot open-vocabulary action detection |
| **Embeddings** | BGE-VL-base + Nomic Embed v1.5 fallback | Single multimodal model, ~0.8 GB VRAM |
| **Vector Store** | ChromaDB | Persistent, local, no server needed |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM + optional ColBERTv2/ColBERT-Att | Dual re-ranking precision |
| **Frame Compression** | DINOv2-small (LongVU-style) | ~85 MB VRAM, perceptual similarity |
| **Video MLLM** | SmolVLM2 / VideoChat-Flash / Qwen3-VL-30B-A3B FP8 / InternVideo3-8B | Quad-backend for scene description + Q&A |
| **LLM** | DeepSeek-V4-Flash (Hermes CLI) / OpenAI-compatible (vLLM, Ollama, llama.cpp, TGI, etc.) | Flexible backends |
| **Video Import** | yt-dlp | 1000+ site support |
| **GPU** | RTX 4070, CUDA 12.8 | All models GPU-accelerated, sequential loading |
| **Metrics** | Prometheus + Grafana | 20+ metrics, production dashboard |
| **Tracing** | OpenTelemetry | Distributed tracing with OTLP export |
| **Container** | Docker, Docker Compose | Multi-stage CUDA build |
| **Auth** | Caddy + Gradio password | Production reverse proxy with auto-HTTPS |

---

## 🗺️ Roadmap

### ✅ Completed

- [x] Core video analysis pipeline (transcription, scene detection, YOLO, CLIP, OCR, diarization, sprite sheets)
- [x] RAG indexing and retrieval (ChromaDB, multi-granularity chunks, re-ranking)
- [x] Chat interface with source citations and clip export
- [x] Production web UI — FastAPI + Jinja2 + HTMX + Alpine.js
- [x] Legacy Gradio web UI
- [x] YouTube URL import (yt-dlp)
- [x] Batch processing queue
- [x] Multi-video library management
- [x] GPU pipeline management (sequential loading for 12 GB VRAM)
- [x] OpenCLIP zero-shot scene classification (ViT-L-14)
- [x] PySceneDetect 0.7+ (Adaptive + Content + Histogram + Hash)
- [x] Speaker diarization (PyAnnote)
- [x] OCR text extraction (PaddleOCR PP-OCRv6)
- [x] Docker deployment (multi-stage CUDA 12.8)
- [x] ColBERTv2 late-interaction re-ranking
- [x] BGE-VL multimodal embedding (replaces dual-model)
- [x] TV-RAG temporal-aware retrieval
- [x] Multi-granularity chunking (60s + 30s sliding + scene + frame)
- [x] Video MLLM integration (SmolVLM2, VideoChat-Flash, Qwen3-VL-30B, InternVideo3-8B)
- [x] Agentic RAG (iterative retrieval with confidence-based early stopping)
- [x] Self-check + re-retrieval (LLM-verified evidence)
- [x] Scene graph retrieval (VGent/ViG-RAG inspired)
- [x] Query classification & routing
- [x] Multi-hop query decomposition
- [x] X-CLIP zero-shot action recognition
- [x] DINOv2 perceptual frame compression (LongVU-style)
- [x] Face recognition (InsightFace SCRFD-10G + ArcFace)
- [x] Entity tracking (ByteTrack via Ultralytics)
- [x] Audio-only processing mode
- [x] Conversation memory (ChromaDB-backed)
- [x] Structured JSON logging (structlog)
- [x] Content-addressable pipeline cache
- [x] PipelineOrchestrator (heuristic video type classification)
- [x] MMR diversity re-ranking
- [x] ColBERT-Att attention-weighted re-ranking
- [x] Real-time streaming/chunked analysis
- [x] Live stream analysis (RTMP/RTSP/HLS)
- [x] Federated video search (MCP-based)
- [x] Video content chaptering (NLTK TextTiling)
- [x] Agentic Video Understanding Agent (7 tools)
- [x] Webcam capture & analysis
- [x] Autonomous Video Curator (MCR closed-loop)
- [x] Hierarchical multi-agent orchestrator (HiCrew-inspired)
- [x] Robust Agent Confidence (Robust-TO inspired)
- [x] Event-Causal RAG (SES graphs, DualStoreMemory)
- [x] Streaming Thinking (amortized reasoning)
- [x] REST API (30+ endpoints, async jobs, SSE streaming)
- [x] Python client SDK
- [x] Async job queue (pure asyncio)
- [x] OpenTelemetry distributed tracing
- [x] Rate limiting + structured error responses
- [x] Structured video reports
- [x] Persistent knowledge graph (SQLite, cross-video)
- [x] Pipeline health monitoring
- [x] Evaluation suite (5 tasks, auto-persisted reports)
- [x] Prometheus metrics + Grafana dashboard
- [x] Webhook notifications
- [x] Adaptive pipeline scaling (per-video auto-tuning)
- [x] MCP tool server (7 tools, stdio + SSE)

---

## ⚖️ License

MIT License — see `LICENSE` for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow the existing code style (ruff, black, mypy) and add tests for new functionality.

```bash
# Before submitting a PR
make check     # Format + lint + typecheck
make test      # Run tests
```
