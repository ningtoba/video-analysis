# Changelog

> **Note:** This changelog is ~190 KB. Detailed entries for versions before 0.44.0
> are preserved in git history (`git log --oneline`). Consider using GitHub Releases
> for curated release notes and trimming this file to the most recent versions.
> To view full history: `git log --reverse --format="%H %s" -- CHANGELOG.md`

## 0.61.0 (2026-06-30) — Slim Refactoring: LLM Vision API Architecture

### Summary

Complete architecture simplification. Replaced 8+ local vision models with a
single LLM Vision API call (GPT-4o/Claude/Gemini/DeepSeek/Ollama) for all
vision tasks — scene understanding, object detection, OCR, people detection,
and action recognition.

**Before:** 55 files / 37,438 lines Python / 8+ ML models / 30+ deps /
674-line .env.example / 3-stage CUDA Dockerfile (1.4 GB+)

**After:** 22 files / 5,898 lines Python / 1 ML model (whisper ASR only) /
15 deps / 15-line .env.example / single-stage Dockerfile (2.26 GB)

### Removed Modules (33)

- `agent.py`, `agent_confidence.py` — Agentic Video Agent & confidence scoring
- `orchestra.py`, `orchestrator.py` — Hierarchical multi-agent orchestrator
- `event_rag.py` — Event-Causal RAG with SES graphs
- `colbert_reranker.py`, `colbert_att_reranker.py` — ColBERT re-rankers
- `scene_graph.py`, `query_router.py` — Scene graph retrieval & query routing
- `self_check.py` — LLM self-check verification loop
- `streaming.py`, `streaming_think.py`, `stream_chat.py` — Streaming pipeline
- `video_mllm.py`, `backends/` — Four Video MLLM backends
- `chapters.py` — NLTK TextTiling chaptering
- `action.py` — X-CLIP zero-shot action recognition
- `face.py` — InsightFace face detection
- `federation.py` — Federated video search
- `curator.py` — Autonomous video curator
- `evaluation.py`, `benchmark.py` — Evaluation suite & benchmarks
- `frame_compression.py` — DINOv2 perceptual compression
- `knowledge_graph.py` — SQLite knowledge graph
- `pipeline_health.py` — Health monitoring dashboard
- `metrics.py` — Prometheus metrics
- `telemetry.py` — OpenTelemetry tracing
- `report.py` — Structured report generator
- `flow.py` — Optical flow analysis
- `memory.py` — Conversation memory
- `mcp_server.py` — MCP tool server
- `config_store.py` — Persistent config store
- `adaptive_scaler.py` — Adaptive pipeline scaler
- `ui/app.py`, `ui/camera.py`, `ui/workflow.py`, `ui/comparison.py`,
  `ui/event_timeline.py`, `ui/knowledge_graph.py`, `ui/monitor.py`,
  `ui/utils.py`, `ui/routes/` — Legacy Gradio app & UI pages

### Architecture Changes

- **LLM Vision API** replaces YOLO, CLIP, PaddleOCR, InsightFace, BGE-VL,
  cross-encoder, and all MLLM backends — one API call handles all vision tasks
- **faster-whisper** remains as the only local ML model for ASR
- **Removed ChromaDB** — analysis stored as JSON, no vector database needed
- **Removed all embedding models** (BGE-VL, Nomic-Embed, Qwen3-VL-Embedding)
- **Removed all re-rankers** (cross-encoder, ColBERTv2, ColBERT-Att)
- **Removed all RAG complexity** (multi-hop, agentic, MMR, temporal decay)

### Configuration

- Simplified `.env.example` from 674 lines to 15 settings
- All settings optional except LLM API key
- LLM providers: openai, anthropic, gemini, deepseek, ollama
- Auto VRAM detection selects optimal Whisper model at startup

### Docker

- Single-stage `python:3.11-slim` Dockerfile (was 3-stage CUDA)
- Entrypoint script auto-detects GPU and downloads best Whisper model
- GPU support via docker-compose `deploy.resources.reservations.devices`
- Single docker-compose.yml for all environments
- Removed `docker-compose.prod.yml`, `Caddyfile`, `Makefile`

### Dependencies

- Reduced from 50+ to 15 core deps
- Removed: torch, torchvision, transformers, sentence-transformers,
  open-clip-torch, ultralytics, chromadb, langchain-text-splitters,
  insightface, paddlepaddle, paddleocr, pyannote.audio, vllm, decord,
  nltk, openai, anthropic, structlog, gradio, mcp, onnxruntime

### Code Size

| Metric | Before | After |
|--------|--------|-------|
| Python files | 55 | 22 |
| Lines of code | 37,438 | 5,898 |
| Dependencies | 50+ | 15 |
| .env.example lines | 674 | 15 |
| Docker stages | 3 | 1 |
| Docker image size | 1.4 GB+ (CUDA) | 2.26 GB (slim) |


## Older Versions (0.1.0 — 0.60.0)

Detailed changelog entries for versions 0.1.0 through 0.60.0 are preserved in git history.
View with: `git log --reverse --format="%H %s" -- CHANGELOG.md`

Key milestones in that range:
- **0.60.0** — Adaptive pipeline stage scaling & resource management
- **0.59.0** — Webhook notification system & infrastructure tests
- **0.58.0–0.54.0** — Event-causal RAG, streaming thinking, InternVideo3 backend
- **0.52.0–0.48.0** — Knowledge graph, multi-agent orchestrator, telemetry, evaluation suite
- **0.44.0–0.30.0** — REST API, live stream analysis, ColBERT reranking, DINOv2 compression
- **0.29.0–0.10.0** — Prometheus metrics, face recognition, MCP server, pipeline orchestrator, tiered storage, action recognition, video MLLM backends, scene graphs, entity tracking
- **0.9.0–0.1.0** — Initial development: Whisper ASR, LLM Vision, search, chat, FFmpeg pipeline

All features from these versions were either consolidated into the current slim architecture
(ASR + LLM Vision API) or removed during the 0.61.0 refactoring.
