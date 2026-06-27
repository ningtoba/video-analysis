"""Video Analysis Platform — Self-hosted video understanding with RAG chatbot.

Modules:
- pipeline: Video ingestion and analysis (transcription, scene detection, OCR, object detection)
- rag: Vector store indexing, retrieval, and re-ranking
- models: Data models and schemas
- chat: LLM-powered Q&A over video context
- config: Configuration management
- llm_provider: Self-contained LLM provider abstraction (Hermes CLI / OpenAI-compatible API)
- stream_chat: MLLM streaming Q&A — token-by-token LLM response streaming (v0.41.0)
- self_check: LLM-based self-check verification + re-retrieval (v0.27.0)
- agent: Agentic Video Understanding Agent with multi-tool dispatch (v0.36.0)
- chapters: Video content chaptering — auto topic segmentation & chapter generation (v0.37.0)
- api: Full REST API layer with OpenAPI docs (v0.41.0)
- client: Python API client for the REST API (v0.49.0)
- telemetry: OpenTelemetry distributed tracing — pipeline, RAG, and API spans (v0.49.0)
- rate_limiter: In-memory token bucket rate limiter for the REST API (v0.49.0)
- error_handlers: Structured JSON error responses for the REST API (v0.49.0)
- job_queue: In-process async job queue for background video processing (v0.43.0)
| agent_confidence: Robust-TO inspired confidence-aware agent — per-frame trustworthiness, evidence scoring, tiered weighting (v0.50.0)
| report: Structured video report generation — comprehensive JSON schema from pipeline results (v0.50.0)
"""

from video_analysis import (
    pipeline,
    rag,
    models,
    chat,
    config,
    scene_graph,
    query_router,
    storage,
    quality,
    memory,
    frame_compression,
    streaming,
    federation,
    llm_provider,
    stream_chat,
    curator,
)

# face module is optional (requires insightface) — import on demand only

# Workflow module (requires Gradio 6.17+ for gr.Workflow)

# Qwen3-VL backend (requires vLLM or transformers — optional, heavy model)

# api and stream_chat modules are imported on demand

__version__ = "0.50.0"

# Re-export streaming module public API at package level
from video_analysis.streaming import (
    StreamingPipeline,
    StreamingChunkResult,
)  # noqa: E402, F401
