"""
Video Analysis Platform — Self-hosted video understanding with RAG chatbot.

Modules:
- pipeline: Video ingestion and analysis (transcription, scene detection, OCR, object detection)
- rag: Vector store indexing, retrieval, and re-ranking
- models: Data models and schemas
- chat: LLM-powered Q&A over video context
- config: Configuration management
- self_check: LLM-based self-check verification + re-retrieval (v0.27.0)
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
)

# face module is optional (requires insightface) — import on demand only

__version__ = "0.28.0"
