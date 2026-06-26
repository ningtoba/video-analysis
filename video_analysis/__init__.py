"""
Video Analysis Platform — Self-hosted video understanding with RAG chatbot.

Modules:
- pipeline: Video ingestion and analysis (transcription, scene detection, OCR, object detection)
- rag: Vector store indexing, retrieval, and re-ranking
- models: Data models and schemas
- chat: LLM-powered Q&A over video context
- config: Configuration management
"""

from video_analysis import (
    pipeline,
    rag,
    models,
    chat,
    config,
    scene_graph,
    query_router,
)

__version__ = "0.19.0"
