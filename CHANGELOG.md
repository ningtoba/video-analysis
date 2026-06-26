# Changelog

## 0.1.0 (2026-06-26)

### Initial Release

- **Core pipeline**: FFmpeg-based scene detection, frame extraction, faster-whisper transcription, YOLO object detection
- **RAG engine**: ChromaDB vector store with hybrid BM25/dense retrieval, cross-encoder re-ranking, temporal context expansion
- **Chat interface**: Video Q&A with source citations (clickable timestamps), conversation history
- **Web UI**: Gradio Blocks with dark theme, video upload, real-time analysis progress, streaming chat
- **CLI mode**: Batch processing and Q&A from the terminal
- **GPU acceleration**: Full CUDA support for RTX 4070 (faster-whisper, sentence-transformers, torch)
- **All local**: No API keys required — runs entirely on self-hosted hardware

### Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py     # Package init
│   ├── __main__.py     # CLI/UI entry point
│   ├── config.py       # Configuration management
│   ├── models.py       # Data models (Scene, Frame, VideoIndex, ChatMessage)
│   ├── pipeline.py     # Video processing pipeline
│   ├── rag.py          # Vector RAG indexing and retrieval
│   └── chat.py         # LLM-powered Q&A
├── ui/
│   └── app.py          # Gradio web interface
├── tests/
│   └── test_basic.py   # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```
