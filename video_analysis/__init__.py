"""Video Analysis Platform — Self-hosted video analysis with LLM-powered Q&A.

Uses faster-whisper for transcription and any LLM Vision API (OpenAI, Anthropic,
Gemini, DeepSeek, Ollama) for scene understanding, object detection, OCR, and Q&A.
"""

try:
    from importlib.metadata import version
    __version__ = version("video-analysis")
except Exception:
    __version__ = "0.0.0"
