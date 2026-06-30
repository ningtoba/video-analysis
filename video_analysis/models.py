"""
Data models for video analysis platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrameInfo:
    """A single extracted frame with metadata."""

    timestamp: float  # seconds
    filepath: str  # path to frame image
    scene_id: Optional[int] = None
    llm_description: Optional[str] = None  # LLM Vision description
    llm_objects: List[str] = field(default_factory=list)  # objects identified by LLM
    llm_ocr: Optional[str] = None  # text read by LLM Vision
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SceneInfo:
    """A detected scene with metadata."""

    start_time: float
    end_time: float
    scene_id: int
    description: Optional[str] = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class VideoAnalysis:
    """Analysis result for a processed video."""

    video_id: str
    filename: str
    duration: float
    title: Optional[str] = None
    transcript: List[TranscriptSegment] = field(default_factory=list)
    scenes: List[SceneInfo] = field(default_factory=list)
    frames: List[FrameInfo] = field(default_factory=list)
    llm_summary: Optional[str] = None  # LLM-generated video summary
    error: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Concatenated transcript text."""
        return "\n".join(seg.text for seg in self.transcript)


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str
    sources: List[dict] = field(default_factory=list)


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
