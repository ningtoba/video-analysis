"""
Data models for video analysis platform.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import timedelta


@dataclass
class FrameInfo:
    """A single extracted frame with metadata."""

    timestamp: float  # seconds
    filepath: str  # path to frame image
    scene_id: Optional[int] = None
    description: Optional[str] = None
    objects: List[dict] = field(default_factory=list)
    # Each object dict can now include:
    #   {"label": "person", "confidence": 0.95, "bbox": [...],
    #    "track_id": 1}  # ByteTrack persistent ID across frames (optional)
    ocr_text: Optional[str] = None
    action: Optional[str] = None  # X-CLIP action recognition label
    action_confidence: Optional[float] = None  # confidence score for the action
    # Face detection (InsightFace, v0.26.0)
    faces: List[dict] = field(default_factory=list)
    # Each face dict:
    #   {"bbox": [x1,y1,x2,y2], "confidence": 0.99, "embedding": [...],
    #    "face_id": "PERSON_0", "gender": "Male", "age": 30}
    # Quality screening metadata (v0.21.0)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SceneInfo:
    """A detected scene with metadata."""

    scene_id: int
    start_time: float
    end_time: float
    key_frames: List[FrameInfo] = field(default_factory=list)
    transcript: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[dict] = field(default_factory=list)


@dataclass
class VideoIndex:
    """Indexed representation of a processed video."""

    video_id: str
    filename: str
    duration: float
    filepath: str
    scenes: List[SceneInfo] = field(default_factory=list)
    transcript: List[TranscriptSegment] = field(default_factory=list)
    full_transcript: str = ""
    chunks: List[dict] = field(default_factory=list)
    sprite_sheet: Optional[str] = None  # path to sprite sheet image
    sprite_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "filename": self.filename,
            "duration": self.duration,
            "filepath": self.filepath,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "key_frames": [
                        {
                            "timestamp": f.timestamp,
                            "filepath": f.filepath,
                            "description": f.description,
                            "objects": f.objects,
                            "ocr_text": f.ocr_text,
                            "action": f.action,
                            "action_confidence": f.action_confidence,
                            "faces": f.faces,
                            "metadata": f.metadata,
                        }
                        for f in s.key_frames
                    ],
                    "transcript": s.transcript,
                    "summary": s.summary,
                }
                for s in self.scenes
            ],
            "transcript": [
                {
                    "start": t.start,
                    "end": t.end,
                    "text": t.text,
                    "speaker": t.speaker,
                }
                for t in self.transcript
            ],
            "full_transcript": self.full_transcript,
            "sprite_sheet": self.sprite_sheet,
            "sprite_metadata": self.sprite_metadata,
        }


@dataclass
class ChatSource:
    """A source citation for a chat response."""

    text: str
    timestamp: float
    frame_path: Optional[str] = None
    scene_id: Optional[int] = None
    relevance_score: float = 0.0


@dataclass
class ChatMessage:
    """A single chat message with sources."""

    role: str  # "user" or "assistant"
    content: str
    sources: List[ChatSource] = field(default_factory=list)


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm."""
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
