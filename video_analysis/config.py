"""
Configuration for the video analysis platform.
"""

from pathlib import Path
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Central configuration for the platform."""

    # Paths
    data_dir: Path = Path(os.environ.get("VIDEO_ANALYSIS_DATA", "data"))
    video_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    thumbnails_dir: Path = field(init=False)
    chroma_path: Path = field(init=False)

    # Processing
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"

    # Frame extraction
    frame_rate: float = 0.5  # 1 frame per 2 seconds default
    scene_threshold: float = 0.3  # PySceneDetect sensitivity

    # YOLO
    yolo_model: str = "yolo26x.pt"  # Latest YOLO26
    yolo_confidence: float = 0.25

    # RAG
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_collection: str = "video_analysis"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    temporal_window: int = 1  # neighbors on each side

    # LLM
    llm_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # Clip export
    clip_export_dir: Path = field(init=False)

    # OCR (PaddleOCR)
    ocr_enabled: bool = True
    ocr_confidence: float = 0.3

    # Diarization (PyAnnote)
    diarize_enabled: bool = True

    # UI
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860
    ui_share: bool = False
    # Library
    library_max_videos: int = 50

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.video_dir = self.data_dir / "videos"
        self.frames_dir = self.data_dir / "frames"
        self.audio_dir = self.data_dir / "audio"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        self.chroma_path = self.data_dir / "chroma"
        self.clip_export_dir = self.data_dir / "clips"
        for d in [
            self.data_dir,
            self.video_dir,
            self.frames_dir,
            self.audio_dir,
            self.thumbnails_dir,
            self.clip_export_dir,
            self.chroma_path,
        ]:
            d.mkdir(parents=True, exist_ok=True)
