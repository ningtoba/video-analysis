"""
REST API for the video analysis platform.

Simplified: video processing, Q&A, search, and library management.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from video_analysis.chat import VideoChat
from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline

logger = logging.getLogger(__name__)

# ── Request/Response models ─────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    error: Optional[str] = None


class ProcessRequest(BaseModel):
    video_path: str
    video_id: Optional[str] = None
    skip_llm_vision: bool = False


class ProcessResponse(BaseModel):
    video_id: str
    status: str
    error: Optional[str] = None
    duration: Optional[float] = None
    num_transcript_segments: int = 0
    num_scenes: int = 0
    num_frames: int = 0


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    duration: float
    title: Optional[str] = None
    has_transcript: bool = False
    has_analysis: bool = False
    num_scenes: int = 0
    num_frames: int = 0


# ── Router ──────────────────────────────────────────────────────────


def create_router(config: Config) -> APIRouter:
    """Create the API router with all endpoints."""
    router = APIRouter(prefix="/api")
    pipeline = VideoPipeline(config)
    chat = VideoChat(config=config)

    # Store in-memory analysis results (persisted to JSON files)
    analyses: Dict[str, dict] = {}
    _load_analyses(config.data_dir / "analyses", analyses)

    @router.post("/videos/process")
    async def process_video(req: ProcessRequest) -> ProcessResponse:
        """Process a video file."""
        path = Path(req.video_path)
        if not path.exists():
            raise HTTPException(404, f"Video not found: {req.video_path}")

        try:
            analysis = pipeline.process_video(
                str(path),
                video_id=req.video_id,
                skip_llm_vision=req.skip_llm_vision,
            )

            # Store analysis
            analysis_dir = config.data_dir / "analyses"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis_path = analysis_dir / f"{analysis.video_id}.json"

            # Convert to dict for JSON storage
            analysis_dict = _analysis_to_dict(analysis)
            analysis_path.write_text(json.dumps(analysis_dict, indent=2))
            analyses[analysis.video_id] = analysis_dict

            return ProcessResponse(
                video_id=analysis.video_id,
                status="error" if analysis.error else "complete",
                error=analysis.error,
                duration=analysis.duration,
                num_transcript_segments=len(analysis.transcript),
                num_scenes=len(analysis.scenes),
                num_frames=len(analysis.frames),
            )
        except Exception as e:
            logger.error("Processing failed: %s", e)
            raise HTTPException(500, str(e))

    @router.post("/videos/{video_id}/query")
    async def query_video(video_id: str, req: QueryRequest) -> QueryResponse:
        """Ask a question about a processed video."""
        analysis_dict = analyses.get(video_id)
        if not analysis_dict:
            raise HTTPException(404, f"Video {video_id} not found")

        from video_analysis.models import VideoAnalysis
        analysis = _dict_to_analysis(analysis_dict)

        answer = chat.ask(req.question, analysis)
        if answer:
            return QueryResponse(answer=answer)
        return QueryResponse(error="Failed to get answer")

    @router.get("/videos")
    async def list_videos() -> List[VideoInfo]:
        """List all processed videos."""
        results = []
        for vid, data in analyses.items():
            results.append(VideoInfo(
                video_id=vid,
                filename=data.get("filename", vid),
                duration=data.get("duration", 0),
                title=data.get("title"),
                has_transcript=len(data.get("transcript", [])) > 0,
                has_analysis=data.get("llm_summary") is not None,
                num_scenes=len(data.get("scenes", [])),
                num_frames=len(data.get("frames", [])),
            ))
        return results

    @router.get("/videos/{video_id}")
    async def get_video(video_id: str) -> dict:
        """Get full analysis for a video."""
        analysis_dict = analyses.get(video_id)
        if not analysis_dict:
            raise HTTPException(404, f"Video {video_id} not found")
        return analysis_dict

    @router.delete("/videos/{video_id}")
    async def delete_video(video_id: str):
        """Delete a video analysis."""
        if video_id in analyses:
            del analyses[video_id]
            # Delete JSON file
            analysis_path = config.data_dir / "analyses" / f"{video_id}.json"
            if analysis_path.exists():
                analysis_path.unlink()
            return {"status": "deleted"}
        raise HTTPException(404, f"Video {video_id} not found")

    @router.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return router


# ── Helper functions ────────────────────────────────────────────────


def _analysis_to_dict(analysis) -> dict:
    """Convert VideoAnalysis to a JSON-serializable dict."""
    return {
        "video_id": analysis.video_id,
        "filename": analysis.filename,
        "duration": analysis.duration,
        "title": analysis.title,
        "llm_summary": analysis.llm_summary,
        "error": analysis.error,
        "transcript": [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in analysis.transcript
        ],
        "scenes": [
            {"scene_id": s.scene_id, "start_time": s.start_time, "end_time": s.end_time,
             "description": s.description}
            for s in analysis.scenes
        ],
        "frames": [
            {"timestamp": f.timestamp, "filepath": f.filepath, "scene_id": f.scene_id,
             "llm_description": f.llm_description, "llm_objects": f.llm_objects,
             "llm_ocr": f.llm_ocr}
            for f in analysis.frames
        ],
    }


def _dict_to_analysis(d: dict):
    """Convert a dict back to a VideoAnalysis (or dict-like object)."""
    from video_analysis.models import VideoAnalysis, TranscriptSegment, SceneInfo, FrameInfo
    analysis = VideoAnalysis(
        video_id=d.get("video_id", ""),
        filename=d.get("filename", ""),
        duration=d.get("duration", 0),
        title=d.get("title"),
        llm_summary=d.get("llm_summary"),
        error=d.get("error"),
        transcript=[
            TranscriptSegment(**s) for s in d.get("transcript", [])
        ],
        scenes=[
            SceneInfo(**s) for s in d.get("scenes", [])
        ],
        frames=[
            FrameInfo(**f) for f in d.get("frames", [])
        ],
    )
    return analysis


def _load_analyses(dir_path: Path, analyses: Dict[str, dict]):
    """Load analysis JSON files from disk into memory."""
    if not dir_path.exists():
        return
    for f in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            vid = data.get("video_id", f.stem)
            analyses[vid] = data
        except Exception as e:
            logger.warning("Failed to load analysis %s: %s", f.name, e)
