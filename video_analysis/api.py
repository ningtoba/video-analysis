"""
REST API for the video analysis platform.

Simplified: video processing, Q&A, search, and library management.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from video_analysis.chat import VideoChat
from video_analysis.config import Config, load_settings, save_settings
from video_analysis.pipeline import VideoPipeline
from video_analysis.model_manager import WHISPER_MODELS, download_whisper_model

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


class SettingsUpdate(BaseModel):
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_model: Optional[str] = None
    whisper_model: Optional[str] = None
    frame_rate: Optional[float] = None
    max_frames_for_llm: Optional[int] = None
    scene_threshold: Optional[float] = None
    host: Optional[str] = None
    port: Optional[int] = None


class ModelDownloadRequest(BaseModel):
    model_name: str


# ── Model download state ─────────────────────────────────────────


_download_status: dict = {
    "downloading": False,
    "current": None,
    "error": None,
    "available": list(WHISPER_MODELS.keys()),
}
_download_lock = threading.Lock()

# ── Router ──────────────────────────────────────────────────────────


def create_router(config: Config) -> APIRouter:
    """Create the API router with all endpoints."""
    router = APIRouter(prefix="/api")
    pipeline = VideoPipeline(config)
    chat = VideoChat(config=config)

    # Store in-memory analysis results (persisted to JSON files)
    analyses: Dict[str, dict] = {}
    _load_analyses(config.data_dir / "analyses", analyses)

    # Load persisted settings on startup and apply to config
    saved_settings = load_settings(config.data_dir)
    if saved_settings:
        _apply_settings_to_config(config, saved_settings)

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

    @router.post("/import-url")
    async def import_url(body: dict):
        """Download a video from a URL (YouTube, etc.) and return local path."""
        url = body.get("url", "").strip()
        if not url:
            raise HTTPException(400, "No URL provided")

        if not config.yt_dlp_enabled:
            raise HTTPException(400, "YouTube/URL import is disabled")

        try:
            import yt_dlp
        except ImportError:
            raise HTTPException(500, "yt-dlp is not installed")

        video_id = str(uuid.uuid4())[:8]
        ydl_opts = {
            "format": "best[height<=1080]",
            "outtmpl": str(config.videos_dir / f"{video_id}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                ext = info.get("ext", "mp4")
                filename = info.get("title", video_id)
        except Exception as e:
            raise HTTPException(500, f"Failed to download URL: {e}")

        video_path = config.videos_dir / f"{video_id}.{ext}"
        if not video_path.exists():
            raise HTTPException(500, "Downloaded file not found")

        return {
            "video_id": video_id,
            "filename": filename,
            "path": str(video_path),
            "message": "Downloaded. Process with POST /api/videos/process",
        }

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
    # Register settings and model endpoints
    add_settings_endpoints(router, config)
    add_model_endpoints(router, config)

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

# ── Settings persistence ─────────────────────────────────────────────


def _load_settings_json(path: Path) -> dict:
    """Load settings from JSON file, return empty dict if missing."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.warning("Failed to load settings from %s", path)
    return {}


def _save_settings_json(path: Path, data: dict):
    """Save settings to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Settings saved to %s", path)


def _config_to_dict(cfg: Config) -> dict:
    """Serialize Config to a JSON-safe dict for the settings API."""
    return {
        "llm_provider": cfg.llm_provider,
        "llm_api_key": cfg.llm_api_key,
        "llm_api_base": cfg.llm_api_base,
        "llm_model": cfg.llm_model,
        "llm_temperature": cfg.llm_temperature,
        "llm_max_tokens": cfg.llm_max_tokens,
        "whisper_model": cfg.whisper_model,
        "whisper_device": cfg.whisper_device,
        "whisper_compute_type": cfg.whisper_compute_type,
        "frame_rate": cfg.frame_rate,
        "max_frames_for_llm": cfg.max_frames_for_llm,
        "scene_threshold": cfg.scene_threshold,
        "scene_detector": cfg.scene_detector,
        "processing_mode": cfg.processing_mode,
        "host": cfg.host,
        "port": cfg.port,
        "data_dir": str(cfg.data_dir),
    }


def _apply_settings_to_config(cfg: Config, settings: dict):
    """Update Config fields from a settings dict (only known keys)."""
    for key, value in settings.items():
        if hasattr(cfg, key):
            # Type-coerce based on current type
            current = getattr(cfg, key)
            if isinstance(current, bool):
                setattr(cfg, key, bool(value))
            elif isinstance(current, int):
                setattr(cfg, key, int(value))
            elif isinstance(current, float):
                setattr(cfg, key, float(value))
            elif isinstance(current, str):
                setattr(cfg, key, str(value))
            elif isinstance(current, Path):
                setattr(cfg, key, Path(str(value)))
            else:
                setattr(cfg, key, value)


# ── Settings endpoints ───────────────────────────────────────────────


def add_settings_endpoints(router: APIRouter, config: Config):
    """Attach settings GET/PUT endpoints to the router."""

    @router.get("/settings")
    async def get_settings():
        """Return current configuration."""
        return _config_to_dict(config)

    @router.put("/settings")
    async def update_settings(body: dict):
        """Update configuration settings and persist to settings.json."""
        _apply_settings_to_config(config, body)
        settings_path = config.data_dir / "settings.json"
        _save_settings_json(settings_path, _config_to_dict(config))
        return {"status": "saved", "settings": _config_to_dict(config)}


# ── Model endpoints ──────────────────────────────────────────────────


def add_model_endpoints(router: APIRouter, config: Config):
    """Attach model list/download/status endpoints to the router."""
    global _download_status

    @router.get("/models")
    async def list_models():
        """List all known Whisper models with download status and sizes."""
        from video_analysis.model_manager import WHISPER_MODELS

        # Check which models are already downloaded
        import os
        import faster_whisper  # noqa: F401 — ensures faster_whisper is importable for cache path

        cache_dir = os.environ.get(
            "WHISPER_CACHE_DIR",
            str(Path(os.path.expanduser("~")) / ".cache" / "faster_whisper"),
        )
        cache_path = Path(cache_dir)

        results = []
        for name, info in WHISPER_MODELS.items():
            model_dir = cache_path / name
            downloaded = model_dir.exists() and any(model_dir.iterdir())
            results.append({
                "name": name,
                "params": info["params"],
                "vram_mb": info["vram_mb"],
                "speed": info["speed"],
                "wer": info["wer"],
                "downloaded": downloaded,
            })

        current = config.whisper_model
        return {
            "models": results,
            "current_model": current,
            "cache_dir": str(cache_path),
        }

    @router.post("/models/download")
    async def download_model(body: dict):
        """Start downloading a Whisper model in the background."""
        model_name = body.get("model_name", "")
        from video_analysis.model_manager import WHISPER_MODELS, download_whisper_model

        if model_name not in WHISPER_MODELS:
            raise HTTPException(400, f"Unknown model: {model_name}")

        with _download_lock:
            if _download_status["downloading"]:
                raise HTTPException(409, f"Already downloading '{_download_status['current']}'")
            _download_status["downloading"] = True
            _download_status["current"] = model_name
            _download_status["error"] = None

        def _do_download():
            try:
                download_whisper_model(model_name)
            except Exception as e:
                logger.error("Model download failed: %s", e)
                with _download_lock:
                    _download_status["error"] = str(e)
            finally:
                with _download_lock:
                    _download_status["downloading"] = False

        thread = threading.Thread(target=_do_download, daemon=True)
        thread.start()

        return {"status": "downloading"}

    @router.get("/models/status")
    async def model_download_status():
        """Return current download status."""
        with _download_lock:
            return dict(_download_status)
