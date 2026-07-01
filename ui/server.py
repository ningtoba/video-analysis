"""
FastAPI web server for the video analysis platform.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from video_analysis.api import create_router
from video_analysis.chat import VideoChat
from video_analysis.config import Config
from video_analysis.llm_provider import LLMProviderConfig, get_llm_provider
from video_analysis.stream_manager import StreamManager
from video_analysis.event_memory import EventMemory

logger = logging.getLogger(__name__)


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or Config()

    app = FastAPI(
        title="Video Analysis Platform",
        version="0.62.0",
        description="Self-hosted video analysis with ASR + LLM Vision — upload & real-time stream",
    )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Set up templates
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    # Include API router
    api_router = create_router(config)
    app.include_router(api_router)

    # Store config on app state
    app.state.config = config

    # Stream manager
    stream_manager = StreamManager()
    app.state.stream_manager = stream_manager

    # Event memory for stream chat RAG
    event_memory = EventMemory(
        db_path=str(config.data_dir / "event_memory.db"),
        retention_days=30,
    )
    app.state.event_memory = event_memory

    # Chat instance
    chat = VideoChat(config=config)

    # ── Lazy LLM init for stream engine ──────────────────────────

    def _init_stream_llm():
        """Initialize the LLM chat function for stream engines from current config."""
        if stream_manager._llm_chat_fn:
            return
        if not config.llm_api_key:
            logger.warning("No LLM API key configured — stream LLM analysis disabled")
            return
        try:
            llm_cfg = LLMProviderConfig(
                provider=config.llm_provider,
                api_key=config.llm_api_key,
                api_base=config.llm_api_base,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            llm = get_llm_provider(llm_cfg)
            stream_manager.set_llm_chat_fn(
                lambda msgs, imgs=None, sys=None: (
                    llm.chat_with_images(msgs, imgs, sys) if imgs else llm.chat(msgs, sys)
                )
            )
            logger.info("Stream LLM initialized: %s/%s", config.llm_provider, config.llm_model)
        except Exception as e:
            logger.warning("Failed to init stream LLM: %s", e)

    # ── Web UI Routes ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {"request": request})

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        return templates.TemplateResponse(request, "settings.html", {"request": request})

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        return templates.TemplateResponse(request, "models.html", {"request": request})

    @app.get("/stream", response_class=HTMLResponse)
    async def stream_page(request: Request):
        """Live stream management page."""
        return templates.TemplateResponse(request, "stream.html", {"request": request})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ── Upload ────────────────────────────────────────────────────

    @app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)):
        if not file.filename:
            raise HTTPException(400, "No file provided")
        video_id = str(uuid.uuid4())[:8]
        ext = Path(file.filename).suffix or ".mp4"
        save_path = config.videos_dir / f"{video_id}{ext}"
        content = await file.read()
        save_path.write_bytes(content)
        return {
            "video_id": video_id,
            "filename": file.filename,
            "path": str(save_path),
            "message": "Uploaded. Process with POST /api/videos/process",
        }

    # ── Chat ──────────────────────────────────────────────────────

    @app.post("/api/chat")
    async def chat_endpoint(question: str = Form(...), video_id: str = Form(...)):
        from video_analysis.api import _load_analyses, _dict_to_analysis
        analyses: dict = {}
        _load_analyses(config.data_dir / "analyses", analyses)
        analysis_dict = analyses.get(video_id)
        if not analysis_dict:
            raise HTTPException(404, f"Analysis not found for {video_id}")
        analysis = _dict_to_analysis(analysis_dict)
        answer = chat.ask(question, analysis)
        if answer:
            return {"answer": answer}
        return {"error": "Failed to generate answer"}

    # ── Stream API Endpoints ──────────────────────────────────────

    @app.post("/api/stream/start")
    async def stream_start(
        source: str = Form(...),
        fps: float = Form(1.0),
        interval: float = Form(30.0),
        motion_threshold: float = Form(0.02),
        buffer_seconds: float = Form(300.0),
    ):
        """Start a real-time stream."""
        _init_stream_llm()
        if not stream_manager._llm_chat_fn:
            raise HTTPException(400, "LLM not configured — set API key in Settings first")
        try:
            db_path = str(config.data_dir / "stream_events.db")
            stream_id = stream_manager.start(
                source=source,
                fps=fps,
                interval=interval,
                motion_threshold=motion_threshold,
                buffer_seconds=buffer_seconds,
                db_path=db_path,
            )
            return {"stream_id": stream_id, "status": "started"}
        except Exception as e:
            logger.error("Stream start failed: %s", e)
            raise HTTPException(500, str(e))

    @app.post("/api/stream/stop")
    async def stream_stop():
        """Stop all streams."""
        stream_manager.stop_all()
        return {"status": "stopped"}

    @app.get("/api/stream/status")
    async def stream_status():
        """Get status of all streams."""
        streams = stream_manager.list()
        return {
            "running": len(streams) > 0,
            "count": len(streams),
            "streams": streams,
        }

    @app.get("/api/stream/events")
    async def stream_events(limit: int = Query(50, ge=1, le=500)):
        """Get events from the first active stream (simplified single-stream view)."""
        streams = stream_manager.list()
        if not streams:
            return {"events": []}
        sid = streams[0]["stream_id"]
        events = stream_manager.get_events(sid, limit)
        return {
            "stream_id": sid,
            "events": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp,
                    "description": e.description[:200] if e.description else "",
                    "triggered_by": e.triggered_by,
                    "motion_score": e.motion_score,
                    "frame_path": e.frame_path,
                }
                for e in events
            ],
        }

    @app.post("/api/stream/chat")
    async def stream_chat(question: str = Form(...)):
        """Ask a question about stream events using RAG over event memory."""
        em: EventMemory = app.state.event_memory

        # Build LLM chat function from current config
        from video_analysis.llm_provider import LLMProviderConfig, get_llm_provider
        llm_cfg = LLMProviderConfig(
            provider=config.llm_provider,
            api_key=config.llm_api_key,
            api_base=config.llm_api_base,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        llm = get_llm_provider(llm_cfg)

        def chat_fn(messages):
            return llm.chat(messages)

        answer = em.query_natural_language(
            stream_id="_stream_",
            question=question,
            llm_chat_fn=chat_fn,
        )

        if answer:
            return {"answer": answer}
        return {"error": "Failed to get answer"}
    # ── Frames ────────────────────────────────────────────────────

    @app.get("/api/videos/{video_id}/frames/{frame_file:path}")
    async def get_frame(video_id: str, frame_file: str):
        from fastapi.responses import FileResponse
        frame_path = config.frames_dir / video_id / frame_file
        if frame_path.exists():
            return FileResponse(str(frame_path))
        raise HTTPException(404, "Frame not found")

    # Health check endpoint from health.py
    try:
        from ui.health import add_health_endpoints
        add_health_endpoints(app, config)
    except ImportError:
        pass

    return app
