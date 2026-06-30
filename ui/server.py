"""
FastAPI web server for the video analysis platform.

Simplified: single-page app with video upload, processing, and Q&A.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from video_analysis.api import create_router
from video_analysis.chat import VideoChat
from video_analysis.config import Config

logger = logging.getLogger(__name__)


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or Config()

    app = FastAPI(
        title="Video Analysis Platform",
        version="0.61.0",
        description="Self-hosted video analysis with ASR + LLM Vision",
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

    # Store config on app state for settings access
    app.state.config = config
    app.state.settings_path = config.data_dir / "settings.json"

    # Chat instance (lazy)
    chat = VideoChat(config=config)

    # ── Web UI Routes ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Main page."""
        return templates.TemplateResponse(request, "index.html", {"request": request})

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        """Settings page."""
        return templates.TemplateResponse(request, "settings.html", {"request": request})

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        """Model download management page."""
        return templates.TemplateResponse(request, "models.html", {"request": request})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)):
        """Upload a video file for processing."""
        if not file.filename:
            raise HTTPException(400, "No file provided")

        # Save file
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

    @app.post("/api/chat")
    async def chat_endpoint(
        question: str = Form(...),
        video_id: str = Form(...),
    ):
        """Ask a question about a video."""
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

    @app.get("/api/videos/{video_id}/frames/{frame_file:path}")
    async def get_frame(video_id: str, frame_file: str):
        """Serve a frame image."""
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
