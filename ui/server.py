"""
Production FastAPI web server for Video Analysis Platform.

Replaces the Gradio UI with a FastAPI + Jinja2 + HTMX + Alpine.js stack.
Backend modules (video_analysis/*) are unchanged; all Gradio component
interactions are mapped to HTMX partials or WebSocket streams.

Start with::

    python -m video_analysis

    uvicorn ui.server:create_app --factory --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from video_analysis import __version__
from video_analysis.api import create_api_router
from video_analysis.config import Config
from video_analysis.job_queue import get_default_manager
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG
from video_analysis.chat import VideoChat

logger = logging.getLogger(__name__)

# ── Global singletons (set once at startup) ──────────────────────────────
_start_time: float = 0.0
_rag: Optional[VideoRAG] = None
_chat: Optional[VideoChat] = None
_pipeline: Optional[VideoPipeline] = None
_config: Optional[Config] = None

# Template directory (relative to this file)
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ═══════════════════════════════════════════════════════════════════════════
# App Factory
# ═══════════════════════════════════════════════════════════════════════════


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Build and return the production FastAPI application.

    Args:
        config: Application configuration.  Created from env vars if omitted.

    Returns:
        A fully configured FastAPI app ready for ``uvicorn.run()``.
    """
    global _start_time, _rag, _chat, _pipeline, _config

    config = config or Config()
    _config = config
    _start_time = time.time()

    # ── Hugging Face authentication ──────────────────────────────────
    if config.hf_token:
        try:
            import huggingface_hub

            huggingface_hub.login(token=config.hf_token)
            logger.info("HF_TOKEN authenticated successfully")
        except Exception as exc:
            logger.warning("HF_TOKEN login failed: %s", exc)

    # ── Build FastAPI app ────────────────────────────────────────────
    app = FastAPI(
        title="Video Analysis Platform",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
    )

    # Static files
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Data thumbnails (sprite sheets served by the pipeline)
    thumb_dir = config.thumbnails_dir
    thumb_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/data/thumbnails", StaticFiles(directory=str(thumb_dir)), name="thumbnails")

    # Video files
    video_dir = config.video_dir
    video_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/data/videos", StaticFiles(directory=str(video_dir)), name="videos")

    # ── Initialise backend singletons ────────────────────────────────
    # CRITICAL: ChromaDB's sqlite3 backend must be imported before any
    # other thread touches sqlite3, or Chroma will crash with
    # "SQLite objects created in a thread can only be used in that same thread".
    # We force a dummy RAG touch on the main thread to load the sqlite3
    # shared-cache extension now.
    try:
        import chromadb  # noqa: F401
    except ImportError:
        pass

    _rag = VideoRAG(config)
    _pipeline = VideoPipeline(config)
    _chat = VideoChat(_rag, config)

    # Shared RAG instance for the existing REST API router
    try:
        from video_analysis.api import set_rag_instance

        set_rag_instance(_rag)
    except Exception:
        pass

    logger.info(
        "Backend initialised: RAG=%s Pipeline=%s Chat=%s",
        type(_rag).__name__,
        type(_pipeline).__name__,
        type(_chat).__name__,
    )

    # ── Include the full REST API router ─────────────────────────────
    app.include_router(create_api_router(config))

    # ── Register all routes ──────────────────────────────────────────
    _setup_routes(app, config)

    # ── Store backend singletons in app.state for route access ───────
    app.state.pipeline = _pipeline
    app.state.rag = _rag
    app.state.chat = _chat
    app.state.config = config

    # ── Register error handlers ──────────────────────────────────────
    try:
        from video_analysis.error_handlers import register_error_handlers

        register_error_handlers(app)
        logger.info("Structured error handlers registered")
    except ImportError as exc:
        logger.debug("Error handlers not available: %s", exc)

    # ── Register rate limiting middleware ────────────────────────────
    _setup_rate_limiter(app, config)

    # ── Job worker lifespan ──────────────────────────────────────────
    @app.router.on_event("startup")
    async def _start_worker():
        manager = get_default_manager()
        if manager._worker_task is None:
            manager._worker_task = asyncio.create_task(manager._worker_loop())

    @app.router.on_event("shutdown")
    async def _stop_worker():
        mgr = get_default_manager()
        if mgr._worker_task is not None:
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except asyncio.CancelledError:
                pass
            mgr._worker_task = None

    return app


# ═══════════════════════════════════════════════════════════════════════════
# Route Registration
# ═══════════════════════════════════════════════════════════════════════════


def _setup_routes(app: FastAPI, config: Config) -> None:
    """Register all application routes on the FastAPI app."""

    # ── Health ───────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        global _start_time, _rag
        uptime = time.time() - _start_time if _start_time > 0 else 0.0
        gpu = False
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            pass
        return {
            "status": "ok",
            "version": __version__,
            "gpu_available": gpu,
            "uptime_seconds": round(uptime, 1),
        }

    @app.get("/api/status")
    async def api_status():
        """Lightweight status badge endpoint (HTMX polling)."""
        global _rag
        try:
            videos = _rag.list_videos() if _rag else []
            return HTMLResponse(
                f'<span class="badge ready">● Ready — {len(videos)} videos indexed</span>'
            )
        except Exception:
            return HTMLResponse('<span class="badge error">● Error</span>')

    # ── Main page ────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Render the full single-page application."""
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "version": __version__,
                "config": config,
            },
        )

    # ── WebSocket: Job Progress ──────────────────────────────────────
    @app.websocket("/ws/jobs/{job_id}")
    async def ws_job_progress(websocket: WebSocket, job_id: str):
        await websocket.accept()
        try:
            manager = get_default_manager()
            last_status = None
            while True:
                job = await manager.get_job(job_id)
                if job is None:
                    await websocket.send_json({"status": "unknown", "error": "Job not found"})
                    break

                status = job.status.value if hasattr(job.status, 'value') else str(job.status)
                # Only send updates when status changes
                if status != last_status:
                    last_status = status
                    payload = {"status": status, "job_id": job_id}
                    if hasattr(job, 'progress_pct'):
                        payload["progress_pct"] = job.progress_pct
                    if hasattr(job, 'progress'):
                        payload["progress"] = job.progress
                    if hasattr(job, 'result'):
                        payload["result"] = job.result
                    if hasattr(job, 'error'):
                        payload["error"] = job.error
                    await websocket.send_json(payload)

                if status in ("completed", "failed", "cancelled"):
                    break

                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("WebSocket error for job %s: %s", job_id, exc)
            try:
                await websocket.send_json({"status": "error", "error": str(exc)})
            except Exception:
                pass

    # ── Lazily register tab routes ───────────────────────────────────
    _register_tab_routes(app, config)


def _register_tab_routes(app: FastAPI, config: Config) -> None:
    """Register route handlers for each UI tab.

    Each tab can be loaded as a full page (GET /tab-name) or as an
    HTMX partial (GET /tab-name?partial=1) for lazy-loading.
    """
    import importlib

    _TAB_ROUTES = [
        ("ui.routes.analysis", "register_analysis_routes"),
        ("ui.routes.import_tab", "register_import_routes"),
        ("ui.routes.batch", "register_batch_routes"),
        ("ui.routes.search", "register_search_routes"),
        ("ui.routes.library", "register_library_routes"),
        ("ui.routes.camera", "register_camera_routes"),
        ("ui.routes.monitor", "register_monitor_routes"),
        ("ui.routes.comparison", "register_comparison_routes"),
        ("ui.routes.knowledge_graph", "register_kg_routes"),
        ("ui.routes.event_timeline", "register_event_routes"),
    ]

    for module_name, func_name in _TAB_ROUTES:
        try:
            module = importlib.import_module(module_name)
            register_fn = getattr(module, func_name, None)
            if register_fn is not None:
                register_fn(app, config, templates)
                logger.debug("Registered routes from %s", module_name)
            else:
                logger.warning("Module %s has no %s function", module_name, func_name)
        except ImportError as exc:
            logger.info("Tab route module %s not available yet: %s", module_name, exc)
        except Exception as exc:
            logger.warning("Failed to register routes from %s: %s", module_name, exc)


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════


def _setup_rate_limiter(app: FastAPI, config: Config) -> None:
    """Install rate-limiting middleware, exempting static assets."""
    try:
        from video_analysis.rate_limiter import TokenBucketLimiter

        _limiter = TokenBucketLimiter(
            capacity=config.rate_limit_capacity,
            rate=config.rate_limit_rate,
        )

        _UNRATED_PREFIXES = (
            "/assets/", "/static/", "/gradio_api/",
            "/theme.css", "/favicon.ico", "/manifest.json",
            "/health", "/docs", "/openapi.json",
            "/ws/",  # WebSocket connections
            "/data/",  # Static data mounts
        )

        if config.rate_limit_enabled:

            @app.middleware("http")
            async def rate_limit_middleware(request: Request, call_next):
                path = request.url.path
                if path == "/health" or path.startswith(_UNRATED_PREFIXES):
                    return await call_next(request)

                client_ip = request.client.host if request.client else "unknown"
                if not await _limiter.consume(client_ip):
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded. Try again later.",
                            "error_code": "RATE_LIMIT_EXCEEDED",
                            "status_code": 429,
                        },
                        headers={"Retry-After": "60"},
                    )
                return await call_next(request)

        logger.info(
            "Rate limiting %s (capacity=%d, rate=%.2f/s)",
            "enabled" if config.rate_limit_enabled else "disabled",
            config.rate_limit_capacity,
            config.rate_limit_rate,
        )
    except ImportError:
        logger.debug("Rate limiter not available")
