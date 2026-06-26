"""
FastAPI health and API endpoints for the video analysis platform.

Provides:
- GET /health          — container health check, GPU status, model states
- GET /api/library     — list of indexed videos (delegates to VideoRAG.list_videos())
- GET /api/video/{id}  — single video info (delegates to VideoRAG.get_library_info())
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from video_analysis import __version__
from video_analysis.api import create_api_router
from video_analysis.config import Config
from video_analysis.rag import VideoRAG

logger = logging.getLogger(__name__)

# ── Startup bookkeeping (set once when create_health_app() is called) ──
_start_time: float = 0.0
_rag: Optional[VideoRAG] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    models_loaded: dict
    uptime_seconds: float


class LibraryItem(BaseModel):
    video_id: str
    filename: str = ""
    num_scenes: int = 0
    num_chunks: int = 0
    duration: float = 0.0
    has_sprite: bool = False


class LibraryResponse(BaseModel):
    count: int
    videos: list[LibraryItem]


class VideoInfoResponse(BaseModel):
    video_id: str
    filename: str = ""
    num_scenes: int = 0
    num_chunks: int = 0
    duration: float = 0.0
    has_sprite: bool = False


def _check_gpu() -> bool:
    """Return True if a CUDA-capable GPU is visible to torch."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _check_models() -> dict:
    """Lightweight probe of model loading state.  Returns a dict keyed by model
    name with status 'loaded', 'not_loaded', or 'error'."""
    models: dict[str, str] = {}

    # faster-whisper (lazy import so we don't force load at startup)
    try:
        import faster_whisper  # noqa: F401

        models["faster-whisper"] = "loaded"
    except ImportError:
        models["faster-whisper"] = "not_loaded"
    except Exception:
        models["faster-whisper"] = "error"

    # torch / CUDA
    try:
        import torch

        models["torch"] = (
            f"v{torch.__version__}" if hasattr(torch, "__version__") else "loaded"
        )
    except ImportError:
        models["torch"] = "not_loaded"
    except Exception:
        models["torch"] = "error"

    # ChromaDB
    try:
        import chromadb

        models["chromadb"] = "loaded"
    except ImportError:
        models["chromadb"] = "not_loaded"
    except Exception:
        models["chromadb"] = "error"

    # open-clip
    try:
        import open_clip

        models["open_clip"] = "loaded"
    except ImportError:
        models["open_clip"] = "not_loaded"
    except Exception:
        models["open_clip"] = "error"

    # sentence-transformers
    try:
        import sentence_transformers  # noqa: F401

        models["sentence_transformers"] = "loaded"
    except ImportError:
        models["sentence_transformers"] = "not_loaded"
    except Exception:
        models["sentence_transformers"] = "error"

    return models


def _setup_routes(app: FastAPI, config: Optional[Config] = None) -> None:
    """Register all routes on the FastAPI app."""

    # Include the full REST API router from video_analysis.api
    app.include_router(create_api_router(config))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        global _start_time, _rag
        uptime = time.time() - _start_time if _start_time > 0 else 0.0
        gpu = _check_gpu()
        models = _check_models()

        # If RAG is initialised, try a quick ping
        rag_ok = False
        if _rag is not None:
            try:
                col: object = _rag.collection
                if hasattr(col, "count"):
                    col.count()  # type: ignore[no-untyped-call]
                rag_ok = True
            except Exception:
                pass

        return HealthResponse(
            status="ok",
            version=__version__,
            gpu_available=gpu,
            models_loaded=models,
            uptime_seconds=round(uptime, 1),
        )

    @app.get("/api/library", response_model=LibraryResponse)
    async def api_library():
        global _rag
        if _rag is None:
            raise HTTPException(status_code=503, detail="RAG engine not initialised")
        try:
            video_ids = _rag.list_videos()
            items: list[LibraryItem] = []
            for vid in video_ids:
                info = _rag.get_library_info(vid)
                if info is not None:
                    items.append(
                        LibraryItem(
                            video_id=info.video_id,
                            filename=info.filename,
                            num_scenes=info.num_scenes,
                            num_chunks=info.num_chunks,
                            duration=info.duration,
                            has_sprite=info.has_sprite,
                        )
                    )
            return LibraryResponse(count=len(items), videos=items)
        except Exception as e:
            logger.error(f"Library listing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/video/{video_id}", response_model=VideoInfoResponse)
    async def api_video(video_id: str):
        global _rag
        if _rag is None:
            raise HTTPException(status_code=503, detail="RAG engine not initialised")
        info = _rag.get_library_info(video_id)
        if info is None:
            raise HTTPException(
                status_code=404, detail=f"Video '{video_id}' not found in library"
            )
        return VideoInfoResponse(
            video_id=info.video_id,
            filename=info.filename,
            num_scenes=info.num_scenes,
            num_chunks=info.num_chunks,
            duration=info.duration,
            has_sprite=info.has_sprite,
        )

    # Prometheus /metrics endpoint (v0.28.0)
    metrics_enabled = config is not None and config.prometheus_enabled

    if metrics_enabled:

        @app.get("/metrics")
        async def metrics():
            from video_analysis.metrics import metrics_endpoint

            content = metrics_endpoint()
            return Response(
                content=content,
                media_type="text/plain; charset=utf-8",
            )

    # Federated Search REST endpoint (v0.33.0)
    federation_enabled = config is not None and config.federation_enabled

    if federation_enabled:

        @app.get("/api/federated/search")
        async def federated_search(
            query: str,
            top_k: int = 10,
            include_local: bool = True,
        ):
            """Search across this instance's index (for consumption by federation peers).

            Returns a JSON payload with ``chunks`` as a list of chunk dicts.
            """
            global _rag
            if _rag is None:
                raise HTTPException(
                    status_code=503, detail="RAG engine not initialised"
                )
            try:
                chunks = _rag.search_all(query=query, top_k=max(top_k, 50))
                return {
                    "query": query,
                    "top_k": top_k,
                    "total_chunks": len(chunks),
                    "chunks": [
                        {
                            "chunk_id": c.chunk_id,
                            "video_id": c.video_id,
                            "text": c.text,
                            "timestamp": c.timestamp,
                            "scene_id": c.scene_id,
                            "score": c.score,
                            "frame_path": c.frame_path,
                            "chunk_type": c.chunk_type,
                        }
                        for c in chunks
                    ],
                }
            except Exception as e:
                logger.error("Federated search endpoint error: %s", e)
                raise HTTPException(status_code=500, detail=str(e))


def _setup_auth_middleware(app: FastAPI, config: Config) -> None:
    """Add HTTP Basic Auth middleware to the FastAPI app if configured.

    Reads credentials from config.ui_auth_username and config.ui_auth_password
    (which default to GRADIO_USER and GRADIO_PASSWORD env vars). The /health
    endpoint is excluded from auth.
    """
    if not config.ui_auth_enabled or not config.ui_auth_password:
        logger.info("UI authentication disabled (no GRADIO_PASSWORD set)")
        return

    expected_user = config.ui_auth_username
    expected_pass = config.ui_auth_password
    logger.info(f"UI authentication enabled for user '{expected_user}'")

    @app.middleware("http")
    async def auth_middleware(request, call_next):
        # Skip auth for health endpoint
        if request.url.path == "/health":
            return await call_next(request)
        # Check auth header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Basic "):
            return Response(
                status_code=401,
                content="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )
        import base64

        try:
            decoded = base64.b64decode(auth_header.removeprefix("Basic ")).decode()
            username, _, password = decoded.partition(":")
            if username == expected_user and password == expected_pass:
                return await call_next(request)
        except Exception:
            pass
        return Response(
            status_code=401,
            content="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


def create_health_app(config: Config) -> FastAPI:
    """Build and configure the FastAPI application.

    The returned app is intended to be used with ``gr.mount_gradio_app()``
    so that the Gradio UI lives at ``/`` while ``/health`` and ``/api/*``
    remain accessible on FastAPI directly.
    """
    global _start_time, _rag

    _start_time = time.time()
    _rag = VideoRAG(config)

    app = FastAPI(
        title="Video Analysis Platform API",
        version=__version__,
        description="REST API for the self-hosted video analysis platform",
    )

    # Pass the module-level RAG instance to the API router
    try:
        from video_analysis.api import set_rag_instance

        set_rag_instance(_rag)
    except ImportError as exc:
        logger.warning("REST API module not available: %s", exc)

    # Routes are registered by _setup_routes (create_api_router included there)
    _setup_routes(app, config)

    # Apply auth middleware if configured
    _setup_auth_middleware(app, config)

    logger.info("Health API initialised — RAG engine ready")
    return app
