"""
Full REST API layer for the video analysis platform.

Provides a comprehensive set of endpoints for video processing, querying,
searching, and management. Designed to be mounted as a FastAPI APIRouter
into the existing FastAPI application in ``ui/health.py``.

Async Job Queue (v0.43.0):
    ``POST /api/videos/process`` now enqueues video processing as a
    background job and returns immediately with a ``job_id``.  Poll
    ``GET /api/jobs/{job_id}`` for completion status and results.

Endpoints:
    POST   /api/videos/process          — Enqueue a video for processing
    GET    /api/jobs/{job_id}           — Poll job status
    GET    /api/jobs                    — List recent jobs
    POST   /api/videos/{video_id}/query — Ask a question about a video
    GET    /api/videos/search           — Cross-video semantic search
    GET    /api/videos/{video_id}/transcript  — Get video transcript
    GET    /api/videos/{video_id}/frames/{timestamp} — Get frame image
    DELETE /api/videos/{video_id}       — Delete a video from the index
    GET    /api/videos/{video_id}/chapters — Get video chapters
    GET    /api/videos/{video_id}       — Get detailed video info
    GET    /api/sse/chat                — SSE streaming chat endpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG, RetrievedChunk, VideoLibraryInfo
from video_analysis.chat import VideoChat
from video_analysis.models import VideoIndex, ChatMessage, ChatSource, format_timestamp
from video_analysis.chapters import ChapterGenerator, ChapteringResult
from video_analysis.job_queue import (
    Job,
    JobManager,
    JobStatus as JQStatus,
    get_default_manager,
)
from video_analysis.evaluation import EvalReportStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    """Request body for POST /api/videos/process (URL mode)."""

    url: str = Field(
        "",
        description="URL of a video to download and process (YouTube, direct link, etc.)",
    )
    file_path: str = Field(
        "", description="Local file path, if not uploading via multipart"
    )


class ProcessResponse(BaseModel):
    """Response from a successful video processing request."""

    video_id: str = Field(..., description="Unique identifier for the processed video")
    filename: str = Field("", description="Original filename")
    duration: float = Field(0.0, description="Video duration in seconds")
    num_scenes: int = Field(0, description="Number of detected scenes")
    num_chunks: int = Field(0, description="Number of indexed chunks")
    full_transcript: str = Field("", description="Full transcription text")
    status: str = Field("processed", description="Processing status")


class QueryRequest(BaseModel):
    """Request body for POST /api/videos/{video_id}/query."""

    query: str = Field(
        ..., min_length=1, description="Natural language question about the video"
    )
    stream: bool = Field(False, description="If true, stream tokens via SSE")


class QueryResponse(BaseModel):
    """Response from a query endpoint (non-streaming)."""

    answer: str = Field(..., description="The LLM-generated answer")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Source citations"
    )


class SearchResult(BaseModel):
    """A single search result from cross-video semantic search."""

    chunk_id: str = Field("", description="Unique chunk identifier")
    video_id: str = Field("", description="Video this chunk belongs to")
    text: str = Field("", description="Chunk text content")
    timestamp: float = Field(0.0, description="Timestamp in seconds")
    scene_id: int = Field(-1, description="Scene index")
    score: float = Field(0.0, description="Relevance score")
    frame_path: Optional[str] = Field(
        None, description="Path to associated frame image"
    )
    chunk_type: str = Field("scene", description="Type of chunk (scene, frame, etc.)")


class SearchResponse(BaseModel):
    """Response from GET /api/videos/search."""

    query: str = Field(..., description="The original search query")
    total_results: int = Field(0, description="Number of results returned")
    results: List[SearchResult] = Field(
        default_factory=list, description="Search result chunks"
    )


class TranscriptSegmentSchema(BaseModel):
    """A single transcript segment."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field("", description="Transcribed text")
    speaker: Optional[str] = Field(None, description="Speaker label if diarized")


class TranscriptResponse(BaseModel):
    """Response from GET /api/videos/{video_id}/transcript."""

    video_id: str = Field(..., description="Video identifier")
    segments: List[TranscriptSegmentSchema] = Field(
        default_factory=list, description="Transcript segments"
    )
    full_transcript: str = Field("", description="Full joined transcript text")


class ChapterSchema(BaseModel):
    """A single chapter within a video."""

    title: str = Field(..., description="Chapter title")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    index: int = Field(..., description="Chapter number (0-based)")
    summary: str = Field("", description="One-line summary")
    transcript_preview: str = Field("", description="First ~200 chars of transcript")
    word_count: int = Field(0, description="Word count")


class ChaptersResponse(BaseModel):
    """Response from GET /api/videos/{video_id}/chapters."""

    video_id: str = Field(..., description="Video identifier")
    chapters: List[ChapterSchema] = Field(
        default_factory=list, description="Generated chapters"
    )
    method: str = Field("", description="Chaptering method used")


class FrameInfoSchema(BaseModel):
    """A single key frame's metadata."""

    timestamp: float = Field(..., description="Frame timestamp in seconds")
    filepath: str = Field("", description="Path to frame image file")
    description: Optional[str] = Field(None, description="Frame description")
    objects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected objects"
    )
    ocr_text: Optional[str] = Field(None, description="OCR text found in frame")
    action: Optional[str] = Field(None, description="Detected action label")
    action_confidence: Optional[float] = Field(
        None, description="Action detection confidence"
    )


class SceneInfoSchema(BaseModel):
    """A single scene's metadata."""

    scene_id: int = Field(..., description="Scene index")
    start_time: float = Field(..., description="Scene start time in seconds")
    end_time: float = Field(..., description="Scene end time in seconds")
    transcript: Optional[str] = Field(None, description="Scene transcript text")
    summary: Optional[str] = Field(None, description="Scene summary")
    key_frames: List[FrameInfoSchema] = Field(
        default_factory=list, description="Key frames in this scene"
    )


class VideoDetailResponse(BaseModel):
    """Detailed video information from GET /api/videos/{video_id}."""

    video_id: str = Field(..., description="Video identifier")
    filename: str = Field("", description="Original filename")
    duration: float = Field(0.0, description="Duration in seconds")
    filepath: str = Field("", description="Path to original video file")
    num_scenes: int = Field(0, description="Number of scenes")
    num_chunks: int = Field(0, description="Number of indexed chunks")
    has_sprite: bool = Field(False, description="Whether a sprite sheet exists")
    scenes: List[SceneInfoSchema] = Field(
        default_factory=list, description="Scene details"
    )
    transcript_summary: str = Field("", description="Full transcript text")
    sprite_sheet: Optional[str] = Field(None, description="Path to sprite sheet image")


class DeleteResponse(BaseModel):
    """Response from DELETE /api/videos/{video_id}."""

    video_id: str = Field(..., description="Video identifier")
    deleted: bool = Field(True, description="Whether deletion succeeded")
    message: str = Field("", description="Status message")


class SSEErrorResponse(BaseModel):
    """Error detail for SSE endpoints."""

    error: str = Field(..., description="Error message")


class VideoListResponse(BaseModel):
    """Response from GET /api/videos."""

    count: int = Field(0, description="Number of videos in the library")
    videos: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of video summaries"
    )


# ---------------------------------------------------------------------------
# Job Queue schemas (v0.43.0)
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    """Response from GET /api/jobs/{job_id}."""

    job_id: str = Field(..., description="Unique job identifier")
    job_type: str = Field(..., description="Type of job (e.g. process_video)")
    status: str = Field(
        ..., description="Job status: pending/running/completed/failed/cancelled"
    )
    progress: str = Field("", description="Human-readable progress message")
    progress_pct: float = Field(
        0.0, description="Estimated completion percentage (0-100)"
    )
    created_at: float = Field(..., description="Unix timestamp when job was created")
    started_at: Optional[float] = Field(
        None, description="Unix timestamp when processing began"
    )
    completed_at: Optional[float] = Field(
        None, description="Unix timestamp when job finished"
    )
    result: Optional[Dict[str, Any]] = Field(
        None, description="Job result (populated on completion)"
    )
    error: Optional[str] = Field(
        None, description="Error message (populated on failure)"
    )


class JobListResponse(BaseModel):
    """Response from GET /api/jobs."""

    total: int = Field(0, description="Total number of jobs")
    jobs: List[JobResponse] = Field(
        default_factory=list, description="Recent job entries"
    )


class EnqueueResponse(BaseModel):
    """Response from POST /api/videos/process (async enqueue mode)."""

    job_id: str = Field(..., description="Job identifier for status polling")
    status: str = Field("pending", description="Initial job status")
    message: str = Field(
        "Video processing enqueued. Poll GET /api/jobs/{job_id} for status.",
        description="Status message",
    )


# ---------------------------------------------------------------------------
# Helper: retrieve full transcript from RAG
# ---------------------------------------------------------------------------


def _job_to_response(job: Job) -> JobResponse:
    """Convert a ``Job`` dataclass to a ``JobResponse`` Pydantic model."""
    return JobResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status.value,
        progress=job.progress,
        progress_pct=job.progress_pct,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
    )


def _get_video_transcript_data(rag: VideoRAG, video_id: str) -> Dict[str, Any]:
    """Fetch transcript segments and full transcript from the RAG index.

    Returns a dict with keys ``segments`` and ``full_transcript``, or
    raises ``HTTPException(404)`` if the video is not found.
    """
    try:
        # Query Chroma for all documents belonging to this video
        results = rag.collection.get(
            where={"video_id": video_id},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.error("Error fetching transcript for %s: %s", video_id, exc)
        raise HTTPException(status_code=503, detail="RAG engine unavailable")

    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    segments = []
    full_text_parts = []
    seen_texts: set = set()
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        text = results["documents"][i]

        # Only include transcript-type chunks for the transcript endpoint
        chunk_type = meta.get("chunk_type", "scene")
        if chunk_type not in ("scene", "fixed_60s", "sliding_30s"):
            continue

        start = meta.get("start_time", 0.0)
        end = meta.get("end_time", 0.0)
        speaker = meta.get("speaker")

        # Deduplicate near-identical text (e.g. scene-level + window duplicates)
        norm = text.strip()[:100]
        if norm not in seen_texts:
            seen_texts.add(norm)
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker,
                }
            )
            full_text_parts.append(text)

    return {
        "segments": segments,
        "full_transcript": "\n".join(full_text_parts),
    }


def _get_chapter_generator(config: Config) -> ChapterGenerator:
    """Create a ChapterGenerator instance (lazy)."""
    return ChapterGenerator(config=config)


def _get_scene_info_from_index(rag: VideoRAG, video_id: str) -> List[Dict[str, Any]]:
    """Fetch scene metadata from the RAG index for a given video.

    Returns a list of scene dicts ordered by scene_id.
    """
    try:
        results = rag.collection.get(
            where={"video_id": video_id},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.error("Error fetching scenes for %s: %s", video_id, exc)
        raise HTTPException(status_code=503, detail="RAG engine unavailable")

    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    scenes_map: Dict[int, Dict[str, Any]] = {}
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        scene_id = meta.get("scene_id", -1)
        if scene_id < 0:
            continue

        if scene_id not in scenes_map:
            scenes_map[scene_id] = {
                "scene_id": scene_id,
                "start_time": meta.get("start_time", 0.0),
                "end_time": meta.get("end_time", 0.0),
                "transcript": None,
                "summary": None,
                "key_frames": [],
            }

        text = results["documents"][i]
        chunk_type = meta.get("chunk_type", "scene")

        if chunk_type == "scene":
            scenes_map[scene_id]["transcript"] = text[:500] if text else None
            # Check if frame_path indicates a key frame
            frame_path = meta.get("frame_path")
            if frame_path:
                scenes_map[scene_id]["key_frames"].append(
                    {
                        "timestamp": meta.get("start_time", 0.0),
                        "filepath": frame_path,
                        "description": None,
                        "objects": [],
                        "ocr_text": None,
                        "action": None,
                        "action_confidence": None,
                    }
                )

    return sorted(scenes_map.values(), key=lambda s: s["scene_id"])


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------


async def _query_event_generator(
    query: str,
    video_id: str,
    rag: VideoRAG,
    chat: VideoChat,
    config: Config,
) -> AsyncGenerator[str, None]:
    """SSE event generator for streaming LLM tokens.

    Retrieves context from the RAG engine, then streams LLM tokens
    via the VideoChat backend.
    """
    q: asyncio.Queue = asyncio.Queue()

    def _run_query() -> None:
        """Run the query in a thread pool and push tokens to the queue."""
        try:
            # Retrieve context
            chunks = rag.retrieve(query, video_id=video_id)
            if video_id and chunks:
                chunks = rag.expand_temporal_context(chunks, video_id)
            context = (
                rag.build_context(chunks) if chunks else "No relevant context found."
            )

            # Build prompt
            prompt = chat._build_prompt(query, context)

            # Get LLM and stream tokens
            llm = chat._get_llm()

            # Try streaming API first
            tokens_streamed = False
            if hasattr(llm, "stream"):
                try:
                    for token in llm.stream(
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=config.llm_max_tokens,
                    ):
                        if token:
                            q.put_nowait(token)
                            tokens_streamed = True
                except Exception:
                    pass

            if not tokens_streamed:
                # Fallback: single-shot chat
                answer = llm.chat(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=config.llm_max_tokens,
                )
                if answer:
                    # Send answer in chunks of ~50 chars to simulate streaming
                    chunk_size = 50
                    for i in range(0, len(answer), chunk_size):
                        q.put_nowait(answer[i : i + chunk_size])

        except Exception as exc:
            logger.error("SSE query error: %s", exc)
            q.put_nowait(f"[Error: {exc}]")
        finally:
            q.put_nowait(None)  # sentinel

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_query)

    while True:
        token = await q.get()
        if token is None:
            break
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"


async def _chat_event_generator(
    query: str,
    config: Config,
) -> AsyncGenerator[str, None]:
    """SSE event generator for the raw chat endpoint (no video_id).

    Uses the LLM directly without RAG context.
    """
    q: asyncio.Queue = asyncio.Queue()

    def _run_chat() -> None:
        """Run a raw LLM chat in a thread pool."""
        try:
            from video_analysis.llm_provider import get_llm_provider, LLMProviderConfig

            cfg = LLMProviderConfig(
                provider=(
                    config.llm_provider if hasattr(config, "llm_provider") else "hermes"
                ),
                api_base=(
                    config.openai_api_base if hasattr(config, "openai_api_base") else ""
                ),
                api_key=(
                    config.openai_api_key if hasattr(config, "openai_api_key") else ""
                ),
                model=config.openai_model if hasattr(config, "openai_model") else "",
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature,
                hermes_model=config.llm_model,
                hermes_max_tokens=config.llm_max_tokens,
            )
            llm = get_llm_provider(cfg)

            tokens_streamed = False
            if hasattr(llm, "stream"):
                try:
                    for token in llm.stream(
                        prompt=query,
                        temperature=config.llm_temperature,
                        max_tokens=config.llm_max_tokens,
                    ):
                        if token:
                            q.put_nowait(token)
                            tokens_streamed = True
                except Exception:
                    pass

            if not tokens_streamed:
                answer = llm.chat(
                    prompt=query,
                    temperature=config.llm_temperature,
                    max_tokens=config.llm_max_tokens,
                )
                if answer:
                    chunk_size = 50
                    for i in range(0, len(answer), chunk_size):
                        q.put_nowait(answer[i : i + chunk_size])

        except Exception as exc:
            logger.error("SSE chat error: %s", exc)
            q.put_nowait(f"[Error: {exc}]")
        finally:
            q.put_nowait(None)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_chat)

    while True:
        token = await q.get()
        if token is None:
            break
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def create_api_router(config: Optional[Config] = None) -> APIRouter:
    """Create and configure the FastAPI APIRouter with all video API endpoints.

    Args:
        config: Application configuration. If None, a default Config is created.

    Returns:
        A configured ``APIRouter`` instance ready to be included in a FastAPI app.
    """
    cfg = config or Config()
    router = APIRouter()

    # Lazy-initialised components shared across endpoints.
    pipeline: Optional[VideoPipeline] = None
    rag: Optional[VideoRAG] = None
    chat: Optional[VideoChat] = None
    chapter_gen: Optional[ChapterGenerator] = None

    def _get_rag() -> VideoRAG:
        nonlocal rag
        # Check module-level RAG first (set by ui/health.py via set_rag_instance)
        import video_analysis.api as _api_mod

        if _api_mod._RAG is not None:  # noqa: SLF001
            rag = _api_mod._RAG
            _api_mod._RAG = None  # clear to avoid stale ref
        if rag is None:
            rag = VideoRAG(cfg)
        return rag

    def _get_chat() -> VideoChat:
        nonlocal chat
        if chat is None:
            chat = VideoChat(rag=_get_rag(), config=cfg)
        return chat

    def _get_pipeline() -> VideoPipeline:
        nonlocal pipeline
        if pipeline is None:
            pipeline = VideoPipeline(cfg)
        return pipeline

    def _get_chapter_gen() -> ChapterGenerator:
        nonlocal chapter_gen
        if chapter_gen is None:
            chapter_gen = ChapterGenerator(config=cfg)
        return chapter_gen

    # ------------------------------------------------------------------
    # Job Queue — internal handlers
    # ------------------------------------------------------------------

    def _process_video_handler(
        manager: JobManager,
        job_id: str,
        video_path: str,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full video pipeline (sync, runs in thread pool).

        Registered as the ``process_video`` handler in the JobManager.
        """
        # Report phase 1: pipeline
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Create fresh instances for background work
            local_pipeline = VideoPipeline(cfg)
            local_rag = VideoRAG(cfg)

            loop.call_soon_threadsafe(
                asyncio.ensure_future,
                manager.update_job(
                    job_id,
                    progress="Running video pipeline (transcription, scene detection, OCR, YOLO...)",
                    progress_pct=10.0,
                ),
            )

            video_index: VideoIndex = local_pipeline.process(video_path)

            loop.call_soon_threadsafe(
                asyncio.ensure_future,
                manager.update_job(
                    job_id,
                    progress="Indexing into RAG vector store...",
                    progress_pct=75.0,
                ),
            )

            local_rag.index_video(video_index)

            result = {
                "video_id": video_index.video_id,
                "filename": video_index.filename,
                "duration": video_index.duration,
                "num_scenes": len(video_index.scenes),
                "num_chunks": len(video_index.chunks),
                "full_transcript": video_index.full_transcript,
                "status": "processed",
            }
            return result
        finally:
            loop.close()

    # Register the handler on the default manager
    _job_manager = get_default_manager()

    # Only register if not already registered (re-entrant route factory calls)
    try:
        _job_manager._worker_registry.get("process_video")
    except KeyError:
        _job_manager.register_handler("process_video", _process_video_handler)

    # ------------------------------------------------------------------
    # POST /api/videos/process  (async — returns job_id immediately)
    # ------------------------------------------------------------------

    @router.post(
        "/api/videos/process",
        response_model=EnqueueResponse,
        summary="Enqueue a video for processing",
    )
    async def api_process_video(
        file: Optional[UploadFile] = File(None, description="Video file upload"),
        url: str = Form("", description="URL to download and process"),
        file_path: str = Form("", description="Local file path to process"),
    ):
        """Enqueue a video for background processing.

        Accepts a file upload, a URL (YouTube, direct link), or a local
        file path.  Returns immediately with a ``job_id`` that can be
        polled via ``GET /api/jobs/{job_id}`` for status and results.

        When the job completes (status=``completed``), the ``result``
        field contains the processed video's metadata including
        ``video_id``.
        """
        loop = asyncio.get_event_loop()

        # Determine source and resolve to a local video_path
        if file is not None and file.filename:
            video_dir = cfg.video_dir
            video_dir.mkdir(parents=True, exist_ok=True)
            dest = video_dir / file.filename
            content = await file.read()
            dest.write_bytes(content)
            video_path = str(dest)

        elif url:
            pipe = _get_pipeline()
            video_dir = cfg.video_dir
            video_dir.mkdir(parents=True, exist_ok=True)
            try:
                video_path = await loop.run_in_executor(
                    None, pipe.download_from_url, url, video_dir
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Failed to download URL: {exc}"
                )
            if not video_path:
                raise HTTPException(
                    status_code=400, detail="Failed to download video from URL"
                )

        elif file_path:
            video_path = file_path
            if not Path(video_path).exists():
                raise HTTPException(
                    status_code=404, detail=f"File not found: {video_path}"
                )
        else:
            raise HTTPException(
                status_code=400, detail="Provide a file upload, url, or file_path"
            )

        # Enqueue the job
        job = await _job_manager.enqueue(
            "process_video",
            video_path=video_path,
        )

        return EnqueueResponse(
            job_id=job.job_id,
            status=job.status.value,
            message="Video processing enqueued. Poll GET /api/jobs/{job_id} for status.",
        )

    # ------------------------------------------------------------------
    # GET /api/jobs/{job_id}  — poll job status
    # ------------------------------------------------------------------

    @router.get(
        "/api/jobs/{job_id}",
        response_model=JobResponse,
        summary="Poll a job's status and results",
    )
    async def api_get_job(job_id: str):
        """Retrieve the current status and results of a background job.

        Returns the job's lifecycle state (pending/running/completed/
        failed/cancelled), progress information, and — on success —
        the ``result`` dict containing the processed video metadata
        (``video_id``, ``filename``, ``duration``, ``num_scenes``,
        ``num_chunks``, ``full_transcript``).
        """
        job = await _job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return _job_to_response(job)

    # ------------------------------------------------------------------
    # GET /api/jobs  — list recent jobs
    # ------------------------------------------------------------------

    @router.get(
        "/api/jobs",
        response_model=JobListResponse,
        summary="List recent processing jobs",
    )
    async def api_list_jobs(
        limit: int = Query(50, description="Maximum number of jobs to return"),
        offset: int = Query(0, description="Number of jobs to skip"),
        status: Optional[str] = Query(
            None, description="Filter by status (pending/running/completed/failed)"
        ),
    ):
        """Return recent processing jobs, newest first.

        Supports pagination via *limit* and *offset*, and optional
        filtering by *status*.
        """
        status_filter = None
        if status:
            try:
                status_filter = JQStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status '{status}'. Valid: pending, running, completed, failed, cancelled",
                )

        jobs = await _job_manager.list_jobs(
            limit=limit, offset=offset, status_filter=status_filter
        )
        return JobListResponse(
            total=len(jobs),
            jobs=[_job_to_response(j) for j in jobs],
        )

    # ------------------------------------------------------------------
    # POST /api/videos/{video_id}/query
    # ------------------------------------------------------------------

    @router.post(
        "/api/videos/{video_id}/query",
        response_model=QueryResponse,
        summary="Ask a question about a video",
    )
    async def api_query_video(video_id: str, body: QueryRequest):
        """Ask a natural language question about a specific video.

        Set ``stream=true`` in the request body to receive tokens via
        Server-Sent Events instead of a single JSON response.
        """
        if body.stream:
            return StreamingResponse(
                _query_event_generator(
                    body.query, video_id, _get_rag(), _get_chat(), cfg
                ),
                media_type="text/event-stream",
            )

        chat_instance = _get_chat()
        loop = asyncio.get_event_loop()

        try:
            message: ChatMessage = await loop.run_in_executor(
                None, chat_instance.ask, body.query, video_id
            )
        except Exception as exc:
            logger.error("Query error for %s: %s", video_id, exc)
            raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

        return QueryResponse(
            answer=message.content,
            sources=[
                {
                    "text": s.text,
                    "timestamp": s.timestamp,
                    "frame_path": s.frame_path,
                    "scene_id": s.scene_id,
                    "relevance_score": s.relevance_score,
                }
                for s in message.sources
            ],
        )

    # ------------------------------------------------------------------
    # GET /api/videos/search
    # ------------------------------------------------------------------

    @router.get(
        "/api/videos/search",
        response_model=SearchResponse,
        summary="Cross-video semantic search",
    )
    async def api_search_videos(
        query: str = Query(..., description="Natural language search query"),
        top_k: int = Query(10, description="Number of results to return"),
    ):
        """Semantic search across all indexed videos.

        Returns relevant chunks sorted by relevance score.
        """
        loop = asyncio.get_event_loop()
        try:
            chunks: List[RetrievedChunk] = await loop.run_in_executor(
                None, _get_rag().search_all, query, top_k
            )
        except Exception as exc:
            logger.error("Search error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

        return SearchResponse(
            query=query,
            total_results=len(chunks),
            results=[
                SearchResult(
                    chunk_id=c.chunk_id,
                    video_id=c.video_id,
                    text=c.text[:500],
                    timestamp=c.timestamp,
                    scene_id=c.scene_id,
                    score=c.score,
                    frame_path=c.frame_path,
                    chunk_type=c.chunk_type,
                )
                for c in chunks
            ],
        )

    # ------------------------------------------------------------------
    # GET /api/videos/{video_id}/transcript
    # ------------------------------------------------------------------

    @router.get(
        "/api/videos/{video_id}/transcript",
        response_model=TranscriptResponse,
        summary="Get video transcript",
    )
    async def api_get_transcript(video_id: str):
        """Retrieve the full transcript for a video.

        Returns both individual timestamped segments and a joined full
        transcript string.
        """
        data = _get_video_transcript_data(_get_rag(), video_id)
        return TranscriptResponse(
            video_id=video_id,
            segments=[TranscriptSegmentSchema(**s) for s in data["segments"]],
            full_transcript=data["full_transcript"],
        )

    # ------------------------------------------------------------------
    # GET /api/videos/{video_id}/frames/{timestamp}
    # ------------------------------------------------------------------

    @router.get(
        "/api/videos/{video_id}/frames/{timestamp:path}",
        summary="Get a frame image at a specific timestamp",
    )
    async def api_get_frame(video_id: str, timestamp: str):
        """Retrieve a frame image at a specific timestamp for a video.

        The timestamp is matched to the closest key frame. Returns the
        image file (JPEG/PNG) directly.
        """
        rag_instance = _get_rag()
        try:
            results = rag_instance.collection.get(
                where={"video_id": video_id},
                include=["metadatas"],
                limit=500,
            )
        except Exception as exc:
            logger.error("Error fetching frame metadata for %s: %s", video_id, exc)
            raise HTTPException(status_code=503, detail="RAG engine unavailable")

        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

        # Parse the target timestamp
        try:
            target_ts = float(timestamp)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid timestamp: {timestamp}"
            )

        # Find the closest frame_path
        best_path: Optional[str] = None
        best_diff: float = float("inf")

        for meta in results["metadatas"]:
            frame_path = meta.get("frame_path")
            if not frame_path:
                continue
            start_time = meta.get("start_time", 0.0)
            diff = abs(start_time - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_path = frame_path

        if not best_path or not Path(best_path).exists():
            raise HTTPException(
                status_code=404, detail=f"No frame found near timestamp {timestamp}s"
            )

        try:
            image_bytes = Path(best_path).read_bytes()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to read frame image: {exc}"
            )

        # Determine media type from extension
        ext = Path(best_path).suffix.lower()
        media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        return Response(content=image_bytes, media_type=media_type)

    # ------------------------------------------------------------------
    # DELETE /api/videos/{video_id}
    # ------------------------------------------------------------------

    @router.delete(
        "/api/videos/{video_id}",
        response_model=DeleteResponse,
        summary="Delete a video from the index",
    )
    async def api_delete_video(video_id: str):
        """Remove a video and all its chunks from the RAG index.

        Does NOT delete the original video file or frame files from disk.
        """
        rag_instance = _get_rag()
        loop = asyncio.get_event_loop()

        # Check existence
        try:
            info = await loop.run_in_executor(
                None, rag_instance.get_library_info, video_id
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"RAG engine error: {exc}")

        if info is None:
            raise HTTPException(
                status_code=404, detail=f"Video '{video_id}' not found in index"
            )

        try:
            await loop.run_in_executor(None, rag_instance.delete_video, video_id)
        except Exception as exc:
            logger.error("Delete error for %s: %s", video_id, exc)
            raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")

        return DeleteResponse(
            video_id=video_id,
            deleted=True,
            message=f"Video '{video_id}' removed from index",
        )

    # ------------------------------------------------------------------
    # GET /api/videos/{video_id}/chapters
    # ------------------------------------------------------------------

    @router.get(
        "/api/videos/{video_id}/chapters",
        response_model=ChaptersResponse,
        summary="Get video chapters",
    )
    async def api_get_chapters(video_id: str):
        """Generate and retrieve video chapters from transcript content.

        Uses the ChapterGenerator to segment the transcript into
        topically-coherent chapters with LLM-generated titles.
        """
        # Fetch transcript data
        data = _get_video_transcript_data(_get_rag(), video_id)
        if not data["segments"]:
            raise HTTPException(
                status_code=404, detail=f"No transcript found for video '{video_id}'"
            )

        loop = asyncio.get_event_loop()
        try:
            result: ChapteringResult = await loop.run_in_executor(
                None,
                _get_chapter_gen().segment_transcript,
                data["segments"],
                video_id,
                None,  # scene_boundaries
                12,  # max_chapters
                2,  # min_chapters
                True,  # use_llm_titles
            )
        except Exception as exc:
            logger.error("Chapter generation error for %s: %s", video_id, exc)
            raise HTTPException(
                status_code=500, detail=f"Chapter generation failed: {exc}"
            )

        return ChaptersResponse(
            video_id=video_id,
            chapters=[
                ChapterSchema(
                    title=c.title,
                    start_time=c.start_time,
                    end_time=c.end_time,
                    index=c.index,
                    summary=c.summary,
                    transcript_preview=c.transcript_preview,
                    word_count=c.word_count,
                )
                for c in result.chapters
            ],
            method=result.method,
        )

    # ------------------------------------------------------------------
    # GET /api/videos — list all indexed videos
    # ------------------------------------------------------------------

    @router.get(
        "/api/videos",
        response_model=VideoListResponse,
        summary="List all indexed videos",
    )
    async def api_list_videos():
        """Return a list of all videos currently indexed in the library."""
        rag_instance = _get_rag()
        loop = asyncio.get_event_loop()

        try:
            video_ids = await loop.run_in_executor(None, rag_instance.list_videos)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"RAG engine error: {exc}")

        items = []
        for vid in video_ids:
            try:
                info = await loop.run_in_executor(
                    None, rag_instance.get_library_info, vid
                )
                if info is not None:
                    items.append(
                        {
                            "video_id": info.video_id,
                            "filename": info.filename,
                            "num_scenes": info.num_scenes,
                            "num_chunks": info.num_chunks,
                            "duration": info.duration,
                            "has_sprite": info.has_sprite,
                        }
                    )
            except Exception:
                items.append(
                    {"video_id": vid, "filename": "", "error": "metadata unavailable"}
                )

        return VideoListResponse(count=len(items), videos=items)

    # ------------------------------------------------------------------
    # GET /api/videos/{video_id}  — detailed video info
    # ------------------------------------------------------------------

    @router.get(
        "/api/videos/{video_id}",
        response_model=VideoDetailResponse,
        summary="Get detailed video information",
    )
    async def api_get_video_detail(video_id: str):
        """Retrieve detailed information about a video including scenes,
        objects, transcript summary, and library metadata.
        """
        rag_instance = _get_rag()
        loop = asyncio.get_event_loop()

        try:
            info: Optional[VideoLibraryInfo] = await loop.run_in_executor(
                None, rag_instance.get_library_info, video_id
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"RAG engine error: {exc}")

        if info is None:
            raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

        # Fetch scene data
        scenes = _get_scene_info_from_index(rag_instance, video_id)

        # Get transcript summary
        transcript_data = _get_video_transcript_data(rag_instance, video_id)

        return VideoDetailResponse(
            video_id=info.video_id,
            filename=info.filename,
            duration=info.duration,
            filepath="",  # Not stored in library info
            num_scenes=info.num_scenes,
            num_chunks=info.num_chunks,
            has_sprite=info.has_sprite,
            scenes=[SceneInfoSchema(**s) for s in scenes],
            transcript_summary=(
                transcript_data["full_transcript"][:2000]
                if transcript_data["full_transcript"]
                else ""
            ),
            sprite_sheet=None,
        )

    # ------------------------------------------------------------------
    # GET /api/sse/chat — raw SSE chat endpoint
    # ------------------------------------------------------------------

    @router.get(
        "/api/sse/chat",
        summary="SSE streaming chat endpoint for real-time LLM token streaming",
    )
    async def api_sse_chat(
        query: str = Query(
            ..., description="The chat message / question to send to the LLM"
        ),
    ):
        """Streaming chat endpoint using Server-Sent Events.

        Sends tokens one at a time as SSE ``data`` events. Each event
        is a JSON object with a ``token`` field. The stream ends with
        ``data: [DONE]``.
        """
        if not query.strip():
            raise HTTPException(
                status_code=400, detail="Query parameter 'query' is required"
            )

        return StreamingResponse(
            _chat_event_generator(query, cfg),
            media_type="text/event-stream",
        )

    # ------------------------------------------------------------------
    # GET /api/evaluations — list historical evaluation reports
    # ------------------------------------------------------------------

    store = EvalReportStore(data_dir=cfg.data_dir if hasattr(cfg, "data_dir") else None)

    @router.get(
        "/api/evaluations",
        summary="List historical evaluation reports",
        description="Returns summaries of saved evaluation reports, newest first.",
    )
    async def api_list_evaluations(
        limit: int = Query(20, ge=1, le=100, description="Max reports to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
    ):
        reports = store.list_reports(limit=limit, offset=offset)
        return {
            "total": len(reports),
            "offset": offset,
            "limit": limit,
            "reports": reports,
        }

    # ------------------------------------------------------------------
    # GET /api/evaluations/{run_id} — get full evaluation report
    # ------------------------------------------------------------------

    @router.get(
        "/api/evaluations/{run_id}",
        summary="Get full evaluation report by run ID",
        description="Returns the complete evaluation report with all task results.",
    )
    async def api_get_evaluation(run_id: str):
        report = store.load_report(run_id)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation report '{run_id}' not found",
            )
        return report.to_dict()

    # ------------------------------------------------------------------
    # GET /api/evaluations/compare — compare multiple reports
    # ------------------------------------------------------------------

    @router.get(
        "/api/evaluations/compare",
        summary="Compare multiple evaluation reports",
        description=(
            "Compare metrics across multiple evaluation runs. "
            "Pass 'run_ids' as a comma-separated list."
        ),
    )
    async def api_compare_evaluations(
        run_ids: str = Query(
            ...,
            description="Comma-separated list of report run IDs to compare",
        ),
    ):
        ids = [r.strip() for r in run_ids.split(",") if r.strip()]
        if len(ids) < 1:
            raise HTTPException(
                status_code=400,
                detail="At least one run_id required",
            )
        result = store.compare_reports(ids)
        if not result.get("report_ids"):
            raise HTTPException(
                status_code=404,
                detail=f"No reports found for IDs: {run_ids}",
            )
        return result

    return router


def set_rag_instance(rag: VideoRAG) -> None:
    """Set the module-level RAG instance (called by ui/health.py at startup).

    Args:
        rag: Initialised VideoRAG instance to share across API routes.
    """
    global _RAG
    _RAG = rag
