# MLLM REST API Endpoints — Design Document

## Overview

The existing `api.py` provides endpoints for video processing, RAG querying, search,
transcripts, chapters, KG, and health. However, it has **zero endpoints** that leverage
`VideoMLLM` (the video-native multimodal LLM backend). The `VideoMLLM` class and its
backends (`InternVideo3`, `Qwen3-VL`, `SmolVLM2`, `VideoChat-Flash`) support
`describe_scene()`, `summarize_video()`, and `answer()` — but these are only callable
from Python code.

This document designs **7 new REST endpoints** to expose the MLLM capabilities via the
API, following the existing `api.py` patterns (lazy init, async thread-pool exec, SSE
streaming, Pydantic schemas).

---

## 1. Existing Patterns (from `api.py`)

- **Route factory**: `create_api_router(config)` returns an `APIRouter`.
- **Lazy init**: Helpers like `_get_rag()`, `_get_chat()`, `_get_kg()`, `_get_health()`.
- **Async thread-pool**: `await loop.run_in_executor(None, sync_fn, ...)` for CPU/GPU work.
- **SSE streaming**: `StreamingResponse(_event_generator(...), media_type="text/event-stream")`.
- **Pydantic schemas**: Separate `*Request` and `*Response` models at module top.
- **Error handling**: `HTTPException(status_code=... , detail=...)`.
- **Path prefix**: All routes under `/api/...`.

---

## 2. New MLLM Singleton

Add a module-level lazy singleton for `VideoMLLM`, similar to `_kg_instance` and
`_health_instance`:

```python
_mllm_instance: Optional[VideoMLLM] = None

def _get_mllm(config: Config) -> VideoMLLM:
    global _mllm_instance
    if _mllm_instance is None:
        from video_analysis.video_mllm import VideoMLLM
        _mllm_instance = VideoMLLM(
            model_name=config.mllm_model_name if hasattr(config, "mllm_model_name") else "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448",
            backend=config.mllm_backend if hasattr(config, "mllm_backend") else "auto",
        )
    return _mllm_instance
```

Add a `set_mllm_instance()` setter (analogous to `set_rag_instance()`) so the UI layer
can inject a pre-configured MLLM at startup.

Additionally, add a separate lazy singleton for backend management:

```python
_mllm_backend_instances: Dict[str, Any] = {}

def _get_mllm_backend(backend_name: str, config: Config) -> Any:
    """Get or create a specific MLLM backend by name."""
    global _mllm_backend_instances
    if backend_name not in _mllm_backend_instances:
        _mllm_backend_instances[backend_name] = VideoMLLM(
            model_name=config.mllm_model_name if hasattr(config, "mllm_model_name") else "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448",
            backend=backend_name,
        )
    return _mllm_backend_instances[backend_name]
```

---

## 3. Pydantic Schemas

Add to the top section of `api.py` (after existing schemas, before `_job_to_response`):

### 3.1. Describe Frame(s)

```python
class MLLMDescribeRequest(BaseModel):
    """Request body for POST /api/videos/{video_id}/describe."""
    timestamps: List[float] = Field(
        ..., min_length=1, max_length=16,
        description="Frame timestamps in seconds (up to 16)",
    )
    prompt: Optional[str] = Field(
        None,
        description="Optional custom description prompt. Defaults to a rich scene description prompt.",
    )
    max_tokens: int = Field(256, ge=32, le=2048, description="Max tokens in response")


class MLLMDescribeResponse(BaseModel):
    """Response from scene description endpoint."""
    video_id: str = Field(..., description="Video identifier")
    descriptions: List[Dict[str, Any]] = Field(
        ...,
        description="List of per-timestamp descriptions",
        examples=[[{"timestamp": 10.5, "description": "A sunlit kitchen..."}]],
    )
    backend: str = Field("", description="Which MLLM backend produced the descriptions")


class MLLMDescribeFramesRequest(BaseModel):
    """Request body for POST /api/mllm/describe-frames (no video ID needed)."""
    frame_paths: List[str] = Field(
        ..., min_length=1, max_length=16,
        description="Paths to frame image files on the server",
    )
    prompt: Optional[str] = Field(
        None,
        description="Optional custom description prompt",
    )
    max_tokens: int = Field(256, ge=32, le=2048, description="Max tokens in response")


class MLLMDescribeFramesResponse(BaseModel):
    """Response from raw frame-by-frame description."""
    descriptions: List[Dict[str, Any]] = Field(
        ...,
        description="List of per-frame descriptions",
    )
    backend: str = Field("", description="Which MLLM backend produced the descriptions")
```

### 3.2. Summarize Video

```python
class MLLMSummarizeRequest(BaseModel):
    """Request body for POST /api/videos/{video_id}/summarize."""
    num_frames: int = Field(32, ge=8, le=64, description="Number of frames to sample")
    prompt: Optional[str] = Field(
        None,
        description="Optional custom summary prompt",
    )
    max_tokens: int = Field(512, ge=64, le=4096, description="Max tokens in summary")
    stream: bool = Field(False, description="If true, stream tokens via SSE")


class MLLMSummarizeResponse(BaseModel):
    """Response from video summarization."""
    video_id: str = Field(..., description="Video identifier")
    summary: str = Field(..., description="Generated video summary")
    num_frames: int = Field(0, description="Number of frames sampled")
    backend: str = Field("", description="Which MLLM backend produced the summary")
```

### 3.3. MLLM Query (bypassing RAG)

```python
class MLLMQueryRequest(BaseModel):
    """Request body for POST /api/videos/{video_id}/mllm-query."""
    query: str = Field(..., min_length=1, description="Natural language question about the video")
    timestamps: List[float] = Field(
        default_factory=list,
        description="Specific frame timestamps to use as visual context. If empty, auto-samples 8 evenly-spaced frames.",
    )
    num_frames: int = Field(8, ge=1, le=32, description="Number of frames to sample (if timestamps not provided)")
    max_tokens: int = Field(512, ge=32, le=4096, description="Max tokens in answer")
    stream: bool = Field(False, description="If true, stream tokens via SSE")


class MLLMQueryResponse(BaseModel):
    """Response from MLLM query (bypasses RAG)."""
    video_id: str = Field(..., description="Video identifier")
    query: str = Field(..., description="The original question")
    answer: str = Field(..., description="The MLLM-generated answer")
    frames_used: List[float] = Field(
        default_factory=list, description="Timestamps of frames used"
    )
    backend: str = Field("", description="Which MLLM backend produced the answer")
```

### 3.4. Backend Management

```python
class MLLMBackendInfo(BaseModel):
    """Information about a single MLLM backend."""
    name: str = Field(..., description="Backend identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field("", description="Short description")
    available: bool = Field(False, description="Whether the backend is currently available (dependencies met)")
    loaded: bool = Field(False, description="Whether the backend is currently loaded into GPU memory")
    model_name: str = Field("", description="Current model identifier for this backend")
    supported_modes: List[str] = Field(
        default_factory=lambda: ["describe_scene", "summarize_video", "answer"],
        description="Supported MLLM operations",
    )


class MLLMBackendListResponse(BaseModel):
    """Response from GET /api/mllm/backends."""
    backends: List[MLLMBackendInfo] = Field(..., description="List of available/configured backends")
    active_backend: str = Field("", description="Currently active backend name")
    default_backend: str = Field("auto", description="Default backend (set by config)")


class MLLMLoadBackendRequest(BaseModel):
    """Request body for POST /api/mllm/backends/{backend}/load."""
    model_name: Optional[str] = Field(
        None,
        description="Optional model identifier override. Uses backend default if not specified.",
    )
    use_fp8: bool = Field(True, description="Enable FP8 quantization (reduces VRAM)")
    thinking_mode: bool = Field(False, description="Enable thinking/reasoning mode (InternVideo3 MCR)")


class MLLMLoadBackendResponse(BaseModel):
    """Response from loading an MLLM backend."""
    backend: str = Field(..., description="Backend name")
    loaded: bool = Field(True, description="Whether loading succeeded")
    model_name: str = Field("", description="Model identifier loaded")
    mode: Optional[str] = Field(None, description="Deployment mode (vllm_server / vllm_offline / transformers)")
    error: Optional[str] = Field(None, description="Error message if loading failed")


class MLLMUnloadBackendResponse(BaseModel):
    """Response from unloading an MLLM backend."""
    backend: str = Field(..., description="Backend name")
    unloaded: bool = Field(True, description="Whether unloading succeeded")
    message: str = Field("", description="Status message")


class MLLMErrorResponse(BaseModel):
    """Error response for MLLM endpoints."""
    error: str = Field(..., description="Error message")
    backend: Optional[str] = Field(None, description="Backend that produced the error")
```

---

## 4. Route Definitions

All routes go inside the `create_api_router()` factory function (or a separate
`create_mllm_router()` factory that returns an `APIRouter` to be included via
`router.include_router(mllm_router)`).

### 4.1. POST /api/videos/{video_id}/describe

Describe specific frame timestamps for a known video.

**Flow:**
1. Look up `video_id` in RAG to find the video's working directory / frame files.
2. For each requested timestamp, find the nearest existing frame file (same pattern as existing `GET /api/videos/{video_id}/frames/{timestamp}`).
3. Pass the frame file paths + optional custom prompt to `mllm.describe_scene()`.
4. Return per-frame descriptions.

```python
@router.post(
    "/api/videos/{video_id}/describe",
    response_model=MLLMDescribeResponse,
    summary="Describe specific frames in a video using MLLM",
)
async def api_mllm_describe(video_id: str, body: MLLMDescribeRequest):
    """Use the Video MLLM to produce rich, natural-language descriptions
    for specific frame timestamps in a video.

    Bypasses CLIP/caption models — uses the full MLLM backend (InternVideo3,
    Qwen3-VL, SmolVLM2, or VideoChat-Flash) for detailed scene understanding.
    """
    mllm = _get_mllm(cfg)
    if not mllm.available:
        raise HTTPException(status_code=503, detail="No MLLM backend is available/loaded")

    # Resolve timestamps to actual frame files on disk
    rag_instance = _get_rag()
    results = rag_instance.collection.get(
        where={"video_id": video_id},
        include=["metadatas"],
        limit=500,
    )
    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    # For each requested timestamp, find the closest frame_path
    descriptions = []
    for ts in body.timestamps:
        best_path = None
        best_diff = float("inf")
        for meta in results["metadatas"]:
            frame_path = meta.get("frame_path")
            if not frame_path:
                continue
            start_time = meta.get("start_time", 0.0)
            diff = abs(start_time - ts)
            if diff < best_diff:
                best_diff = diff
                best_path = frame_path

        if best_path and Path(best_path).exists():
            desc = await loop.run_in_executor(
                None,
                mllm.describe_scene,
                [best_path],
                body.prompt,
            )
            descriptions.append({
                "timestamp": ts,
                "closest_frame_time": meta.get("start_time", ts),
                "description": desc or "Failed to generate description",
                "frame_path": best_path,
            })
        else:
            descriptions.append({
                "timestamp": ts,
                "description": f"No frame found near timestamp {ts}s",
            })

    return MLLMDescribeResponse(
        video_id=video_id,
        descriptions=descriptions,
        backend=mllm._resolved_backend or "unknown",
    )
```

### 4.2. POST /api/videos/{video_id}/summarize

Full-video summary using MLLM.

**Flow:**
1. Resolve `video_id` to a video file path (from RAG library info).
2. Call `mllm.summarize_video(video_path, num_frames=body.num_frames, prompt=body.prompt)`.
3. If `stream=true`, use SSE to stream tokens.

```python
@router.post(
    "/api/videos/{video_id}/summarize",
    response_model=MLLMSummarizeResponse,
    summary="Generate a full video summary using MLLM",
)
async def api_mllm_summarize(video_id: str, body: MLLMSummarizeRequest):
    """Use the Video MLLM to generate a comprehensive summary of an entire video.

    Frames are automatically sampled evenly across the video duration.
    Set ``stream=true`` to receive the summary tokens via Server-Sent Events.
    """
    mllm = _get_mllm(cfg)
    if not mllm.available:
        raise HTTPException(status_code=503, detail="No MLLM backend is available/loaded")

    # Resolve video file path
    rag_instance = _get_rag()
    info = rag_instance.get_library_info(video_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    # Find video file on disk (check video_dir, data_dir, etc.)
    video_path = _resolve_video_path(cfg, info)
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file not found on disk for '{video_id}'")

    if body.stream:
        return StreamingResponse(
            _mllm_summarize_event_generator(mllm, video_path, body, cfg),
            media_type="text/event-stream",
        )

    summary = await loop.run_in_executor(
        None,
        mllm.summarize_video,
        video_path,
        body.num_frames,
        body.prompt,
    )

    return MLLMSummarizeResponse(
        video_id=video_id,
        summary=summary or "Failed to generate summary",
        num_frames=body.num_frames,
        backend=mllm._resolved_backend or "unknown",
    )
```

**SSE generator** (analogous to `_query_event_generator`):

```python
async def _mllm_summarize_event_generator(
    mllm, video_path, body, config
) -> AsyncGenerator[str, None]:
    q = asyncio.Queue()

    def _run():
        try:
            summary = mllm.summarize_video(video_path, body.num_frames, body.prompt)
            if summary:
                # Yield in chunks to simulate streaming (MLLM summarize is typically non-streaming)
                chunk_size = 50
                for i in range(0, len(summary), chunk_size):
                    q.put_nowait(summary[i:i+chunk_size])
        except Exception as exc:
            logger.error("MLLM summarize SSE error: %s", exc)
            q.put_nowait(f"[Error: {exc}]")
        finally:
            q.put_nowait(None)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run)
    while True:
        token = await q.get()
        if token is None:
            break
        yield f"data: {json.dumps({'token': token, 'backend': mllm._resolved_backend})}\n\n"
    yield "data: [DONE]\n\n"
```

### 4.3. POST /api/videos/{video_id}/mllm-query

Multi-frame Q&A via MLLM, bypassing RAG entirely.

**Flow:**
1. Resolve frame paths from timestamps (or auto-sample if no timestamps given).
2. Call `mllm.answer(query, frames=frame_paths)`.
3. Return answer with frame timestamps used.

```python
@router.post(
    "/api/videos/{video_id}/mllm-query",
    response_model=MLLMQueryResponse,
    summary="Ask a question about a video using MLLM (bypasses RAG)",
)
async def api_mllm_query(video_id: str, body: MLLMQueryRequest):
    """Ask a question about a video using the Video MLLM directly,
    bypassing the text-based RAG pipeline entirely.

    The MLLM sees actual video frames and can answer visual questions
    that text-based RAG cannot handle (e.g., "What color is the car?",
    "What is the person wearing?").
    """
    mllm = _get_mllm(cfg)
    if not mllm.available:
        raise HTTPException(status_code=503, detail="No MLLM backend is available/loaded")

    rag_instance = _get_rag()
    results = rag_instance.collection.get(
        where={"video_id": video_id},
        include=["metadatas"],
        limit=500,
    )
    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    # Resolve frame paths
    frame_paths = []
    timestamps_used = []

    if body.timestamps:
        # Use specified timestamps
        for ts in body.timestamps:
            best_path, best_time = _find_closest_frame(results["metadatas"], ts)
            if best_path:
                frame_paths.append(best_path)
                timestamps_used.append(best_time)
    else:
        # Auto-sample evenly spaced frames
        all_metas = [m for m in results["metadatas"] if m.get("frame_path")]
        if all_metas:
            step = max(1, len(all_metas) // body.num_frames)
            for meta in all_metas[::step][:body.num_frames]:
                fp = meta.get("frame_path")
                if fp and Path(fp).exists():
                    frame_paths.append(fp)
                    timestamps_used.append(meta.get("start_time", 0.0))

    if not frame_paths:
        raise HTTPException(status_code=404, detail="No frame images found for this video")

    if body.stream:
        return StreamingResponse(
            _mllm_query_event_generator(mllm, body.query, frame_paths, body.max_tokens, cfg),
            media_type="text/event-stream",
        )

    answer = await loop.run_in_executor(
        None,
        mllm.answer,
        body.query,
        frame_paths,
        None,  # no video_path
    )

    return MLLMQueryResponse(
        video_id=video_id,
        query=body.query,
        answer=answer or "Failed to generate answer",
        frames_used=timestamps_used,
        backend=mllm._resolved_backend or "unknown",
    )
```

### 4.4. POST /api/mllm/describe-frames

Raw frame-by-frame description (no video ID required). Useful for ad-hoc frame
analysis without an indexed video.

```python
@router.post(
    "/api/mllm/describe-frames",
    response_model=MLLMDescribeFramesResponse,
    summary="Describe arbitrary frame images using MLLM (no video ID needed)",
)
async def api_mllm_describe_frames(body: MLLMDescribeFramesRequest):
    """Describe arbitrary frame image files on the server using the Video MLLM.

    Unlike ``POST /api/videos/{video_id}/describe``, this endpoint does not
    require an indexed video. The caller provides direct paths to frame image
    files on the server filesystem.

    Useful for ad-hoc analysis, previewing frames before indexing, or
    describing frames from external sources.
    """
    mllm = _get_mllm(cfg)
    if not mllm.available:
        raise HTTPException(status_code=503, detail="No MLLM backend is available/loaded")

    # Validate frame paths
    valid_paths = [fp for fp in body.frame_paths if Path(fp).exists()]
    if not valid_paths:
        raise HTTPException(status_code=400, detail="None of the provided frame paths exist on disk")

    descriptions = []
    backend_name = mllm._resolved_backend or "unknown"

    for fp in valid_paths:
        desc = await loop.run_in_executor(
            None,
            mllm.describe_scene,
            [fp],
            body.prompt,
        )
        descriptions.append({
            "frame_path": fp,
            "description": desc or "Failed to generate description",
        })

    return MLLMDescribeFramesResponse(
        descriptions=descriptions,
        backend=backend_name,
    )
```

### 4.5. GET /api/mllm/backends

List available/configured MLLM backends and their status.

```python
@router.get(
    "/api/mllm/backends",
    response_model=MLLMBackendListResponse,
    summary="List available MLLM backends",
)
async def api_mllm_list_backends():
    """List all configured MLLM backends and their current status.

    Returns information about each backend including whether its
    dependencies are met and whether it is currently loaded into GPU memory.
    """
    backends = _get_configured_backends()
    mllm = _get_mllm(cfg)

    backend_infos = []
    for name, info in backends.items():
        loaded = _is_backend_loaded(name, mllm)
        backend_infos.append(MLLMBackendInfo(
            name=name,
            display_name=info["display_name"],
            description=info["description"],
            available=_check_backend_availability(name),
            loaded=loaded,
            model_name=info.get("model_name", ""),
        ))

    active = mllm._resolved_backend or "none"
    return MLLMBackendListResponse(
        backends=backend_infos,
        active_backend=active,
        default_backend=cfg.mllm_backend if hasattr(cfg, "mllm_backend") else "auto",
    )


def _get_configured_backends() -> Dict[str, Dict[str, str]]:
    """Return the set of known MLLM backends with metadata."""
    return {
        "videochat_flash": {
            "display_name": "VideoChat-Flash",
            "description": "OpenGVLab hierarchical compression video MLLM (ICLR 2026)",
            "model_name": "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448",
        },
        "smolvlm2": {
            "display_name": "SmolVLM2",
            "description": "HuggingFaceTB compact vision-language models (2.2B, 500M, 256M)",
            "model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        },
        "qwen3_vl": {
            "display_name": "Qwen3-VL-30B-A3B",
            "description": "Qwen MoE VLM with 30B total / 3B active params, FP8, 128K context",
            "model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        },
        "internvideo3": {
            "display_name": "InternVideo3-8B",
            "description": "OpenGVLab SOTA open-weight video MLLM with MCR reasoning (Video-MME 73.8)",
            "model_name": "OpenGVLab/InternVideo3-8B-Instruct",
        },
    }


def _is_backend_loaded(backend_name: str, mllm: VideoMLLM) -> bool:
    """Check if a specific backend is currently loaded in the default MLLM instance."""
    if mllm._resolved_backend == backend_name and mllm._model is not None:
        return True
    if backend_name in _mllm_backend_instances:
        instance = _mllm_backend_instances[backend_name]
        return instance._model is not None or instance._llm is not None
    return False


def _check_backend_availability(backend_name: str) -> bool:
    """Check if a backend's dependencies are satisfied (without loading)."""
    try:
        import torch
    except ImportError:
        return False

    if backend_name == "internvideo3":
        # Check if vLLM server responds or vLLM/transformers is importable
        try:
            from video_analysis.backends.internvideo3 import InternVideo3Backend
            b = InternVideo3Backend()
            return b._check_vllm_server() or True  # optimistic for non-server modes
        except ImportError:
            return False
    elif backend_name == "qwen3_vl":
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
    elif backend_name in ("videochat_flash", "smolvlm2"):
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
    return False
```

### 4.6. POST /api/mllm/backends/{backend}/load

Load a specific MLLM backend into GPU memory.

```python
@router.post(
    "/api/mllm/backends/{backend}/load",
    response_model=MLLMLoadBackendResponse,
    summary="Load a specific MLLM backend into GPU memory",
)
async def api_mllm_load_backend(
    backend: str,
    body: MLLMLoadBackendRequest = MLLMLoadBackendRequest(),
):
    """Load a specific MLLM backend into GPU memory.

    Supported backends: ``videochat_flash``, ``smolvlm2``, ``qwen3_vl``,
    ``internvideo3``.

    Loading a new backend automatically unloads the previously active backend
    to free GPU memory (since most systems have limited VRAM).
    """
    supported = ["videochat_flash", "smolvlm2", "qwen3_vl", "internvideo3"]
    if backend not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend '{backend}'. Supported: {', '.join(supported)}",
        )

    # Unload current backend first (VRAM management)
    mllm = _get_mllm(cfg)
    if mllm._model is not None:
        mllm.unload()

    # Create a new MLLM instance with the requested backend
    loaded_mllm = _get_mllm_backend(backend, cfg)
    model_name = body.model_name or _get_default_model_for_backend(backend)

    if hasattr(loaded_mllm, "load"):
        success = loaded_mllm.load()
    elif hasattr(loaded_mllm, "_ensure_loaded"):
        success = loaded_mllm._ensure_loaded()
    else:
        success = False

    if not success:
        return MLLMLoadBackendResponse(
            backend=backend,
            loaded=False,
            model_name=model_name,
            mode=None,
            error=f"Failed to load {backend} backend — check dependencies, model availability, and GPU memory",
        )

    return MLLMLoadBackendResponse(
        backend=backend,
        loaded=True,
        model_name=model_name,
        mode=loaded_mllm._mode if hasattr(loaded_mllm, "_mode") else loaded_mllm._resolved_backend,
    )


def _get_default_model_for_backend(backend: str) -> str:
    models = {
        "videochat_flash": "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448",
        "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "qwen3_vl": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        "internvideo3": "OpenGVLab/InternVideo3-8B-Instruct",
    }
    return models.get(backend, "")
```

### 4.7. POST /api/mllm/backends/{backend}/unload

Unload a backend from GPU memory.

```python
@router.post(
    "/api/mllm/backends/{backend}/unload",
    response_model=MLLMUnloadBackendResponse,
    summary="Unload an MLLM backend from GPU memory",
)
async def api_mllm_unload_backend(backend: str):
    """Unload a specific MLLM backend from GPU memory, freeing VRAM for
    other pipeline stages.

    If the specified backend is the currently active backend in the default
    MLLM instance, it is unloaded and the MLLM instance is marked as unavailable.
    """
    # Check if this is the active backend
    mllm = _get_mllm(cfg)
    freed_active = False

    if mllm._resolved_backend == backend and mllm._model is not None:
        mllm.unload()
        freed_active = True

    # Also check the backend-specific instances
    if backend in _mllm_backend_instances:
        instance = _mllm_backend_instances[backend]
        if hasattr(instance, "unload"):
            instance.unload()
        del _mllm_backend_instances[backend]

    return MLLMUnloadBackendResponse(
        backend=backend,
        unloaded=True,
        message=(
            f"Backend '{backend}' unloaded from GPU memory"
            f"{' (was active backend)' if freed_active else ''}"
        ),
    )
```

---

## 5. Helper: Find Closest Frame

```python
def _find_closest_frame(metadatas: List[Dict[str, Any]], target_ts: float) -> Tuple[Optional[str], float]:
    """Find the frame_path closest to a target timestamp."""
    best_path = None
    best_time = 0.0
    best_diff = float("inf")
    for meta in metadatas:
        frame_path = meta.get("frame_path")
        if not frame_path:
            continue
        start_time = meta.get("start_time", 0.0)
        diff = abs(start_time - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_path = frame_path
            best_time = start_time
    return best_path, best_time
```

---

## 6. Helper: Resolve Video File Path

```python
def _resolve_video_path(config: Config, library_info: Any) -> Optional[str]:
    """Try to find the actual video file on disk from library info."""
    # Strategy 1: Check config's video_dir
    video_dir = config.video_dir if hasattr(config, "video_dir") else None
    if video_dir:
        candidates = [
            video_dir / library_info.filename,
            video_dir / f"{library_info.video_id}.mp4",
            video_dir / f"{library_info.video_id}.webm",
            video_dir / f"{library_info.video_id}.mkv",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

    # Strategy 2: Check data_dir
    data_dir = config.data_dir if hasattr(config, "data_dir") else None
    if data_dir:
        data_candidates = [
            data_dir / "videos" / library_info.filename,
            data_dir / "videos" / f"{library_info.video_id}.mp4",
        ]
        for c in data_candidates:
            if c.exists():
                return str(c)

    # Strategy 3: Use the video_id to check known locations
    for base in [Path("."), Path.home() / "videos", Path.home() / "Projects"]:
        for ext in [".mp4", ".webm", ".mkv", ".avi", ".mov"]:
            p = base / f"{library_info.video_id}{ext}"
            if p.exists():
                return str(p)

    return None
```

---

## 7. Client Integration (client.py)

Add corresponding methods to `VideoAnalysisClient`:

```python
# --- MLLM Endpoints ---

def mllm_describe(self, video_id: str, timestamps: List[float],
                  prompt: Optional[str] = None, max_tokens: int = 256) -> Dict[str, Any]:
    """Describe specific frames in a video using the Video MLLM."""
    return self._post(
        f"/api/videos/{video_id}/describe",
        json={"timestamps": timestamps, "prompt": prompt, "max_tokens": max_tokens},
    )

def mllm_summarize(self, video_id: str, num_frames: int = 32,
                    prompt: Optional[str] = None, max_tokens: int = 512,
                    stream: bool = False) -> Dict[str, Any]:
    """Generate a full video summary using the Video MLLM."""
    return self._post(
        f"/api/videos/{video_id}/summarize",
        json={
            "num_frames": num_frames, "prompt": prompt,
            "max_tokens": max_tokens, "stream": stream,
        },
    )

def mllm_query(self, video_id: str, query: str,
               timestamps: Optional[List[float]] = None,
               num_frames: int = 8, max_tokens: int = 512,
               stream: bool = False) -> Dict[str, Any]:
    """Ask a visual question about a video using MLLM (bypasses RAG)."""
    body = {
        "query": query,
        "timestamps": timestamps or [],
        "num_frames": num_frames,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    return self._post(f"/api/videos/{video_id}/mllm-query", json=body)

def mllm_describe_frames(self, frame_paths: List[str],
                          prompt: Optional[str] = None,
                          max_tokens: int = 256) -> Dict[str, Any]:
    """Describe arbitrary frame images using MLLM (no video ID needed)."""
    return self._post(
        "/api/mllm/describe-frames",
        json={"frame_paths": frame_paths, "prompt": prompt, "max_tokens": max_tokens},
    )

def mllm_list_backends(self) -> Dict[str, Any]:
    """List all configured MLLM backends and their status."""
    return self._get("/api/mllm/backends")

def mllm_load_backend(self, backend: str, model_name: Optional[str] = None,
                       use_fp8: bool = True, thinking_mode: bool = False) -> Dict[str, Any]:
    """Load a specific MLLM backend into GPU memory."""
    return self._post(
        f"/api/mllm/backends/{backend}/load",
        json={
            "model_name": model_name,
            "use_fp8": use_fp8,
            "thinking_mode": thinking_mode,
        },
    )

def mllm_unload_backend(self, backend: str) -> Dict[str, Any]:
    """Unload a specific MLLM backend from GPU memory."""
    return self._post(f"/api/mllm/backends/{backend}/unload")

def mllm_summarize_stream(self, video_id: str, num_frames: int = 32,
                           prompt: Optional[str] = None) -> Any:
    """Stream a video summary via SSE."""
    import requests
    url = self._url(f"/api/videos/{video_id}/summarize")
    resp = requests.post(
        url,
        json={"num_frames": num_frames, "prompt": prompt, "stream": True},
        stream=True,
        timeout=self.timeout,
    )
    if not resp.ok:
        raise APIError(resp.status_code, resp.text)
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = raw_line[6:]
        if payload == "[DONE]":
            break
        yield json.loads(payload)

def mllm_query_stream(self, video_id: str, query: str, num_frames: int = 8) -> Any:
    """Stream an MLLM query answer via SSE."""
    import requests
    url = self._url(f"/api/videos/{video_id}/mllm-query")
    resp = requests.post(
        url,
        json={"query": query, "num_frames": num_frames, "stream": True},
        stream=True,
        timeout=self.timeout,
    )
    if not resp.ok:
        raise APIError(resp.status_code, resp.text)
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = raw_line[6:]
        if payload == "[DONE]":
            break
        yield json.loads(payload)
```

---

## 8. Integration Points

| Integration | Location | Changes Needed |
|---|---|---|
| **Singleton init** | `api.py` module top | Add `_mllm_instance`, `_mllm_backend_instances`, `_get_mllm()`, `set_mllm_instance()` |
| **Pydantic schemas** | `api.py` ~line 260 | Insert 7 new request/response models after `VideoListResponse` |
| **Route factory** | Inside `create_api_router()` | Add 7 endpoint functions after existing routes (~line 1637, before `return router`) |
| **SSE generators** | `api.py` after `_chat_event_generator` | Add `_mllm_summarize_event_generator`, `_mllm_query_event_generator` |
| **Helpers** | `api.py` after `_get_scene_info_from_index` | Add `_find_closest_frame()`, `_resolve_video_path()` |
| **Client** | `client.py` after `compare_evaluations` | Add 10 new methods |
| **Config** | `config.py` (add if missing) | Add `mllm_model_name`, `mllm_backend`, `mllm_max_frames` fields |

---

## 9. Summary: What's Missing vs. What's Being Added

| Capability | Existing | New |
|---|---|---|
| MLLM singleton | ❌ None | ✅ `_get_mllm()` lazy singleton |
| Scene description | ❌ Not exposed | ✅ `POST /api/videos/{video_id}/describe` |
| Raw frame description | ❌ Not exposed | ✅ `POST /api/mllm/describe-frames` |
| Video summary | ❌ Not exposed | ✅ `POST /api/videos/{video_id}/summarize` (+ SSE streaming) |
| Visual Q&A (bypass RAG) | ❌ Not exposed | ✅ `POST /api/videos/{video_id}/mllm-query` (+ SSE streaming) |
| List backends | ❌ Not exposed | ✅ `GET /api/mllm/backends` |
| Load backend | ❌ Not exposed | ✅ `POST /api/mllm/backends/{backend}/load` |
| Unload backend | ❌ Not exposed | ✅ `POST /api/mllm/backends/{backend}/unload` |
| Client methods | ❌ None | ✅ 10 new methods in `VideoAnalysisClient` |
| Stream generators | ❌ None | ✅ 2 new SSE generators |
| Frame resolution helper | ❌ None (inline in frame endpoint) | ✅ `_find_closest_frame()` helper |
| Video path resolution | ❌ None | ✅ `_resolve_video_path()` helper |
