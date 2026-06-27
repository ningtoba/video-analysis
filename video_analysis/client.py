"""
Python API Client for the Video Analysis Platform REST API.

Provides a high-level synchronous and asynchronous client for interacting
with a running video-analysis instance via its REST API endpoints.

Usage:
    from video_analysis.client import VideoAnalysisClient

    # Synchronous client
    client = VideoAnalysisClient(base_url="http://localhost:7860")
    health = client.health()

    # Asynchronous client
    import asyncio
    result = asyncio.run(client.health_async())
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class HealthInfo:
    """Response from the /health endpoint."""

    status: str
    version: str
    gpu_available: bool
    models_loaded: Dict[str, str]
    uptime_seconds: float


@dataclass
class VideoInfo:
    """Summary information about an indexed video."""

    video_id: str
    filename: str = ""
    num_scenes: int = 0
    num_chunks: int = 0
    duration: float = 0.0
    has_sprite: bool = False


@dataclass
class JobInfo:
    """Status information about a background processing job."""

    job_id: str
    job_type: str
    status: str  # pending, running, completed, failed, cancelled
    progress: str = ""
    progress_pct: float = 0.0
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class SearchResult:
    """A single result from semantic video search."""

    chunk_id: str = ""
    video_id: str = ""
    text: str = ""
    timestamp: float = 0.0
    scene_id: int = -1
    score: float = 0.0
    frame_path: Optional[str] = None
    chunk_type: str = "scene"


@dataclass
class TranscriptSegment:
    """A single transcript segment with timing."""

    start: float
    end: float
    text: str = ""
    speaker: Optional[str] = None


@dataclass
class QueryResult:
    """Result from a question-answering query."""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Chapter:
    """A single chapter within a video."""

    title: str
    start_time: float
    end_time: float
    index: int
    summary: str = ""
    transcript_preview: str = ""
    word_count: int = 0


@dataclass
class EvaluationReport:
    """A single evaluation run report."""

    run_id: str
    timestamp: str
    version: str
    tasks: List[Dict[str, Any]]
    summary: Dict[str, Any]
    passed: bool = True


@dataclass
class EvaluationComparison:
    """Cross-report evaluation comparison result."""

    report_ids: List[str]
    comparison_data: Dict[str, Any]
    regressions: List[Dict[str, Any]]
    improvements: List[Dict[str, Any]]


@dataclass
class MLLMBackendInfo:
    """Status of a single MLLM backend."""

    name: str = ""
    available: bool = False
    loaded: bool = False
    requires_server: bool = False


@dataclass
class MLLMBackendsResult:
    """Response from GET /api/mllm/backends."""

    configured_backend: str = "auto"
    resolved_backend: Optional[str] = None
    backends: List[MLLMBackendInfo] = field(default_factory=list)


@dataclass
class MLLMDescribeResult:
    """Result from POST /api/mllm/describe or /api/mllm/summarize."""

    description: str = ""
    backend: str = ""
    error: Optional[str] = None


@dataclass
class MLLMQueryResult:
    """Result from POST /api/mllm/query."""

    answer: str = ""
    backend: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Client Errors
# ---------------------------------------------------------------------------


class ClientError(Exception):
    """Base exception for client errors."""


class ConnectionError(ClientError):
    """Raised when the API server cannot be reached."""


class APIError(ClientError):
    """Raised when the API returns a non-2xx status code."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(f"[{status_code}] {detail}")


# ---------------------------------------------------------------------------
# Synchronous Client
# ---------------------------------------------------------------------------


class VideoAnalysisClient:
    """High-level Python client for the Video Analysis REST API.

    Args:
        base_url: Root URL of the running video-analysis instance
            (e.g. ``http://localhost:7860``).
        timeout: Default HTTP request timeout in seconds.

    Usage::

        client = VideoAnalysisClient("http://localhost:7860")
        health = client.health()
        print(f"Version: {health.version}")
        print(f"GPU: {health.gpu_available}")

        # Process a video
        job = client.process_video(file_path="/path/to/video.mp4")
        print(f"Job: {job.job_id}")

        # Poll for completion
        while job.status in ("pending", "running"):
            import time
            time.sleep(2)
            job = client.get_job(job.job_id)
        print(f"Video ID: {job.result['video_id']}")

        # Ask a question
        result = client.query(job.result['video_id'], "What is in this video?")
        print(f"Answer: {result.answer}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id: str = ""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "The `requests` library is required for synchronous client usage. "
                "Install with: pip install requests"
            )
        self._requests = requests

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request and return parsed JSON."""
        url = self._url(path)
        kwargs.setdefault("timeout", self.timeout)
        try:
            resp = self._requests.request(method, url, **kwargs)
        except self._requests.exceptions.ConnectionError as exc:
            raise ConnectionError(f"Cannot connect to {url}: {exc}") from exc
        except self._requests.exceptions.Timeout as exc:
            raise ConnectionError(
                f"Request to {url} timed out after {self.timeout}s"
            ) from exc

        if not resp.ok:
            try:
                body = resp.json()
                detail = body.get("detail", resp.text)
                error_code = body.get("error_code")
            except Exception:
                detail = resp.text
                error_code = None
            raise APIError(
                status_code=resp.status_code,
                detail=str(detail),
                error_code=error_code,
            )

        try:
            return resp.json()
        except Exception:
            return resp.text

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    def _delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> HealthInfo:
        """Get the health status of the video-analysis instance."""
        data = self._get("/health")
        return HealthInfo(
            status=data.get("status", "unknown"),
            version=data.get("version", "unknown"),
            gpu_available=data.get("gpu_available", False),
            models_loaded=data.get("models_loaded", {}),
            uptime_seconds=float(data.get("uptime_seconds", 0.0)),
        )

    def health_async(self) -> Any:
        """Async health check — returns raw data for use with asyncio."""
        import asyncio

        return asyncio.to_thread(self.health)

    # ------------------------------------------------------------------
    # Video Management
    # ------------------------------------------------------------------

    def list_videos(self) -> List[VideoInfo]:
        """List all indexed videos.

        Returns:
            A list of :class:`VideoInfo` objects.
        """
        data = self._get("/api/library")
        videos = data.get("videos", [])
        return [
            VideoInfo(
                video_id=v.get("video_id", ""),
                filename=v.get("filename", ""),
                num_scenes=v.get("num_scenes", 0),
                num_chunks=v.get("num_chunks", 0),
                duration=float(v.get("duration", 0.0)),
                has_sprite=v.get("has_sprite", False),
            )
            for v in videos
        ]

    def get_video(self, video_id: str) -> VideoInfo:
        """Get detailed information about a single video.

        Args:
            video_id: The video's unique identifier.

        Returns:
            A :class:`VideoInfo` with full details.
        """
        data = self._get(f"/api/video/{video_id}")
        return VideoInfo(
            video_id=data.get("video_id", video_id),
            filename=data.get("filename", ""),
            num_scenes=data.get("num_scenes", 0),
            num_chunks=data.get("num_chunks", 0),
            duration=float(data.get("duration", 0.0)),
            has_sprite=data.get("has_sprite", False),
        )

    def get_video_detail(self, video_id: str) -> Dict[str, Any]:
        """Get full video detail from the REST API (scenes, frames, etc.).

        Note: This hits ``/api/videos/{video_id}`` (the richer REST endpoint),
        which returns scene-level metadata.

        Returns:
            A dict with full video details.
        """
        return self._get(f"/api/videos/{video_id}")

    def delete_video(self, video_id: str) -> Dict[str, Any]:
        """Delete a video from the ChromaDB index.

        Args:
            video_id: The video's unique identifier.

        Returns:
            A dict with ``video_id``, ``deleted``, and ``message`` keys.
        """
        return self._delete(f"/api/videos/{video_id}")

    # ------------------------------------------------------------------
    # Video Processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        poll: bool = False,
        poll_interval: float = 2.0,
        poll_timeout: float = 600.0,
        **upload_kwargs: Any,
    ) -> JobInfo:
        """Enqueue a video for background processing and return the job.

        Args:
            file_path: Local path to a video file to process.
            url: URL (YouTube, direct link) to download and process.
            poll: If True, synchronously wait for the job to complete.
            poll_interval: Seconds between status polls.
            poll_timeout: Max seconds to wait for completion.

        Returns:
            A :class:`JobInfo` with the job status and (if polled)
            the result.

        Raises:
            APIError: If the server rejects the request.
            ValueError: If neither ``file_path`` nor ``url`` is provided.
        """
        if file_path and url:
            raise ValueError("Provide either file_path or url, not both.")
        if url:
            data = self._post(
                "/api/videos/process",
                data={"url": url},
            )
        elif file_path:
            data = self._post(
                "/api/videos/process",
                data={"file_path": file_path},
            )
        else:
            raise ValueError("Provide either file_path or url.")

        job_id = data.get("job_id", "")
        job = self.get_job(job_id)

        if poll:
            deadline = time.time() + poll_timeout
            while job.status in ("pending", "running"):
                if time.time() > deadline:
                    raise APIError(
                        408,
                        f"Job {job_id} did not complete within {poll_timeout}s",
                    )
                time.sleep(poll_interval)
                job = self.get_job(job_id)

        return job

    def upload_video(self, file_path: str) -> JobInfo:
        """Upload a local video file for processing via multipart upload.

        Args:
            file_path: Path to the video file to upload.

        Returns:
            A :class:`JobInfo` with the job status.
        """
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "video/mp4")}
            data = self._request("POST", "/api/videos/process", files=files)

        return self.get_job(data.get("job_id", ""))

    # ------------------------------------------------------------------
    # Job Management
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> JobInfo:
        """Poll the status of a background processing job.

        Args:
            job_id: The job's unique identifier.

        Returns:
            A :class:`JobInfo` with current status.
        """
        data = self._get(f"/api/jobs/{job_id}")
        return JobInfo(
            job_id=data.get("job_id", job_id),
            job_type=data.get("job_type", ""),
            status=data.get("status", "unknown"),
            progress=data.get("progress", ""),
            progress_pct=float(data.get("progress_pct", 0.0)),
            created_at=float(data.get("created_at", 0.0)),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
        )

    def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[JobInfo]:
        """List recent processing jobs.

        Args:
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.
            status: Optional filter (pending, running, completed, failed).

        Returns:
            A list of :class:`JobInfo` objects.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        data = self._get("/api/jobs", **params)
        jobs = data.get("jobs", [])
        return [
            JobInfo(
                job_id=j.get("job_id", ""),
                job_type=j.get("job_type", ""),
                status=j.get("status", "unknown"),
                progress=j.get("progress", ""),
                progress_pct=float(j.get("progress_pct", 0.0)),
                created_at=float(j.get("created_at", 0.0)),
                started_at=j.get("started_at"),
                completed_at=j.get("completed_at"),
                result=j.get("result"),
                error=j.get("error"),
            )
            for j in jobs
        ]

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> JobInfo:
        """Synchronously wait for a job to complete.

        Args:
            job_id: The job's unique identifier.
            poll_interval: Seconds between status polls.
            timeout: Max seconds to wait.

        Returns:
            The final :class:`JobInfo` (completed or failed).
        """
        deadline = time.time() + timeout
        while True:
            job = self.get_job(job_id)
            if job.status not in ("pending", "running"):
                return job
            if time.time() > deadline:
                raise APIError(
                    408,
                    f"Job {job_id} did not complete within {timeout}s",
                )
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Query / Q&A
    # ------------------------------------------------------------------

    def query(
        self,
        video_id: str,
        question: str,
        stream: bool = False,
    ) -> QueryResult:
        """Ask a natural language question about a video.

        Args:
            video_id: The video's unique identifier.
            question: The natural language question.
            stream: If True, returns immediately (caller should use
                ``query_stream()`` for token-by-token access).

        Returns:
            A :class:`QueryResult` with answer and source citations.
        """
        if stream:
            # Return a placeholder; use query_stream() for actual streaming
            self._post(
                f"/api/videos/{video_id}/query",
                json={"query": question, "stream": True},
            )
            return QueryResult(
                answer="Streaming response — use query_stream()", sources=[]
            )

        data = self._post(
            f"/api/videos/{video_id}/query",
            json={"query": question, "stream": False},
        )
        return QueryResult(
            answer=data.get("answer", ""),
            sources=data.get("sources", []),
        )

    def query_stream(
        self,
        video_id: str,
        question: str,
    ) -> Any:
        """Ask a question and receive token-by-token SSE stream.

        This is a generator that yields token dicts as they arrive.

        Yields:
            Dicts with ``token`` key for each token chunk.
            A ``[DONE]`` sentinel signals completion.
        """
        try:
            import requests as req_lib
        except ImportError:
            raise ImportError("The `requests` library is required.")

        url = self._url(f"/api/videos/{video_id}/query")
        resp = req_lib.post(
            url,
            json={"query": question, "stream": True},
            stream=True,
            timeout=self.timeout,
        )
        if not resp.ok:
            raise APIError(resp.status_code, resp.text)

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line_str: str = str(raw_line)
            if not line_str.startswith("data: "):
                continue
            payload = line_str[6:]  # strip "data: "
            if payload == "[DONE]":
                break
            yield json.loads(payload)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Perform cross-video semantic search.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            A list of :class:`SearchResult` sorted by relevance.
        """
        data = self._get(
            "/api/videos/search",
            query=query,
            top_k=top_k,
        )
        results = data.get("results", [])
        return [
            SearchResult(
                chunk_id=r.get("chunk_id", ""),
                video_id=r.get("video_id", ""),
                text=r.get("text", ""),
                timestamp=float(r.get("timestamp", 0.0)),
                scene_id=int(r.get("scene_id", -1)),
                score=float(r.get("score", 0.0)),
                frame_path=r.get("frame_path"),
                chunk_type=r.get("chunk_type", "scene"),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Transcript & Chapters
    # ------------------------------------------------------------------

    def get_transcript(
        self,
        video_id: str,
    ) -> Dict[str, Any]:
        """Get the full transcript for a video.

        Args:
            video_id: The video's unique identifier.

        Returns:
            A dict with ``segments`` (list of :class:`TranscriptSegment`)
            and ``full_transcript`` (string).
        """
        data = self._get(f"/api/videos/{video_id}/transcript")
        segments = [
            TranscriptSegment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=s.get("text", ""),
                speaker=s.get("speaker"),
            )
            for s in data.get("segments", [])
        ]
        return {
            "segments": segments,
            "full_transcript": data.get("full_transcript", ""),
        }

    def get_chapters(self, video_id: str) -> List[Chapter]:
        """Get auto-generated chapters for a video.

        Args:
            video_id: The video's unique identifier.

        Returns:
            A list of :class:`Chapter` objects.
        """
        data = self._get(f"/api/videos/{video_id}/chapters")
        chapters = data.get("chapters", [])
        return [
            Chapter(
                title=c.get("title", ""),
                start_time=float(c.get("start_time", 0.0)),
                end_time=float(c.get("end_time", 0.0)),
                index=int(c.get("index", 0)),
                summary=c.get("summary", ""),
                transcript_preview=c.get("transcript_preview", ""),
                word_count=int(c.get("word_count", 0)),
            )
            for c in chapters
        ]

    # ------------------------------------------------------------------
    # Frame Retrieval
    # ------------------------------------------------------------------

    def get_frame(self, video_id: str, timestamp: float) -> bytes:
        """Get a frame image at a specific timestamp.

        Args:
            video_id: The video's unique identifier.
            timestamp: Timestamp in seconds.

        Returns:
            Raw image bytes (JPEG or PNG).
        """
        try:
            import requests as req_lib
        except ImportError:
            raise ImportError("The `requests` library is required.")

        url = self._url(f"/api/videos/{video_id}/frames/{timestamp}")
        resp = req_lib.get(url, timeout=self.timeout)
        if not resp.ok:
            raise APIError(resp.status_code, resp.text)
        return resp.content

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def list_evaluations(self) -> List[Dict[str, Any]]:
        """List saved evaluation reports.

        Returns:
            A list of report summary dicts (paginated, newest first).
        """
        data = self._get("/api/evaluations")
        return data.get("reports", [])

    def get_evaluation(self, run_id: str) -> EvaluationReport:
        """Get a full evaluation report by run ID.

        Args:
            run_id: The evaluation run's unique identifier.

        Returns:
            An :class:`EvaluationReport` with full task results.
        """
        data = self._get(f"/api/evaluations/{run_id}")
        return EvaluationReport(
            run_id=data.get("run_id", run_id),
            timestamp=data.get("timestamp", ""),
            version=data.get("version", ""),
            tasks=data.get("tasks", []),
            summary=data.get("summary", {}),
            passed=data.get("passed", True),
        )

    def compare_evaluations(self, run_ids: List[str]) -> EvaluationComparison:
        """Compare multiple evaluation reports side-by-side.

        Args:
            run_ids: List of evaluation run IDs to compare.

        Returns:
            An :class:`EvaluationComparison` with diff data.
        """
        ids_param = ",".join(run_ids)
        data = self._get(
            "/api/evaluations/compare",
            run_ids=ids_param,
        )
        return EvaluationComparison(
            report_ids=run_ids,
            comparison_data=data.get("comparison_data", {}),
            regressions=data.get("regressions", []),
            improvements=data.get("improvements", []),
        )

    # ------------------------------------------------------------------
    # Video MLLM Direct API (v0.55.0)
    # ------------------------------------------------------------------

    def get_mllm_backends(self) -> MLLMBackendsResult:
        """List available MLLM backends and their status.

        Returns:
            An :class:`MLLMBackendsResult` with backend status info.
        """
        data = self._get("/api/mllm/backends")
        backends = [
            MLLMBackendInfo(
                name=b.get("name", ""),
                available=b.get("available", False),
                loaded=b.get("loaded", False),
                requires_server=b.get("requires_server", False),
            )
            for b in data.get("backends", [])
        ]
        return MLLMBackendsResult(
            configured_backend=data.get("configured_backend", "auto"),
            resolved_backend=data.get("resolved_backend"),
            backends=backends,
        )

    def mllm_describe(
        self,
        frames: List[str],
        prompt: str = "Describe what's happening in these frames in detail.",
        max_tokens: int = 256,
    ) -> MLLMDescribeResult:
        """Describe frames using the video MLLM.

        Args:
            frames: List of frame file paths to describe.
            prompt: Optional custom prompt.
            max_tokens: Max tokens in response.

        Returns:
            An :class:`MLLMDescribeResult` with the description.
        """
        data = self._post(
            "/api/mllm/describe",
            json={
                "frames": frames,
                "prompt": prompt,
                "max_tokens": max_tokens,
            },
        )
        return MLLMDescribeResult(
            description=data.get("description", ""),
            backend=data.get("backend", ""),
            error=data.get("error"),
        )

    def mllm_summarize(
        self,
        video_id: str = "",
        video_path: Optional[str] = None,
        prompt: str = "Summarize the key content, events, and subjects of this video.",
        num_frames: int = 32,
    ) -> MLLMDescribeResult:
        """Summarize a video using the MLLM.

        Args:
            video_id: Indexed video ID.
            video_path: Local video file path.
            prompt: Custom summary prompt.
            num_frames: Number of frames to sample.

        Returns:
            An :class:`MLLMDescribeResult` with the summary.
        """
        data = self._post(
            "/api/mllm/summarize",
            json={
                "video_id": video_id,
                "video_path": video_path,
                "prompt": prompt,
                "num_frames": num_frames,
            },
        )
        return MLLMDescribeResult(
            description=data.get("description", ""),
            backend=data.get("backend", ""),
            error=data.get("error"),
        )

    def mllm_query(
        self,
        query: str,
        video_id: str = "",
        video_path: Optional[str] = None,
        num_frames: int = 16,
    ) -> MLLMQueryResult:
        """Ask a visual question via the MLLM (bypassing RAG).

        Args:
            query: Natural language question.
            video_id: Indexed video ID.
            video_path: Local video file path.
            num_frames: Number of frames to sample.

        Returns:
            An :class:`MLLMQueryResult` with the answer.
        """
        data = self._post(
            "/api/mllm/query",
            json={
                "query": query,
                "video_id": video_id,
                "video_path": video_path,
                "num_frames": num_frames,
            },
        )
        return MLLMQueryResult(
            answer=data.get("answer", ""),
            backend=data.get("backend", ""),
            error=data.get("error"),
        )

    def mllm_load_backend(
        self,
        backend: str = "auto",
        model_size: str = "2.2B",
        use_fp8: bool = True,
    ) -> Dict[str, Any]:
        """Load a specific MLLM backend onto GPU.

        Args:
            backend: Backend to load (auto, internvideo3, qwen3_vl, smolvlm2, videochat_flash).
            model_size: Model size for SmolVLM2.
            use_fp8: Enable FP8 quantization.

        Returns:
            Dict with success, backend, resolved_backend, and message keys.
        """
        return self._post(
            "/api/mllm/backends/load",
            json={
                "backend": backend,
                "model_size": model_size,
                "use_fp8": use_fp8,
            },
        )

    def mllm_unload_backend(self) -> Dict[str, Any]:
        """Unload the current MLLM backend from GPU memory.

        Returns:
            Dict with success and message keys.
        """
        return self._post("/api/mllm/backends/unload")


# ---------------------------------------------------------------------------
# Async Client (helper functions)
# ---------------------------------------------------------------------------


async def async_health(base_url: str = "http://localhost:7860") -> HealthInfo:
    """Convenience async helper for checking health.

    Args:
        base_url: Root URL of the video-analysis instance.

    Returns:
        A :class:`HealthInfo` object.
    """
    client = VideoAnalysisClient(base_url=base_url)
    return await client.health_async()
