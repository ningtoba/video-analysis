"""
In-Process Async Job Queue — background video processing with job tracking.

Provides an in-process async job queue (zero external dependencies — no Redis, no
Celery) for offloading long-running video pipeline work from the HTTP request
path. Jobs are processed sequentially in a background asyncio.TaskGroup (Python
3.11+), with full status tracking, error capture, and result storage.

The JobManager singleton is started once at application startup via its
``lifespan`` contextmanager and stopped gracefully on shutdown.  All mutable
state lives in a thread-safe dict behind an asyncio.Lock.

Usage:
    manager = JobManager()
    job = await manager.enqueue("process", video_path="/tmp/video.mp4")
    # job.job_id is the handle the caller returns immediately
    # Later, poll GET /api/jobs/{job_id} which calls manager.get_job(job_id)

Thread safety: all public methods are coroutines that acquire the internal lock
before mutation.  The background worker runs in an asyncio Task, so there is no
GIL concern with the synchronous parts of the pipeline (which are dispatched to
a ThreadPoolExecutor inside the worker coroutine).

No external dependencies beyond Python 3.11 stdlib.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════


class JobStatus(str, Enum):
    """Lifecycle states for a processing job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A single background processing job.

    Attributes:
        job_id: Unique identifier (UUID v4 hex).
        job_type: Discriminator — e.g. ``"process_video"``.
        status: Current lifecycle status.
        progress: Human-readable progress message (optional).
        progress_pct: Estimated completion percentage (0-100, optional).
        created_at: Unix timestamp when the job was enqueued.
        started_at: Unix timestamp when processing began (or None).
        completed_at: Unix timestamp when the job finished (or None).
        params: Keyword parameters for the processing function.
        result: Structured result dict (populated on success).
        error: Human-readable error message (populated on failure).
        traceback: Optional detailed traceback string.
    """

    job_id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    progress: str = ""
    progress_pct: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# Manager
# ═══════════════════════════════════════════════════════════════════════════


class _BackgroundWorkerRegistry:
    """Registry of callables keyed by job_type for dispatch.

    External code registers handlers via ``register()``.  The JobManager
    looks up the handler when a job is ready to process.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Any] = {}

    def register(self, job_type: str, handler: Any) -> None:
        """Register *handler* for jobs of *job_type*.

        The handler must be a callable accepting (**params) -> dict.
        It may be sync (dispatched to a thread pool) or async (run directly).
        """
        if job_type in self._handlers:
            logger.warning("Overwriting existing handler for job_type=%r", job_type)
        self._handlers[job_type] = handler

    def get(self, job_type: str) -> Any:
        """Return the handler for *job_type*, or raise KeyError."""
        return self._handlers[job_type]


class JobManager:
    """Singleton async job queue manager.

    Usage:
        manager = JobManager()
        async with manager.lifespan(app):
            job = await manager.enqueue("process_video", video_path="...")
            status = await manager.get_job(job.job_id)
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        poll_interval: float = 0.5,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._poll_interval = poll_interval
        self._lock = asyncio.Lock()
        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._worker_registry = _BackgroundWorkerRegistry()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._shutdown_event = asyncio.Event()

    # -- registration -------------------------------------------------------

    def register_handler(self, job_type: str, handler: Any) -> None:
        """Register a callable for the given *job_type*."""
        self._worker_registry.register(job_type, handler)

    # -- lifecycle ----------------------------------------------------------

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """FastAPI lifespan context — start worker on enter, stop on exit.

        Usage:
            app = FastAPI(lifespan=manager.lifespan)
        """
        await self._start()
        yield
        await self._stop()

    async def _start(self) -> None:
        """Start the background worker task and semaphore."""
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("JobManager started (max_concurrent=%s)", self._max_concurrent)

    async def _stop(self) -> None:
        """Signal shutdown and wait for the worker to finish."""
        self._shutdown_event.set()
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
                logger.warning("JobManager worker task cancelled on shutdown")
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("JobManager stopped")

    # -- public API ---------------------------------------------------------

    async def enqueue(self, job_type: str, **params: Any) -> Job:
        """Create a new job and add it to the processing queue.

        Returns the Job immediately (the ``job_id`` field is the handle
        the caller should return to the user for polling).
        """
        job = Job(
            job_id=uuid.uuid4().hex,
            job_type=job_type,
            params=params,
        )
        async with self._lock:
            self._jobs[job.job_id] = job
        await self._queue.put(job.job_id)
        logger.info("Enqueued job %s (type=%s)", job.job_id[:8], job_type)
        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Return the Job for *job_id*, or None if not found."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[JobStatus] = None,
    ) -> List[Job]:
        """Return recent jobs, newest first.

        Supports pagination with *limit*/*offset* and optional *status_filter*.
        """
        async with self._lock:
            all_jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )
        if status_filter:
            all_jobs = [j for j in all_jobs if j.status == status_filter]
        return all_jobs[offset : offset + limit]

    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[str] = None,
        progress_pct: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        traceback: Optional[str] = None,
    ) -> Optional[Job]:
        """Update fields on an existing job.

        Returns the updated Job, or None if not found.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
                if status == JobStatus.RUNNING and job.started_at is None:
                    job.started_at = time.time()
                if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = time.time()
            if progress is not None:
                job.progress = progress
            if progress_pct is not None:
                job.progress_pct = progress_pct
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if traceback is not None:
                job.traceback = traceback
            return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job.  Returns True if cancelled, False otherwise.

        Running jobs cannot be cancelled (they are already in-progress on
        the worker task).  Returns False for already-completed or not-found
        jobs.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.PENDING:
                return False
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            return True

    # -- internal -----------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Background loop: pull job IDs from the queue and process them."""
        sem = self._semaphore
        assert sem is not None  # set during _start

        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                # Poll the queue with a timeout so we can check shutdown.
                job_id = await asyncio.wait_for(
                    self._queue.get(), timeout=self._poll_interval
                )
            except asyncio.TimeoutError:
                continue

            async with sem:
                job = await self.get_job(job_id)
                if job is None:
                    continue
                if job.status == JobStatus.CANCELLED:
                    logger.info("Skipping cancelled job %s", job_id[:8])
                    continue

                # Mark running
                await self.update_job(
                    job_id,
                    status=JobStatus.RUNNING,
                    progress="Initialising...",
                    progress_pct=0.0,
                )

                try:
                    handler = self._worker_registry.get(job.job_type)
                except KeyError:
                    await self.update_job(
                        job_id,
                        status=JobStatus.FAILED,
                        error=f"No handler registered for job_type={job.job_type!r}",
                        progress="Failed — unknown job type",
                    )
                    continue

                # Dispatch — handler may be sync or async
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(self, job_id, **job.params)
                    else:
                        loop = asyncio.get_event_loop()
                        # run_in_executor only supports positional args, so
                        # we wrap the call in a lambda that unpacks kwargs.
                        result = await loop.run_in_executor(
                            None,
                            lambda: handler(self, job_id, **job.params),
                        )
                    await self.update_job(
                        job_id,
                        status=JobStatus.COMPLETED,
                        progress="Complete",
                        progress_pct=100.0,
                        result=result,
                    )
                except Exception as exc:
                    logger.exception("Job %s failed", job_id[:8])
                    await self.update_job(
                        job_id,
                        status=JobStatus.FAILED,
                        progress=f"Failed: {exc}",
                        error=str(exc),
                        traceback=_format_exc(),
                    )

    @property
    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the job queue."""
        # Counts are computed without the lock since we only need approximate
        # values for monitoring/dashboarding.
        counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for job in self._jobs.values():
            s = job.status.value
            if s in counts:
                counts[s] += 1
        return {
            "total_jobs": len(self._jobs),
            "queue_depth": self._queue.qsize(),
            "max_concurrent": self._max_concurrent,
            "counts": counts,
        }


# -- helpers ----------------------------------------------------------------


def _format_exc() -> str:
    """Return the current exception traceback as a compact string."""
    import traceback

    return "".join(traceback.format_exc())


# -- default handler --------------------------------------------------------
# These are registered by api.py at import time.

_DEFAULT_MANAGER: Optional[JobManager] = None


def get_default_manager() -> JobManager:
    """Return the module-level default JobManager singleton.

    Created lazily on first access.
    """
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = JobManager()
    return _DEFAULT_MANAGER


def reset_default_manager() -> None:
    """Reset the default manager (for testing)."""
    global _DEFAULT_MANAGER
    _DEFAULT_MANAGER = None
