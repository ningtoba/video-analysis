"""
Pipeline Benchmarking — per-stage VRAM tracking, timing, and profiling.

Provides ``PipelineBenchmark`` context manager that wraps pipeline execution
to capture per-stage wall-clock time, GPU memory deltas, and optional
pytest-benchmark integration.

Usage::

    from video_analysis.benchmark import PipelineBenchmark

    with PipelineBenchmark("process_video") as bm:
        index = pipeline.process("test.mp4")

    print(bm.report())
    # Stage          Duration (s)  VRAM Start (MiB)  VRAM Peak (MiB)
    # audio_extract          2.34              512               768
    # scene_detect          15.21              768              1024
    # ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageRecord:
    """Profiling data for a single pipeline stage."""

    name: str
    duration_s: float
    vram_start_mib: float = 0.0
    vram_peak_mib: float = 0.0
    vram_end_mib: float = 0.0


class GPUProfiler:
    """Context manager that captures GPU memory around a code block.

    Works as a lightweight context manager in pipeline stages.  Use it
    standalone or let ``PipelineBenchmark`` collect the records.

    Example::

        with GPUProfiler("transcribe") as prof:
            result = whisper_model.transcribe(audio)

        print(f"Peak VRAM: {prof.peak_mib:.0f} MiB")
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.start_mib: float = 0.0
        self.peak_mib: float = 0.0
        self.end_mib: float = 0.0
        self.elapsed: float = 0.0
        self._gpu_available = False
        self._start = 0.0

    def __enter__(self) -> "GPUProfiler":
        self._start = time.perf_counter()
        try:
            import torch

            if torch.cuda.is_available():
                self._gpu_available = True
                torch.cuda.reset_peak_memory_stats()
                self.start_mib = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                self.start_mib = 0.0
        except ImportError:
            self.start_mib = 0.0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
        try:
            import torch

            if self._gpu_available and torch.cuda.is_available():
                self.peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self.end_mib = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                self.peak_mib = 0.0
                self.end_mib = 0.0
        except ImportError:
            self.peak_mib = 0.0
            self.end_mib = 0.0

        logger.debug(
            "[GPUProfiler] %s — elapsed=%.2f start=%.0fMiB " "peak=%.0fMiB end=%.0fMiB",
            self.label or "?",
            self.elapsed,
            self.start_mib,
            self.peak_mib,
            self.end_mib,
        )

    def to_stage_record(self, name: str) -> StageRecord:
        return StageRecord(
            name=name,
            duration_s=self.elapsed,
            vram_start_mib=self.start_mib,
            vram_peak_mib=self.peak_mib,
            vram_end_mib=self.end_mib,
        )


class PipelineBenchmark:
    """Collects per-stage profiling records during pipeline execution.

    Stages push records via ``record_stage(name, profiler)``, and the
    final report is emitted as structured JSON and human-readable table.

    Usage::

        bm = PipelineBenchmark("my_video.mp4")
        with GPUProfiler("transcribe") as prof:
            ...
        bm.record_stage("transcribe", prof)
        print(bm.report())
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.stages: list[StageRecord] = []
        self._start = time.perf_counter()
        self._end: float | None = None

    def __enter__(self) -> "PipelineBenchmark":
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()

    def record_stage(self, name: str, profiler: GPUProfiler) -> None:
        """Record a stage from a GPUProfiler instance."""
        self.stages.append(profiler.to_stage_record(name))
        logger.info(
            "[Benchmark] %s — %.2fs (start: %.0f MiB, peak: %.0f MiB)",
            name,
            profiler.elapsed,
            profiler.start_mib,
            profiler.peak_mib,
        )

    def finish(self) -> None:
        """Mark the end of the benchmark."""
        self._end = time.perf_counter()

    @property
    def total_duration(self) -> float:
        return (self._end or time.perf_counter()) - self._start

    def report(self) -> str:
        """Return a formatted benchmark report table."""
        lines = [
            f"Pipeline Benchmark: {self.label or 'unnamed'}",
            f"Total: {self.total_duration:.2f}s, Stages: {len(self.stages)}",
            "",
            f"{'Stage':<25} {'Duration (s)':<14} {'VRAM Start':<12} {'VRAM Peak':<12} {'VRAM End':<12}",
            "-" * 75,
        ]
        for s in self.stages:
            lines.append(
                f"{s.name:<25} {s.duration_s:<14.2f} "
                f"{s.vram_start_mib:<12.0f} {s.vram_peak_mib:<12.0f} "
                f"{s.vram_end_mib:<12.0f}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Return benchmark data as a dict (JSON-serialisable)."""
        return {
            "label": self.label,
            "total_duration_s": self.total_duration,
            "stages": [
                {
                    "name": s.name,
                    "duration_s": s.duration_s,
                    "vram_start_mib": s.vram_start_mib,
                    "vram_peak_mib": s.vram_peak_mib,
                    "vram_end_mib": s.vram_end_mib,
                }
                for s in self.stages
            ],
        }
