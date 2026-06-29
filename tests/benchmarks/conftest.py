"""Benchmark infrastructure: GPUProfiler context manager."""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class GPUProfiler:
    """Context manager that profiles GPU memory usage around a code block.

    Uses ``torch.cuda`` APIs when a GPU is available; otherwise records
    ``n/a`` for all GPU metrics.

    Example usage::

        with GPUProfiler("encode frames") as prof:
            result = pipeline._encode_frames(frames)

        print(f"Peak GPU memory: {prof.peak_mib:.0f} MiB")
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.start_mib: float = 0.0
        self.peak_mib: float = 0.0
        self.end_mib: float = 0.0
        self.elapsed: float = 0.0
        self._gpu_available: bool = False

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

        if self.label:
            logger.info(
                "[GPUProfiler] %s — elapsed=%.2fs start=%.0fMiB "
                "peak=%.0fMiB end=%.0fMiB",
                self.label,
                self.elapsed,
                self.start_mib,
                self.peak_mib,
                self.end_mib,
            )

    @property
    def gpu_used(self) -> bool:
        """Whether GPU was available during profiling."""
        return self._gpu_available

