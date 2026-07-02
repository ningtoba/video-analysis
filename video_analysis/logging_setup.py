"""
Structured JSON logging setup for the video analysis platform.

Uses structlog for smart output:
- TTY → colored ConsoleRenderer
- File/pipe → JSONRenderer

Usage:
    from video_analysis.logging_setup import setup_logging, PipelineLogger

    setup_logging()  # call once at startup
    pl = PipelineLogger()
    pl.log_stage_start("transcribe", video_id="abc123")
"""

import logging as stdlib_logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _simple_filter_by_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Simple level filter that works with PrintLoggerFactory.

    This replaces structlog.stdlib.filter_by_level which requires
    a stdlib logger with .disabled/.level attributes.
    """
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }
    event_level = levels.get(event_dict.get("level", "").upper(), 0)
    if event_level < _LogConfig.filter_level:
        raise structlog.DropEvent
    return event_dict


class _LogConfig:
    """Sentinel class holding logging configuration state."""

    filter_level: int = stdlib_logging.INFO
    initialized: bool = False


def _is_tty() -> bool:
    """Return True if stdout is a terminal (TTY)."""
    return sys.stdout.isatty()


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure structlog once at application startup.

    Parameters
    ----------
    level : str, optional
        Log level string (e.g. "DEBUG", "INFO", "WARNING", "ERROR").
        Falls back to the STRUCTURED_LOGGING_LEVEL env var, then "INFO".
    fmt : str, optional
        Output format: "console", "json", or "auto".
        Falls back to the STRUCTURED_LOGGING_FORMAT env var, then "auto".
        "auto" → ConsoleRenderer on TTY, JSONRenderer otherwise.
    """
    if _LogConfig.initialized:
        return

    import os
    log_level = (level or os.environ.get("STRUCTURED_LOGGING_LEVEL") or "INFO").upper()
    log_format = (fmt or os.environ.get("STRUCTURED_LOGGING_FORMAT") or "auto").lower()

    # Map string level to numeric logging constant
    numeric_level = getattr(stdlib_logging, log_level, stdlib_logging.INFO)

    # Set the global filter level for the non-stdlib filter processor
    _LogConfig.filter_level = numeric_level

    # Choose renderer
    if log_format == "console" or (log_format == "auto" and _is_tty()):
        renderer: Any = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            _simple_filter_by_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libs respect our level
    stdlib_logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=False,
    )

    _LogConfig.initialized = True


class PipelineLogger:
    """Stage-aware logger for pipeline progress and errors.

    Each method logs a structured event with consistent key-value pairs
    that can be consumed by JSON log aggregators or viewed in the console.
    """

    def __init__(self, logger_name: str = "pipeline") -> None:
        self._logger = structlog.get_logger(logger_name)

    def log_stage_start(
        self,
        stage: str,
        video_id: str,
        **kwargs: Any,
    ) -> None:
        """Log the start of a pipeline stage.

        Parameters
        ----------
        stage : str
            Pipeline stage name (e.g. 'transcribe', 'detect_scenes').
        video_id : str
            Identifier for the video being processed.
        **kwargs
            Additional key-value pairs to include in the log event.
        """
        self._logger.info("stage_start", stage=stage, video_id=video_id, **kwargs)

    def log_stage_end(
        self,
        stage: str,
        video_id: str,
        duration: float,
        **kwargs: Any,
    ) -> None:
        """Log the successful completion of a pipeline stage.

        Parameters
        ----------
        stage : str
            Pipeline stage name.
        video_id : str
            Identifier for the video.
        duration : float
            Wall-clock duration in seconds for the stage.
        **kwargs
            Additional key-value pairs to include in the log event.
        """
        self._logger.info("stage_end", stage=stage, video_id=video_id, duration=duration, **kwargs)

    def log_error(
        self,
        stage: str,
        video_id: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        """Log an error that occurred during a pipeline stage.

        Parameters
        ----------
        stage : str
            Pipeline stage name where the error occurred.
        video_id : str
            Identifier for the video being processed.
        error : str
            Human-readable error description or exception message.
        **kwargs
            Additional key-value pairs to include in the log event.
        """
        self._logger.error("stage_error", stage=stage, video_id=video_id, error=error, **kwargs)
