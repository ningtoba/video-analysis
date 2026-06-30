"""
OpenTelemetry tracing — distributed tracing for the video analysis platform.

Provides a ``TelemetryContext`` context manager for tracing pipeline stages,
RAG retrieval, API requests, and evaluation runs.  Spans are exported via
OTLP gRPC/HTTP, configured through standard ``OTEL_*`` environment variables.

The module uses lazy initialisation so importing it has zero side effects —
the tracer provider is created on first span creation.  When ``opentelemetry``
packages are not installed, all operations are no-ops and no imports are
attempted.

Usage (auto-initialising)::

    from video_analysis.telemetry import TelemetryContext, trace_pipeline, get_trace_id

    # Context manager — synchronous or async
    with TelemetryContext("transcribe", video_id="abc123", model="large-v3") as ctx:
        result = do_things()
        ctx.set_attribute("duration_s", 42.5)
        ctx.set_status_ok()
    # on exception, automatically records error status

    # Decorator for async pipeline functions
    @trace_pipeline(stage="rag_retrieval", attributes={"method": "hybrid"})
    async def retrieve_chunks(video_id: str) -> list[str]:
        ...

    # Extract trace context from incoming HTTP headers
    from video_analysis.telemetry import parent_span_from_headers

    headers = {"traceparent": "00-...-...-01"}
    span = parent_span_from_headers(headers, "api_request")
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Coroutine
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Mapping,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

_TRACE_ID_FORMAT: str = "032x"

# ── Module-level state ──────────────────────────────────────────────────

_initialised = False
_tracer_provider: Any = None
_tracer: Any = None

# Sentinel for opentelemetry availability
enabled: bool = True  # will be set to False if import fails


# ── TracingSpan Protocol ────────────────────────────────────────────────


@runtime_checkable
class TracingSpan(Protocol):
    """Protocol describing the minimal span interface used throughout this module.

    Concrete implementations can be OpenTelemetry span objects or the
    local ``_NoopSpan`` when OpenTelemetry is not installed.
    """

    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...

    def add_event(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> None: ...

    def set_status(self, status: Any) -> None: ...

    def end(self) -> None: ...

    def is_recording(self) -> bool: ...

    def get_span_context(self) -> Any: ...

    def update_name(self, name: str) -> None: ...


# ── No-op span (used when opentelemetry is not installed) ──────────────


class _NoopSpan(TracingSpan):
    """A no-op span that satisfies the TracingSpan protocol.

    All methods are safe to call — they silently do nothing.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def get_span_context(self) -> Any:
        return None

    def update_name(self, name: str) -> None:
        pass


class _NoopTracer:
    """A no-op tracer that returns ``_NoopSpan`` instances."""

    def start_span(self, name: str, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> contextmanager:  # type: ignore[type-arg]
        @contextmanager
        def _noop_cm() -> Generator[_NoopSpan, None, None]:
            yield _NoopSpan()

        return _noop_cm()  # type: ignore[return-value]


# ── Lazy initialisation ─────────────────────────────────────────────────


def _ensure_telemetry() -> None:
    """Lazy-init the OpenTelemetry tracer provider on first use."""
    global _initialised, _tracer_provider, _tracer, enabled

    if _initialised:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "opentelemetry packages not installed — tracing is disabled. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-grpc"
        )
        enabled = False

        class _DummyTrace:
            @staticmethod
            def get_current_span() -> _NoopSpan:
                return _NoopSpan()

            @staticmethod
            def use_span(span: Any) -> contextmanager:  # type: ignore[type-arg]
                @contextmanager
                def _noop_cm() -> Generator[_NoopSpan, None, None]:
                    yield _NoopSpan()

                return _noop_cm()  # type: ignore[return-value]

            def __getattr__(self, name: str) -> Any:
                return lambda *args, **kwargs: None

        trace = _DummyTrace()  # type: ignore[assignment]
        _tracer = _NoopTracer()
        _initialised = True
        return

    # Build resource with service name
    import os

    service_name = os.environ.get("OTEL_SERVICE_NAME", "video-analysis")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

    # Configure OTLP exporter if endpoint is set, else use console export
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            logger.info("OTLP tracing enabled, endpoint=%s", otlp_endpoint)
        except Exception as exc:
            logger.warning("Failed to configure OTLP exporter: %s", exc)
    else:
        # Console exporter as fallback — good for local development
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info(
                "OTLP tracing disabled (no OTEL_EXPORTER_OTLP_ENDPOINT), using console exporter"
            )
        except Exception as exc:
            logger.warning("Failed to configure console exporter: %s", exc)

    # Set global tracer provider
    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    try:
        _version = __import__("importlib.metadata").metadata.version("video-analysis")
    except Exception:
        _version = "0.0.0"
    _tracer = trace.get_tracer(service_name, _version)
    enabled = True
    _initialised = True


def _get_tracer() -> Any:
    """Return the current tracer, initialising if needed."""
    _ensure_telemetry()
    return _tracer


# ── Status codes ────────────────────────────────────────────────────────


# Lazy-import status objects so we don't fail at module level
_STATUS_UNSET: Any = None
_STATUS_OK: Any = None
_STATUS_ERROR: Any = None
_OTEL_STATUS_CLS: Any = None


def _ensure_status_codes() -> None:
    """Lazy-import OpenTelemetry status codes (no-op when not available)."""
    global _STATUS_UNSET, _STATUS_OK, _STATUS_ERROR, _OTEL_STATUS_CLS
    if _STATUS_UNSET is not None:
        return
    try:
        from opentelemetry.trace.status import Status, StatusCode

        _OTEL_STATUS_CLS = Status
        _STATUS_UNSET = StatusCode.UNSET
        _STATUS_OK = StatusCode.OK
        _STATUS_ERROR = StatusCode.ERROR
    except ImportError:
        _STATUS_UNSET = 0
        _STATUS_OK = 1
        _STATUS_ERROR = 2


def _make_status(status_code: Any, description: str = "") -> Any:
    """Create a proper Status object (no-op fallback when OTel unavailable)."""
    _ensure_status_codes()
    if _OTEL_STATUS_CLS is not None:
        return _OTEL_STATUS_CLS(status_code, description)
    return _Status(status_code, description)


class _Status:
    """Fallback status holder when opentelemetry is not installed."""

    def __init__(self, status_code: Any, description: str = "") -> None:
        self.status_code = status_code
        self.description = description


# ── TelemetryContext ────────────────────────────────────────────────────


class TelemetryContext:
    """Context manager for tracing a single pipeline stage or operation.

    Use inside ``with`` blocks to automatically create a span that is
    closed on exit.  If the block raises, the span is recorded as an
    error.  All methods are safe to call even when OpenTelemetry is not
    installed — they silently become no-ops.

    Args:
        name: Span / operation name (e.g. ``"transcribe"``, ``"rag_retrieval"``).
        **attributes: Initial key-value attributes to set on the span.

    Example::

        with TelemetryContext("transcribe", video_id="abc123") as ctx:
            result = transcribe(video)
            ctx.set_attribute("duration_s", result.duration)
            ctx.set_status_ok()
    """

    def __init__(self, name: str, **attributes: Any) -> None:
        self._name = name
        self._initial_attributes = attributes
        self._span: Any = _NoopSpan()
        self._tracer: Any = _NoopTracer()
        self._exited = False

    def __enter__(self) -> TelemetryContext:
        self._tracer = _get_tracer()
        self._span = self._tracer.start_span(self._name)
        if self._initial_attributes:
            self._span.set_attributes(self._initial_attributes)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> Optional[bool]:
        if self._exited:
            return None
        self._exited = True

        if exc_type is not None:
            # Record error status
            _ensure_status_codes()
            self._span.set_status(_make_status(_STATUS_ERROR, str(exc_val)))
            self._span.set_attribute("error.type", exc_type.__name__)
            self._span.set_attribute("success", False)
            self._span.add_event("exception", {"exception.message": str(exc_val)})
        else:
            # OK only if explicitly set by user; otherwise leave as UNSET
            pass

        self._span.end()
        return None  # Do not suppress exceptions

    async def __aenter__(self) -> TelemetryContext:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> Optional[bool]:
        return self.__exit__(exc_type, exc_val, exc_tb)

    # ── Public API ──────────────────────────────────────────────────────

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single attribute on the current span.

        Args:
            key: Attribute name.
            value: Attribute value (int, float, str, bool, or sequence thereof).
        """
        self._span.set_attribute(key, value)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        """Set multiple attributes at once.

        Args:
            attributes: Mapping of attribute names to values.
        """
        self._span.set_attributes(attributes)

    def add_event(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> None:
        """Record a timed event on the current span.

        Args:
            name: Event name.
            attributes: Optional key-value event attributes.
        """
        self._span.add_event(name, attributes)

    def set_status_ok(self) -> None:
        """Mark the span as successful (StatusCode.OK)."""
        _ensure_status_codes()
        self._span.set_status(_make_status(_STATUS_OK))
        self._span.set_attribute("success", True)

    def set_status_error(self, description: str = "") -> None:
        """Mark the span as errored (StatusCode.ERROR).

        Args:
            description: Optional error description.
        """
        _ensure_status_codes()
        self._span.set_status(_make_status(_STATUS_ERROR, description))
        self._span.set_attribute("success", False)
        self._span.set_attribute("error.type", "error")

    def get_span_context(self) -> Any:
        """Return the underlying span context (or ``None`` if no active span)."""
        return self._span.get_span_context()

    def update_name(self, name: str) -> None:
        """Change the span name mid-operation.

        Args:
            name: New span name.
        """
        self._span.update_name(name)


# ── Decorator for async pipeline functions ──────────────────────────────


P = ParamSpec("P")
R = TypeVar("R")


def trace_pipeline(
    stage: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
    capture_return: bool = False,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]],
    Callable[P, Coroutine[Any, Any, R]],
]:
    """Decorator that wraps an async pipeline function in a tracing span.

    The span is created with the given stage name and optional attributes.
    On successful completion the span is marked OK; on exception it is
    marked ERROR with the exception details recorded.

    Args:
        stage: The pipeline stage name (used as the span name).
        attributes: Optional static attributes to attach to every span.
        capture_return: If ``True``, set a ``result`` attribute on the span
            from the function's return value (converted via ``str()``).

    Returns:
        Decorated async function.

    Example::

        @trace_pipeline("rag_retrieval", attributes={"method": "hybrid"})
        async def retrieve(video_id: str, query: str) -> list[str]:
            ...
    """
    attributes = dict(attributes) if attributes else {}

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = _get_tracer()
            span = tracer.start_span(stage)
            if attributes:
                span.set_attributes(attributes)

            try:
                result = await func(*args, **kwargs)
                _ensure_status_codes()
                span.set_status(_make_status(_STATUS_OK))
                span.set_attribute("success", True)
                if capture_return:
                    span.set_attribute("result", str(result))
                return result
            except Exception as exc:
                _ensure_status_codes()
                span.set_status(_make_status(_STATUS_ERROR, str(exc)))
                span.set_attribute("success", False)
                span.set_attribute("error.type", type(exc).__name__)
                span.add_event("exception", {"exception.message": str(exc)})
                raise
            finally:
                span.end()

        return wrapper

    return decorator


# ── Context manager for async pipeline functions ────────────────────────


@asynccontextmanager
async def pipeline_span(
    stage: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
) -> AsyncIterator[TelemetryContext]:
    """Async context manager for tracing pipeline spans.

    This is an alternative to ``@trace_pipeline`` that gives you access
    to the ``TelemetryContext`` inside the async block, allowing you to
    set attributes or record events during execution.

    Args:
        stage: The pipeline stage name (used as the span name).
        attributes: Optional static attributes to attach to the span.

    Yields:
        A ``TelemetryContext`` instance bound to the active span.

    Example::

        async with pipeline_span("rag_retrieval", attributes={"method": "hybrid"}) as ctx:
            chunks = await retrieve(video_id, query)
            ctx.set_attribute("chunk_count", len(chunks))
            ctx.set_status_ok()
    """
    ctx = TelemetryContext(stage, **(attributes or {}))
    await ctx.__aenter__()
    try:
        yield ctx
        ctx.set_status_ok()
    except BaseException as exc:
        ctx.set_status_error(str(exc))
        raise
    finally:
        ctx._span.end()


# ── Parent span from HTTP headers (W3C TraceContext) ────────────────────


def parent_span_from_headers(
    headers: Mapping[str, str],
    span_name: str = "api_request",
    *,
    initial_attributes: Optional[Mapping[str, Any]] = None,
) -> TelemetryContext:
    """Create a ``TelemetryContext`` linked to a parent span from incoming HTTP headers.

    Extracts W3C ``traceparent`` and optional ``tracestate`` headers to
    continue a remote trace.  If the headers are missing or invalid, a new
    root span is created (safe fallback).

    Args:
        headers: HTTP request headers (case-insensitive key lookup).
        span_name: Name for the new child span (default ``"api_request"``).
        initial_attributes: Optional attributes to set on the new span.

    Returns:
        A ``TelemetryContext`` whose span is a child of the extracted
        remote span (or a new root span if extraction fails).

    Example::

        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "rojo=00f067aa0ba902b7",
        }
        ctx = parent_span_from_headers(headers, "rag_lookup",
                                       initial_attributes={"method": "hybrid"})
        with ctx:
            ...
    """
    traceparent: Optional[str] = None
    tracestate: Optional[str] = None

    # Case-insensitive header lookup
    for k, v in headers.items():
        lower_k = k.lower().strip()
        if lower_k == "traceparent":
            traceparent = v.strip()
        elif lower_k == "tracestate":
            tracestate = v.strip()

    tracer = _get_tracer()

    ctx: TelemetryContext
    span: Any

    if traceparent and _is_opentelemetry_available():
        try:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            carrier: Dict[str, str] = {}
            if traceparent:
                carrier["traceparent"] = traceparent
            if tracestate:
                carrier["tracestate"] = tracestate

            propagator = TraceContextTextMapPropagator()
            parent_context = propagator.extract(carrier=carrier)

            span = tracer.start_span(span_name, context=parent_context)
            ctx = TelemetryContext.__new__(TelemetryContext)
            ctx._name = span_name
            ctx._initial_attributes = dict(initial_attributes) if initial_attributes else {}
            ctx._span = span
            ctx._tracer = tracer
            ctx._exited = False

            if initial_attributes:
                span.set_attributes(initial_attributes)

            return ctx
        except Exception as exc:
            logger.debug("Failed to extract trace context from headers: %s", exc)

    # Fallback: create a new root span
    ctx = TelemetryContext(span_name, **(initial_attributes or {}))
    # Manually enter so the span is started
    tracer_actual = _get_tracer()
    ctx._tracer = tracer_actual
    ctx._span = tracer_actual.start_span(span_name)
    if initial_attributes:
        ctx._span.set_attributes(initial_attributes)
    ctx._exited = False
    return ctx


# ── Get the current trace ID ────────────────────────────────────────────


def get_trace_id() -> str:
    """Return the current trace ID as a hex string.

    If there is an active span in the current context, its trace ID is
    returned.  Otherwise a random UUID (dash-free hex string suitable for
    log correlation) is generated.

    Returns:
        A 32-character hex string representing the trace ID.

    Example::

        trace_id = get_trace_id()
        logger.info("Processing request", trace_id=trace_id)
    """
    if not _initialised:
        _ensure_telemetry()

    try:
        from opentelemetry import trace

        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
        if span_context and span_context.trace_id != 0:
            # Format as 32-char zero-padded hex
            return format(span_context.trace_id, _TRACE_ID_FORMAT)
    except (ImportError, RuntimeError, AttributeError):
        pass

    # Generate a trace-id-like UUID (32 hex chars, no dashes)
    return uuid.uuid4().hex


# ── Internal helpers ────────────────────────────────────────────────────


def _is_opentelemetry_available() -> bool:
    """Return ``True`` if OpenTelemetry packages were successfully loaded."""
    return enabled and _initialised


# ── Minimal re-export of the tracer for advanced use ────────────────────


def get_tracer() -> Any:
    """Return the module-level OTel tracer, initialising if needed.

    This is an escape hatch for advanced usage; most code should use
    ``TelemetryContext`` or ``@trace_pipeline`` instead.
    """
    return _get_tracer()


def force_flush() -> None:
    """Force flush any pending spans to the exporter.

    Safe to call when OpenTelemetry is not installed (no-op).
    """
    if _initialised and _tracer_provider is not None:
        try:
            _tracer_provider.force_flush()
        except Exception as exc:
            logger.warning("Failed to flush spans: %s", exc)
