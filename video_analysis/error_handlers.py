"""Structured error handling for the FastAPI REST API.

Provides Pydantic models, exception classes, and a registration function
for consistent JSON error responses across all API endpoints.

Every error response includes:
- ``detail`` — human-readable description
- ``error_code`` — machine-readable error identifier
- ``status_code`` — HTTP status code (mirrored from the response)
- ``timestamp`` — ISO-8601 formatted UTC timestamp
- ``path`` — request URL path that triggered the error

Usage::

    from fastapi import FastAPI
    from video_analysis.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)

    # Then raise StandardHTTPError anywhere in your route handlers:
    from video_analysis.error_handlers import StandardHTTPError
    raise StandardHTTPError(status_code=404, detail="Video not found")
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════════


class ErrorDetail(BaseModel):
    """Structured error response body.

    Returned as the body of every error response from the API.  The top-level
    ``detail`` field is compatible with FastAPI's default error schema; extra
    fields (``error_code``, ``timestamp``, etc.) provide richer context for
    clients and debugging.
    """

    detail: str = Field(..., description="Human-readable error description")
    error_code: str = Field(..., description="Machine-readable error identifier")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    path: str = Field(..., description="Request URL path that triggered the error")


class ValidationErrorItem(BaseModel):
    """A single field validation error from Pydantic."""

    loc: List[str] = Field(default_factory=list, description="Location of the error (field path)")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type identifier")


class ValidationErrorResponse(BaseModel):
    """422 response body with per-field validation details."""

    detail: str = Field("Request validation failed", description="Top-level summary")
    error_code: str = Field("VALIDATION_ERROR", description="Error identifier")
    status_code: int = Field(422, description="HTTP status code")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    path: str = Field(..., description="Request URL path")
    errors: List[ValidationErrorItem] = Field(
        default_factory=list, description="Per-field validation errors"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Exception classes
# ═══════════════════════════════════════════════════════════════════════════


class StandardHTTPError(Exception):
    """Application-level HTTP error with structured metadata.

    Raise this inside route handlers to produce a consistent JSON error
    response via ``register_error_handlers()``.

    Args:
        status_code: HTTP status code (4xx or 5xx).
        detail: Human-readable error description.
        error_code: Machine-readable identifier (defaults to ``"HTTP_{code}"``).

    Example::

        raise StandardHTTPError(
            status_code=404,
            detail="The requested video was not found",
            error_code="VIDEO_NOT_FOUND",
        )
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code or f"HTTP_{status_code}"
        super().__init__(self.detail)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 formatted string."""
    return datetime.now(timezone.utc).isoformat()


def _build_error_detail(
    detail: str,
    error_code: str,
    status_code: int,
    request: Request,
) -> Dict[str, Any]:
    """Build a serialisable dict representing an ``ErrorDetail``.

    Args:
        detail: Human-readable description.
        error_code: Machine-readable error identifier.
        status_code: HTTP status code.
        request: The incoming FastAPI request (used for ``path``).

    Returns:
        Dict matching the ``ErrorDetail`` schema.
    """
    return {
        "detail": detail,
        "error_code": error_code,
        "status_code": status_code,
        "timestamp": _now_iso(),
        "path": str(request.url.path),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Error handler implementations
# ═══════════════════════════════════════════════════════════════════════════


async def _standard_http_error_handler(
    request: Request,
    exc: StandardHTTPError,
) -> JSONResponse:
    """Handle ``StandardHTTPError`` -> structured JSON response."""
    content = _build_error_detail(
        detail=exc.detail,
        error_code=exc.error_code,
        status_code=exc.status_code,
        request=request,
    )
    return JSONResponse(status_code=exc.status_code, content=content)


async def _http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Handle FastAPI's ``HTTPException`` -> structured JSON response."""
    content = _build_error_detail(
        detail=str(exc.detail),
        error_code=f"HTTP_{exc.status_code}",
        status_code=exc.status_code,
        request=request,
    )
    return JSONResponse(status_code=exc.status_code, content=content)


async def _request_validation_handler(
    request: Request,
    exc: FastAPIRequestValidationError,
) -> JSONResponse:
    """Handle FastAPI ``RequestValidationError`` -> 422 with field-level errors."""
    now = _now_iso()
    path = str(request.url.path)

    errors: List[Dict[str, Any]] = []
    for e in exc.errors():
        errors.append(
            {
                "loc": list(e.get("loc", [])),
                "msg": e.get("msg", ""),
                "type": e.get("type", ""),
            }
        )

    content: Dict[str, Any] = {
        "detail": "Request validation failed",
        "error_code": "VALIDATION_ERROR",
        "status_code": 422,
        "timestamp": now,
        "path": path,
        "errors": errors,
    }
    return JSONResponse(status_code=422, content=content)


# ═══════════════════════════════════════════════════════════════════════════
# Catch-all middleware for unhandled exceptions
# ═══════════════════════════════════════════════════════════════════════════


class _CatchAllMiddleware(BaseHTTPMiddleware):
    """Middleware that catches any unhandled exception and returns a 500 JSON
    response.

    This runs inside the Starlette middleware stack, *below* the
    ``ServerErrorMiddleware``, so it intercepts exceptions before
    ``ServerErrorMiddleware`` re-raises them.  This is necessary because
    ``app.add_exception_handler(Exception, ...)`` does not prevent
    ``ServerErrorMiddleware`` from re-raising the exception to the ASGI
    server / test client.

    The original exception and traceback are logged server-side; only a
    generic message is returned to the client.
    """

    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        try:
            response = await call_next(request)
            return response  # type: ignore[return-value]
        except Exception as exc:
            logger.error(
                "Unhandled exception processing %s %s\n%s",
                request.method,
                request.url.path,
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            content = _build_error_detail(
                detail="Internal server error",
                error_code="INTERNAL_ERROR",
                status_code=500,
                request=request,
            )
            return JSONResponse(status_code=500, content=content)


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════


def register_error_handlers(app: FastAPI) -> None:
    """Register structured error handlers on a FastAPI application.

    Registers handlers for the following exception types:

    - ``StandardHTTPError`` -> configurable 4xx/5xx with ``ErrorDetail``
    - ``HTTPException`` (FastAPI) -> any 4xx/5xx with ``ErrorDetail``
    - ``RequestValidationError`` (FastAPI / Pydantic) -> 422 with per-field
      error details
    - Any other unhandled ``Exception`` -> 500 with a sanitised generic
      message (via middleware)

    Args:
        app: The FastAPI application instance to attach handlers to.

    Example::

        from fastapi import FastAPI
        from video_analysis.error_handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)
    """
    # Register exception handlers for well-known exception types.
    app.add_exception_handler(StandardHTTPError, _standard_http_error_handler)
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(FastAPIRequestValidationError, _request_validation_handler)

    # Catch-all for any unhandled exception.  Uses middleware instead of
    # add_exception_handler(Exception, ...) because Starlette's
    # ServerErrorMiddleware always re-raises the exception even after the
    # handler runs, which breaks TestClient and other ASGI consumers.
    app.add_middleware(_CatchAllMiddleware)

    logger.info("Registered structured error handlers on %s", app.title or "FastAPI app")
