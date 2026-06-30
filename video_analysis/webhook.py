"""
Webhook Notification System — event-driven HTTP callbacks for pipeline events.

Provides a configurable webhook dispatcher that fires HTTP POST requests
to registered URLs when specific pipeline, evaluation, or health events occur:

    WebhookEvent:
        - pipeline.complete      — video processing finished
        - pipeline.error         — video processing failed
        - eval.complete          — evaluation run finished
        - health.alert           — health monitor created a new alert
        - health.critical        — critical-severity health alert

Usage:
    from video_analysis.webhook import webhook_dispatcher

    # Fire an event (non-blocking — runs in a thread pool)
    webhook_dispatcher.fire("pipeline.complete", {
        "video_id": "abc123",
        "filename": "my_video.mp4",
        "duration": 320.5,
    })

Design:
    - Zero external dependencies — uses urllib.request (stdlib)
    - Configurable via WebhookConfig or Config.webhook_* fields
    - Thread pool executor for non-blocking delivery
    - Per-URL timeout and retry (1 retry, 5s timeout)
    - Logs failures without crashing the caller
    - Supports multiple webhook URLs (comma-separated)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: float = 5.0  # seconds per POST
MAX_RETRIES: int = 1
DEFAULT_MAX_WORKERS: int = 4

# ---------------------------------------------------------------------------
# Webhook event type constants
# ---------------------------------------------------------------------------

EVENT_PIPELINE_COMPLETE = "pipeline.complete"
EVENT_PIPELINE_ERROR = "pipeline.error"
EVENT_EVAL_COMPLETE = "eval.complete"
EVENT_HEALTH_ALERT = "health.alert"
EVENT_HEALTH_CRITICAL = "health.critical"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WebhookConfig:
    """Per-webhook URL configuration."""

    url: str
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = MAX_RETRIES
    headers: Dict[str, str] = field(
        default_factory=lambda: {
            "Content-Type": "application/json",
            "User-Agent": "VideoAnalysis-Webhook/0.59.0",
        }
    )

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid webhook URL: {self.url}")


# ---------------------------------------------------------------------------
# Webhook dispatcher
# ---------------------------------------------------------------------------


class WebhookDispatcher:
    """Thread-safe webhook dispatcher with configurable URL list.

    Fires events to all registered URLs asynchronously via a thread pool.
    Fire-and-forget: failures are logged but never raised to the caller.
    """

    def __init__(
        self,
        urls: Optional[List[str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        self._configs: List[WebhookConfig] = []
        self._timeout = timeout
        self._lock = threading.Lock()
        self._executor: Optional[threading.Thread] = None

        if urls:
            for u in urls:
                self.add_url(u)

    # ------------------------------------------------------------------
    # URL management
    # ------------------------------------------------------------------

    def add_url(self, url: str) -> None:
        """Register a webhook URL."""
        cfg = WebhookConfig(url=url, timeout=self._timeout)
        with self._lock:
            # Avoid duplicates
            if not any(c.url == url for c in self._configs):
                self._configs.append(cfg)

    def remove_url(self, url: str) -> bool:
        """Unregister a webhook URL. Returns True if found and removed."""
        with self._lock:
            for i, c in enumerate(self._configs):
                if c.url == url:
                    self._configs.pop(i)
                    return True
        return False

    @property
    def urls(self) -> List[str]:
        """Return a copy of registered URLs."""
        with self._lock:
            return [c.url for c in self._configs]

    @property
    def enabled(self) -> bool:
        """True if at least one webhook URL is registered."""
        with self._lock:
            return len(self._configs) > 0

    # ------------------------------------------------------------------
    # Event firing
    # ------------------------------------------------------------------

    def fire(self, event: str, payload: Dict[str, Any]) -> None:
        """Fire a webhook event to all registered URLs (non-blocking).

        Runs HTTP POST in a daemon thread so the caller is never blocked.
        """
        if not self.enabled:
            return

        body = json.dumps(
            {
                "event": event,
                "timestamp": time.time(),
                "payload": payload,
            },
            default=str,
        )

        t = threading.Thread(
            target=self._deliver,
            args=(body,),
            daemon=True,
        )
        t.start()

    def fire_blocking(self, event: str, payload: Dict[str, Any]) -> List[str]:
        """Fire a webhook event synchronously.

        Returns a list of error messages (empty = all succeeded).
        Useful for testing.
        """
        if not self.enabled:
            return ["no webhooks configured"]

        body = json.dumps(
            {
                "event": event,
                "timestamp": time.time(),
                "payload": payload,
            },
            default=str,
        )

        return self._deliver(body)

    def _deliver(self, body: str) -> List[str]:
        """Deliver a payload to all registered URLs. Returns error list."""
        errors: List[str] = []
        with self._lock:
            configs = list(self._configs)

        for cfg in configs:
            err = self._post(cfg, body)
            if err:
                errors.append(err)
        return errors

    @staticmethod
    def _post(cfg: WebhookConfig, body: str) -> Optional[str]:
        """Send one POST request. Returns error string or None on success."""
        for attempt in range(cfg.max_retries + 1):
            try:
                req = urllib.request.Request(
                    cfg.url,
                    data=body.encode("utf-8"),
                    headers=cfg.headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                    status = resp.status
                    if 200 <= status < 300:
                        return None
                    # Non-2xx — retry unless it's a client error 4xx
                    if 400 <= status < 500:
                        return f"webhook {cfg.url}: HTTP {status} (client error, not retried)"
            except urllib.error.HTTPError as e:
                if attempt < cfg.max_retries and e.code >= 500:
                    continue
                return f"webhook {cfg.url}: HTTP {e.code} after {attempt + 1} attempt(s)"
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < cfg.max_retries:
                    continue
                return f"webhook {cfg.url}: {e} after {attempt + 1} attempt(s)"
        return None  # Should not reach here

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all registered webhook URLs."""
        with self._lock:
            self._configs.clear()

    def __repr__(self) -> str:
        return f"WebhookDispatcher(urls={self.urls})"


# ---------------------------------------------------------------------------
# Singleton dispatcher (lazy init)
# ---------------------------------------------------------------------------

_webhook_dispatcher: Optional[WebhookDispatcher] = None
_webhook_lock = threading.Lock()


def get_webhook_dispatcher(
    config=None,
) -> WebhookDispatcher:
    """Get or create the global webhook dispatcher singleton.

    If *config* is provided and the singleton doesn't exist yet, it's
    initialised from the config's webhook fields.
    """
    global _webhook_dispatcher
    if _webhook_dispatcher is None:
        with _webhook_lock:
            if _webhook_dispatcher is None:
                urls = []
                timeout = DEFAULT_TIMEOUT
                if config is not None:
                    if hasattr(config, "webhook_urls") and config.webhook_urls:
                        urls = config.webhook_urls
                    if hasattr(config, "webhook_timeout"):
                        timeout = config.webhook_timeout
                _webhook_dispatcher = WebhookDispatcher(urls=urls, timeout=timeout)
    return _webhook_dispatcher


def set_webhook_dispatcher(dispatcher: WebhookDispatcher) -> None:
    """Replace the global dispatcher (for testing)."""
    global _webhook_dispatcher
    _webhook_dispatcher = dispatcher


def reset_webhook_dispatcher() -> None:
    """Reset the global dispatcher to None (for testing)."""
    global _webhook_dispatcher
    _webhook_dispatcher = None


# Convenience alias
webhook_dispatcher = get_webhook_dispatcher
