"""
Tests for the Webhook Notification System (v0.59.0).

Tests the WebhookDispatcher, WebhookConfig, and singleton get/set/reset
functions in video_analysis/webhook.py.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from video_analysis.webhook import (
    EVENT_EVAL_COMPLETE,
    EVENT_HEALTH_ALERT,
    EVENT_HEALTH_CRITICAL,
    EVENT_PIPELINE_COMPLETE,
    WebhookConfig,
    WebhookDispatcher,
    get_webhook_dispatcher,
    reset_webhook_dispatcher,
    set_webhook_dispatcher,
)

# ═══════════════════════════════════════════════════════════════════════
# WebhookConfig tests
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookConfig:
    def test_default_headers(self):
        cfg = WebhookConfig(url="http://example.com/hook")
        assert cfg.url == "http://example.com/hook"
        assert cfg.timeout == 5.0
        assert cfg.max_retries == 1
        assert cfg.headers["Content-Type"] == "application/json"

    def test_https_url_accepted(self):
        cfg = WebhookConfig(url="https://hooks.example.com/callback")
        assert cfg.url == "https://hooks.example.com/callback"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid webhook URL"):
            WebhookConfig(url="ftp://bad-scheme.com/hook")

    def test_custom_timeout_and_retries(self):
        cfg = WebhookConfig(url="http://example.com/hook", timeout=10.0, max_retries=3)
        assert cfg.timeout == 10.0
        assert cfg.max_retries == 3


# ═══════════════════════════════════════════════════════════════════════
# WebhookDispatcher tests
# ═══════════════════════════════════════════════════════════════════════


class TestWebhookDispatcherInit:
    def test_empty_init(self):
        d = WebhookDispatcher()
        assert not d.enabled
        assert d.urls == []

    def test_init_with_urls(self):
        d = WebhookDispatcher(urls=["http://a.com/h", "http://b.com/h"])
        assert d.enabled
        assert len(d.urls) == 2
        assert "http://a.com/h" in d.urls

    def test_duplicate_urls_skipped(self):
        d = WebhookDispatcher(urls=["http://a.com/h", "http://a.com/h"])
        assert len(d.urls) == 1


class TestWebhookDispatcherUrlManagement:
    def test_add_url(self):
        d = WebhookDispatcher()
        d.add_url("http://example.com/hook")
        assert d.enabled
        assert "http://example.com/hook" in d.urls

    def test_add_duplicate_skipped(self):
        d = WebhookDispatcher(urls=["http://a.com/h"])
        d.add_url("http://a.com/h")
        assert len(d.urls) == 1

    def test_remove_url(self):
        d = WebhookDispatcher(urls=["http://a.com/h", "http://b.com/h"])
        assert d.remove_url("http://a.com/h")
        assert len(d.urls) == 1
        assert "http://a.com/h" not in d.urls

    def test_remove_nonexistent(self):
        d = WebhookDispatcher()
        assert not d.remove_url("http://nonexistent.com/hook")

    def test_clear(self):
        d = WebhookDispatcher(urls=["http://a.com/h", "http://b.com/h"])
        d.clear()
        assert not d.enabled
        assert d.urls == []


class TestWebhookDispatcherFire:
    def test_fire_noop_when_disabled(self):
        """fire() is a no-op when no URLs configured."""
        d = WebhookDispatcher()
        # Should not raise
        d.fire("test.event", {"key": "value"})

    def test_fire_blocking_returns_error_empty_urls(self):
        d = WebhookDispatcher()
        errors = d.fire_blocking("test.event", {})
        assert errors == ["no webhooks configured"]

    def test_fire_blocking_delivers_payload(self):
        """Use a local HTTP server to verify actual delivery."""
        received: List[bytes] = []

        class TestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                received.append(body)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass  # suppress HTTP server logs

        server = HTTPServer(("127.0.0.1", 0), TestHandler)
        port = server.server_port
        url = f"http://127.0.0.1:{port}/webhook"

        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        d = WebhookDispatcher(urls=[url])
        errors = d.fire_blocking("pipeline.complete", {"video_id": "v001", "filename": "test.mp4"})

        t.join(timeout=3)
        server.server_close()

        assert errors == []
        assert len(received) == 1
        payload = json.loads(received[0])
        assert payload["event"] == "pipeline.complete"
        assert payload["payload"]["video_id"] == "v001"

    def test_fire_blocking_returns_errors_on_failure(self):
        d = WebhookDispatcher(urls=["http://127.0.0.1:1/nonexistent"])
        errors = d.fire_blocking("test.event", {"key": "val"})
        assert len(errors) == 1
        assert "webhook" in errors[0]

    def test_fire_blocking_multiple_urls(self):
        received: List[str] = []

        class TestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        # Start two servers
        server1 = HTTPServer(("127.0.0.1", 0), TestHandler)
        server2 = HTTPServer(("127.0.0.1", 0), TestHandler)
        port1 = server1.server_port
        port2 = server2.server_port

        threads = []
        for s in (server1, server2):
            t = threading.Thread(target=s.handle_request, daemon=True)
            t.start()
            threads.append(t)

        d = WebhookDispatcher(
            urls=[
                f"http://127.0.0.1:{port1}/hook1",
                f"http://127.0.0.1:{port2}/hook2",
            ]
        )
        errors = d.fire_blocking("test.event", {"n": 1})

        for t in threads:
            t.join(timeout=3)
        server1.server_close()
        server2.server_close()

        assert errors == []
        assert len(received) == 2
        assert "/hook1" in received
        assert "/hook2" in received

    def test_fire_blocking_correct_headers(self):
        """Verify Content-Type and User-Agent headers are sent."""
        header_store: Dict[str, str] = {}

        class TestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                header_store["content_type"] = self.headers.get("Content-Type", "")
                header_store["user_agent"] = self.headers.get("User-Agent", "")
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), TestHandler)
        port = server.server_port
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        d = WebhookDispatcher(urls=[f"http://127.0.0.1:{port}/hook"])
        d.fire_blocking("test.event", {})

        t.join(timeout=3)
        server.server_close()

        assert header_store.get("content_type") == "application/json"
        assert "VideoAnalysis" in header_store.get("user_agent", "")


class TestWebhookDispatcherPayloadFormat:
    def test_payload_contains_event_timestamp_payload(self):
        """Verify JSON payload structure."""
        received: List[Dict[str, Any]] = []

        class TestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                received.append(body)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), TestHandler)
        port = server.server_port
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        d = WebhookDispatcher(urls=[f"http://127.0.0.1:{port}/hook"])
        d.fire_blocking("pipeline.complete", {"video_id": "abc"})

        t.join(timeout=3)
        server.server_close()

        assert len(received) == 1
        payload = received[0]
        assert "event" in payload
        assert "timestamp" in payload
        assert "payload" in payload
        assert payload["event"] == "pipeline.complete"
        assert payload["payload"]["video_id"] == "abc"


class TestWebhookDispatcherThreadSafety:
    def test_concurrent_add_remove(self):
        """Add and remove URLs from multiple threads."""
        d = WebhookDispatcher()

        def adder(url: str):
            for _ in range(50):
                d.add_url(url)

        def remover(url: str):
            for _ in range(30):
                d.remove_url(url)

        threads = []
        for i in range(5):
            t = threading.Thread(target=adder, args=(f"http://hook{i}.com/h",))
            threads.append(t)
        for _ in range(3):
            t = threading.Thread(target=remover, args=("http://hook0.com/h",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should still have valid state (no exceptions)
        assert isinstance(d.urls, list)


# ═══════════════════════════════════════════════════════════════════════
# Singleton tests
# ═══════════════════════════════════════════════════════════════════════


class TestSingleton:
    def setup_method(self):
        reset_webhook_dispatcher()

    def test_get_returns_singleton(self):
        d1 = get_webhook_dispatcher()
        d2 = get_webhook_dispatcher()
        assert d1 is d2

    def test_get_with_config_sets_urls(self):
        """Initialise from a config-like object."""
        config = MagicMock()
        config.webhook_urls = ["http://a.com/h", "http://b.com/h"]
        config.webhook_timeout = 10.0

        d = get_webhook_dispatcher(config)
        assert d.enabled
        assert len(d.urls) == 2

    def test_get_without_config_returns_empty_dispatcher(self):
        d = get_webhook_dispatcher()
        assert not d.enabled
        assert d.urls == []

    def test_set_replaces_singleton(self):
        custom = WebhookDispatcher(urls=["http://custom.com/hook"])
        set_webhook_dispatcher(custom)

        d = get_webhook_dispatcher()
        assert d is custom
        assert "http://custom.com/hook" in d.urls

    def test_reset_clears_singleton(self):
        d1 = get_webhook_dispatcher()
        reset_webhook_dispatcher()

        d2 = get_webhook_dispatcher()
        assert d2 is not d1  # new instance

    def test_config_init_once_only(self):
        """Config is only used on first init — subsequent calls return existing singleton."""
        d1 = get_webhook_dispatcher()  # no config → empty

        config = MagicMock()
        config.webhook_urls = ["http://new.com/hook"]
        d2 = get_webhook_dispatcher(config)  # should NOT re-init
        assert d2 is d1


# ═══════════════════════════════════════════════════════════════════════
# Event constants
# ═══════════════════════════════════════════════════════════════════════


class TestEventConstants:
    def test_event_constants(self):
        assert EVENT_PIPELINE_COMPLETE == "pipeline.complete"
        assert EVENT_EVAL_COMPLETE == "eval.complete"
        assert EVENT_HEALTH_ALERT == "health.alert"
        assert EVENT_HEALTH_CRITICAL == "health.critical"
