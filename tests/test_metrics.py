"""
Tests for the Prometheus metrics module (v0.28.0).

Coverage targets:
- Import and initialisation
- Counter increments and label propagation
- Histogram observations
- No-op fallback when prometheus_client absent
- metrics_endpoint() function
- Config field defaults and env var override
- Integration: /metrics endpoint on health app
"""

import os
import sys
import time
from unittest.mock import patch

import pytest

from video_analysis.config import Config

# ── Helpers ──────────────────────────────────────────────────────────────


def _reimport_metrics():
    """Force re-import of video_analysis.metrics to reset state."""
    for mod in list(sys.modules.keys()):
        if "video_analysis.metrics" in mod:
            del sys.modules[mod]
    from video_analysis import metrics as m

    return m


# ── Tests ────────────────────────────────────────────────────────────────


class TestMetricsInit:
    """Initialisation and lazy-population."""

    def test_import(self):
        """Module imports cleanly."""
        from video_analysis import metrics  # noqa: F811

        assert metrics.__doc__ is not None

    def test_lazy_init(self):
        """Metrics are None until _ensure_metrics() is called."""
        m = _reimport_metrics()
        assert m.pipeline_runs_total is None
        assert m.pipeline_duration_seconds is None
        assert m.gpu_memory_bytes is None

        # Calling a convenience function triggers init
        m.increment_pipeline_run(mode="test", success=True)
        assert m.pipeline_runs_total is not None

    def test_increment_pipeline_run(self):
        """Counters increment correctly."""
        m = _reimport_metrics()
        m.increment_pipeline_run(mode="video_full", success=True)
        assert m.pipeline_runs_total.labels(mode="video_full")._value.get() == 1.0
        assert (
            m.pipeline_runs_success_total.labels(mode="video_full")._value.get() == 1.0
        )
        assert (
            m.pipeline_runs_failure_total.labels(mode="video_full")._value.get() == 0.0
        )

    def test_increment_pipeline_run_failure(self):
        """Failed runs increment failure counter."""
        m = _reimport_metrics()
        m.increment_pipeline_run(mode="audio_only", success=False)
        assert (
            m.pipeline_runs_failure_total.labels(mode="audio_only")._value.get() == 1.0
        )

    def test_observe_pipeline_duration(self):
        """Duration histograms record observations."""
        m = _reimport_metrics()
        m.observe_pipeline_duration(42.5, mode="video_full")
        # Histogram doesn't expose simple ._value; verify no exception

    def test_increment_question(self):
        """Question counter increments."""
        m = _reimport_metrics()
        m.increment_question(method="agentic")
        assert m.questions_answered_total.labels(method="agentic")._value.get() == 1.0

    def test_update_gpu_memory(self):
        """update_gpu_memory runs without error."""
        m = _reimport_metrics()
        # Should not raise even without GPU
        m.update_gpu_memory()

    def test_update_chroma_collection_size(self):
        """Chroma collection gauge updates."""
        m = _reimport_metrics()
        m.update_chroma_collection_size(42)
        assert m.chroma_collection_size._value.get() == 42.0
        # Negative should clamp to 0
        m.update_chroma_collection_size(-5)
        assert m.chroma_collection_size._value.get() == 0.0

    def test_metrics_endpoint(self):
        """metrics_endpoint() returns prometheus text."""
        m = _reimport_metrics()
        m.increment_pipeline_run(mode="video_full", success=True)
        text = m.metrics_endpoint()
        assert "va_pipeline_runs_total" in text
        assert "TYPE" in text
        assert text.startswith("#") or "va_" in text

    def test_metrics_endpoint_empty(self):
        """Empty metrics still produce valid output."""
        m = _reimport_metrics()
        text = m.metrics_endpoint()
        # Should have at least HELP and TYPE lines
        assert "# HELP" in text


class TestConfig:
    """Config integration for Prometheus."""

    def test_config_defaults(self):
        """Prometheus defaults in Config."""
        cfg = Config()
        assert cfg.prometheus_enabled is True
        assert cfg.prometheus_metrics_prefix == "va_"

    def test_config_env_override(self):
        """PROMETHEUS_ENABLED=false disables metrics."""
        with patch.dict(os.environ, {"PROMETHEUS_ENABLED": "false"}):
            cfg = Config()
            assert cfg.prometheus_enabled is False


class TestHealthEndpoint:
    """Integration with FastAPI health app."""

    def test_metrics_route_registered(self):
        """/metrics route exists when prometheus enabled."""
        from ui.health import create_health_app

        cfg = Config()
        app = create_health_app(cfg)
        routes = {r.path for r in app.routes}
        assert "/metrics" in routes

    def test_metrics_route_disabled(self):
        """/metrics route not registered when prometheus disabled."""
        with patch.dict(os.environ, {"PROMETHEUS_ENABLED": "false"}):
            from ui.health import create_health_app

            cfg = Config()
            app = create_health_app(cfg)
            routes = {r.path for r in app.routes}
            assert "/metrics" not in routes

    def test_metrics_returns_valid_text(self):
        """GET /metrics returns valid prometheus text (via direct endpoint call)."""
        from video_analysis.metrics import metrics_endpoint, _ensure_metrics

        # Force re-init in a clean subprocess to avoid global registry pollution
        import subprocess, sys

        code = """
from video_analysis.metrics import metrics_endpoint, _ensure_metrics
_ensure_metrics()
from video_analysis.metrics import increment_pipeline_run
increment_pipeline_run(mode="test_fn", success=True)
text = metrics_endpoint()
print("OK" if "va_pipeline_runs_total" in text else "NO_METRIC")
print("HAS_HELP" if "# HELP" in text else "NO_HELP")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "OK" in result.stdout
        assert "HAS_HELP" in result.stdout

    def test_metrics_route_returns_text(self):
        """Actual FastAPI /metrics endpoint returns valid prometheus text (subprocess)."""
        import subprocess, sys

        code = """
from ui.health import create_health_app
from video_analysis.config import Config
from prometheus_client import REGISTRY
import uvicorn, sys, threading, time

# Remove any pre-existing va_ metrics registered from prior test imports
for c in list(REGISTRY._collector_to_names.keys()):
    if hasattr(c, '_name') and 'va_' in (c._name or ''):
        try:
            REGISTRY.unregister(c)
        except (KeyError, ValueError):
            pass

cfg = Config()
app = create_health_app(cfg)

def run():
    uvicorn.run(app, host='127.0.0.1', port=18999, log_level='error')

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(1.5)

import urllib.request
resp = urllib.request.urlopen('http://127.0.0.1:18999/metrics')
data = resp.read().decode()
print(f"STATUS={resp.status}")
print(f"HAS_HELP={'# HELP' in data}")
print(f"HAS_METRIC={'va_pipeline_runs_total' in data}")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr[:300]}")
        assert result.returncode == 0, f"Process failed: {result.stderr}"
        assert "STATUS=200" in result.stdout
        assert "HAS_HELP=True" in result.stdout
        assert "HAS_METRIC=True" in result.stdout


class TestNoopFallback:
    """Behaviour when prometheus_client is not installed."""

    def test_noop_import(self):
        """Module handles missing prometheus_client gracefully."""
        m = _reimport_metrics()
        with patch.dict(sys.modules, {"prometheus_client": None}):
            # Force re-init with no prometheus_client
            for mod in list(sys.modules.keys()):
                if "video_analysis.metrics" in mod:
                    del sys.modules[mod]
            from video_analysis import metrics as m2

            # All convenience functions should work without error
            m2.increment_pipeline_run(mode="test", success=True)
            m2.observe_pipeline_duration(10.0, mode="test")
            m2.increment_question(method="test")
            m2.update_gpu_memory()
            m2.update_chroma_collection_size(10)
            text = m2.metrics_endpoint()
            assert "not installed" in text


class TestVersion:
    """Version bump verification."""

    def test_version(self):
        from video_analysis import __version__

        assert __version__ == "0.36.0"

    def test_pyproject_version(self):
        import tomllib

        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == "0.36.0"
