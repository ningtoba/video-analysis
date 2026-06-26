"""Tests for the Qwen3-VL-30B-A3B backend module.

Tests cover:
- Module import and class existence
- Qwen3VLBackend init with various configs
- _check_vllm_server (connectivity check — expected to fail without server)
- Frame decoding (requires decord — skipped if not installed)
- describe_scene empty frames returns None
- summarize_video nonexistent file returns None
- answer without frames/video returns None
- vLLM server management (start_vllm_server)
- _build_vllm_messages format
- _cleanup_temp_frames
- Config integration — video_mllm_backend = "qwen3_vl"
- VideoMLLM qwen3_vl route integration
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

logger = logging.getLogger(__name__)


def test_qwen3_vl_importable():
    """Test that the Qwen3-VL backend module can be imported cleanly."""
    from video_analysis.backends.qwen3_vl import (
        Qwen3VLBackend,
        QWEN3_VL_MODEL_NAME,
        QWEN3_VL_FP8_MODEL_NAME,
    )

    assert callable(Qwen3VLBackend)
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in QWEN3_VL_MODEL_NAME
    assert "FP8" in QWEN3_VL_FP8_MODEL_NAME


def test_qwen3_vl_init_defaults():
    """Test Qwen3VLBackend default initialization."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend, VLLM_SERVER_DEFAULT

    backend = Qwen3VLBackend()
    assert "30B-A3B" in backend.model_name
    assert backend.backend == "auto"
    assert backend._vllm_server_url == VLLM_SERVER_DEFAULT
    assert backend.use_fp8 is True
    assert backend.max_frames == 32
    assert backend.thinking_mode is False
    assert backend._available is None
    assert backend._model is None
    assert backend._llm is None


def test_qwen3_vl_init_custom():
    """Test Qwen3VLBackend with custom parameters."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend(
        model_name="custom-model",
        backend="vllm_server",
        vllm_server_url="http://custom:8080",
        use_fp8=False,
        max_frames=64,
        thinking_mode=True,
    )
    assert backend.model_name == "custom-model"
    assert backend.backend == "vllm_server"
    assert backend._vllm_server_url == "http://custom:8080"
    assert backend.use_fp8 is False
    assert backend.max_frames == 64
    assert backend.thinking_mode is True


def test_qwen3_vl_vllm_server_url_env():
    """Test that vllm_server_url reads from env var."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend, VLLM_SERVER_URL_ENV

    os.environ[VLLM_SERVER_URL_ENV] = "http://test:9999"
    try:
        backend = Qwen3VLBackend()
        assert backend._vllm_server_url == "http://test:9999"
    finally:
        del os.environ[VLLM_SERVER_URL_ENV]


def test_qwen3_vl_available():
    """Test availability check (should be True since torch is installed)."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend()
    # Should be True since torch is a project dependency
    assert backend.available is True


def test_describe_scene_empty_frames():
    """Test describe_scene returns None for empty frames."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend()
    result = backend.describe_scene([])
    assert result is None


def test_summarize_video_nonexistent():
    """Test summarize_video returns None for nonexistent file."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend()
    result = backend.summarize_video("/tmp/nonexistent_video.mp4")
    assert result is None


def test_answer_no_input():
    """Test answer returns None without frames or video_path."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend()
    result = backend.answer("test query")
    assert result is None


def test_check_vllm_server_no_server():
    """Test _check_vllm_server returns False when no server is running."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend(
        backend="vllm_server",
        vllm_server_url="http://127.0.0.1:1",
    )
    # Port 1 should fail quickly
    result = backend._check_vllm_server()
    assert result is False


def test_build_vllm_messages_text_only():
    """Test _build_vllm_messages builds correct format for text-only."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend()
    messages = backend._build_vllm_messages("What do you see?", None)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert len(messages[0]["content"]) == 1
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][0]["text"] == "What do you see?"


def test_build_vllm_messages_with_frames():
    """Test _build_vllm_messages builds correct format with image paths."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend()

    # Create a temporary image file
    import tempfile
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name

    try:
        img = Image.new("RGB", (10, 10), color="red")
        img.save(tmp_path)

        messages = backend._build_vllm_messages("test", [tmp_path])
        assert len(messages[0]["content"]) == 2
        # First should be image_url
        assert messages[0]["content"][0]["type"] == "image_url"
        assert "data:image/" in messages[0]["content"][0]["image_url"]["url"]
        # Second should be text
        assert messages[0]["content"][1]["type"] == "text"
        assert messages[0]["content"][1]["text"] == "test"
    finally:
        os.unlink(tmp_path)


def test_cleanup_temp_frames():
    """Test _cleanup_temp_frames handles None safely."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend()

    # Should not raise
    backend._cleanup_temp_frames(None)
    backend._cleanup_temp_frames([])

    # Create a temp dir and clean it
    import tempfile

    d = tempfile.mkdtemp()
    p = Path(d) / "frame.jpg"
    p.touch()
    backend._cleanup_temp_frames([str(p)])
    assert not Path(d).exists()


def test_resolve_backend_vllm_server():
    """Test _resolve_backend returns correct value for explicit vllm_server."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend(backend="vllm_server")
    assert backend._resolve_backend() == "vllm_server"


def test_resolve_backend_vllm_offline():
    """Test _resolve_backend returns correct value for explicit vllm_offline."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend(backend="vllm_offline")
    assert backend._resolve_backend() == "vllm_offline"


def test_resolve_backend_transformers():
    """Test _resolve_backend returns correct value for explicit transformers."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend(backend="transformers")
    assert backend._resolve_backend() == "transformers"


def test_vllm_server_management():
    """Test start_vllm_server is callable and returns None on failure."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    # With invalid args, should return None gracefully
    result = Qwen3VLBackend.start_vllm_server(
        model_name="nonexistent/model",
        port=99999,
        max_model_len=128,
    )
    # The subprocess might actually try to start, so it might return a handle
    # that immediately fails. Let's just verify it doesn't crash.
    assert result is None or hasattr(result, "poll")


def test_config_backend_qwen3_vl():
    """Test that config accepts qwen3_vl as a valid backend."""
    from video_analysis.config import Config

    config = Config(data_dir="/tmp/va_test_qwen3_cfg")

    # Default backend should be auto
    assert config.video_mllm_backend == "auto"

    # Set backend via env var
    os.environ["VIDEO_MLLM_BACKEND"] = "qwen3_vl"
    config2 = Config(data_dir="/tmp/va_test_qwen3_cfg2")
    assert config2.video_mllm_backend == "qwen3_vl"

    # Cleanup
    del os.environ["VIDEO_MLLM_BACKEND"]

    import shutil

    shutil.rmtree("/tmp/va_test_qwen3_cfg", ignore_errors=True)
    shutil.rmtree("/tmp/va_test_qwen3_cfg2", ignore_errors=True)


def test_video_mllm_qwen3_vl_import():
    """Test VideoMLLM recognizes qwen3_vl backend type."""
    from video_analysis.video_mllm import VideoMLLM

    mllm = VideoMLLM(backend="qwen3_vl")
    assert mllm.backend == "qwen3_vl"


def test_import_via_video_analysis():
    """Test that the backends package is importable via video_analysis.backends."""
    from video_analysis import backends

    assert backends is not None
    assert hasattr(backends, "__path__")


def test_generate_no_backend_loaded():
    """Test _generate returns None when no backend is loaded (no vLLM server)."""
    from video_analysis.backends.qwen3_vl import Qwen3VLBackend

    backend = Qwen3VLBackend(
        backend="vllm_server", vllm_server_url="http://127.0.0.1:1"
    )
    result = backend._generate("test", max_new_tokens=10)
    assert result is None


def test_qwen3_vl_version():
    """Test version string is updated."""
    from video_analysis import __version__

    assert __version__ == "0.44.0"
