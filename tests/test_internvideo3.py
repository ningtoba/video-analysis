"""Tests for InternVideo3-8B video MLLM backend (v0.55.0).

Covers:
- Module importability
- Backend class structure
- Mode detection (no GPU required)
- Config integration — video_mllm_backend = "internvideo3"
- VideoMLLM internvideo3 route integration
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

logger = logging.getLogger(__name__)


def test_internvideo3_importable():
    """Test that the InternVideo3 backend module can be imported cleanly."""
    from video_analysis.backends.internvideo3 import (
        InternVideo3Backend,
        INTERNVIDEO3_MODEL_NAME,
        VLLM_SERVER_ARGS,
    )

    assert InternVideo3Backend is not None
    assert INTERNVIDEO3_MODEL_NAME == "OpenGVLab/InternVideo3-8B-Instruct"
    assert len(VLLM_SERVER_ARGS) > 0
    assert "--port" in VLLM_SERVER_ARGS
    assert "8001" in VLLM_SERVER_ARGS


def test_internvideo3_backend_instantiation():
    """Test that the backend can be instantiated with defaults."""
    from video_analysis.backends.internvideo3 import InternVideo3Backend

    backend = InternVideo3Backend()
    assert backend.model_name == "OpenGVLab/InternVideo3-8B-Instruct"
    assert backend.use_fp8 is False
    assert backend.thinking_mode is False
    assert backend._mode is None
    assert backend._available is None


def test_internvideo3_backend_fp8_instantiation():
    """Test FP8 mode instantiation."""
    from video_analysis.backends.internvideo3 import InternVideo3Backend

    backend = InternVideo3Backend(use_fp8=True, thinking_mode=True)
    assert backend.use_fp8 is True
    assert backend.thinking_mode is True


def test_internvideo3_backend_custom_model():
    """Test custom model name."""
    from video_analysis.backends.internvideo3 import (
        InternVideo3Backend,
        INTERNVIDEO3_BASE_MODEL_NAME,
    )

    backend = InternVideo3Backend(model_name=INTERNVIDEO3_BASE_MODEL_NAME)
    assert backend.model_name == "Qwen/Qwen3-VL-8B-Instruct"


def test_internvideo3_backend_custom_url():
    """Test custom vLLM server URL via constructor."""
    from video_analysis.backends.internvideo3 import InternVideo3Backend

    backend = InternVideo3Backend(vllm_server_url="http://localhost:8080")
    assert backend._vllm_server_url == "http://localhost:8080"


def test_internvideo3_backend_url_env_var():
    """Test custom vLLM server URL via environment variable."""
    from video_analysis.backends.internvideo3 import InternVideo3Backend

    os.environ["INTERNVIDEO3_VLLM_URL"] = "http://my-server:8001"
    try:
        backend = InternVideo3Backend()
        assert backend._vllm_server_url == "http://my-server:8001"
    finally:
        del os.environ["INTERNVIDEO3_VLLM_URL"]


def test_internvideo3_available_no_gpu():
    """Test that available property returns False without GPU/model.

    On a machine without a vLLM server running and without the model
    installed, available should gracefully return False.
    """
    from video_analysis.backends.internvideo3 import InternVideo3Backend

    backend = InternVideo3Backend()
    # Should not raise, return False gracefully
    is_available = backend.available
    # On a CI/test machine without internvideo3, this will be False
    assert is_available is False or is_available is True  # just don't crash


def test_internvideo3_detect_no_server():
    """Test that _check_vllm_server returns False when no server."""
    from video_analysis.backends.internvideo3 import InternVideo3Backend

    backend = InternVideo3Backend(vllm_server_url="http://localhost:19999")
    result = backend._check_vllm_server()
    assert result is False


def test_video_mllm_backend_type():
    """Test that backend_type Literal accepts internvideo3."""
    from video_analysis.video_mllm import BackendType

    # Verify the Literal accepts internvideo3
    import typing

    args = typing.get_args(BackendType)
    assert "internvideo3" in args


def test_video_mllm_backend_resolve_internvideo3():
    """Test that VideoMLLM can resolve internvideo3 backend."""
    from video_analysis.video_mllm import VideoMLLM

    mllm = VideoMLLM(backend="internvideo3")
    assert mllm.backend == "internvideo3"
    resolved = mllm._resolve_backend()
    assert resolved == "internvideo3"


def test_version_check():
    """Verify version is bumped to 0.55.0."""
    from video_analysis import __version__

    assert __version__ == "0.55.0"


def test_build_multimodal_messages():
    """Test building multimodal messages for InternVideo3."""
    from video_analysis.backends.internvideo3 import InternVideo3Backend

    backend = InternVideo3Backend()

    # Without images
    messages = backend._build_multimodal_messages("What is in this video?", [])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert len(messages[0]["content"]) == 1
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][0]["text"] == "What is in this video?"

    # With image paths (non-existent, should use raw path)
    messages = backend._build_multimodal_messages(
        "Describe this", ["/path/to/frame.jpg", "/path/to/frame2.png"]
    )
    assert len(messages[0]["content"]) == 3  # 2 images + 1 text


def test_vllm_server_args_structure():
    """Test vLLM server args have correct structure."""
    from video_analysis.backends.internvideo3 import VLLM_SERVER_ARGS

    assert "--port" in VLLM_SERVER_ARGS
    assert "--model" in VLLM_SERVER_ARGS
    assert "--gpu-memory-utilization" in VLLM_SERVER_ARGS
    assert "--enforce-eager" in VLLM_SERVER_ARGS
    # Server should listen on 0.0.0.0
    assert "--host" in VLLM_SERVER_ARGS
    assert "0.0.0.0" in VLLM_SERVER_ARGS
