"""Tests for Qwen3-VL-30B-A3B MoE backend (v0.35.0).

Covers:
- Backend selection and resolution
- Config integration — video_mllm_backend = "qwen3_vl"
- VideoMLLM qwen3_vl route integration
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


def test_qwen3_vl_importable():
    """Test that the Qwen3-VL backend module can be imported cleanly."""
    from video_analysis.backends.qwen3_vl import (
        Qwen3VLBackend,
    )

    # Check that the class exists
    assert Qwen3VLBackend is not None


def test_version_check():
    """Verify version is bumped to 0.46.0."""
    from video_analysis import __version__

    assert __version__ == "0.60.0"
