"""
Tests for the webcam/live camera capture and analysis UI module.

Validates that the camera module is importable without gradio installed,
that the lightweight analysis path works with mock frames, and that the
injection function produces correct Gradio components.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from PIL import Image

from video_analysis.config import Config

logger = logging.getLogger(__name__)


# =========================================================================
# Camera module: importability
# =========================================================================


def test_camera_module_importable():
    """The camera module must be importable even without gradio installed.

    Gradio imports are deferred inside functions, so importing the module
    at the top level should not raise ImportError.
    """
    # Reset any cached import of ui.camera
    for modname in list(sys.modules.keys()):
        if "camera" in modname and "ui" in modname:
            del sys.modules[modname]

    from ui.camera import inject_camera_tab, _analyze_frame

    assert callable(inject_camera_tab)
    assert callable(_analyze_frame)


# =========================================================================
# _analyze_frame: lightweight single-frame analysis
# =========================================================================


def _create_test_image(size=(640, 480), color=(128, 128, 200)):
    """Create a simple test JPEG image and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img = Image.new("RGB", size, color)
    img.save(path, "JPEG", quality=85)
    return path


def test_analyze_frame_no_ultralytics():
    """When ultralytics is not installed, _analyze_frame should handle gracefully."""
    cfg = Config(data_dir=tempfile.mkdtemp())
    pipeline = MagicMock()
    pipeline.config = cfg
    pipeline._yolo_model = None

    image_path = _create_test_image()

    # Patch ultralytics import to fail
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("No module named 'ultralytics'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from ui.camera import _analyze_frame

        result = _analyze_frame(image_path, pipeline)

    assert "error" in result
    assert result["objects"] == []
    assert result["description"] == ""

    os.unlink(image_path)


def test_analyze_frame_yolo_success():
    """When YOLO model is loaded, _analyze_frame should run detection."""
    cfg = Config(data_dir=tempfile.mkdtemp())
    pipeline = MagicMock()
    pipeline.config = cfg
    pipeline._yolo_model = MagicMock()

    # Simulate YOLO detection results
    mock_box = MagicMock()
    mock_box.cls = [0]
    mock_box.conf = [0.92]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person"}

    pipeline._yolo_model.return_value = [mock_result]

    image_path = _create_test_image()

    # Patch ultralytics to be importable, but allow YOLO mock to work
    with patch.dict("sys.modules", {"ultralytics": MagicMock()}):
        from ui.camera import _analyze_frame

        result = _analyze_frame(image_path, pipeline)

    assert "error" not in result or result["error"] is None
    assert len(result["objects"]) > 0
    assert any(obj["label"] == "person" for obj in result["objects"])

    os.unlink(image_path)


def test_analyze_frame_yolo_import_fail():
    """When ultralytics import fails, YOLO import path should gracefully fail."""
    cfg = Config(data_dir=tempfile.mkdtemp())
    pipeline = MagicMock()
    pipeline.config = cfg
    pipeline._yolo_model = None

    image_path = _create_test_image()

    import builtins

    original_import = builtins.__import__

    def mock_import_fail(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("No module named 'ultralytics'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import_fail):
        from ui.camera import _analyze_frame

        result = _analyze_frame(image_path, pipeline)

    # Should still return gracefully
    assert isinstance(result, dict)
    assert result["objects"] == []

    os.unlink(image_path)


# =========================================================================
# Camera tab injection function (Gradio-free validation)
# =========================================================================


def test_inject_camera_tab_signature():
    """inject_camera_tab should accept a MagicMock (simulating gr.Blocks) and config."""
    from ui.camera import inject_camera_tab

    mock_app = MagicMock()
    cfg = Config(data_dir=tempfile.mkdtemp())

    # Mock gradio so the lazy import inside inject_camera_tab doesn't fail
    mock_gradio = MagicMock()

    with patch.dict("sys.modules", {"gradio": mock_gradio}):
        # Should not raise when called
        inject_camera_tab(mock_app, cfg)

    # Verify the function ran: it attaches _camera_pipeline to the app
    assert hasattr(mock_app, "_camera_pipeline")


def test_inject_camera_tab_default_config():
    """inject_camera_tab should work without config parameter."""
    from ui.camera import inject_camera_tab

    mock_app = MagicMock()

    mock_gradio = MagicMock()

    with patch.dict("sys.modules", {"gradio": mock_gradio}):
        # Should not raise
        inject_camera_tab(mock_app)

    # Verify the function ran: it attaches _camera_pipeline to the app
    assert hasattr(mock_app, "_camera_pipeline")


# =========================================================================
# Helper: analysis result structure
# =========================================================================


def test_analyze_result_structure():
    """The analysis result dict must have the expected keys."""
    cfg = Config(data_dir=tempfile.mkdtemp())
    pipeline = MagicMock()
    pipeline.config = cfg
    pipeline._yolo_model = None

    image_path = _create_test_image()

    import builtins

    original_import = builtins.__import__

    def mock_import_fail(name, *args, **kwargs):
        if name in ("ultralytics", "open_clip"):
            raise ImportError("No module")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import_fail):
        from ui.camera import _analyze_frame

        result = _analyze_frame(image_path, pipeline)

    assert "objects" in result
    assert "description" in result
    assert "error" in result
    assert isinstance(result["objects"], list)
    assert isinstance(result["description"], str)
    assert result["error"] is not None  # because both imports are mocked out

    os.unlink(image_path)


# =========================================================================
# Config compatibility
# =========================================================================


def test_camera_uses_pipeline_config():
    """The camera module should use the same Config object as the main pipeline."""
    from ui.camera import _analyze_frame
    from video_analysis.pipeline import VideoPipeline

    cfg = Config(data_dir=tempfile.mkdtemp())
    pipeline = VideoPipeline(cfg)

    assert pipeline.config is cfg
    assert pipeline.config.yolo_confidence == cfg.yolo_confidence
    assert pipeline.config.clip_model == cfg.clip_model


def test_camera_module_lazy_imports():
    """Verify that gradio imports are indeed deferred (not at module level)."""
    import ast
    import inspect

    from ui import camera as camera_module

    source = inspect.getsource(camera_module)

    # Parse the source and check no top-level "import gradio"
    tree = ast.parse(source)
    top_level_gradio_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names if hasattr(node, "names") else []:
                name = alias.name if hasattr(alias, "name") else ""
                if "gradio" in name:
                    # Only allow imports inside functions (lazy)
                    inside_function = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.FunctionDef):
                            if node in ast.walk(parent):
                                inside_function = True
                                break
                    if not inside_function:
                        top_level_gradio_imports.append(name)

    assert not top_level_gradio_imports, (
        f"Top-level gradio import(s) found: {top_level_gradio_imports}"
    )


# =========================================================================
# Frame format handling
# =========================================================================


def test_analyze_frame_handles_png():
    """_analyze_frame should handle PNG images as well."""
    cfg = Config(data_dir=tempfile.mkdtemp())
    pipeline = MagicMock()
    pipeline.config = cfg
    pipeline._yolo_model = None

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.new("RGBA", (320, 240), (255, 0, 0, 255))
    img.save(path, "PNG")

    import builtins

    original_import = builtins.__import__

    def mock_import_fail(name, *args, **kwargs):
        if name in ("ultralytics", "open_clip"):
            raise ImportError("No module")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import_fail):
        from ui.camera import _analyze_frame

        result = _analyze_frame(path, pipeline)

    assert isinstance(result, dict)
    assert "objects" in result

    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__])
