"""
Webcam / live camera capture and real-time analysis tab for Gradio 6.

Provides a lightweight analysis path that reuses existing pipeline models
(YOLO detection, CLIP description) on single frames captured from a webcam
or uploaded image file.  No new heavy dependencies.

Gradio imports are lazily deferred so this module is importable without
gradio installed (e.g. for testing).
"""

import logging
import os
import tempfile
from typing import Optional

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline

logger = logging.getLogger(__name__)

# ── Helper: lightweight analysis of a single frame ──────────────────────


def _analyze_frame(
    image_path: str,
    pipeline: VideoPipeline,
) -> dict:
    """Run YOLO detection + CLIP description on a single saved frame.

    Args:
        image_path: Path to the saved JPEG/PNG frame.
        pipeline: Initialised VideoPipeline instance (models loaded lazily).

    Returns:
        dict with keys: ``objects`` (list of detected labels),
        ``description`` (CLIP scene label string), ``error`` (if any).
    """
    from PIL import Image

    result: dict = {"objects": [], "description": "", "error": None}

    # ── YOLO detection ──────────────────────────────────────────────
    yolo_objects = []
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics not installed, skipping YOLO detection")
        result["error"] = "ultralytics not installed"
        return result

    if pipeline._yolo_model is None:
        try:
            logger.info("Loading YOLO model for camera analysis")
            pipeline._yolo_model = YOLO(pipeline.config.yolo_model)
        except Exception:
            try:
                pipeline._yolo_model = YOLO("yolo26n.pt")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                result["error"] = str(e)
                # Still attempt CLIP below

    if pipeline._yolo_model is not None:
        try:
            det_results = pipeline._yolo_model(
                image_path,
                conf=pipeline.config.yolo_confidence,
                verbose=False,
            )
            seen = set()
            for r in det_results:
                for box in r.boxes:
                    label = r.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if label not in seen or conf > 0.5:
                        yolo_objects.append(
                            {"label": label, "confidence": round(conf, 3)}
                        )
                        seen.add(label)
            result["objects"] = yolo_objects
        except Exception as e:
            logger.warning(f"YOLO detection error on frame: {e}")
            if result.get("error") is None:
                result["error"] = str(e)

    # ── CLIP description ────────────────────────────────────────────
    try:
        import torch
        import open_clip
    except ImportError:
        logger.warning("open-clip-torch not installed, skipping CLIP description")
        return result

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the pipeline's cached CLIP model or load fresh
    if pipeline._clip_model is None:
        try:
            logger.info("Loading OpenCLIP for camera analysis")
            model, _, preprocess = open_clip.create_model_and_transforms(
                pipeline.config.clip_model,
                pretrained=pipeline.config.clip_pretrained_dataset,
                device=device,
            )
            tokenizer = open_clip.get_tokenizer(pipeline.config.clip_model)
            pipeline._clip_model = model
            pipeline._clip_preprocess = preprocess
            pipeline._clip_tokenizer = tokenizer
        except Exception as e:
            logger.warning(f"Failed to load OpenCLIP: {e}")
            if result.get("error") is None:
                result["error"] = str(e)
            return result

    from video_analysis.pipeline import DEFAULT_CLIP_LABELS

    try:
        img = Image.open(image_path).convert("RGB")
        image_input = pipeline._clip_preprocess(img).unsqueeze(0).to(device)
        text_tokens = pipeline._clip_tokenizer(DEFAULT_CLIP_LABELS).to(device)

        import torch.nn.functional as F

        with torch.no_grad():
            image_features = pipeline._clip_model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
            text_features = pipeline._clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_values, top_indices = similarity.topk(3, dim=-1)

        labels_list = []
        for k in range(3):
            idx = int(top_indices[0, k].item())
            val = float(top_values[0, k].item())
            labels_list.append(f"{DEFAULT_CLIP_LABELS[idx]} ({val:.0f}%)")
        labels_str = ", ".join(labels_list)
        result["description"] = labels_str
    except Exception as e:
        logger.warning(f"CLIP description error: {e}")
        if result.get("error") is None:
            result["error"] = str(e)

    return result


# ── Build the camera tab ────────────────────────────────────────────────


def inject_camera_tab(app, config: Optional[Config] = None):
    """Add a '📷 Camera' tab to an existing ``gr.Blocks`` app.

    Args:
        app: A ``gr.Blocks`` instance (must be in ``with`` context).
        config: Application configuration.
    """
    import gradio as gr

    cfg = config or Config()
    pipeline = VideoPipeline(cfg)

    # Keep a reference so we can clean up models later
    app._camera_pipeline = pipeline

    CAMERA_CSS = """
    .camera-feed { border-radius: 12px; overflow: hidden; }
    .camera-results {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .camera-results h4 { margin: 0 0 0.5rem 0; color: var(--accent); }
    .camera-results .object-tag {
        display: inline-block;
        background: rgba(124,58,237,0.12);
        color: var(--accent);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 0.15rem;
        font-family: monospace;
    }
    .camera-results .description {
        color: var(--text);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    """

    with gr.TabItem("📷 Camera", id="camera"):
        with gr.Row(equal_height=False):
            # LEFT: camera feed
            with gr.Column(scale=3, min_width=480):
                gr.Markdown("### Live Camera Feed")

                source_selector = gr.Radio(
                    choices=["Webcam 0", "Webcam 1", "Upload Image"],
                    value="Webcam 0",
                    label="Camera Source",
                )

                # Webcam live feed
                webcam = gr.Image(
                    sources=["webcam"],
                    label="Webcam Feed",
                    height=360,
                    streaming=False,
                    visible=True,
                    elem_classes="camera-feed",
                    container=True,
                )

                # Upload alternative
                upload_img = gr.Image(
                    sources=["upload"],
                    label="Upload Image for Analysis",
                    height=360,
                    visible=False,
                    elem_classes="camera-feed",
                    container=True,
                )

                with gr.Row():
                    capture_btn = gr.Button(
                        "📸 Capture & Analyze",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    auto_toggle = gr.Button(
                        "▶️ Start Auto-Analysis",
                        variant="secondary",
                        size="lg",
                        scale=2,
                    )
                    clear_cam_btn = gr.Button("🗑 Clear", scale=1)

                auto_interval = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Auto-Analysis Interval (seconds)",
                )

                status_display = gr.HTML(
                    '<span class="badge ready" style="margin-top:0.5rem">● Camera ready</span>'
                )

            # RIGHT: analysis results
            with gr.Column(scale=4, min_width=500):
                gr.Markdown("### 📊 Analysis Results")

                # Captured frame thumbnail
                captured_preview = gr.Image(
                    label="Captured Frame",
                    height=240,
                    visible=False,
                )

                # Results area
                results_html = gr.HTML(
                    '<p style="color:var(--text-muted);">Capture a frame to see analysis results.</p>'
                )

                # Raw JSON toggle
                with gr.Accordion("📋 Raw Detection Data", open=False):
                    raw_json = gr.JSON(label="Raw Results")

        # ── State ────────────────────────────────────────────────────
        camera_running = gr.State(False)

        # ── Source toggling ──────────────────────────────────────────
        def _switch_source(source: str):
            show_webcam = source.startswith("Webcam")
            return {
                webcam: gr.update(visible=show_webcam),
                upload_img: gr.update(visible=not show_webcam),
            }

        source_selector.change(
            fn=_switch_source,
            inputs=[source_selector],
            outputs=[webcam, upload_img],
        )

        # ── Single capture & analyze ─────────────────────────────────
        def _do_capture(
            webcam_val,
            upload_val,
            source: str,
        ):
            """Grab the current frame from whichever source is active and analyse it."""
            # Determine which image to use
            if source.startswith("Webcam"):
                img_data = webcam_val
            else:
                img_data = upload_val

            if img_data is None:
                return (
                    gr.update(visible=False),
                    '<p style="color:#ef4444;">⚠️ No frame available. Point your webcam or upload an image first.</p>',
                    None,
                    '<span class="badge error">● No frame</span>',
                )

            # Save to temp file
            from PIL import Image as PILImage
            import numpy as np

            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = tmp.name

            try:
                # img_data could be numpy array (webcam) or file path string
                if isinstance(img_data, str):
                    # Uploaded image path
                    img = PILImage.open(img_data).convert("RGB")
                elif isinstance(img_data, np.ndarray):
                    img = PILImage.fromarray(img_data)
                else:
                    return (
                        gr.update(visible=False),
                        '<p style="color:#ef4444;">⚠️ Unrecognised image format.</p>',
                        None,
                        '<span class="badge error">● Error</span>',
                    )

                img.save(tmp_path, "JPEG", quality=90)

                # Run lightweight analysis
                analysis = _analyze_frame(tmp_path, pipeline)

                # Build results display
                obj_tags = ""
                for obj in analysis.get("objects", []):
                    label = obj["label"]
                    conf = obj.get("confidence", 0)
                    obj_tags += f'<span class="object-tag">{label} ({conf:.0%})</span> '

                desc_html = ""
                if analysis.get("description"):
                    desc_html = (
                        f'<div class="description">'
                        f"<strong>Scene:</strong> {analysis['description']}"
                        f"</div>"
                    )

                error_html = ""
                if analysis.get("error"):
                    error_html = (
                        f'<p style="color:#f59e0b;font-size:0.85rem;margin-top:0.5rem;">'
                        f"⚠️ Partial error: {analysis['error']}</p>"
                    )

                html = f"""<div class="camera-results">
                    <h4>🎯 Detected Objects</h4>
                    <div>{obj_tags if obj_tags else '<span style="color:var(--text-muted);font-size:0.85rem;">No objects detected.</span>'}</div>
                    {desc_html}
                    {error_html}
                </div>"""

                status_val = (
                    '<span class="badge ready">● Analysis complete</span>'
                    if not analysis.get("error")
                    else '<span class="badge error">● Partial error</span>'
                )

                return (
                    gr.update(visible=True, value=tmp_path),
                    html,
                    analysis,
                    status_val,
                )
            except Exception as e:
                logger.error(f"Capture error: {e}", exc_info=True)
                return (
                    gr.update(visible=False),
                    f'<p style="color:#ef4444;">❌ Capture failed: {str(e)[:200]}</p>',
                    None,
                    '<span class="badge error">● Error</span>',
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        capture_btn.click(
            fn=_do_capture,
            inputs=[webcam, upload_img, source_selector],
            outputs=[captured_preview, results_html, raw_json, status_display],
        )

        # ── Auto-analysis mode ───────────────────────────────────────
        def _auto_analysis_loop(
            webcam_val,
            upload_val,
            source: str,
            running: bool,
            interval: float,
        ):
            """Generator that yields frames periodically while running."""
            import time as _time

            if not running:
                return

            while running:
                if source.startswith("Webcam"):
                    img_data = webcam_val
                else:
                    img_data = upload_val

                if img_data is None:
                    yield (
                        gr.update(visible=False),
                        '<p style="color:#f59e0b;">⏳ Waiting for camera feed...</p>',
                        None,
                        '<span class="badge busy">● Waiting for feed</span>',
                        running,
                    )
                    _time.sleep(2)
                    continue

                # Save to temp and analyse
                from PIL import Image as PILImage
                import numpy as np

                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp_path = tmp.name
                try:
                    if isinstance(img_data, str):
                        img = PILImage.open(img_data).convert("RGB")
                    elif isinstance(img_data, np.ndarray):
                        img = PILImage.fromarray(img_data)
                    else:
                        _time.sleep(interval)
                        continue

                    img.save(tmp_path, "JPEG", quality=90)
                    analysis = _analyze_frame(tmp_path, pipeline)

                    obj_tags = ""
                    for obj in analysis.get("objects", []):
                        label = obj["label"]
                        conf = obj.get("confidence", 0)
                        obj_tags += (
                            f'<span class="object-tag">{label} ({conf:.0%})</span> '
                        )

                    desc_html = ""
                    if analysis.get("description"):
                        desc_html = (
                            f'<div class="description">'
                            f"<strong>Scene:</strong> {analysis['description']}"
                            f"</div>"
                        )

                    error_html = ""
                    if analysis.get("error"):
                        error_html = (
                            f'<p style="color:#f59e0b;font-size:0.85rem;margin-top:0.5rem;">'
                            f"⚠️ Partial error: {analysis['error']}</p>"
                        )

                    html = f"""<div class="camera-results">
                        <h4>🎯 Detected Objects</h4>
                        <div>{obj_tags if obj_tags else '<span style="color:var(--text-muted);font-size:0.85rem;">No objects detected.</span>'}</div>
                        {desc_html}
                        {error_html}
                    </div>"""

                    status_val = (
                        '<span class="badge ready">● Auto-analysis running</span>'
                        if not analysis.get("error")
                        else '<span class="badge error">● Error in auto-analysis</span>'
                    )

                    yield (
                        gr.update(visible=True, value=tmp_path),
                        html,
                        analysis,
                        status_val,
                        running,
                    )
                except Exception as e:
                    logger.error(f"Auto-analysis error: {e}")
                    yield (
                        gr.update(),
                        f'<p style="color:#ef4444;">⚠️ {str(e)[:100]}</p>',
                        None,
                        '<span class="badge error">● Error</span>',
                        running,
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                _time.sleep(interval)

        # We cannot use a direct generator loop with Gradio's click/change in a
        # straightforward way. Instead, use a JS-based periodic trigger approach:
        # the toggle button sets running state, and a JavaScript interval calls
        # the capture endpoint repeatedly.
        #
        # For simplicity, we implement the toggle as: clicking "Start" triggers
        # a single capture + runs a JS interval that clicks the capture button
        # every N seconds. The "Stop" button clears the interval.

        # Inject JS for auto-analysis interval
        cam_auto_js = gr.HTML(
            """<script>
(function() {
  'use strict';
  window.__camIntervalId = null;

  window.__startCamAuto = function(intervalSec) {
    if (window.__camIntervalId) return;  // already running
    const captureBtn = document.querySelector('#camera button:has-text("📸 Capture")') ||
                       Array.from(document.querySelectorAll('#camera button')).find(
                         b => b.textContent.includes('Capture')
                       );
    if (!captureBtn) {
      console.warn('[CameraAuto] Capture button not found');
      return;
    }
    // Trigger a capture immediately
    captureBtn.click();
    // Then repeat
    window.__camIntervalId = setInterval(function() {
      const btn = Array.from(document.querySelectorAll('#camera button')).find(
        b => b.textContent.includes('Capture')
      );
      if (btn) btn.click();
    }, intervalSec * 1000);
    console.log('[CameraAuto] Auto-analysis started, interval=' + intervalSec + 's');
  };

  window.__stopCamAuto = function() {
    if (window.__camIntervalId) {
      clearInterval(window.__camIntervalId);
      window.__camIntervalId = null;
      console.log('[CameraAuto] Auto-analysis stopped');
    }
  };
})();
</script>""",
            visible=True,
        )

        def _toggle_auto(running: bool):
            """Flip the running state and return updated button text + status."""
            new_state = not running
            btn_text = "⏹ Stop Auto-Analysis" if new_state else "▶️ Start Auto-Analysis"
            status_val = (
                '<span class="badge busy">● Auto-analysis running</span>'
                if new_state
                else '<span class="badge ready">● Auto-analysis stopped</span>'
            )
            return new_state, btn_text, status_val

        auto_toggle.click(
            fn=_toggle_auto,
            inputs=[camera_running],
            outputs=[camera_running, auto_toggle, status_display],
        )

        # Clear button
        def _clear_camera():
            return (
                gr.update(value=None),
                gr.update(value=None, visible=False),
                '<p style="color:var(--text-muted);">Capture cleared.</p>',
                None,
                '<span class="badge ready">● Camera ready</span>',
            )

        clear_cam_btn.click(
            fn=_clear_camera,
            outputs=[webcam, captured_preview, results_html, raw_json, status_display],
        )

    # ── Inject CSS ──────────────────────────────────────────────────
    gr.HTML(f"<style>{CAMERA_CSS}</style>", visible=True)
