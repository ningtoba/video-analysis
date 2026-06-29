"""
Monitoring Dashboard — System metrics, evaluation runner, and job queue viewer.

Provides an interactive Gradio tab for real-time system health visualization
and evaluation task execution directly from the UI.

Usage:
    from ui.monitor import inject_monitor_tab

    with gr.Blocks() as app:
        with gr.Tabs():
            inject_monitor_tab(...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import gradio as gr

from video_analysis.config import Config
from video_analysis.job_queue import get_default_manager
from video_analysis.evaluation import EvaluationRunner
from video_analysis import metrics as va_metrics

logger = logging.getLogger(__name__)

MONITOR_DARK_CSS = """
.monitor-card {
    background: var(--surface, #1e1b2e);
    border: 1px solid var(--border, #3d3a50);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.monitor-card:hover { border-color: var(--primary, #7c3aed); }
.monitor-card .label {
    color: var(--text-muted, #9895b0);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.monitor-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text, #e2e0f0);
    margin: 0.25rem 0;
}
.monitor-card .value.success { color: #34d399; }
.monitor-card .value.warning { color: #fbbf24; }
.monitor-card .value.error { color: #ef4444; }
.monitor-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
}
"""


def _build_metric_html(
    label: str,
    value: str,
    status_class: str = "",
) -> str:
    """Render a single metric card as HTML."""
    cls = f"value {status_class}".strip() if status_class else "value"
    return (
        f'<div class="monitor-card">'
        f'<div class="label">{label}</div>'
        f'<div class="{cls}">{value}</div>'
        f"</div>"
    )


def _build_metrics_row(metrics: List[Dict[str, str]]) -> str:
    """Render a row of metric cards."""
    cards = "".join(
        _build_metric_html(m["label"], m["value"], m.get("status", "")) for m in metrics
    )
    return f'<div class="monitor-row">{cards}</div>'


def _collect_system_metrics() -> Dict[str, Any]:
    """Collect current system metrics from the Prometheus counters.

    Returns a dict with live metric values for display.
    """
    va_metrics._ensure_metrics()

    try:
        # Try to read actual Prometheus metric values
        pipeline_total = int(va_metrics.pipeline_runs_total._value.get())
    except (AttributeError, TypeError):
        pipeline_total = 0

    try:
        pipeline_success = int(va_metrics.pipeline_runs_success_total._value.get())
    except (AttributeError, TypeError):
        pipeline_success = 0

    try:
        pipeline_failures = int(va_metrics.pipeline_runs_failure_total._value.get())
    except (AttributeError, TypeError):
        pipeline_failures = 0

    try:
        questions = int(va_metrics.questions_answered_total._value.get())
    except (AttributeError, TypeError):
        questions = 0

    try:
        videos = int(va_metrics.videos_indexed_total._value.get())
    except (AttributeError, TypeError):
        videos = 0

    try:
        gpu_mem = int(va_metrics.gpu_memory_bytes._value.get())
    except (AttributeError, TypeError):
        gpu_mem = 0

    gpu_gb = gpu_mem / (1024**3) if gpu_mem > 0 else 0.0

    return {
        "pipeline_total": pipeline_total,
        "pipeline_success": pipeline_success,
        "pipeline_failures": pipeline_failures,
        "questions_answered": questions,
        "videos_indexed": videos,
        "gpu_memory_gb": gpu_gb,
    }


def _build_system_metrics_html(metrics: Dict[str, Any]) -> str:
    """Build the system metrics section HTML."""
    pipeline_rate = metrics["pipeline_success"] / max(metrics["pipeline_total"], 1)
    pipeline_class = (
        "success"
        if pipeline_rate >= 0.9
        else "warning" if pipeline_rate >= 0.7 else "error"
    )
    failure_str = str(metrics["pipeline_failures"])
    failure_class = (
        "success"
        if metrics["pipeline_failures"] == 0
        else "warning" if metrics["pipeline_failures"] < 5 else "error"
    )

    return _build_metrics_row(
        [
            {"label": "Pipeline Runs", "value": str(metrics["pipeline_total"])},
            {
                "label": "Success Rate",
                "value": f"{pipeline_rate * 100:.0f}%",
                "status": pipeline_class,
            },
            {"label": "Failures", "value": failure_str, "status": failure_class},
            {"label": "Videos Indexed", "value": str(metrics["videos_indexed"])},
            {"label": "Questions Asked", "value": str(metrics["questions_answered"])},
            {
                "label": "GPU Memory",
                "value": (
                    f"{metrics['gpu_memory_gb']:.2f} GB"
                    if metrics["gpu_memory_gb"] > 0
                    else "N/A"
                ),
                "status": "warning" if metrics["gpu_memory_gb"] > 10 else "",
            },
        ]
    )


def _build_job_queue_html(cfg: Config) -> str:
    """Build the job queue status section."""
    import asyncio
    import inspect

    try:
        jm = get_default_manager()
        raw = jm.list_jobs()
        # Work with both sync mocks and real async managers
        if inspect.iscoroutine(raw):
            # Called during synchronous UI construction — run inline
            raw = asyncio.run(raw)
        jobs = list(raw) if isinstance(raw, (list, tuple)) else []
    except Exception:
        return '<div class="monitor-card">Job queue not available (no manager instance).</div>'

    if not jobs:
        return '<div class="monitor-card">No jobs in queue.</div>'

    rows = []
    for j in jobs[-20:]:  # last 20
        status_badge = j.status.value if hasattr(j.status, "value") else str(j.status)
        status_class_map = {
            "pending": "warning",
            "running": "warning",
            "completed": "success",
            "failed": "error",
        }
        sc = status_class_map.get(status_badge.lower(), "")
        pct = (
            f"{j.progress_pct}%"
            if hasattr(j, "progress_pct") and j.progress_pct
            else ""
        )
        progress = f" ({pct})" if pct else ""
        rows.append(
            f'<div class="monitor-card" style="padding: 0.5rem 1rem; display: flex; '
            f'justify-content: space-between; align-items: center;">'
            f'<span style="font-family: monospace; font-size: 0.85rem;">{j.job_id[:12]}...</span>'
            f'<span><span class="badge {sc}">{status_badge}</span>{progress}</span>'
            f"</div>"
        )

    return "".join(rows)


def _build_metrics_snapshot_html(cfg: Config) -> str:
    """Build full metrics snapshot HTML for the dashboard."""
    metrics = _collect_system_metrics()
    html = "<h3 style='margin-top:0'>System Metrics</h3>"
    html += _build_system_metrics_html(metrics)
    html += "<h3>Job Queue</h3>"
    html += _build_job_queue_html(cfg)
    return html


# ── Evaluation Runner Callbacks ──────────────────────────────────────────


def _run_eval_task(cfg: Config, task_names: str) -> str:
    """Run evaluation tasks and return results as formatted text."""
    try:
        runner = EvaluationRunner(cfg)
        task_list: Optional[List[str]] = None
        if task_names and task_names.strip():
            task_list = [t.strip() for t in task_names.split(",") if t.strip()]
        report = runner.run_all(task_names=task_list)
    except Exception as exc:
        logger.exception("Evaluation runner failed")
        return f"**Error:** {exc}"

    results = []
    for r in report.results:
        status_icon = "✅" if r.all_passed else "❌" if r.status == "fail" else "⚠️"
        metrics_str = ", ".join(
            f"{m.name}={m.value:.3f}{' ✅' if m.passed else ''}" for m in r.metrics
        )
        results.append(
            f"{status_icon} **{r.task_name}** — {r.status} ({r.duration_ms:.0f}ms)\n"
            f"   {metrics_str}\n"
            f"   {r.task_description}"
        )
    summary = report.summary()
    return "\n\n".join(results) + f"\n\n---\n**Summary:** {summary}"


def inject_monitor_tab(
    app: gr.Blocks,
    cfg: Config,
) -> None:
    """Inject the Monitoring dashboard tab into a Gradio Blocks app.

    Args:
        app: The Gradio Blocks instance to add the tab to.
        cfg: Application configuration.
    """
    with gr.TabItem("📊 Monitor", id="monitor"):
        with gr.Row():
            refresh_btn = gr.Button(
                "🔄 Refresh Metrics", size="sm", variant="secondary"
            )
            eval_task_input = gr.Textbox(
                label="Eval Tasks (comma-separated, empty=all)",
                placeholder="e.g. retrieval_precision, scene_boundary_accuracy",
                scale=2,
            )
            run_eval_btn = gr.Button(
                "▶ Run Evaluation", size="sm", variant="primary", scale=1
            )

        with gr.Row():
            metrics_html = gr.HTML(value=_build_metrics_snapshot_html(cfg))

        with gr.Row():
            eval_output = gr.Markdown(
                value="Run evaluation tasks to see results here.",
                label="Evaluation Results",
            )

        # Refresh handler
        def _refresh_metrics(cfg: Config) -> str:
            return _build_metrics_snapshot_html(cfg)

        refresh_btn.click(
            fn=_refresh_metrics,
            inputs=[gr.State(cfg)],
            outputs=[metrics_html],
        )

        # Eval run handler
        def _on_run_eval(cfg: Config, task_str: str) -> str:
            return _run_eval_task(cfg, task_str)

        run_eval_btn.click(
            fn=_on_run_eval,
            inputs=[gr.State(cfg), eval_task_input],
            outputs=[eval_output],
        )

        # Inject CSS (separate from content to avoid rendering as visible text)
        gr.HTML(f"<style>{MONITOR_DARK_CSS}</style>", visible=True)
