"""
Cross-Report Evaluation Dashboard — compare quality metrics across runs.

Provides a Gradio tab for viewing historical evaluation reports, comparing
quality metrics side-by-side across different pipeline versions and configs,
and visually tracking regressions and improvements over time.

This turns the evaluation harness from a CLI-only tool into a visual
analytics interface for quality monitoring.

Usage (integration):
    from ui.comparison import inject_comparison_tab

    with gr.Blocks() as app:
        with gr.Tabs():
            inject_comparison_tab(config)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from video_analysis.config import Config
from video_analysis.evaluation import EvalReportStore, EvaluationRunner

logger = logging.getLogger(__name__)

COMPARISON_DARK_CSS = """
.comp-card {
    background: var(--surface, #1e1b2e);
    border: 1px solid var(--border, #3d3a50);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.comp-card:hover { border-color: var(--primary, #7c3aed); }
.comp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.comp-table th {
    color: var(--text-muted, #9895b0);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.75rem;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border, #3d3a50);
    text-align: left;
}
.comp-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(61, 58, 80, 0.4);
}
.comp-table .regression { color: #ef4444; font-weight: 700; }
.comp-table .improvement { color: #34d399; font-weight: 700; }
.comp-table .neutral { color: #e2e0f0; }
.comp-badge-pass { color: #34d399; font-weight: 700; }
.comp-badge-fail { color: #ef4444; font-weight: 700; }
.comp-badge-skip { color: #9895b0; }
.comp-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e0f0;
    margin: 1rem 0 0.5rem;
}
.comp-subheader {
    font-size: 0.85rem;
    color: #9895b0;
    margin: 0 0 1rem;
}
.comp-status-bar {
    display: flex;
    gap: 0.5rem;
    margin: 0.5rem 0 1rem;
}
.comp-status-pass {
    background: #065f46;
    color: #34d399;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.comp-status-fail {
    background: #7f1d1d;
    color: #ef4444;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
"""


def _load_report_store(config: Config) -> EvalReportStore:
    """Get or create the persistent report store."""
    return EvalReportStore(data_dir=config.data_dir)


def _refresh_report_list(config: Config) -> str:
    """Refresh the list of available evaluation reports as HTML."""
    store = _load_report_store(config)
    reports = store.list_reports(limit=50)

    if not reports:
        return (
            '<div class="comp-card">'
            '<div class="comp-subheader">No evaluation reports found. '
            "Run an evaluation first (via the Monitoring tab or CLI)."
            "</div></div>"
        )

    lines = ['<div class="comp-card">']
    lines.append(
        '<div class="comp-header">📋 Evaluation History</div>'
        f'<div class="comp-subheader">{len(reports)} report(s) on disk</div>'
    )

    for r in reports:
        passed_class = "comp-badge-pass" if r["passed"] else "comp-badge-fail"
        status_label = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f'<div style="display:flex;align-items:center;gap:1rem;'
            f'padding:0.4rem 0;border-bottom:1px solid rgba(61,58,80,0.3);">'
            f'<span class="{passed_class}">{status_label}</span>'
            f'<span style="color:#9895b0;font-size:0.8rem;">{r["run_id"]}</span>'
            f'<span style="color:#e2e0f0;flex:1;">{r["summary"]}</span>'
            f'<span style="color:#9895b0;font-size:0.8rem;">v{r["version"]}</span>'
            f'<span style="color:#9895b0;font-size:0.8rem;">'
            f'{r.get("total_tasks", 0)} tasks</span>'
            f"</div>"
        )

    lines.append("</div>")
    return "\n".join(lines)


def _run_compare(config: Config, report_ids_text: str) -> str:
    """
    Compare selected reports and return rendered HTML.
    Accepts comma/space-separated report IDs.
    """
    if not report_ids_text or not report_ids_text.strip():
        return (
            '<div class="comp-card">'
            '<div class="comp-subheader">Enter at least one report ID.</div>'
            "</div>"
        )

    run_ids = [
        r.strip() for r in report_ids_text.replace(",", " ").split() if r.strip()
    ]
    if not run_ids:
        return (
            '<div class="comp-card">'
            '<div class="comp-subheader">No valid report IDs provided.</div>'
            "</div>"
        )

    store = _load_report_store(config)
    comparison = store.compare_reports(run_ids)

    report_ids = comparison.get("report_ids", [])
    if not report_ids:
        return (
            '<div class="comp-card">'
            '<div class="comp-subheader">None of the specified reports were found.</div>'
            "</div>"
        )

    lines: List[str] = []
    lines.append(
        '<div class="comp-card">'
        f'<div class="comp-header">📊 Cross-Report Comparison</div>'
        f'<div class="comp-subheader">Comparing {len(report_ids)} report(s): '
        f'{", ".join(report_ids)}</div>'
    )

    # Version info
    version_comp = comparison.get("version_comparison", {})
    lines.append('<div style="margin:0.5rem 0;">')
    for rid, ver in version_comp.items():
        lines.append(
            f'<span style="display:inline-block;margin-right:1rem;'
            f"padding:0.15rem 0.5rem;background:var(--surface,#1e1b2e);"
            f'border-radius:6px;font-size:0.8rem;">'
            f'<span style="color:#9895b0;">{rid}:</span> '
            f'<span style="color:#e2e0f0;">v{ver}</span></span>'
        )
    lines.append("</div>")

    # Task comparison tables
    task_comp = comparison.get("task_comparison", {})
    if not task_comp:
        lines.append('<div class="comp-subheader">No tasks to compare.</div>')
    else:
        for task_name, task_data in task_comp.items():
            lines.append(
                f'<div class="comp-header" style="font-size:0.95rem;margin-top:1.25rem;">'
                f"🔍 {task_name}</div>"
            )
            metrics_dict = task_data.get("metrics", {})
            if not metrics_dict:
                lines.append(
                    '<div class="comp-subheader" style="font-style:italic;">'
                    "No metrics available</div>"
                )
                continue

            # Build table
            lines.append('<table class="comp-table">')
            # Header row: Metric | Report1 | Report2 | ...
            lines.append("<tr><th>Metric</th>")
            for rid in report_ids:
                ver = version_comp.get(rid, "?")
                lines.append(f"<th>{rid} (v{ver})</th>")
            lines.append("</tr>")

            for mname, mdata in metrics_dict.items():
                lines.append("<tr>")
                lines.append(f'<td style="font-weight:600;">{mname}</td>')
                values = []
                for rid in report_ids:
                    entry = mdata.get(rid)
                    if entry is None:
                        values.append(("N/A", ""))
                    else:
                        val = entry.get("value", "?")
                        unit = entry.get("unit", "")
                        passed = entry.get("passed")
                        display = f"{val}" + (f" {unit}" if unit else "")
                        cls = ""
                        if passed is True:
                            cls = "improvement"
                        elif passed is False:
                            cls = "regression"
                        values.append((display, cls))

                for display, cls in values:
                    lines.append(
                        f'<td class="{cls if cls else "neutral"}">{display}</td>'
                    )
                lines.append("</tr>")

            lines.append("</table>")

    lines.append("</div>")

    # Raw JSON toggle section
    lines.append(
        '<div class="comp-card">'
        '<div class="comp-header" style="font-size:0.95rem;">📄 Raw Data</div>'
        f'<pre style="max-height:300px;overflow:auto;font-size:0.75rem;'
        f'color:#9895b0;background:#151225;padding:0.75rem;border-radius:8px;">'
        f"{json.dumps(comparison, indent=2, default=str)[:5000]}"
        f"</pre></div>"
    )

    return "\n".join(lines)


def _delete_old_reports(config: Config) -> str:
    """Delete all evaluation reports from disk."""
    store = _load_report_store(config)
    count = 0
    for f in store.reports_dir.glob("report_*.json"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return f"Deleted {count} evaluation report(s)."


def inject_comparison_tab(
    config: Config,
    tab_id: str = "evaluation-comparison",
    tab_label: str = "📈 Eval Comparison",
) -> None:
    """Inject the cross-report evaluation comparison tab into a Gradio app.

    Call this inside a ``gr.Tabs()`` context manager.

    Args:
        config: Application configuration.
        tab_id: HTML element ID for the tab.
        tab_label: Label displayed on the tab button.
    """
    with gr.Tab(tab_label, elem_id=tab_id):
        gr.HTML(
            value=(
                '<div class="comp-header">📈 Evaluation Comparison Dashboard</div>'
                '<div class="comp-subheader">'
                "Compare quality metrics across different pipeline runs "
                "to detect regressions and track improvements."
                "</div>"
            )
        )
        gr.Markdown(
            "**Reports are automatically saved every time an evaluation runs** "
            "(via the Monitoring tab, CLI `--eval`, or API endpoint).\n\n"
            "To compare reports, enter the **report IDs** separated by spaces or commas:"
        )
        report_ids_input = gr.Textbox(
            label="Report IDs",
            placeholder="e.g. a1b2c3d4 e5f6g7h8 or a1b2c3d4, e5f6g7h8",
            lines=1,
        )
        compare_btn = gr.Button("📊 Compare Reports", variant="primary", scale=1)
        refresh_btn = gr.Button("🔄 Refresh Report List", variant="secondary", scale=0)
        delete_btn = gr.Button("🗑️ Clear All Reports", variant="stop", scale=0)
        output = gr.HTML(label="Comparison Results")

        # Report list
        report_list_html = gr.HTML(value=_refresh_report_list(config))

        compare_btn.click(
            fn=_run_compare,
            inputs=[gr.State(config), report_ids_input],
            outputs=output,
        )
        refresh_btn.click(
            fn=_refresh_report_list,
            inputs=gr.State(config),
            outputs=report_list_html,
        )
        delete_btn.click(
            fn=_delete_old_reports,
            inputs=gr.State(config),
            outputs=report_list_html,
        ).then(
            fn=lambda: "",
            outputs=output,
        )

    # Inject CSS
    gr.HTML(f"<style>{COMPARISON_DARK_CSS}</style>")
