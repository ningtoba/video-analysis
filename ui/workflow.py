"""
Gradio 6 Workflow integration for the video analysis pipeline.

Wraps each pipeline stage as a callable node in Gradio's visual
Workflow canvas (gr.Workflow, introduced in Gradio 6.17.0).

Users can drag nodes, connect edges, and run the pipeline visually.
Each node auto-generates input/output ports from function signatures.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import gradio as gr

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG
from video_analysis.models import VideoIndex

logger = logging.getLogger(__name__)

WORKFLOW_JSON = Path("data/workflow.json")


# ── Pipeline Stage Functions (wrapped as Workflow nodes) ──


def stage_download_url(url: str) -> str:
    """Download a video from a URL and return the local file path."""
    config = Config()
    path = VideoPipeline.download_from_url(url, config.video_dir)
    return str(path or "")


def stage_process_video(video_path: str) -> str:
    """Run the full analysis pipeline on a local video file.

    Returns a JSON string with video_id, duration, scene_count.
    """
    if not video_path or not Path(video_path).exists():
        return json.dumps({"error": "Video file not found", "video_id": ""})

    config = Config()
    pipeline = VideoPipeline(config)
    index = pipeline.process(video_path)
    return json.dumps(
        {
            "video_id": index.video_id,
            "duration": index.duration,
            "scene_count": len(index.scenes),
            "transcript_segments": len(index.transcript),
        }
    )


def stage_index_video(result_json: str) -> str:
    """Index a processed video into ChromaDB for Q&A retrieval."""
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "Invalid input", "indexed": False})

    if not data.get("video_id"):
        return json.dumps({"error": "No video_id", "indexed": False})

    config = Config()
    pipeline = VideoPipeline(config)
    rag = VideoRAG(config)
    video_path = config.video_dir / f"{data['video_id']}.mp4"

    if not video_path.exists():
        return json.dumps({"error": f"Video not found: {video_path}", "indexed": False})

    index = pipeline.process(str(video_path))
    rag.index_video(index)
    return json.dumps(
        {
            "indexed": True,
            "video_id": data["video_id"],
            "chunks": len(index.scenes),
        }
    )


def stage_ask_question(question: str, video_id_json: str) -> str:
    """Ask a question about an indexed video.

    video_id_json: JSON string from stage_index_video containing video_id.
    question: Natural language question about the video.
    """
    try:
        data = json.loads(video_id_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "Invalid video index input"})

    video_id = data.get("video_id", "")
    if not video_id:
        return json.dumps({"error": "No video_id"})

    config = Config()
    rag = VideoRAG(config)
    from video_analysis.chat import VideoChat

    chat = VideoChat(rag, config)
    response = chat.ask(question, video_id=video_id)
    return json.dumps(
        {
            "answer": response.content,
            "sources": len(response.sources) if response.sources else 0,
        }
    )


# ── Workflow Builder ──


def build_workflow(config: Optional[Config] = None) -> gr.Blocks:
    """Build the Gradio Workflow tab."""
    config = config or Config()

    # Define the pipeline stage functions with explicit names
    bind_map = {
        "Download URL": stage_download_url,
        "Process Video": stage_process_video,
        "Index Video": stage_index_video,
        "Ask Question": stage_ask_question,
    }

    # Default edges: a linear pipeline
    default_edges = [
        ("Download URL", "Process Video"),
        ("Process Video", "Index Video"),
        ("Index Video", "Ask Question"),
    ]

    # Load or initialize the workflow graph file
    graph_path = config.data_dir / "workflow.json"
    if not graph_path.exists():
        graph_path.parent.mkdir(parents=True, exist_ok=True)

    with gr.Blocks() as workflow_app:
        gr.Markdown("### 🧩 Visual Pipeline Builder")
        gr.Markdown(
            "Drag pipeline stages from the sidebar onto the canvas, "
            "connect their ports, and run the pipeline visually. "
            "Each stage auto-generates input/output ports from its signature."
        )

        # The Workflow canvas — the core Gradio 6.17+ component
        gr.Workflow(
            bind=bind_map,
            edges=default_edges,
            graph=str(graph_path),
            label="Video Analysis Pipeline",
        )

        gr.Markdown("---")
        gr.Markdown(
            "**API access:** Each pipeline output is exposed as a named endpoint. "
            "Use `gradio_client` to call individual stages programmatically."
        )

    return workflow_app


def inject_workflow_tab(app: gr.Blocks, config: Config):
    """Inject a Workflow tab into an existing Blocks app.

    This is used when the Workflow is embedded as a tab alongside the
    main Gradio app.
    """
    with app:
        with gr.TabItem("🧩 Pipeline", id="workflow"):
            gr.Markdown("### 🧩 Visual Pipeline Builder")
            gr.Markdown(
                "Connect pipeline stages visually. Drag nodes, wire ports, "
                "and run the full analysis pipeline from the canvas."
            )

            bind_map = {
                "Download URL": stage_download_url,
                "Process Video": stage_process_video,
                "Index Video": stage_index_video,
                "Ask Question": stage_ask_question,
            }

            default_edges = [
                ("Download URL", "Process Video"),
                ("Process Video", "Index Video"),
                ("Index Video", "Ask Question"),
            ]

            graph_path = config.data_dir / "workflow.json"
            if not graph_path.exists():
                graph_path.parent.mkdir(parents=True, exist_ok=True)

            gr.Workflow(
                bind=bind_map,
                edges=default_edges,
                graph=str(graph_path),
                label="Video Analysis Pipeline",
            )
