"""
Agentic Video Understanding Agent — multi-tool video analysis agent.

Inspired by CVPR 2026 Agentic Video Summarization, LongVideoAgent (multi-agent
reasoning with long videos), and VideoWebArena (long-context multimodal agents).

Instead of a static RAG pipeline, this module implements a dynamic, tool-using
agent that orchestrates multiple video understanding capabilities through an
LLM reasoning loop. The agent can:

1. analyze_frames — sample frames from timestamps and send to Video MLLM
2. search_rag — query the vector index for relevant text context
3. detect_objects — run YOLO object detection on specific frames
4. extract_text — OCR text from specific frames
5. search_transcript — find spoken phrases with timestamps
6. temporal_grounding — identify precise moments when events occur
7. summarize_video — produce a structured multi-section summary

The agent uses the configured Video MLLM (Qwen3-VL, VideoChat-Flash, or
SmolVLM2) as its visual reasoning engine, combined with tool invocation for
retrieval, detection, and OCR capabilities.

Usage:
    from video_analysis.agent import VideoUnderstandingAgent

    agent = VideoUnderstandingAgent(video_path="/path/to/video.mp4", rag=rag_instance)
    result = agent.query("What objects appear around 2:30 and who is speaking?")
    print(result.answer)
    print(result.evidence)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

from video_analysis.config import Config
from video_analysis.rag import VideoRAG, RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AgentToolResult:
    """Result from a single tool invocation."""

    tool_name: str
    success: bool
    data: str  # text representation of the result
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentQueryResult:
    """Final result from the agent's reasoning loop.

    Contains the answer, all evidence gathered, and the reasoning trace.
    """

    query: str
    answer: str
    confidence: float  # 0.0 - 1.0, how confident the agent is
    evidence: List[AgentToolResult] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    tools_used: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class AgentTools:
    """Collection of tools the agent can invoke.

    Each tool returns an AgentToolResult with a text-serializable data field
    that can be fed back into the LLM's context.
    """

    def __init__(
        self,
        config: Config,
        rag: Optional[VideoRAG] = None,
        video_path: Optional[str] = None,
        video_id: Optional[str] = None,
    ):
        self.config = config
        self.rag = rag
        self.video_path = Path(video_path) if video_path else None
        self.video_id = video_id
        self._mllm = None
        self._yolo = None
        self._ocr = None
        self._transcript_data: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Lazy-loaded components
    # ------------------------------------------------------------------

    def _get_mllm(self):
        """Lazy-load the Video MLLM for visual reasoning."""
        if self._mllm is None:
            try:
                from video_analysis.video_mllm import VideoMLLM

                from video_analysis.video_mllm import BackendType, ModelSizeType

                backend_val: BackendType = (
                    self.config.video_mllm_backend
                    if self.config.video_mllm_backend
                    in ("auto", "videochat_flash", "smolvlm2", "qwen3_vl")
                    else "auto"
                )
                size_val: ModelSizeType = (
                    self.config.video_mllm_model_size
                    if self.config.video_mllm_model_size in ("2.2B", "500M", "256M")
                    else "2.2B"
                )
                self._mllm = VideoMLLM(
                    model_name=self.config.video_mllm_model,
                    backend=backend_val,
                    model_size=size_val,
                )
            except Exception as exc:
                logger.warning("Could not load Video MLLM for agent: %s", exc)
                return None
        return self._mllm if self._mllm and self._mllm.available else None

    def _get_yolo(self):
        """Lazy-load YOLO model for object detection."""
        if self._yolo is None:
            try:
                from ultralytics import YOLO

                self._yolo = YOLO(self.config.yolo_model)
            except Exception as exc:
                logger.warning("Could not load YOLO for agent: %s", exc)
                return None
        return self._yolo

    def _get_ocr(self):
        """Lazy-load PaddleOCR for text extraction."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR

                self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception as exc:
                logger.warning("Could not load PaddleOCR for agent: %s", exc)
                return None
        return self._ocr

    # ------------------------------------------------------------------
    # Individual tools
    # ------------------------------------------------------------------

    def analyze_frames(
        self, timestamps: List[float], prompt: str = "Describe what you see in detail."
    ) -> AgentToolResult:
        """Sample frames at given timestamps and analyze with Video MLLM.

        Args:
            timestamps: List of timestamps in seconds to sample frames.
            prompt: Visual analysis prompt for the MLLM.

        Returns:
            AgentToolResult with the MLLM's description.
        """
        if not self.video_path or not self.video_path.exists():
            return AgentToolResult(
                tool_name="analyze_frames",
                success=False,
                data="Video file not available.",
                metadata={"timestamps": timestamps},
            )

        mllm = self._get_mllm()
        if mllm is None:
            return AgentToolResult(
                tool_name="analyze_frames",
                success=False,
                data="Video MLLM not available — cannot analyze frames. "
                "Set VIDEO_MLLM_ENABLED=true and ensure a model is configured.",
                metadata={"timestamps": timestamps},
            )

        try:
            # Extract frames at specified timestamps
            import cv2

            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            for ts in timestamps:
                frame_idx = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    from PIL import Image

                    pil_img = Image.fromarray(frame_rgb)
                    frames.append(pil_img)
            cap.release()

            if not frames:
                return AgentToolResult(
                    tool_name="analyze_frames",
                    success=False,
                    data="Could not extract frames at the specified timestamps.",
                    metadata={"timestamps": timestamps},
                )

            # Use describe_scene for the sampled frames
            description = mllm.describe_scene(frames, prompt=prompt)
            return AgentToolResult(
                tool_name="analyze_frames",
                success=True,
                data=description or "No description generated.",
                metadata={
                    "timestamps": timestamps,
                    "num_frames": len(frames),
                },
            )
        except Exception as exc:
            logger.exception("analyze_frames failed")
            return AgentToolResult(
                tool_name="analyze_frames",
                success=False,
                data=f"Error analyzing frames: {exc}",
                metadata={"timestamps": timestamps},
            )

    def search_rag(self, query: str, top_k: int = 5) -> AgentToolResult:
        """Search the RAG vector index for relevant text context.

        Searches transcript, scene descriptions, OCR text, and object
        detection results indexed during pipeline processing.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            AgentToolResult with formatted search results.
        """
        if self.rag is None:
            return AgentToolResult(
                tool_name="search_rag",
                success=False,
                data="RAG index not available.",
                metadata={"query": query},
            )

        try:
            chunks = self.rag.retrieve(
                query=query,
                video_id=self.video_id,
                top_k=top_k,
            )
            if not chunks:
                return AgentToolResult(
                    tool_name="search_rag",
                    success=True,
                    data="No relevant chunks found in the index.",
                    metadata={"query": query, "num_results": 0},
                )

            # Format results as readable text
            lines = []
            for i, c in enumerate(chunks, 1):
                ts = f"{c.timestamp:.1f}s" if c.timestamp else "N/A"
                scene = f"Scene {c.scene_id}" if c.scene_id else "N/A"
                ctype = c.chunk_type or "unknown"
                text_preview = c.text[:300].replace("\n", " ")
                lines.append(
                    f"[{i}] {ctype} @ {ts} ({scene}), score={c.score:.3f}\n"
                    f"    {text_preview}"
                )

            return AgentToolResult(
                tool_name="search_rag",
                success=True,
                data="\n\n".join(lines),
                metadata={
                    "query": query,
                    "num_results": len(chunks),
                },
            )
        except Exception as exc:
            logger.exception("search_rag failed")
            return AgentToolResult(
                tool_name="search_rag",
                success=False,
                data=f"Error searching RAG index: {exc}",
                metadata={"query": query},
            )

    def detect_objects(self, timestamp: float) -> AgentToolResult:
        """Run YOLO object detection on a frame at the given timestamp.

        Args:
            timestamp: Timestamp in seconds.

        Returns:
            AgentToolResult with detected objects and confidence scores.
        """
        if not self.video_path or not self.video_path.exists():
            return AgentToolResult(
                tool_name="detect_objects",
                success=False,
                data="Video file not available.",
                metadata={"timestamp": timestamp},
            )

        yolo = self._get_yolo()
        if yolo is None:
            return AgentToolResult(
                tool_name="detect_objects",
                success=False,
                data="YOLO model not available.",
                metadata={"timestamp": timestamp},
            )

        try:
            import cv2

            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return AgentToolResult(
                    tool_name="detect_objects",
                    success=False,
                    data="Could not extract frame at the given timestamp.",
                    metadata={"timestamp": timestamp},
                )

            results = yolo(frame, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[cls_id]
                    detections.append(f"{label} ({conf:.2f})")

            if not detections:
                return AgentToolResult(
                    tool_name="detect_objects",
                    success=True,
                    data="No objects detected in this frame.",
                    metadata={"timestamp": timestamp, "num_objects": 0},
                )

            return AgentToolResult(
                tool_name="detect_objects",
                success=True,
                data="Detected objects: " + ", ".join(detections),
                metadata={
                    "timestamp": timestamp,
                    "num_objects": len(detections),
                    "objects": detections,
                },
            )
        except Exception as exc:
            logger.exception("detect_objects failed")
            return AgentToolResult(
                tool_name="detect_objects",
                success=False,
                data=f"Error detecting objects: {exc}",
                metadata={"timestamp": timestamp},
            )

    def extract_text(self, timestamp: float) -> AgentToolResult:
        """Run OCR text extraction on a frame at the given timestamp.

        Args:
            timestamp: Timestamp in seconds.

        Returns:
            AgentToolResult with extracted text.
        """
        if not self.video_path or not self.video_path.exists():
            return AgentToolResult(
                tool_name="extract_text",
                success=False,
                data="Video file not available.",
                metadata={"timestamp": timestamp},
            )

        ocr = self._get_ocr()
        if ocr is None:
            return AgentToolResult(
                tool_name="extract_text",
                success=False,
                data="PaddleOCR not available — install with: pip install paddleocr",
                metadata={"timestamp": timestamp},
            )

        try:
            import cv2

            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return AgentToolResult(
                    tool_name="extract_text",
                    success=False,
                    data="Could not extract frame at the given timestamp.",
                    metadata={"timestamp": timestamp},
                )

            result = ocr.ocr(frame, cls=True)
            texts = []
            for line_group in result:
                if line_group is None:
                    continue
                for line in line_group:
                    texts.append(line[1][0])

            if not texts:
                return AgentToolResult(
                    tool_name="extract_text",
                    success=True,
                    data="No text detected in this frame.",
                    metadata={"timestamp": timestamp, "num_texts": 0},
                )

            return AgentToolResult(
                tool_name="extract_text",
                success=True,
                data="Extracted text: " + " | ".join(texts),
                metadata={
                    "timestamp": timestamp,
                    "num_texts": len(texts),
                    "texts": texts,
                },
            )
        except Exception as exc:
            logger.exception("extract_text failed")
            return AgentToolResult(
                tool_name="extract_text",
                success=False,
                data=f"Error extracting text: {exc}",
                metadata={"timestamp": timestamp},
            )

    def search_transcript(self, query: str, top_k: int = 5) -> AgentToolResult:
        """Search transcript segments for matching spoken content.

        When the pipeline has produced a transcript, this tool searches the
        transcript text for matches and returns the most relevant segments
        with their timestamps.

        Args:
            query: Text to search for in the transcript.
            top_k: Maximum number of matching segments to return.

        Returns:
            AgentToolResult with matching transcript segments.
        """
        if self.rag is None:
            return AgentToolResult(
                tool_name="search_transcript",
                success=False,
                data="RAG index not available.",
                metadata={"query": query},
            )

        try:
            # Use ChromaDB to search for transcript chunks specifically
            chunks = self.rag.retrieve(
                query=query,
                video_id=self.video_id,
                top_k=top_k * 2,
            )

            # Filter to transcript-type chunks
            transcript_chunks = [
                c
                for c in chunks
                if c.chunk_type in ("transcript", "fixed_60s", "sliding_30s")
                or c.text.strip()
            ]

            if not transcript_chunks:
                return AgentToolResult(
                    tool_name="search_transcript",
                    success=True,
                    data="No matching transcript segments found.",
                    metadata={"query": query, "num_results": 0},
                )

            transcript_chunks = transcript_chunks[:top_k]
            lines = []
            for i, c in enumerate(transcript_chunks, 1):
                ts = f"{c.timestamp:.1f}s" if c.timestamp else "N/A"
                text_preview = c.text[:200]
                speaker = ""
                if c.metadata and "speaker" in c.metadata:
                    speaker = f" [{c.metadata['speaker']}]"
                lines.append(f"[{i}] @ {ts}{speaker}\n    {text_preview}")

            return AgentToolResult(
                tool_name="search_transcript",
                success=True,
                data="Transcript matches:\n\n" + "\n\n".join(lines),
                metadata={
                    "query": query,
                    "num_results": len(transcript_chunks),
                },
            )
        except Exception as exc:
            logger.exception("search_transcript failed")
            return AgentToolResult(
                tool_name="search_transcript",
                success=False,
                data=f"Error searching transcript: {exc}",
                metadata={"query": query},
            )

    def temporal_grounding(self, event_description: str) -> AgentToolResult:
        """Find precise timestamps where a described event occurs.

        Uses the RAG index to locate temporal segments matching the event
        description, combining semantic search with transcript and scene
        metadata.

        Args:
            event_description: Description of the event to locate.

        Returns:
            AgentToolResult with matched timestamps and context.
        """
        if self.rag is None:
            return AgentToolResult(
                tool_name="temporal_grounding",
                success=False,
                data="RAG index not available.",
                metadata={"event": event_description},
            )

        try:
            search_query = f"{event_description} — find the precise timestamp and scene"
            chunks = self.rag.retrieve(
                query=search_query,
                video_id=self.video_id,
                top_k=10,
            )

            if not chunks:
                return AgentToolResult(
                    tool_name="temporal_grounding",
                    success=True,
                    data="Could not locate this event in the video index.",
                    metadata={"event": event_description, "num_results": 0},
                )

            # Group by timestamp and present the most relevant moments
            seen_ts = set()
            events = []
            for c in chunks:
                ts_rounded = round(c.timestamp, 1) if c.timestamp else 0.0
                if ts_rounded in seen_ts:
                    continue
                seen_ts.add(ts_rounded)
                text_preview = c.text[:150].replace("\n", " ")
                ctype = c.chunk_type or "unknown"
                events.append(
                    f"  @ {ts_rounded:.1f}s (scene {c.scene_id}, {ctype}): {text_preview}"
                )

            return AgentToolResult(
                tool_name="temporal_grounding",
                success=True,
                data="Temporal matches for event:\n" + "\n".join(events[:8]),
                metadata={
                    "event": event_description,
                    "num_matches": len(events),
                    "timestamps": [
                        round(c.timestamp, 1) for c in chunks[:8] if c.timestamp
                    ],
                },
            )
        except Exception as exc:
            logger.exception("temporal_grounding failed")
            return AgentToolResult(
                tool_name="temporal_grounding",
                success=False,
                data=f"Error locating event: {exc}",
                metadata={"event": event_description},
            )

    def summarize_video(self, num_frames: int = 16) -> AgentToolResult:
        """Generate a structured multi-section summary of the entire video.

        Uses the Video MLLM to analyze sampled frames from across the video
        and produces a structured summary with sections for visual content,
        spoken content, key events, and objects.

        Args:
            num_frames: Number of frames to sample evenly across the video.

        Returns:
            AgentToolResult with the structured summary.
        """
        if not self.video_path or not self.video_path.exists():
            return AgentToolResult(
                tool_name="summarize_video",
                success=False,
                data="Video file not available.",
            )

        mllm = self._get_mllm()
        if mllm is None:
            # Fall back to RAG-based summarization if MLLM is unavailable
            return self._summarize_from_rag()

        try:
            import cv2

            cap = cv2.VideoCapture(str(self.video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            if total_frames <= 0:
                return AgentToolResult(
                    tool_name="summarize_video",
                    success=False,
                    data="Could not determine video length.",
                )

            # Sample frames evenly across the video
            if num_frames > total_frames:
                num_frames = max(2, total_frames)

            step = max(1, total_frames // num_frames)
            ts_step = duration / num_frames if duration > 0 else 1.0
            timestamps = [i * ts_step for i in range(min(num_frames, total_frames))]

            frame_result = self.analyze_frames(
                timestamps[: min(num_frames, 8)],  # Limit frames for MLLM context
                prompt=(
                    "Provide a detailed description of this video frame. "
                    "Include: visible objects, people, activities, setting/location, "
                    "text/overlays, and any notable visual elements."
                ),
            )

            # Also gather RAG context for transcript-based summary
            rag_summary = self._summarize_from_rag()

            # Combine visual + transcript summaries
            visual_info = frame_result.data if frame_result.success else ""
            transcript_info = rag_summary.data if rag_summary.success else ""

            sections = [
                f"## Video Summary (duration: {duration:.0f}s)\n",
            ]
            if visual_info:
                sections.append(f"### Visual Content\n{visual_info}\n")
            if transcript_info:
                sections.append(f"### Transcript Highlights\n{transcript_info}\n")

            return AgentToolResult(
                tool_name="summarize_video",
                success=True,
                data="\n".join(sections),
                metadata={
                    "duration_seconds": duration,
                    "num_frames_sampled": len(timestamps),
                    "mllm_available": True,
                },
            )
        except Exception as exc:
            logger.exception("summarize_video failed")
            # Fallback to RAG-only summary
            return self._summarize_from_rag()

    def _summarize_from_rag(self) -> AgentToolResult:
        """Fallback summarization from RAG index when MLLM is unavailable."""
        if self.rag is None:
            return AgentToolResult(
                tool_name="summarize_video",
                success=False,
                data="Neither Video MLLM nor RAG index available.",
            )

        try:
            # Get key scenes and stats from the index
            meta = self.rag.collection.get(include=["metadatas"])
            if not meta["ids"]:
                return AgentToolResult(
                    tool_name="summarize_video",
                    success=True,
                    data="No indexed data found for this video.",
                )

            # Extract unique scenes and count
            scene_ids = set()
            num_transcripts = 0
            num_frames = 0
            filenames = set()
            for m in meta["metadatas"]:
                if m.get("video_id") != self.video_id:
                    continue
                sid = m.get("scene_id")
                if sid is not None:
                    scene_ids.add(sid)
                ctype = m.get("chunk_type", "")
                if ctype == "transcript":
                    num_transcripts += 1
                elif ctype == "frame":
                    num_frames += 1
                fname = m.get("filename")
                if fname:
                    filenames.add(fname)

            # Search for overall content
            summary_chunks = self.rag.retrieve(
                query="summary of video content key events objects people",
                video_id=self.video_id,
                top_k=5,
            )
            content_lines = []
            for c in summary_chunks:
                content_lines.append(
                    f"  @ {c.timestamp:.1f}s (scene {c.scene_id}): {c.text[:200]}"
                )

            lines = [
                f"## Video Summary (RAG-based)",
                f"Filename(s): {', '.join(filenames) or 'unknown'}",
                f"Scenes indexed: {len(scene_ids)}",
                f"Transcript segments: {num_transcripts}",
                f"Frame descriptions: {num_frames}",
                "",
                "### Key Content:",
            ] + content_lines

            return AgentToolResult(
                tool_name="summarize_video",
                success=True,
                data="\n".join(lines),
                metadata={
                    "num_scenes": len(scene_ids),
                    "num_chunks": len(meta["ids"]),
                    "mllm_available": False,
                },
            )
        except Exception as exc:
            logger.exception("_summarize_from_rag failed")
            return AgentToolResult(
                tool_name="summarize_video",
                success=False,
                data=f"Summarization failed: {exc}",
            )


# ---------------------------------------------------------------------------
# Agent orchestrator
# ---------------------------------------------------------------------------


class VideoUnderstandingAgent:
    """Agentic video understanding orchestrator.

    Takes a natural language question about a video and iteratively:
    1. Determines which tools to invoke
    2. Invokes tools and collects results
    3. Synthesizes a final answer from all evidence gathered

    The agent follows a two-phase approach:
    - **Analysis phase**: Invoke tools in parallel/dispatch order to gather
      evidence about the video
    - **Synthesis phase**: Use the LLM to compose a final answer from all
      evidence collected

    This mirrors the CVPR 2026 Agentic Video Summarization paradigm where
    the model actively decides what to look at rather than passively
    processing pre-selected frames.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        rag: Optional[VideoRAG] = None,
        video_path: Optional[str] = None,
        video_id: Optional[str] = None,
    ):
        self.config = config or Config()
        self.rag = rag
        self.video_path = video_path
        self.video_id = video_id
        self._tools = AgentTools(self.config, rag, video_path, video_id)

    def query(
        self,
        question: str,
        context: Optional[List[RetrievedChunk]] = None,
        max_tools: int = 5,
    ) -> AgentQueryResult:
        """Answer a question about the video using the tool-using agent.

        The agent dynamically selects tools based on the question type:

        - "what do you see / describe" → analyze_frames
        - "find / locate / when did" → temporal_grounding + search_transcript
        - "what objects" → detect_objects
        - "what text / what does it say" → extract_text
        - "summarize / overview" → summarize_video
        - general questions → search_rag + analyze_frames

        Args:
            question: Natural language question about the video.
            context: Optional pre-retrieved RAG chunks to bootstrap.
            max_tools: Maximum number of tool invocations (default: 5).

        Returns:
            AgentQueryResult with the answer, evidence, and reasoning trace.
        """
        import time

        start = time.time()

        evidence: List[AgentToolResult] = []
        steps: List[str] = []
        answer_parts: List[str] = []
        q_lower = question.lower().strip()

        # Phase 1: Determine tool strategy based on question type
        steps.append(f"Analyzing question type: {question[:80]}...")

        # Bootstrap with any provided context chunks
        if context:
            rag_text = "\n".join(
                f"@ {c.timestamp:.1f}s ({c.chunk_type}, scene {c.scene_id}): {c.text[:200]}"
                for c in context[:5]
            )
            evidence.append(
                AgentToolResult(
                    tool_name="context_bootstrap",
                    success=True,
                    data=f"Pre-retrieved context:\n{rag_text}",
                    metadata={"num_chunks": len(context)},
                )
            )
            steps.append("Bootstrapped with pre-retrieved RAG context.")

        # Classify question and dispatch tools
        if any(
            w in q_lower
            for w in ["summarize", "summary", "overview", "what happens", "what's in"]
        ):
            steps.append("Question type: summarization → invoking summarize_video")
            steps.append("Also searching RAG for comprehensive context.")
            result = self._tools.summarize_video(num_frames=16)
            evidence.append(result)
            if result.success:
                answer_parts.append(result.data)

            # Also get transcript-based context
            tx = self._tools.search_transcript(question, top_k=5)
            evidence.append(tx)
            if tx.success:
                answer_parts.append(tx.data)

        elif any(
            w in q_lower
            for w in ["find", "locate", "when did", "where does", "at what time"]
        ):
            steps.append(
                f"Question type: temporal grounding → searching for '{question}'"
            )
            tg = self._tools.temporal_grounding(question)
            evidence.append(tg)
            if tg.success:
                answer_parts.append(tg.data)

            # Also search transcript for spoken mentions
            tx = self._tools.search_transcript(question, top_k=5)
            evidence.append(tx)
            if tx.success:
                answer_parts.append(tx.data)

        elif any(
            w in q_lower
            for w in ["objects", "detect", "what can you see", "what is in", "visible"]
        ):
            steps.append("Question type: object detection → sampling key frames")
            # Try to extract timestamps from question
            timestamps = self._extract_timestamps(question)
            if timestamps:
                for ts in timestamps[:3]:
                    do = self._tools.detect_objects(ts)
                    evidence.append(do)
                    if do.success:
                        answer_parts.append(do.data)
                        steps.append(f"Detected objects at {ts:.1f}s")
            else:
                # Sample evenly — early, mid, late
                for ts in [30.0, 120.0, 300.0]:
                    do = self._tools.detect_objects(ts)
                    evidence.append(do)
                    if do.success:
                        answer_parts.append(do.data)

        elif any(
            w in q_lower
            for w in ["text", "ocr", "read", "caption", "subtitle", "what does it say"]
        ):
            steps.append("Question type: OCR/text extraction")
            timestamps = self._extract_timestamps(question) or [30.0, 120.0, 300.0]
            for ts in timestamps[:3]:
                ocr_result = self._tools.extract_text(ts)
                evidence.append(ocr_result)
                if ocr_result.success:
                    answer_parts.append(ocr_result.data)

        elif any(
            w in q_lower
            for w in [
                "transcript",
                "speak",
                "said",
                "say",
                "dialogue",
                "conversation",
                "narrator",
            ]
        ):
            steps.append("Question type: transcript search")
            tx = self._tools.search_transcript(question, top_k=10)
            evidence.append(tx)
            if tx.success:
                answer_parts.append(tx.data)

        elif any(w in q_lower for w in ["who", "person", "people", "face", "speaker"]):
            steps.append("Question type: person/face identification")
            tx = self._tools.search_transcript(question, top_k=5)
            evidence.append(tx)
            if tx.success:
                answer_parts.append(tx.data)

            # Also analyze frames for people
            timestamps = self._extract_timestamps(question) or [60.0, 180.0]
            for ts in timestamps[:2]:
                fa = self._tools.analyze_frames(
                    [ts],
                    prompt="Describe the people visible in this frame — their appearance, "
                    "position, and any identifying features.",
                )
                evidence.append(fa)
                if fa.success:
                    answer_parts.append(fa.data)

        else:
            # General question: search RAG + optionally analyze frames
            steps.append("Question type: general → searching RAG index")
            sr = self._tools.search_rag(question, top_k=8)
            evidence.append(sr)

            if sr.success and "No relevant" not in sr.data:
                answer_parts.append(sr.data)

            # If the question seems visual, also sample frames
            if any(
                w in q_lower
                for w in ["look", "scene", "see", "show", "appear", "background"]
            ):
                steps.append("Question appears visual — also sampling frames")
                timestamps = self._extract_timestamps(question) or [30.0, 120.0]
                fa = self._tools.analyze_frames(
                    timestamps[:3],
                    prompt=f"Question: {question}\n\nDescribe what you see in these frames "
                    f"that is relevant to answering this question.",
                )
                evidence.append(fa)
                if fa.success:
                    answer_parts.append(fa.data)

        # Phase 2: Synthesize answer
        steps.append("Synthesizing final answer from all evidence.")

        all_evidence_text = "\n\n---\n\n".join(
            f"[Tool: {e.tool_name}]\n{e.data}" for e in evidence if e.success
        )
        tools_used = sum(1 for e in evidence if e.success)

        # Always include RAG search for any remaining context
        if not any(e.tool_name == "search_rag" for e in evidence):
            sr = self._tools.search_rag(question, top_k=5)
            evidence.append(sr)
            if sr.success:
                all_evidence_text += f"\n\n---\n\n[Tool: search_rag]\n{sr.data}"

        # Build final answer from evidence
        if answer_parts:
            final_answer = "\n\n".join(answer_parts)
        else:
            # Try to build something from the evidence
            text_parts = [e.data for e in evidence if e.success and len(e.data) > 20]
            if text_parts:
                final_answer = "\n\n".join(text_parts[:3])
            else:
                final_answer = (
                    "I was unable to gather sufficient evidence to answer this question. "
                    "Please ensure the video has been fully processed by the pipeline and "
                    "the RAG index has been built."
                )

        # Estimate confidence
        num_success = sum(1 for e in evidence if e.success)
        total_tools = max(len(evidence), 1)
        confidence = min(1.0, num_success / total_tools)

        elapsed = time.time() - start

        return AgentQueryResult(
            query=question,
            answer=final_answer,
            confidence=confidence,
            evidence=evidence,
            reasoning_steps=steps,
            tools_used=tools_used,
            duration_seconds=elapsed,
        )

    @staticmethod
    def _extract_timestamps(text: str) -> List[float]:
        """Extract timestamp references from a question string.

        Supports formats: MM:SS, H:MM:SS, X seconds, Xs, X minutes.

        Args:
            text: Question text possibly containing timestamps.

        Returns:
            Sorted list of timestamp floats in seconds.
        """
        timestamps: List[float] = []

        # Match MM:SS or H:MM:SS
        time_pattern = re.compile(r"(?:(\d+):)?(\d+):(\d+)")
        for match in time_pattern.finditer(text):
            h = int(match.group(1)) if match.group(1) else 0
            m = int(match.group(2))
            s = int(match.group(3))
            total = h * 3600 + m * 60 + s
            if total > 0:
                timestamps.append(float(total))

        # Match "X seconds" or "Xs" or "X second"
        sec_pattern = re.compile(r"(\d+)\s*(?:seconds?|s)\b", re.IGNORECASE)
        for match in sec_pattern.finditer(text):
            ts = float(match.group(1))
            if ts > 0:
                timestamps.append(ts)

        # Match "X minutes" or "X minute"
        min_pattern = re.compile(r"(\d+)\s*(?:minutes?|min)\b", re.IGNORECASE)
        for match in min_pattern.finditer(text):
            ts = float(match.group(1)) * 60
            if ts > 0:
                timestamps.append(ts)

        return sorted(set(timestamps))

    def generate_report(self, video_id: Optional[str] = None) -> str:
        """Generate a comprehensive video analysis report.

        Combines multiple tool outputs into a structured markdown report
        with sections:

        - Overview (duration, scenes, objects found)
        - Visual content summary
        - Key transcript segments
        - Detected objects
        - OCR text
        - Timeline of key events

        Args:
            video_id: Override video ID filter.

        Returns:
            Markdown-formatted report.
        """
        vid = video_id or self.video_id
        sections = []

        # Overview from RAG
        if self.rag:
            meta = self.rag.collection.get(include=["metadatas"])
            if meta["ids"]:
                scene_ids = set()
                chunk_types = {}
                filename = "unknown"
                for m in meta["metadatas"]:
                    if m.get("video_id") != vid:
                        continue
                    sid = m.get("scene_id")
                    if sid is not None:
                        scene_ids.add(sid)
                    ct = m.get("chunk_type", "unknown")
                    chunk_types[ct] = chunk_types.get(ct, 0) + 1
                    fn = m.get("filename")
                    if fn:
                        filename = fn

                sections.append(
                    f"## Overview\n"
                    f"- **Filename**: {filename}\n"
                    f"- **Video ID**: {vid}\n"
                    f"- **Scenes**: {len(scene_ids)}\n"
                    f"- **Chunks**: {sum(chunk_types.values())} total\n"
                    f"- **Types**: {', '.join(f'{k}: {v}' for k, v in sorted(chunk_types.items()))}\n"
                )

        # Visual summary
        if self.video_path:
            vs = self._tools.summarize_video(num_frames=12)
            if vs.success:
                sections.append(vs.data)

        # Object detection summary
        obj_result = self._tools.detect_objects(30.0)
        if (
            obj_result.success
            and obj_result.data != "No objects detected in this frame."
        ):
            sections.append(f"## Key Objects\n{obj_result.data}")

        # Transcript highlights
        tx = self._tools.search_transcript("key moments important", top_k=5)
        if tx.success and "No matching" not in tx.data:
            sections.append(f"## Transcript Highlights\n{tx.data}")

        if not sections:
            sections.append(
                f"## Video Analysis Report\n"
                f"- **Video ID**: {vid or 'unknown'}\n"
                f"- **Status**: No indexed data available. Process the video through "
                f"the analysis pipeline first.\n"
            )

        return "\n\n".join(sections)
