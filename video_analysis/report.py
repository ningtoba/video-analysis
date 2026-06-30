"""
Structured Video Report Generation — comprehensive JSON-schema video analysis reports.

Generates a complete, structured video report from pipeline analysis results
with full type annotations. The report schema covers:

  - Video metadata (duration, resolution, FPS, hash)
  - Timeline summary (scene boundaries, frames)
  - Per-scene breakdowns (objects, text, speakers, key moments)
  - Full transcript report (speakers, gaps, key phrases)
  - Object catalog (unique objects, frequency, timeline)
  - Action summary
  - OCR summary
  - Face summary
  - Chapter summaries
  - Curation results
  - RAG statistics
  - Quality metrics (blur, brightness, occlusion)

The report can be serialised to JSON for API consumption, saved/loaded from
disk, and rendered as human-readable markdown.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_analysis.config import Config
from video_analysis.models import VideoIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report schema dataclasses
# ---------------------------------------------------------------------------

REPORT_SCHEMA_VERSION = "1.0"

# Display / analysis limits
_MAX_TOP_ENTITIES: int = 20  # top objects, key phrases
_MAX_TOP_ACTIONS: int = 10
_MAX_TOP_SCENES: int = 10
_MAX_TOP_THEMES: int = 8
_MAX_DESCRIPTION_PREVIEW: int = 200
_MAX_SILENT_PERIODS: int = 50
_MIN_WORD_LENGTH_KEY_PHRASE: int = 3
_SILENT_GAP_THRESHOLD_S: float = 2.0
_TRUNCATION_DECIMALS: int = 1

# Checksum
_CHECKSUM_CHUNK_SIZE: int = 65536  # 64KB
_CHECKSUM_HEX_LENGTH: int = 16


@dataclass
class VideoMetadata:
    """Metadata about the analysed video."""

    title: str = ""
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    file_size: int = 0
    video_id: str = ""
    file_path: str = ""
    checksum: str = ""  # SHA-256 of first 64KB + file size
    processing_time_seconds: float = 0.0
    pipeline_version: str = "0.0.0"
    source_url: str = ""  # YouTube URL or original source


@dataclass
class QualityMetrics:
    """Per-frame quality summary for the entire video."""

    mean_blur_score: float = 0.0
    mean_brightness_score: float = 0.0
    mean_trustworthiness: float = 0.0
    total_frames_assessed: int = 0
    frames_below_threshold: int = 0
    worst_frame_quality: float = 0.0
    best_frame_quality: float = 0.0


@dataclass
class TimelineSummary:
    """High-level timeline overview."""

    num_scenes: int = 0
    num_frames: int = 0
    total_transcript_duration: float = 0.0
    scene_boundaries: List[float] = field(default_factory=list)
    mean_scene_duration: float = 0.0
    min_scene_duration: float = 0.0
    max_scene_duration: float = 0.0


@dataclass
class KeyMoment:
    """A notable moment in the video."""

    timestamp: float = 0.0
    description: str = ""
    source: str = ""  # "transcript", "scene_change", "object_appearance", "action"


@dataclass
class SceneReport:
    """Detailed information about a single scene."""

    scene_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    description: str = ""
    objects: List[str] = field(default_factory=list)
    text_spoken: str = ""
    speakers: List[str] = field(default_factory=list)
    key_moments: List[KeyMoment] = field(default_factory=list)
    thumbnail_path: str = ""
    quality_score: float = 0.0


@dataclass
class TranscriptReport:
    """Full transcript analysis report."""

    total_segments: int = 0
    total_duration: float = 0.0
    speaker_count: int = 0
    speakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # speakers: {"SPEAKER_00": {"segment_count": 10, "total_words": 500, "word_count_pct": 0.5}}
    key_phrases: List[str] = field(default_factory=list)
    total_words: int = 0
    silent_periods: List[Tuple[float, float]] = field(default_factory=list)
    # silent_periods: [(start, end), ...] in seconds
    language_detected: str = ""


@dataclass
class ObjectCatalog:
    """Catalog of all detected objects."""

    unique_objects: List[str] = field(default_factory=list)
    total_detections: int = 0
    object_frequency: Dict[str, int] = field(default_factory=dict)
    # object_frequency: {"person": 42, "car": 15}
    top_objects: List[Tuple[str, int, int]] = field(default_factory=list)
    # top_objects: [(name, count, scenes_visible), ...]  # noqa
    object_timeline: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ActionSummary:
    """Summary of recognised actions."""

    actions: List[Dict[str, Any]] = field(default_factory=list)
    action_categories: Dict[str, int] = field(default_factory=dict)
    dominant_actions: List[Tuple[str, float]] = field(default_factory=list)
    total_action_frames: int = 0


@dataclass
class OCRSummary:
    """Summary of OCR text extraction."""

    total_text_frames: int = 0
    unique_texts: List[str] = field(default_factory=list)
    text_timeline: List[Dict[str, Any]] = field(default_factory=list)
    language_hints: List[str] = field(default_factory=list)


@dataclass
class FaceSummary:
    """Summary of face detection and recognition."""

    total_faces: int = 0
    unique_identities: List[str] = field(default_factory=list)
    face_timeline: List[Dict[str, Any]] = field(default_factory=list)
    most_frequent: str = ""
    face_frequency: Dict[str, int] = field(default_factory=dict)


@dataclass
class ChapterSummary:
    """Summary of a single video chapter."""

    chapter_id: int = 0
    title: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    summary: str = ""
    scene_ids: List[int] = field(default_factory=list)


@dataclass
class CurationSummary:
    """Summary of autonomous curation results."""

    iterations: int = 0
    entities_found: List[str] = field(default_factory=list)
    exploration_path: List[str] = field(default_factory=list)
    report_path: str = ""
    curiosity_score: float = 0.0


@dataclass
class RAGStats:
    """ChromaDB indexing statistics."""

    total_chunks: int = 0
    chunk_types: Dict[str, int] = field(default_factory=dict)
    embedding_model: str = ""
    top_k_retrieval: int = 0
    has_scene_graph: bool = False
    has_colbert_reranker: bool = False


@dataclass
class VideoReport:
    """Top-level structured video analysis report.

    Schema version: 1.0
    """

    schema_version: str = REPORT_SCHEMA_VERSION
    generated_at: str = ""
    video: VideoMetadata = field(default_factory=VideoMetadata)
    timeline: TimelineSummary = field(default_factory=TimelineSummary)
    scenes: List[SceneReport] = field(default_factory=list)
    transcript: TranscriptReport = field(default_factory=TranscriptReport)
    objects: ObjectCatalog = field(default_factory=ObjectCatalog)
    actions: ActionSummary = field(default_factory=ActionSummary)
    ocr: OCRSummary = field(default_factory=OCRSummary)
    faces: FaceSummary = field(default_factory=FaceSummary)
    chapters: List[ChapterSummary] = field(default_factory=list)
    curation: Optional[CurationSummary] = None
    rag_stats: RAGStats = field(default_factory=RAGStats)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generates structured VideoReport from pipeline results.

    Supports three construction modes:
      1. from_video_index(video_index) — from an in-memory VideoIndex object
      2. from_video_id(video_id) — reads stored ChromaDB collections
      3. from_pipeline_result(result) — from raw pipeline dict output

    Usage:
        generator = ReportGenerator(config, rag=rag_instance)
        report = generator.from_video_index(video_index)
        json_str = generator.to_json(report)
        generator.save(report, Path("report.json"))
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        rag: Optional[Any] = None,
        pipeline_version: str = "0.0.0",
    ):
        self._config = config or Config()
        self._rag = rag
        self._pipeline_version = pipeline_version
        from video_analysis import __version__ as va_version

        self._va_version = va_version

    def from_video_index(
        self,
        index: VideoIndex,
        processing_time: float = 0.0,
        video_path: Optional[Path] = None,
    ) -> VideoReport:
        """Build a full VideoReport from a VideoIndex object.

        Args:
            index: Completed VideoIndex from the pipeline.
            processing_time: Total pipeline processing time in seconds.
            video_path: Path to the video file (for checksum and size).

        Returns:
            Fully populated VideoReport.
        """
        now = datetime.now(timezone.utc).isoformat()

        # --- Video metadata ---
        metadata = VideoMetadata(
            video_id=index.video_id,
            title=Path(index.filepath).name if index.filepath else index.video_id,
            duration=index.duration,
            file_path=index.filepath,
            pipeline_version=self._pipeline_version or self._va_version or "0.0.0",
            processing_time_seconds=processing_time,
        )

        if video_path and video_path.exists():
            metadata.file_size = video_path.stat().st_size
            metadata.checksum = self._compute_checksum(video_path)

        # --- Timeline ---
        scene_boundaries = [s.start_time for s in index.scenes]
        if index.scenes:
            scene_durations = [s.end_time - s.start_time for s in index.scenes]
            timeline = TimelineSummary(
                num_scenes=len(index.scenes),
                num_frames=sum(len(s.key_frames) for s in index.scenes),
                total_transcript_duration=(
                    sum(seg.end - seg.start for seg in index.transcript)
                    if index.transcript
                    else 0.0
                ),
                scene_boundaries=scene_boundaries,
                mean_scene_duration=(
                    sum(scene_durations) / len(scene_durations) if scene_durations else 0.0
                ),
                min_scene_duration=min(scene_durations) if scene_durations else 0.0,
                max_scene_duration=max(scene_durations) if scene_durations else 0.0,
            )
        else:
            timeline = TimelineSummary()

        # --- Scenes ---
        scenes: List[SceneReport] = []
        for s in index.scenes:
            objects_in_scene: List[str] = []
            key_moments: List[KeyMoment] = []
            speakers: List[str] = []

            for f in s.key_frames:
                if f.objects:
                    for obj in f.objects:
                        label = obj.get("label", str(obj))
                        if label not in objects_in_scene:
                            objects_in_scene.append(label)

                if f.timestamp:
                    if f.description:
                        key_moments.append(
                            KeyMoment(
                                timestamp=f.timestamp,
                                description=f.description[:_MAX_DESCRIPTION_PREVIEW],
                                source="frame_description",
                            )
                        )

            scene_report = SceneReport(
                scene_id=s.scene_id,
                start_time=s.start_time,
                end_time=s.end_time,
                duration=s.end_time - s.start_time,
                description=s.summary or "",
                objects=objects_in_scene,
                text_spoken=s.transcript or "",
                speakers=speakers,
                key_moments=key_moments,
                thumbnail_path=(s.key_frames[0].filepath if s.key_frames else ""),
                quality_score=1.0,
            )
            scenes.append(scene_report)

        # --- Transcript ---
        transcript_report = TranscriptReport()
        if index.transcript:
            speakers_dict: Dict[str, Dict[str, Any]] = {}
            total_words = 0
            for seg in index.transcript:
                words = seg.text.split()
                word_count = len(words)
                total_words += word_count
                speaker = seg.speaker or "UNKNOWN"
                if speaker not in speakers_dict:
                    speakers_dict[speaker] = {"segment_count": 0, "total_words": 0}
                speakers_dict[speaker]["segment_count"] += 1
                speakers_dict[speaker]["total_words"] += word_count

            # Compute percentages
            for sp in speakers_dict.values():
                sp["word_count_pct"] = round(sp["total_words"] / max(total_words, 1), 3)

            # Find silent periods (gaps > 2s between consecutive segments)
            silent_periods: List[Tuple[float, float]] = []
            sorted_segs = sorted(index.transcript, key=lambda x: x.start)
            for i in range(1, len(sorted_segs)):
                gap = sorted_segs[i].start - sorted_segs[i - 1].end
                if gap > _SILENT_GAP_THRESHOLD_S:
                    silent_periods.append(
                        (
                            round(sorted_segs[i - 1].end, _TRUNCATION_DECIMALS),
                            round(sorted_segs[i].start, _TRUNCATION_DECIMALS),
                        )
                    )

            # Extract key phrases (words appearing in multiple segments)
            word_freq: Dict[str, int] = {}
            for seg in index.transcript:
                for w in set(w.lower().strip(".,!?;:") for w in seg.text.split()):
                    if len(w) > _MIN_WORD_LENGTH_KEY_PHRASE:
                        word_freq[w] = word_freq.get(w, 0) + 1
            key_phrases_sorted = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[
                :_MAX_TOP_ENTITIES
            ]
            key_phrases = [k for k, _ in key_phrases_sorted]

            transcript_report = TranscriptReport(
                total_segments=len(index.transcript),
                total_duration=timeline.total_transcript_duration,
                speaker_count=len(speakers_dict),
                speakers=speakers_dict,
                key_phrases=key_phrases,
                total_words=total_words,
                silent_periods=silent_periods[:_MAX_SILENT_PERIODS],
            )

        # --- Objects ---
        unique_objects: List[str] = []
        object_freq: Dict[str, int] = {}
        total_detections = 0
        for s in index.scenes:
            for f in s.key_frames:
                for obj in f.objects:
                    label = obj.get("label", str(obj))
                    if label not in unique_objects:
                        unique_objects.append(label)
                    object_freq[label] = object_freq.get(label, 0) + 1
                    total_detections += 1

        top_objects_list = sorted(object_freq.items(), key=lambda x: x[1], reverse=True)[
            :_MAX_TOP_ENTITIES
        ]

        object_catalog = ObjectCatalog(
            unique_objects=unique_objects,
            total_detections=total_detections,
            object_frequency=object_freq,
            top_objects=[(name, count, 0) for name, count in top_objects_list],
        )

        # --- Actions ---
        actions_summary = ActionSummary()
        action_counter: Dict[str, int] = {}
        actions_list: List[Dict[str, Any]] = []
        for s in index.scenes:
            for f in s.key_frames:
                if f.action:
                    actions_list.append(
                        {
                            "timestamp": f.timestamp,
                            "action": f.action,
                            "confidence": f.action_confidence or 0.0,
                            "scene_id": s.scene_id,
                        }
                    )
                    action_counter[f.action] = action_counter.get(f.action, 0) + 1

        if actions_list:
            dominant = sorted(action_counter.items(), key=lambda x: x[1], reverse=True)[
                :_MAX_TOP_ACTIONS
            ]
            actions_summary = ActionSummary(
                actions=actions_list,
                action_categories=action_counter,
                dominant_actions=[(name, float(cnt)) for name, cnt in dominant],
                total_action_frames=len(actions_list),
            )

        # --- OCR ---
        ocr_summary = OCRSummary()
        ocr_frames = 0
        unique_texts: List[str] = []
        ocr_timeline: List[Dict[str, Any]] = []
        for s in index.scenes:
            for f in s.key_frames:
                if f.ocr_text:
                    ocr_frames += 1
                    clean = " ".join(f.ocr_text.split())
                    ocr_timeline.append(
                        {
                            "timestamp": f.timestamp,
                            "text": f.ocr_text,
                            "scene_id": s.scene_id,
                        }
                    )
                    if clean and clean not in unique_texts:
                        unique_texts.append(clean)

        if ocr_frames:
            ocr_summary = OCRSummary(
                total_text_frames=ocr_frames,
                unique_texts=unique_texts,
                text_timeline=ocr_timeline,
            )

        # --- Faces ---
        face_summary = FaceSummary()
        face_counter: Dict[str, int] = {}
        total_faces = 0
        face_timeline: List[Dict[str, Any]] = []
        for s in index.scenes:
            for f in s.key_frames:
                for face in f.faces:
                    total_faces += 1
                    face_id = face.get("face_id", face.get("id", "unknown"))
                    if face_id != "unknown":
                        face_counter[face_id] = face_counter.get(face_id, 0) + 1
                    face_timeline.append(
                        {
                            "timestamp": f.timestamp,
                            "face_id": face_id,
                            "confidence": face.get("confidence", 0.0),
                            "bbox": face.get("bbox", []),
                            "scene_id": s.scene_id,
                        }
                    )

        most_frequent_id = max(face_counter, key=lambda k: face_counter[k]) if face_counter else ""

        if total_faces:
            face_summary = FaceSummary(
                total_faces=total_faces,
                unique_identities=sorted(face_counter.keys()),
                face_timeline=face_timeline,
                most_frequent=most_frequent_id,
                face_frequency=face_counter,
            )

        # --- RAG stats ---
        rag_stats = RAGStats(
            embedding_model=self._config.embedding_model,
            top_k_retrieval=self._config.top_k_retrieval,
            has_scene_graph=self._config.scene_graph_enabled,
            has_colbert_reranker=self._config.colbert_reranker_enabled
            or self._config.colbert_att_reranker_enabled,
            total_chunks=len(index.chunks) if index.chunks else 0,
            chunk_types={},
        )

        if index.chunks:
            for c in index.chunks:
                ct = c.get("chunk_type", "unknown")
                rag_stats.chunk_types[ct] = rag_stats.chunk_types.get(ct, 0) + 1

        report = VideoReport(
            schema_version=REPORT_SCHEMA_VERSION,
            generated_at=now,
            video=metadata,
            timeline=timeline,
            scenes=scenes,
            transcript=transcript_report,
            objects=object_catalog,
            actions=actions_summary,
            ocr=ocr_summary,
            faces=face_summary,
            rag_stats=rag_stats,
        )

        return report

    def from_video_id(self, video_id: str) -> VideoReport:
        """Build a report from stored ChromaDB data for a video ID.

        Requires a RAG instance with access to the video's collection.

        Args:
            video_id: Video ID to build report for.

        Returns:
            VideoReport populated from indexed data.
        """
        now = datetime.now(timezone.utc).isoformat()

        metadata = VideoMetadata(video_id=video_id)

        # Try to get metadata from RAG
        timeline = TimelineSummary()
        object_catalog = ObjectCatalog()
        transcript_report = TranscriptReport()
        rag_stats = RAGStats(
            embedding_model=self._config.embedding_model,
            top_k_retrieval=self._config.top_k_retrieval,
        )
        quality = QualityMetrics()

        if self._rag:
            try:
                meta = self._rag.collection.get(include=["metadatas"])
                if meta["ids"]:
                    scene_ids: set = set()
                    chunk_type_counts: Dict[str, int] = {}
                    filenames: set = set()
                    objects_found: List[str] = []
                    total_chunks = 0

                    for m in meta["metadatas"]:
                        if m.get("video_id") != video_id:
                            continue
                        sid = m.get("scene_id")
                        if sid is not None:
                            scene_ids.add(sid)
                        ct = m.get("chunk_type", "unknown")
                        chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1
                        fn = m.get("filename")
                        if fn:
                            filenames.add(fn)
                        total_chunks += 1
                        # Collect unique objects from metadata
                        obj = m.get("objects")
                        if obj and isinstance(obj, str):
                            for o in obj.split(","):
                                o = o.strip()
                                if o and o not in objects_found:
                                    objects_found.append(o)

                    timeline = TimelineSummary(
                        num_scenes=len(scene_ids),
                    )
                    metadata.title = ", ".join(filenames) if filenames else video_id
                    object_catalog = ObjectCatalog(
                        unique_objects=objects_found,
                        total_detections=total_chunks,
                    )
                    rag_stats = RAGStats(
                        total_chunks=total_chunks,
                        chunk_types=chunk_type_counts,
                        embedding_model=self._config.embedding_model,
                        top_k_retrieval=self._config.top_k_retrieval,
                        has_scene_graph=self._config.scene_graph_enabled,
                        has_colbert_reranker=self._config.colbert_reranker_enabled
                        or self._config.colbert_att_reranker_enabled,
                    )
            except Exception as exc:
                logger.warning("Failed to read RAG data for report: %s", exc)

        report = VideoReport(
            schema_version=REPORT_SCHEMA_VERSION,
            generated_at=now,
            video=metadata,
            timeline=timeline,
            objects=object_catalog,
            transcript=transcript_report,
            rag_stats=rag_stats,
            quality_metrics=quality,
        )
        return report

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, report: VideoReport, indent: int = 2) -> str:
        """Serialise a VideoReport to pretty-printed JSON.

        Args:
            report: The report to serialise.
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(asdict(report), indent=indent, default=str)

    def save(self, report: VideoReport, path: Path) -> Path:
        """Save a VideoReport as JSON to disk.

        Args:
            report: The report to save.
            path: Destination file path.

        Returns:
            The path the report was saved to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(report), encoding="utf-8")
        logger.info("Report saved to %s", path)
        return path

    def load(self, path: Path) -> VideoReport:
        """Load a VideoReport from a JSON file on disk.

        Args:
            path: Path to the JSON report file.

        Returns:
            Deserialised VideoReport.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is malformed.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return self._dict_to_report(data)

    @staticmethod
    def _dict_to_report(data: dict) -> VideoReport:
        """Recursively reconstruct a VideoReport from a dict.

        Handles nested dataclass reconstruction.
        """
        # Reconstruct nested dataclasses
        if "video" in data and isinstance(data["video"], dict):
            data["video"] = VideoMetadata(**data["video"])
        if "timeline" in data and isinstance(data["timeline"], dict):
            data["timeline"] = TimelineSummary(**data["timeline"])
        if "transcript" in data and isinstance(data["transcript"], dict):
            data["transcript"] = TranscriptReport(**data["transcript"])
        if "objects" in data and isinstance(data["objects"], dict):
            data["objects"] = ObjectCatalog(**data["objects"])
        if "actions" in data and isinstance(data["actions"], dict):
            data["actions"] = ActionSummary(**data["actions"])
        if "ocr" in data and isinstance(data["ocr"], dict):
            data["ocr"] = OCRSummary(**data["ocr"])
        if "faces" in data and isinstance(data["faces"], dict):
            data["faces"] = FaceSummary(**data["faces"])
        if "rag_stats" in data and isinstance(data["rag_stats"], dict):
            data["rag_stats"] = RAGStats(**data["rag_stats"])
        if "quality_metrics" in data and isinstance(data["quality_metrics"], dict):
            data["quality_metrics"] = QualityMetrics(**data["quality_metrics"])
        # Reconstruct scene list
        if "scenes" in data and isinstance(data["scenes"], list):
            data["scenes"] = [
                SceneReport(**s) if isinstance(s, dict) else s for s in data["scenes"]
            ]
        # Reconstruct key moments inside scenes
        for s in data["scenes"]:
            if hasattr(s, "key_moments") and s.key_moments:
                s.key_moments = [
                    KeyMoment(**km) if isinstance(km, dict) else km for km in s.key_moments
                ]
        # Reconstruct chapters
        if "chapters" in data and isinstance(data["chapters"], list):
            data["chapters"] = [
                ChapterSummary(**c) if isinstance(c, dict) else c for c in data["chapters"]
            ]
        # Curation
        if "curation" in data and isinstance(data["curation"], dict):
            data["curation"] = CurationSummary(**data["curation"])

        return VideoReport(**data)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def summary_text(report: VideoReport) -> str:
        """Render a VideoReport as human-readable markdown.

        Args:
            report: The report to render.

        Returns:
            Markdown-formatted summary string.
        """
        lines = [
            "# Video Analysis Report",
            "",
            f"**Schema**: v{report.schema_version} | **Generated**: {report.generated_at}",
            "",
            "## 📹 Video",
            f"- **ID**: {report.video.video_id or 'N/A'}",
            f"- **Title**: {report.video.title or 'N/A'}",
            f"- **Duration**: {report.video.duration:.1f}s ({_fmt_duration(report.video.duration)})",
            f"- **Resolution**: {report.video.width}x{report.video.height}",
            f"- **FPS**: {report.video.fps:.2f}",
            f"- **File size**: {_fmt_size(report.video.file_size)}",
            f"- **Pipeline**: v{report.video.pipeline_version} in {report.video.processing_time_seconds:.1f}s",
            "",
            "## ⏱ Timeline",
            f"- **Scenes**: {report.timeline.num_scenes}",
            f"- **Frames indexed**: {report.timeline.num_frames}",
            f"- **Transcript duration**: {report.timeline.total_transcript_duration:.1f}s",
            f"- **Mean scene**: {report.timeline.mean_scene_duration:.1f}s "
            f"(min: {report.timeline.min_scene_duration:.1f}s, "
            f"max: {report.timeline.max_scene_duration:.1f}s)",
            f"- **Scene boundaries**: {len(report.timeline.scene_boundaries)} detected",
            "",
            "## 📝 Transcript",
            f"- **Segments**: {report.transcript.total_segments}",
            f"- **Total words**: {report.transcript.total_words:,}",
            f"- **Speakers**: {report.transcript.speaker_count}",
        ]

        if report.transcript.speakers:
            for sp, data in report.transcript.speakers.items():
                pct = data.get("word_count_pct", 0.0) * 100
                lines.append(
                    f"  - **{sp}**: {data.get('segment_count', 0)} segments, "
                    f"{data.get('total_words', 0)} words ({pct:.0f}%)"
                )

        if report.transcript.silent_periods:
            lines.append(f"- **Silent periods**: {len(report.transcript.silent_periods)} gaps > 2s")

        lines.extend(
            [
                "",
                "## 🎯 Objects",
                f"- **Unique objects**: {len(report.objects.unique_objects)}",
                f"- **Total detections**: {report.objects.total_detections}",
            ]
        )

        if report.objects.top_objects:
            lines.append("- **Top objects**:")
            for name, count, scenes in report.objects.top_objects[:_MAX_TOP_ENTITIES]:
                lines.append(f"  - {name}: {count} detections")

        lines.extend(
            [
                "",
                "## 🔍 RAG Index",
                f"- **Total chunks**: {report.rag_stats.total_chunks}",
            ]
        )
        if report.rag_stats.chunk_types:
            for ct, count in sorted(report.rag_stats.chunk_types.items()):
                lines.append(f"  - {ct}: {count}")
        lines.append(f"- **Embedding model**: {report.rag_stats.embedding_model}")
        lines.append(f"- **Scene graph**: {'yes' if report.rag_stats.has_scene_graph else 'no'}")

        if report.scenes:
            lines.extend(
                [
                    "",
                    "## 🎬 Scene Breakdown",
                ]
            )
            for s in report.scenes[:_MAX_TOP_SCENES]:
                lines.append(
                    f"### Scene {s.scene_id} ({_fmt_duration(s.start_time)} - {_fmt_duration(s.end_time)})"
                )
                if s.description:
                    lines.append(f"{s.description[:_MAX_DESCRIPTION_PREVIEW]}")
                if s.objects:
                    lines.append(f"Objects: {', '.join(s.objects[:_MAX_TOP_ENTITIES])}")
                lines.append("")

            if len(report.scenes) > _MAX_TOP_SCENES:
                lines.append(f"*... and {len(report.scenes) - _MAX_TOP_SCENES} more scenes*")

        lines.append("---")
        lines.append(f"*Report generated by video-analysis v{report.video.pipeline_version}*")

        return "\n".join(lines)

    @staticmethod
    def to_chunk_context(report: VideoReport) -> str:
        """Format a report for injection into RAG context for LLM queries.

        Returns a compact, LLM-friendly text representation of the most
        important facts from the report.
        """
        lines = [
            "## Video Context (from structured report)",
            f"- Video: **{report.video.title or report.video.video_id}**",
            f"- Duration: {_fmt_duration(report.video.duration)}",
            f"- Scenes: {report.timeline.num_scenes}",
        ]
        if report.objects.top_objects:
            top = report.objects.top_objects[:5]
            lines.append(f"- Key objects: {', '.join(f'{n} ({c}x)' for n, c, _ in top)}")
        if report.transcript.speakers:
            sp_summary = ", ".join(f"{s}" for s in report.transcript.speakers.keys())
            lines.append(f"- Speakers: {sp_summary}")
        if report.transcript.total_words:
            lines.append(f"- Transcript: {report.transcript.total_words} words")
        if report.transcript.key_phrases:
            lines.append(
                f"- Key themes: {', '.join(report.transcript.key_phrases[:_MAX_TOP_THEMES])}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_checksum(video_path: Path) -> str:
        """Compute a fast checksum of the first 64KB + file size."""
        try:
            h = hashlib.sha256()
            with open(video_path, "rb") as f:
                chunk = f.read(_CHECKSUM_CHUNK_SIZE)
                h.update(chunk)
            file_size = video_path.stat().st_size
            h.update(str(file_size).encode())
            return h.hexdigest()[:_CHECKSUM_HEX_LENGTH]
        except (OSError, PermissionError) as exc:
            logger.warning("Could not compute checksum for %s: %s", video_path, exc)
            return ""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    if seconds < 0:
        return "0:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes <= 0:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ===================================================================
# FastAPI Router
# ===================================================================


def create_report_router(
    report_generator: Optional[ReportGenerator] = None,
):
    """Create a FastAPI APIRouter with video report endpoints.

    Endpoints:
        GET /api/reports/{video_id}          — Full report as JSON
        GET /api/reports/{video_id}/summary  — Markdown summary
        GET /api/reports/compare             — Compare two video reports

    Args:
        report_generator: A ``ReportGenerator`` instance. If None, a
            default one will be created on first use.

    Returns:
        A configured ``APIRouter`` ready to be included in a FastAPI app.
    """
    from fastapi import APIRouter, HTTPException, Query
    from fastapi.responses import PlainTextResponse, Response

    router = APIRouter()

    # Shared generator (lazy initialised)
    _generator: Optional[ReportGenerator] = report_generator

    def _get_gen() -> ReportGenerator:
        nonlocal _generator
        if _generator is None:
            _generator = ReportGenerator()
        return _generator

    @router.get("/api/reports/{video_id}")
    async def get_report(video_id: str):
        """Return the full ``VideoReport`` as JSON."""
        try:
            gen = _get_gen()
            report_path = gen._config.data_dir / "reports" / f"{video_id}.json"
            if report_path.exists():
                report = gen.load(report_path)
            else:
                report = gen.from_video_id(video_id)
            return Response(
                content=gen.to_json(report),
                media_type="application/json",
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        except Exception as exc:
            logger.error("Error generating report for video_id=%s: %s", video_id, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/api/reports/{video_id}/summary")
    async def get_report_summary(video_id: str):
        """Return the video report as a human-readable markdown summary."""
        try:
            gen = _get_gen()
            report_path = gen._config.data_dir / "reports" / f"{video_id}.json"
            if report_path.exists():
                report = gen.load(report_path)
            else:
                report = gen.from_video_id(video_id)
            return PlainTextResponse(
                content=gen.summary_text(report),
                media_type="text/markdown",
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        except Exception as exc:
            logger.error("Error generating summary for video_id=%s: %s", video_id, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/api/reports/compare")
    async def compare_reports(
        video_ids: str = Query(..., description="Comma-separated video IDs"),
    ):
        """Compare two or more video reports side-by-side.

        Returns a JSON object with a comparison table of key metrics
        across the requested videos.
        """
        ids = [v.strip() for v in video_ids.split(",") if v.strip()]
        if len(ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="Provide at least two video_ids (comma-separated)",
            )

        gen = _get_gen()
        reports: List[VideoReport] = []
        for vid in ids:
            try:
                report_path = gen._config.data_dir / "reports" / f"{vid}.json"
                if report_path.exists():
                    reports.append(gen.load(report_path))
                else:
                    reports.append(gen.from_video_id(vid))
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Video {vid} not found")

        comparison = _build_comparison(reports, ids)
        return comparison

    return router


def _build_comparison(reports: List[VideoReport], video_ids: List[str]) -> Dict[str, Any]:
    """Build a comparison dict across multiple video reports."""
    comparison: Dict[str, Any] = {
        "video_ids": video_ids,
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
    }

    metric_defs = [
        ("duration", "Duration", "s"),
        ("num_scenes", "Scenes", ""),
        ("num_frames", "Frames", ""),
        ("total_segments", "Transcript Segments", ""),
        ("speaker_count", "Speakers", ""),
        ("total_detections", "Object Detections", ""),
        ("total_text_frames", "OCR Frames", ""),
        ("total_faces", "Faces", ""),
    ]

    for key, label, unit in metric_defs:
        vals: List[Dict[str, Any]] = []
        for i, report in enumerate(reports):
            val = _get_metric(report, key)
            vals.append({"video_id": video_ids[i], "value": val, "unit": unit})
        comparison["metrics"][label] = vals

    # Scene counts
    comparison["scene_counts"] = {vid: len(r.scenes) for vid, r in zip(video_ids, reports)}

    # Common objects
    all_obj_sets = [set(r.objects.unique_objects) for r in reports]
    if len(all_obj_sets) >= 2:
        common = set.intersection(*all_obj_sets) if all_obj_sets else set()
        comparison["common_objects"] = sorted(common)

    return comparison


def _get_metric(report: VideoReport, key: str) -> Any:
    """Extract a named metric from a VideoReport."""
    mapping: Dict[str, Any] = {
        "duration": report.video.duration,
        "num_scenes": report.timeline.num_scenes,
        "num_frames": report.timeline.num_frames,
        "total_segments": report.transcript.total_segments,
        "speaker_count": report.transcript.speaker_count,
        "total_detections": report.objects.total_detections,
        "total_text_frames": report.ocr.total_text_frames,
        "total_faces": report.faces.total_faces,
    }
    return mapping.get(key, None)
