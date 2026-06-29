"""
Video Content Chaptering — Automatic topic segmentation & chapter generation.

Takes a video transcript and segments it into meaningful chapters/topics
using NLP-based topic segmentation (TextTiling algorithm from NLTK) and
LLM-powered chapter title generation.

Architecture:
  1. TRANSCRIPT SEGMENTATION — NLTK TextTiling splits the full transcript
     into topically-coherent segments based on lexical score changes.
  2. CHAPTER TITLE GENERATION — The LLM (via Hermes CLI) generates a
     concise, descriptive title and optional summary for each chapter.
  3. CHAPTER INDEX STORAGE — Chapters are stored alongside the VideoIndex
     and can be retrieved for structured video browsing, agent-generated
     reports, and UI display.
  4. UI INTEGRATION — A 'Chapters' tab in Gradio displays the table of
     contents with clickable timestamps (future: integrated into agent).

Usage:
    from video_analysis.chapters import ChapterGenerator

    chapters = ChapterGenerator()
    result = chapters.segment_transcript(
        transcript_segments=[{"start": 0.0, "text": "Hello and welcome..."}, ...],
        video_id="my_video",
    )
    # result has .segments (list[ChapterSegment]) and .chapters (list[Chapter])
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_analysis.config import Config

logger = logging.getLogger(__name__)

# TextTiling algorithm defaults (NLTK)
_TEXTTILE_PSEUDO_SENTENCE_SIZE = 200
_TEXTTILE_BLOCK_COMPARISON_SIZE = 40

# Segmentation thresholds
_MIN_TEXT_CHARS_FOR_SEGMENT = 50
_MIN_WORDS_FOR_TEXTTILING = 200
_CHAPTER_WORDS_PER_GROUP = 150
_DEFAULT_TARGET_CHAPTERS = 6
_MAX_CHAPTERS_DEFAULT = 12
_MIN_CHAPTERS_DEFAULT = 2
_MIN_WORDS_FOR_LLM_TITLE = 30

# LLM chapter title generation
_LLM_MAX_INPUT_CHARS = 2000
_LLM_TITLE_SANITY_MAX = 120
_LLM_TITLE_MAX_CHARS = 60

# Heuristic title fallback
_HEURISTIC_SENTENCE_MIN_LEN = 20
_HEURISTIC_SENTENCE_MAX_LEN = 200
_HEURISTIC_TITLE_TRUNCATE = 57

# Report generation
_CHAPTER_PREVIEW_MAX_CHARS = 200


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ChapterSegment:
    """A single transcript segment with chapter assignment."""

    start: float  # timestamp in seconds
    end: float  # timestamp in seconds
    text: str
    chapter_index: int = -1  # assigned chapter index (-1 = unassigned)
    speaker: Optional[str] = None


@dataclass
class Chapter:
    """A single chapter within a video."""

    title: str
    start_time: float  # seconds
    end_time: float  # seconds
    index: int  # chapter number (0-based)
    summary: str = ""  # LLM-generated one-line summary
    transcript_preview: str = ""  # first ~200 chars of chapter transcript
    word_count: int = 0


@dataclass
class ChapteringResult:
    """Result of the chapter generation process."""

    video_id: str
    chapters: List[Chapter]
    num_segments: int
    method: str  # "texttiling", "llm_only", "uniform"
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "chapters": [
                {
                    "title": c.title,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "index": c.index,
                    "summary": c.summary,
                    "transcript_preview": c.transcript_preview,
                    "word_count": c.word_count,
                }
                for c in self.chapters
            ],
            "num_segments": self.num_segments,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Chapter Generator
# ---------------------------------------------------------------------------


class ChapterGenerator:
    """Generates video chapters from transcript data.

    Uses NLTK TextTiling for topic segmentation (when available) with
    a pure-Python fallback, then generates chapter titles via LLM.
    """

    def __init__(self, config: Optional[Config] = None, llm=None):
        self.config = config or Config()
        self._llm = llm  # optional LLMProvider instance
        self._texttiler = None  # lazy-loaded

    def _init_texttiler(self) -> bool:
        """Initialize the NLTK TextTiling tokenizer.

        Returns True if successful, False if NLTK is unavailable.
        """
        if self._texttiler is not None:
            return True

        try:
            from nltk.tokenize import TextTilingTokenizer

            # Download required NLTK data if needed
            try:
                import nltk

                nltk.data.find("tokenizers/punkt")
            except (LookupError, OSError):
                try:
                    import nltk

                    nltk.download("punkt", quiet=True)
                    nltk.download("punkt_tab", quiet=True)
                except Exception:
                    pass

            # Use reasonable defaults — 200 word pseudo-sentence size,
            # which works well for transcript text at ~150 wpm.
            self._texttiler = TextTilingTokenizer(
                k=_TEXTTILE_PSEUDO_SENTENCE_SIZE,
                w=_TEXTTILE_BLOCK_COMPARISON_SIZE,
                stopwords=None,  # use NLTK default stopwords
            )
            logger.info("NLTK TextTilingTokenizer initialized.")
            return True
        except ImportError:
            logger.warning(
                "NLTK not installed — will use alternative chaptering methods. "
                "Install with: pip install nltk"
            )
            return False
        except Exception as exc:
            logger.warning(f"Could not init TextTilingTokenizer: {exc}")
            return False

    def _texttile_segment(self, full_text: str) -> List[Tuple[int, int]]:
        """Run TextTiling on a full transcript string.

        Args:
            full_text: The complete transcript text (word-level timestamps
                       already flattened to readable paragraphs).

        Returns:
            List of (start_char, end_char) tuples indicating topic boundaries
            found by TextTiling.
        """
        tiler = self._texttiler
        if tiler is None:
            return []

        try:
            segments = tiler.tokenize(full_text)
            # segments is a list of strings — we need char ranges
            char_ranges: List[Tuple[int, int]] = []
            pos = 0
            for seg_text in segments:
                seg_len = len(seg_text)
                end = pos + seg_len
                # Skip very short segments (<50 chars = noise)
                if seg_len >= _MIN_TEXT_CHARS_FOR_SEGMENT:
                    char_ranges.append((pos, end))
                pos = end + 1  # +1 for whitespace
            return char_ranges
        except Exception as exc:
            logger.warning(f"TextTiling failed: {exc}")
            return []

    def _segment_uniform(
        self, transcript: List[ChapterSegment], target_chapters: int = _DEFAULT_TARGET_CHAPTERS
    ) -> List[List[ChapterSegment]]:
        """Fallback: divide transcript into uniform time-based segments.

        Used when TextTiling is unavailable or fails. Divides the total
        duration into equal-length buckets and assigns segments accordingly.

        Args:
            transcript: List of transcript segments with timestamps.
            target_chapters: Desired number of chapters.

        Returns:
            List of segment groups, each group becoming a chapter.
        """
        if not transcript:
            return []

        total_duration = transcript[-1].end
        if total_duration <= 0:
            return [transcript]

        chapter_duration = total_duration / max(target_chapters, 1)
        groups: List[List[ChapterSegment]] = []
        current_group: List[ChapterSegment] = []
        current_start = 0.0

        for seg in transcript:
            if seg.start >= current_start + chapter_duration and current_group:
                groups.append(current_group)
                current_group = []
                current_start += chapter_duration
            current_group.append(seg)

        if current_group:
            groups.append(current_group)

        return groups

    def _segment_by_scene_boundaries(
        self,
        transcript: List[ChapterSegment],
        scene_times: Optional[List[float]] = None,
    ) -> List[List[ChapterSegment]]:
        """Segment transcript using scene detection boundaries.

        This provides better chapter boundaries than uniform segmentation
        when scene detection metadata is available.

        Args:
            transcript: List of transcript segments.
            scene_times: List of scene boundary timestamps in seconds.

        Returns:
            List of segment groups.
        """
        if not scene_times or len(scene_times) < 2:
            return self._segment_uniform(transcript, target_chapters=_DEFAULT_TARGET_CHAPTERS)

        groups: List[List[ChapterSegment]] = []
        boundary_idx = 0
        current_group: List[ChapterSegment] = []

        for seg in transcript:
            # Check if we've passed the next scene boundary
            while (
                boundary_idx < len(scene_times) - 1
                and seg.start >= scene_times[boundary_idx + 1]
            ):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                boundary_idx += 1
            current_group.append(seg)

        if current_group:
            groups.append(current_group)

        return groups

    def _get_llm(self):
        """Lazy-load the LLM provider."""
        if self._llm is None:
            from video_analysis.llm_provider import get_llm_provider, LLMProviderConfig

            cfg = LLMProviderConfig(
                provider=os.environ.get("LLM_PROVIDER", "hermes"),
                api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("OPENAI_MODEL", "qwen2.5"),
                max_tokens=512,
                temperature=0.3,
                timeout=30,
                hermes_model=self.config.llm_model,
                hermes_max_tokens=512,
            )
            self._llm = get_llm_provider(cfg)
            logger.info("ChapterGenerator using LLM provider: %s", self._llm.name)
        return self._llm

    def _generate_title_via_llm(
        self, chapter_text: str, chapter_index: int, total_chapters: int
    ) -> Tuple[str, str]:
        """Generate a chapter title using the LLM provider.

        Falls back to a descriptive title using simple heuristics.

        Args:
            chapter_text: The full transcript text of this chapter.
            chapter_index: 0-based chapter index.
            total_chapters: Total number of chapters.

        Returns:
            Tuple of (title: str, summary: str).
        """
        # Truncate input to avoid overflowing
        truncated_text = chapter_text[:_LLM_MAX_INPUT_CHARS]

        prompt = (
            f"You are analyzing a video transcript. This is chapter "
            f"{chapter_index + 1} of {total_chapters}.\n\n"
            f"Transcript segment:\n{truncated_text}\n\n"
            f"Generate:\n"
            f"1. A short, descriptive CHAPTER TITLE (max {_LLM_TITLE_MAX_CHARS} chars, no quotes)\n"
            f"2. A one-sentence CHAPTER SUMMARY (max 150 chars)\n\n"
            f'Format: {{"title": "...", "summary": "..."}}\n'
            f"Return only valid JSON, nothing else."
        )

        llm = self._get_llm()
        parsed = llm.structured_chat(
            prompt=prompt,
            temperature=0.3,
            max_tokens=512,
            timeout=30,
        )
        if parsed:
            title = parsed.get("title", "").strip()
            summary = parsed.get("summary", "").strip()
            if title:
                title = title.strip("\"' ")
                summary = summary.strip("\"' ")
                if len(title) <= _LLM_TITLE_SANITY_MAX:
                    return title, summary

        # Fallback: generate a heuristic title
        return self._generate_heuristic_title(chapter_text, chapter_index)

    def _generate_heuristic_title(
        self, chapter_text: str, chapter_index: int
    ) -> Tuple[str, str]:
        """Generate a simple heuristic title from the first sentence.

        Fallback when LLM is unavailable.
        """
        # Try to extract the first meaningful sentence
        sentences = re.split(r"[.!?\n]+", chapter_text.strip())
        first_sentence = ""
        for s in sentences:
            s = s.strip()
            if _HEURISTIC_SENTENCE_MIN_LEN < len(s) < _HEURISTIC_SENTENCE_MAX_LEN:
                first_sentence = s
                break

        if first_sentence:
            title = (
                first_sentence[:_HEURISTIC_TITLE_TRUNCATE] + "…"
                if len(first_sentence) > _LLM_TITLE_MAX_CHARS
                else first_sentence
            )
        else:
            title = f"Chapter {chapter_index + 1}"

        return title, ""

    def _build_transcript_paragraph(
        self,
        segments: List[ChapterSegment],
    ) -> str:
        """Build a readable paragraph from timestamped segments.

        Merges short contiguous segments and formats as readable text.
        """
        lines = []
        for seg in segments:
            speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
            text = seg.text.strip()
            if text:
                lines.append(f"{speaker_prefix}{text}")
        return " ".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_transcript(
        self,
        transcript_segments: List[Dict[str, Any]],
        video_id: str = "unknown",
        scene_boundaries: Optional[List[float]] = None,
        max_chapters: int = 12,
        min_chapters: int = 2,
        use_llm_titles: bool = True,
    ) -> ChapteringResult:
        """Segment a video transcript into chapters.

        Args:
            transcript_segments: List of dicts with keys:
                - start (float): segment start time in seconds
                - end (float): segment end time in seconds
                - text (str): segment text content
                - speaker (str, optional): speaker label
            video_id: The video ID for metadata.
            scene_boundaries: Optional list of scene boundary timestamps
                (from PySceneDetect) to use as anchor points.
            max_chapters: Maximum number of chapters to generate.
            min_chapters: Minimum number of chapters.
            use_llm_titles: If True, use LLM for title generation.

        Returns:
            ChapteringResult with the generated chapters.
        """
        import time

        start_time = time.time()

        if not transcript_segments:
            return ChapteringResult(
                video_id=video_id,
                chapters=[],
                num_segments=0,
                method="none",
                duration_seconds=0.0,
                error="No transcript segments provided.",
            )

        # Convert to ChapterSegment objects
        segments: List[ChapterSegment] = []
        for seg in transcript_segments:
            segments.append(
                ChapterSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", "").strip(),
                    speaker=seg.get("speaker"),
                )
            )

        # Remove empty segments
        segments = [s for s in segments if s.text]

        if not segments:
            return ChapteringResult(
                video_id=video_id,
                chapters=[],
                num_segments=0,
                method="none",
                duration_seconds=time.time() - start_time,
                error="All transcript segments were empty.",
            )

        total_words = sum(len(s.text.split()) for s in segments)
        total_duration = segments[-1].end - segments[0].start

        # Strategy 1: Try TextTiling-based segmentation
        segment_groups: List[List[ChapterSegment]] = []
        method = "uniform"

        if total_words >= _MIN_WORDS_FOR_TEXTTILING:
            tiler_ok = self._init_texttiler()
            if tiler_ok:
                full_text = self._build_transcript_paragraph(segments)
                char_ranges = self._texttile_segment(full_text)

                if len(char_ranges) >= min_chapters:
                    # Map char ranges back to segment indices
                    segment_groups = self._map_char_ranges_to_segments(
                        segments, full_text, char_ranges
                    )
                    method = "texttiling"

        # Strategy 2: Scene boundary segmentation
        if not segment_groups and scene_boundaries:
            segment_groups = self._segment_by_scene_boundaries(
                segments, scene_boundaries
            )
            method = "scene_boundaries"

        # Strategy 3: Uniform time-based fallback
        if not segment_groups:
            target = min(max_chapters, max(min_chapters, total_words // _CHAPTER_WORDS_PER_GROUP))
            segment_groups = self._segment_uniform(segments, target_chapters=target)
            method = "uniform"

        # Limit chapters
        if len(segment_groups) > max_chapters:
            segment_groups = self._merge_groups(segment_groups, max_chapters)

        # Build Chapter objects
        chapters: List[Chapter] = []
        cumulative_start = segments[0].start if segments else 0.0

        for idx, group in enumerate(segment_groups):
            if not group:
                continue

            chapter_start = group[0].start
            chapter_end = group[-1].end
            chapter_text = self._build_transcript_paragraph(group)
            chapter_words = sum(len(s.text.split()) for s in group)

            title = f"Chapter {idx + 1}"
            summary = ""

            if use_llm_titles and chapter_words >= _MIN_WORDS_FOR_LLM_TITLE:
                gen_title, gen_summary = self._generate_title_via_llm(
                    chapter_text, idx, len(segment_groups)
                )
                if gen_title:
                    title = gen_title
                if gen_summary:
                    summary = gen_summary
            else:
                gen_title, gen_summary = self._generate_heuristic_title(
                    chapter_text, idx
                )
                title = gen_title
                summary = gen_summary

            # Extract a preview
            preview = chapter_text[:_CHAPTER_PREVIEW_MAX_CHARS].replace("\n", " ").strip()

            chapters.append(
                Chapter(
                    title=title,
                    start_time=chapter_start,
                    end_time=chapter_end,
                    index=idx,
                    summary=summary,
                    transcript_preview=preview,
                    word_count=chapter_words,
                )
            )

        elapsed = time.time() - start_time

        return ChapteringResult(
            video_id=video_id,
            chapters=chapters,
            num_segments=len(segments),
            method=method,
            duration_seconds=elapsed,
        )

    def _map_char_ranges_to_segments(
        self,
        segments: List[ChapterSegment],
        full_text: str,
        char_ranges: List[Tuple[int, int]],
    ) -> List[List[ChapterSegment]]:
        """Map TextTiling character ranges back to transcript segments.

        Since we build full_text from segments with spaces, we need to
        find which segments fall into which char range.
        """
        # Build char-to-segment mapping
        char_to_seg: List[int] = []
        for seg_idx, seg in enumerate(segments):
            seg_words = seg.text.split()
            for w in seg_words:
                char_to_seg.append(seg_idx)
                char_to_seg.append(seg_idx)  # +1 for space
            # Add a space between segments
            char_to_seg.append(seg_idx)

        groups: List[List[ChapterSegment]] = []
        seen_segments: set = set()

        for start_char, end_char in char_ranges:
            group_indices: set = set()
            for i in range(start_char, min(end_char, len(char_to_seg))):
                seg_idx = char_to_seg[i]
                if seg_idx < len(segments):
                    group_indices.add(seg_idx)

            if not group_indices:
                continue

            group = [
                segments[idx]
                for idx in sorted(group_indices)
                if idx not in seen_segments
            ]
            for idx in group_indices:
                seen_segments.add(idx)

            if group:
                groups.append(group)

        # Add any remaining segments not covered by TextTiling
        remaining = [s for i, s in enumerate(segments) if i not in seen_segments]
        if remaining:
            if groups and remaining:
                # Merge remaining into the last chapter
                groups[-1].extend(remaining)
            elif remaining:
                groups.append(remaining)

        return groups

    def _merge_groups(
        self,
        groups: List[List[ChapterSegment]],
        target_count: int,
    ) -> List[List[ChapterSegment]]:
        """Merge adjacent groups until we reach target_count.

        Merges the smallest groups first to balance chapter sizes.
        """
        while len(groups) > target_count:
            # Find the smallest group (by segment count)
            smallest_idx = min(
                range(1, len(groups)),  # skip first group
                key=lambda i: len(groups[i]),
            )
            # Merge with previous group
            groups[smallest_idx - 1].extend(groups[smallest_idx])
            groups.pop(smallest_idx)
        return groups

    def generate_chapter_report(
        self,
        result: ChapteringResult,
        video_filename: str = "unknown",
    ) -> str:
        """Generate a markdown chapter report from a ChapteringResult.

        Args:
            result: The ChapteringResult from segment_transcript().
            video_filename: The video filename for the report header.

        Returns:
            Markdown-formatted chapter report string.
        """
        from video_analysis.models import format_timestamp

        if not result.chapters:
            return f"# Chapter Report: {video_filename}\n\n_No chapters generated._\n"

        lines = [
            f"# Chapter Report: {video_filename}",
            f"",
            f"- **Method**: {result.method}",
            f"- **Chapters**: {len(result.chapters)}",
            f"- **Transcript Segments**: {result.num_segments}",
            f"",
            f"---",
            f"",
        ]

        for ch in result.chapters:
            start_ts = format_timestamp(ch.start_time)
            end_ts = format_timestamp(ch.end_time)
            duration = ch.end_time - ch.start_time
            duration_str = (
                f"{int(duration // 60)}m {int(duration % 60)}s"
                if duration >= 60
                else f"{duration:.1f}s"
            )

            lines.append(f"## Chapter {ch.index + 1}: {ch.title}")
            lines.append(f"")
            lines.append(f"⏱ **{start_ts} → {end_ts}** ({duration_str})")
            if ch.summary:
                lines.append(f"📝 {ch.summary}")
            lines.append(f"📊 {ch.word_count} words in this chapter")
            lines.append(f"")
            lines.append(f"> {ch.transcript_preview}")
            lines.append(f"")

        return "\n".join(lines)

    def generate_agent_chapter_context(self, result: ChapteringResult) -> str:
        """Generate a concise chapter context string for the agent reasoning loop.

        Used by VideoUnderstandingAgent to provide chapter-aware context.

        Args:
            result: The ChapteringResult.

        Returns:
            Compact chapter description string.
        """
        from video_analysis.models import format_timestamp

        if not result.chapters:
            return "No chapters available."

        parts = [
            f"Video has {len(result.chapters)} chapters (segmented via {result.method}):"
        ]
        for ch in result.chapters:
            start_ts = format_timestamp(ch.start_time)
            parts.append(f'  Ch{ch.index + 1}: "{ch.title}" @ {start_ts}')

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Integration helper — extract transcript segments from RAG or VideoIndex
# ---------------------------------------------------------------------------


def extract_transcript_from_rag(
    rag_instance: Any,
    video_id: str,
) -> List[Dict[str, Any]]:
    """Extract transcript segments from the ChromaDB RAG index.

    Searches the collection for transcript-type chunks matching the
    given video_id and returns them as a list of {start, end, text, speaker}
    dicts, sorted by timestamp.

    Args:
        rag_instance: A VideoRAG instance with an initialized collection.
        video_id: The video ID to search for.

    Returns:
        Sorted list of transcript segment dicts.
    """
    segments: List[Dict[str, Any]] = []

    try:
        col = rag_instance.collection
        # Query all transcript chunks for this video
        results = col.get(
            where={"$and": [{"video_id": video_id}, {"chunk_type": "transcript"}]},
            include=["metadatas", "documents"],
        )

        if not results or not results["ids"]:
            # Fallback: try all chunk types for this video
            results = col.get(
                where={"video_id": video_id},
                include=["metadatas", "documents"],
            )

        if results and results["ids"]:
            for meta, doc in zip(results["metadatas"], results["documents"]):
                seg: Dict[str, Any] = {
                    "start": float(meta.get("start_time", meta.get("timestamp", 0.0))),
                    "end": float(meta.get("end_time", meta.get("timestamp", 0.0))),
                    "text": str(doc or meta.get("text", "")),
                    "speaker": meta.get("speaker"),
                }
                if seg["text"].strip():
                    segments.append(seg)

        # Sort by start time
        segments.sort(key=lambda s: s["start"])

    except Exception as exc:
        logger.warning(f"Could not extract transcript from RAG: {exc}")

    return segments
