"""
Real-time streaming video analysis pipeline.

Processes video in configurable time-window chunks, yielding incremental
results (scenes, transcript, objects) before the full video is processed.
Inspired by StreamingVLM (ICLR 2026) and ThinkStream (ECCV 2026).

Modes:
  - chunked_file   : Split an existing video into chunks, process each,
                     yield results incrementally (reduced latency to first result).
  - file_watch     : Watch a file being written (e.g. OBS live recording)
                     and process chunks as they become available.
  - segment_based  : Use FFmpeg segment mode for fine-grained boundaries.

Usage::

    from video_analysis.streaming import StreamingPipeline

    pipeline = StreamingPipeline()
    for result in pipeline.process_streaming("video.mp4", chunk_duration=30.0):
        print(f"Chunk {result.chunk_index}: {len(result.scenes)} scenes")
    final_index = pipeline.final_index()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

from video_analysis.config import Config
from video_analysis.models import (
    SceneInfo,
    TranscriptSegment,
    VideoIndex,
)

# Deferred import to avoid circular import at module load time
# (rag.py imports from config/models, which don't import streaming)
from video_analysis.rag import VideoRAG  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class StreamingChunkResult:
    """Result from processing a single streaming chunk of a video.

    Attributes:
        chunk_index: Zero-based index of this chunk in the sequence.
        start_time: Start time (seconds) of the chunk in the original video.
        end_time: End time (seconds) of the chunk in the original video.
        duration: Actual duration (seconds) of this chunk.
        scenes: List of scenes detected within this chunk.
        transcript_segments: List of transcript segments within this chunk.
        full_transcript: Concatenated transcript text for this chunk.
        objects_found: Unique object labels detected in this chunk.
        has_video: False for audio-only chunks (e.g. silent regions).
        metadata: Arbitrary metadata dict for extensibility.
    """

    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    scenes: List[SceneInfo] = field(default_factory=list)
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    full_transcript: str = ""
    objects_found: List[str] = field(default_factory=list)
    has_video: bool = True
    metadata: dict = field(default_factory=dict)


def _ffprobe_duration(video_path: Path) -> Optional[float]:
    """Get total duration (seconds) of a video file via FFprobe.

    Returns None if the file does not exist or FFprobe fails.
    """
    if not video_path.exists():
        logger.warning(f"Video file not found: {video_path}")
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"FFprobe failed for {video_path}: {result.stderr.strip()}")
            return None
        data = json.loads(result.stdout)
        duration_str = data.get("format", {}).get("duration")
        if duration_str is not None:
            return float(duration_str)
        logger.warning(f"No duration found in ffprobe output for {video_path}")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"FFprobe error for {video_path}: {e}")
        return None


class StreamingPipeline:
    """Process video in streaming chunks, yielding incremental results.

    The streaming pipeline delegates all GPU-heavy processing to the existing
    :class:`VideoPipeline` — it does NOT load GPU models itself. Each chunk
    is extracted as a temporary video segment via FFmpeg copy-mode, then
    processed by the existing pipeline.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._pipeline: Optional[Any] = None
        self._temp_dir: Path = self.config.data_dir / "tmp"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._all_scenes: List[SceneInfo] = []
        self._all_transcript_segments: List[TranscriptSegment] = []
        self._all_transcript_text: List[str] = []
        self._all_objects: set = set()
        self._processed_chunks: int = 0
        self._video_id: Optional[str] = None
        self._chunk_results: List[StreamingChunkResult] = []

    # ------------------------------------------------------------------
    # Lazy pipeline initialisation
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        """Get or create the underlying VideoPipeline (lazy init)."""
        if self._pipeline is None:
            # Deferred import to avoid loading GPU models at module level
            from video_analysis.pipeline import VideoPipeline

            self._pipeline = VideoPipeline(self.config)
        return self._pipeline

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    def process_streaming(
        self,
        video_path: str,
        chunk_duration: float = 30.0,
        overlap: float = 2.0,
        incremental_index: bool = True,
        max_chunks: Optional[int] = None,
    ) -> Generator[StreamingChunkResult, None, VideoIndex]:
        """Process an existing video file in streaming chunks.

        Yields a :class:`StreamingChunkResult` after each chunk is processed,
        then returns the final merged :class:`VideoIndex`.

        Args:
            video_path: Path to the video file.
            chunk_duration: Seconds per chunk (default 30).
            overlap: Seconds of overlap between adjacent chunks for
                context continuity (default 2).
            incremental_index: If True, index each chunk to ChromaDB as
                it's processed (default True).
            max_chunks: Optional limit on the number of chunks to process.
                None = no limit (process entire video).

        Yields:
            StreamingChunkResult for each processed chunk.

        Returns:
            The final merged VideoIndex covering all chunks.
        """
        video_path_obj = Path(video_path)
        if not video_path or not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_id = video_path_obj.stem
        self._video_id = video_id

        # Get total duration
        duration = _ffprobe_duration(video_path_obj)
        if duration is None or duration <= 0:
            raise ValueError(
                f"Cannot determine duration for {video_path}. "
                f"File may be empty or corrupt."
            )

        # Compute segment boundaries
        segments = self._segment_video(video_path_obj, chunk_duration, overlap)

        if max_chunks is not None:
            segments = segments[:max_chunks]

        for chunk_idx, (start, end) in enumerate(segments):
            if start >= duration:
                break

            actual_end = min(end, duration)
            result = self._process_segment(
                video_path_obj, start, actual_end, chunk_idx, video_id
            )
            if result is not None:
                # Accumulate
                self._chunk_results.append(result)
                self._all_scenes.extend(result.scenes)
                self._all_transcript_segments.extend(result.transcript_segments)
                self._all_transcript_text.append(result.full_transcript)
                self._all_objects.update(result.objects_found)
                self._processed_chunks += 1

                # Optionally index incrementally
                if incremental_index:
                    self._index_chunk(result)

                yield result

            # Stop if we've exhausted the video
            if actual_end >= duration:
                break

        # Build final merged index
        final_index = self._build_final_index(video_id, video_path_obj, duration)

        # Index the final merged index if we didn't do incremental indexing
        if not incremental_index:
            self._index_final(final_index)

        return final_index

    def process_live(
        self,
        source: str,
        chunk_duration: float = 10.0,
        incremental_index: bool = True,
        poll_interval: float = 1.0,
    ) -> Generator[StreamingChunkResult, None, None]:
        """Watch a file being written (e.g. OBS recording) and process
        chunks as they become available.

        This is an infinite generator — callers should ``break`` when done.

        Args:
            source: Path to a file being written, or RTSP/RTMP URL.
            chunk_duration: Seconds per processing chunk (default 10).
            incremental_index: If True, index incrementally (default True).
            poll_interval: Seconds between file-size checks (default 1).

        Yields:
            StreamingChunkResult for each completed chunk.
        """
        from pathlib import Path as _Path

        source_path = _Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Live source not found: {source}")

        video_id = source_path.stem
        self._video_id = video_id

        # For file_watch mode, poll for file size changes
        last_size = source_path.stat().st_size
        last_end: float = 0.0
        chunk_idx: int = 0

        # Use a local VideoPipeline to avoid holding GPU models in this generator
        pipeline = self._get_pipeline()

        while True:
            time.sleep(poll_interval)

            if not source_path.exists():
                logger.warning(f"Live source disappeared: {source}")
                break

            current_size = source_path.stat().st_size
            if current_size <= last_size:
                # File hasn't grown — skip this polling cycle
                continue

            # Probe the current duration
            duration = _ffprobe_duration(source_path)
            if duration is None or duration <= last_end:
                continue

            # If we have at least chunk_duration seconds of new content
            if duration - last_end >= chunk_duration:
                chunk_start = last_end
                chunk_end = last_end + chunk_duration

                # Process this chunk
                result = self._process_segment(
                    source_path, chunk_start, chunk_end, chunk_idx, video_id
                )
                if result is not None:
                    self._chunk_results.append(result)
                    self._all_scenes.extend(result.scenes)
                    self._all_transcript_segments.extend(result.transcript_segments)
                    self._all_transcript_text.append(result.full_transcript)
                    self._all_objects.update(result.objects_found)
                    self._processed_chunks += 1

                    if incremental_index:
                        self._index_chunk(result)

                    yield result

                last_end = chunk_end
                chunk_idx += 1

            last_size = current_size

    # ------------------------------------------------------------------
    # Segment computation
    # ------------------------------------------------------------------

    def _segment_video(
        self, video_path: Path, chunk_duration: float, overlap: float
    ) -> List[Tuple[float, float]]:
        """Generate non-overlapping segment timestamps using FFprobe.

        Extracts full duration first, then divides into chunk_duration
        segments with the given overlap.

        Returns:
            List of (start_time, end_time) tuples.
        """
        duration = _ffprobe_duration(video_path)
        if duration is None or duration <= 0:
            logger.warning(
                f"Cannot segment video (duration={duration}) — returning empty list"
            )
            return []

        segments: List[Tuple[float, float]] = []
        step = chunk_duration
        # Grid boundaries: 0, step, 2*step, 3*step, ...
        grid_pos = 0.0
        while grid_pos < duration:
            chunk_start = grid_pos
            chunk_end = min(grid_pos + step, duration)
            # Apply overlap for all but the first segment
            if len(segments) > 0 and overlap > 0:
                chunk_start = max(chunk_start - overlap, 0.0)
            segments.append((chunk_start, chunk_end))
            grid_pos += step

        return segments

    # ------------------------------------------------------------------
    # Segment processing
    # ------------------------------------------------------------------

    def _process_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        chunk_index: int,
        video_id: str,
    ) -> Optional[StreamingChunkResult]:
        """Extract a segment from the video as a temp file, run the pipeline
        on it, and return the result.

        Creates a temp segment via FFmpeg stream copy::

            ffmpeg -ss START -to END -i VIDEO -c copy segment.mp4

        Then processes it through the existing VideoPipeline methods.

        Args:
            video_path: Path to the original video.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            chunk_index: Zero-based chunk index.
            video_id: Video ID for naming.

        Returns:
            StreamingChunkResult, or None if the segment could not be processed.
        """
        duration = end_time - start_time
        if duration <= 0:
            logger.warning(
                f"Segment {chunk_index}: invalid duration {duration:.2f}s — skipping"
            )
            return None

        # Create temp segment file
        segment_filename = (
            f"{video_id}_chunk_{chunk_index:04d}_{start_time:.1f}-{end_time:.1f}.mp4"
        )
        segment_path = self._temp_dir / segment_filename

        try:
            # Use FFmpeg stream copy for speed — no re-encoding
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-i",
                str(video_path),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(segment_path),
            ]
            logger.debug(
                f"Extracting segment {chunk_index}: [{start_time:.1f}s - {end_time:.1f}s]"
            )
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # generous timeout for large segments
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"FFmpeg segment extraction failed for chunk {chunk_index}: "
                f"{e.stderr.strip() or e}"
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout for segment {chunk_index}")
            return None

        if not segment_path.exists() or segment_path.stat().st_size == 0:
            logger.warning(f"Segment {chunk_index} produced empty file — skipping")
            return None

        # Process the segment through the existing pipeline
        try:
            pipeline = self._get_pipeline()
            segment_index = pipeline.process(str(segment_path))

            # Extract objects found in this chunk
            objects_found: set = set()
            for scene in segment_index.scenes:
                for frame in scene.key_frames:
                    if frame.objects:
                        for obj in frame.objects:
                            label = obj.get("label", "")
                            if label:
                                objects_found.add(label)

            # Build result
            result = StreamingChunkResult(
                chunk_index=chunk_index,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                scenes=segment_index.scenes,
                transcript_segments=segment_index.transcript,
                full_transcript=segment_index.full_transcript,
                objects_found=sorted(objects_found),
                has_video=True,
                metadata={
                    "segment_file": str(segment_path),
                    "video_id": video_id,
                    "original_path": str(video_path),
                },
            )

            # Cleanup pipeline GPU models after each chunk to stay within VRAM
            pipeline.cleanup()

            return result

        except Exception as e:
            logger.error(f"Pipeline processing failed for segment {chunk_index}: {e}")
            # Try to clean up regardless
            try:
                pipeline = self._get_pipeline()
                pipeline.cleanup()
            except Exception:
                pass
            return None
        finally:
            # Remove the temp segment file
            try:
                if segment_path.exists():
                    segment_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove temp segment {segment_path}: {e}")

    # ------------------------------------------------------------------
    # Incremental indexing
    # ------------------------------------------------------------------

    def _index_chunk(self, result: StreamingChunkResult) -> None:
        """Index a single chunk result into ChromaDB incrementally.

        Builds a lightweight VideoIndex from the chunk's data and delegates
        to VideoRAG.index_video().

        Args:
            result: The chunk result to index.
        """
        try:
            rag = VideoRAG(self.config)
            video_id = result.metadata.get("video_id", f"stream_{uuid.uuid4().hex[:8]}")

            # Build a minimal VideoIndex for this chunk
            chunk_index = VideoIndex(
                video_id=video_id,
                filename=f"{video_id}_chunk_{result.chunk_index:04d}",
                duration=result.duration,
                filepath=result.metadata.get("segment_file", ""),
                scenes=result.scenes,
                transcript=result.transcript_segments,
                full_transcript=result.full_transcript,
            )
            rag.index_video(chunk_index)
            logger.debug(
                f"Indexed chunk {result.chunk_index} "
                f"({len(result.scenes)} scenes, {len(result.transcript_segments)} transcript segs)"
            )
        except Exception as e:
            logger.warning(f"Failed to index chunk {result.chunk_index}: {e}")

    def _index_final(self, index: VideoIndex) -> None:
        """Index the final merged VideoIndex into ChromaDB.

        Args:
            index: The final merged VideoIndex.
        """
        try:
            rag = VideoRAG(self.config)
            rag.index_video(index)
            logger.info(
                f"Indexed final merged index for {index.video_id} "
                f"({len(index.scenes)} scenes total)"
            )
        except Exception as e:
            logger.warning(f"Failed to index final index: {e}")

    # ------------------------------------------------------------------
    # Final index assembly
    # ------------------------------------------------------------------

    def _build_final_index(
        self, video_id: str, video_path: Path, duration: float
    ) -> VideoIndex:
        """Build a merged VideoIndex from all accumulated chunk results.

        Args:
            video_id: Video identifier.
            video_path: Path to the original video file.
            duration: Total video duration in seconds.

        Returns:
            A single VideoIndex combining all chunks.
        """
        return VideoIndex(
            video_id=video_id,
            filename=video_path.name,
            duration=duration,
            filepath=str(video_path),
            scenes=self._all_scenes,
            transcript=self._all_transcript_segments,
            full_transcript=" ".join(self._all_transcript_text),
        )

    def final_index(self) -> Optional[VideoIndex]:
        """Return the accumulated final VideoIndex after streaming.

        Must be called after :meth:`process_streaming` completes.

        Returns:
            The merged VideoIndex, or None if no chunks were processed.
        """
        if not self._all_scenes and not self._all_transcript_segments:
            return None
        return VideoIndex(
            video_id=self._video_id or "unknown",
            filename=f"{self._video_id or 'unknown'}.mp4",
            duration=(
                sum(r.duration for r in self._chunk_results)
                if self._chunk_results
                else 0.0
            ),
            filepath="",
            scenes=self._all_scenes,
            transcript=self._all_transcript_segments,
            full_transcript=" ".join(self._all_transcript_text),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return processing statistics."""
        return {
            "chunks_processed": self._processed_chunks,
            "total_scenes": len(self._all_scenes),
            "total_transcript_segments": len(self._all_transcript_segments),
            "unique_objects": len(self._all_objects),
            "video_id": self._video_id,
        }
