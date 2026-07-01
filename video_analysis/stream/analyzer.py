"""
LLM Vision scheduler — decides WHEN to call LLM for occasional scene descriptions.

Object detection is now handled by YOLO in engine.py; this module only handles
infrequent Vision LLM calls for high-level scene context (every 5 minutes).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from video_analysis.stream.sampler import SampledFrame

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN = 60.0
PERIODIC_INTERVAL = 300.0


def _encode_frame(frame_bgr: np.ndarray, max_size: int = 1024) -> Optional[str]:
    """Encode a frame as base64 JPEG for LLM Vision API."""
    try:
        h, w = frame_bgr.shape[:2]
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame = frame_bgr

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning("Frame encoding failed: %s", e)
        return None


class LLMAnalyzer:
    """Schedules and executes occasional LLM Vision scene descriptions.

    Heavy object detection is handled by YOLO in the stream engine; this
    class only provides optional high-level scene context at reduced frequency.

    Args:
        chat_fn: Callable that takes (messages, images, system) and returns text.
        cooldown_seconds: Minimum seconds between LLM calls.
        periodic_interval: Seconds between periodic scene descriptions.
    """

    def __init__(
        self,
        chat_fn: Callable[..., Optional[str]],
        cooldown_seconds: float = DEFAULT_COOLDOWN,
        periodic_interval: float = PERIODIC_INTERVAL,
    ):
        self._chat = chat_fn
        self._cooldown = cooldown_seconds
        self._periodic_interval = periodic_interval
        self._last_analysis_time = 0.0
        self._last_description = ""
        self._analysis_count = 0

    def should_analyze_periodic(self, now: float) -> bool:
        return (now - self._last_analysis_time) >= self._periodic_interval

    def should_analyze_motion(self, now: float, motion_score: float) -> bool:
        if motion_score < 0.1:
            return False
        if (now - self._last_analysis_time) < self._cooldown:
            return False
        return True

    def analyze(
        self,
        frames: list[SampledFrame],
        triggered_by: str = "periodic",
        motion_score: float = 0.0,
    ) -> Optional[str]:
        """Send frames to LLM Vision for a scene description.

        Args:
            frames: List of frames to analyze (latest first recommended).
            triggered_by: "periodic" or "motion".
            motion_score: Motion score that triggered this (if applicable).

        Returns:
            LLM description text, or None on failure.
        """
        if not frames:
            return None

        now = time.time()
        if (now - self._last_analysis_time) < self._cooldown:
            logger.debug("LLM cooldown active, skipping analysis")
            return None

        prompt = (
            "Describe the current scene in this video feed briefly:\n"
            "1. What type of scene (indoor/outdoor/office/street/etc.)\n"
            "2. Are there people visible? Approximately how many?\n"
            "3. Any vehicles, animals, or notable objects?\n"
            "4. Overall activity level (quiet/moderate/busy)\n\n"
            "Return ONLY valid JSON:\n"
            '{"scene_type": "...", "description": "...", '
            '"people_count": 0, "activity_level": "quiet|moderate|busy"}'
        )

        if motion_score > 0:
            prompt += f"\nMotion score: {motion_score:.3f}"

        prompt += "\n\nCurrent frame(s):"

        # Encode frames (limit to 3 for cost/speed)
        encoded_frames: list[str] = []
        for sf in frames[:3]:
            encoded = _encode_frame(sf.frame_bgr)
            if encoded:
                encoded_frames.append(encoded)

        if not encoded_frames:
            logger.warning("No frames could be encoded")
            return None

        try:
            response = self._chat(
                messages=[{"role": "user", "content": prompt}],
                images=encoded_frames,
            )

            if response:
                self._last_analysis_time = now
                self._analysis_count += 1

                # Try to extract description from JSON
                try:
                    json_str = response.strip()
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    data = json.loads(json_str)
                    desc = data.get("description", response[:300])
                except json.JSONDecodeError:
                    desc = response[:500]

                self._last_description = desc
                logger.info(
                    "LLM analysis #%d (%s): %s",
                    self._analysis_count, triggered_by, desc[:80],
                )
                return desc

        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)

        return None

    @property
    def analysis_count(self) -> int:
        return self._analysis_count

    @property
    def last_description(self) -> str:
        return self._last_description
