"""
LLM Vision scheduler — decides WHEN to call LLM and builds the prompts.

Dual-mode: periodic (every N seconds) AND event-triggered (on motion).
Enforces cooldown to prevent runaway API costs.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Callable, List, Optional

import cv2
import numpy as np
from PIL import Image

from video_analysis.stream.sampler import SampledFrame

logger = logging.getLogger(__name__)

# Minimum interval between LLM Vision calls (seconds)
DEFAULT_COOLDOWN = 15.0

# Prompt for periodic analysis
PERIODIC_PROMPT = """You are watching a live video feed. Describe what's happening in this scene.

Focus on:
1. What is the main subject/scene type (indoor, outdoor, office, street, etc.)
2. Are there people? How many? What are they doing?
3. Are there vehicles or animals?
4. Any visible text or signs
5. Overall activity level (quiet/busy/chaotic)

Return ONLY valid JSON:
{
  "scene_type": "...",
  "description": "Brief 1-2 sentence scene description",
  "people_count": 0,
  "activity_level": "quiet|moderate|busy",
  "objects": ["visible", "items"],
  "text_visible": "",
  "changes_since_last": "if available, describe changes from previous frame"
}"""

# Prompt for motion-triggered analysis
MOTION_PROMPT = """MOTION DETECTED in the video feed. Examine these frames carefully.

The first frame is the most recent. Earlier frames show what happened just before.

Describe:
1. What changed between frames?
2. Is this an event that needs attention?
3. Are there people/vehicles involved?
4. What is the urgency level?

Return ONLY valid JSON:
{
  "event_type": "motion|person|vehicle|unknown",
  "description": "What happened and what changed",
  "urgency": "low|medium|high",
  "people_involved": false,
  "needs_attention": false,
  "changes": ["list", "of", "changes"]
}"""


def _encode_frame(frame_bgr: np.ndarray, max_size: int = 1024) -> Optional[str]:
    """Encode a frame as base64 JPEG for LLM Vision API."""
    try:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning("Frame encode failed: %s", e)
        return None


class LLMAnalyzer:
    """Schedules and executes LLM Vision analysis on sampled frames.

    Args:
        chat_fn: Callable that takes (messages, images, system) and returns text.
        cooldown_seconds: Minimum seconds between LLM calls.
        periodic_interval: Seconds between periodic analyses.
        motion_cooldown: Seconds to wait after motion-triggered analysis.
    """

    def __init__(
        self,
        chat_fn: Callable[..., Optional[str]],
        cooldown_seconds: float = DEFAULT_COOLDOWN,
        periodic_interval: float = 30.0,
        motion_cooldown: float = 10.0,
    ):
        self._chat = chat_fn
        self._cooldown = cooldown_seconds
        self._periodic_interval = periodic_interval
        self._motion_cooldown = motion_cooldown
        self._last_analysis_time = 0.0
        self._last_motion_analysis_time = 0.0
        self._last_description = ""
        self._analysis_count = 0

    def should_analyze_periodic(self, now: float) -> bool:
        """Check if periodic analysis is due."""
        return (now - self._last_analysis_time) >= self._periodic_interval

    def should_analyze_motion(self, now: float, motion_score: float) -> bool:
        """Check if motion-triggered analysis should fire."""
        if motion_score < 0.1:
            return False
        if (now - self._last_motion_analysis_time) < self._motion_cooldown:
            return False
        if (now - self._last_analysis_time) < self._cooldown:
            return False
        return True

    def analyze(
        self,
        frames: List[SampledFrame],
        triggered_by: str = "periodic",
        motion_score: float = 0.0,
    ) -> Optional[str]:
        """Send frames to LLM Vision for analysis.

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

        # Select prompt based on trigger
        if triggered_by == "motion":
            prompt = MOTION_PROMPT
        else:
            prompt = PERIODIC_PROMPT

        # Add temporal context
        if self._last_description:
            prompt += f"\nPrevious scene description: {self._last_description[:200]}"
        if motion_score > 0:
            prompt += f"\nMotion score: {motion_score:.3f}"

        prompt += "\n\nCurrent frame(s):"

        # Encode frames (limit to 3 for cost/speed)
        encoded_frames: List[str] = []
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
                if triggered_by == "motion":
                    self._last_motion_analysis_time = now
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
