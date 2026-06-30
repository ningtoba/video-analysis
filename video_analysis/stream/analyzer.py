"""
LLM Vision scheduler — decides WHEN to call LLM and builds the prompts.

Dual-mode: periodic (every N seconds) AND event-triggered (on motion).
Enforces cooldown to prevent runaway API costs.
Maintains an event chain: each analysis receives the previous descriptions
so the LLM describes *what changed* rather than re-describing from scratch.
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

DEFAULT_COOLDOWN = 15.0

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
  "changes_since_last": "describe what changed since previous observation"
}"""

MOTION_PROMPT = """MOTION DETECTED in the video feed. Examine these frames carefully.

Describe what's happening, focusing on what changed or what caused the motion.
Return ONLY valid JSON:
{
  "scene_type": "...",
  "description": "Brief description of the current scene",
  "people_count": 0,
  "activity_level": "quiet|moderate|busy",
  "motion_cause": "what likely caused the motion (person walking, vehicle, animal, etc.)",
  "objects": ["visible", "items"],
  "text_visible": "",
  "changes_since_last": "what has changed compared to the previous observation"
}"""


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
    """Schedules and executes LLM Vision analysis on sampled frames.

    Maintains an event chain: each analysis receives previous descriptions
    as context so the LLM describes *what changed* rather than re-describing
    the entire scene every time.

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
        # Event chain: keeps recent descriptions for temporal context
        self._event_chain: list[str] = []
        self._max_chain_length = 5
        self._analysis_count = 0

    def should_analyze_periodic(self, now: float) -> bool:
        return (now - self._last_analysis_time) >= self._periodic_interval

    def should_analyze_motion(self, now: float, motion_score: float) -> bool:
        if motion_score < 0.1:
            return False
        if (now - self._last_motion_analysis_time) < self._motion_cooldown:
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
        """Send frames to LLM Vision for analysis.

        Feeds the event chain into the prompt so the LLM describes
        *what changed* rather than re-describing from scratch.

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

        # Select base prompt
        if triggered_by == "motion":
            prompt = MOTION_PROMPT
        else:
            prompt = PERIODIC_PROMPT

        # Inject event chain: tell the LLM what was seen before
        if self._event_chain:
            prompt += "\n\n### Previous observations (oldest → newest):\n"
            for i, prev_desc in enumerate(self._event_chain[-self._max_chain_length:], 1):
                prompt += f"{i}. {prev_desc[:300]}\n"
            prompt += (
                "\nDescribe what HAS CHANGED compared to the previous observations above. "
                "Focus on: new objects, people entering/leaving, movement, "
                "or any notable differences. If nothing significant changed, say so briefly."
            )
        else:
            prompt += "\n\n(This is the first observation of this scene — describe it fully.)"

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

                # Store in event chain
                self._last_description = desc
                self._event_chain.append(desc)
                # Trim chain
                if len(self._event_chain) > self._max_chain_length:
                    self._event_chain = self._event_chain[-self._max_chain_length:]

                logger.info(
                    "LLM analysis #%d (%s, chain=%d): %s",
                    self._analysis_count, triggered_by,
                    len(self._event_chain), desc[:80],
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
