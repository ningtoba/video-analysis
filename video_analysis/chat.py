"""
Chat module — LLM-powered Q&A over video analysis results.

Uses the configured LLM provider (OpenAI, Anthropic, Gemini, etc.)
with transcript + frame descriptions as context. No vector DB needed.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from video_analysis.config import Config
from video_analysis.llm_provider import LLMProviderConfig, get_llm_provider
from video_analysis.models import ChatMessage, VideoAnalysis

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a video analysis assistant. Answer questions about video content based on the provided transcript and visual analysis.

Rules:
1. Answer based ONLY on the provided context
2. When referencing specific moments, include timestamps in HH:MM:SS format
3. If the context doesn't have enough information, say so
4. Be concise and factual

Context includes:
- Transcript: timestamped dialogue/narration
- Visual descriptions: what's visible in frames at specific timestamps
- Detected objects and text visible in frames"""


class VideoChat:
    """Chat interface over a processed video's analysis."""

    def __init__(self, config: Optional[Config] = None, llm=None):
        self.config = config or Config()
        self._llm = llm
        self.history: List[ChatMessage] = []

    def _get_llm(self):
        """Lazy-load the LLM provider."""
        if self._llm is None:
            cfg = LLMProviderConfig(
                provider=self.config.llm_provider,
                api_key=self.config.llm_api_key,
                api_base=self.config.llm_api_base,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            self._llm = get_llm_provider(cfg)
        return self._llm

    def _build_context(self, analysis: VideoAnalysis) -> str:
        """Build context string from video analysis."""
        parts = []
        parts.append(f"Video: {analysis.title or analysis.filename}")
        parts.append(f"Duration: {analysis.duration:.1f}s")

        # Transcript
        if analysis.transcript:
            parts.append("\n--- Transcript ---")
            for seg in analysis.transcript[:100]:  # Limit to first 100 segments
                speaker = f" [{seg.speaker}]" if seg.speaker else ""
                parts.append(f"[{seg.start:.1f}s-{seg.end:.1f}s]{speaker}: {seg.text}")

        # Frame descriptions
        frames_with_desc = [f for f in analysis.frames if f.llm_description]
        if frames_with_desc:
            parts.append("\n--- Visual Analysis ---")
            for frame in frames_with_desc[:30]:  # Limit to 30 frames
                desc = frame.llm_description or ""
                objects = ""
                if frame.llm_objects:
                    objects = f" [Objects: {', '.join(frame.llm_objects[:10])}]"
                ocr = ""
                if frame.llm_ocr:
                    ocr = f" [Text: {frame.llm_ocr}]"
                parts.append(f"[{frame.timestamp:.1f}s] {desc}{objects}{ocr}")

        # Summary
        if analysis.llm_summary:
            parts.append(f"\n--- Summary ---\n{analysis.llm_summary}")

        return "\n".join(parts)

    def ask(self, question: str, analysis: VideoAnalysis) -> Optional[str]:
        """Ask a question about a video's analysis."""
        llm = self._get_llm()
        if not llm or not self.config.llm_api_key:
            logger.warning("No LLM provider configured")
            return None

        context = self._build_context(analysis)
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with timestamp citations where relevant:"

        try:
            response = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM_PROMPT,
            )
            if response:
                self.history.append(ChatMessage(role="user", content=question))
                self.history.append(ChatMessage(role="assistant", content=response))
            return response
        except Exception as e:
            logger.error("Chat failed: %s", e)
            return None

    def clear_history(self):
        self.history.clear()
