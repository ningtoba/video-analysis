"""
Query Router — classifies user queries and routes to optimal retrieval strategy.

Implements the **query classification & multi-modal routing** architecture
described in the project roadmap.  Routes user queries into one of four
retrieval strategies based on query type:

| Route | Query Type | Example | Retrieval Strategy |
|-------|-----------|---------|-------------------|
| ``text`` | Factual, narrative | "What did the speaker say about X?" | Standard ChromaDB dense + temporal expansion |
| ``visual`` | Visual content, objects | "What color was the car?" | BGE-VL visual search + Video MLLM if available |
| ``temporal`` | Time/sequence | "What happened before the explosion?" | ChromaDB + temporal decay weighting + scene graph |
| ``multimodal`` | Complex, multi-aspect | "Why did the protagonist leave?" | Multi-hop decomposition + combined retrieval |

The router uses the LLM for classification when available,
with keyword-based fallback for offline/resource-constrained
environments. The LLM call is a single lightweight prompt (no model load).

**LLM Provider Integration:** Instead of directly calling the Hermes CLI,
this module uses the :class:`LLMProvider` abstraction, supporting both
Hermes CLI and OpenAI-compatible backends.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from video_analysis.config import Config

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    """Enumeration of supported query routing strategies."""

    TEXT = "text"
    VISUAL = "visual"
    TEMPORAL = "temporal"
    MULTIMODAL = "multimodal"

    def __str__(self) -> str:
        return self.value


@dataclass
class RoutingDecision:
    """Result of query routing analysis."""

    route: QueryRoute
    confidence: float = 1.0  # 0.0 to 1.0
    sub_queries: List[str] = None  # for multimodal/multi-hop decomposition
    reasoning: str = ""

    def __post_init__(self):
        if self.sub_queries is None:
            self.sub_queries = []


# Keyword patterns for lightweight route matching (fallback when LLM is unavailable)
_TEMPORAL_KEYWORDS = re.compile(
    r"\b(before|after|during|then|earlier|later|previously|subsequently|"
    r"first|second|next|last|finally|sequence|timeline|when did|"
    r"what happened (before|after)|chronolog|order of events)\b",
    re.IGNORECASE,
)

_VISUAL_KEYWORDS = re.compile(
    r"\b(see|look|appear|visible|color|colour|shape|size|wearing|"
    r"on screen|in the frame|show[s]?|display|what does .* look like|"
    r"what (is|are) .* (in|on) the (frame|image|scene)|"
    r"what (can|could) you see|describe (the |this )?(scene|video|image|frame))\b",
    re.IGNORECASE,
)

_MULTIMODAL_KEYWORDS = re.compile(
    r"\b(why did|how did|compare|contrast|explain|summarize|"
    r"what (is|are) the (reason|cause|effect|relationship)|"
    r"what (happened|transpired)|tell me about|describe (in detail))\b",
    re.IGNORECASE,
)


# LLM-based routing prompt template (cheap and fast)
_ROUTING_SYSTEM_PROMPT = """You are a query classification system for a video analysis platform.
Classify the user's question into ONE of these routes:

- **text**: Factual questions about spoken content, dialogue, or information.
  Example: "What did the speaker say about the budget?"
- **visual**: Questions about visual elements — objects, colors, people's appearance, on-screen text.
  Example: "What color was the car in the chase scene?"
- **temporal**: Questions about timing, sequence of events, or what happened before/after something.
  Example: "What happened right before the explosion?"
- **multimodal**: Complex questions requiring multiple types of evidence — combining visuals, text, timing.
  Example: "Why did the character leave the room?"

Respond in this exact JSON format:
{"route": "text|visual|temporal|multimodal", "confidence": 0.0-1.0, "reasoning": "Brief reason"}"""


class QueryRouter:
    """Classifies user queries and determines optimal retrieval strategy.

    Two-tier approach:
    1. **LLM-based classification** — uses the configured LLM for accurate routing
       (fast, single-turn prompt).
    2. **Keyword-based fallback** — regex patterns for lightweight, offline routing.

    Uses the :class:`LLMProvider` abstraction instead of directly calling
    the Hermes CLI, enabling OpenAI-compatible backends (vLLM, Ollama, etc.).

    Args:
        config: Platform config (for LLM endpoint).
        prefer_llm: If True (default), tries LLM first; falls back to keywords.
                   If False, uses keywords only (faster, no external call).
        llm: Optional pre-configured LLMProvider instance.
    """

    def __init__(self, config: Optional[Config] = None, prefer_llm: bool = True, llm=None):
        self.config = config or Config()
        self.prefer_llm = prefer_llm
        self._llm = llm  # optional LLMProvider instance

    def _get_llm(self):
        """Lazy-load the LLM provider."""
        if self._llm is None:
            from video_analysis.llm_provider import LLMProviderConfig, get_llm_provider

            cfg = LLMProviderConfig(
                provider=os.environ.get("LLM_PROVIDER", "hermes"),
                api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("OPENAI_MODEL", "qwen2.5"),
                max_tokens=256,
                temperature=0.1,
                timeout=30,
                hermes_model=self.config.llm_model,
                hermes_max_tokens=256,
            )
            self._llm = get_llm_provider(cfg)
            logger.info("QueryRouter using LLM provider: %s", self._llm.name)
        return self._llm

    def classify(self, query: str) -> RoutingDecision:
        """Classify a user query into the optimal retrieval route.

        Args:
            query: The user's natural language question.

        Returns:
            RoutingDecision with route, confidence, and reasoning.
        """
        if self.prefer_llm:
            try:
                decision = self._classify_llm(query)
                if decision is not None and decision.confidence >= 0.5:
                    return decision
            except Exception as e:
                logger.debug(f"LLM routing failed ({e}), falling back to keywords")

        return self._classify_keyword(query)

    def _classify_llm(self, query: str) -> Optional[RoutingDecision]:
        """Classify using the configured LLM via LLMProvider.

        Uses a lightweight single-turn prompt with no model load.
        """
        prompt = f"""{_ROUTING_SYSTEM_PROMPT}

Question: {query}

JSON response:"""

        llm = self._get_llm()
        parsed = llm.structured_chat(
            prompt=prompt,
            temperature=0.1,
            max_tokens=256,
            timeout=30,
        )

        if parsed is None:
            return None

        route_str = parsed.get("route", "text").lower()
        route = (
            QueryRoute(route_str) if route_str in [r.value for r in QueryRoute] else QueryRoute.TEXT
        )
        confidence = min(float(parsed.get("confidence", 1.0)), 1.0)
        reasoning = parsed.get("reasoning", "")

        return RoutingDecision(
            route=route,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _classify_keyword(self, query: str) -> RoutingDecision:
        """Fast keyword-based routing fallback."""
        if _TEMPORAL_KEYWORDS.search(query):
            return RoutingDecision(
                route=QueryRoute.TEMPORAL,
                confidence=0.6,
                reasoning="Matched temporal keywords",
            )
        if _VISUAL_KEYWORDS.search(query):
            return RoutingDecision(
                route=QueryRoute.VISUAL,
                confidence=0.6,
                reasoning="Matched visual keywords",
            )
        if _MULTIMODAL_KEYWORDS.search(query):
            return RoutingDecision(
                route=QueryRoute.MULTIMODAL,
                confidence=0.6,
                reasoning="Matched multimodal keywords",
            )
        return RoutingDecision(
            route=QueryRoute.TEXT,
            confidence=0.8,
            reasoning="Default text route (no specific keywords matched)",
        )
