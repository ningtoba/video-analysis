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

The router uses the LLM itself (via Hermes CLI) for classification when
available, with keyword-based fallback for offline/resource-constrained
environments.  The LLM call is a single lightweight prompt (no model load).
"""

import json
import logging
import re
import subprocess
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

    Args:
        config: Platform config (for LLM endpoint).
        prefer_llm: If True (default), tries LLM first; falls back to keywords.
                   If False, uses keywords only (faster, no external call).
    """

    def __init__(self, config: Optional[Config] = None, prefer_llm: bool = True):
        self.config = config or Config()
        self.prefer_llm = prefer_llm

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
        """Classify using the configured LLM via Hermes CLI.

        Uses a lightweight single-turn prompt with no model load.
        """
        prompt = f"""{_ROUTING_SYSTEM_PROMPT}

Question: {query}

JSON response:"""

        try:
            result = subprocess.run(
                [
                    "hermes",
                    "chat",
                    "-q",
                    "-m",
                    self.config.llm_model,
                    "-t",
                    "0.1",  # low temperature for deterministic output
                    "--max-tokens",
                    "256",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                # Extract JSON from the output
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    match = re.search(r"\{[^}]+\}", output)
                    if match:
                        parsed = json.loads(match.group())
                    else:
                        raise

                route_str = parsed.get("route", "text").lower()
                route = (
                    QueryRoute(route_str)
                    if route_str in [r.value for r in QueryRoute]
                    else QueryRoute.TEXT
                )
                confidence = min(float(parsed.get("confidence", 1.0)), 1.0)
                reasoning = parsed.get("reasoning", "")

                return RoutingDecision(
                    route=route,
                    confidence=confidence,
                    reasoning=reasoning,
                )
        except FileNotFoundError:
            logger.debug("Hermes CLI not available for routing")
        except subprocess.TimeoutExpired:
            logger.debug("LLM routing timed out")
        except Exception as e:
            logger.debug(f"LLM routing error: {e}")

        return None

    def _classify_keyword(self, query: str) -> RoutingDecision:
        """Classify using keyword/pattern matching — lightweight fallback."""
        q = query.strip()

        # Count keyword matches for each category
        temporal_score = len(_TEMPORAL_KEYWORDS.findall(q))
        visual_score = len(_VISUAL_KEYWORDS.findall(q))
        multimodal_score = len(_MULTIMODAL_KEYWORDS.findall(q))

        # Find the best match
        scores = {
            QueryRoute.TEMPORAL: temporal_score,
            QueryRoute.VISUAL: visual_score,
            QueryRoute.MULTIMODAL: multimodal_score,
        }

        best_route = QueryRoute.TEXT
        best_score = 0
        for route, score in scores.items():
            if score > best_score:
                best_score = score
                best_route = route

        # Compute confidence based on how decisive the match is
        total = sum(scores.values()) or 1
        confidence = min(best_score / total, 1.0) if best_score > 0 else 0.5

        return RoutingDecision(
            route=best_route,
            confidence=confidence,
            reasoning=f"keyword match: {dict((k.value, v) for k, v in scores.items())}",
        )

    def classify_and_decompose(self, query: str) -> RoutingDecision:
        """Classify a query and decompose multimodal/complex queries into sub-questions.

        For ``multimodal`` routes, generates sub-questions that can be used
        for multi-hop retrieval.  For other routes, returns the query as-is.

        Uses the LLM for decomposition when available; falls back to heuristics.
        """
        decision = self.classify(query)

        if decision.route == QueryRoute.MULTIMODAL:
            sub_queries = self._decompose_multimodal(query)
            if sub_queries:
                decision.sub_queries = sub_queries

        return decision

    def _decompose_multimodal(self, query: str) -> List[str]:
        """Decompose a complex multimodal query into simpler sub-questions.

        Uses the LLM to break down complex queries into focused sub-questions
        that can be answered independently via text-only, visual-only, or
        temporal-only retrieval.
        """
        decompose_prompt = f"""Break this complex question about a video into 2-4 focused sub-questions.
Each sub-question should target ONE type of information (text/visual/temporal).
Make each sub-question standalone and answerable from video context.

Original question: {query}

Return as a JSON array of strings ONLY:
["sub-question 1", "sub-question 2", ...]"""

        try:
            result = subprocess.run(
                [
                    "hermes",
                    "chat",
                    "-q",
                    "-m",
                    self.config.llm_model,
                    "-t",
                    "0.1",
                    "--max-tokens",
                    "512",
                ],
                input=decompose_prompt,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                try:
                    sub_queries = json.loads(output)
                except json.JSONDecodeError:
                    # Try extracting array from markdown
                    match = re.search(r'\["[^\]]+"\]', output)
                    if match:
                        sub_queries = json.loads(match.group())
                    else:
                        return []
                if isinstance(sub_queries, list) and len(sub_queries) >= 2:
                    logger.info(
                        f"Decomposed query into {len(sub_queries)} sub-questions: {sub_queries}"
                    )
                    return sub_queries
        except Exception as e:
            logger.debug(f"Query decomposition failed: {e}")

        # Fallback: naive heuristic decomposition
        return self._heuristic_decompose(query)

    def _heuristic_decompose(self, query: str) -> List[str]:
        """Fallback decomposition using heuristics when LLM is unavailable.

        Extracts entity-related questions and temporal context as separate
        sub-questions.
        """
        sub_queries = [query]

        # If query asks "why", split into "what happened" + "what was the context"
        if query.lower().startswith("why"):
            sub_queries = [
                f"What happened in the video related to: {query}",
                f"What was the context around: {query}",
            ]

        # If query has multiple entities with "and", split
        if " and " in query.lower() and len(query) > 50:
            parts = re.split(r"\band\b", query)
            if len(parts) >= 2:
                sub_queries = [p.strip().strip("?") + "?" for p in parts[:3]]

        return sub_queries
