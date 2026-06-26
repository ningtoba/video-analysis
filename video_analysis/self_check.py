"""
Self-Check + Re-Retrieval — LLM-verified answer-evidence alignment.

Inspired by Self-RAG (ICLR 2024), CRAG (Corrective RAG, 2024), and DSLM2
(Dynamic Self-Correction for LLMs, 2025). This module adds a verification
layer after retrieval that:

1. Generates a draft answer from retrieved evidence
2. Checks whether the evidence actually supports the answer
3. Re-retrieves when confidence is low (using decomposition or reformulation)
4. Returns the final verified answer with confidence metadata

The self-check operates as the 4th round in the agentic_retrieve pipeline
(or standalone), adding LLM-grade verification on top of embedding-based
confidence thresholds.

Usage:
    from video_analysis.self_check import SelfCheckRAG

    checker = SelfCheckRAG(config)
    result = checker.verify(query, chunks, video_id)
    if result.verdict == "supported":
        # answer is trustworthy
    elif result.verdict == "unsupported":
        # re-retrieval was attempted
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple

from video_analysis.config import Config
from video_analysis.rag import VideoRAG, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class SelfCheckResult:
    """Result of a self-check + re-retrieval pass."""

    query: str
    draft_answer: str
    verdict: str  # "supported" | "partial" | "unsupported" | "error"
    confidence_score: float  # 0.0 - 1.0
    gaps: List[str] = field(default_factory=list)
    corrected_answer: str = ""
    retrieval_rounds: int = 1
    total_chunks_used: int = 0


class SelfCheckRAG:
    """LLM-based verification layer for RAG retrieval.

    After retrieval produces top-k chunks, this module asks the LLM to:

    1. Draft a concise answer from the evidence
    2. Rate whether the evidence supports the answer (supported/partial/unsupported)
    3. Identify specific gaps (missing timestamps, conflicting info, incomplete coverage)
    4. On "unsupported" or "partial" with significant gaps:
       a. Reformulate the query to address gaps
       b. Re-retrieve with the reformulated query
       c. Re-verify until confidence is adequate or max rounds reached

    The self-check is designed as a **drop-in enhancement** to the existing
    agentic_retrieve pipeline — specifically as Round 4 (LLM verification)
    after Rounds 1-3 (embedding search, multi-hop, scene-graph expansion).

    **LLM Provider Integration:** Instead of directly calling the Hermes CLI,
    this module uses the :class:`LLMProvider` abstraction, supporting both
    Hermes CLI and OpenAI-compatible backends.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        rag: Optional[VideoRAG] = None,
        llm=None,
    ):
        self.config = config or Config()
        self.rag = rag  # optional — set later if not provided at init
        self._llm = llm  # optional LLMProvider instance
        self._cached_verdicts: dict = {}  # query hash -> verdict cache

    def _get_llm(self):
        """Lazy-load the LLM provider."""
        if self._llm is None:
            from video_analysis.llm_provider import get_llm_provider, LLMProviderConfig

            cfg = LLMProviderConfig(
                provider=os.environ.get("LLM_PROVIDER", "hermes"),
                api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("OPENAI_MODEL", "qwen2.5"),
                max_tokens=1024,
                temperature=0.1,
                timeout=60,
                hermes_model=self.config.llm_model,
                hermes_max_tokens=1024,
            )
            self._llm = get_llm_provider(cfg)
            logger.info("SelfCheckRAG using LLM provider: %s", self._llm.name)
        return self._llm

    def verify(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        video_id: Optional[str] = None,
        max_rounds: int = 2,
    ) -> SelfCheckResult:
        """Run self-check on retrieved chunks.

        Args:
            query: The user's question.
            chunks: Retrieved chunks (from agentic_retrieve or retrieve).
            video_id: Optional video filter for re-retrieval.
            max_rounds: Max verification+reretrieval rounds (default: 2).

        Returns:
            SelfCheckResult with verdict and corrected answer.
        """
        if not chunks:
            return SelfCheckResult(
                query=query,
                draft_answer="",
                verdict="unsupported",
                confidence_score=0.0,
                gaps=["No retrieved evidence to verify"],
                retrieval_rounds=1,
                total_chunks_used=0,
            )

        current_query = query
        all_chunks = list(chunks)
        seen_queries: Set[str] = {query.lower().strip()}

        for round_num in range(1, max_rounds + 1):
            # Step 1: Build evidence string
            evidence_text = self._build_evidence_text(all_chunks)

            # Step 2: Ask LLM to draft answer and check evidence
            result = self._check_evidence(current_query, evidence_text, round_num)

            if round_num == 1:
                result.total_chunks_used = len(all_chunks)

            # If supported with high confidence, early stop
            if result.verdict == "supported" and result.confidence_score >= 0.7:
                logger.info(
                    f"Self-check round {round_num}: "
                    f"verdict={result.verdict}, "
                    f"confidence={result.confidence_score:.2f} — early stop"
                )
                result.retrieval_rounds = round_num
                result.total_chunks_used = len(all_chunks)
                return result

            # If we have gaps but the verdict isn't "unsupported", try re-retrieval
            if result.verdict in ("partial", "unsupported"):
                # Step 3: Reformulate query to address gaps
                reformulated = self._reformulate_query(
                    current_query, result.gaps, result.draft_answer
                )

                q_lower = reformulated.lower().strip()
                if reformulated and q_lower not in seen_queries:
                    seen_queries.add(q_lower)
                    current_query = reformulated
                    logger.info(
                        f"Self-check round {round_num}: "
                        f"re-retrieving with reformulated query: {reformulated[:80]}"
                    )

                    # Step 4: Re-retrieve with reformulated query
                    if self.rag is not None:
                        new_chunks = self._re_retrieve(
                            current_query, video_id, all_chunks
                        )
                        if new_chunks:
                            all_chunks = self._merge_chunks(all_chunks, new_chunks)

                    result.retrieval_rounds = round_num + 1
                    result.total_chunks_used = len(all_chunks)
                    continue

            # No re-retrieval needed or possible
            result.retrieval_rounds = round_num
            result.total_chunks_used = len(all_chunks)
            return result

        # Reached max rounds — return best result
        final_result = self._check_evidence(
            current_query, self._build_evidence_text(all_chunks), max_rounds
        )
        final_result.retrieval_rounds = max_rounds
        final_result.total_chunks_used = len(all_chunks)
        return final_result

    def _build_evidence_text(self, chunks: List[RetrievedChunk]) -> str:
        """Build a structured evidence string from retrieved chunks."""
        lines = []
        for i, chunk in enumerate(chunks[:10]):  # max 10 chunks for LLM context
            header = f"[{i + 1}]"
            if chunk.timestamp > 0:
                from video_analysis.models import format_timestamp

                header += f" @ {format_timestamp(chunk.timestamp)}"
            if chunk.chunk_type:
                header += f" ({chunk.chunk_type})"
            if chunk.scene_id >= 0:
                header += f" [Scene {chunk.scene_id}]"
            header += f" score={chunk.score:.3f}"
            lines.append(header)
            # Truncate long evidence per chunk
            text = chunk.text[:600].strip()
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def _check_evidence(
        self, query: str, evidence: str, round_num: int
    ) -> SelfCheckResult:
        """Ask the LLM to verify evidence against the query.

        Returns a SelfCheckResult with the LLM's assessment.
        """
        prompt = f"""You are a rigorous fact-checking layer for a video analysis RAG system. Your job is to evaluate whether the retrieved evidence supports answering the user's question.

USER QUESTION: {query}

RETRIEVED EVIDENCE:
{evidence}

INSTRUCTIONS:
1. First, produce a concise draft answer (1-3 sentences) based SOLELY on the provided evidence.
2. Then, determine the VERDICT:
   - "supported" — The evidence fully supports a complete, accurate answer with specific timestamps and details.
   - "partial" — The evidence supports part of the answer, but some aspects are missing, vague, or conflicting.
   - "unsupported" — The evidence does not contain enough information to answer the question.
3. Rate your confidence (0.0 to 1.0) based on:
   - How much of the evidence directly addresses the query
   - Presence of timestamp-specific citations
   - Consistency across multiple evidence chunks
   - Coverage of all aspects of the question
4. If verdict is "partial" or "unsupported", list specific gaps (e.g., "No evidence about object colors", "Missing timestamp for event X", "Evidence only covers first half of the question")

Respond in this exact JSON format:
{{
    "draft_answer": "...",
    "verdict": "supported|partial|unsupported",
    "confidence": 0.0-1.0,
    "gaps": ["gap1", "gap2"]
}}"""

        llm = self._get_llm()
        parsed = llm.structured_chat(
            prompt=prompt,
            temperature=0.1,
            max_tokens=1024,
            timeout=60,
        )

        if parsed is None:
            return SelfCheckResult(
                query=query,
                draft_answer="",
                verdict="unsupported",
                confidence_score=0.0,
                gaps=["LLM verification call failed"],
            )

        confidence = max(0.0, min(1.0, parsed.get("confidence", 0.0)))
        return SelfCheckResult(
            query=query,
            draft_answer=parsed.get("draft_answer", ""),
            verdict=parsed.get("verdict", "unsupported"),
            confidence_score=confidence,
            gaps=parsed.get("gaps", []),
        )

    def _reformulate_query(
        self, original_query: str, gaps: List[str], draft_answer: str
    ) -> str:
        """Reformulate the query to address identified gaps."""
        gaps_text = (
            "\n".join(f"- {g}" for g in gaps)
            if gaps
            else "No specific gaps identified."
        )
        prompt = f"""You are helping a video analysis RAG system improve its retrieval. The system asked about:

ORIGINAL QUERY: {original_query}

DRAFT ANSWER (from partial evidence):
{draft_answer[:500]}

IDENTIFIED GAPS:
{gaps_text}

Reformulate the original query to address these gaps. Focus on what's MISSING from the evidence.
Keep it concise (1-2 sentences). Output ONLY the reformulated query, no explanation."""

        llm = self._get_llm()
        reformulated = llm.chat(
            prompt=prompt,
            temperature=0.1,
            max_tokens=256,
            timeout=30,
        )
        if reformulated and len(reformulated) > 10:
            return reformulated.strip().strip('"').strip("'")

        return original_query

    def _re_retrieve(
        self,
        reformulated_query: str,
        video_id: Optional[str],
        existing_chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Re-retrieve with a reformulated query.

        Uses the existing RAG's agentic_retrieve or standard retrieve method.
        Filters out chunks already seen to maximize new information.
        """
        if self.rag is None:
            return []

        try:
            existing_ids: Set[str] = {c.chunk_id for c in existing_chunks}

            if self.config.agentic_retrieval_enabled:
                new_chunks = self.rag.agentic_retrieve(
                    reformulated_query, video_id=video_id
                )
            elif self.config.query_routing_enabled or self.config.multi_hop_enabled:
                new_chunks = self.rag.routed_retrieve(
                    reformulated_query, video_id=video_id
                )
            else:
                new_chunks = self.rag.retrieve(reformulated_query, video_id=video_id)

            # Filter to only genuinely new chunks
            novel = [c for c in new_chunks if c.chunk_id not in existing_ids]
            logger.info(
                f"Re-retrieval: {len(new_chunks)} total, " f"{len(novel)} novel chunks"
            )
            return novel

        except Exception as e:
            logger.warning(f"Re-retrieval failed: {e}")
            return []

    def _merge_chunks(
        self,
        existing: List[RetrievedChunk],
        new_chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Merge existing and new chunks, deduplicating by chunk_id.

        New chunks are scored slightly higher to surface fresh evidence
        in the verification pass.
        """
        merged: dict = {}
        # Add new chunks first (with slight score bump)
        for c in new_chunks:
            merged[c.chunk_id] = RetrievedChunk(
                text=c.text,
                score=c.score + 0.05,
                timestamp=c.timestamp,
                video_id=c.video_id,
                scene_id=c.scene_id,
                chunk_type=c.chunk_type,
                chunk_id=c.chunk_id,
                metadata=c.metadata,
            )
        # Add existing (will not overwrite new, preserving the score bump)
        for c in existing:
            if c.chunk_id not in merged:
                merged[c.chunk_id] = c
        return list(merged.values())
