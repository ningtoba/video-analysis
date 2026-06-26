# Query Classification & Routing + Multi-Hop Decomposition for Video RAG

> **Date:** 2026-06-26
> **Context:** Implementation research for video-analysis v0.13.0 roadmap items:
> 1. "Query classification & routing (text/visual/temporal modality dispatch)"
> 2. "Multi-hop query decomposition (sub-question → retrieve → reason)"
> **LLM backend:** DeepSeek-V4-Flash via Hermes CLI (`hermes chat -q`)
> **Current embedding:** BGE-VL-base (text, image, composed)
> **Current retrieval:** ChromaDB + TV-RAG temporal decay + cross-encoder re-ranking + quad-chunk strategy

---

## Table of Contents

1. [Overview: Why Query Routing & Multi-Hop Decomposition](#1-overview)
2. [Query Classification & Routing](#2-query-classification--routing)
   - [2.1 Modality-Based Routing Taxonomy](#21-modality-based-routing-taxonomy)
   - [2.2 LLM-Based Classifier (No Extra Model)](#22-llm-based-classifier-no-extra-model)
   - [2.3 Routing Strategies Per Modality](#23-routing-strategies-per-modality)
   - [2.4 Implementation Pattern](#24-implementation-pattern)
   - [2.5 Performance Considerations](#25-performance-considerations)
3. [Multi-Hop Query Decomposition](#3-multi-hop-query-decomposition)
   - [3.1 The Multi-Hop Pattern](#31-the-multi-hop-pattern)
   - [3.2 Decomposition Strategies](#32-decomposition-strategies)
   - [3.3 Implementation Pattern](#33-implementation-pattern)
   - [3.4 Execution Loop](#34-execution-loop)
4. [Lightweight Python Libraries & Patterns](#4-lightweight-python-libraries--patterns)
5. [Integration Into Existing Codebase](#5-integration-into-existing-codebase)
   - [5.1 New Module: `video_analysis/router.py`](#51-new-module-video_analysisrouterpy)
   - [5.2 Modifications to `video_analysis/chat.py`](#52-modifications-to-video_analysischatpy)
   - [5.3 Modifications to `video_analysis/rag.py`](#53-modifications-to-video_analysisragpy)
   - [5.4 Config Additions](#54-config-additions)
6. [Latency Budget Analysis](#6-latency-budget-analysis)
7. [Alternatives Considered](#7-alternatives-considered)
8. [Recommendations](#8-recommendations)

---

## 1. Overview

### The Problem

Currently, **every user query** follows the same pipeline regardless of intent:

```
User query → BGE-VL text embedding → ChromaDB → cross-encoder re-rank → LLM
```

This works well for text-heavy questions ("What did the speaker say about X?") but is suboptimal for:

- **Visual queries** ("Show me scenes with a red car") — best answered by retrieving frame descriptions or matched image embeddings
- **Temporal queries** ("What happened right before the explosion?") — best with TV-RAG temporal weighting and nearby scene expansion
- **Multimodal queries** ("Find the person wearing a blue shirt talking about AI") — benefits from composed image+text retrieval
- **Complex/holistic queries** ("Summarize the speaker's argument and how the visual evidence supports it") — needs multi-hop sub-question decomposition

### The Solution: Two-Phase Query Processing

```
User query
  │
  ├── [Phase 1] QUERY CLASSIFIER (LLM call, ~50ms)
  │     Determines: modality (text/visual/temporal/multimodal/complex)
  │                 sub-questions (if complex)
  │                 query_time (if temporal)
  │
  ├── [Phase 2a] ROUTED RETRIEVAL (per modality)
  │     Routes to optimal retrieval strategy
  │
  └── [Phase 2b] MULTI-HOP EXECUTION (if complex)
        Decomposes → retrieves per sub-question → reasons → aggregates
```

---

## 2. Query Classification & Routing

### 2.1 Modality-Based Routing Taxonomy

Based on analysis of real video Q&A datasets (MLVU, Video-MME, EgoSchema, NExT-QA), user queries for video content can be classified into these modalities:

| Class | Description | Example Queries | Optimal Retrieval Strategy |
|-------|------------|-----------------|--------------------------|
| **text** | About spoken content, dialogue, narration | "What did the narrator say about climate change?" | Text embedding → ChromaDB (transcript chunks) |
| **visual** | About objects, scenes, actions, visuals | "Show me scenes with cars" / "What color was the dress?" | Text embedding → ChromaDB (frame description chunks) OR BGE-VL image matching |
| **temporal** | About timing, sequence, when things happen | "What happened at 5:30?" / "What was shown before the finale?" | Text embedding → ChromaDB + TV-RAG temporal decay + temporal context expansion |
| **multimodal** | Combines visual + textual elements | "Find the person in red talking about politics" / "Show me the chart the speaker referenced" | BGE-VL composed embedding (image+text) OR multi-query retrieval |
| **complex** | Multi-step, comparative, holistic | "Compare the two scenes where the speaker changes tone" / "Why did the experiment fail?" | Multi-hop decomposition → sub-question retrieval → reasoning chain |
| **metadata** | About the video itself | "How long is this video?" / "Who uploaded this?" | Metadata-only (no vector search needed) |

### 2.2 LLM-Based Classifier (No Extra Model)

The recommended approach uses **a single lightweight LLM call with structured output** to classify the query. No separate classification model is needed — DeepSeek-V4-Flash handles this efficiently.

#### Design Principles

1. **Single-turn classification** — one LLM call, not a multi-turn conversation
2. **Structured output** — JSON with all classification fields
3. **Cheap, fast** — use a small model or low max_tokens
4. **Deterministic extraction** — avoid free-form reasoning in the routing step

#### Prompt Template

```python
QUERY_CLASSIFICATION_PROMPT = """You are a query classifier for a video analysis system.
Analyze the user's query and output a JSON object with these fields:

1. "modality": one of ["text", "visual", "temporal", "multimodal", "complex", "metadata"]
2. "query_time": float or null — if the query references a specific timestamp, extract it in seconds. e.g. "at 5:30" → 330.0. Otherwise null.
3. "sub_queries": list of strings or null — if the query is complex and would benefit from being broken into sub-questions, list them. Otherwise null.
4. "key_entities": list of strings — important nouns/entities to focus retrieval on.
5. "confidence": float 0.0-1.0 — how confident you are in the classification.

Classification rules:
- **text**: Query asks about spoken content, dialogue, narration, opinions, explanations.
- **visual**: Query asks about objects, scenes, colors, actions, visual elements, "show me", "find scenes".
- **temporal**: Query includes a time reference ("at X:XX", "before", "after", "during", "when" in a temporal sense) OR asks about sequence/order of events.
- **multimodal**: Query combines visual elements with textual/spoken content. E.g., "the person wearing X saying Y".
- **complex**: Query requires multiple retrieval steps, comparison, reasoning, or holistic understanding. E.g., "compare", "why", "how did", "what changed".
- **metadata**: Query asks about video properties (duration, filename, upload date, etc.)

User query: {query}

Respond with ONLY the JSON object, no other text:"""

def classify_query(query: str, llm_call_fn) -> QueryClassification:
    """
    Classify a user query using the LLM.
    
    Args:
        query: The user's natural language question
        llm_call_fn: Function that takes a prompt string and returns response text
    
    Returns:
        QueryClassification dataclass
    """
    prompt = QUERY_CLASSIFICATION_PROMPT.format(query=query)
    response = llm_call_fn(prompt, max_tokens=300, temperature=0.0)
    
    try:
        result = json.loads(response)
        return QueryClassification(
            modality=result.get("modality", "text"),
            query_time=result.get("query_time"),
            sub_queries=result.get("sub_queries"),
            key_entities=result.get("key_entities", []),
            confidence=result.get("confidence", 0.5),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Query classification parse failed: {e}")
        return QueryClassification(modality="text")  # safe fallback
```

### 2.3 Routing Strategies Per Modality

Once classified, the router dispatches to the optimal retrieval strategy:

```python
def route_retrieval(
    classification: QueryClassification,
    query: str,
    rag: VideoRAG,
    top_k: int = 20,
) -> List[RetrievedChunk]:
    """Route a query to the optimal retrieval strategy based on its modality."""
    
    modality = classification.modality
    query_time = classification.query_time
    
    if modality == "metadata":
        # Metadata queries — no vector search needed
        # Handled separately in chat.py
        return []
    
    elif modality == "visual":
        # Visual queries: weight frame chunks higher
        # Use BGE-VL image embedding if a reference image exists,
        # otherwise use text embedding with frame chunk bonus
        return rag.retrieve(
            query=query,
            boost_chunk_types=["frame"],
            top_k=top_k,
        )
    
    elif modality == "temporal":
        # Temporal queries: apply TV-RAG temporal decay
        # If query has an explicit timestamp, use it as query_time
        return rag.retrieve(
            query=query,
            query_time=query_time,  # TV-RAG temporal weighting
            top_k=top_k,
        )
    
    elif modality == "multimodal":
        # Multimodal queries: try BGE-VL composed retrieval
        # (image + text) if we have a key frame from the user,
        # otherwise do multi-query: text + visual-weighted
        return rag.retrieve_multimodal(
            query=query,
            top_k=top_k,
        )
    
    elif modality == "complex":
        # Complex queries: handled by multi-hop decomposition
        # (see Section 3) — this just does an initial broad pass
        return rag.retrieve(
            query=query,
            top_k=top_k * 2,  # broader initial retrieval
        )
    
    else:  # text — the default
        return rag.retrieve(
            query=query,
            top_k=top_k,
        )
```

### 2.4 Implementation Pattern

The cleanest pattern is a simple pipeline that adds minimal overhead:

```python
@dataclass
class QueryClassification:
    modality: str = "text"  # text, visual, temporal, multimodal, complex, metadata
    query_time: Optional[float] = None
    sub_queries: Optional[List[str]] = None
    key_entities: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class RoutedResult:
    """Result of query routing — either a single retrieval or multi-hop chain."""
    chunks: List[RetrievedChunk]
    classification: QueryClassification
    retrieval_strategy: str  # name of strategy used
    sub_results: Optional[List["RoutedResult"]] = None  # for multi-hop


class QueryRouter:
    """
    Lightweight query router that classifies + routes queries.
    Uses the LLM to classify queries — no separate model needed.
    """
    
    def __init__(self, rag: VideoRAG, llm_call_fn: Callable):
        self.rag = rag
        self.llm_call_fn = llm_call_fn
    
    def route(self, query: str) -> RoutedResult:
        # Phase 1: Classify
        classification = self._classify(query)
        
        if classification.modality == "complex" and classification.sub_queries:
            # Phase 2b: Multi-hop for complex queries
            return self._execute_multi_hop(classification, query)
        else:
            # Phase 2a: Single retrieval for all other types
            chunks = route_retrieval(classification, query, self.rag)
            return RoutedResult(
                chunks=chunks,
                classification=classification,
                retrieval_strategy=classification.modality,
            )
```

### 2.5 Performance Considerations

| Query Type | Classification LLM Call | Retrieval Cost | Total Latency | vs Current Baseline |
|-----------|------------------------|----------------|---------------|-------------------|
| text | ~50ms (300 tokens, temp=0) | Same as now | ~+50ms | +15% |
| visual | ~50ms | Same + chunk weighting | ~+50ms | +15% |
| temporal | ~50ms | Same (TV-RAG already built) | ~+50ms | +15% |
| multimodal | ~50ms | BGE-VL composed (maybe +2 frames) | ~+50-100ms | +20% |
| complex | ~50ms | 2-5 sub-queries × retrieval | ~+500ms-2s | +100-400% |

**Key insight:** For 80% of queries (text, visual, temporal, multimodal), the overhead is just the LLM classification call (~50ms). The expensive path (complex) is rare but dramatically improves quality for hard questions.

**Optimization option:** Cache the classifier result for near-identical queries (e.g., follow-ups like "what about the blue one?" after "show me the red car").

---

## 3. Multi-Hop Query Decomposition

### 3.1 The Multi-Hop Pattern

Multi-hop decomposition breaks complex questions into a chain of simpler sub-questions, each answered by retrieval + reasoning, feeding the next step.

```
                                          ┌─────────────┐
User query ──→ Query Classifier ──→       │ Sub-questions│
 ("complex")                  │           │ (LLM decom-  │
                              │           │  position)   │
                              │           └──────┬───────┘
                              │          ┌───────┴────────┐
                              │          │  Sub-Q1 ──→ R1  │
                              │          │  Sub-Q2 ──→ R2  │──→ Aggregate ──→ Final Answer
                              │          │  Sub-Q3 ──→ R3  │
                              │          └────────────────┘
                              │
                              └──→ Single retrieval (for non-complex)
```

### 3.2 Decomposition Strategies

Three strategies for decomposing queries, chosen based on query structure:

#### Strategy A: Sequential Decomposition (most common)
Sub-questions are answered in order; later sub-questions can use results from earlier ones.

```python
# Example: "Why did the experiment fail after the temperature spike?"
# Step 1: "Find scenes showing the temperature spike" → [chunk at 12:30]
# Step 2: "What happens in the scenes after the temperature spike?" → [chunks at 13:00-15:00]
# Step 3: "Based on these, explain why the experiment failed" → final answer
```

#### Strategy B: Parallel Decomposition
Independent sub-questions answered simultaneously, then merged.

```python
# Example: "Compare the speaker's opening statement with their closing remarks"
# Parallel:
#   "What did the speaker say in the opening?" → R1
#   "What did the speaker say in closing?" → R2
# Merge: combine R1 + R2 → "Compare these two statements"
```

#### Strategy C: Recursive Decomposition
If a sub-question itself is complex, recursively decompose it. (Rare in practice for video Q&A — most complex queries only need 2-3 hops.)

### 3.3 Implementation Pattern

```python
QUERY_DECOMPOSITION_PROMPT = """You are decomposing a complex video analysis question into simpler sub-questions.
Each sub-question should be independently answerable from video content (transcripts, scene descriptions, objects, OCR).

Original question: {query}

Rules:
1. Output 2-4 sub-questions max (2-3 is typical)
2. Each sub-question must be answerable by retrieving video content
3. Sub-questions can depend on previous answers — order them logically
4. The final sub-question should synthesize previous answers
5. If the query doesn't need decomposition, return a single sub-question

Output as JSON:
{{
  "needs_decomposition": true/false,
  "sub_questions": [
    "Sub-question 1 targeting specific video content",
    "Sub-question 2 building on sub-question 1",
    "Sub-question 3 (optional, synthesis)"
  ],
  "reasoning": "Brief explanation of why this decomposition helps"
}}"""

MULTI_HOP_SYNTHESIS_PROMPT = """You are answering a complex video analysis question.
You have broken it into sub-questions and retrieved relevant content for each.

Original question: {query}

Sub-question results:
{sub_results}

Based on the retrieved video content for each sub-question, provide a comprehensive answer.
Include specific timestamps where relevant. If the evidence doesn't fully answer the question,
acknowledge what's known vs. uncertain.

Answer:"""
```

### 3.4 Execution Loop

```python
class MultiHopExecutor:
    """
    Executes multi-hop query decomposition.
    
    Flow:
    1. Decompose complex query into sub-questions (LLM call)
    2. For each sub-question (sequentially):
       a. Route + retrieve (using QueryRouter with sub-question)
       b. Store retrieved chunks + brief answer
    3. Synthesize all sub-question results into final answer (LLM call)
    """
    
    def __init__(self, router: QueryRouter, llm_call_fn: Callable):
        self.router = router
        self.llm_call_fn = llm_call_fn
    
    def execute(
        self,
        query: str,
        classification: QueryClassification,
        max_hops: int = 4,
    ) -> RoutedResult:
        sub_questions = classification.sub_queries or [query]
        sub_questions = sub_questions[:max_hops]
        
        all_chunks = []
        sub_results = []
        accumulated_context = ""
        
        for i, sub_q in enumerate(sub_questions):
            # Include previous results for context-dependent sub-questions
            enriched_query = self._enrich_sub_query(sub_q, accumulated_context)
            
            # Route and retrieve for this sub-question
            result = self.router.route(enriched_query)
            
            # Build context with previous answers
            partial_answer = self._summarize_sub_result(
                sub_q, result.chunks, accumulated_context
            )
            
            all_chunks.extend(result.chunks)
            sub_results.append({
                "sub_question": sub_q,
                "chunks": result.chunks,
                "partial_answer": partial_answer,
            })
            accumulated_context += f"\nSub-question {i+1}: {sub_q}\nResult: {partial_answer}\n"
        
        # Deduplicate chunks across hops
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique_chunks.append(c)
        
        return RoutedResult(
            chunks=unique_chunks,
            classification=classification,
            retrieval_strategy="multi_hop_decomposition",
            sub_results=sub_results,
        )
    
    def _enrich_sub_query(self, sub_q: str, context: str) -> str:
        """Enrich a sub-question with accumulated context from previous hops."""
        if not context:
            return sub_q
        return f"{sub_q}\n\n[Context from previous analysis: {context[:500]}]"
    
    def _summarize_sub_result(
        self, sub_q: str, chunks: List[RetrievedChunk], context: str
    ) -> str:
        """Quick summarization of a sub-question's retrieval results."""
        if not chunks:
            return "No relevant content found."
        
        snippets = [
            f"[{format_timestamp(c.timestamp)}] {c.text[:200]}"
            for c in chunks[:3]
        ]
        return "\n".join(snippets)
```

---

## 4. Lightweight Python Libraries & Patterns

### No Dedicated Library Needed

The **key insight** is that neither query classification/routing nor multi-hop decomposition requires a dedicated library. The patterns are simple enough to implement in <200 lines of Python using:

1. **LLM structured output** — via Hermes CLI (`hermes chat -q`) with JSON parsing
2. **Existing embedding model** (BGE-VL) — handles text, image, and composed retrieval
3. **Existing ChromaDB** — with metadata filters for chunk_type weighting

### What Others Use

| Library/Framework | Approach | Pros | Cons | Verdict |
|------------------|----------|------|------|---------|
| **LlamaIndex Routers** | `RouterQueryEngine` with `LLMSingleSelector` | Full framework, handles routing + decomposition | Heavy dependency, couples to LlamaIndex abstractions | Overkill for this project |
| **LangChain** | `RunnableBranch` + `create_route_node` | Flexible, works with any LLM | Forces LangChain dependency chain | Overkill — LangChain isn't used here |
| **DSPy** | Programmatic optimizers + signatures | No prompt engineering, self-optimizing | Different paradigm, learning curve | Interesting but not needed |
| **DIY** (this plan) | Single LLM call + JSON parse + if/else routing | ~150 lines, zero deps, debuggable | Must write/maintain prompts | **Best fit** for this codebase |

### Minimal Dependency Pattern

The entire router can be implemented with **zero new dependencies**:

```python
# What you need:
# 1. json (stdlib) — for parsing structured output
# 2. dataclasses (stdlib) — for type-safe classification
# 3. re (stdlib) — for optional timestamp extraction fallback
# 4. Existing: hermes CLI, BGE-VL, ChromaDB — already in the project

# No new pip installs required.
```

### Timestamp Extraction Pattern (Optional — can replace LLM call for this)

If you want to avoid the LLM call just for timestamp extraction, a regex approach works for common patterns:

```python
import re

TIMESTAMP_PATTERNS = [
    (r"(?:at|around|about)\s*(\d+):(\d+)(?::(\d+))?", lambda m: 
        int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)),
    (r"(\d+):(\d+)(?::(\d+))?\s*(?:timestamp|mark|point)", lambda m:
        int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)),
    (r"(?:(\d+)\s*(?:hour|hr)s?\s*)?(?:(\d+)\s*(?:min|minute)s?\s*)?(?:(\d+)\s*(?:sec|second)s?)?", ...),
]

def extract_timestamp(query: str) -> Optional[float]:
    """Extract timestamp from query using regex patterns."""
    for pattern, converter in TIMESTAMP_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return converter(match)
    return None
```

But the LLM approach is simpler, handles more edge cases, and the classification is needed anyway — so bundling timestamp extraction into the same LLM call is more efficient.

---

## 5. Integration Into Existing Codebase

### 5.1 New Module: `video_analysis/router.py`

This is the main new file (~200-250 lines).

```python
"""
Query routing and multi-hop decomposition for video RAG.

Routes user queries to optimal retrieval strategies based on
modality classification, and handles multi-hop decomposition
for complex queries.

Pattern: Single LLM call for classification → dispatch to retrieval strategy
No additional classification models needed.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from video_analysis.rag import VideoRAG, RetrievedChunk
from video_analysis.models import format_timestamp

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class QueryClassification:
    """Result of classifying a user query."""
    modality: str = "text"         # text, visual, temporal, multimodal, complex, metadata
    query_time: Optional[float] = None
    sub_queries: Optional[List[str]] = None
    key_entities: List[str] = field(default_factory=list)
    confidence: float = 0.5
    raw: str = ""                  # raw LLM response for debugging


@dataclass
class SubQueryResult:
    """Result of a single sub-query in multi-hop decomposition."""
    sub_question: str
    chunks: List[RetrievedChunk]
    partial_answer: str = ""


@dataclass
class RoutedResult:
    """Result of query routing — single or multi-hop."""
    chunks: List[RetrievedChunk]
    classification: QueryClassification
    retrieval_strategy: str = "standard"
    sub_results: List[SubQueryResult] = field(default_factory=list)


# ──────────────────────────────────────────────
# LLM prompt for classification + decomposition
# ──────────────────────────────────────────────

CLASSIFICATION_PROMPT = """You are a query classifier for a video analysis system. 
Analyze the user's query and output ONLY valid JSON (no markdown, no explanation).

{{
  "modality": "text|visual|temporal|multimodal|complex|metadata",
  "query_time": null or number (seconds, e.g. 330.0 for "at 5:30"),
  "sub_questions": null or ["q1", "q2", "q3"],
  "key_entities": ["entity1", "entity2"],
  "confidence": 0.0 to 1.0
}}

Classification rules:
- **text**: Asks about spoken content, dialogue, narration, explanations
- **visual**: Asks about objects, scenes, colors, actions, "show me", "find scenes with"
- **temporal**: References time ("at X:XX", "before", "after", "during") or sequence
- **multimodal**: Combines visual + textual elements ("person wearing X saying Y")
- **complex**: Requires multi-step reasoning, comparison, "why", "how", "compare", "summarize overall"
- **metadata**: About video properties (duration, filename)

If the query is complex, provide 2-4 sub_questions that break it into simpler retrievable steps.
The final sub_question should synthesize previous answers.

Query: {query}

JSON:"""


# ──────────────────────────────────────────────
# Router implementation
# ──────────────────────────────────────────────

class QueryRouter:
    """
    Classifies user queries and routes them to optimal retrieval strategies.
    
    Uses a single lightweight LLM call for both modality classification
    and (for complex queries) sub-question decomposition.
    """
    
    def __init__(
        self,
        rag: VideoRAG,
        llm_call_fn: Callable,
        enable_classification: bool = True,
    ):
        self.rag = rag
        self.llm_call_fn = llm_call_fn
        self.enable_classification = enable_classification
    
    def route(self, query: str, top_k: int = 20) -> RoutedResult:
        """Classify and route a query to the optimal retrieval strategy."""
        
        if not self.enable_classification:
            # Bypass: use standard text retrieval (legacy behavior)
            chunks = self.rag.retrieve(query, top_k=top_k)
            return RoutedResult(
                chunks=chunks,
                classification=QueryClassification(modality="text"),
            )
        
        # Phase 1: Classify
        classification = self._classify(query)
        
        # Phase 2: Route
        if classification.modality == "complex" and classification.sub_queries:
            return self._route_complex(classification, query, top_k)
        elif classification.modality == "visual":
            return self._route_visual(classification, query, top_k)
        elif classification.modality == "temporal":
            return self._route_temporal(classification, query, top_k)
        elif classification.modality == "multimodal":
            return self._route_multimodal(classification, query, top_k)
        elif classification.modality == "metadata":
            return self._route_metadata(classification)
        else:
            return self._route_text(classification, query, top_k)
    
    def _classify(self, query: str) -> QueryClassification:
        """Classify query via LLM call."""
        try:
            prompt = CLASSIFICATION_PROMPT.format(query=query)
            response = self.llm_call_fn(prompt, max_tokens=300, temperature=0.0)
            
            # Clean response — strip markdown fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
                cleaned = cleaned.rsplit("```", 1)[0] if "```" in cleaned else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            
            result = json.loads(cleaned)
            return QueryClassification(
                modality=result.get("modality", "text"),
                query_time=result.get("query_time"),
                sub_queries=result.get("sub_questions"),
                key_entities=result.get("key_entities", []),
                confidence=result.get("confidence", 0.5),
                raw=response,
            )
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning(f"Query classification failed: {e}")
            return QueryClassification(modality="text")
    
    def _route_text(self, classification, query, top_k):
        """Route: standard text retrieval (default)."""
        chunks = self.rag.retrieve(query, top_k=top_k)
        return RoutedResult(chunks=chunks, classification=classification, retrieval_strategy="text")
    
    def _route_visual(self, classification, query, top_k):
        """Route: visual retrieval — boost frame chunks."""
        chunks = self.rag.retrieve(
            query,
            top_k=top_k,
            boost_chunk_types={"frame": 1.5, "scene": 1.2},
        )
        return RoutedResult(chunks=chunks, classification=classification, retrieval_strategy="visual")
    
    def _route_temporal(self, classification, query, top_k):
        """Route: temporal-aware retrieval with TV-RAG decay."""
        chunks = self.rag.retrieve(
            query,
            query_time=classification.query_time,
            top_k=top_k,
        )
        # Extra temporal context expansion
        if chunks:
            chunks = self.rag.expand_temporal_context(chunks, chunks[0].video_id)
        return RoutedResult(chunks=chunks, classification=classification, retrieval_strategy="temporal")
    
    def _route_multimodal(self, classification, query, top_k):
        """Route: multimodal — try composed BGE-VL retrieval."""
        chunks = self.rag.retrieve_multimodal(query, top_k=top_k)
        if not chunks:
            chunks = self.rag.retrieve(query, top_k=top_k)
        return RoutedResult(chunks=chunks, classification=classification, retrieval_strategy="multimodal")
    
    def _route_metadata(self, classification):
        """Route: metadata query — signals chat.py to handle directly."""
        return RoutedResult(chunks=[], classification=classification, retrieval_strategy="metadata")
    
    def _route_complex(self, classification, query, top_k):
        """Route: multi-hop decomposition for complex queries."""
        executor = MultiHopExecutor(self)
        result = executor.execute(query, classification, top_k)
        result.classification = classification
        return result


# ──────────────────────────────────────────────
# Multi-hop executor
# ──────────────────────────────────────────────

class MultiHopExecutor:
    """
    Executes multi-hop query decomposition.
    
    For each sub-question:
    1. Route + retrieve
    2. Summarize findings
    3. Feed into next sub-question
    
    After all hops: aggregate chunks for final LLM answer.
    """
    
    def __init__(self, router: QueryRouter):
        self.router = router
    
    def execute(
        self,
        query: str,
        classification: QueryClassification,
        top_k: int = 20,
        max_hops: int = 4,
    ) -> RoutedResult:
        sub_questions = (classification.sub_queries or [query])[:max_hops]
        
        all_chunks = []
        sub_results = []
        accumulated_context = ""
        
        for i, sub_q in enumerate(sub_questions):
            # Enrich with context from previous hops
            enriched = sub_q
            if accumulated_context:
                enriched = f"{sub_q}\n\nPrevious findings: {accumulated_context[:500]}"
            
            # Route without complex (avoid recursion)
            simple_class = QueryClassification(modality="text")
            result = self.router._route_text(simple_class, enriched, top_k)
            
            # Build partial summary
            partial = self._summarize(sub_q, result.chunks)
            
            all_chunks.extend(result.chunks)
            sub_results.append(SubQueryResult(
                sub_question=sub_q,
                chunks=result.chunks,
                partial_answer=partial,
            ))
            accumulated_context += f"\nQ{i+1}: {sub_q}\nA{i+1}: {partial}\n"
        
        # Deduplicate chunks
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique_chunks.append(c)
        
        return RoutedResult(
            chunks=unique_chunks,
            classification=classification,
            retrieval_strategy="multi_hop",
            sub_results=sub_results,
        )
    
    def _summarize(self, sub_q: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant content found for this sub-question."
        snippets = [
            f"[{format_timestamp(c.timestamp)}] {c.text[:200]}"
            for c in chunks[:3]
        ]
        return "\n".join(snippets)
```

### 5.2 Modifications to `video_analysis/chat.py`

The `VideoChat.ask()` method needs minimal changes — integrate router before retrieval:

```python
class VideoChat:
    def __init__(self, rag: VideoRAG, config: Optional[Config] = None):
        self.rag = rag
        self.config = config or Config()
        self.history: List[ChatMessage] = []
        self.router = QueryRouter(
            rag=rag,
            llm_call_fn=self._call_llm_with_temperature,
            enable_classification=config.query_classification_enabled,
        )
    
    def _call_llm_with_temperature(self, prompt: str, **kwargs) -> str:
        """Call LLM with configurable params (used by router for classification)."""
        max_tokens = kwargs.get("max_tokens", 300)
        temperature = kwargs.get("temperature", 0.0)
        result = subprocess.run(
            ["hermes", "chat", "-q", "-m", self.config.llm_model,
             "-t", str(temperature), "--max-tokens", str(max_tokens)],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        raise RuntimeError(f"LLM call failed: {result.stderr[:200]}")
    
    def ask(self, query: str, video_id: Optional[str] = None) -> ChatMessage:
        # Step 1: Route the query (classify + retrieve)
        routed = self.router.route(query)
        
        if routed.retrieval_strategy == "metadata":
            # Handle metadata directly
            answer = self._answer_metadata(query, video_id)
            message = ChatMessage(role="assistant", content=answer, sources=[])
            self.history.append(ChatMessage(role="user", content=query))
            self.history.append(message)
            return message
        
        if not routed.chunks:
            return ChatMessage(
                role="assistant",
                content="I couldn't find relevant information about that.",
                sources=[],
            )
        
        # Expand temporal context for non-multi-hop results
        if video_id and routed.retrieval_strategy != "multi_hop":
            routed.chunks = self.rag.expand_temporal_context(routed.chunks, video_id)
        
        # Build context and prompt
        context = self.rag.build_context(routed.chunks)
        
        # For multi-hop results, include sub-question chain
        if routed.sub_results:
            context += "\n\n[Sub-question Analysis]\n"
            for sr in routed.sub_results:
                context += f"Q: {sr.sub_question}\nA: {sr.partial_answer}\n\n"
        
        prompt = self._build_prompt(query, context)
        answer = self._call_llm(prompt)
        sources = self.rag.get_source_citations(routed.chunks)
        
        message = ChatMessage(role="assistant", content=answer, sources=sources)
        self.history.append(ChatMessage(role="user", content=query))
        self.history.append(message)
        return message
```

### 5.3 Modifications to `video_analysis/rag.py`

Two small additions:

1. **`chunk_type boosting`** in `retrieve()` — allow boosting scores for specific chunk types:

```python
def retrieve(
    self,
    query: str,
    video_id: Optional[str] = None,
    top_k: Optional[int] = None,
    query_time: Optional[float] = None,
    boost_chunk_types: Optional[dict] = None,  # NEW
) -> List[RetrievedChunk]:
    # ... existing retrieval logic ...
    
    # Apply chunk_type boosting (NEW)
    for chunk in chunks:
        if boost_chunk_types and chunk.chunk_type in boost_chunk_types:
            chunk.score *= boost_chunk_types[chunk.chunk_type]
    
    # ... rest of existing logic ...
```

2. **`retrieve_multimodal()`** — new method for composed image+text retrieval:

```python
def retrieve_multimodal(
    self,
    query: str,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Multimodal retrieval using BGE-VL composed embedding.
    Currently uses text-only embedding since we don't have a user-provided image,
    but scores frame chunks higher since visual queries benefit from frame content.
    """
    return self.retrieve(
        query,
        top_k=top_k or self.config.top_k_retrieval,
        boost_chunk_types={"frame": 1.5, "scene": 1.2},
    )
```

### 5.4 Config Additions

Add to `video_analysis/config.py`:

```python
# Query routing (v0.13.0)
query_classification_enabled: bool = bool(
    os.environ.get("QUERY_CLASSIFICATION_ENABLED", "true").lower() == "true"
)  # LLM-based query classification before retrieval
multi_hop_enabled: bool = bool(
    os.environ.get("MULTI_HOP_ENABLED", "true").lower() == "true"
)  # Multi-hop decomposition for complex queries
```

---

## 6. Latency Budget Analysis

### Current Pipeline Baseline

| Step | Time | Notes |
|------|------|-------|
| BGE-VL text embedding | ~20ms | GPU, batch=1 |
| ChromaDB query (top_k=20) | ~5ms | HNSW index |
| Cross-encoder re-rank | ~50ms | MiniLM on GPU |
| TV-RAG temporal expansion | ~10ms | Scene fetch from Chroma |
| **Total retrieval** | **~85ms** | |
| LLM answer generation | ~2-10s | DeepSeek-V4-Flash |
| **Total end-to-end** | **~2-10s** | |

### With Query Classification & Routing (non-complex queries)

| Step | Time | Delta |
|------|------|-------|
| LLM classification call | ~50ms | **+50ms** |
| Routing dispatch | ~1ms | +1ms |
| Retrieval (same as above) | ~85ms | 0 |
| **Total retrieval** | **~136ms** | **+51ms** |
| LLM answer generation | ~2-10s | 0 |
| **Total end-to-end** | **~2-10s** | **+~5%** |

### With Multi-Hop Decomposition (complex queries, ~10-20% of queries)

| Step | Time | Notes |
|------|------|-------|
| LLM classification + decomposition | ~80ms | Same call, generates sub-questions |
| Hop 1: retrieve + summarize | ~150ms | |
| Hop 2: retrieve + summarize | ~150ms | Context feeds forward |
| Hop 3: retrieve + summarize | ~150ms | |
| Deduplication | ~5ms | |
| **Total retrieval** | **~535ms** | **+450ms** |
| Synthesis LLM call | ~3-10s | Longer due to richer context |
| **Total end-to-end** | **~3.5-10.5s** | **+~35%** for complex only |

### Optimization Options

1. **Parallel sub-questions** — If sub-questions are independent (Strategy B), execute them concurrently instead of sequentially. Cuts ~450ms → ~200ms for 3-hops.
2. **Skip classifier for simple follow-ups** — If the user's query is a short follow-up ("and the blue one?"), use the previous classification.
3. **Cache classification** — Session-level cache for identical queries within the same conversation.

---

## 7. Alternatives Considered

### Alternative 1: Embedding-Based Classifier (No LLM)

**Approach:** Use sentence embeddings + KNN or logistic regression to classify query modality.

**Pros:**
- ~5ms instead of ~50ms
- Deterministic, reproducible
- No LLM call needed

**Cons:**
- Requires labeled training data (modality labels for queries)
- Can't extract timestamps or sub-questions
- Brittle to novel query phrasing
- Overhead of maintaining a classifier model

**Verdict:** Not worth it — the LLM approach is more flexible, and 50ms is negligible compared to 2-10s answer generation.

### Alternative 2: Rule-Based Routing Only

**Approach:** Use keyword/regex rules to detect modality without any LLM call.

**Pros:**
- Zero additional latency
- Extremely simple

**Cons:**
- Fragile — misses subtle phrasing
- Can't handle complex multi-hop queries
- No timestamp extraction for natural language ("right after the explosion")

**Verdict:** Good as a fallback or pre-filter, but insufficient as the primary approach.

### Alternative 3: Dedicated Small Classification Model

**Approach:** Fine-tune a BERT-sized classifier (110M params) on query modality data.

**Pros:**
- Fast (~10ms)
- Open-source

**Cons:**
- Need to create/modify training data
- Additional model = more VRAM pressure (12 GB constraint)
- Can't decompose queries or extract timestamps
- Doesn't scale to new modalities without retraining

**Verdict:** Over-engineered for this use case.

---

## 8. Recommendations

### Recommended Implementation

1. **Use LLM-based classification** (DeepSeek-V4-Flash via Hermes) — single call classifies modality, extracts timestamps, and generates sub-questions for complex queries.

2. **Start with the router as a thin wrapper** — the `QueryRouter` class is self-contained (~200 lines), no new dependencies, and can be disabled via config for zero behavior change.

3. **Add `retrieve_multimodal()` and `boost_chunk_types` to `rag.py`** — minimal changes to existing code.

4. **Integrate into `chat.py`** — replace the current `rag.retrieve()` call with `router.route()`.

5. **Implement multi-hop as a progressive enhancement** — start with non-complex routing (handles 80% of queries with ~50ms overhead), then add multi-hop for complex queries.

### Effort Estimates

| Task | Files | Effort | Priority |
|------|-------|--------|----------|
| Create `video_analysis/router.py` (~200 lines) | New file | 1h | P0 |
| Add `chunk_type boosting` + `retrieve_multimodal()` to `rag.py` | `rag.py` | 15min | P0 |
| Update `chat.py` to use router | `chat.py` | 15min | P0 |
| Add config fields | `config.py` | 5min | P0 |
| Add `boost_chunk_types` to `VideoRAG.retrieve()` signature | `rag.py` | 5min | P0 |
| **Total** | **5 files** | **~1.5h** | |

### Architecture Diagram (Final)

```
User Query
  │
  ├──→ QueryRouter.classify() ──→ LLM (DeepSeek-V4-Flash)
  │     returns: {modality, query_time, sub_questions, entities}
  │
  ├── text ─────────→ VideoRAG.retrieve() ──→ ChromaDB ──→ Re-rank
  ├── visual ───────→ VideoRAG.retrieve(boost=frame) ──→ ChromaDB ──→ Re-rank
  ├── temporal ─────→ VideoRAG.retrieve(query_time=) ──→ ChromaDB + TV-RAG ──→ Expand ──→ Re-rank
  ├── multimodal ───→ VideoRAG.retrieve_multimodal() ──→ ChromaDB (frame-weighted) ──→ Re-rank
  ├── metadata ─────→ Direct answer (no retrieval)
  │
  └── complex ──────→ MultiHopExecutor.execute()
                        ├── Hop 1: sub_q1 → retrieve → summarize
                        ├── Hop 2: sub_q2 + context → retrieve → summarize
                        ├── Hop 3: sub_q3 + context → retrieve → summarize
                        └── Aggregate all chunks → LLM synthesis
                              │
                              └──→ Answer with citations
```

---

## Key Sources

- **Multi-RAG (arxiv 2407.15829)** — Multi-query routing for domain-specific RAG
- **Corrective RAG (arxiv 2401.15884)** — Self-correction feedback loop for RAG quality
- **Self-RAG (arxiv 2310.11511)** — On-demand retrieval and self-reflection
- **LlamaIndex Router System** — `RouterQueryEngine` / `LLMSingleSelector` pattern
- **LangChain RunnableBranch** — Conditional routing based on LLM output
- **VGent (NeurIPS 2025)** — Graph-based video RAG with multi-hop reasoning (arxiv 2510.14032)
- **ViG-RAG (AAAI 2026)** — Temporal + semantic hybrid graph reasoning for video
