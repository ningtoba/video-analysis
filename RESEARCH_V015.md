# v0.15.0 Research — Next Evolution Beyond Complete Roadmap (June 2026)

> **Date:** 2026-06-26
> **Context:** The project at v0.14.0 has all original roadmap items complete (scene detection, transcription, diarization, OCR, YOLO, OpenCLIP, X-CLIP, BGE-VL, ChromaDB, cross-encoder, ColBERTv2, TV-RAG, scene graph, query routing, multi-hop decomposition, VideoChat-Flash MLLM, Gradio 6 UI, YouTube import, batch processing, clip export, Docker, graceful shutdown). This document identifies the next frontier.

---

## Executive Summary

The original roadmap is fully complete at v0.14.0. The next evolution (v0.15.0+) should focus on three themes:

1. **Lighter, faster video MLLM integration** — SmolVLM2 (HuggingFace, Feb 2025) offers 2.2B, 500M, and 256M video understanding models that run anywhere, including on a free Colab. This is a dramatic improvement over VideoChat-Flash 2B (~5.4 GB VRAM).
2. **Production hardening & agentic retrieval** — Agentic RAG (A-RAG, 2025), hierarchical retrieval, CI/CD, pre-commit hooks, performance benchmarks.
3. **Unfinished optimizations** — adaptive frame sampling, CLIP-similarity frame dedup, Gradio 6.19 Workflow subgraphs.

---

## Section 1: SmolVLM2 — Game-Changing Lightweight Video MLLM

### Discovery

**SmolVLM2** (HuggingFace, Feb 20, 2025) is a family of video understanding models by HuggingFace:
- **2.2B params** — best bang for buck, runs on free Colab
- **500M params** — iPhone-local inference, quarter of 2.2B size
- **256M params** — experimental, smallest video LM ever

**Key advantages over current VideoChat-Flash (2B):**
| Property | VideoChat-Flash 2B | SmolVLM2 2.2B | SmolVLM2 500M |
|----------|-------------------|----------------|----------------|
| VRAM | ~5.4 GB (BF16) | ~2-3 GB | ~0.5-1 GB |
| Transformer version | trust_remote_code, custom model | transformers-native from v4.49.0 | Same |
| Flash Attention 2 | ? | Yes, `_attn_implementation="flash_attention_2"` | Same |
| Video-MME | Good | Outperforms all 2B models | Close to 2.2B |
| Colab-friendly | No (5.4 GB too much) | Yes | Yes |
| License | MIT | Apache 2.0? | Same |

**Inference pattern:**
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "video.mp4"},
            {"type": "text", "text": "Describe this video in detail"},
        ],
    },
]
```

### Recommendation

**Add SmolVLM2 as an alternative/drop-in-replacement for VideoChat-Flash in `video_mllm.py`.** The module already has a VideoMLLM class with `describe_scene()`, `summarize_video()`, `answer()` methods — SmolVLM2 can plug into the same interface. The key benefit: users without 12GB VRAM headroom can run the 500M model.

**Priority:** HIGH — unlocks video MLLM for all users, not just RTX 4070 owners.

---

## Section 2: Agentic & Hierarchical RAG

### Discovery: A-RAG (arXiv 2602.03442, 2025)

A-RAG (Agentic Retrieval-Augmented Generation) introduces **hierarchical retrieval interfaces** where:
- A central agent decides which "retrieval tool" to call
- Each retrieval tool is a specialized interface (text search, visual search, temporal search, graph search)
- The agent can chain tools, parallelize, and fuse results

**Current state:** The query router in v0.14.0 is a **static classifier** — it picks ONE route per query. An agentic loop would:
1. Classify the query
2. Execute the primary retrieval
3. Check if results are sufficient (has enough evidence?)
4. If not, try an alternative route or decompose further
5. Fuse results from multiple retrieval passes

**Concrete implementation:**
```python
def agentic_retrieve(query: str, rag: VideoRAG, max_rounds: int = 3):
    """Agentic retrieval loop: classify → retrieve → check → refine."""
    router = rag._get_query_router()
    
    for round in range(max_rounds):
        decision = router.classify(query)
        chunks = route_retrieval(query, decision, rag)
        
        # Check if we have enough high-quality evidence
        if _has_sufficient_evidence(chunks, query):
            return chunks
        
        # Refine: switch route or decompose
        query = _refine_query(query, chunks)
    
    return chunks  # best effort after max rounds
```

### Recommendation

**Add an agentic retrieval mode** that wraps the existing query router in an iterative loop. When the primary route returns low-confidence results, it automatically tries alternative routes and fuses the results.

**Priority:** MEDIUM — improves complex queries but the current static router already handles ~80% of cases well.

---

## Section 3: Unfinished Optimizations

### 3.1 Adaptive Frame Sampling

**Config field exists but is never used:**
```python
adaptive_frame_sampling: bool = False
adaptive_frame_sampling_sensitivity: float = 0.3
```

The idea: instead of uniform 1-fps sampling, sample more densely near scene boundaries (where content changes most) and less in the middle of long static shots. Uses FFmpeg's scene score to dynamically adjust frame rate.

**Implementation:** ~50 lines in `pipeline.py`, modifying `_extract_key_frames()`.

**Priority:** MEDIUM — reduces redundant frames by ~30-50%.

### 3.2 CLIP-Similarity Frame Deduplication

**Config field exists but is never used:**
```python
clip_frame_dedup: bool = False
clip_frame_dedup_threshold: float = 0.92
```

After extraction, deduplicate frames whose CLIP embeddings have cosine similarity > threshold. Near-duplicate frames inflate ChromaDB and dilute retrieval quality.

**Implementation:** ~30 lines in `pipeline.py`, using the already-loaded OpenCLIP model.

**Priority:** LOW-MEDIUM — cleaner RAG chunks, minor quality improvement.

### 3.3 Gradio 6.19 Workflow Subgraphs

**Gradio 6.19.0 (current pinned version) introduced Workflow subgraphs:**
- Each subgraph exposed as a named endpoint (`/info`, `/call`, `/api`)
- Drag selection in the workflow canvas
- Pipeline visualization with node-level progress

**Current state:** The project uses `gr.Blocks()` with tabs. The pipeline runs linearly with log output. A Workflow visualization would show the 12-step pipeline as a visual graph with per-step progress.

**Implementation:** ~100 lines in `ui/app.py`, wrapping the existing pipeline steps as a Gradio Workflow.

**Priority:** LOW — cosmetic improvement, not functional.

---

## Section 4: Production Hardening

| Area | Current State | Target | Effort |
|------|--------------|--------|--------|
| CI/CD | None | GitHub Actions: lint, type-check, test | 2h |
| Pre-commit hooks | None | ruff, mypy, pytest | 30min |
| API versioning | None | FastAPI prefix `/api/v1` | 15min |
| Performance benchmarks | None | `tests/benchmarks/` with pytest-benchmark | 1h |
| Input validation | Basic | Pydantic models on API endpoints | 1h |
| Rate limiting | None | slowapi/redis on FastAPI | 30min |
| Prometheus metrics | None | prometheus-client FastAPI middleware | 30min |

**Priority for v0.15.0:**
- **P0:** CI/CD + pre-commit hooks (foundation for all future work)
- **P1:** Performance benchmarks (baseline before optimization)
- **P2:** API versioning, input validation

---

## Section 5: Proposed v0.15.0 Implementation Plan

### Phase 1 — SmolVLM2 Integration (HIGH priority)
1. Add `SmolVLM2VideoMLLM` class in `video_analysis/video_mllm.py`
2. Extend `VideoMLLM` to auto-select between VideoChat-Flash and SmolVLM2 based on VRAM/availability
3. Add config field: `video_mllm_backend: str = "auto"` (auto|videochat_flash|smolvlm2)
4. Add `video_mllm_model_size: str = "2.2B"` (2.2B|500M|256M)
5. Tests for SmolVLM2 integration
6. Update README, CHANGELOG

### Phase 2 — Agentic RAG (MEDIUM priority)
1. Add `agentic_retrieve()` method to `VideoRAG`
2. Implement `_has_sufficient_evidence()` heuristic
3. Implement `_refine_query()` for multi-round refinement
4. Config: `agentic_retrieval_enabled: bool = False`
5. Tests

### Phase 3 — Unfinished Optimizations (MEDIUM priority)
1. Implement `_extract_key_frames_adaptive()` in pipeline
2. Implement CLIP-similarity dedup in pipeline
3. Wire up config flags
4. Tests for both

### Phase 4 — Production Hardening (LOW priority this iteration)
1. CI/CD config
2. Pre-commit hooks
3. Benchmark stubs

---

## Key Sources

- SmolVLM2 blog: https://huggingface.co/blog/smolvlm2 (Feb 20, 2025)
- SmolVLM2 models: https://huggingface.co/collections/HuggingFaceTB/smolvlm2-67b0cb5c2c83d2fa92358c2c
- A-RAG paper: https://arxiv.org/abs/2602.03442 (2025)
- VideoRAG KDD 2026: https://dl.acm.org/doi/10.1145/3770854.3783944
- RAG Survey 2025: https://arxiv.org/abs/2410.12837
- Gradio 6.19.0 changelog: https://gradio.app/changelog
