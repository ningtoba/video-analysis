# v0.15.0 Deep-Dive Research — SmolVLM2, Agentic RAG & Production Hardening

> **Date:** 2026-06-26 (Iteration #1, Worker #0)  
> **Project:** video-analysis (v0.14.0 → v0.15.0)  
> **Hardware:** RTX 4070 (12 GB VRAM), CachyOS  
> **LLM:** DeepSeek-V4-Flash via Hermes Agent

---

## Executive Summary

This research confirms the v0.15.0 direction proposed in `RESEARCH_V015.md` and adds new findings from targeted web extraction of HuggingFace model pages, Gradio documentation, and production hardening research papers. The three priority areas remain:

1. **SmolVLM2 integration** (HIGH) — unlocks video MLLM at 500M and 2.2B sizes, both Apache 2.0
2. **Agentic RAG** (MEDIUM) — wraps existing query router in an iterative retrieval loop
3. **Production hardening** (MEDIUM) — CI/CD, pre-commit, benchmarks, infrastructure cleanup

Additionally, this researcher discovered that **adaptive frame sampling and CLIP-similarity frame dedup are already fully implemented** in `pipeline.py` (lines 527-646). They just need to be enabled and tested.

---

## Section 1: SmolVLM2 — Verified Specifications

### Model Availability & VRAM

| Model | Params | VRAM (BF16) | VRAM (int8) | Video-MME | MLVU | MVBench | License |
|-------|--------|-------------|-------------|-----------|------|---------|---------|
| **SmolVLM2-2.2B-Instruct** | 2.2B | ~5.2 GB | ~2.6 GB | **52.1** | 55.2 | 46.27 | Apache 2.0 |
| **SmolVLM2-500M-Video-Instruct** | 500M | ~1-2 GB | ~0.5-1 GB | **42.2** | 47.3 | 39.73 | Apache 2.0 |
| **SmolVLM2-256M-Video-Instruct** | 256M | ~0.5-1 GB | ~256 MB | **33.7** | 40.6 | 32.7 | Apache 2.0 |

### Comparison with Current VideoChat-Flash

| Property | VideoChat-Flash 2B | SmolVLM2 2.2B | SmolVLM2 500M |
|----------|-------------------|----------------|----------------|
| VRAM | ~5.4 GB (BF16) | ~5.2 GB (BF16) | ~1-2 GB (BF16) |
| Loading API | `AutoModel` + `trust_remote_code=True` | `AutoModelForImageTextToText` (transformers-native) | Same |
| Required transformers | Custom fork | ≥v4.49.0 | Same |
| Flash Attention 2 | Requires config | `_attn_implementation="flash_attention_2"` | Same |
| Video input | Custom processor API | Chat template (`{"type": "video", "path": "..."}`) | Same |
| Video dependency | Custom | **decord** (`pip install decord`) | Same |
| License | MIT | Apache 2.0 | Apache 2.0 |
| Native video handling | Yes (hierarchical compression) | Yes (10 FPS uniform sampling) | Yes |
| Image understanding | Good | Better (MathVista 51.5, MMMU 42) | Adequate |

### Key Integration Facts

1. **SmolVLM2 uses `AutoModelForImageTextToText`** — the standard transformers class, not `trust_remote_code`. This means it's compatible with the latest transformers without custom code.
2. **Video input via chat templates** — no custom processor calls needed. The pattern is:
   ```python
   processor.apply_chat_template(messages, ...)
   ```
3. **Requires `decord` package** for video decoding. Add to requirements.txt.
4. **500M model can run on CPU** in a pinch (~2-3s per 32-frame video).
5. **The 2.2B model outperforms VideoChat-Flash** on Video-MME (52.1 vs ~50) despite being similar VRAM.
6. **SmolVLM2 500M is the real game-changer** — at ~1-2 GB VRAM, it runs alongside the existing pipeline without needing to unload other models.

### Integration Architecture

```
VideoMLLM (abstract wrapper)
├── _backend: str  # "auto" | "videochat_flash" | "smolvlm2"
├── _model_size: str  # "2.2B" | "500M" | "256M" (only for smolvlm2)
│
├── describe_scene(frames) → str
│   ├── VideoChatFlashBackend.describe_scene()
│   └── SmolVLM2Backend.describe_scene()
│
├── summarize_video(video_path) → str
│   ├── VideoChatFlashBackend.summarize_video()
│   └── SmolVLM2Backend.summarize_video()
│
└── answer(query, frames, video_path) → str
    ├── VideoChatFlashBackend.answer()
    └── SmolVLM2Backend.answer()
```

---

## Section 2: Agentic RAG — Verified Feasibility

### Current State

The project already has at v0.14.0:
- ✅ Query router (4 routes: text/visual/temporal/multimodal)
- ✅ Multi-hop decomposition (sub-questions → retrieve → merge → re-rank)
- ✅ Scene graph (K-hop expansion)
- ✅ TV-RAG temporal decay
- ✅ Cross-encoder + optional ColBERTv2 re-ranking

### What Agentic RAG Adds

The current query router is **single-pass**: it classifies the query, executes one route, and returns. Agentic RAG makes this iterative:

```
Input: Query
├── Round 1: Classify → Retrieve → Score Results
│   ├── Confidence ≥ threshold? → Return
│   └── Confidence < threshold? → Go to Round 2
├── Round 2: Refine Query → Alternative Route → Retrieve → Score
│   ├── Confidence ≥ threshold? → Return
│   └── Still poor? → Fuse Round 1 + Round 2 results → Return
└── Best effort after max_rounds
```

### Implementation Plan

```python
def agentic_retrieve(
    self, query: str, video_id: str = None,
    max_rounds: int = 3, min_confidence: float = 0.5
) -> List[RetrievedChunk]:
    all_chunks = []
    current_query = query
    router = self._get_query_router()

    for round_idx in range(max_rounds):
        # Classify current query
        decision = router.classify_and_decompose(current_query)
        # Execute retrieval
        chunks = self.retrieve(current_query, video_id=video_id)
        all_chunks.extend(chunks)

        # Check confidence: avg score of top-k chunks
        if chunks:
            avg_score = sum(c.score for c in chunks[:3]) / min(3, len(chunks))
            if avg_score >= min_confidence:
                break  # Good enough!

        # Refine: try a different route or decomposition
        if round_idx == 0 and decision.sub_queries:
            # Try multi-hop if we haven't yet
            chunks = self._multi_hop_retrieve(
                query, decision.sub_queries, video_id
            )
            all_chunks.extend(chunks)
        elif round_idx == 1:
            # Try scene-graph expansion
            sg = self._get_scene_graph()
            if sg:
                chunks = sg.expand_chunks(chunks)
                all_chunks.extend(chunks)

    # Deduplicate and re-rank
    return self._deduplicate_and_rerank(all_chunks, query, top_k)
```

### New Config Fields

```python
agentic_retrieval_enabled: bool = False  # Enable agentic retrieval loop
agentic_max_rounds: int = 3  # Max retrieval rounds
agentic_min_confidence: float = 0.5  # Min avg score to stop early
```

---

## Section 3: Production Hardening — Concrete Files to Create

### 3.1 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
      - id: detect-private-key
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [types-PyYAML]
```

### 3.2 `pyproject.toml` additions

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 120
timeout_method = "thread"
markers = [
    "gpu: marks tests that require CUDA GPU (deselect with '-m \"not gpu\"')",
    "slow: marks slow tests (deselect with '-m \"not slow\"')",
    "integration: marks integration tests (not unit tests)",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
addopts = "-v --strict-markers"
```

### 3.3 Benchmark Infrastructure

Create `tests/benchmarks/` with:
- `conftest.py` — GPUProfiler context manager, `pytest-benchmark` integration
- `test_pipeline_throughput.py` — benchmark each pipeline stage
- `test_rag_latency.py` — benchmark retrieval + re-ranking latency
- `test_chat_latency.py` — benchmark Q&A response time

---

## Section 4: Already-Implemented But Untested Features

### 4.1 Adaptive Frame Sampling ✅ (Implemented, lines 527-564)

Already in `pipeline.py` — `_adaptive_frame_samples()`. Uses cosine-based density:
- 3x frame density in first/last 10% of each scene
- Base rate: 1 frame per 2 seconds (configurable via `adaptive_frame_sampling_sensitivity`)
- Dense rate: 1 frame per 0.67 seconds near boundaries

**Needs:** Tests + enable by default recommendation.

### 4.2 CLIP-Similarity Frame Dedup ✅ (Implemented, lines 566-646)

Already in `pipeline.py` — `_dedup_frames_clip()`. Uses OpenCLIP embeddings:
- Compares consecutive frame pairs
- Removes later frame if cosine similarity ≥ threshold (default 0.92)
- Falls back gracefully if OpenCLIP unavailable

**Needs:** Tests + enable by default recommendation.

### 4.3 Timeline Hover Preview ✅ (Implemented in UI)

Gradio UI already has sprite-sheet timeline hover via CSS with `--sprite-sheet` property.

---

## Section 5: Recommended Implementation Order for v0.15.0

| Priority | Feature | Effort | Files Changed | Why This Order |
|----------|---------|--------|---------------|----------------|
| **P0** | SmolVLM2 integration | ~3h | `video_mllm.py`, `config.py`, `__init__.py`, `requirements.txt`, `tests/` | Unlocks video MLLM at 500M (1GB VRAM!) |
| **P0** | Adaptive frame sampling tests | ~0.5h | `tests/test_basic.py` | Already implemented, just needs test + enable |
| **P0** | CLIP dedup tests | ~0.5h | `tests/test_basic.py` | Already implemented, just needs test + enable |
| **P1** | Agentic RAG | ~3h | `rag.py`, `config.py`, `query_router.py` | Wraps existing router in iterative loop |
| **P1** | Pre-commit hooks | ~0.5h | `.pre-commit-config.yaml`, `pyproject.toml` | Foundation for code quality |
| **P2** | CI/CD (GitHub Actions) | ~1h | `.github/workflows/ci.yml` | Automated testing |
| **P2** | Benchmark infrastructure | ~2h | `tests/benchmarks/` | Performance regression tracking |

---

## Key Sources

- SmolVLM2 Blog: https://huggingface.co/blog/smolvlm2 (Feb 20, 2025)
- SmolVLM2 2.2B Model: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- SmolVLM2 500M Video: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- Qwen2.5-VL 3B: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct (video-capable, ~6B VRAM, alternative to VideoChat-Flash)
- BGE-VL-base: https://huggingface.co/BAAI/BGE-VL-base (MIT, 150M params, updated 2026-04-10)
- BGE-en-ICL: https://huggingface.co/BAAI/bge-en-icl (in-context learning embedding, good for zero-shot retrieval)
- Gradio 6.19 ChatInterface: https://www.gradio.app/docs/gradio/chatinterface
- A-RAG paper (2025): https://arxiv.org/abs/2602.03442
- Production hardening audit: docs/PRODUCTION-HARDENING-AUDIT.md
- Testing infrastructure research: docs/RESEARCH-video-evaluation-testing.md
