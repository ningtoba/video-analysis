# Video Analysis Platform — Research Findings (Iteration 1)

## Overview

Research conducted June 26, 2026. Coverage: video frame extraction, video understanding AI,
RAG architectures for video context, web UI frameworks, and production deployment for
self-hosted video analysis pipelines.

---

## 1. Scene Detection & Frame Extraction

### Status Quo (current codebase)

- **PySceneDetect 0.7** (released May 3, 2026) — the latest stable release already in use.
  Supports AdaptiveDetector, ContentDetector, HistogramDetector, HashDetector.
- FFmpeg `select='gt(scene,threshold)'` fallback.
- Frame extraction at 1 frame per 2 seconds, plus midpoint per scene.

### Findings

**PySceneDetect 0.7 is still the best OSS option.** It reached v0.7 in May 2026 with
production maturity. No competitive OSS alternative has emerged for scene-cut detection
since 2024.

**Content-aware adaptive sampling** is the key improvement area: instead of uniform
1-fps sampling, we could use:

1. **Scene-change-weighted sampling** — sample more densely near scene boundaries
   (where content changes most), less in the middle of long static shots.
2. **Motion-based adaptive sampling** — use FFmpeg's scene score or optical flow to
   dynamically adjust frame rate: high motion = higher fps, static = lower fps.
3. **CLIP-similarity-based keyframe selection** — after extraction, deduplicate frames
   that are semantically near-duplicates (cosine sim > 0.95 in CLIP space).

**No real AI-powered scene detection tools** have replaced PySceneDetect at the OSS
level. Commercial services (Twelve Labs, Google Video AI) use proprietary ML, but no
comparable open model has been released for shot boundary detection specifically.

### Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| High | Implement motion-based adaptive frame sampling | Reduces redundant frames by 30-50% |
| Medium | Add CLIP-similarity dedup after frame extraction | Better RAG quality, fewer chunks |
| Low | Upgrade PySceneDetect usage to v0.7 API | Minor stability improvements |

---

## 2. Video Understanding AI Models

### Status Quo (current codebase)

- **OpenCLIP** ViT-B-32 / ViT-L-14 — zero-shot scene classification (25 predefined labels)
- **YOLO** (`yolo26x.pt`) — object detection on key frames
- **faster-whisper** (large-v3) — transcription
- **PyAnnote 3.1** — speaker diarization
- **PaddleOCR** — on-screen text extraction

### Findings

**Video foundation models have advanced dramatically (2024-2026):**

| Model | Released | VRAM | Notes |
|-------|----------|------|-------|
| **InternVideo2.5** (OpenGVLab) | 2025 | ~4-6 GB (encoder only) | State-of-the-art video encoder, 1B params. Apache 2.0 |
| **VideoChat-Flash** (OpenGVLab) | 2025 (ICLR 2026) | ~16 GB (7B model) | Hierarchical compression for long video. 7B model with 2B small variant available |
| **LLaVA-Video 7B** | 2025 | ~16 GB | Strong on Video-MME, LongVideoBench |
| **VideoMAE V2** (MCG-NJU) | 2023 (CVPR 2024) | ~4 GB | Archived repo, superseded by InternVideo |
| **TimeSformer** (Meta) | 2021 (ICML) | ~4 GB | **Archived** Jan 2025 — deprecated |

**Key finding: InternVideo2 is the successor to VideoMAE/TimeSformer** for action
recognition and video understanding. The pipeline should target InternVideo2.5 for
action recognition instead of the originally planned VideoMAE or TimeSformer.

**OpenCLIP remains the best zero-shot frame-level descriptor** that fits in 12 GB VRAM.
Video foundation models (InternVideo2, LLaVA-Video) are too heavy for real-time batch
processing alongside all other models on a 12 GB card.

**Nomic Embed Multimodal** (released 2025) is a new multimodal embedding model that
can embed images and text into the same vector space — this is directly relevant for
video frame search.

### Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| **High** | **Add InternVideo2 as optional action recognition model** | Unblocks roadmap item |
| Medium | Add frame dedup via CLIP similarity | Cleaner RAG chunks |
| Low | Evaluate Nomic Embed Multimodal for image+text search | Multimodal retrieval |
| Low | Consider LLaVA-Video for advanced video Q&A | Better than prompt-only approach |

---

## 3. RAG Architectures for Video Context

### Status Quo (current codebase)

- **ChromaDB** persistent local vector store
- **Nomic Embed Text v1.5** (768-dim) — dense retrieval
- **Cross-encoder MS MARCO MiniLM** — first-stage re-ranking
- **ColBERTv2 via RAGatouille** — optional late-interaction re-ranking
- **Temporal context expansion** (±1 neighbor scene)
- **Hybrid** transcript + scene descriptions + frame objects + OCR

### Findings

**The current RAG stack is already state-of-the-art for a self-hosted setup.**
Key improvements discovered:

1. **Nomic Embed v2** (not yet released at time of writing) — keep monitoring.
   Current v1.5 with Matryoshka dimensions is excellent.

2. **BGE-VL (FlagEmbedding)** — Released March 2025. **Multimodal embedding** that
   supports text-to-image, image-to-text, and combined search. MIT license.
   This is the most relevant breakthrough: it can embed video frames directly as
   visual vectors, enabling image-level similarity search in ChromaDB (not just
   text-based search). Requires ~2 GB VRAM.

3. **ColBERTv3** — RAGatouille continues to be maintained, ColBERTv3 has improved
   compression for the late interaction mechanism. Already optionally integrated.

4. **MegaPairs** dataset (released with BGE-VL) — massive multimodal pairs for
   training custom embeddings. Not needed for inference but relevant if fine-tuning.

5. **ChromaDB 28.6k stars**, actively maintained. Major improvements in the rust
   rewrite backend. No need to switch vector DBs.

6. **Temporal-aware retrieval** could be improved with:
   - **Hierarchical indexing**: full video → scenes → frames as a tree, query
     at each level
   - **Timestamp-weighted retrieval**: score = similarity × time_decay_weight
   - **Cross-video temporal patterns**: queries about "before/after" events

### Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| **High** | **Integrate BGE-VL for multimodal (image+text) frame search** | Major retrieval quality boost |
| Medium | Implement hierarchical chunk indexing (video→scene→frame) | Better structured retrieval |
| Medium | Add timestamp weighting to retrieval scoring | Better temporal precision |
| Low | Upgrade to ColBERTv3 when RAGatouille supports it | Minor precision improvement |

---

## 4. Web UI Frameworks

### Status Quo (current codebase)

- **Gradio 6 Blocks** — v6.19.0 latest (as of June 2026)
- FastAPI for health/API endpoints
- Mounted via `gr.mount_gradio_app()`
- Custom CSS dark theme, timeline JS, Shadow DOM fixes
- Gradio shared (public URL) disabled by default

### Findings

**Gradio 6 is the right choice.** Alternatives evaluated:

| Framework | Stars | Video Support | Pros | Cons |
|-----------|-------|---------------|------|------|
| **Gradio 6** | 40k+ | Native Video component, chatbot, streaming | Best video+chat combo, gr.mount_gradio_app() for FastAPI | Shadow DOM quirks, JS injection limited |
| Streamlit | 40k+ | Video via st.video | Simpler | No native chatbot, poor for interactive Q&A |
| NiceGUI | 10k+ | Video via ui.video | Python-native UI | Smaller community |
| Dash (Plotly) | 22k+ | Video via html.Video | Enterprise-grade | Heavy for self-hosted |

**Gradio remains the best fit** because:
- Native `gr.Video()` component with timeline, seek, controls
- `gr.Chatbot()` with bubble layout, perfect for Q&A
- `gr.mount_gradio_app()` allows FastAPI alongside
- Extensive component ecosystem for file uploads, progress, tabs
- Active development (6.19.0 = very fresh)

**Gradio 6 improvements since v0.6.0 of the project:**
- Workflow subgraphs (6.19.0) — multi-step pipeline visualization
- Svelte 5 migration of core components (6.18.0)
- Better ARIA accessibility
- OAuth session handling

### Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| High | **Implement Gradio auth via env vars** (roadmap item) | Security |
| Medium | Upgrade to Gradio 6.19.x for latest fixes | Stability |
| Medium | Leverage gr.Workflow for visual pipeline display | Better UX |

---

## 5. Production Deployment

### Status Quo (current codebase)

- Multi-stage Docker build (python:3.11-slim → nvidia/cuda:12.8-runtime)
- docker-compose with GPU passthrough, health checks, persistent volumes
- FastAPI health endpoint at /health
- Sequential model loading for 12 GB VRAM management
- Non-root user in container

### Findings

**GPU Memory Management:**

1. **Sequential loading is the right approach** for 12 GB VRAM on RTX 4070.
   Total VRAM needed if all models loaded simultaneously:
   - faster-whisper large-v3: ~4 GB
   - OpenCLIP ViT-L-14: ~2 GB
   - ChromaDB (sentence-transformers): ~1 GB
   - YOLO: ~1 GB
   - Total concurrent: ~8 GB (fits, but barely during transcription)

2. **`torch.cuda.empty_cache()`** after each model unload is critical — we should
   verify this is being called between pipeline stages.

3. **Background queue workers** via Redis/Celery would be overkill for a single-GPU
   setup. The existing sequential batch queue is correct.

**Containerization:**
- Dockerfile uses CUDA 12.8 runtime — good, but torch wheels may only compile
  against cu128 until torch 3.0 ships. Keep this aligned.
- HEALTHCHECK uses `/health` endpoint with 120s start period — correct.
- Mem limit 16g with 4g swap — appropriate for 12 GB card.

**Missing pieces for production readiness:**
- **Gradio auth** (env vars)
- **Rate limiting** on FastAPI endpoints
- **Graceful shutdown** catch SIGTERM for ongoing processing

### Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| High | Implement Gradio auth via env vars | Security |
| Medium | Add `torch.cuda.empty_cache()` calls between pipeline stages | VRAM stability |
| Low | Add graceful SIGTERM handling | Clean shutdowns |
| Low | Add rate limiting on API endpoints | Production hardening |

---

## Summary of High-Impact Items for Next Iteration

| # | Area | Action | Effort | Impact |
|---|------|--------|--------|--------|
| 1 | UI/Security | **Gradio auth via env vars** | Small | **High** |
| 2 | RAG | **BGE-VL multimodal embeddings for frame search** | Medium | **High** |
| 3 | Video AI | **InternVideo2 action recognition** | Medium | **High** |
| 4 | Pipeline | **Adaptive frame sampling (motion-based)** | Medium | Medium |
| 5 | Pipeline | **CLIP similarity frame dedup** | Small | Medium |
| 6 | Pipeline | **GPU memory: explicit empty_cache() calls** | Small | Medium |
| 7 | RAG | Hierarchical chunk indexing | Medium | Medium |

---

## Key Sources

- PySceneDetect v0.7: https://github.com/Breakthrough/PySceneDetect (released May 3, 2026)
- InternVideo2.5: https://github.com/OpenGVLab/InternVideo2 (Apache 2.0, 2.3k stars)
- VideoChat-Flash: https://github.com/OpenGVLab/VideoChat-Flash (ICLR 2026, 526 stars)
- LLaVA-Video: https://github.com/LLaVA-VL/LLaVA-NeXT (4.7k stars)
- RAGatouille: https://github.com/AnswerDotAI/RAGatouille (3.9k stars, Apache 2.0)
- ChromaDB: https://github.com/chroma-core/chroma (28.6k stars, actively maintained)
- BGE-VL (FlagEmbedding): https://github.com/FlagOpen/FlagEmbedding (11.9k stars)
- VideoMAE V2: https://github.com/MCG-NJU/VideoMAE (1.8k stars)
- Nomic Embed Multimodal: https://huggingface.co/nomic-ai (Apache 2.0)
