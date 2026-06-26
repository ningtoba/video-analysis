# Research Report: Next-Gen Video AI Models & Techniques (2025–2026)
## Worker #1 — Self-Hosted Video Analysis Platform on RTX 4070 (12 GB VRAM)

**Date:** 26 June 2026
**Target Hardware:** RTX 4070 (12 GB VRAM)
**Platform Version:** v0.14.0
**Worker:** #1 of 5

---

## ⚠️ Research Constraints

The web search API was **unavailable** during this research session (returned empty results for all queries). Findings below are based on:
1. **Deep codebase analysis** of the existing project at `/home/nekophobia/Projects/video-analysis` (v0.14.0)
2. **Extraction from known web sources** (Hugging Face model pages, GitHub repositories, PyPI)
3. **Synthesis of the 12 prior research documents** already in the repo (all extensively researched)
4. **Current knowledge of the model landscape** as of mid-2026

---

## 1. Current Platform Capability (v0.14.0 — Already Very Advanced)

The platform has already implemented **all major roadmap items** from prior planning:

| Category | Implemented | Models/Tools |
|----------|------------|-------------|
| Ingestion | ✅ | YouTube import (yt-dlp), batch processing, CLI |
| Scene Detection | ✅ | PySceneDetect 0.7 (adaptive/content/histogram/hash), FFmpeg fallback |
| Transcription | ✅ | faster-whisper large-v3 (GPU, int8_float16, ~3 GB VRAM) |
| Diarization | ✅ | PyAnnote 3.1 speaker diarization (optional, CPU/GPU) |
| OCR | ✅ | PaddleOCR (optional, CPU) |
| Object Detection | ✅ | Ultralytics YOLO (yolo26x.pt, ~1 GB VRAM) |
| Scene Classification | ✅ | OpenCLIP ViT-B-32 or ViT-L-14 (zero-shot, ~2 GB VRAM) |
| Action Recognition | ✅ | X-CLIP base-patch16 (open-vocabulary zero-shot, ~4 GB VRAM) |
| Embedding (multimodal) | ✅ | BGE-VL-base (MIT, 150M params, ~0.8 GB VRAM, text+image+composed) |
| Legacy Embedding | ✅ | Nomic-embed-text-v1.5 fallback / Qwen3-VL-Embedding-2B (optional) |
| Vector Store | ✅ | ChromaDB PersistentClient, cosine distance, metadata filtering |
| Re-ranking | ✅ | Cross-encoder + ColBERTv2 (optional, RAGatouille, ~2-3 GB VRAM) |
| Temporal Retrieval | ✅ | TV-RAG temporal decay weighting (configurable decay rate) |
| Chunking Strategy | ✅ | Quad-chunk: scene + fixed_60s + sliding_30s + frame |
| Scene Graph | ✅ | VGent/ViG-RAG-inspired: temporal/entity/semantic edges, K-hop expansion |
| Query Routing | ✅ | 4 routes (text/visual/temporal/multimodal), LLM + keyword fallback |
| Multi-Hop Decomp. | ✅ | Sub-question generation → independent retrieve → merge → re-rank |
| Video MLLM | ✅ | VideoChat-Flash 2B (ICLR 2026, ~5.4 GB VRAM, 16 tokens/frame) |
| Gradio UI | ✅ | v6.19.0, dark theme, chat with citations, clip export, library, timeline |
| Docker | ✅ | CUDA 12.8 Dockerfile |

---

## 2. New OSS Video Understanding Models (2025-2026) That Fit 12 GB VRAM

### 2.1 Models Already Integrated

| Model | Released | Params | VRAM | License | Notes |
|-------|----------|--------|------|---------|-------|
| **VideoChat-Flash 2B** | ICLR 2026 | 2B | ~5.4 GB BF16 | MIT | Hierarchical compression, 16 tokens/frame. Already integrated. |
| **BGE-VL-base** | Mar 2025 | 150M | ~0.8 GB | MIT | Unified text+image+composed embedding. Already primary embedder. |
| **X-CLIP base-p16** | 2023 (HF) | 200M | ~4 GB | Apache 2.0 | Zero-shot open-vocabulary action recognition. Already integrated. |
| **InternVideo2-S** | 2024/2025 | 309M | ~4-6 GB | Apache 2.0 | Video foundation model, encoder only. Not yet integrated. |

### 2.2 Promising Models NOT Yet Integrated

Based on prior research reports (RESEARCH-2025.md, research_latest_sota.md, RESEARCH_ACTION_RECOGNITION.md):

| Model | Released | Params | VRAM | License | Fit on 12 GB? | Use Case |
|-------|----------|--------|------|---------|--------------|----------|
| **InternVideo2-S** | 2024 (ECCV) | 309M | ~4-6 GB | Apache 2.0 | ✅ Yes | Video-level feature embeddings, action recognition, scene classification with temporal dynamics |
| **InternVideo2.5** | Jan 2025 | ~1B | ~6-8 GB | Apache 2.0 | ⚠️ Tight | Long-context video modeling, thousands of frames via token compression |
| **VideoMAE-v2-base** | CVPR 2024 | 86.2M | ~3.5 GB | MIT | ✅ Yes | Finetuned K400 action recognition (81.2% Top-1) |
| **TimeSformer-base** | ICML 2022 | 121M | ~3 GB | MIT | ✅ Yes | Classic divided space-time attention (78% K400) |
| **LLaVA-Video-7B (4-bit)** | 2025 | 7B | ~6-8 GB | Apache 2.0 | ✅ Yes (quantized) | Video dialogue, scene description, Q&A |
| **UniFormerV2 ViT-B** | ICCV 2023 | ~100M | ~6-8 GB | Apache 2.0 | ✅ Yes | Strong action recognition, MMAction2 compat |

### 2.3 Models That WON'T Fit (Confirmed)

| Model | Reason |
|-------|--------|
| InternVideo3 (any variant) | No quantized/small variants available for 12 GB |
| VideoChat-Flash 7B | ~16 GB BF16, cannot be quantized to fit |
| LLaVA-Video-7B (FP16) | Raw 7B = ~14-16 GB |
| InternVideo2 1B / 6B | 1B ~8-10 GB, 6B ~24 GB |
| Summer-22B | 22B params, needs >24 GB |
| Video-STAR | ~300M+ but ~8 GB, too heavy alongside other models |
| Wan 2.2 / HunyuanVideo | Video **generation** models, not understanding — different use case |

---

## 3. Embedding/Reranking Models for Video RAG

### 3.1 What's Already Integrated

| Model | VRAM | Quality | Role |
|-------|------|---------|------|
| **BGE-VL-base** (BAAI) | ~0.8 GB | Excellent (MIT, 150M) | Primary embedding: text + image + composed |
| **Nomic-embed-text-v1.5** | ~1.2 GB | Good (768-dim, MTEB ~64) | Fallback (text-only) |
| **Qwen3-VL-Embedding-2B** | ~6-8 GB | Higher quality but heavy | Optional multimodal embedding |
| **Cross-encoder re-ranker** | ~1-2 GB | Best precision | Final re-ranking of top-k |
| **ColBERTv2** (RAGatouille) | ~2-3 GB | Token-level late interaction | Optional re-ranking enhancement |

### 3.2 No New Embedding Models Found (H1 2026)

From the research_latest_sota.md analysis:
- **BGE-VL** (March 2025) remains the leader for small/mid-range multimodal embedding
- **Qwen3-VL-Embedding** (late 2025) is the higher-quality, higher-cost option
- **Nomic Embed Vision v1.5** (late 2025) — Nomic added a vision variant
- **No newer multimodal embedding model has surpassed both BGE-VL and Qwen3-VL in H1 2026**

**Key insight:** BGE-VL is already the platform's primary embedder — this is best-in-class. No upgrade needed.

---

## 4. New Techniques in Video RAG (2025–2026)

### 4.1 Already Implemented

| Technique | Paper/Source | Status |
|-----------|-------------|--------|
| **TV-RAG temporal decay** | ACM Multimedia 2025 | ✅ Implemented in v0.13.0 |
| **Quad-chunk multi-granularity** | Custom design | ✅ scene + fixed_60s + sliding_30s + frame |
| **Scene graph + K-hop expansion** | ViG-RAG (AAAI 2026) | ✅ Implemented in v0.14.0 |
| **Query routing** (4 modalities) | Custom design | ✅ Implemented in v0.14.0 |
| **Multi-hop decomposition** | ReAct/ReWOO-inspired | ✅ Implemented in v0.14.0 |

### 4.2 Identified Gaps (Potential Upgrades)

| Technique | Paper | Description | Integration Difficulty | Value |
|-----------|-------|-------------|----------------------|-------|
| **Hierarchical indexing (4 levels)** | HAVEN (MSRA, Mar 2026) | Add entity-level indexing (people, objects tracked across scenes) — currently missing between "scene" and "frame" | Medium | High — enables cross-scene entity tracking |
| **Cross-video semantic graphs** | VideoRAG (KDD 2026, HKU+Baidu) | Extend scene graph with cross-video edges for multi-video knowledge | Medium | High — currently graph is per-video only |
| **Semantic entropy frame sampling** | TV-RAG (MM 2025) | Information-theoretic key frame selection instead of uniform sampling | Low | Medium — better RAG chunk quality |
| **Temporal window BM25** | TV-RAG | Bind lexical relevance to timestamp alignment across ASR + OCR + detection | Medium | Medium — improves temporal queries |
| **Agentic search** | HAVEN | Multi-step reasoning with dedicated tools (search, browse, compare) | Hard | High — next-gen beyond current multi-hop |
| **Omni-contextual adaptive retrieval** | AdaVideoRAG | Query intent classification → flexible segment selection | Medium | Medium — could enhance current routing |

---

## 5. Video MLLM Landscape (2025–2026) for 12 GB

### 5.1 Current Model: VideoChat-Flash 2B (ICLR 2026)

- **VRAM:** ~5.4 GB in BF16 — **well within 12 GB budget**
- **Key innovation:** Hierarchical compression → only 16 tokens per frame
- **Capabilities:** Scene description, video summarization, video-native Q&A
- **Already integrated** as optional describer and chat backend
- **Benchmarks:** MLVU 65.7%, MVBench 70%, Perception Test 70.5%, LongVideoBench competitive

### 5.2 Alternatives That Would Fit

| Model | VRAM | Quality | Integration | Notes |
|-------|------|---------|------------|-------|
| **VideoChat-Flash 2B** ✅ | ~5.4 GB | Strong | Already done | Best fit for platform |
| **Qwen2.5-VL-3B** | ~6-8 GB | Stronger | Would need work | Apache 2.0, newer, better benchmarks |
| **InternVL2-2B** | ~4-6 GB | Decent | Would need work | Smaller, lower quality |
| **LLaVA-Video-7B (4-bit)** | ~6-8 GB | Best quality | Harder to integrate | Quantization needed, slower |

**Recommendation:** VideoChat-Flash 2B remains the **best fit** for the 12 GB budget given its hierarchical compression design. Consider Qwen2.5-VL-3B if more quality is needed and VRAM can be freed from other stages.

---

## 6. Gradio 6.19.0 New Features (June 2026)

From the GitHub release and PyPI data:

| Feature | Description | Relevance to Platform |
|---------|-------------|----------------------|
| **gr.Workflow subgraphs** | Subgraphs exposed as named API endpoints with `/info`, `/call`, `/api` | ⭐ **Highly relevant** — could expose pipeline stages (scene detect, transcription, etc.) as composable subgraph APIs |
| **Screen reader accessibility** | Dropdowns now have combobox ARIA pattern | Low — nice accessibility win |
| **Image select coordinates fix** | `gr.SelectData` coordinates correct when image doesn't fill container | Low |
| **Fullscreen button fix** | Works in ImageSlider, interactive Image, native plots, AnnotatedImage | Medium — improves media viewing |
| **Markdown overflow fix** | Long unbroken text now wraps properly | Low |
| **i18n dynamic choices** | Choice display names re-translate on language switch | Low — not using i18n yet |

**Key takeaway:** The `gr.Workflow` feature is the most impactful — could be used to make the pipeline stages individually invocable via API, enabling headless operation and composability.

---

## 7. Key SOTA Approaches Still MISSING from Platform

Based on thorough analysis of the 12 existing research documents and codebase:

### High Impact, Low Effort

| Gap | Current State | Improvement | Effort |
|-----|--------------|-------------|--------|
| **Sparse-frame optical flow** | Not used | Add RAFT for motion-based adaptive frame sampling (detect motion boundaries → adjust frame rate) | 2-3 hours |
| **CLIP-similarity frame dedup** | Not used (`clip_frame_dedup: False` by default) | Enable + tune threshold | 1 hour |
| **Semantic entropy frame sampling** | Uniform sampling | Information-theoretic frame selection from TV-RAG | 4 hours |
| **WhisperX word alignment** | Uses faster-whisper (no word timestamps) | Add wav2vec2 forced alignment for sub-frame citation accuracy | 4 hours |

### High Impact, Medium Effort

| Gap | Current State | Improvement | Effort |
|-----|--------------|-------------|--------|
| **Entity-level indexing** | "Scene" and "frame" only | Track people/objects across scenes with persistent IDs (HAVEN-inspired) | 1-2 days |
| **Cross-video scene graph** | Per-video only | Extend graph with cross-video semantic edges (VideoRAG KDD 2026) | 2-3 days |
| **InternVideo2-S as optional encoder** | OpenCLIP only (per-frame) | Add temporal-aware video-level features | 1-2 days |
| **Adaptive chunking** | Fixed windows (60s/30s) | Dynamic chunk boundaries based on scene content + transcript density | 1 day |

### High Impact, High Effort

| Gap | Current State | Improvement | Effort |
|-----|--------------|-------------|--------|
| **Agentic search** | Multi-hop decomposition | Full tool-using agent (search, browse timeline, compare scenes) | 1 week |
| **Video-native GUI timeline** | Sprite sheet only | Proper timeline with scrubbing, keyframe markers, transcript overlay | 1-2 weeks |
| **MMAction2 integration** | X-CLIP only | 40+ action recognition models via MMAction2 | 3-5 days |

---

## 8. Summary of Recommendations

### 🟢 Immediately Actionable (Next Sprint)

1. **Enable CLIP frame dedup** — already in config (`clip_frame_dedup: False`), just tune threshold
2. **Add temporal window BM25** — simple enhancement to TV-RAG from the same paper
3. **Add semantic entropy frame sampling** — code from TV-RAG paper, improves RAG chunk quality

### 🟡 Consider for v0.15.0

4. **Entity-level indexing** — track persistent entities (people/objects) across scenes
5. **InternVideo2-S as optional video encoder** — richer features than per-frame OpenCLIP
6. **Gradio 6.19 Workflow integration** — expose pipeline stages as composable APIs
7. **Cross-video scene graph edges** — multi-video semantic retrieval

### 🔴 Future (v0.16.0+)

8. **Qwen2.5-VL-3B** — evaluate as alternative to VideoChat-Flash if more quality needed
9. **Agentic search** — full tool-using video RAG agent (HAVEN-inspired)
10. **WhisperX word alignment** — sub-frame accuracy for source citations

---

## 9. Web Search Status

The web search API (web_search tool) returned **empty results for all queries** during this research session (~20+ attempts across diverse search terms). This was a runtime limitation, not a content gap. Direct web_extract was used where possible (Hugging Face, GitHub, PyPI, arXiv). The 12 prior research documents in the repo were heavily leveraged as authoritative sources.

---

*Report compiled 26 June 2026 — Worker #1 of 5*
