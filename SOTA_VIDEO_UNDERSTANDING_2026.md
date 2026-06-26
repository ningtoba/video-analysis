# State-of-the-Art Video Understanding AI — July-December 2026 (Updated)

> **Research conducted:** 2026-06-26  
> **Previous research:** SOTA_VIDEO_UNDERSTANDING_2026.md (June 2026, for v0.14→v0.15)  
> **This update:** Covers H2 2026 landscape for v0.16.0 roadmap items  
> **Hardware target:** NVIDIA RTX 4070 (12 GB VRAM)  
> **Project:** video-analysis v0.16.0  
> **Current models in project:** VideoChat-Flash 2B, SmolVLM2, X-CLIP, BGE-VL, Jina v4, Qwen3-Embedding, faster-whisper

---

## Table of Contents

1. [Video MLLMs Under 3B Parameters (H2 2026)](#1-video-mllms-under-3b-parameters)
2. [New Video/Multimodal Embedding Models](#2-new-embedding-models)
3. [Multi-Object Tracking: Beyond ByteTrack](#3-multi-object-tracking)
4. [CVPR/ECCV/NeurIPS 2026 Papers on Video RAG & Scene Graphs](#4-top-venue-papers-2026)
5. [Gradio 6.19+ Workflow & Component Features](#5-gradio-619-features)
6. [Synthesis & v0.16 Roadmap Impact](#6-synthesis)

---

## 1. Video MLLMs Under 3B Parameters

### 1.1 VideoThinker-R1-3B (CVPR 2026 Poster)

| Property | Detail |
|----------|--------|
| **Params** | ~3.0B (based on Qwen2.5-VL-3B-Instruct; actual ~4B with embedding) |
| **VRAM (BF16)** | ~6-7 GB ✅ Fits 12GB |
| **License** | MIT ✅ |
| **Release date** | April 2026 (model), May 2026 (updated) |
| **Paper** | arxiv:2605.01324 — "Beyond Perceptual Shortcuts: Causal-Inspired Debiasing Optimization for Generalizable Video Reasoning in Lightweight MLLMs" |
| **Authors** | Jingze Wu, Quan Zhang, Hongfei Suo, Zeqiang Cai, Hongbo Chen |
| **Venue** | **CVPR 2026 Poster** |

**Key innovation:** Causal-Inspired Debiasing Policy Optimization (CDPO). Two-stage training:
1. **Bias Aware Training** — trains a "bias model" that embodies shortcut behaviors
2. **CDPO** — fine-tunes the primary model with repulsive objective away from bias model

**Benchmarks:**
- No SFT required — uses only 1% of training data for RL
- **Surpasses VideoRFT-3B** by +3.2% average on standard benchmarks, +7% on Video-MME
- **Outperforms Video-UTR-7B** by +2.1% on MVBench, +3.8% on TempCompass

**v0.16 relevance:** Strong candidate to replace/complement VideoChat-Flash 2B for reasoning tasks. Fits 12GB easily. Fine-tune of Qwen2.5-VL-3B (Apache 2.0 base model).

### 1.2 Video-Scan (cudabenchmarktest/video-scan)

| Property | Detail |
|----------|--------|
| **Params** | ~2B |
| **VRAM** | ~5-6 GB ✅ |
| **License** | Not specified |
| **Release date** | May 20, 2026 |

Qwen2.5-VL-2B based video understanding model. Minimal downloads (18 total). Potential for lightweight scanning.

### 1.3 Summary: Video MLLMs for 12GB

| Model | Params | VRAM BF16 | Fits 12GB? | License | Best For |
|-------|--------|-----------|------------|---------|----------|
| **VideoThinker-R1-3B** | 3B | ~6-7 GB | **Yes** ✅ | MIT ✅ | Video reasoning (new SOTA for its size) |
| **VideoChat-Flash 2B** *(current)* | 2B | ~5.4 GB | Yes ✅ | MIT ✅ | Long video (99.1% NIAH@10K) |
| **SmolVLM2 2.2B** *(current)* | 2.2B | ~2-3 GB | Yes ✅ | Apache 2.0 ✅ | General video understanding |
| **Video-Scan** | 2B | ~5-6 GB | Yes ✅ | ? | Lightweight scanning |
| **VideoKR-7B/8B** | 7-9B | ~14-18 GB | ❌ | — | Too large for 12GB |
| **MOSS-Video-11B** | 11B | ~22 GB | ❌ | Apache 2.0 | Too large native; OK if 4-bit |

---

## 2. New Embedding Models

### 2.1 Current State
Already using: **BGE-VL** (BAAI/bge-vl-base), **Jina v4**, **Qwen3-Embedding**

### 2.2 No Significant New Video Embedding Models Found

**Key finding:** No new dedicated video/multimodal embedding models beyond BGE-VL, Jina v4, and Qwen3-Embedding have appeared since mid-2025. The embedding model landscape has consolidated:

| Model | Release | Pipeline | Best For |
|-------|---------|----------|----------|
| **BGE-VL** (BAAI) | Mid-2025 | Image+text → embedding | Multimodal search (already integrated) |
| **Jina v4** (Jina AI) | 2025 | Text → embedding | Text retrieval (already integrated) |
| **Qwen3-Embedding** (Alibaba) | Early 2026 | Image+text → embedding | Composed retrieval (already integrated) |

**Research directions (not production models):**
- **MJEPA** (Meta, arXiv:2606.25225, June 2026): Joint-embedding predictive architecture for audio-visual learning. Self-supervised video representation. Not a retrieval embedding.
- **V-JEPA** (Meta, 2025/2026): Video Joint-Embedding Predictive Architecture. Could be used as frozen feature extractor.

**v0.16 recommendation:** Stick with current embedding stack. Monitor for domain-specific fine-tuned variants.

---

## 3. Multi-Object Tracking: Beyond ByteTrack

### 3.1 Polycepta (June 2026) — Most Promising NEW Approach

| Property | Detail |
|----------|--------|
| **Paper** | arXiv:2606.23604 — "Polycepta: Object-Centric Appearance Estimation for Multi-Object Tracking" |
| **Authors** | Mohamed Nagy, Naoufel Werghi, Jorge Dias, Majid Khonji |
| **Date** | June 22, 2026 |
| **Venue** | Not yet announced (very recent) |

**Key innovation:** Reformulates appearance modeling as a **recursive estimation problem** rather than frame-wise matching. Constructs and continuously updates an **independent appearance state for each tracked object**.

**Key properties:**
- Appearance quality **improves over time** during inference
- **Unseen class generalization**
- **90.57 Hz** inference speed
- Integrates with existing tracking-by-detection pipelines
- **SOTA on KITTI:** 92.27% MOTA

**v0.16 relevance:** Perfect for entity tracking. Recursive appearance estimation maintains entity identities across video clips/scenes.

### 3.2 Other MOT Frameworks

| Tracker | Year | Key Feature | License | Suitable for 12GB? |
|---------|------|-------------|---------|-------------------|
| **ByteTrack** | 2022 | Association by detection score | MIT ✅ | ✅ (already integrated) |
| **BoT-SORT** | 2022 | ByteTrack + Kalman + camera motion | MIT ✅ | ✅ |
| **OC-SORT** | 2023 | Observation-centric association | MIT ✅ | ✅ |
| **Deep OC-SORT** | 2024 | OC-SORT + deep appearance | MIT ✅ | ✅ |
| **MOTRv3** | 2024 | End-to-end transformer tracking | Apache 2.0 ✅ | ⚠️ (heavy) |
| **DEVA** | 2024 | Decoupled long-range video tracking | MIT ✅ | ✅ (lighter than MOTR) |
| **Polycepta** | June 2026 | Recursive appearance estimation | ? | ✅ (90.57 Hz) |

### 3.3 Recommendation for v0.16 Entity Tracking

**Primary choice:** Implement **BoT-SORT** or **OC-SORT** as direct ByteTrack replacement — well-established, MIT-licensed, YOLO-compatible.

**Research track:** Monitor **Polycepta** — recursive appearance estimation is architecturally aligned with entity tracking.

**Cross-video pipeline:**
1. YOLO → ByteTrack/BoT-SORT (within-video tracking)
2. BGE-VL embeddings from tracked crops (cross-video identity matching)
3. Optional: lightweight ReID (osnet_x0_25)

---

## 4. Top Venue Papers 2026 — Video RAG, Scene Graphs, Understanding

### 4.1 CVPR 2026 Papers

#### DSFlash: Panoptic Scene Graph Generation
- **arXiv:** 2603.10538 — **CVPR 2026**
- **Key:** 56 FPS panoptic scene graph generation from video on RTX 3090. Comprehensive (not just salient) relationship extraction. Trains in <24h on GTX 1080.
- **v0.16 relevance:** Directly applicable to scene graph retrieval system. Much faster than existing SGG.

#### VideoThinker-R1-3B (covered in §1.1)
- **CVPR 2026 Poster**
- Causal debiasing for lightweight video MLLMs

### 4.2 ECCV 2026 Papers

#### PhysRAG: Physics-Aware Video Generation via RAG
- **arXiv:** 2606.26916 — **ECCV 2026**
- Uses RAG to inject physics knowledge into video diffusion models.
- **v0.16 relevance:** Not directly for analysis, but RAG-for-video-knowledge pattern is transferable.

### 4.3 Other Notable Video Papers (2026)

#### Decoupling Semantics and Logic for Video RAG
- **arXiv:** 2606.07924 — **ACL 2026 MAGMAR Workshop (Oral, Retrieval No.1)**
- Training-free, two-stage cascaded Video RAG pipeline:
  - Stage 1: Dense retrieval on visual summaries only
  - Stage 2: LLM agent for cognitive reranking with persona/hallucination constraints
- **v0.16 relevance:** **HIGH** — directly aligns with the agentic RAG (3-round iterative) approach

#### Robust-TO: Confidence-Aware Tool Orchestration
- **arXiv:** 2606.26904
- Agentic video understanding with per-frame trustworthiness
- 56.4% accuracy on clean, 54.3% on corrupted — beats Gemini-2.5-Pro (46.2%)
- Uses GRPO reward for correctness + evidence reliability + efficiency
- **v0.16 relevance:** Confidence-weighted evidence framework applicable to agentic RAG pipeline

#### HarmVideoBench (June 2026)
- Multi-layered harmful video understanding benchmark (19 LVLMs)
- BCR method: reasoning boundary prediction + dynamic context retrieval (84.4%)

#### ProtoKV: Streaming Video Understanding
- **arXiv:** 2606.26762
- Delayed query handling with summary-state memory
- Relevant if platform adds live/streaming analysis

#### EVIS: Event-Aware Video Segmentation (IEEE TIP)
- Decomposes video into events via learnable Event Query
- Object-Pixel-Hybrid Learning for long-video tracking
- **v0.16 relevance:** Event decomposition could enhance scene detection output

### 4.4 Video RAG Papers (2026) Summary

| Paper | Venue | Application | Relevance |
|-------|-------|-------------|-----------|
| **PhysRAG** (2606.26916) | ECCV 2026 | Video gen + physical RAG | Pattern: knowledge injection via RAG |
| **Decoupling Semantics+Logic** (2606.07924) | ACL 2026 MAGMAR | Training-free cascaded Video RAG | **High** — two-stage: semantic retrieval + LLM reranking |
| **LongLive-RAG** (2606.02553) | — | Long video gen + RAG history | RAG-as-memory pattern |

---

## 5. Gradio 6.19+ Workflow & Component Features

### 5.1 Latest Release: Gradio 6.19.0 (June 17, 2026)

#### ⭐ Workflow Subgraphs as API Endpoints (#13524)
**The single most impactful new feature for v0.16.**

Each subgraph (output node) in `gr.Workflow` is now exposed as a **named endpoint** reusing existing `/info`, `/call/{api_name}`, and `/api/{api_name}` machinery:
- `gradio_client` can call workflow subgraphs programmatically
- API recorder / MCP servers work against visually-defined workflows
- Endpoints re-derive live when the graph is saved (no restart)
- "View API" panel in the canvas shows available endpoints

**Implementation** (PR #13524 by @abidlabs, merged June 17, 2026):
- `workflow_api.py`: Schema-v2 graph model, topological sort, subgraph extraction
- `WorkflowExecutor`: Python executor running subject's upstream DAG
- One named API endpoint per subject via hidden real components
- `WorkflowEndpointManager`: Tears down and rebuilds endpoints on every save

**v0.16 impact:** Each video analysis pipeline step (scene detection, transcription, YOLO, etc.) can be a Workflow subgraph, exposed as individual API endpoints. Enables composable analysis pipelines via visual canvas.

#### Other 6.18.0 → 6.19.0 Features

| Feature | Version | Description |
|---------|---------|-------------|
| Workflow drag selection | 6.18.0 | Select multiple nodes by dragging |
| Workflow: preserve input state | 6.18.0 | Dropdown/radio/checkbox persist across saves |
| Workflow: local HF token | 6.18.0 | Write-token auth for spaces |
| Svelte 5: Plot, Chatbot, Tabs | 6.18.0 | Improved reactivity |
| OAuth session expiry | 6.18.0 | Expired sessions logged out |
| Script tag warning in gr.HTML | 6.18.0 | Security improvement |
| JS functions with fn=None | 6.18.0 | `js` works even without Python fn |
| Dropdown accessibility | 6.19.0 | Combobox ARIA pattern |
| Fullscreen button fix | 6.19.0 | ImageSlider, Image, native plots |
| i18n dynamic translation | 6.19.0 | Choices retranslated on language switch |

### 5.2 Recommended Implementation for v0.16

1. **Wrap each pipeline step** as a `gr.Workflow` node (input_node, output_node)
2. **Define subgraphs** for common pipelines ("full_analysis", "quick_search", "entity_track")
3. **Expose subgraphs** as API endpoints (automatic in 6.19.0)
4. **MCP integration** — enable `mcp_server=True` for AI agent access

---

## 6. Synthesis & v0.16 Roadmap Impact

### 6.1 What's New Since Previous Research

| Category | Finding | Impact on v0.16 |
|----------|---------|-----------------|
| **New video MLLMs** | VideoThinker-R1-3B (MIT, CVPR 2026) | ✅ Replace VideoChat-Flash 2B for reasoning |
| **New video MLLMs** | Video-Scan (2B) | ✅ Potential lightweight option |
| **Embedding models** | No new dedicated video embedding models | ✅ Stick with current stack |
| **MOT replacement** | Polycepta (June 2026) | 🔍 Wait for code; use BoT-SORT/OC-SORT now |
| **CVPR 2026 SGG** | DSFlash — 56 FPS panoptic SGG | ✅ Replace/enhance scene graph pipeline |
| **ECCV 2026 Video RAG** | Decoupling Semantics and Logic paper | 🔍 Adapt two-stage cascaded Video RAG pattern |
| **Gradio 6.19.0** | Workflow subgraphs as API endpoints (#13524) | ⭐ Most impactful — composable pipeline UI |

### 6.2 Prioritized v0.16 Plan

| Priority | Item | Effort | Notes |
|----------|------|--------|-------|
| **P0** | Entity tracking (within-video) | Medium | ByteTrack→BoT-SORT; Polycepta upcoming |
| **P1** | Gradio Workflow subgraphs | Medium | **6.19.0 makes this much easier** |
| **P1** | Cross-video entity matching | Medium | BGE-VL + lightweight ReID |
| **P2** | Sparse optical flow | Medium | RAFT/GMA for motion as auxiliary signal |
| **P2** | DSFlash scene graphs | Medium | 56 FPS, CVPR 2026 quality |
| **P3** | VideoThinker-R1-3B integration | Small | Plug-and-play, MIT license |

### 6.3 Model Stack for v0.16

```
Video Input
  → Scene Detection (PySceneDetect)
  → Frame Extraction
  → YOLO + BoT-SORT + Entity Track  │  Sparse Optical Flow (RAFT/GMA)
  → VideoThinker-R1-3B (or VCF-2B fallback)
  → BGE-VL Embed + DSFlash Scene Graph (CVPR 2026)
  → ChromaDB (BGE-VL + Qwen3-Embedding)
  → Agentic RAG (3-round) + Confidence-aware orchestration
  → Gradio 6.19 Workflow (subgraph APIs) + MCP + FastAPI
```

### 6.4 Key Sources

- VideoThinker-R1-3B: https://huggingface.co/Falconss1/VideoThinker-R1-3B | arxiv:2605.01324
- Polycepta MOT: arxiv:2606.23604
- DSFlash SGG: arxiv:2603.10538 (CVPR 2026)
- PhysRAG: arxiv:2606.26916 (ECCV 2026)
- Decoupling Semantics & Logic Video RAG: arxiv:2606.07924 (ACL 2026)
- Robust-TO: arxiv:2606.26904
- Gradio 6.19.0 PR #13524: https://github.com/gradio-app/gradio/pull/13524
- Gradio releases: https://github.com/gradio-app/gradio/releases
- MJEPA: arxiv:2606.25225
- HarmVideoBench: arxiv:2606.27187
- EVIS: arxiv:2606.26994
- ProtoKV: arxiv:2606.26762
