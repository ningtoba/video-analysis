# State-of-the-Art Video Understanding AI — June 2026

> **Research conducted:** 2026-06-26
> **Hardware target:** NVIDIA RTX 4070 (12 GB VRAM)
> **Project:** video-analysis v0.14.0 → v0.15.0

---

## Table of Contents

1. [Video MLLMs](#1-video-mllms)
2. [Action Recognition](#2-action-recognition)
3. [Scene Understanding](#3-scene-understanding)
4. [Frame Extraction & Sampling](#4-frame-extraction--sampling)
5. [Synthesis & Recommendations](#5-synthesis--recommendations)
6. [Sources](#6-sources)

---

## 1. Video MLLMs

### 1.1 The Landscape (June 2026)

| Model | Params | VRAM (BF16) | 12GB OK? | License | Notes |
|-------|--------|-------------|----------|---------|-------|
| **VideoChat-Flash 2B** | 2.0B | ~5.4 GB | Yes | MIT | ICLR 2026, 16 tok/frame, 99.1% NIAH@10K |
| **VideoChat-Flash 7B** | 6.9B | ~14 GB | No | MIT | Best quality, too large |
| **SmolVLM2 2.2B** | 2.2B | ~2-3 GB | Yes | Apache 2.0 | Outperforms all 2B on Video-MME |
| **SmolVLM2 500M** | 0.5B | ~0.5-1 GB | Yes | Apache 2.0 | Close to 2.2B, iPhone-local |
| **SmolVLM2 256M** | 0.256B | ~0.3 GB | Yes | Apache 2.0 | Experimental, smallest video LM |
| **VideoChat2-Flash-2B@224** | 2.0B | ~5 GB | Yes | MIT | InternVideo + Qwen backbone |
| **VideoChat2-Flash-7B@448** | 6.9B | ~15 GB | No | MIT | Best overall, >12GB |
| **Qwen3-Omni / 3.5-Omni** | 7B+ | ~14 GB+ | No | Apache 2.0 | Native multi-modal, 256K ctx |
| **Qwen2.5-VL-7B** | 7.0B | ~14 GB | No | Apache 2.0 | Needs quantization |
| **Gemini 2.5 Flash** | API | N/A | API | Proprietary | 1M context, ~258 tok/frame |
| **LLaVA-Video-7B** | 6.9B | ~14 GB | No | Apache 2.0 | Strong dense captioning |
| **InternVideo2.5** | Encoder | varies | Yes (enc only) | MIT | Backbone for VideoChat |

### 1.2 VideoChat-Flash (ICLR 2026) — Current Project Model

**Architecture:** Hierarchical Video Token Compression (HiCo)
- Compression ratio ~50:1 with almost no performance loss
- Encodes each frame into just **16 tokens** (vs ~256 for Gemini)
- 99.1% NIAH accuracy at 10,000+ frames
- Can process videos up to 3 hours long
- Multi-stage short-to-long learning with LongVid dataset

**Status:** Integrated in video_analysis/video_mllm.py as OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448. Provides describe_scene(), summarize_video(), answer().

**Latest:** Strong results on VideoEval-Pro benchmark (June 2025).

### 1.3 SmolVLM2 (HuggingFace, Feb 2025) — Recommended New Primary

**Key breakthrough:** First video LM family that runs on free Colab/consumer hardware.

**Architecture:**
- Standard transformers (AutoModelForImageTextToText) no trust_remote_code needed
- Native Flash Attention 2 support
- Straightforward video path input, model handles sampling
- Accepts up to 50 evenly-sampled frames by default

**VRAM savings:**
| Model | BF16 VRAM | Colab | RTX 4070 (12GB) |
|-------|-----------|-------|-----------------|
| SmolVLM2 2.2B | ~2-3 GB | Yes | Yes + 9GB headroom |
| SmolVLM2 500M | ~0.5-1 GB | Yes | Tiny footprint |
| VideoChat-Flash 2B | ~5.4 GB | Borderline | Yes (tighter) |

**Performance:** Outperforms all 2B models on Video-MME.

**Downsides vs VideoChat-Flash:**
- Less proven on 1h+ videos
- No token-level compression (256-384 tokens/frame)

### 1.4 New 2026 Trends

1. **Hierarchical compression** — HiCo (50:1), ToMe, PyramidDrop
2. **Linear-time attention** — Mamba-Transformer hybrids (Jamba, Nemotron Nano 2)
3. **Unified universal encoders** — EUPE (<100M params) distills CLIP+DINOv2+SAM+Depth
4. **Agentic video understanding** — Models reason about when/where to look
5. **On-device** — EdgeTAM at 16 FPS on iPhone, SmolVLM2 500M on phone

**NOT suitable for 12GB:** VideoChat-Flash 7B (~14GB), Qwen3-Omni (7B+), LLaVA-Video-7B.


## 2. Action Recognition

### 2.1 Current State: X-CLIP (Microsoft)

**Currently in project:** microsoft/xclip-base-patch16-zero-shot (~200M params, ~4 GB VRAM)

**Strengths:** Zero-shot open-vocabulary, 26 default categories, per-frame.

**Weaknesses:** No temporal reasoning, ~4 GB dedicated VRAM.

### 2.2 Best Alternatives for RTX 4070 (12 GB)

| Model | Params | VRAM | Temporal | Zero-shot | Notes |
|-------|--------|------|----------|-----------|-------|
| X-CLIP (current) | 200M | ~4 GB | No | Yes | Already integrated |
| **TimeSformer** | 120M | ~3 GB | Yes | Yes (via CLIP text) | SOTA Kinetics-400/600 |
| VideoMAE-v2-giant | 600M | ~6 GB | Yes | No (FT) | Best supervised |
| UniformerV2 | 125M | ~3 GB | Yes | No (FT) | Efficient, strong SSv2 |
| Motionformer | 180M | ~4 GB | Yes | No (FT) | Trajectory attention |
| SlowFast | 34M | ~2 GB | Yes | No (FT) | Classic, very light |

### 2.3 Recommendation

**Keep X-CLIP as lightweight per-frame option. Add TimeSformer as video-native backend.** TimeSformer uses ~3 GB VRAM, does divided spatial-temporal attention on 8-16 frame clips, and can use CLIP text embeddings for zero-shot classification. Falls back to X-CLIP when VRAM is constrained.


## 3. Scene Understanding

### 3.1 Current State

- **OpenCLIP** (ViT-B-32 / ViT-L-14) — 25 label zero-shot classification
- **VideoChat-Flash 2B** — optional NL scene descriptions

### 3.2 SOTA Approaches (2026)

| Approach | VRAM | Quality | Speed | Notes |
|----------|------|---------|-------|-------|
| OpenCLIP zero-shot | 0.5-2 GB | Basic | Fast | 25 labels current |
| **SmolVLM2 2.2B** | 2-3 GB | Rich NL | Medium | Full scene description |
| VideoChat-Flash 2B | 5.4 GB | Rich NL | Medium | Current MLLM |
| SAM 2 + GroundingDINO | 4-6 GB | Dense | Slow | Segment-then-classify |
| **EUPE universal** | <1 GB | Multi-task | Fast | Seg+depth+VLM <100M |
| DINOv2 + clustering | 0.5 GB | Good | Fast | Unsupervised parsing |

### 3.3 Key Developments

**EUPE (Efficient Universal Perception Encoder, Meta 2025-2026):**
- <100M params, distills DINOv2 + SAM 2 + CLIP + SigLIP into one backbone
- Matches domain experts on classification/dense prediction/VLM
- Ideal OpenCLIP replacement: one encoder for scene classification + segmentation + depth

**SAM 2 / EdgeTAM (Meta 2025-2026):**
- SAM 2 extends SAM with video memory module (FIFO queues + object pointers)
- EdgeTAM: 87.7 J&F on DAVIS, 16 FPS on iPhone 15 Pro Max

### 3.4 Recommendation

**Primary:** SmolVLM2 2.2B for scene descriptions (richer, uses less VRAM than VideoChat-Flash). OpenCLIP as fallback.
**Future:** EUPE as unified perception backbone.


## 4. Frame Extraction & Sampling

### 4.1 Current State

- Uniform 0.5 fps sampling
- PySceneDetect: adaptive/content/ffmpeg/histogram/hash detectors
- Config fields adaptive_frame_sampling and clip_frame_dedup exist but are **UNUSED**

### 4.2 SOTA in 2026

**A. Uniform 1 fps — Industry Standard**
Default across all major VLMs (Gemini, Qwen, LLaVA-Video). ~258 tokens/frame default resolution, ~66 at low.

**B. Adaptive Keyframe Sampling (AKS) — CVPR 2025**
Plug-and-play module with prompt-frame matching scores. Improves all downstream QA.

**C. DINOv2 Feature Dedup — Meta 2026**
Sample at 1 fps, compute DINOv2 features in 8-frame windows, drop similar neighbors.
~45.9% frame retention. Vision-centric features ideal for inter-frame comparison.

**D. M-LLM Frame Selector — Amazon 2025 (arXiv 2502.19680)**
Lightweight LLM selector with spatial + temporal pseudo-labeling. Improves all benchmarks.

**E. Small VLM Frame Strategy (arXiv 2509.14769)**
Finding: **Uniform FPS sampling is best for small VLMs.** Evenly spaced frames outperform keyframe-based.

### 4.3 Recommendation

**Phase 1 (v0.15.0, LOW effort):** Wire up existing adaptive_frame_sampling and clip_frame_dedup config fields. Config exists but pipeline.py never reads them. ~50 + ~30 lines of code.

**Phase 2 (v0.15.0+):** DINOv2-based frame deduplication (~0.5 GB VRAM).

**Phase 3 (v0.16.0):** AKS for prompt-aware frame selection during MLLM QA.


## 5. Synthesis & Recommendations

### Recommended Stack (RTX 4070 12GB)

| Module | Current | Recommended | VRAM Change | Priority |
|--------|---------|-------------|-------------|----------|
| Video MLLM | VideoChat-Flash 2B (~5.4 GB) | **SmolVLM2 2.2B** (~2-3 GB) + keep VChat-Flash option | -2.5 GB | HIGH |
| Action rec | X-CLIP 200M (~4 GB) | Keep X-CLIP, add TimeSformer (opt) | Same / +3 opt | MEDIUM |
| Scene desc | OpenCLIP + opt VChat | **SmolVLM2 2.2B** primary, OpenCLIP fallback | -2.5 GB | HIGH |
| Frame extract | Uniform 0.5 fps | Wire up adaptive + CLIP dedup (existing config) | Same | MEDIUM |
| Embedding | BGE-VL-base (~0.8 GB) | Keep | Same | Already good |

### Pipeline VRAM Budget (with SmolVLM2)

```
Total VRAM:               12.0 GB
OS/Drivers/CUDA:         ~0.5 GB
Available for AI:       ~11.5 GB

With SmolVLM2 2.2B:
  faster-whisper large-v3:   ~4.0 GB  (unloaded after transcription)
  YOLO26x:                   ~2.0 GB  (unloaded after detection)
  SmolVLM2 2.2B:            ~2.5 GB  (description + QA)
  BGE-VL-base:              ~0.8 GB  (stays loaded for embedding)
  X-CLIP:                   ~0 GB    (on-demand, unloaded after)
  --------------------------------
  Peak concurrent:          ~9.3 GB  12.0 GB — Comfortable
```

With VideoChat-Flash (current):
```
  VideoChat-Flash 2B:       ~5.4 GB  (instead of 2.5 GB)
  --------------------------------
  Peak concurrent:         ~12.2 GB  12.0 GB — Must sequentialize
```

### Key Decisions for v0.15.0

1. **Replace VideoChat-Flash with SmolVLM2 as primary** — frees 2-3 GB VRAM
2. **Multi-backend VideoMLLM** — auto selects SmolVLM2, fallback to VideoChat-Flash, optional Gemini API
3. **Wire up adaptive frame sampling + CLIP dedup** — config exists but unimplemented, low effort
4. **Keep X-CLIP** — still SOTA for zero-shot per-frame action recognition
5. **Consider EUPE** for unified perception <100M parameters


## 6. Sources

### Papers
- VideoChat-Flash: Li et al., ICLR 2026. arXiv:2501.00574
- SmolVLM2: HuggingFace, Feb 2025. arXiv:2504.05299
- A-RAG: Du et al., Feb 2026. arXiv:2602.03442
- Adaptive Keyframe Sampling (AKS): CVPR 2025
- M-LLM Frame Selection: Hu et al., 2025. arXiv:2502.19680
- Frame Sampling for SVLMs: arXiv:2509.14769
- VideoBrain: arXiv:2602.04094
- Efficient Video Intelligence 2026: Vikas Chandra (Meta)

### Models
- SmolVLM2: huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- VideoChat-Flash: github.com/OpenGVLab/VideoChat-Flash
- X-CLIP: huggingface.co/microsoft/xclip-base-patch16-zero-shot
- SAM 2 / EdgeTAM: Meta Research
- EUPE: Vikas Chandra (Meta)

### Web
- Gemini API video guide (April 2026) — 258 tokens/frame default
- Forasoft "Video VLMs In 2026 — Frame Sampling Vs Token Streaming" (May 31, 2026)
- HuggingFace Blog "SmolVLM2: Bringing Video Understanding to Every Device" (Feb 20, 2025)

---

*Research compiled by Hermes Agent, June 26, 2026*
