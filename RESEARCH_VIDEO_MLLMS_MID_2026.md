# Video MLLM Research Report — Mid-2026

**Research conducted:** June 26-27, 2026
**Context:** video-analysis platform running on RTX 4070 (12GB VRAM)
**Current backends:** Qwen3-VL-30B-A3B (FP8 via vLLM), SmolVLM2, VideoChat-Flash, BGE-VL-base

---

## 1. NEW Video MLLMs That Fit 12GB VRAM

### InternVideo3 (NEW — June 10, 2026)
- **ArXiv:** 2606.12195 (submitted June 10, 2026)
- **Base:** Built on Qwen3-VL-8B
- **Key innovation:** Multimodal Contextual Reasoning (MCR) + M^2LA (KV-cache compression)
- **VRAM footprint:** ~8B params — fits 12GB at INT4/NF4 (~6GB) or FP8 (~10GB)
- **Benchmark results (open-weight 8B-class):**
  - **Video-MME: 73.8** (best; Qwen3-VL-8B: 71.4, Eagle2.5: 72.4)
  - **MLVU: 77.3** (best; Qwen3-VL-8B: 57.6)
  - **EgoSchema: 76.6** (best; Qwen3-VL-8B: 69.8)
  - **VRBench: 69.4** (best; Qwen3-VL-8B: 59.4)
  - **Short QA Avg: 69.0** (best; Qwen3-VL-8B: 65.3)
  - **1.84x decode speedup** vs Qwen3-VL-8B via M^2LA at 32K tokens
- **Verdict: STRONGEST CANDIDATE — best open-weight video MLLM as of June 2026**

### VideoLLaMA3-7B (Jan 2025, Apache 2.0)
- **ArXiv:** 2501.13106 — Code: github.com/DAMO-NLP-SG/VideoLLaMA3
- **Weights:** huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B
- **Key:** Any-resolution Vision Tokenization (AVT), Differential Frame Pruner (DiffFP)
- **VRAM:** ~6GB INT4 — runs on 12GB
- **Verdict: SOLID OPTION, available now**

### Models NOT found (no significant 2025-2026 updates):
- VideoPrism2, Mantis-8B, VideoChat2, VideoMamba, LanguageBind

---

## 2. Efficient Video MLLM Inference

### M^2LA (InternVideo3) — KV-cache compression, 1.84x faster decode
### FlashAttention-3 — H100 gains; RTX 4070 sees marginal benefit vs FA2
### torch.compile — Limited for variable video lengths; useful for text decode
### KV-cache Quantization — KIVI (2-4x), FP8 (vLLM), W4A16
### Speculative Decoding — Pair SmolVLM2 draft with Qwen3-VL

---

## 3. Video Frame Selection

### DiffFP (VideoLLaMA3): Frame pruning by inter-frame similarity
### Your project already has: DINO compression, PySceneDetect, configurable frame rate
### Enhancement: Add DiffFP-style redundancy filter as preprocessing step

---

## 4. Lightweight Approaches for 12GB

### Current (fits comfortably):
- Qwen3-VL-30B-A3B FP8 (~8-10GB), VideoChat-Flash 2B (~5GB), SmolVLM2 (~1-3GB), BGE-VL (~0.8GB)

### Top additions:
1. **InternVideo3 (8B)** — when weights release (~6-10GB)
2. **VideoLLaMA3-7B** — available now (~6GB Q4)

---

## 5. Video Embedding Models

### BGE-VL-base (current): 150M, 0.8GB — still the best at its VRAM cost
### Upgrade path: Qwen3-VL-Embedding-2B (~4GB, already in config)
### No dramatically better video-specific embedding model at similar size

---

## Summary Recommendations

### HIGHEST PRIORITY — Add InternVideo3 when weights release
### SECONDARY — Add VideoLLaMA3-7B (available now)
### EFFICIENCY — Add DiffFP-style frame pruning for all backends
### EMBEDDING — Keep BGE-VL-base; Qwen3-VL-Embedding-2B as optional upgrade

## Sources
1. arxiv.org/abs/2606.12195 (InternVideo3, June 2026)
2. arxiv.org/abs/2501.13106 (VideoLLaMA3, Jan 2025)
3. huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B
4. huggingface.co/BAAI/BGE-VL-base
5. github.com/DAMO-NLP-SG/VideoLLaMA3
6. github.com/OpenGVLab/InternVideo
