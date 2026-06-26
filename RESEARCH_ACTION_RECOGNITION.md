# Video Action Recognition for Self-Hosted Pipeline — Research Report (2025–2026)

**Date:** 2025-06-26
**Target Hardware:** RTX 4070 (12 GB VRAM)
**Existing Stack:** OpenCLIP ViT-B-32/L-14, YOLO, faster-whisper, PySceneDetect, ChromaDB
**Python:** 3.14

---

## 1. Executive Summary

The existing pipeline processes keyframes per scene with image-level models (OpenCLIP,
YOLO). Adding action/activity recognition is the natural next step. **The best fit for
this pipeline is X-CLIP (open-vocabulary, HuggingFace transformers-native, works on
keyframes, ~4 GB VRAM)**, followed by VideoMAE (highest closed-vocabulary accuracy,
86.5M params, 80.9% K400 top-1). For a lightweight first integration, TimeSformer base
provides the simplest path with the smallest memory footprint (~3 GB).

---

## 2. Landscape Overview

### 2.1 Open-Vocabulary Action Recognition (Zero-Shot, No Training Set)

| Model | Params | K400 Top-1 | ZS HMDB-51 | ZS UCF-101 | VRAM | Notes |
|-------|--------|-----------|------------|------------|------|-------|
| **X-CLIP** (base-p16) | ~200M | 80.4% (FT) | 44.6% | 72.0% | ~4 GB | Best overall fit — HF transformers, open-vocab, frame seqs |
| **X-CLIP** (base-p32) | ~200M | 80.4% | ~43% | ~70% | ~4 GB | Slightly less accurate |
| **FROSTER** | ~300M | ~83% | ~50% | ~76% | ~6 GB | ICLR 2024, frozen CLIP teacher |
| **Video-STAR** (2025) | ~300M+ | ~85% | — | — | ~8 GB | ICLR 2026, too heavy for 12GB total |
| **Open-VCLIP** | ~150M | ~77% | — | — | ~3 GB | Simpler, less accurate |

**X-CLIP base-patch16-zero-shot** (`microsoft/xclip-base-patch16-zero-shot` on HF)
requires zero training data, accepts action text queries, ranks frames against them.

### 2.2 Closed-Vocabulary Models (Kinetics-400 Supervised)

| Model | Params | K400 Top-1 | Input Frames | VRAM | FPS (4070) | HuggingFace ID |
|-------|--------|-----------|-------------|------|-----------|----------------|
| **VideoMAE-base** | 86.5M | **80.9%** | 16 | ~3.5 GB | ~40 | `MCG-NJU/videomae-base-finetuned-kinetics` |
| **VideoMAE-large** | 305M | 85.2% | 16 | ~7 GB | ~15 | `MCG-NJU/videomae-large-finetuned-kinetics` |
| **VideoMAEv2-base** | 86.2M | 81.2% | 16 | ~3.5 GB | ~40 | `OpenGVLab/VideoMAEv2-Base` |
| **TimeSformer-base** | 121M | 78.0% | 8 | ~3 GB | ~50 | `facebook/timesformer-base-finetuned-k400` |
| **ViViT-base** | 86M | 77.1% | 32 | ~4 GB | ~30 | `google/vivit-b-16x2-kinetics400` |
| **V-JEPA2 ViT-L** | 300M | ~83% | 64 | ~8 GB | ~10 | `facebook/vjepa2-vitl-fpc64-256` |
| **SlowFast R50** | 34M | 76.8% | 32 | ~2.5 GB | ~60 | Via PyTorchVideo |
| **TSN ResNet50** | 24M | 72.8% | 25 clips | ~1.5 GB | ~100 | Via MMAction2 |
| **TSM ResNet50** | 24M | 74.1% | 8 | ~1.5 GB | ~90 | Via MMAction2 |
| **MobileOne-S4 TSN** | 13.7M | 73.7% | 25 clips | ~1.0 GB | ~120 | Via MMAction2 |

### 2.3 Keyframe Compatibility

The pipeline extracts 1-5 keyframes per scene. Models that work with sparse frames:

| Model | Keyframe-Friendly | Why |
|-------|-----------------|-----|
| **X-CLIP** | ✅ Perfect | List of N images (2-32), cross-frame attention |
| **TimeSformer** | ✅ Good | 8 frames, spatial+temporal attention |
| **VideoMAE** | ✅ Good | 16 frames as patch sequences |
| **TSN** | ✅ Designed for it | Sparse temporal sampling strategy |
| **V-JEPA2** | ✅ Good | Flexible frame input |
| **SlowFast** | ⚠️ Needs dense | Two pathways need dense temporal stream |

---

## 3. Python Package Matrix

| Package | PyPI | Install | Video Models | HF Integration | Notes |
|---------|------|---------|-------------|---------------|-------|
| **transformers** | ✅ | `pip install transformers` | X-CLIP, VideoMAE, TimeSformer, ViViT, V-JEPA2 | ✅ Native | **Already installed** |
| **pytorchvideo** | ✅ | `pip install pytorchvideo` | SlowFast, R(2+1)D, MViT | ⚠️ torch.hub | Meta-maintained |
| **mmaction2** | ✅ | `pip install mmaction2` | 40+ models (needs mmcv-full) | ❌ No | Heavy deps, 2-day setup |
| **open-clip-torch** | ✅ | already installed | Image CLIP only (no temporal) | ❌ No | Could extend with custom pooling |

**Integration Difficulty:**

- **transformers**: 🟢 Easy — 2 hours, `pipeline("video-classification")` works
- **pytorchvideo**: 🟡 Medium — 4 hours, custom data pipeline
- **mmaction2**: 🔴 Hard — 2 days, mmcv-full, config system

---

## 4. VRAM Budget (RTX 4070 12GB)

**Already loaded in pipeline:**
- OpenCLIP ViT-B-32: ~1.5 GB (or ViT-L-14: ~4 GB)
- faster-whisper large-v3: ~3 GB (int8_float16)
- YOLO: ~2 GB
- **Total baseline: ~6.5 GB (or ~9 GB with ViT-L-14)**

**Remaining: ~3-5.5 GB (or ~3 GB with ViT-L-14)**

| Action Model | VRAM | Fits with ViT-L-14? | Fits with ViT-B-32? | Best Strategy |
|-------------|------|-------------------|-------------------|--------------|
| TimeSformer-base | ~3 GB | ⚠️ Tight | ✅ Yes | Load, infer, unload |
| X-CLIP base-p16 | ~4 GB | ⚠️ Tight | ✅ Yes | Load, infer, unload |
| VideoMAE-base | ~3.5 GB | ⚠️ Tight | ✅ Yes | Load, infer, unload |
| VideoMAE-large | ~7 GB | ❌ No | ⚠️ Too tight | Would spill over |
| TSN + MobileOne | ~1 GB | ✅ Yes | ✅ Yes | Can coexist |

**Strategy**: Load action model after image models finish per scene, then unload.
The `cleanup()` pattern already exists.

---

## 5. Integration Pattern

### Option A: X-CLIP Open-Vocabulary (Recommended)

```python
from transformers import AutoModelForVideoClassification, AutoImageProcessor
import torch

class ActionRecognizer:
    def __init__(self, model_name="microsoft/xclip-base-patch16-zero-shot",
                 device="cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_name).to(device)
        self.model.eval()

    def classify(self, frames: list, candidate_actions: list[str]) -> list[dict]:
        inputs = self.processor(frames, text=candidate_actions,
                                return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        results = []
        for fp in probs:
            top_idx = fp.argmax().item()
            results.append({
                "action": candidate_actions[top_idx],
                "confidence": fp[top_idx].item(),
                "all_probs": {candidate_actions[i]: fp[i].item()
                              for i in range(len(candidate_actions))},
            })
        return results
```

### Option B: VideoMAE — One-Liner

```python
from transformers import pipeline
pipe = pipeline("video-classification",
                model="MCG-NJU/videomae-base-finetuned-kinetics")
result = pipe(frames)  # [{"label": "archery", "score": 0.95}, ...]
```

---

## 6. Default Action Categories

```python
DEFAULT_ACTION_LABELS = [
    "a person walking", "a person running", "a person sitting",
    "a person standing", "people talking", "a person speaking",
    "a person cooking", "a person eating", "a person typing",
    "a person reading", "a person dancing", "a person exercising",
    "a person driving", "a person riding a bicycle",
    "a person playing an instrument", "a person using a phone",
    "a person shaking hands", "a person clapping", "a person jumping",
    "a person fighting", "a person throwing", "a person lifting",
    "no person visible",
]
```

---

## 7. Ranked Recommendations

| # | Model | VRAM | K400 Acc | Open-Vocab | Keyframe | Integration |
|---|-------|------|---------|-----------|----------|-------------|
| 🥇 1 | X-CLIP base-p16 | ~4 GB | 80.4% | ✅ Yes | ✅ Perfect | 🟢 Easy |
| 🥈 2 | VideoMAE-base | ~3.5 GB | 80.9% | ❌ (400 cls) | ✅ Good | 🟢 Easy |
| 🥉 3 | TimeSformer-base | ~3 GB | 78.0% | ❌ (400 cls) | ✅ Good | 🟢 Easy |
| 4 | TSN + MobileOne | ~1 GB | 73.7% | ❌ (400 cls) | ✅ Perfect | 🔴 Hard (MMAction2) |

### Implementation Plan

**Phase 1 (2-4 hours):** Add X-CLIP. Follows existing OpenCLIP zero-shot pattern,
uses `transformers` already installed, allows free-form action queries.

**Phase 2 (optional):** Add VideoMAE as secondary classifier for unlabeled fallback
(400 Kinetics action categories).

**Phase 3 (nice-to-have):** Per-scene classification using VideoMAE on the full
keyframe sequence for temporal consistency.

---

## 8. HuggingFace Model Zoo

| Model ID | Type | Size | Notes |
|----------|------|------|-------|
| `microsoft/xclip-base-patch16-zero-shot` | Open-vocabulary | 200M | **#1 pick** |
| `microsoft/xclip-base-patch32` | Closed-vocabulary | 200M | K400 fine-tuned |
| `MCG-NJU/videomae-base-finetuned-kinetics` | Closed-vocabulary | 86.5M | 80.9% K400 |
| `MCG-NJU/videomae-large-finetuned-kinetics` | Closed-vocabulary | 305M | 85.2% K400 |
| `OpenGVLab/VideoMAEv2-Base` | Pretrain only | 86.2M | CVPR 2023 |
| `facebook/timesformer-base-finetuned-k400` | Closed-vocabulary | 121M | 78% K400, lowest VRAM |
| `google/vivit-b-16x2-kinetics400` | Closed-vocabulary | 86M | 77.1% K400 |
| `facebook/vjepa2-vitl-fpc64-256` | Self-supervised | 300M | Meta 2025 |

---

## 9. Pipeline Changes Needed

**New FrameInfo fields:**
```python
action: Optional[str] = None
action_confidence: Optional[float] = None
action_all_probs: Optional[dict] = None
```

**New Config fields:**
```python
action_recognizer: str = "xclip"
action_model_name: str = "microsoft/xclip-base-patch16-zero-shot"
action_unload_after_inference: bool = True
```

**New pipeline step (after CLIP classification):**
```python
if hasattr(self.config, 'action_recognition_enabled') and self.config.action_recognition_enabled:
    self._classify_actions(scenes)
```

**New dependency:** None needed — `transformers` already installed.

---

## 10. Key References

- X-CLIP: https://arxiv.org/abs/2208.02816 (ECCV 2022 Oral)
- VideoMAE: https://arxiv.org/abs/2203.12602 (NeurIPS 2022 Spotlight)
- TimeSformer: https://arxiv.org/abs/2102.05095 (ICML 2021)
- FROSTER: https://arxiv.org/abs/2403.01560 (ICLR 2024)
- MMAction2: https://github.com/open-mmlab/mmaction2
- PyTorchVideo: https://pytorchvideo.org/
- HF Video Models: https://huggingface.co/models?pipeline_tag=video-classification
