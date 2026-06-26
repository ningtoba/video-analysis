# Video Analysis Research 2024–2026

> Comprehensive survey of state-of-the-art approaches for a self-hosted video analysis
> platform targeting a **12GB RTX 4070** GPU.
>
> Current stack: FFmpeg scene filter, OpenCLIP ViT-B-32, YOLO v26, faster-whisper.
> This document identifies the most practical upgrades and additions.

---

## Table of Contents

1. [Frame Extraction & Scene Detection](#1-frame-extraction--scene-detection)
2. [Video Understanding Models](#2-video-understanding-models)
3. [Speech & Audio Processing](#3-speech--audio-processing)
4. [Motion Analysis & Action Recognition](#4-motion-analysis--action-recognition)
5. [Text Extraction from Video (OCR)](#5-text-extraction-from-video-ocr)
6. [Emerging Approaches & Trends](#6-emerging-approaches--trends)
7. [Recommendations Summary](#7-recommendations-summary)

---

## 1. Frame Extraction & Scene Detection

### 1.1 TransNet V2 (Shot Boundary Detection)

| Property | Value |
|---|---|
| **Paper** | [TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection](https://www.researchgate.net/publication/385306316) (2024) |
| **Repository** | [github.com/soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) |
| **License** | MIT |
| **Type** | Deep learning shot boundary detection (not heuristic) |
| **GPU RAM** | ~2–3 GB (tiny, runs easily on any GPU) |
| **RTX 4070 (12 GB)** | ✅ Practical |

**Overview:** A deep convolutional network designed specifically for shot transition detection. Unlike FFmpeg's `scene` filter (which uses simple pixel-difference heuristics), TransNetV2 learns to detect cuts, fades, and dissolves from data. Its PyTorch inference path is lightweight.

**F1 Scores reported:**
- ClipShots: 77.9%
- BBC Planet Earth: 96.2%
- RAI: 93.9%

**Key advantage over current approach:** The current `pipeline.py` uses FFmpeg's `select='gt(scene,0.3)'` — a single-threshold heuristic that misses dissolves, fades, and slow transitions. TransNetV2 catches these.

**Usage pattern (PyTorch inference):**

```python
from transnetv2 import TransNetV2

model = TransNetV2()
video_frames, preds, single_frame_preds = model.predict_video("video.mp4")

# preds[i] == True means frame i is a shot boundary
boundary_indices = [i for i, p in enumerate(preds) if p]
boundary_times = [i / fps for i in boundary_indices]
```

**Note:** The research paper "[AUTOSHOT: A Short Video Dataset and State-of-the-Art](https://openreview.net/pdf?id=u89Eq-_3oE4)" (2024) shows newer methods outperforming TransNetV2 on short-form video, but TransNetV2 remains the gold standard for general-purpose footage.

### 1.2 PySceneDetect v2 (Content-Aware Detector)

| Property | Value |
|---|---|
| **Repository** | [github.com/Breakthrough/PySceneDetect](https://github.com/breakthrough/pyscenedetect) |
| **License** | BSD 3-Clause |
| **Latest version** | v0.6.x (2024–2025 active development) |
| **GPU RAM** | CPU only (OpenCV-based) |
| **RTX 4070 (12 GB)** | ✅ Practical (uses no GPU) |

**Overview:** PySceneDetect v0.6 adds a **content-aware** detector that uses HSV histogram differences combined with adaptive thresholding — still heuristic but more robust than FFmpeg's scene filter. It also has a `detect-threshold` mode for fade/cut detection.

**Key advantage over FFmpeg alone:** PySceneDetect's `ContentDetector` with adaptive threshold is more accurate than FFmpeg's single threshold parameter, and its `AdaptiveDetector` (added in v0.6) handles varying content types automatically.

**Recommended hybrid approach:** Use TransNetV2 as primary detector, PySceneDetect's `AdaptiveDetector` as fallback.

```python
# PySceneDetect v0.6
from scenedetect import open_video, AdaptiveDetector

video = open_video("video.mp4")
detector = AdaptiveDetector(adaptive_threshold=3.0)
scene_list = detector.detect(video)

# scene_list contains timecodes for each scene
for scene in scene_list:
    print(f"Scene: {scene[0].get_timecode()} -> {scene[1].get_timecode()}")
```

### 1.3 Two-Stage Hybrid Approach (Best Practice)

Recent work (Soucek and Lokoc, 2024) demonstrates a two-stage pipeline:

1. **PySceneDetect** (fast, heuristic-based) → initial scene splits
2. **TransNetV2** → compensates for false negatives from stage 1

This is the recommended approach: fast first pass + deep learning refinement.

---

## 2. Video Understanding Models

### 2.1 InternVideo2 / InternVideo2.5

| Property | Value |
|---|---|
| **Paper** | [InternVideo2: Scaling Foundation Models for Multimodal Video Understanding](https://arxiv.org/abs/2403.15377) (ECCV 2024) |
| **Repository** | [github.com/OpenGVLab/InternVideo](https://github.com/opengvlab/internvideo) |
| **License** | Apache 2.0 |
| **Model sizes** | S (309M), B (309M), L (???), 1B, 6B |
| **GPU RAM** | S/B: ~4–6 GB; 1B: ~8–10 GB; 6B: ~24 GB (FP16) |
| **RTX 4070 (12 GB)** | ✅ InternVideo2-S/B/L ✅, 1B ⚠️ (tight), 6B ❌ |

**Overview:** State-of-the-art video foundation model family achieving 92.1% Top-1 on Kinetics-400. InternVideo2 unifies masked video modeling, crossmodal contrastive learning, and next-token prediction. InternVideo2.5 (released Jan 2025) adds long-context capabilities for processing thousands of frames.

**Key capabilities:**
- Action recognition
- Video-text retrieval
- Video captioning
- Video question answering
- Temporal localization
- **Audio understanding** (video+audio+speech)

**Practical use for 12 GB GPU:** The InternVideo2-S (distilled, small model) runs comfortably on 12 GB and can be used for:
- Generating video-level feature embeddings (~1024-dim)
- Action recognition
- Scene classification that understands temporal dynamics (unlike OpenCLIP which is per-frame)

```python
# InternVideo2 inference pattern (multi-modality)
from transformers import AutoModel, AutoConfig

model = AutoModel.from_pretrained(
    "OpenGVLab/InternVideo2-Stage2-1B",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).cuda()

# Video frames → video embedding
video_tensor = load_video_frames("video.mp4", num_frames=16)  # shape: [1, 16, 3, H, W]
video_emb = model.get_video_features(video_tensor)
```

### 2.2 LLaVA-NeXT (Video)

| Property | Value |
|---|---|
| **Blog** | [LLaVA-NeXT: A Strong Zero-shot Video Understanding Model](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) (Apr 2024) |
| **Repository** | [github.com/LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) |
| **License** | Apache 2.0 |
| **Model sizes** | 7B, 13B, 34B, 72B, 110B |
| **GPU RAM** | 7B: ~14–16 GB (FP16), ~8 GB (4-bit) |
| **RTX 4070 (12 GB)** | ⚠️ 7B with 4-bit quantization, ❌ FP16 |

**Overview:** Surprising finding — LLaVA-NeXT trained only on **image** data achieves strong zero-shot video understanding. It processes videos by uniformly sampling N frames and feeding them as a sequence to the LLM. DPO training with AI feedback on videos yields further improvements.

**Key insight:** The image-only-trained model transfers to video zero-shot because it uses a unified visual representation. This means you don't necessarily need video-specific training data.

**Practical use for 12 GB GPU:** With 4-bit quantization (bitsandbytes):
```python
# LLaVA-NeXT video inference (4-bit quantized on 12 GB)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "liuhaotian/llava-v1.6-vicuna-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_name=get_model_name_from_path(model_path),
    load_4bit=True,  # 4-bit quantization
    device="cuda",
)

# Uniformly sample 8 frames from video
frames = sample_video_frames("video.mp4", num_frames=8)
# Each frame processed through vision tower → LLM understands temporal sequence
```

### 2.3 Video-LLaVA

| Property | Value |
|---|---|
| **Paper** | [Video-LLaVA: Learning United Visual Representation by Alignment Before Projection](https://arxiv.org/abs/2311.10122) (EMNLP 2024) |
| **Repository** | [github.com/PKU-YuanGroup/Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) |
| **License** | Apache 2.0 |
| **Model sizes** | 7B |
| **GPU RAM** | ~16 GB (FP16), ~8–9 GB (4-bit) |
| **RTX 4070 (12 GB)** | ⚠️ 7B with 4-bit quantization |

**Overview:** Video-LLaVA jointly trains on images and videos using a unified visual representation (LanguageBind encoder). The key innovation is aligning visual features **before** the projection layer to the LLM, enabling shared representations between images and videos.

**Practical for 12 GB?** Tight but possible with 4-bit quantization and Flash Attention 2. Expect ~6–8 tokens/s on RTX 4070.

### 2.4 VideoChat2 (InternVideo2 Stage 3)

| Property | Value |
|---|---|
| **Repository** | part of [OpenGVLab/InternVideo](https://github.com/opengvlab/internvideo) |
| **License** | Apache 2.0 |
| **Model sizes** | 7B (based on InternLM2) |
| **GPU RAM** | ~14–16 GB (FP16) |
| **RTX 4070 (12 GB)** | ⚠️ with 4-bit quantization |

**Overview:** VideoChat2 is the video-dialogue model built on InternVideo2 encoders + InternLM2. Stage 3 training enables conversational video understanding. Released Aug 2024 with a longer context window.

---

## 3. Speech & Audio Processing

### 3.1 WhisperX

| Property | Value |
|---|---|
| **Repository** | [github.com/m-bain/whisperX](https://github.com/m-bain/whisperX) |
| **License** | BSD 2-Clause |
| **Stars** | 22.7k ⭐ |
| **GPU RAM** | ~5–8 GB (large-v2 + alignment model + diarization) |
| **RTX 4070 (12 GB)** | ✅ Practical |

**Overview:** WhisperX extends faster-whisper with three critical features the current platform lacks:

1. **Word-level timestamps** — using wav2vec2 forced alignment (vs. current code which relies on `seg.words` which is often None)
2. **Speaker diarization** — using pyannote-audio to label "who spoke when"
3. **Batched inference** — up to 70× realtime with large-v2
4. **VAD preprocessing** — reduces hallucination

**Current gap in platform:** `pipeline.py` line 332–341 tries to access `seg.words` but this attribute is frequently `None` or unreliable with faster-whisper alone. WhisperX's wav2vec2 alignment produces accurate word boundaries.

```python
# WhisperX — drop-in upgrade for faster-whisper
import whisperx

device = "cuda"
audio = whisperx.load_audio("audio.wav")

# 1. Transcribe with word-level timestamps
model = whisperx.load_model("large-v3", device, compute_type="int8_float16")
result = model.transcribe(audio, batch_size=16)
# result["segments"][i]["words"] has start, end, word for every word

# 2. Align words (wav2vec2 forced alignment)
align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
result = whisperx.align(
    result["segments"], align_model, metadata, audio, device=device,
    return_char_alignments=False,
)

# 3. Diarize (assign speaker labels)
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token="YOUR_HF_TOKEN",  # requires HF login for pyannote
    device=device,
)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)
# result["segments"][i]["speaker"] = "SPEAKER_00" or "SPEAKER_01"
```

**Limitation:** PyAnnote speaker diarization requires accepting their license on HuggingFace and providing an auth token. The diarization itself uses ~2–3 GB GPU memory on top of the Whisper model.

### 3.2 PyAnnote Audio (Speaker Diarization 3.1)

| Property | Value |
|---|---|
| **HuggingFace** | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| **License** | MIT (but gated — requires user agreement) |
| **GPU RAM** | ~2–3 GB |
| **RTX 4070 (12 GB)** | ✅ Practical |

**Overview:** State-of-the-art speaker diarization pipeline combining voice activity detection (VAD), speaker change detection, overlapped speech detection, and embedding clustering.

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN",
)
pipeline.to(torch.device("cuda"))

diarization = pipeline("audio.wav")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s -> {turn.end:.1f}s")
```

### 3.3 Silero VAD (Voice Activity Detection)

| Property | Value |
|---|---|
| **Repository** | [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| **License** | MIT |
| **GPU RAM** | <500 MB |
| **RTX 4070 (12 GB)** | ✅ Trivial |

**Overview:** Pre-trained VAD model that detects speech segments in audio. Much more accurate than the VAD filter built into faster-whisper. Can be used as a preprocessing step before transcription to reduce hallucination.

```python
import torch
import silero_vad

vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
)
(get_speech_timestamps, _, _, _, _) = utils

speech_timestamps = get_speech_timestamps(
    audio_waveform, vad_model,
    threshold=0.5,  # sensitivity
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
)
```

---

## 4. Motion Analysis & Action Recognition

### 4.1 UniFormerV2

| Property | Value |
|---|---|
| **Paper** | [UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer](https://arxiv.org/abs/2211.09552) (ICCV 2023) |
| **Repository** | [github.com/OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2) |
| **License** | Apache 2.0 |
| **Backbones** | ViT-B, ViT-L |
| **GPU RAM** | ViT-B: ~6–8 GB; ViT-L: ~12–14 GB |
| **RTX 4070 (12 GB)** | ✅ ViT-B ✅, ViT-L ⚠️ (tight) |

**Overview:** Arms a frozen image ViT with a lightweight video UniFormer block for temporal modeling. Achieves strong action recognition by leveraging pre-trained image Transformers — no need to train from scratch on video.

**Practical use for 12 GB GPU:** The ViT-B variant runs comfortably on 12 GB and provides:
- Action recognition on 16–32 frame clips
- Feature extraction for downstream tasks
- Can be used via MMAction2 integration

```python
# UniFormerV2 via mmaction2
from mmaction.apis import inference_recognizer, init_recognizer

config = "configs/recognition/uniformerv2/uniformerv2_base_32x8_k400.py"
checkpoint = "checkpoints/uniformerv2_base_k400.pth"
model = init_recognizer(config, checkpoint, device="cuda:0")

result = inference_recognizer(model, "video.mp4")
# result['pred_label'] = 'playing guitar', 'running', etc.
```

### 4.2 VideoMAE v2

| Property | Value |
|---|---|
| **Paper** | [VideoMAE v2: Scaling Video Masked Autoencoders](https://arxiv.org/abs/2303.16727) (2023) |
| **Repository** | Part of [VideoMAE](https://github.com/OpenGVLab/VideoMAE) |
| **License** | MIT |
| **GPU RAM** | Base: ~6 GB; Large: ~12 GB |
| **RTX 4070 (12 GB)** | ✅ Base ✅, Large ⚠️ (tight) |

**Overview:** Video masked autoencoder that learns spatiotemporal representations by masking random patches in video clips. v2 scales the approach with a dual masking strategy.

**Practical note:** VideoMAE is primarily designed for training/fine-tuning. For inference-only use, UniFormerV2 or InternVideo2-S are more practical.

### 4.3 TimeSformer

| Property | Value |
|---|---|
| **Paper** | [TimeSformer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) (ICML 2022) |
| **Repository** | [github.com/facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer) |
| **License** | MIT |
| **GPU RAM** | ~6 GB (Base) |
| **RTX 4070 (12 GB)** | ✅ Practical |

**Overview:** Pioneering video Transformer using divided space-time attention. Now somewhat superseded by newer methods but remains a solid, well-understood baseline.

### 4.4 RAFT (Optical Flow)

| Property | Value |
|---|---|
| **Paper** | [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) (ECCV 2020) |
| **Repository** | [github.com/princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) |
| **License** | BSD 3-Clause |
| **GPU RAM** | ~2–3 GB |
| **RTX 4070 (12 GB)** | ✅ Practical |

**Overview:** Dense optical flow estimation between pairs of frames. Useful for:
- Detecting motion boundaries (which often correlate with scene changes)
- Identifying camera movement (pan, tilt, zoom)
- Computing motion magnitude as a video activity signal

```python
# RAFT optical flow between consecutive frames
import torch
from raft import RAFT

model = RAFT()
model.load_state_dict(torch.load("raft-things.pth"))
model.cuda()

flow_low, flow_up = model(frame1, frame2, iters=12, test_mode=True)
# flow_up shape: [1, 2, H, W] — x and y displacement per pixel
motion_magnitude = torch.norm(flow_up, dim=1).mean().item()
```

**Integration for scene detection:** Compute motion magnitude between consecutive frames. Very low motion during a fade-to-black → potential scene boundary.

---

## 5. Text Extraction from Video (OCR)

### 5.1 PaddleOCR (PP-OCRv5)

| Property | Value |
|---|---|
| **Repository** | [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| **Latest version** | v3.0 (May 2025) with PP-OCRv5 |
| **License** | Apache 2.0 |
| **Stars** | 83.9k ⭐ |
| **GPU RAM** | ~1–2 GB (PP-OCRv5), ~0.5 GB (PP-OCRv4 mobile) |
| **RTX 4070 (12 GB)** | ✅ Trivial |

**Overview:** The state-of-the-art open-source OCR engine as of 2025–2026. PP-OCRv5 (released May 2025) is a 5-million-parameter specialist model that beats billion-parameter VLMs (like Qwen2.5-VL 72B) on OmniDocBench. Three-stage pipeline: detection → direction classification → recognition.

**Key advantages for video:**
- Extremely lightweight — runs easily alongside other models on the same GPU
- Handles scene text (signs, captions, overlays, lower thirds)
- PP-OCRv5 achieves ~97% accuracy on scene text benchmarks
- Supports 100+ languages
- GPU-accelerated end-to-end

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,  # detect text orientation
    lang='en',
    use_gpu=True,
    gpu_mem=1024,  # limit GPU memory to 1 GB
)

# Process each key frame
for frame in key_frames:
    result = ocr.ocr(frame.filepath, cls=True)
    # result = [[[bbox, (text, confidence)], ...], ...]
    for line in result[0]:
        bbox = line[0]  # 4 corner points
        text = line[1][0]
        confidence = line[1][1]
        frame.ocr_text = text
```

**Alternative: RapidOCR** — A reimplementation of PaddleOCR that is GPU-agnostic (doesn't require PaddlePaddle).  Uses ONNX Runtime. Slightly less accurate but much easier to install (no PaddlePaddle dependency).

```python
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
result, elapse = engine(frame_path)
# result = [bbox, text, confidence]
```

**Recommendation:** Install `paddleocr` with `paddlepaddle-gpu` for best accuracy, or `rapidocr-onnxruntime` for hassle-free installation on any GPU.

### 5.2 TrOCR

| Property | Value |
|---|---|
| **Repository** | [github.com/microsoft/unilm/tree/master/trocr](https://github.com/microsoft/unilm) |
| **License** | MIT |
| **GPU RAM** | ~2–4 GB |
| **RTX 4070 (12 GB)** | ✅ Practical |

**Overview:** Transformer-based OCR (encoder-decoder) from Microsoft. Better at reading handwritten text than PaddleOCR, but slower and heavier. For video scene text (usually printed/synthetic), PaddleOCR is the better choice.

### 5.3 Current Platform Gap

`FrameInfo` already has an `ocr_text` field (models.py line 20), but nothing populates it — the `_describe_scenes_clip` method only runs OpenCLIP classification. Adding PaddleOCR to the pipeline would populate `ocr_text` for every key frame.

---

## 6. Emerging Approaches & Trends

### 6.1 InternVideo2.5 — Long-Context Video Modeling

**What it is:** Released Jan 2025. Extends InternVideo2 to handle **thousands of frames** with hierarchical token compression. Uses dense vision task annotations and direct preference optimization.

**Why it matters:** Previous video models max out at 32–64 frames. InternVideo2.5 can process entire long videos (10+ minutes) in a single forward pass via token compression.

**Relevance to 12 GB GPU:** The compression mechanism makes it more memory-efficient than prior approaches — may fit in ~10 GB with the smaller backbone.

### 6.2 InternVideo3 — Agent-Based Contextual Reasoning

**What it is:** InternVideo3 treats video understanding as an agent-based problem — a model that can call tools, refine its observations, and answer complex queries through multi-step reasoning.

**Relevance:** Emergent trend toward "video agents" rather than monolithic video models. This aligns well with the existing platform architecture (pipeline-based processing + RAG).

### 6.3 Video LLMs as Feature Extractors

**Trend:** Rather than using OpenCLIP for per-frame features, newer approaches use Video LLMs (InternVideo2, LLaVA-NeXT) to generate **rich semantic descriptions** of scenes. These descriptions are then indexed (via sentence-transformers) for RAG.

**Relevance to current platform:** Replace OpenCLIP zero-shot classification with internVideo2 feature embeddings or LLaVA-NeXT-generated scene descriptions. Both provide richer semantic understanding.

### 6.4 Hybrid Audio-Visual Scene Detection

**Trend:** Combining visual shot detection (TransNetV2) with **audio scene change detection** (using spectrogram analysis or wav2vec2 features). A scene cut in both modalities is a much stronger boundary signal than either alone.

### 6.5 Efficient Video Transformers

**Trend:** Token merging, hierarchical attention, and frame sampling strategies that reduce the computational cost of processing many frames. Key papers to watch:
- **VideoMamba** (2024) — State space models for video
- **VideoPruner** — Pruning irrelevant frames before LLM processing
- **Any-resolution video models** — Processing varying resolution inputs

---

## 7. Recommendations Summary

### High Priority — Ready to Integrate (Works on 12 GB)

| Component | Replace/Add | Model | Ease | Impact |
|---|---|---|---|---|
| **Scene detection** | Replace FFmpeg heuristic | TransNetV2 + PySceneDetect AdaptiveDetector | Easy | **High** — catches dissolves/fades missed today |
| **Word-level transcripts** | Upgrade faster-whisper → WhisperX | WhisperX (large-v3) with wav2vec2 alignment | Medium | **High** — accurate word timestamps for search |
| **OCR** | Add to pipeline | PaddleOCR PP-OCRv5 or RapidOCR | Easy | **High** — currently gap in FrameInfo.ocr_text |
| **Speaker diarization** | Add to pipeline | pyannote/speaker-diarization-3.1 | Medium | **Medium** — who said what per segment |
| **VAD** | Add preprocessing | Silero VAD | Easy | **Medium** — reduces hallucination |

### Medium Priority — Test on 12 GB

| Component | Model | Notes |
|---|---|---|
| **Action recognition** | UniFormerV2-B (via MMAction2) | Heavy but adds temporal understanding |
| **Video feature embeddings** | InternVideo2-S (distilled) | Better than OpenCLIP for video tasks |
| **Scene description (LLM)** | LLaVA-NeXT-7B (4-bit) | Rich semantic descriptions per scene |
| **Optical flow** | RAFT | Motion analysis + camera movement detection |

### Low Priority — Watch

| Component | Reason |
|---|---|
| InternVideo2-1B/6B | Requires >12 GB or heavy quantization |
| Video-LLaVA 7B (FP16) | Too big for 12 GB |
| VideoChat2 7B | Alternative to LLaVA-NeXT |
| VideoMAE v2 Large | Training-focused, not inference |

### Architecture Recommendation

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: TransNetV2 → shot boundaries (deep learning)        │
│  Step 2: PySceneDetect AdaptiveDetector → fallback           │
│  Step 3: WhisperX → transcript + word timestamps + speakers  │
│  Step 4: Silero VAD → audio preprocessing                    │
│  Step 5: PaddleOCR → text detection per key frame            │
│  Step 6: InternVideo2-S → video-level feature embeddings     │
│  Step 7: UniFormerV2-B → action recognition per scene        │
│  Step 8: RAFT → optical flow / motion analysis               │
│  Step 9: LLaVA-NeXT (4-bit) → scene description (optional)   │
│                                                              │
│  Output: VideoIndex with richer annotations                  │
└─────────────────────────────────────────────────────────────┘
```

### GPU Memory Budget (Target: 12 GB)

```
                 ┌──────────────────┐
  WhisperX       │ 5–6 GB           │ ← releases after transcription
                 ├──────────────────┤
  TransNetV2     │ 2 GB             │ ← tiny, run anytime
                 ├──────────────────┤
  PaddleOCR      │ 1–2 GB           │ ← tiny, can share CUDA context
                 ├──────────────────┤
  UniFormerV2    │ 6–8 GB           │ ← run after Whisper is unloaded
                 ├──────────────────┤
  InternVideo2-S │ 4–5 GB           │ ← run after other models freed
                 ├──────────────────┤
  LLaVA-NeXT     │ 8 GB (4-bit)     │ ← heavy, run last / optionally
                 └──────────────────┘
```

**Key insight:** Models don't need to be loaded simultaneously. The pipeline unloads each model after its stage completes (see `pipeline.cleanup()`). This is already partially implemented with the `clip_unload_after_inference` flag.

---

## Quick-Start Commands

```bash
# Install new dependencies
pip install transnetv2                   # TransNetV2 (PyTorch)
pip install whisperx                      # WhisperX (drop-in upgrade)
pip install paddlepaddle-gpu paddleocr    # PaddleOCR (GPU)
pip install rapidocr-onnxruntime          # or RapidOCR (no PaddlePaddle)
pip install pyannote.audio                # Speaker diarization
pip install silero-vad                    # Voice activity detection
pip install mmaction2                     # UniFormerV2 / action recognition

# For video LLMs (optional, heavy):
pip install transformers accelerate bitsandbytes
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
```

---

*Research compiled June 2026. Focus on practical, working solutions for a 12 GB RTX 4070.*
