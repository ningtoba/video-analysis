# Video-Specific Evaluation, Testing & Quality Improvements

**Scope:** Research findings for video-analysis platform v0.14.0  
**Date:** 2026-06-26  
**Author:** Hermes Agent (subagent)

---

## Table of Contents

1. [Video Quality Assessment (VQA) Models](#1-video-quality-assessment-vqa-models)
2. [Benchmark Datasets for Pipeline Evaluation](#2-benchmark-datasets-for-pipeline-evaluation)
3. [Automated Pipeline Output Accuracy Evaluation](#3-automated-pipeline-output-accuracy-evaluation)
4. [Testing Frameworks for Video Pipelines](#4-testing-frameworks-for-video-pipelines)
5. [Performance Benchmarking for Pipeline Stages](#5-performance-benchmarking-for-pipeline-stages)
6. [Integration Test Infrastructure for GPU Tests](#6-integration-test-infrastructure-for-gpu-tests)
7. [Video Deduplication / Near-Duplicate Detection](#7-video-deduplication--near-duplicate-detection)
8. [Data Versioning for Video Datasets](#8-data-versioning-for-video-datasets)
9. [Error Handling & Recovery in Long-Running Pipelines](#9-error-handling--recovery-in-long-running-pipelines)
10. [Priority Recommendations](#10-priority-recommendations)

---

## 1. Video Quality Assessment (VQA) Models

### Can any run in ≤12 GB VRAM? **Yes.**

Three state-of-the-art open-source NR-VQA (No-Reference Video Quality Assessment) models are feasible:

| Model | Venue | VRAM Est. | Notes |
|-------|-------|-----------|-------|
| **VQA² (Q-Future)** | ACMMM 2025 | ~8-10 GB | LLaVA-OV / Qwen2.5-VL based. Updated 2025/5: 4× memory-efficient training pipeline. Can score AND explain quality in language. |
| **VQAThinker** | AAAI 2026 | ~8-12 GB | First open-source NR-VQA + RL. Does quality scoring + understanding/explanation. Excellent generalization to OOD videos. |
| **RQ-VQA** | CVPRW 2025 | ~6-8 GB | Swin-B + SlowFast backbone. Lightweight, efficient. Strong on NTIRE/compression benchmarks. |

**Recommendation:** RQ-VQA has the smallest footprint. VQAThinker or VQA² are better if explainable output is desired (they produce natural-language quality descriptions alongside scores).

### How to integrate:
```python
# Pseudo-integration for RQ-VQA (lightest option)
from rqvqa import RQ_VQA
model = RQ_VQA(device="cuda")  # ~6-8 GB VRAM
score = model.predict(video_path)  # returns MOS score (0-100)
explanation = model.explain(video_path)  # if supported
```

### Potential use in video-analysis:
- **Post-pipeline quality gate**: skip or flag videos below quality threshold before expensive processing
- **Scene-level quality**: measure per-scene quality to detect encoding artifacts
- **Metadata enrichment**: store quality score in `VideoIndex` for downstream filtering

---

## 2. Benchmark Datasets for Pipeline Evaluation

### Video Understanding Benchmarks (for MLLM/chat accuracy)

| Dataset | Year | Description | Size | Best For |
|---------|------|-------------|------|----------|
| **Video-MME** | CVPR 2025 | First comprehensive MLLM video eval; 30s-60min videos, 6 domains, 29 tasks | ~900 videos | End-to-end chat accuracy |
| **Video-MME-v2** | 2026 | Redesigned from first principles; progressive difficulty; anti-bias | ~1000 videos | Harder, more robust eval |
| **LVBench** | 2025 | Long video understanding (TV series, sports, surveillance) | ~150 long videos | Long-video RAG evaluation |
| **EvalVerse** | 2026 | Pipeline-aware + expert-calibrated; treats evaluation as scientific problem | ~1000 videos | Pipeline-stage evaluation |
| **OVO-Bench** | CVPR 2025 | Online video understanding (real-world streaming scenarios) | ~500 videos | Real-time/fast inference testing |

### Video Quality Benchmarks (for VQA model evaluation)

| Dataset | Description | Frames | Licenses |
|---------|-------------|--------|----------|
| **KoNViD-1k** | 1,200 videos, diverse content, human MOS | 1200 | Research |
| **LSVQ** (YouTube-UGC) | 39k videos, in-the-wild quality | 39k | Research |
| **NTIRE** (various years) | Compression/transmission artifacts | varies | Challenge |
| **Tencent Video Dataset** | Streaming/codec distortions | varies | Research |

### For this project:
- **Video-MME** is the closest match to the project's use case (Q&A + RAG over video)
- **EvalVerse** is most interesting because it's pipeline-aware — could validate individual pipeline stages
- Both are too large (GBs) to check into git, but could be referenced as external download targets

---

## 3. Automated Pipeline Output Accuracy Evaluation

### What to measure

The video-analysis pipeline produces: transcriptions, scene timestamps, detected objects, OCR text, scene descriptions, action labels, embeddings. Each can be systematically evaluated:

| Pipeline Output | Metric | Ground Truth Source |
|----------------|--------|-------------------|
| **Transcription** | WER (Word Error Rate) | Manual transcript or benchmark subset |
| **Scene boundaries** | Precision/Recall vs human-annotated splits | Annotated video (e.g., MovieNet, BBC) |
| **Object detection** | mAP (mean Average Precision) | COCO-style annotations |
| **OCR** | CER / F1 | Known text overlay positions |
| **Scene description** | CLIP-score, BLEU, BERTScore | Human-written captions |
| **Action recognition** | Top-1/Top-5 accuracy | Kinetics class labels |
| **RAG retrieval** | MRR, Hit@k, NDCG | Relevance judgments |
| **Chat Q&A** | GPT-4-as-judge, BERTScore | Ground-truth Q&A pairs |

### Implementation approach:

```python
# Pseudo-code for automated eval pipeline
class PipelineEvaluator:
    def __init__(self, benchmark_videos: list[Path], ground_truth: dict):
        self.videos = benchmark_videos
        self.gt = ground_truth

    def evaluate_transcription(self, pipeline) -> dict:
        results = {}
        for video in self.videos:
            transcript = pipeline.transcribe(video)
            gt = self.gt[video]["transcript"]
            results[video] = {
                "wer": compute_wer(transcript, gt),
                "cer": compute_cer(transcript, gt),
            }
        return results

    def evaluate_scene_detection(self, pipeline) -> dict:
        # Precision/Recall of scene boundaries within tolerance
        ...

    def evaluate_rag_retrieval(self, pipeline, questions: list[str]) -> dict:
        # MRR, Hit@k for known correct documents
        ...

    def full_regression(self, pipeline) -> dict:
        return {
            "transcription": self.evaluate_transcription(pipeline),
            "scene_detection": self.evaluate_scene_detection(pipeline),
            "rag": self.evaluate_rag_retrieval(pipeline),
            "chat": self.evaluate_chat(pipeline),
        }
```

### Qualification framework (golden test videos):

Create a small curated set of test videos (~3-5, each 30-60 seconds) with known ground truth:
1. **Simple interview** (one speaker, static background) — test transcription + diarization
2. **Action scene** (multiple objects, motion) — test YOLO + scene detection + X-CLIP
3. **Text-heavy** (slides/signs) — test OCR
4. **Mixed audio** (music + speech) — test transcription robustness
5. **Compressed/artifact** — test quality gate

Store small videos (<5 MB each) in `tests/data/`; larger ones as external references.

---

## 4. Testing Frameworks for Video Processing Pipelines

### Current state
- 1 test file (`tests/test_basic.py`, ~1330 lines)
- 85+ tests covering: config, data models, pipeline imports, sprite sheet generation, URL parsing, RAG imports, health endpoints
- No `conftest.py`, no fixtures, no pytest markers, no `pytest.ini`
- No GPU-dependent tests (all runnable on CPU)
- All tests use real FFmpeg for synthetic test video generation (testsrc filter)

### What's missing

#### a) Synthetic/Deterministic test video generation
Replace FFmpeg `testsrc` heavy calls with lightweight alternatives:

```python
import cv2
import numpy as np

def make_test_video(
    path: Path,
    duration: float = 5.0,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Create a deterministic test video with known frame content."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(int(duration * fps)):
        # Deterministic content: gradient + timestamp
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i:04d}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return path
```

Benefits: fully deterministic, no FFmpeg dependency, fast, known frame content for OCR/object verification.

#### b) Mock model inference
```python
@pytest.fixture
def mock_pipeline(monkeypatch):
    """Replace model inference calls with deterministic stubs."""
    def mock_yolo(frame):
        return [{"label": "person", "confidence": 0.95, "bbox": [0,0,100,200]}]

    monkeypatch.setattr("video_analysis.pipeline.VideoPipeline._infer_yolo", mock_yolo)
    return VideoPipeline(Config(data_dir="/tmp/test"))
```

#### c) Fixture-based conftest.py
```python
# tests/conftest.py
import pytest

@pytest.fixture
def temp_config():
    """Provide a clean Config with temp data dir and auto-cleanup."""
    import tempfile, shutil
    from video_analysis.config import Config

    tmp = Path(tempfile.mkdtemp(prefix="va_test_"))
    cfg = Config(data_dir=tmp)
    yield cfg
    shutil.rmtree(tmp, ignore_errors=True)

@pytest.fixture
def test_video_path(tmp_path):
    """Synthetic deterministic test video."""
    path = tmp_path / "test.mp4"
    make_test_video(path)
    return path
```

Saves ~100 lines of repeated cleanup boilerplate.

#### d) Snapshot/golden file testing
- **pytest-regtest** (https://pypi.org/project/pytest-regtest/): register expected output, re-run to diff
- **pytest-approvaltests**: multi-format approval tests
- **syrupy** (https://github.com/tophat/syrupy): pytest-native snapshot testing

Useful for: verifying sprite sheet generation produces identical images, RAG returns expected chunks, chat responses are deterministic given fixed inputs.

#### e) Deterministic time for temporal tests
Use `freezegun` or `time-machine` to freeze `time.time()` during TV-RAG temporal decay tests.

### Recommended additions

| Tool | Purpose | Install |
|------|---------|---------|
| `pytest-xdist` | Parallel test execution (`-n auto`) | `pytest-xdist` |
| `pytest-timeout` | Kill hung tests | `pytest-timeout` |
| `pytest-randomly` | Randomize test order to detect coupling | `pytest-randomly` |
| `syrupy` | Snapshot testing for deterministic outputs | `syrupy` |
| `pytest-benchmark` | Benchmark fixtures | `pytest-benchmark` |
| `pytest-repeat` | Flaky test detection | `pytest-repeat` |

### Recommended pytest.ini

```ini
# pytest.ini (or pyproject.toml [tool.pytest.ini_options])
[pytest]
testpaths = tests
timeout = 120
timeout_method = thread
markers =
    gpu: marks tests that require CUDA GPU (deselect with '-m "not gpu"')
    slow: marks slow tests (deselect with '-m "not slow"')
    integration: marks integration tests (not unit tests)
    vqa: video quality assessment tests
filterwarnings =
    ignore::DeprecationWarning
addopts = -v --strict-markers
```

---

## 5. Performance Benchmarking for Pipeline Stages

### Current state
- README has a static performance table (times for a 10-minute video on RTX 4070)
- No automated benchmarking

### Proposed approach

Use `pytest-benchmark` for automated, tracked performance tests:

```python
# tests/benchmark_pipeline.py
import pytest

@pytest.mark.benchmark
@pytest.mark.slow
def test_pipeline_throughput(benchmark, temp_config, test_video_path):
    """Benchmark full pipeline on a fixed synthetic video."""
    pipeline = VideoPipeline(temp_config)

    result = benchmark(pipeline.process_video, test_video_path)

    # Benchmark reports min/max/mean/std/ops
    # Can set thresholds
    assert result is not None
```

### Key metrics to benchmark per stage

| Stage | Metric | Instrumentation |
|-------|--------|----------------|
| Audio extraction | wall time, CPU% | `time` decorator |
| Transcription | wall time, realtime factor | compare duration vs wall |
| Scene detection | wall time, scenes/sec | FFmpeg/PySceneDetect |
| Frame extraction | wall time, frames/sec | count output frames |
| YOLO detection | wall time, frames/sec, VRAM | nvidia-smi polling |
| CLIP description | wall time, frames/sec | count described frames |
| OCR | wall time, frames/sec | paddleocr throughput |
| Sprite sheet | wall time | - |
| RAG indexing | wall time, chunks/sec | chunks indexed |
| Cross-encoder | wall time, queries/sec | throughput |

### Monitoring during benchmark

```python
import subprocess, json, time

class GPUProfiler:
    """Context manager that polls nvidia-smi during a benchmark run."""
    def __enter__(self):
        self.readings = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits", "-l", "1"],
            stdout=subprocess.PIPE, text=True
        )
        return self

    def __exit__(self, *args):
        self.proc.terminate()
        for line in self.proc.stdout or []:
            parts = line.strip().split(", ")
            if len(parts) == 3:
                self.readings.append({
                    "gpu_util": float(parts[0]),
                    "mem_used_mb": float(parts[1]),
                })

    @property
    def peak_vram_mb(self) -> float:
        return max(r["mem_used_mb"] for r in self.readings) if self.readings else 0

    @property
    def avg_gpu_util(self) -> float:
        return sum(r["gpu_util"] for r in self.readings) / len(self.readings) if self.readings else 0
```

### Historical tracking
Store benchmark results in `benchmarks/` directory (JSON files). Use pytest-benchmark's `--benchmark-save` + `--benchmark-compare` for regression detection.

---

## 6. Integration Test Infrastructure for GPU-Dependent Tests

### Challenge
GPU tests need `nvidia-smi` / CUDA available. The project is designed for CUDA acceleration but tests currently run CPU-only.

### Strategy

#### Mark-based conditional skipping

```python
# tests/conftest.py
import shutil
import pytest

def has_cuda() -> bool:
    """Check if CUDA GPU is available."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def has_nvidia_gpu() -> bool:
    """Check for any NVIDIA GPU (even without torch)."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and int(result.stdout.strip()) > 0
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return False

skip_if_no_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
require_gpu = pytest.mark.skipif(not has_nvidia_gpu(), reason="No NVIDIA GPU found")
```

#### Usage in tests

```python
# tests/test_gpu.py
@pytest.mark.gpu
@skip_if_no_cuda
def test_whisper_cuda_inference(temp_config):
    """Test that whisper can run on GPU."""
    from video_analysis.pipeline import VideoPipeline
    pipeline = VideoPipeline(temp_config)
    assert pipeline.whisper_device == "cuda"
    # Run a short transcription
    ...

@pytest.mark.gpu
@require_gpu
@skip_if_no_cuda
def test_model_vram_management(temp_config):
    """Test that models load/unload within 12GB budget."""
    pipeline = VideoPipeline(temp_config)
    with GPUProfiler() as prof:
        pipeline.process_video(test_video_path)
    assert prof.peak_vram_mb <= 12000, f"VRAM exceeded: {prof.peak_vram_mb} MB"
```

#### CI pipeline design

```
CI (CPU-only, every PR):
  pytest tests/ -m "not gpu" -v --timeout=120 -x

Scheduled / Release (GPU runner):
  pytest tests/ -v --timeout=300 -x
  pytest tests/ -m "gpu" -v --timeout=600
```

- Self-hosted GPU runner (the same RTX 4070 machine) tagged with `gpu` label
- CPU CI runs on GitHub Actions or similar free tier
- GPU CI runs periodically (nightly) or on releases

#### Environment variable control

```python
@pytest.fixture(autouse=True)
def _skip_gpu_on_env(request):
    """Skip GPU tests if SKIP_GPU_TESTS=1 is set."""
    if request.node.get_closest_marker("gpu") and os.environ.get("SKIP_GPU_TESTS"):
        pytest.skip("GPU tests disabled via SKIP_GPU_TESTS")
```

---

## 7. Video Deduplication / Near-Duplicate Detection

### Available tools

| Tool | Approach | License | Notes |
|------|----------|---------|-------|
| **videohash** (akamhy) | Perceptual video hashing → 64-bit hash | MIT | `pip install videohash`; 377★, active. Frames → imagehash → compare. Fastest option. |
| **videohash2** | Fork of videohash, similar API | MIT | `pip install videohash2` |
| **Frame-level CLIP/BGE-VL embeddings** | Existing BGE-VL embeddings per frame → FAISS index | MIT | Already part of the stack! Embedding-based dedup. |

### Best approach for this project

The project already has BGE-VL embeddings. Near-duplicate detection is a natural extension:

```python
# video_analysis/dedup.py - Proposed new module
import numpy as np
from pathlib import Path
from typing import List, Tuple

class VideoDedup:
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold

    def find_duplicates(self, video_indexes: List[VideoIndex]) -> List[Tuple[str, str, float]]:
        """Find near-duplicate videos using existing BGE-VL embeddings."""
        duplicates = []
        # Compare scene-level embedding clusters across videos
        for i, vi in enumerate(video_indexes):
            for j, vj in enumerate(video_indexes[i+1:]):
                score = self._compute_similarity(vi, vj)
                if score >= self.threshold:
                    duplicates.append((vi.video_id, vj.video_id, score))
        return duplicates

    def _compute_similarity(self, a: VideoIndex, b: VideoIndex) -> float:
        """Cosine similarity of aggregated scene embeddings."""
        # Aggregate per-scene embeddings, compute max similarity
        ...
```

For lightweight use (no ML), `videohash` is a 1-liner:

```python
from videohash import VideoHash
hash1 = VideoHash("video1.mp4")  # 64-bit hash
hash2 = VideoHash("video2.mp4")
similarity = 1 - (hash1 - hash2) / 64  # 1.0 = identical, 0.0 = completely different
```

### Use cases
- **Library dedup**: detect when user tries to re-import the same video
- **Batch optimization**: skip videos already processed (by hash)
- **Smart grouping**: group near-duplicates for batch comparison

---

## 8. Data Versioning for Video Datasets

### Options

| Tool | Description | Video-Specific |
|------|-------------|----------------|
| **DVC** (Data Version Control) | Git-like versioning for data; stores pointers in git, data in remote storage | Yes — dvc.annotate, dvc.ls for large files |
| **DVC + S3/GCS** | Remote cache for video files; `dvc push/pull` | Common ML pipeline pattern |
| **DVC Studio** | Web UI for dataset comparison | Yes |
| **Git LFS** | Git Large File Storage | Simple but limited; no pipeline tracking |
| **LFS + DVC** | Hybrid: LFS for raw videos, DVC for derived artifacts | Complex |

### Why DVC fits

DVC is designed for ML pipelines with large data. It tracks:
- Raw input videos (versioned, stored remotely)
- Intermediate artifacts (extracted frames, audio)
- Pipeline stages and their dependencies
- Metric changes across pipeline versions

### Minimal setup

```bash
# Initialize DVC
cd video-analysis
dvc init

# Track raw video datasets
dvc add data/videos/
dvc add data/benchmark_videos/
git add data/videos.dvc data/benchmark_videos.dvc .dvc/config
git commit -m "track video datasets with DVC"

# Define pipeline stages
dvc run -n extract_frames \
  -p frames_per_scene,scene_detector \
  -d data/videos/raw.mp4 \
  -o data/frames/ \
  python -m video_analysis.pipeline --extract-frames data/videos/raw.mp4

# Show pipeline DAG
dvc dag
```

### Integration with this project

```yaml
# dvc.yaml - Proposed pipeline definition
stages:
  transcribe:
    cmd: python -m video_analysis.pipeline --transcribe ${video}
    deps:
      - ${video}
    outs:
      - data/audio/${video.name}.wav
      - data/transcripts/${video.name}.json
    metrics:
      - metrics/${video.name}_transcript.json:
          cache: false

  analyze:
    cmd: python -m video_analysis.pipeline --full ${video}
    deps:
      - ${video}
      - data/transcripts/${video.name}.json
    outs:
      - data/index/${video.name}.chroma
      - data/sprites/${video.name}.jpg
    metrics:
      - metrics/${video.name}_analysis.json:
          cache: false
```

### Caveats
- Video files are large; DVC works best with a remote cache (S3, GCS, or NFS)
- DVC adds complexity — useful once datasets exceed ~10 GB or have multiple versions
- For simpler cases, a `data/` symlink or `rsync` script suffices

---

## 9. Error Handling & Recovery in Long-Running Pipelines

### Current state
- Graceful SIGTERM/SIGINT shutdown (saves partial results)
- Each stage has try/except blocks
- No checkpoint/resume, no structured retry logic

### Proposed improvements

#### a) Checkpoint-based resume

```python
# video_analysis/checkpoint.py - Proposed new module
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class PipelineCheckpoint:
    video_id: str
    completed_stages: list[str]  # ["audio", "transcribe", "scenes", "objects", ...]
    current_stage: str
    partial_video_index: Optional[dict] = None

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> Optional["PipelineCheckpoint"]:
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return None

class ResumablePipeline:
    STAGES = ["extract_audio", "transcribe", "detect_scenes",
              "extract_frames", "detect_objects", "describe_scenes",
              "extract_ocr", "diarize", "generate_sprite", "index_rag"]

    def process_video(self, video_path: Path) -> VideoIndex:
        ckpt_path = self._checkpoint_path(video_path)
        checkpoint = PipelineCheckpoint.load(ckpt_path)

        start_stage = checkpoint.current_stage if checkpoint else self.STAGES[0]
        partial_index = checkpoint.partial_video_index if checkpoint else None

        for stage in self.STAGES[self.STAGES.index(start_stage):]:
            try:
                result = self._run_stage(stage, video_path, partial_index)
                self._save_checkpoint(ckpt_path, stage, result)
                partial_index = result
            except Exception as e:
                logger.error(f"Stage {stage} failed: {e}")
                # Partial results saved; can resume next run
                raise
```

#### b) Exponential backoff retry for transient failures

```python
import time
from functools import wraps

def retry(max_retries=3, base_delay=1.0, backoff=2.0, exceptions=(Exception,)):  # noqa: B008
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff ** attempt)
                        logger.warning(f"{func.__name__} failed (attempt {attempt+1}), "
                                       f"retrying in {delay:.1f}s: {e}")
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

# Usage in pipeline
@retry(max_retries=2, exceptions=(ConnectionError, TimeoutError))
def _transcribe(self, audio_path):
    ...
```

#### c) Graceful degradation for optional stages

```python
def _run_with_fallback(self, stage_name: str, primary_fn, fallback_fn):
    """Run a stage; if it fails, log and run fallback or skip."""
    try:
        return primary_fn()
    except (ImportError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.warning(f"{stage_name} failed: {e}")
        if fallback_fn:
            logger.info(f"  → falling back to {fallback_fn.__name__}")
            return fallback_fn()
        logger.info(f"  → skipping {stage_name}")
        return None
```

#### d) Structured logging for post-mortem analysis

```python
# Use structured logging with correlation IDs
import structlog
logger = structlog.get_logger()

@pipeline_timing
def process_video(self, video_path):
    log = logger.bind(video_id=video_path.stem, pipeline_run=uuid4().hex)
    log.info("pipeline_start", file_size=video_path.stat().st_size)
    ...
    log.info("pipeline_complete", duration=elapsed, stages_completed=len(stages))
```

#### e) Circuit breaker for persistent hardware failures

```python
class ModelCircuitBreaker:
    """Prevent repeated OOM crashes on the same GPU."""
    def __init__(self, max_failures=3, reset_after=300):
        self.failures = {}
        self.max_failures = max_failures
        self.reset_after = reset_after
        self.last_failure = {}

    def try_call(self, model_name, fn):
        now = time.time()
        if model_name in self.failures and self.failures[model_name] >= self.max_failures:
            if now - self.last_failure.get(model_name, 0) < self.reset_after:
                raise RuntimeError(f"Circuit breaker open for {model_name}")
            self.failures[model_name] = 0  # Reset after cooldown

        try:
            return fn()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            self.failures[model_name] = self.failures.get(model_name, 0) + 1
            self.last_failure[model_name] = now
            raise
```

### When to apply each

| Scenario | Strategy |
|----------|----------|
| Network timeout (yt-dlp) | Exponential backoff retry |
| GPU OOM | Circuit breaker + sequential unloading |
| ImportError (optional dep) | Graceful degradation + skip |
| User kills process (SIGTERM) | Checkpoint save (already done) |
| FFmpeg corruption on input | Validate video before processing |
| ChromaDB connection failure | Retry with backoff |

---

## 10. Priority Recommendations

### Tier 1 — Quick wins (implementation effort: hours)

1. **Add `conftest.py`** with reusable fixtures (`temp_config`, `test_video_path`) — removes ~100 lines of cleanup boilerplate from current tests
2. **Add `pytest.ini`** with markers (`gpu`, `slow`, `integration`) and sensible defaults
3. **Synthetic test video fixture** via OpenCV (deterministic, no FFmpeg dependency)
4. **`videohash` integration** for library deduplication (single `pip install`, 2-3 new functions)

### Tier 2 — Medium effort (days)

5. **`GPUProfiler` context manager** for VRAM tracking in tests
6. **Pytest `@skip_if_no_cuda` / `@require_gpu` markers** — enables GPU test infrastructure
7. **Checkpoint/resume system** for pipeline (survives crashes mid-processing)
8. **Retry decorator + circuit breaker** for transient failures
9. **`pytest-benchmark` integration** with per-stage performance tracking
10. **Snapshot tests for deterministic outputs** (sprite sheet images, metadata JSON)

### Tier 3 — Long-term (weeks)

11. **Full automated pipeline eval** with curated benchmark videos and ground truth
12. **DVC pipeline definition** for data versioning (when dataset grows beyond ~10 GB)
13. **VQA model integration** (RQ-VQA or VQAThinker) for quality gating
14. **CI pipeline with GPU runner** for nightly integration tests
15. **Video-MME or LVBench integration** as evaluation suite for MLLM/chat accuracy

### Packages to add

```toml
# pyproject.toml additions

# Testing
pytest>=8.0
pytest-xdist>=3.6
pytest-timeout>=2.3
pytest-randomly>=3.15
pytest-benchmark>=4.0
pytest-repeat>=0.9
syrupy>=4.6

# Video deduplication
videohash>=3.0

# GPU profiling (built-in, no dep needed)
# Uses nvidia-smi in subprocess

# Optional: VQA
# rqvqa (if published on PyPI)
# videohash2>=1.0

# Optional: data versioning
# dvc>=3.0
```
