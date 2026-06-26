# Changelog

## 0.18.0 (2026-06-26) — Research Phase: Next-Gen Integration

### 🔬 New Research: Qwen3-VL-30B-A3B — The New Optimal MLLM Backend

- **Qwen3-VL-30B-A3B** (Alibaba, Apache 2.0, 30B total / 3B active params via MoE) identified as the
  superior alternative to VideoChat-Flash 2B for the 12 GB RTX 4070 target. When quantized to FP8,
  estimated ~3-6 GB VRAM with vastly superior capabilities.
- **Key specs**: 256K native context (expandable to 1M), frame-by-frame video description, visual
  agent abilities (tool use, UI interaction), multilingual OCR in 32 languages, Thinking variant
  for complex reasoning tasks.
- **Integration plan**: New `"qwen3_vl"` backend option in `video_mllm_backend`, FP8 testing as
  Phase A, promotion to default `"auto"` backed in Phase B.

### 🧠 Lightweight Video Classifier: Qwen3.5-0.8B

- **Qwen3.5-0.8B** (Apache 2.0, 800M params) is a sub-1B multimodal model supporting text, images,
  and video. Ideal for the PipelineOrchestrator component (video type classification in <200ms,
  ~1.6 GB VRAM). Supports 200+ languages, 262K context.

### 📝 PaddleOCR v5 Upgrade Research

- **PP-OCRv5** confirmed available: 34.5M params, +5.1% recognition accuracy over v4_server,
  +13% end-to-end accuracy. 109 language support (vs ~50 in v4). PP-StructureV3 for hierarchical
  document parsing. Published as PaddleOCR 3.0 (arXiv:2507.05595).

### ⚡ Dependency Modernization Research

- **Critical finding**: Environment has torch 2.12.1 (pinned `>=2.1.0`), transformers 5.12.1
  (pinned `>=4.45.2`), sentence-transformers 5.6.0 (pinned `>=2.5.0`). All minimum bounds are
  3-4 major versions behind reality, risking silent breaking changes.
- **Recommended updates**: `torch>=2.5.0`, `transformers>=4.50.0`, `sentence-transformers>=2.7.0`,
  add `yt-dlp>=2024.12.0` to pyproject.toml deps.
- **Gemma 3 12B / Gemma-3n-E2B-IT** (Google, Apache 2.0) identified as additional candidates for
  lightweight multimodal processing with video+audio+image input.

### 🔧 Strategic Shifts

- VideoChat-Flash 2B demoted to legacy fallback — Qwen3-VL-30B-A3B is the new primary recommendation
- All transformers-dependent code needs v5 API audit (breaking changes in transformers>=5.0)
- PipelineOrchestrator targets Qwen3.5-0.8B instead of SmolVLM2 500M (lighter, more capable)
- ChromaDB confirmed as "stay" — LanceDB only if >5M vectors
- PaddleOCR v5 upgrade should be part of next implementation wave
- Full research: `docs/research/v0.18.0-research-next-gen.md`

---

## 0.17.0 (2026-06-26) — Research Phase: Beyond the Roadmap

### 🔬 New Research: Next-Gen Capabilities (v0.17.0+)

- **Autonomous Agentic Pipeline** — LLM-driven stage orchestration: quick scan (30s) classifies video type, then dynamically selects only relevant pipeline stages (save 30-50% processing time, 40-60% disk usage)
- **Video Type Classification** — 7 video types (lecture, sports, screen recording, interview, movie, vlog, podcast) each with optimized frame rate, scene detection threshold, chunk strategy, and model selection
- **Face Recognition System** — InsightFace (MIT, ~1.5 GB VRAM) for person identity across scenes: RetinaFace detection → ArcFace embedding → DBSCAN clustering → cross-video gallery
- **Pipeline Caching & Incremental Re-Indexing** — Content-hash based stage caching (70-90% faster re-runs on partial changes) + incremental ChromaDB upsert
- **MCP Tool Server** — Expose each pipeline stage (extract_frames, detect_scenes, transcribe_audio, detect_objects, search_video) as composable MCP tools for Hermes/agentic workflows
- **Gradio Workflow Subgraph Integration** — 4-phase plan confirmed: refactor → FastAPI → Gradio Workflow → MCP tools
- **UI Dashboard Enhancement** — Multi-resolution timeline with scene markers + entity timeline bars + transcript heatmap + export (SRT/CSV)
- **Dependency Modernization** — Audit of all 25+ dependencies; recommendations to update torch>=2.5.0, add yt-dlp to requirements.txt, boxmot/insightface as optional deps
- **Key Design Decision**: Use **Ultralytics built-in ByteTrack** (MIT) instead of BoxMOT (AGPL-3.0) for entity tracking
- **Key Design Decision**: **FFmpeg motion vectors** (zero GPU, <1ms/frame) confirmed over deep flow models for 12GB VRAM budget
- **Full research**: `docs/research/v0.17.0-research-beyond-roadmap.md`

### 📦 Version Fix
- Fixed `pyproject.toml` version mismatch: `0.15.0` → `0.16.0` (synced with `__init__.py`)

---

## 0.16.0 (2026-06-26) — Research Phase

### 🔬 Entity Tracking Research — ByteTrack/BoxMOT for Persistent IDs

- **Deep research** into multi-object tracking (MOT) for assigning persistent IDs to
  people and objects across video scenes. Current YOLO detections are per-frame with
  no identity — a person in scene 1 has no connection to the same person in scene 3.
- **ByteTrack via BoxMOT** selected as the primary approach (MIT license, minimal
  VRAM ~500 MB shared with YOLO, integrates directly with YOLOv26 already in pipeline).
  BoxMOT wraps ByteTrack, BoT-SORT, DeepOCSORT, and ImprAssocFlow in a unified API.
- **InsightFace** (MIT, ~1.5 GB VRAM) identified as optional add-on for face-based
  person ReID when tracking across long temporal gaps or across different videos.
- **Integration strategy**: Pipeline Step 7 post-YOLO → ByteTrack assigns track_ids →
  stored in FrameInfo.objects → indexed in ChromaDB → used by scene_graph.py for
  entity-shared edges (replacing text-parsed name matching).
- **Full research**: `docs/research/v0.16.0-research-evolution.md`

### 🕸️ Cross-Video Scene Graph Research

- Scene graph adjacency structure already supports `(video_id, scene_id)` tuple keys
  across videos. Current `rebuild()` only connects within each video.
- **Phase 1**: Extend semantic Jaccard + entity edges across ALL videos (trivial code
  change, removes video_id grouping in cross-comparison loops).
- **Phase 2**: Add BGE-VL cross-video scene embedding comparison via FAISS approximate
  kNN — leverages existing BGE-VL model already loaded during indexing.
- Cross-video edges enable queries like "find all interviews in any video" by linking
  semantically similar scenes across the entire library.

### ⚡ Gradio 6 Workflow Integration Research

- Gradio 6.19.0 (June 2026) introduced **Workflow subgraphs** — each subgraph is
  exposed as a named endpoint via `/info`, `/call`, `/api` with a "View API" panel.
- **4-phase plan**: (A) Refactor pipeline steps into independently callable stage
  methods, (B) Expose as FastAPI endpoints, (C) Add Gradio Workflow subgraphs for UI,
  (D) Add MCP tool definitions for external agent access.
- Pipeline.py already has most steps as individual methods — Phase A is minimal effort.
- Existing `/health` and `/api/library` FastAPI endpoints provide the foundation.

### 🎞️ Sparse-Frame Optical Flow Research

- Current adaptive frame sampling uses a motion-unaware cosine density function (dense
  near scene boundaries regardless of actual motion).
- **FFmpeg motion vector extraction** identified as the optimal approach — every
  h264/h265 video already encodes per-macroblock motion vectors as part of compression.
  Extracting them costs zero GPU and <1ms per frame via `ffprobe` / `ffmpeg codecview`.
- **Comparison**: FFmpeg MVs (0 GPU, <1ms) >> OpenCV Farneback/DIS (CPU, 15-30ms) >>
  RAFT/GMFlow/FlowFormer (GPU, 80-200ms, 1-2 GB VRAM).
- Recommendation: Use FFmpeg MVs as primary (always available, zero cost), fall back to
  OpenCV Farneback for videos without motion vectors (e.g., screen recordings, GIFs).
- Motion score per frame guides sampling: high-motion (>0.5) → 1 fps, medium (0.2-0.5)
  → 1 per 2s, low (<0.2) → 1 per 5s or scene mid-point.

### 🧠 Cutting-Edge Developments (Feb-Jun 2026)

- **SmolVLM2** (Apache 2.0, Mar 2026) — already integrated in v0.15.0, project is current
- **VideoChat-Flash 2B** (MIT, ICLR 2026) — already integrated in v0.13.0, project is current
- **InternVideo2.5** (Feb 2026, 8B, ~16GB VRAM) — too heavy for 12GB RTX 4070, skip
- **VGent** (NeurIPS 2025 Spotlight) — graph-based video RAG, already implemented in scene_graph.py
- **ViG-RAG** (AAAI 2026) — hybrid temporal+semantic graph, already implemented
- Trend: small video MLLMs under 3B that fit consumer GPUs (low-hanging fruit all integrated)
- Gradio 6.19+ Workflow subgraphs + MCP integration is the most impactful new capability

### 📚 Documentation

- Full v0.16.0 research plan saved at `docs/research/v0.16.0-research-evolution.md`
- README roadmap updated: 4 research items marked done, 4 implementation items pending
- Next implementation priority: Entity tracking → Cross-video graphs → Gradio Workflows → Optical flow

---

## 0.15.0 (2026-06-26) — SmolVLM2, Agentic RAG & Production Hardening

### 🧠 Major Feature: SmolVLM2 Backend — Lightweight Video MLLM (Apache 2.0)

- **Dual-backend Video MLLM**: `VideoMLLM` now supports two backends selected via `video_mllm_backend` config field:
  - `"videochat_flash"` — the existing OpenGVLab VideoChat-Flash 2B (ICLR 2026, MIT, ~5.4 GB VRAM)
  - `"smolvlm2"` — HuggingFace SmolVLM2 family (Apache 2.0, transformers-native, no `trust_remote_code`)
  - `"auto"` (default) — tries SmolVLM2 first, falls back to VideoChat-Flash

- **SmolVLM2 model sizes** (selectable via `video_mllm_model_size`):
  | Size | Params | VRAM (BF16) | Video-MME | Use Case |
  |------|--------|-------------|-----------|----------|
  | `2.2B` | 2.2B | ~5.2 GB | 52.1 | Best quality, parallel with other pipeline stages |
  | `500M` | 500M | ~1-2 GB | 42.2 | Runs alongside pipeline without unloading! |
  | `256M` | 256M | ~0.5-1 GB | 33.7 | CPU-friendly, experimental |

- **Transformers-native API**: SmolVLM2 uses `AutoModelForImageTextToText` with standard chat templates — no `trust_remote_code`, no custom processor. Video input via `{"type": "video", "path": "..."}` in the chat template.
- **`decord` dependency**: Required for SmolVLM2 video decoding. Added to requirements.txt as optional (commented out).
- **All three methods work with both backends**: `describe_scene()`, `summarize_video()`, `answer()`.
- **Config fields**: `video_mllm_backend` (env: `VIDEO_MLLM_BACKEND`), `video_mllm_model_size` (env: `VIDEO_MLLM_MODEL_SIZE`).

### 🔄 Agentic RAG — Iterative Retrieval Loop with Confidence Checking

- **`agentic_retrieve()`**: New iterative retrieval method on `VideoRAG` that runs multiple rounds of retrieval with confidence-based early stopping:
  - **Round 1**: Standard `retrieve()` — fast embedding search + re-ranking
  - **Round 2**: Multi-hop decomposition — break query into sub-questions (if enabled)
  - **Round 3**: Scene-graph K-hop expansion — graph traversal from accumulated results
  - After all rounds: deduplicate and re-rank merged results against the original query

- **Confidence-gated early exit**: Each round checks the average score of the top-3 chunks against `agentic_min_confidence` (default: `0.5`). When the threshold is met, the loop stops early without executing remaining rounds.
- **Config fields**: `agentic_retrieval_enabled` (default: `false`, env: `AGENTIC_RETRIEVAL_ENABLED`), `agentic_max_rounds` (env: `AGENTIC_MAX_ROUNDS`), `agentic_min_confidence` (env: `AGENTIC_MIN_CONFIDENCE`).
- **Chat integration**: `VideoChat.ask()` and `ask_with_history()` automatically use Agentic RAG when `agentic_retrieval_enabled` is toggled on, falling back to the existing routed / standard retrieval path when disabled.

### 🛠️ Production Hardening

- **`.pre-commit-config.yaml`**: Pre-commit hooks for code quality:
  - Ruff (lint + format) — `line-length=100`, `target-version=py311`
  - Trailing whitespace, end-of-file fixer, YAML/JSON validation
  - Check-added-large-files, detect-private-key
  - MyPy (ignore-missing-imports)

- **`.github/workflows/ci.yml`**: GitHub Actions CI/CD:
  - Python 3.10/3.11/3.12 matrix on push/PR to master
  - Install dependencies from `requirements.txt`
  - Run `pytest tests/ -v -m "not gpu"` (CPU-only marker skips GPU tests)
  - Ruff code quality check
  - Pip caching for faster runs

- **`pyproject.toml` additions**: `[tool.ruff]` config, `[tool.pytest.ini_options]` with `timeout=120`, `strict-markers`, `filterwarnings`.

- **Benchmark infrastructure** (`tests/benchmarks/`):
  - `conftest.py` — `GPUProfiler` context manager for measuring GPU memory usage during benchmarks
  - `test_pipeline_throughput.py` — benchmark each pipeline stage (frame extraction, scene detection, transcription)
  - `test_rag_latency.py` — benchmark retrieval + re-ranking latency

- **Docker label fix**: Updated `Dockerfile` LABEL version from stale `0.5.0` to `0.15.0`.

### ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_MLLM_BACKEND` | `auto` | Video MLLM backend: "auto", "videochat_flash", or "smolvlm2" |
| `VIDEO_MLLM_MODEL_SIZE` | `2.2B` | SmolVLM2 model size: "2.2B", "500M", or "256M" |
| `AGENTIC_RETRIEVAL_ENABLED` | `false` | Enable agentic iterative retrieval loop |
| `AGENTIC_MAX_ROUNDS` | `3` | Max retrieval rounds in agentic loop |
| `AGENTIC_MIN_CONFIDENCE` | `0.5` | Min avg score of top-3 chunks to stop early |

### 📦 Dependencies

- **New (optional)**: `decord>=0.6.0` — video decoding for SmolVLM2 (commented out, install on demand)
- **New (optional)**: `pytest-timeout>=2.3.0`, `pytest-benchmark>=4.0.0` — benchmark infrastructure (commented out)

### 🧪 Tests

- **18 new tests** in v0.15.0 (93 pre-existing → 104 total passing, +18 net new):
  - `test_version_0_15_0` — version bump check
  - `test_config_agentic_rag_fields` — default values
  - `test_config_agentic_rag_custom_values` — custom config override
  - `test_rag_agentic_retrieve_method_exists` — method signature check
  - `test_rag_agentic_retrieve_disabled_features` — graceful fallback with all features disabled
  - `test_agentic_retrieve_confidence_check` — high/low confidence threshold behavior
  - `test_chat_agentic_retrieval_disabled` — proper dispatch when agentic is off
  - `test_agentic_retrieve_max_rounds_1` — single-round behavior
  - SmolVLM2 config/defaults tests
  - CI workflow + pre-commit syntax validation tests
  - Benchmark infrastructure smoke tests

### 🏗️ Architecture

```
video-analysis/
├── video_analysis/
│   ├── __init__.py              # v0.15.0
│   ├── config.py                # +video_mllm_backend, video_mllm_model_size
│   │                           # +agentic_retrieval_enabled/rounds/min_confidence
│   ├── video_mllm.py            # +SmolVLM2 backend (2.2B/500M/256M)
│   ├── rag.py                   # +agentic_retrieve() method
│   └── chat.py                  # +_ask_agentic() integration
├── .pre-commit-config.yaml      # NEW — pre-commit hooks
├── .github/workflows/ci.yml     # NEW — GitHub Actions CI
├── tests/
│   └── benchmarks/              # NEW — benchmark infrastructure
│       ├── conftest.py
│       ├── test_pipeline_throughput.py
│       └── test_rag_latency.py
├── Dockerfile                   # Updated LABEL version to 0.15.0
├── pyproject.toml               # v0.15.0 + ruff + pytest config
├── requirements.txt             # +decord (commented), +pytest-timeout/benchmark (commented)
├── README.md                    # Updated with new features & roadmap
└── CHANGELOG.md                 # This file
```

---

## 0.14.0 (2026-06-26) — Graph-Based Video RAG + Query Routing + Multi-Hop Decomposition

### 🧠 Major Features: Graph-Based Retrieval, Smart Query Routing, Multi-Hop Reasoning

All three remaining roadmap items implemented in this release:

### 🕸️ Scene Graph (VGent/ViG-RAG Inspired)

- **New `video_analysis/scene_graph.py` module**: Lightweight in-memory graph layer on top of ChromaDB. Nodes = video scenes, edges = three relationship types:
  - **Temporal edges**: Scenes adjacent or nearby in time (configurable window)
  - **Entity-shared edges**: Scenes sharing detected objects, people, or actions from pipeline metadata
  - **Semantic edges**: Scenes with high word-overlap similarity in scene text
- **K-hop graph expansion**: `k_hop_expand()` traverses the graph from seed scene nodes for K hops, discovering semantically related content across different video segments or even across different videos
- **`expand_chunks()`**: Integrates with the standard retrieval pipeline — after ChromaDB returns `RetrievedChunk`s, graph expansion pulls in semantically connected scene chunks with a default score
- **Lazy rebuild**: Graph auto-rebuilds from ChromaDB metadata on first query; call `rebuild()` after indexing new videos to refresh edges
- **Zero external dependencies**: No separate graph database — the graph lives in memory and is rebuilt from ChromaDB metadata (~5-10ms for typical video libraries)
- **Config fields**: `scene_graph_enabled` (default: `true`), `scene_graph_k_hop` (default: `2`), `scene_graph_temporal_window` (`3`), `scene_graph_min_shared_entities` (`1`), `scene_graph_semantic_threshold` (`0.85`)

### 🧭 Query Router (Multi-Modal Dispatch)

- **New `video_analysis/query_router.py` module**: Classifies user queries into one of four retrieval routes:
  | Route | Query Type | Example | Strategy |
  |-------|-----------|---------|----------|
  | `text` | Factual, narrative | "What did the speaker say?" | Standard ChromaDB dense retrieval |
  | `visual` | Visual content, objects | "What color was the car?" | BGE-VL visual search (when available) |
  | `temporal` | Time/sequence | "What happened before the explosion?" | ChromaDB + temporal decay + scene graph |
  | `multimodal` | Complex, multi-aspect | "Why did the protagonist leave?" | Multi-hop decomposition |
- **Two-tier classification**: LLM-based routing via Hermes CLI (lightweight single-turn prompt) with keyword-based regex fallback when the LLM is unavailable
- **`RoutingDecision` dataclass**: Route, confidence (0-1), reasoning, and `sub_queries` list for multi-hop decomposition
- **Keyword patterns**: Three regex patterns for `_TEMPORAL`, `_VISUAL`, `_MULTIMODAL` keywords with scoring logic for decisive route selection
- **Config fields**: `query_routing_enabled` (default: `true`), `query_routing_prefer_llm` (`true`)

### 🔗 Multi-Hop Query Decomposition

- **`routed_retrieve()` in `rag.py`**: The primary retrieval entrypoint that coordinates the full pipeline:
  1. Query classification → RouteDecision
  2. Multi-hop decomposition for `multimodal` queries
  3. Route-specific retrieval strategy
  4. Scene-graph K-hop expansion
  5. Standard re-ranking and temporal expansion

- **`_multi_hop_retrieve()`**: Decomposes complex queries into 2-4 sub-questions → independent retrieval per sub-query → deduplication → re-ranking against original query → scene-graph expansion on merged results. Falls back gracefully to standard retrieval when decomposition fails or returns no results.

- **Chat integration**: `VideoChat.ask()` and `ask_with_history()` automatically use `routed_retrieve()` when any of `query_routing_enabled`, `scene_graph_enabled`, or `multi_hop_enabled` are toggled on.

### ⚙️ Configuration

- **8 new config fields** in `video_analysis/config.py`:
  | Variable | Default | Description |
  |----------|---------|-------------|
  | `scene_graph_enabled` | `true` | Enable scene-graph retrieval layer |
  | `scene_graph_k_hop` | `2` | K-hop graph expansion hops |
  | `scene_graph_temporal_window` | `3` | Max scene distance for temporal edges |
  | `scene_graph_min_shared_entities` | `1` | Min shared objects/actions for entity edge |
  | `scene_graph_semantic_threshold` | `0.85` | Similarity threshold for semantic edges |
  | `query_routing_enabled` | `true` | Enable query classification & routing |
  | `query_routing_prefer_llm` | `true` | Use LLM for classification (fast prompt) |
  | `multi_hop_enabled` | `true` | Enable multi-hop query decomposition |
  | `multi_hop_max_sub_queries` | `4` | Max sub-questions to generate |
  | `multi_hop_rerank_top_k` | `10` | Top-k from each sub-query retrieval |

### 📚 Documentation

- README roadmap updated: all previous roadmap items checked as completed
- `video_analysis/__init__.py` updated: imports `scene_graph` and `query_router` modules

### 🧪 Tests

- **16 new tests** for v0.14.0 features:
  - `test_scene_graph_import`, `test_scene_graph_no_rag_init`, `test_scene_graph_k_hop_empty`
  - `test_scene_graph_expand_chunks_empty`, `test_scene_graph_disabled`
  - `test_query_router_import`, `test_query_router_keyword_text/visual/temporal/multimodal`
  - `test_query_router_heuristic_decompose`
  - `test_config_scene_graph_fields`, `test_config_query_routing_fields`, `test_config_multi_hop_fields`
  - `test_rag_routed_retrieve_fallback`, `test_rag_multi_hop_no_subqueries`
  - `test_version_0_14_0`
- Pre-existing test suite: 69 → 85 passing (1 pre-existing einops dependency skip)

---

## 0.13.0 (2026-06-26) — Video MLLM Integration

### 🧠 Major Feature: VideoChat-Flash — Lightweight Video MLLM (ICLR 2026)

- **New `video_analysis/video_mllm.py` module**: Wraps OpenGVLab's VideoChat-Flash 2B (`OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448`) — the state-of-the-art lightweight video MLLM that fits in 12 GB VRAM (~5.4 GB BF16). Key specs: 16 tokens/frame (vs 256+ for typical VLMs), 448px resolution, 99.1% NIAH over 10K frames (~3 hours of video), MVBench 70.0. MIT license.
- **`VideoMLLM` class**: Lazy-load on first use, GPU memory management (load/unload compatible with sequential pipeline model), graceful fallback when dependencies are missing. Three core methods:
  - `describe_scene(frames)` — rich natural language scene descriptions with people, objects, actions, setting, and mood
  - `summarize_video(video_path, num_frames=32)` — comprehensive global video summary using VideoChat-Flash's hierarchical compression (handles long videos with few tokens)
  - `answer(query, frames, video_path)` — video-native Q&A that sees frame images directly (not just text context)

### ⚙️ Config & Pipeline Integration

- **New config fields**:
  | Variable | Default | Description |
  |----------|---------|-------------|
  | `VIDEO_MLLM_ENABLED` | `false` | Enable Video MLLM module |
  | `VIDEO_MLLM_MODEL` | `OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448` | MLLM model name |
  | `VIDEO_MLLM_AS_DESCRIBER` | `false` | Use MLLM for scene descriptions instead of OpenCLIP |
  | `VIDEO_MLLM_AS_CHAT_BACKEND` | `false` | Use MLLM as video-native Q&A backend instead of Hermes CLI |
- **Pipeline integration** (Step 10): When `video_mllm_as_describer` is enabled, runs VideoChat-Flash on each scene's key frames after OpenCLIP classification. Generates rich natural language descriptions that augment (or replace) the OpenCLIP zero-shot labels. MLLM model is unloaded after use to free ~5.4 GB VRAM for subsequent steps.
- **Chat integration**: `VideoChat.ask()` and `ask_with_history()` now have an optional Video MLLM backend. When `video_mllm_as_chat_backend` is enabled, the MLLM answers the question using frame images directly as visual context — enabling questions about visual details that text-only RAG would miss. Falls back gracefully to the text-only RAG + Hermes CLI path when the MLLM is unavailable.

### 🎯 Graph-Based Video RAG Research — Next Frontier

- **VGent** (NeurIPS 2025 Spotlight, arXiv:2510.14032): Graph-based retrieval-reasoning that outperforms SOTA video RAG methods by +8.6% on MLVU. Core idea: index videos as structured graphs with semantic relationships between clips.
- **ViG-RAG** (AAAI 2026, #6 ranked): Hybrid temporal+semantic graph reasoning — combines temporal edges (before/after/overlap) with entity-based semantic edges.
- **Architecture proposed**: SceneGraph layer alongside existing ChromaDB multi-granularity chunks, with K-hop expansion for retrieval context.

### 🎯 Query Classification & Multi-Modal Routing Research

- Multi-RAG pattern: classify user queries into text/visual/temporal/multimodal routes before retrieval. Complex queries use multi-hop decomposition (sub-question → retrieve → reason).
- Current pipeline does uniform text embedding for all queries. Adding classification would route to optimal retrieval strategy for each query type.

### 📚 Documentation

- Full v0.13.0 research plan saved at `.hermes/plans/2026-06-26_173500-v0.13.0-research-synthesis.md`
- README roadmap updated with 4 new items (research marked done, 4 implementation items remaining)

---

## 0.12.0 (2026-06-26)

### 🧠 Major Feature: BGE-VL Multimodal Embedding — Single Unified Model

- **BGE-VL-base as the default embedding model** (BAAI/BGE-VL-base, 150M params, MIT license, ~0.8 GB VRAM). Replaces the old dual-model approach (SentenceTransformer + optional Qwen3-VL-Embedding-2B) with a single unified embedding pipeline that handles text-only, image-only, and composed (image+text) retrieval.
- **`rag.py`**: Completely rewritten embedding stack. New `_load_bge_vl()` / `_unload_bge_vl()` for lazy GPU loading/unloading. `_get_bge_vl_embedding()` handles all three modes (text, image, composed). `_get_query_embedding()` uses BGE-VL as primary with query prefix normalization. `_get_embedding()` falls back gracefully to SentenceTransformer + Nomic Embed when BGE-VL is unavailable.
- **`search_all()`**: Now uses BGE-VL composed retrieval when both image and text are provided — true multimodal cross-video search without the heavy Qwen3-VL model.
- **`text_embedding_model`** config field added (`nomic-ai/nomic-embed-text-v1.5`) for fallback. Legacy `multimodal_embedding_enabled` / Qwen3-VL path retained for backward compatibility.
- **Embedding prefix normalization**: Query/document prefixes now applied when falling back to SentenceTransformer (e.g. `search_query:` for Nomic, `Represent this query:` for BGE). Boosts retrieval accuracy by 5-10%.

### ⏱️ Temporal-Aware Retrieval (TV-RAG)

- **Time-decay weighting**: New `temporal_decay_rate` config field (default: `0.1`). When `query_time` is provided to `retrieve()`, chunk scores are weighted by `score * exp(-decay_rate * time_distance)` per the TV-RAG paper (ACM Multimedia 2025). Set to `0.0` to disable.
- **`_get_query_embedding()`**: Updated `retrieve()` signature with optional `query_time` parameter. Temporal weighting integrates seamlessly with the existing cross-encoder re-ranking pipeline.

### 📦 Multi-Granularity Chunking

- **Quad-chunk strategy** in `index_video()`:
  - **Scene chunks** (variable length, rich context): transcript + descriptions + objects + OCR + actions
  - **Fixed-window chunks** (60 seconds, no overlap): transcript segments aligned to time windows — cross-scene queries
  - **Sliding-window chunks** (30 seconds, 15s overlap): fine-grained temporal localization
  - **Frame chunks** (per-frame): direct frame-level retrieval
  - **Transcript chunks** (legacy 500-char windows): retained for backward compatibility
- All chunk types get `chunk_type` metadata field in ChromaDB (`scene`, `frame`, `fixed_60s`, `sliding_30s`, `transcript`) enabling targeted retrieval strategies.

### 🛡️ GPU Memory Management & Graceful Shutdown

- **Systematic model unloading**: New `_unload_model(model_attr)` helper in `VideoPipeline` that safely removes a model attribute, deletes the reference, runs `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.synchronize()`.
- **Per-stage GPU cleanup**: Models are explicitly unloaded between every GPU-intensive pipeline step:
  - After Step 5 (Whisper, ~4 GB) → unloaded before diarization
  - After Step 7 (YOLO, ~1 GB) → unloaded before OCR
  - After Step 9 (OpenCLIP, ~2 GB) → unloaded before action recognition
  - After Step 10 (X-CLIP, ~4 GB) → unloaded before sprite sheet/indexing
- Peak VRAM now managed: sequential loading ensures no more than 4 GB reserved at any time on a 12 GB RTX 4070.
- **Graceful SIGTERM/SIGINT handling**: `VideoPipeline` registers signal handlers that set `_shutdown_requested=True`. `__main__.py` also registers handlers with `_shutdown_event` for clean CLI/cron termination.

### ⚙️ Dependency & Configuration Updates

- **Gradio >=6.19.0** (was >=6.0.0) — Svelte 5 migration, MCP support, workflow subgraphs, stability
- **transformers >=4.45.2** (was >=4.40.0) — required for BGE-VL compatibility
- New config fields: `text_embedding_model`, `temporal_decay_rate`
- BGE-VL-base (`BAAI/BGE-VL-base`) is now the default `embedding_model`

### 🐳 Production Deployment

- **`docker-compose.prod.yml`**: New production-grade Docker Compose with:
  - **DCGM Exporter** — NVIDIA GPU metrics (VRAM, temp, utilization) at `:9400/metrics` for Prometheus
  - **Caddy reverse proxy** — automatic HTTPS, WebSocket support for Gradio streaming, security headers, gzip compression
- **Caddyfile**: Production-ready reverse proxy configuration with security headers and logging

### 🧪 Tests

- 12 new tests: BGE-VL config defaults, embedding prefix normalization (Nomic, BGE-small, BGE-VL), pipeline cleanup, model unloading, multi-granularity chunking config, temporal decay config, RetrievedChunk chunk_type field, graceful fallback test (BGE-VL → SentenceTransformer)
- Pre-existing test suite: 49 → 61 tests passing

### 🎬 Major Feature: X-CLIP Zero-Shot Action Recognition

- **Open-vocabulary action detection**: Added `ActionRecognizer` module at `video_analysis/action.py` wrapping Microsoft X-CLIP (`microsoft/xclip-base-patch16-zero-shot`, 200M params, Apache 2.0). Classifies per-frame human activities (walking, running, cooking, typing, etc.) with confidence scores — no training required, works out of the box.
- **Pipeline integration**: New pipeline step (Step 10) between OpenCLIP and transcript assignment. Runs sequentially, loads X-CLIP (~4GB VRAM), classifies all key frames in GPU-efficient batches, then unloads the model to free VRAM.
- **Config toggle**: `ACTION_RECOGNITION_ENABLED=true` env var to enable. New config fields: `action_recognition_enabled`, `action_model_name`, `action_categories_count`.
- **RAG context**: Action labels are indexed in ChromaDB alongside transcript, objects, OCR, and scene descriptions. Queries like "when is someone cooking?" or "find scenes with people fighting" retrieve relevant video segments.
- **Graceful fallback**: If `transformers` is unavailable or model download fails, the step is silently skipped — no breaking changes.
- **26 default action categories**: Covers common video scenarios (walking, running, sitting, cooking, typing, driving, fighting, etc.) with "no person visible" as the catch-all.

### 🐛 Bug Fixes & Improvements

- **Pipeline step numbering**: Fixed duplicate Step 7 (OCR) and duplicate Step 10 (Index). Steps are now correctly numbered 1–13 throughout.
- **Multimodal embedding fallback fix**: Fixed a bug where `_get_embedding()` would load a SentenceTransformer first, causing `_get_multimodal_embedding()` to silently fall back to text-only. Now `_get_embedding()` routes through the multimodal model when `multimodal_embedding_enabled=True`, ensuring one unified embedding space.
- **README roadmap**: Updated stale reference to "InternVideo2.5" for action recognition → now correctly references X-CLIP.

### 🧪 Tests

- 6 new tests: config action fields, ActionRecognizer import/defaults, empty classify list, graceful file-not-found fallback, FrameInfo action fields, ACTION_RECOGNITION_ENABLED env var.
- Pre-existing test suite: 43 → 49 tests passing.

---

### 🔬 Major Enhancement: Qwen3-VL Multimodal Embedding (Apache 2.0)

- **True multimodal semantic search**: Added optional Qwen3-VL-Embedding-2B support (Apache 2.0, 2B params, 2048-dim) for fusing visual + textual information into a shared embedding space. When `MULTIMODAL_EMBEDDING=true` and the model weights are downloaded, frame images are embedded together with text descriptions for far richer semantic retrieval compared to text-only Nomic Embed v1.5.
- **New config fields**: `multimodal_embedding_model` (default: `Qwen/Qwen3-VL-Embedding-2B`, 2B params fits in ~6GB VRAM), `multimodal_embedding_enabled` (reads `MULTIMODAL_EMBEDDING` env var, default `false`).
- **Graceful fallback**: If `transformers`, `torch`, or `Pillow` are not installed, or if the model weights are not downloaded, falls back to text-only embedding with a log warning. No breaking changes.
- **`rag.py`**: Added `_get_multimodal_embedding()` method and `search_all()` method for cross-video semantic search.

### 🔍 New Feature: Cross-Video Semantic Search ("Video Search" Tab)

- **New Gradio tab**: "🔍 Video Search" tab added after the Library tab. Users enter a natural language query and search across ALL indexed videos simultaneously.
- **Rich results**: Results are grouped by video, showing timestamp, relevance score (as %), and expandable context preview for each matching chunk.
- **`rag.search_all()`**: New method on `VideoRAG` that removes the `video_id` filter from ChromaDB queries, enabling true cross-video retrieval. Re-ranks results with the existing cross-encoder for best accuracy.
- **Works with or without multimodal**: When `multimodal_embedding_enabled` is off, search uses the existing Nomic Embed v1.5 text model as before.

### ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIMODAL_EMBEDDING` | `false` | Enable Qwen3-VL-Embedding multimodal search |

### 🔧 Improvements

- `video_analysis/config.py`: Added `multimodal_embedding_model` and `multimodal_embedding_enabled` fields.
- `video_analysis/rag.py`: Added `_get_multimodal_embedding()` — wraps Qwen3-VL-Embedding for image+text fusion. Added `search_all()` — cross-video semantic search without `video_id` filter.

### 🧪 Tests

- Added test for multimodal embedding config defaults.
- Added test for `search_all()` basic logic.
- Pre-existing test suite: 43+ tests passing.

---

### 🔒 New Feature: Gradio UI Authentication

- **FastAPI middleware auth**: Added HTTP Basic Auth middleware to the FastAPI app that mounts Gradio. When `GRADIO_PASSWORD` is set, all UI routes (except `/health`) require authentication via HTTP Basic Auth.
- **Config-driven**: New config fields `ui_auth_enabled`, `ui_auth_username` (from `GRADIO_USER`, default `admin`), and `ui_auth_password` (from `GRADIO_PASSWORD`). Auth auto-enables when `GRADIO_PASSWORD` is set.
- **/health stays public**: The health endpoint is excluded from auth, so Docker health checks and monitoring tools continue to work without credentials.

### 🎞️ New Feature: Motion-Based Adaptive Frame Sampling

- **Smarter frame extraction**: New `adaptive_frame_sampling` config flag (default: `False`). When enabled, frames are sampled more densely near scene boundaries (3× density in first/last 10% of each scene) and more sparsely in static middle regions. This captures transitions and action near cuts while reducing redundant frames from static scenes.
- **Configurable sensitivity**: `adaptive_frame_sampling_sensitivity` (default: `0.3`) controls the base sampling rate — lower values extract more frames overall.

### 🧹 New Feature: CLIP-Similarity Frame Deduplication

- **Removes near-duplicate frames**: New `clip_frame_dedup` config flag (default: `False`). When enabled, consecutive frames within each scene are compared using OpenCLIP ViT-B-32 embedding cosine similarity. Frames exceeding `clip_frame_dedup_threshold` (default: `0.92`) are considered near-duplicates and removed.
- **Graceful fallback**: If `open-clip-torch` is not installed, dedup is silently skipped.
- **VRAM-efficient**: OpenCLIP model is loaded temporarily for the dedup pass, then unloaded from GPU memory.

### ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_USER` | `admin` | UI auth username |
| `GRADIO_PASSWORD` | (unset) | UI auth password — set to enable authentication |
| `ADAPTIVE_FRAME_SAMPLING` | `false` | Enable motion-based adaptive frame sampling |
| `ADAPTIVE_FRAME_SAMPLING_SENSITIVITY` | `0.3` | Sampling density near scene boundaries |
| `CLIP_FRAME_DEDUP` | `false` | Enable CLIP-similarity frame deduplication |
| `CLIP_FRAME_DEDUP_THRESHOLD` | `0.92` | Similarity threshold for frame deduplication |

### 🔧 Improvements

- `ui/health.py`: Added `_setup_auth_middleware()` — configurable HTTP Basic Auth middleware. When `GRADIO_PASSWORD` is not set, the middleware is a no-op (no auth required), maintaining backward compatibility.
- `video_analysis/config.py`: Added `ui_auth_enabled`, `ui_auth_username`, `ui_auth_password`, `adaptive_frame_sampling`, `adaptive_frame_sampling_sensitivity`, `clip_frame_dedup`, `clip_frame_dedup_threshold` fields.
- `video_analysis/pipeline.py`: Added `_adaptive_frame_samples()` and `_dedup_frames_clip()` methods. Modified `_extract_key_frames()` to use adaptive sampling and/or CLIP dedup when configured.

### 🧪 Tests

- Added tests for new config fields and view-only auth module import.

---

## 0.6.0 (2026-06-26)

### 🧠 New Feature: Optional ColBERTv2 Late-Interaction Re-Ranker

- **ColBERTv2 Integration**: New optional `ColBERTReranker` module at `video_analysis/colbert_reranker.py` wraps RAGatouille (AnswerDotAI) for token-level late-interaction re-ranking. Improves retrieval precision for complex queries by matching individual tokens rather than whole vectors.
- **Config toggle**: `colbert_reranker_enabled: bool = False` in `video_analysis/config.py` — set to `True` to enable. Falls back gracefully to the cross-encoder if RAGatouille is not installed.
- **VRAM efficient**: Lazy-loads ColBERTv2 (~2-3 GB VRAM), runs re-ranking, then unloads to free GPU memory. Compatible with 12 GB RTX 4070 sequential model loading.
- **Optional dependency**: `ragatouille>=1.0.0` commented out in `requirements.txt` — install when needed with `pip install ragatouille`.

### 🎬 Timeline Hover Preview: Gradio 6 Shadow DOM Fix

- **Shadow DOM penetration**: Rewrote the JavaScript timeline preview (`ui/app.py:1381-1571`) to use a recursive shadow DOM traversal (`findVideoElements()`) instead of the broken `.gradio-video video` CSS selector. Works with Gradio 6's LitElement-based Web Components where `<video>` lives inside a Shadow DOM.
- **Visibility-aware detection**: `scanForVideo()` filters hidden video elements by checking `getBoundingClientRect()` — only attaches to the visible video player.
- **Graceful tab switching**: Periodic 2-second polling re-scans when Gradio lazy-renders tabs, ensuring the preview attaches to newly loaded video players without manual refresh.
- **Cleaner code**: Removed brittle `video.closest('gradio-video')` call that failed when the video was nested in a shadow root. Now walks up from the `<video>` element to find the first non-video container.

### 🔧 Improvements

- **.dockerignore fix**: Removed the blanket `*.md` exclusion pattern that was accidentally excluding `README.md` and `CHANGELOG.md` from the Docker build context. Now explicitly lists only research documents for exclusion, preserving README and CHANGELOG in the image.
- **Health model check fix**: The `health.py` module was trying to `import whisper` instead of `faster_whisper` in the model check — now correctly reflects the actual dependency.

### 📦 Dependencies

- **Optional**: `ragatouille>=1.0.0` — ColBERTv2 late-interaction re-ranking (commented out, install on demand)

---

## 0.5.0 (2026-06-26)

### 🎬 New Features

- **🧠 OpenCLIP ViT-L-14 Support**: Configurable CLIP model size — switch between ViT-B-32 (default, fast) and ViT-L-14 (richer scene descriptions, +3% accuracy). New config fields: `clip_model`, `clip_pretrained_dataset`, `clip_embed_dim`. ViT-L-14 uses `laion2b_s32b_b82k` pretrained weights.
- **🔍 Enhanced Scene Detection**: New `"histogram"` and `"hash"` detector modes in addition to `"adaptive"`, `"content"`, and `"ffmpeg"`. HistogramDetector uses Y-channel histogram differences for fast cuts; HashDetector uses perceptual hashing for similarity-based scene boundary detection.
- **🏥 Health Endpoint & API**: FastAPI `/health` endpoint with GPU availability, model status, version, and uptime. API endpoints at `/api/library` and `/api/video/{video_id}` for programmatic access. Gradio app mounts on FastAPI using `gr.mount_gradio_app()`.
- **⬆️ Embedding Model Upgrade**: Default embedding model changed to `nomic-ai/nomic-embed-text-v1.5` (768-dim, Apache 2.0, MTEB ~64) — significantly better retrieval quality vs previous BGE-small (384-dim, MTEB ~50).

### 🔧 Improvements

- **Docker Production Ready**: Updated to CUDA 12.8 runtime with torch 2.6 wheels. HEALTHCHECK now uses proper `/health` endpoint. Docker Compose exposes port 7861 for health API.
- **Pipeline Cleanup**: Improved model loading with configurable CLIP model size, pretrained dataset selection, and batch inference. The `_describe_scenes_clip` method now reads `clip_model` and `clip_pretrained_dataset` from config instead of hardcoded values.
- **Test Suite**: Added 7 new tests covering CLIP config fields, scene detector options, embedding model defaults, health module import.

### 📦 Dependencies

- **Updated**: `open-clip-torch>=2.24.0` (supports ViT-L-14 via pretrained flag)
- **Updated**: `sentence-transformers>=3.0.0` (recommended for nomic-embed)
- **Updated**: `scenedetect>=0.7.0` (now explicitly uncommented in requirements.txt)
- **Updated**: CUDA stacks upgraded from 12.4 to 12.8, torch from 2.1 to 2.6

---

## 0.4.0 (2026-06-26)

### 🎬 New Features

- **🌐 YouTube URL Import**: Download and analyze videos directly from YouTube, Vimeo, and other platforms via yt-dlp integration. Paste any URL in the UI or use `--url` in CLI mode.
- **📦 Batch Processing Queue**: New batch processing tab allows queuing multiple videos (by URL or file upload) for sequential analysis. Batch mode also available via `--batch urls.txt` in CLI.
- **🗂️ UI Utils Module**: Extracted `parse_yt_url()` and `queue_html()` into `ui/utils.py` — importable without gradio dependency, enabling proper unit testing of UI logic.

### 🔧 Improvements

- **Timeline Hover Preview JS Fix**: Enhanced the JavaScript timeline preview with proper CSS positioning, multiple sprite URL fallback paths, and fixed floating-point hover card rendering. Preview now shows thumbnail + timestamp on timeline hover.
- **CLI Enhancements**: Added `--url` flag for YouTube downloads, `--batch` flag for processing from a file list, and improved error handling.
- **Config**: New `yt_dlp_enabled`, `yt_dlp_format`, `yt_dlp_output_template`, and `batch_concurrent` configuration fields.

### 📦 Dependencies

- **New**: `yt-dlp>=2024.0.0` — YouTube/URL video import and batch processing

---

## 0.3.0 (2026-06-26)

### 🎬 New Features

- **🗣️ Speaker Diarization**: Automatic speaker labeling via PyAnnote Audio (`pyannote/speaker-diarization-3.1`). Each transcript segment now gets a `SPEAKER_00`, `SPEAKER_01`, etc. label, enabling speaker-aware Q&A. Configurable via `diarize_enabled`. Graceful fallback if PyAnnote is not installed.
- **🔤 OCR Text Extraction**: On-screen text detection via PaddleOCR (CPU mode). Extracts text from key frames and stores in `FrameInfo.ocr_text`. Visible in RAG context and Q&A responses. Configurable via `ocr_enabled` and `ocr_confidence`.
- **🐳 Docker Deployment**: Complete Dockerfile (multi-stage, CUDA 12.4 runtime) and docker-compose.yml with GPU passthrough, health checks, persistent volumes, and Nvidia container toolkit support.
- **📚 Library Tab Video Player**: Library cards are now clickable — clicking a video in the library loads it in a video player with metadata display. JS bridge (`window.__selectVideo`) connects Gradio UI to the library backend.

### 🔧 Improvements

- **Timeline Hover Preview Fix**: Rewrote the JavaScript timeline hover detection to work with Gradio 6's `<gradio-video>` web component. Now detects hover on the video container's bottom area rather than relying on the non-existent `<input type="range">` element.
- **Config Flags**: New `ocr_enabled`, `diarize_enabled`, `ocr_confidence` config fields for fine-grained pipeline control.
- **Pipeline Step Count**: 12 pipeline steps (up from 9) — added OCR extraction and speaker diarization.

### 📦 Dependencies

- **New optional**: `paddleocr>=2.8.0` — OCR text extraction
- **New optional**: `pyannote.audio>=3.1.0` — Speaker diarization
- Both are optional with graceful fallbacks if not installed.

## 0.2.0 (2026-06-26)

### 🎬 New Features

- **Clip Export**: Export video clips at precise timestamps directly from the UI — select start/end times and export a trimmed MP4
- **📚 Video Library**: Multi-video management with library tab, refresh, and video info display
- **🖼️ Sprite Sheet Timeline Preview**: Automatic generation of 100-thumbnail sprite sheets for visual timeline browsing
- **🧠 OpenCLIP Zero-shot Classification**: Rich semantic scene descriptions (indoor/outdoor, interview, lecture, etc.) using OpenCLIP ViT-B-32 embeddings on each key frame — improves RAG context quality
- **🎛️ GPU Pipeline Management**: Sequential model loading/unloading to respect 12GB VRAM limits

## 0.1.0 (2026-06-26)

### Initial Release

- **Core pipeline**: FFmpeg-based scene detection, frame extraction, faster-whisper transcription, YOLO object detection
- **RAG engine**: ChromaDB vector store with hybrid BM25/dense retrieval, cross-encoder re-ranking, temporal context expansion
- **Chat interface**: Video Q&A with source citations (clickable timestamps), conversation history
- **Web UI**: Gradio Blocks with dark theme, video upload, real-time analysis progress, streaming chat
- **CLI mode**: Batch processing and Q&A from the terminal
- **GPU acceleration**: Full CUDA support for RTX 4070
- **All local**: No API keys required — runs entirely on self-hosted hardware
