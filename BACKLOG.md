# Engineering Backlog — video-analysis v0.60.0

> **Priority levels:** Critical → High → Medium → Low  
> **Effort:** Small (<1 day) | Medium (2–3 days) | Large (1 week) | XLarge (2+ weeks)

---

## Infrastructure & CI/CD

### [VA-001] Add GitHub Actions CI/CD Pipeline

- **Category:** ci-cd
- **Priority:** critical
- **Impact:** high
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `.github/workflows/ci.yml`, `.github/workflows/publish.yml`
- **Reasoning:** The project has zero CI/CD. Pre-commit hooks exist (ruff, mypy) but are never enforced automatically. No test runner runs on push/PR. No Docker image is published to any registry. This is the single biggest production gap — every change must be manually tested, and there is no quality gate for contributions. A basic CI pipeline should: (1) run `pytest` (with `-m "not gpu"` for CPU runners), (2) run `ruff check` + `ruff format --check`, (3) run `mypy` on typed modules, (4) build the Docker image and smoke-test it starts. A publish workflow should tag and push to Docker Hub / GHCR on version tags.

---

### [VA-002] Add `.env.example` Template

- **Category:** dev-experience
- **Priority:** high
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `.env.example`
- **Reasoning:** There are 60+ environment variables documented in the README but no `.env.example` for quick setup. New contributors must manually piece together which env vars are required vs optional. A template with commented defaults would drastically lower the onboarding barrier. Include all `_ENABLED` toggles, API keys (with placeholder values), and path overrides.

---

### [VA-003] Add `Makefile` with Common Dev Tasks

- **Category:** dev-experience
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `Makefile`
- **Reasoning:** No convenience targets exist. Common operations require memorizing or copy-pasting commands. A `Makefile` should provide: `make install`, `make test`, `make lint`, `make format`, `make typecheck`, `make docker-build`, `make clean`. This standardizes the dev workflow across contributors.

---

### [VA-004] Fix Build Backend — Use `setuptools.build_meta` Instead of Legacy

- **Category:** tech-debt
- **Priority:** high
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `pyproject.toml`
- **Reasoning:** The build backend is `setuptools.backends._legacy:_Backend`, which was deprecated in setuptools 68+. The modern `setuptools.build_meta` should be used instead. Legacy backend support may be removed in future setuptools versions, which would break `pip install`.

---

### [VA-005] Add `py.typed` Marker File

- **Category:** tech-debt
- **Priority:** low
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/py.typed`
- **Reasoning:** Several modules (`error_handlers.py`, `rate_limiter.py`, `config.py`) use modern type hints with `from __future__ import annotations`. Without a `py.typed` marker, downstream consumers using `mypy` or `pyright` won't benefit from the package's type information. This is a one-line empty file in the package directory.

---

### [VA-006] Eliminate Version Drift — Single Source of Truth for `__version__`

- **Category:** tech-debt
- **Priority:** medium
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `pyproject.toml`, `video_analysis/__init__.py`
- **Reasoning:** Version `0.60.0` is duplicated in `pyproject.toml` and `video_analysis/__init__.py`. These will inevitably drift. Use `importlib.metadata` to read the version from the installed package metadata at runtime, or use a dynamic version file that `pyproject.toml` reads via `tool.setuptools.dynamic`.

---

### [VA-007] Add `LICENSE` File to Repository Root

- **Category:** documentation
- **Priority:** medium
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `LICENSE`
- **Reasoning:** `pyproject.toml` declares the license as MIT, but no license file exists in the repo root. This is a legal ambiguity — users and contributors cannot see the actual terms. Add a standard MIT `LICENSE` file.

---

## Architecture & Code Quality

### [VA-008] Refactor `Config.__post_init__` — Extract Declarative Env-Override System

- **Category:** architecture
- **Priority:** critical
- **Impact:** high
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/config.py`
- **Reasoning:** `__post_init__` is ~323 lines of repetitive, error-prone env-var-override blocks. Each follows the same pattern (`os.environ.get()` → `try/except ValueError: pass`), with inconsistent handling: some fields read env vars at the class-body level AND in `__post_init__`, creating ambiguity about which wins. Silent `ValueError` swallowing (~15 instances) hides configuration errors. Extract a declarative override map:

```python
ENV_OVERRIDES: ClassVar[dict[str, tuple[str, Callable[[str], Any]]]] = {
    "video_mllm_enabled": ("VIDEO_MLLM_ENABLED", _parse_bool),
    "streaming_chunk_duration": ("STREAMING_CHUNK_DURATION", _parse_positive_float),
}
```

Consider migrating to `pydantic-settings` or `environs` to eliminate this pattern entirely.

---

### [VA-009] Split God Module — `pipeline.py` (1,964 lines)

- **Category:** architecture
- **Priority:** critical
- **Impact:** high
- **Effort:** xlarge
- **Status:** pending
- **Dependencies:** VA-010
- **Affected Files:** `video_analysis/pipeline.py`, `video_analysis/stages/`
- **Reasoning:** `pipeline.py` handles ~15 distinct responsibilities: scene detection, frame extraction, ASR transcription, YOLO object detection, OCR, face recognition, CLIP classification, X-CLIP action recognition, video MLLM, sprite sheets, URL downloading, audio extraction, corruption checks, graceful shutdown, and GPU memory management. It's a monolith where a single import error or logic bug in one stage can break the entire pipeline. Create a `stages/` package with an abstract `PipelineStage` ABC:

```python
class PipelineStage(ABC):
    @abstractmethod
    async def run(self, video_path: Path, config: Config,
                  intermediate: StageContext) -> StageResult: ...
```

Each stage (TranscriptionStage, SceneDetectionStage, YOLOStage, OCRStage, etc.) becomes a separate module. The pipeline orchestrator becomes a slim coordinator that discovers and invokes stages.

---

### [VA-010] Split God Module — `rag.py` (1,753 lines)

- **Category:** architecture
- **Priority:** critical
- **Impact:** high
- **Effort:** xlarge
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/rag.py`, `video_analysis/retrieval/`
- **Reasoning:** `rag.py` handles ChromaDB management, BGE-VL embeddings, SentenceTransformer fallback, Qwen3-VL multimodal embeddings, cross-encoder re-ranking, ColBERTv2, ColBERT-Att, MMR diversity, scene graph integration, query routing, self-check RAG, event-causal RAG integration, speaker diarization in scoring, temporal re-weighting, chunking strategies (fixed/sliding/transcript), and library management. Extract into a `retrieval/` package with:
- `retrieval/embedders/` — abstract `Embedder` interface with BGE-VL, nomic, Qwen3-VL implementations
- `retrieval/rerankers/` — cross-encoder, ColBERTv2, ColBERT-Att, MMR
- `retrieval/chunking/` — fixed, sliding, transcript-based strategies
- `retrieval/store.py` — ChromaDB wrapper
- `retrieval/pipeline.py` — orchestration of the retrieval flow

---

### [VA-011] Split God Module — `api.py` (1,644 lines)

- **Category:** architecture
- **Priority:** high
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/api.py`, `video_analysis/routes/`
- **Reasoning:** The API module has inline Pydantic schemas, all endpoint logic, lazy singleton management, and inline helper functions. Split into a `routes/` package (mirroring the `ui/routes/` structure): `routes/video.py`, `routes/search.py`, `routes/chat.py`, `routes/health.py`, `routes/jobs.py`. Shared schemas go into `routes/schemas.py`. The main `api.py` becomes a router aggregator and app factory.

---

### [VA-012] Split God Module — `orchestra.py` (1,493 lines)

- **Category:** architecture
- **Priority:** high
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/orchestra.py`
- **Reasoning:** The multi-agent orchestrator has RouterAgent + 6 specialist agents + evidence synthesis all in one file. Extract each specialist agent into its own module (`agents/visual_agent.py`, `agents/text_agent.py`, `agents/temporal_agent.py`, etc.) and keep `orchestra.py` as the routing/coordination layer.

---

### [VA-013] Consolidate Overlapping Orchestration — `orchestrator.py` vs `orchestra.py` vs `agent.py`

- **Category:** architecture
- **Priority:** high
- **Impact:** medium
- **Effort:** large
- **Status:** pending
- **Dependencies:** VA-009, VA-012
- **Affected Files:** `video_analysis/orchestrator.py`, `video_analysis/orchestra.py`, `video_analysis/agent.py`, `video_analysis/classifier.py`
- **Reasoning:** Three modules overlap:
- `orchestrator.py` (355 lines) — video type detection + pipeline stage selection, reimplements logic from `classifier.py`
- `orchestra.py` (1,493 lines) — hierarchical multi-agent routing
- `agent.py` (1,176 lines) — flat multi-tool reasoning agent
- `classifier.py` (1,176 lines) — 3-tier media type classifier
`orchestrator.py` and `classifier.py` both sniff video type via extension/ffprobe. `agent.py` and `orchestra.py` solve the same problem (answering questions about video) at different abstraction levels. De-duplicate video type detection into `classifier.py`, decide whether `agent.py` should be deprecated in favor of `orchestra.py`, and rename `orchestrator.py` to avoid confusion (e.g., `pipeline_profiler.py`).

---

### [VA-014] Add Abstract Embedder and Retriever Interfaces

- **Category:** architecture
- **Priority:** high
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** VA-010
- **Affected Files:** `video_analysis/rag.py`
- **Reasoning:** BGE-VL, SentenceTransformer/nomic, and Qwen3-VL embedding are all inlined as conditional blocks inside `rag.py`. There is no abstract `Embedder` interface, making it impossible to add a new embedding model without editing the central RAG module. Define:

```python
class Embedder(ABC):
    @abstractmethod
    def embed_text(self, texts: list[str]) -> np.ndarray: ...
    @abstractmethod
    def embed_image(self, images: list[Image.Image]) -> np.ndarray: ...
    @abstractmethod
    def embed_composed(self, text: str, image: Image.Image) -> np.ndarray: ...
```

Same for retrieval: an abstract `Retriever` interface would allow swapping ChromaDB for LanceDB, Qdrant, or FAISS without touching RAG internals.

---

### [VA-015] Add Abstract VideoMLLM Backend Interface

- **Category:** architecture
- **Priority:** medium
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/video_mllm.py`, `video_analysis/backends/`
- **Reasoning:** `llm_provider.py` has a clean ABC (`LLMProvider` with `chat()` and `stream_chat()`), but `video_mllm.py` has no similar interface. Each backend (VideoChat-Flash, SmolVLM2, Qwen3-VL, InternVideo3) is loaded via conditional logic. Define `VideoMLLMBackend(ABC)` with `analyze(video_path, prompt) → str` and have each backend implement it. This mirrors the clean pattern already established in `llm_provider.py`.

---

### [VA-016] Consolidate ColBERT Reranker Modules

- **Category:** architecture
- **Priority:** medium
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:** VA-010
- **Affected Files:** `video_analysis/colbert_reranker.py`, `video_analysis/colbert_att_reranker.py`
- **Reasoning:** Two separate modules exist for ColBERT-based reranking. `colbert_reranker.py` (137 lines) is a thin RAGatouille wrapper that could fold into `colbert_att_reranker.py` (329 lines) as a base-class or configuration option. This reduces module count and makes the ColBERT code path easier to maintain.

---

### [VA-055] Add Structured Exception Hierarchy (`exceptions.py`)

- **Category:** architecture
- **Priority:** high
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/exceptions.py`, `video_analysis/error_handlers.py`, `video_analysis/pipeline.py`, `video_analysis/rag.py`, `video_analysis/streaming.py`, `video_analysis/chat.py`, `video_analysis/agent.py`
- **Reasoning:** `error_handlers.py` defines `StandardHTTPError` and structured JSON error responses for the API layer, but the internal modules raise generic `Exception`, `ValueError`, or `RuntimeError` with no taxonomy. There is no dedicated `exceptions.py` with a typed hierarchy. Define:

```python
class VideoAnalysisError(Exception):
    """Base for all application-level errors."""

class PipelineError(VideoAnalysisError): ...
class StageNotFoundError(PipelineError): ...
class StageExecutionError(PipelineError): ...

class RetrievalError(VideoAnalysisError): ...
class EmbeddingError(RetrievalError): ...
class IndexError(RetrievalError): ...

class ConfigError(VideoAnalysisError): ...
class ValidationError(ConfigError): ...

class DependencyError(VideoAnalysisError): ...
class ResourceExhaustedError(VideoAnalysisError): ...

class WebhookDeliveryError(VideoAnalysisError): ...
```

This hierarchy lets `error_handlers.py` catch specific types (e.g., `PipelineError` → 500, `ValidationError` → 422) without inspecting error message strings. It also enables proper `except PipelineError` clauses instead of `except Exception` throughout the codebase. Complements VA-038 (error handling standardization) by providing the concrete exception types that the standard handlers will catch.

---

## Production Hardening

### [VA-017] Add API Authentication / Authorization

- **Category:** security
- **Priority:** critical
- **Impact:** high
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/api.py`, `video_analysis/config.py`, `video_analysis/error_handlers.py`
- **Reasoning:** The REST API has zero authentication. Every endpoint (video processing, search, chat, system configuration) is accessible to anyone who can reach port 7860. While the production compose file sets `GRADIO_PASSWORD`, there is no REST API auth. Add JWT-based token authentication or API key validation via `fastapi.Security` / `fastapi.Depends`. At minimum, require `Authorization: Bearer <API_KEY>` for all non-health endpoints. The rate limiter is a necessary but insufficient safeguard. This is critical before any production deployment that is not behind a VPN.

---

### [VA-018] Fix Memory Leak in Rate Limiter — Add Bucket Eviction

- **Category:** performance
- **Priority:** high
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/rate_limiter.py`
- **Reasoning:** The `_buckets` dict in the token bucket rate limiter grows unboundedly. Each unique client IP creates a new bucket that lives forever. In a long-running server with many unique clients, this is a memory leak. Add a `_max_clients` parameter with LRU eviction or a periodic cleanup sweep that removes stale buckets older than a configurable TTL.

---

### [VA-019] Add Webhook Retry Queue with Persistence

- **Category:** reliability
- **Priority:** high
- **Impact:** high
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/webhook.py`, `video_analysis/config.py`, `video_analysis/job_queue.py`
- **Reasoning:** Webhooks are currently fire-and-forget with a single retry attempt and no disk persistence. If the server restarts during delivery, pending webhooks are lost. If the receiving endpoint is down, the event is silently dropped after one retry. Add:
- Exponential backoff retry with configurable max attempts
- SQLite-backed delivery queue for crash recovery
- Dead-letter tracking for permanently failed deliveries
- Optional delivery receipt confirmation (expect 2xx within timeout)

---

### [VA-020] Add ChromaDB / SQLite Migration Strategy

- **Category:** infrastructure
- **Priority:** medium
- **Impact:** high
- **Effort:** large
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/knowledge_graph.py`, `video_analysis/pipeline_health.py`, `video_analysis/rag.py`
- **Reasoning:** Both ChromaDB and the SQLite database (`knowledge_graph.db`, pipeline health DB) have no versioned migration path. Schema changes between releases could corrupt or invalidate existing data directories. Users who upgrade from v0.50 to v0.60 may lose their indexed videos. Implement:
- A schema version table in SQLite with Alembic-style migration runs
- ChromaDB collection metadata storing the schema version, with upgrade scripts
- A `migrate` CLI command that runs pending migrations
- Backward-compatibility checks on startup

---

### [VA-021] Add Horizontal Scaling Strategy — External Job Queue

- **Category:** architecture
- **Priority:** medium
- **Impact:** medium
- **Effort:** xlarge
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/job_queue.py`, `video_analysis/pipeline.py`, `video_analysis/api.py`, `docker-compose.prod.yml`
- **Reasoning:** The current job queue is in-process `asyncio.TaskGroup`. This means: (1) only one process can run at a time, (2) a server restart kills running jobs, (3) ML models (whisper, CLIP, ChromaDB) all load in the same process, consuming GPU memory together. For multi-user or production deployments, an external job queue (Celery + Redis/RabbitMQ, or `arq` + Redis) is needed to distribute work across workers. This also enables running ASR on one GPU, vision models on another.

---

### [VA-022] Add Graceful Degradation for Optional Dependencies

- **Category:** reliability
- **Priority:** high
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/pipeline.py`, `video_analysis/rag.py`, `video_analysis/face.py`, `video_analysis/config.py`
- **Reasoning:** Several modules have optional dependencies (insightface for face recognition, xclip-base-patch16 for action recognition, opencv-contrib for certain operations), but failures during lazy import may crash the pipeline with inscrutable errors. Add consistent `_check_dependencies()` patterns that:

1. Attempt import at stage initialization, not at module load time
2. Log a clear warning with installation instructions on failure
3. Raise a specific `OptionalDependencyError` that the pipeline can catch and skip
4. Include a `pip install` command in the error message

---

## Testing

### [VA-023] Add Tests for `logging_setup.py`

- **Category:** testing
- **Priority:** high
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `tests/test_logging_setup.py`, `video_analysis/logging_setup.py`
- **Reasoning:** `logging_setup.py` has zero test coverage despite being loaded at startup and configuring all structured logging. It uses global mutable state (`_SIMPLE_FILTER_LEVEL` modified with `global`) which makes tests non-deterministic when called multiple times. Tests should cover: TTY-aware renderer selection, JSON vs console format, level filtering, `PipelineLogger` stage-level logging, and the `_simple_filter_by_level` processor.

---

### [VA-024] Add Tests for `config_store.py`

- **Category:** testing
- **Priority:** high
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `tests/test_config_store.py`, `video_analysis/config_store.py`
- **Reasoning:** `config_store.py` handles JSON persistence of runtime Config changes, type coercion, filtered serialization, and backward-compatible upgrades. Zero tests exist. This module is critical for the runtime settings UI and data integrity across restarts. Tests must cover: save/load cycle, partial updates, type coercion from strings, upgrade of old configs with missing fields, singleton management, and concurrent access.

---

### [VA-025] Add End-to-End Integration Tests

- **Category:** testing
- **Priority:** high
- **Impact:** high
- **Effort:** large
- **Status:** pending
- **Dependencies:** VA-001
- **Affected Files:** `tests/test_e2e.py`
- **Reasoning:** No cross-module end-to-end tests exist. Each module is tested in isolation with mocks, but there is no test that runs the full pipeline → ChromaDB indexing → RAG query → API response flow. Add a CI-integrated test that:

1. Processes a small test video (e.g., 5-second synthetic video)
2. Indexes results into ChromaDB
3. Queries the RAG system and asserts relevant results
4. Hits the REST API endpoints
5. Verifies the report output schema
This catches integration bugs that unit tests miss (e.g., schema mismatches between pipeline output and RAG input).

---

### [VA-026] Organize Test Files into Subdirectories

- **Category:** testing
- **Priority:** low
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `tests/` (directory structure)
- **Reasoning:** All 48 test files live in a flat `tests/` directory. As the project grows, this becomes unwieldy. Organize into subdirectories: `tests/unit/`, `tests/integration/`, `tests/benchmarks/`, `tests/evaluation/` with corresponding `conftest.py` files and marker configuration.

---

### [VA-056] Add/Improve `conftest.py` with Shared Fixtures

- **Category:** testing
- **Priority:** high
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `tests/conftest.py`, `tests/unit/conftest.py`, `tests/integration/conftest.py`
- **Reasoning:** The root `tests/conftest.py` is minimal — it only registers custom markers and enables `pytest_asyncio`. There are no shared fixtures, so every test module that needs a Config object, a temp directory, a sample video path, or a mock pipeline result creates its own. This leads to fixture duplication across the 48+ test files. Add shared fixtures:

```python
@pytest.fixture
def test_config() -> Config:
    """Config with defaults overridden for test isolation."""
    return Config(
        data_dir=tmp_path / "data",
        chroma_persist_dir=tmp_path / "chroma",
        # Disable GPU-dependent features
        video_mllm_enabled=False,
        face_recognition_enabled=False,
        action_recognition_enabled=False,
    )

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Isolated data directory for a single test."""
    d = tmp_path / "data"
    d.mkdir(parents=True)
    return d

@pytest.fixture
def sample_video(tmp_data_dir: Path) -> Path:
    """Path to a small test video (5s synthetic)."""
    # Generate or locate a pre-committed fixture
    ...

@pytest.fixture
def mock_pipeline_result() -> PipelineResult:
    """A realistic but synthetic PipelineResult for retrieval tests."""
    ...
```

These fixtures reduce boilerplate, ensure test isolation, and make it trivial to add new tests without copy-pasting setup logic. Place shared fixtures in `tests/conftest.py`, domain-specific ones in `tests/unit/conftest.py` and `tests/integration/conftest.py`.

---

## Security

### [VA-027] Add Secrets Management — Move Beyond Plain Env Vars

- **Category:** security
- **Priority:** high
- **Impact:** high
- **Effort:** medium
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/config.py`, `docker-compose.prod.yml`, `Dockerfile`
- **Reasoning:** `HF_TOKEN`, `OPENAI_API_KEY`, `GRADIO_PASSWORD`, `LLM_API_KEY` are all passed as plain environment variables. These leak through process listings, Docker inspect, and logs (if not redacted). For production deployments, integrate with Docker secrets, Kubernetes Secrets, or HashiCorp Vault. At minimum: (1) add an `.env` file gitignore entry, (2) document Docker secrets usage in compose files, (3) add a `--secrets` CLI flag for reading secrets from files.

---

### [VA-028] Add Security Scanning to CI

- **Category:** security
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** VA-001
- **Affected Files:** `.github/workflows/ci.yml`
- **Reasoning:** No dependency vulnerability scanning or SAST (Static Application Security Testing) is configured. Add: (1) `pip-audit` or `safety` for Python dependency CVE scanning, (2) `bandit` for Python SAST, (3) GitHub's `Dependabot` for automated dependency update PRs, (4) `trivy` or `grype` for Docker image vulnerability scanning.

---

### [VA-029] Add Input Validation and Sanitization for Video Uploads

- **Category:** security
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/api.py`, `video_analysis/pipeline.py`, `video_analysis/__main__.py`
- **Reasoning:** The pipeline accepts arbitrary file paths and URLs for video processing. `yt-dlp` downloads could in theory point to malicious content. `ffprobe` and OpenCV process untrusted input. Add: file extension whitelist, MIME-type verification, maximum file size enforcement, URL scheme allowlisting (http/https only), and path traversal protection for local files.

---

## Observability & Monitoring

### [VA-030] Fix Telemetry No-op Boilerplate — Replace with `Protocol` Pattern

- **Category:** observability
- **Priority:** medium
- **Impact:** low
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/telemetry.py`
- **Reasoning:** The telemetry module has ~100+ lines of no-op sentinel classes (`_NoopSpan`, `_NoopTracer`, `_NOOP_SPAN`, `_NOOP_TRACER`) plus `cast()` calls and `Any` types throughout. Replace with a `Protocol` for the tracer/span interface and a single `_NoopOTel` module that replaces the entire `opentelemetry` module on import failure. This eliminates the `Any` usage and makes the no-op path testable.

---

### [VA-031] Add Rate Limit Enforcement for Gradio UI in Addition to REST API

- **Category:** security
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/stream_chat.py`, `video_analysis/chat.py`, `ui/app.py`
- **Reasoning:** The token bucket rate limiter only covers the REST API. The Gradio UI (web chat, streaming, batch processing) has no rate limiting. A user could submit unlimited chat queries or batch jobs through the UI, overwhelming the ML models. Apply rate limiting to the Gradio event handlers as well, or centralize rate limiting at the application level (before the REST/UI split).

---

### [VA-032] Improve `pipeline_health.py` — Add Proactive Alerting Integration

- **Category:** observability
- **Priority:** medium
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/pipeline_health.py`, `video_analysis/webhook.py`
- **Reasoning:** `pipeline_health.py` monitors drift detection, anomaly detection, and composite health scoring, but it is purely internal — it stores data locally but doesn't integrate with external alerting systems. Wire it into the webhook system so that: (1) anomalies trigger webhook notifications, (2) sustained degradation (e.g., 3 consecutive low-health scores) triggers an alert, (3) recovery from a degraded state sends a resolved notification. Add optional integration with Slack/Discord webhooks.

---

## Documentation & Developer Experience

### [VA-033] Add `CONTRIBUTING.md` Guide

- **Category:** documentation
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `CONTRIBUTING.md`
- **Reasoning:** The project has 48 test files, comprehensive pre-commit hooks, and a structured development setup, but no contributor guide. Add a `CONTRIBUTING.md` covering: (1) how to set up the dev environment, (2) how to run tests (`pytest -m "not gpu"`), (3) linting and formatting expectations, (4) PR process, (5) how to add a new backend/embedder/stage, (6) how to write tests for a new module. This lowers the barrier for community contributions.

---

### [VA-034] Generate Auto-Updated API Documentation

- **Category:** documentation
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/api.py`, `docs/api.md`
- **Reasoning:** The FastAPI server auto-generates OpenAPI docs at `/docs`, but there is no static API documentation in the repo. Add: (1) a CI step that exports OpenAPI JSON and commits it to `docs/openapi.json`, (2) a `docs/api.md` that provides human-readable endpoint reference with examples, (3) links from README to the API docs. This is especially important since the repo has a Python SDK client (`client.py`) — API docs help users understand what endpoints are available.

---

### [VA-035] Add MCP Server Documentation

- **Category:** documentation
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `docs/mcp.md`, `video_analysis/mcp_server.py`
- **Reasoning:** The MCP server exposes 7 tools (pipeline, streaming, RAG, chat, federation) as MCP tools, but there is no documentation on how to configure an MCP client to connect to it, what tools are available, or what arguments each tool expects. Add a `docs/mcp.md` covering: MCP client configuration (Claude Desktop, VS Code Cline, etc.), tool reference with argument descriptions, example queries, and connection troubleshooting.

---

## Quality of Life & Developer Experience

### [VA-036] Fix Global Mutable State in `logging_setup.py`

- **Category:** tech-debt
- **Priority:** medium
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/logging_setup.py`
- **Reasoning:** `_SIMPLE_FILTER_LEVEL` is a module-level variable modified via `global` inside `setup_logging()`. Calling `setup_logging()` twice changes behavior. The `basicConfig(force=True)` call clobbers any existing logger configuration. Fix: (1) make `_SIMPLE_FILTER_LEVEL` a function-local closure or class attribute, (2) use `basicConfig(force=False)` and check if handlers are already configured, (3) add idempotency guard (`_initialized` flag).

---

### [VA-037] Replace `typing.List` with `list` in `config.py`

- **Category:** tech-debt
- **Priority:** low
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/config.py`
- **Reasoning:** The project targets Python 3.10+ but `config.py` uses `from typing import List` with `List[str]` instead of the native `list[str]` syntax. This is a minor style inconsistency that modern tooling (ruff format) would flag. Apply `list[str]` consistently across the file.

---

### [VA-038] Standardize Error Handling Across All Modules

- **Category:** tech-debt
- **Priority:** medium
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** VA-055
- **Affected Files:** `video_analysis/pipeline.py`, `video_analysis/rag.py`, `video_analysis/streaming.py`, `video_analysis/chat.py`, `video_analysis/agent.py`, `video_analysis/error_handlers.py`
- **Reasoning:** `error_handlers.py` provides structured JSON error responses for the FastAPI layer, but the internal modules (`pipeline.py`, `rag.py`, `streaming.py`) use inconsistent error handling — sometimes raising generic `Exception`, sometimes logging and returning `None`, sometimes using `try/except: pass`. Once VA-055 provides the concrete exception hierarchy, update all modules to raise specific types (`PipelineError`, `RetrievalError`, `ConfigError`, `DependencyError`) that `error_handlers.py` can catch and format. This ensures consistent error surfaces whether accessed via API, CLI, or MCP.

---

### [VA-039] Add Per-Stage Pipeline Caching with Invalidation

- **Category:** performance
- **Priority:** medium
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** VA-009
- **Affected Files:** `video_analysis/cache.py`, `video_analysis/pipeline.py`
- **Reasoning:** The cache module (`cache.py`, 506 lines) uses SHA-256 content-addressable caching per stage, but there is no selective invalidation — clearing cache is all-or-nothing via the API. Add: (1) per-stage cache invalidation, (2) TTL-based expiration for stages whose models or config may change (e.g., re-running with a different CLIP model), (3) cache statistics endpoint showing hit/miss ratios per stage, (4) optional S3/GCS backend for distributed deployments.

---

### [VA-040] Add Feature Flag System to Replace Env Var Explosion

- **Category:** architecture
- **Priority:** medium
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:** VA-008
- **Affected Files:** `video_analysis/config.py`, `video_analysis/feature_flags.py`
- **Reasoning:** With 20+ boolean feature toggles (`*_enabled`), the config class has become a flat namespace. Adding a new flag requires: adding a dataclass field, adding an env-var override block in `__post_init__`, and updating the README. Extract a `FeatureFlag` system:

```python
@dataclass
class FeatureFlags:
    video_mllm: bool = False
    face_recognition: bool = False
    action_recognition: bool = False
    streaming: bool = False
    federation: bool = False
    orchestra: bool = False
    curator: bool = False
    # ...
```

Load from env vars using a convention like `FEATURE_VIDEO_MLLM_ENABLED` with automatic `__post_init__` reflection, eliminating the manual override blocks.

---

### [VA-041] Add `docker healthcheck` Endpoint

- **Category:** observability
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/api.py`, `Dockerfile`
- **Reasoning:** The Dockerfile has `HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl --fail http://localhost:7860/health || exit 1` but the `/health` endpoint currently only does an HTTP 200 check. Make the health endpoint more meaningful: check that ChromaDB is reachable, verify disk space for `data/` is not full, confirm that ML models are loaded (if configured), and return a detailed JSON response with component status. This enables Docker orchestration platforms to make informed restart decisions.

---

### [VA-042] Add Non-NVIDIA GPU Support to Adaptive Scaler

- **Category:** performance
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/adaptive_scaler.py`
- **Reasoning:** `get_free_vram_gb()` uses `pynvml` (NVIDIA-only), falling back to returning `None` on AMD ROCm or Apple MPS. This means the scaler runs blind on non-NVIDIA GPUs. Add: (1) ROCm support via `amdsmi`, (2) MPS support via `torch.cuda.is_available()` fallback with Metal-reported memory, (3) graceful fallback to CPU-available-memory when no GPU is detected.

---

### [VA-043] Reduce Coupling Between `agent_confidence.py` and `agent.py`

- **Category:** architecture
- **Priority:** medium
- **Impact:** medium
- **Effort:** medium
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/agent_confidence.py`, `video_analysis/agent.py`
- **Reasoning:** `agent_confidence.py` imports deeply from `agent.py` internals, making any refactor of the agent module a breaking change for confidence scoring. Extract the interfaces that `agent_confidence.py` depends on into a shared `agent_types.py` or `agent_protocol.py` module with Protocols/ABCs. This decouples the confidence system from the agent implementation and allows either module to be refactored independently.

---

### [VA-044] Standardize ASR Backend Abstraction for Future Backends

- **Category:** architecture
- **Priority:** medium
- **Impact:** low
- **Effort:** medium
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/pipeline.py`, `video_analysis/config.py`
- **Reasoning:** The config file notes Qwen3-ASR, Moonshine, and Parakeet as future ASR backends, with inline comments that they have "no pipeline integration code yet." The enum values exist in spirit but not in code. The only working backend is `faster-whisper`. Before adding more ASR backends, define an `ASRBackend(ABC)` interface (mirroring `LLMProvider`) so that new backends can be added without editing `pipeline.py`. The enum should either be removed until backends are implemented, or tracked in a feature roadmap.

---

### [VA-045] Add Unified Graph Layer — Merge `knowledge_graph.py` and `scene_graph.py`

- **Category:** architecture
- **Priority:** low
- **Impact:** low
- **Effort:** large
- **Status:** pending
- **Dependencies:** —
- **Affected Files:** `video_analysis/knowledge_graph.py`, `video_analysis/scene_graph.py`
- **Reasoning:** `knowledge_graph.py` (904 lines) is a persistent SQLite cross-video entity store. `scene_graph.py` (372 lines) is a transient in-memory per-video scene graph built from ChromaDB metadata. Both model entities and relationships. A unified graph layer would provide a common query interface, allow the scene graph to optionally persist to SQLite, and reduce conceptual overhead. However, this is a lower priority than the god-module splits since the two serve different query purposes.

---

### [VA-046] Add Pre-commit Hook for `ruff check` and `mypy`

- **Category:** dev-experience
- **Priority:** low
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `.pre-commit-config.yaml`
- **Reasoning:** The `.pre-commit-config.yaml` already has ruff and mypy configured, but the `mypy` hook may not run on all Python files due to the `Any` usage in telemetry and other modules. Ensure the pre-commit hooks are actually functional and pass on the current codebase. Add a CI check that verifies pre-commit hooks pass (using `pre-commit run --all-files`).

---

### [VA-047] Add Benchmarking Infrastructure as CI Gate

- **Category:** performance
- **Priority:** low
- **Impact:** low
- **Effort:** medium
- **Status:** pending
- **Dependencies:** VA-001
- **Affected Files:** `tests/benchmarks/`, `.github/workflows/benchmark.yml`
- **Reasoning:** The project has benchmark test files (`tests/benchmarks/`) and a `benchmark.py` profiling context manager, but there is no automated benchmarking pipeline. Add a CI workflow that: (1) runs benchmarks on a GPU runner, (2) stores historical results, (3) alerts on significant regressions (>10% slowdown). This prevents performance degradation during feature accretion, which is a real risk given the rapid pace of new features (12+ arXiv-backed modules).

---

### [VA-048] Refactor `__init__.py` — Replace Eager Imports with Lazy Re-exports

- **Category:** architecture
- **Priority:** medium
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/__init__.py`
- **Reasoning:** The `__init__.py` eagerly imports most modules at package init time (`from video_analysis import pipeline, rag, models, ...`). While this doesn't cause circular imports (the layered architecture prevents it), it does mean that `import video_analysis` triggers all module-level code execution. Some modules import heavy libraries (torch, transformers, chromadb) at the top level. Replace with lazy imports using `importlib.import_module` or use `__getattr__` for module-level lazy loading. This reduces startup time by 50%+ when only a subset of features are needed.

---

### [VA-049] Add `py.typed` Marker for `backends/` Package

- **Category:** tech-debt
- **Priority:** low
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/backends/py.typed`
- **Reasoning:** The `backends/` package is not marked as typed since there's no `py.typed` there. The Qwen3-VL and InternVideo3 backends have type hints. Add the marker.

---

### [VA-050] Add Data Retention and Cleanup Policy

- **Category:** operations
- **Priority:** medium
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/config.py`, `video_analysis/api.py`, `video_analysis/storage.py`
- **Reasoning:** The `data/` directory grows unboundedly — video files, extracted frames (3 tiers), audio files, ChromaDB index, clip exports, eval reports. There is no data retention policy or cleanup mechanism exposed through the API. Add: (1) configurable max storage budget (e.g., `max_storage_gb: int = 100`), (2) periodic cleanup that removes oldest videos when over budget, (3) API endpoint to query storage usage and trigger manual cleanup, (4) configurable frame TTL (delete frames older than N days, keeping metadata/index).

---

### [VA-051] Add Structured Logging for All External API Calls

- **Category:** observability
- **Priority:** low
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `video_analysis/llm_provider.py`, `video_analysis/webhook.py`, `video_analysis/rag.py`
- **Reasoning:** External API calls (LLM provider, webhook delivery, ChromaDB operations) are not consistently logged with request/response details, timing, and error status. Add structured logging at `DEBUG` level for: LLM API round-trips, webhook delivery attempts (with response code), ChromaDB query performance, federated peer queries. This is invaluable for debugging production issues.

---

### [VA-052] Remove `requirements.txt` — Use `pyproject.toml` as Single Source

- **Category:** tech-debt
- **Priority:** low
- **Impact:** low
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `requirements.txt`, `Dockerfile`, `pyproject.toml`
- **Reasoning:** `requirements.txt` duplicates the dependency list from `pyproject.toml`. This is a maintenance burden — any change to deps must be made in two places. The Dockerfile currently does `COPY requirements.txt . && pip install -r requirements.txt`. Switch to `pip install .` in the Dockerfile, or generate `requirements.txt` from `pyproject.toml` via `pip-compile` as a build step. Remove the duplicate file to prevent drift.

---

### [VA-053] Add Pre-commit Hook for Security Scanning (detect-secrets / truffleHog)

- **Category:** security
- **Priority:** low
- **Impact:** medium
- **Effort:** small
- **Status:** pending
- **Dependencies:**
- **Affected Files:** `.pre-commit-config.yaml`
- **Reasoning:** The `.pre-commit-config.yaml` does not include any secret detection hook. Add `detect-secrets` or `trufflehog` to prevent accidental commits of API keys (HF_TOKEN, LLM_API_KEY, OPENAI_API_KEY) that may be hardcoded during development.

---

### [VA-054] Add Plugin/Extension System for Custom Stages and Backends

- **Category:** architecture
- **Priority:** low
- **Impact:** medium
- **Effort:** xlarge
- **Status:** pending
- **Dependencies:** VA-009, VA-010, VA-015
- **Affected Files:** `video_analysis/plugin.py`, `pyproject.toml`
- **Reasoning:** Adding a new processing stage (e.g., a custom ML model for a specific domain) currently requires editing `pipeline.py`. Adding a new embedding model requires editing `rag.py`. A plugin system using `importlib.metadata.entry_points` would allow third-party extensions without forking the repo. Define `video_analysis.stages`, `video_analysis.embedders`, `video_analysis.backends` entry points. This is a long-term goal (post-1.0) but worth noting in the backlog.

---

## Summary

| Category | Count | Critical | High | Medium | Low |
|----------|-------|----------|------|--------|-----|
| Infrastructure & CI/CD | 7 | 1 | 2 | 2 | 2 |
| Architecture & Code Quality | 9 | 3 | 5 | 1 | 0 |
| Production Hardening | 6 | 1 | 3 | 1 | 1 |
| Testing | 5 | 0 | 4 | 0 | 1 |
| Security | 3 | 1 | 1 | 1 | 0 |
| Observability & Monitoring | 3 | 0 | 0 | 3 | 0 |
| Documentation & DevEx | 3 | 0 | 0 | 3 | 0 |
| Quality of Life & DevEx | 11 | 0 | 0 | 8 | 3 |
| **Total** | **47** | **6** | **15** | **19** | **7** |
