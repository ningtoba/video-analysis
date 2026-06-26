# Production Hardening & Deployment Audit — Video Analysis Platform v0.14.0

> Audit date: 2026-06-26
> Platform: Self-hosted video analysis (FastAPI + Gradio + CUDA 12.8)
> Hardware: RTX 4070 (12 GB VRAM), Docker Compose, NVIDIA Container Toolkit

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Already Implemented (What's Done)](#2-already-implemented-whats-done)
3. [Missing: Critical Gaps](#3-missing-critical-gaps)
4. [Missing: High-Priority Gaps](#4-missing-high-priority-gaps)
5. [Missing: Medium-Priority Gaps](#5-missing-medium-priority-gaps)
6. [Missing: Nice-to-Have Improvements](#6-missing-nice-to-have-improvements)
7. [Detailed Recommendations](#7-detailed-recommendations)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Executive Summary

The platform has a **solid foundation** with many production best practices already in place (GPU memory management, graceful shutdown, health checks, non-root container user, multi-stage Docker build, DCGM GPU monitoring, Caddy reverse proxy with HTTPS, auth middleware). However, several **critical gaps** remain that would prevent a production deployment from being reliable, secure, or observable:

| Area | Status | Priority |
|------|--------|----------|
| GPU Memory Management | ✅ Done (per-stage unloading) | — |
| Graceful Shutdown (SIGTERM) | ✅ Done (pipeline + main) | — |
| Health Check | ✅ Done (FastAPI `/health`) | — |
| Docker Multi-stage Build | ✅ Done (python:3.11-slim → nvidia/cuda:12.8) | — |
| Non-root User | ✅ Done | — |
| Auth (Gradio + FastAPI) | ✅ Done (env vars + HTTP Basic) | — |
| Caddy Reverse Proxy + HTTPS | ✅ Done (Caddyfile + docker-compose.prod.yml) | — |
| DCGM GPU Monitoring | ✅ Done (docker-compose.prod.yml) | — |
| **CI/CD Pipeline** | ❌ Missing | 🔴 Critical |
| **Pre-commit Hooks / Code Quality** | ❌ Missing | 🔴 Critical |
| **Rate Limiting** | ❌ Missing | 🔴 Critical |
| **Comprehensive Error Handling** | ⚠️ Partial | 🟠 High |
| **Input Validation** | ⚠️ Partial | 🟠 High |
| **Logging Improvements** | ⚠️ Partial | 🟠 High |
| **Graceful Shutdown (Uvicorn/FastAPI)** | ❌ Missing | 🟠 High |
| **Prometheus Metrics / Observability** | ❌ Missing | 🟠 High |
| **API Versioning** | ❌ Missing | 🟡 Medium |
| **CORS Configuration** | ⚠️ Partial | 🟡 Medium |
| **Security Headers in App** | ⚠️ Partial | 🟡 Medium |
| **.env.example / Docs** | ❌ Missing | 🟡 Medium |
| **Backup/Restore Strategy** | ❌ Missing | 🟡 Medium |
| **Tests (GPU-dependent)** | ⚠️ Partial | 🟡 Medium |
| **Docker Image Tagging/Versioning** | ❌ Missing | 🟢 Nice-to-have |
| **Gradio Level Parameter** | ❌ Missing | 🟢 Nice-to-have |
| **Multi-worker / Horizontal Scaling** | ❌ Missing | 🟢 Nice-to-have |

---

## 2. Already Implemented (What's Done)

### ✅ GPU Memory Management
- Sequential model loading/unloading via `_unload_model()` with `torch.cuda.empty_cache()` + `gc.collect()` + `torch.cuda.synchronize()`
- Pipeline calls `cleanup()` to free all GPU memory
- Config-driven feature toggles (action recognition, OCR, diarization, MLLM, etc.)

### ✅ Graceful Shutdown
- `__main__.py` registers `SIGTERM` + `SIGINT` handlers (`_signal_handler`)
- `VideoPipeline` also registers signal handlers with `_shutdown_requested` flag
- `_shutdown_event` threading.Event for CLI/cron termination

### ✅ Health Check
- FastAPI `/health` endpoint returning GPU status, model states, uptime, version
- Docker `HEALTHCHECK` with 120s start period (model loading)

### ✅ Docker
- Multi-stage build: `python:3.11-slim` (builder) → `nvidia/cuda:12.8.0-runtime-ubuntu22.04` (runtime)
- Non-root `video-analysis` user with `no-new-privileges:true`
- `mem_limit: 16g` with `memswap_limit: 4g`
- JSON-file logging with rotation (max-size 10m, max-file 3)
- `.dockerignore` properly excluding secrets, git, test/docs, data

### ✅ Authentication
- FastAPI middleware with HTTP Basic Auth (excludes `/health`)
- `GRADIO_USER` / `GRADIO_PASSWORD` env vars
- Auth documented in `production-security-best-practices.md`

### ✅ Production Compose (`docker-compose.prod.yml`)
- DCGM Exporter for NVIDIA GPU metrics (`:9400/metrics`)
- Caddy reverse proxy with automatic HTTPS, security headers, gzip, JSON logging
- Separate `video-analysis-net` Docker network
- Named volumes for persistence

### ✅ Caddyfile
- Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy`
- Gzip encoding, WebSocket support, JSON access logs

---

## 3. Missing: Critical Gaps

### 🔴 3.1 No CI/CD Pipeline
**Severity:** CRITICAL — no automated testing, linting, or deployment

The repository has **zero** CI/CD configuration. No `.github/workflows/`, no `.gitlab-ci.yml`, no `Makefile`, no pre-commit config.

**What's missing:**
- GitHub Actions (or equivalent) workflow for `push` and `PR` events
- Unit test runner (pytest with GPU-skipping markers for CI environments without GPUs)
- Docker image build + optional push to registry
- Linting (ruff, mypy) and formatting (black) checks
- Security scanning (bandit, trufflehog)
- Pre-commit hooks configuration

**Challenges specific to GPU repos:**
- GitHub-hosted runners don't have NVIDIA GPUs → tests must use `@pytest.mark.skipif(not torch.cuda.is_available())`
- Self-hosted runner with GPU passthrough for full integration tests
- Docker build with CUDA base images is large (~6 GB) → use Docker layer caching

**Recommended approach:**
```yaml
# .github/workflows/ci.yml structure:
# 1. lint (ruff + mypy) — runs on ubuntu-latest, no GPU
# 2. test-core (pytest with gpu-marked tests skipped) — ubuntu-latest
# 3. test-gpu (pytest with gpu-marked tests) — self-hosted GPU runner
# 4. docker-build (Docker build without push) — ubuntu-latest
# 5. docker-push (on tag/release) — needs registry creds
```

### 🔴 3.2 No Pre-commit Hooks / Code Quality Tools
**Severity:** CRITICAL — no automated code quality enforcement

**Missing:**
- `.pre-commit-config.yaml` with hooks for: ruff, mypy, trailing-whitespace, end-of-file-fixer, check-yaml, check-json, detect-private-key
- `pyproject.toml` is missing `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` sections
- No `ruff.toml` or `.flake8`
- No `Makefile` for common tasks (lint, test, build, clean)

### 🔴 3.3 No Rate Limiting
**Severity:** CRITICAL — no protection against abuse

The FastAPI app has no rate-limiting middleware. Both the Gradio UI and REST API endpoints (`/api/library`, `/api/video/{id}`) are unthrottled.

**What's needed:**
```python
# With slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    ...
```

**Also needed in docker-compose.prod.yml or Caddyfile:**
- Caddy v2.8+ rate_limit directive
- Or Nginx rate limiting if swapped

---

## 4. Missing: High-Priority Gaps

### 🟠 4.1 Comprehensive Error Handling
**Severity:** HIGH — inconsistent error handling across pipeline

**Current state (what's done):**
- Pipeline has try/except around PySceneDetect with FFmpeg fallback
- Fallback to 30s uniform chunks if all scene detection fails
- Graceful handling of missing optional deps (yt-dlp, paddleocr, pyannote, scenedetect)
- `_extract_ocr` handles missing PaddleOCR gracefully
- `_diarize` handles missing pyannote gracefully

**What's missing:**
- **No centralized error handler** in FastAPI app — every route handler does its own try/except
- `_check_models()` catches bare `Exception` — individual error types are lost
- No retry logic for transient failures (network/flaky GPU operations)
- No circuit breaker for flaky downstream dependencies (HuggingFace model downloads, yt-dlp)
- Pipeline `download_from_url()` returns `None` on failure — caller may not check
- Subprocess calls use `timeout` but some catch generic `Exception`

### 🟠 4.2 Input Validation
**Severity:** HIGH — insufficient validation of user-supplied paths/URLs

**Current state:**
- `parse_yt_url()` regex validates against known video site patterns
- Clip export validates `start >= end`
- Batch processing checks file existence

**What's missing:**
- **No path traversal protection** — `video_id` is used directly in file operations (e.g., `config.video_dir / f"{vid}.mp4"`). A malicious `video_id` like `../../etc/passwd` could escape the data directory
- **No file upload size limits** — Gradio's `gr.Video()` accepts files of any size; could OOM the container
- **No URL validation beyond regex** — yt-dlp is called without URL validation beyond basic pattern matching
- **No content-type verification** on uploaded files — `gr.Video()` accepts `.mp4` but the actual MIME type is not checked
- **No sanitization** of user-supplied filenames for batch processing

### 🟠 4.3 Logging Improvements
**Severity:** HIGH — inconsistent and non-structured logging

**Current state:**
- Standard `logging.basicConfig()` with basic format
- Docker JSON-file logging with rotation
- Caddy JSON access logs

**What's missing:**
- **Structured logging** (JSON format with request IDs, timestamps, levels)
- **Log levels are not configurable at runtime** — only via `--verbose` flag at startup
- **No request ID correlation** — cannot trace a single user request through pipeline, FastAPI, and Gradio
- **No audit log** — who uploaded/deleted what, when
- **No per-module log level control** (e.g., debug for pipeline, info for everything else)
- **Caddy logs go to file but app logs go to stdout** — no centralized log aggregation

### 🟠 4.4 Graceful Shutdown (Uvicorn/FastAPI)
**Severity:** HIGH — FastAPI/Gradio has no shutdown handler

**Current state:**
- Pipeline and `__main__.py` handle SIGTERM, but:
  - **No `lifespan` handler** on FastAPI — no `startup`/`shutdown` events registered
  - Gradio's internal Uvicorn server may not propagate shutdown signals cleanly
  - If Gradio is running a long inference, the process may be killed mid-stream
  - Models in GPU memory are not explicitly unloaded on shutdown
  - Ongoing `subprocess.run()` calls (ffmpeg) are not terminated

**What's needed:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown — runs on SIGTERM
    logger.info("Shutting down...")
    pipeline.cleanup()  # unload GPU models
    # Wait for ongoing operations
```

### 🟠 4.5 Prometheus Metrics / Observability
**Severity:** HIGH — no application-level metrics

**Current state:**
- DCGM Exporter provides GPU-level metrics (VRAM, temp, utilization) at `:9400/metrics`
- **But no application-level metrics** beyond Docker health check

**What's missing:**
- **Prometheus FastAPI middleware** (`pip install prometheus-fastapi-instrumentator`)
  - Request count, latency, error rate per endpoint
  - Request size, response size
  - Active requests gauge
- **Custom metrics:**
  - `video_analysis_videos_processed_total` (counter)
  - `video_analysis_pipeline_duration_seconds` (histogram per video)
  - `video_analysis_gpu_memory_allocated_bytes` (gauge)
  - `video_analysis_chat_queries_total` (counter)
  - `video_analysis_model_loading_duration_seconds` (histogram)
  - `video_analysis_vr_allocation` (gauge per model)
- **Prometheus target** configuration (scrape config)
- **Grafana dashboard** for visualization

---

## 5. Missing: Medium-Priority Gaps

### 🟡 5.1 API Versioning
**Current state:**
- `/api/library` and `/api/video/{id}` are unversioned
- Breaking changes would break consumers

**Recommendation:**
```python
from fastapi import APIRouter
v1_router = APIRouter(prefix="/api/v1")
v1_router.add_api_route("/library", api_library, methods=["GET"])
v1_router.add_api_route("/video/{video_id}", api_video, methods=["GET"])
app.include_router(v1_router)
```

### 🟡 5.2 CORS Configuration
**Current state:**
- No explicit CORS middleware in FastAPI app
- Caddy handles `Access-Control-*` headers via reverse proxy
- But if FastAPI is accessed directly (non-proxy), CORS is wide open

**Recommendation:**
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 🟡 5.3 Security Headers in App
**Current state:**
- Caddyfile sets security headers for reverse proxy
- But if accessed outside Caddy, headers are missing

**Recommendation:**
- Add `SecurityMiddleware` or headers middleware to FastAPI app as defense-in-depth

### 🟡 5.4 .env.example / Setup Documentation
**Current state:**
- No `.env.example` file
- Docker Compose files reference `${GRADIO_PASSWORD}`, `${DOMAIN}`, etc. but no example file shows required/optional vars
- No documentation on setting up secrets for first-time users

**Recommendation:**
- Create `.env.example` with all supported env vars and documentation
- Add a `First Time Setup` section to README referencing the `.env.example`

### 🟡 5.5 Backup/Restore Strategy
**Current state:**
- Data persists in `./data` volume (videos, frames, audio, chroma, thumbnails, clips)
- No backup documentation or scripts

**Recommendation:**
- Document backup procedures (ChromDB data + video files)
- Add a simple backup script (`scripts/backup.sh`)
- Consider ChromaDB export/import for portability

### 🟡 5.6 Tests — Limited GPU Coverage
**Current state:**
- 40+ unit tests covering config, models, UI utils, error handling
- Tests use temporary directories and clean up after themselves
- Sprite sheet test generates a real test video via FFmpeg
- Graceful fallback tests for missing optional dependencies

**What's missing:**
- **No GPU-dependent tests** — all tests run CPU-only or mock GPU
- **No integration tests** — no test that runs the full pipeline
- **No test for RAG indexing/retrieval** (requires chromadb)
- **No test for chat module**
- **No test for health API endpoints** (requires FastAPI TestClient)
- **No `pytest.ini` or `pyproject.toml` pytest config**

### 🟡 5.7 Docker Image Tagging/Versioning
**Current state:**
- Image is always tagged `video-analysis:latest`
- No semantic versioning in Docker tags
- No Docker image labels for build info (git commit, build date)

**Recommendation:**
- Tag images with both `v0.14.0` and `latest` during CI
- Add `--build-arg BUILD_DATE=$(date -Iseconds) --build-arg GIT_COMMIT=$(git rev-parse HEAD)`
- Add LABEL `org.opencontainers.image.revision` and `org.opencontainers.image.created`

---

## 6. Missing: Nice-to-Have Improvements

### 🟢 6.1 Gradio `level` Parameter
- Gradio 6's `level` parameter controls what features are shown (1=basic, 2=advanced, 3=expert)
- Not used; advanced features (like clip export, batch processing) are always visible
- Could simplify UX for first-time users

### 🟢 6.2 Multi-worker / Async Pipeline
- Pipeline is single-threaded, sequential
- Could use FastAPI `BackgroundTasks` or Celery for async processing
- Gradio queue can handle concurrent uploads but pipeline itself is synchronous
- Could use `ThreadPoolExecutor` for parallel operations within pipeline (e.g., run OCR + YOLO concurrently)

### 🟢 6.3 Immutable/Read-only Root Filesystem
- Docker security best practice: `read_only: true` for root filesystem
- Requires identifying all write paths and adding tmpfs mounts for them
- `tmpfs: /tmp` for runtime temp files

### 🟢 6.4 Container CPU/Memory Limits in Production
- docker-compose.yml has `mem_limit: 16g` with `memswap_limit: 4g`
- docker-compose.prod.yml has no resource limits at all
- Should add `cpus: "4"` or similar CPU limits

### 🟢 6.5 Dependency Pinning
- `requirements.txt` uses `>=` version specifiers
- For reproducible production builds, pin exact versions with hashes
- Can use `pip freeze > requirements-lock.txt` after testing
- Or use pip-tools (`pip-compile`)

---

## 7. Detailed Recommendations

### 7.1 CI/CD Pipeline (Critical)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:
    branches: [master, main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff mypy
      - run: ruff check . --ignore E501
      - run: mypy video_analysis --ignore-missing-imports

  test-core:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: |
          sudo apt-get install -y ffmpeg
          pip install -r requirements.txt
          pip install pytest
      - run: pytest tests/ -v -m "not gpu" --ignore=tests/test_gpu.py

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - run: docker build -t video-analysis:${{ github.sha }} .
      - run: docker images video-analysis
```

### 7.2 Pre-commit Configuration (Critical)

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
      - id: detect-private-key
  - repo: https://github.com/ambv/black
    rev: 25.1.0
    hooks:
      - id: black
```

### 7.3 Rate Limiting (Critical)

Add to `ui/health.py` in `create_health_app()`:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

Also add to Caddyfile:
```
rate_limit {
    zone dynamic {
        key {remote_host}
        events 10
        window 1m
    }
}
```

### 7.4 FastAPI Lifespan + Graceful Shutdown (High)

Replace current bare FastAPI constructor with lifespan-aware version in `ui/health.py`:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _start_time, _rag
    _start_time = time.time()
    _rag = VideoRAG(config)
    logger.info("Application started")
    yield
    # Shutdown
    logger.info("Shutting down application...")
    if _rag is not None:
        _rag._unload_bge_vl()
    # Signal all pipelines to stop
```

### 7.5 Prometheus Metrics (High)

Add to dependencies and wire in `create_health_app()`:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### 7.6 Path Traversal Protection (High)

Add validation for all user-supplied `video_id` values:

```python
import re

def validate_video_id(video_id: str) -> bool:
    """Prevent path traversal attacks."""
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', video_id))

def validate_filename(filename: str) -> bool:
    """Allow only safe filenames."""
    return not any(c in filename for c in ['..', '/', '\\', '\0'])
```

### 7.7 Upload Size Limits (High)

In `ui/app.py` or FastAPI config:

```python
# FastAPI
app = FastAPI()
app.max_request_size = 1024 * 1024 * 1024  # 1 GB

# Gradio — set file size limit
video_input = gr.Video(
    label="Upload video (max 2GB)",
    sources=["upload"],
    file_types=[".mp4", ".webm", ".mov"],
)
```

Or at the Docker/reverse-proxy level:
```
# Caddyfile
request_body {
    max_size 2GB
}
```

### 7.8 Structured Logging (High)

Replace `logging.basicConfig()` with structured logging:

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),  # dev
        # structlog.processors.JSONRenderer()  # production
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### 7.9 CORS Configuration (Medium)

In `ui/health.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "").split(",") if os.environ.get("CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7.10 .env.example (Medium)

Create `.env.example`:

```env
# === Required ===
GRADIO_PASSWORD=change_me_to_a_strong_password

# === Domain (for Caddy HTTPS) ===
DOMAIN=video-analysis.example.com

# === Optional: Features ===
GRADIO_USER=admin
ACTION_RECOGNITION_ENABLED=false
VIDEO_MLLM_ENABLED=false
MULTIMODAL_EMBEDDING=false

# === Optional: Model Config ===
WHISPER_MODEL=large-v3
EMBEDDING_MODEL=BAAI/BGE-VL-base

# === Optional: Security (CORS, rate limiting) ===
CORS_ORIGINS=https://video-analysis.example.com
```

---

## 8. Implementation Roadmap

### Phase 1 — Critical (Week 1)
1. `.pre-commit-config.yaml` + ruff/mypy configuration
2. `.github/workflows/ci.yml` — lint + test-core + docker-build
3. Rate limiting via slowapi (FastAPI) + Caddy rate_limit
4. `Makefile` with common tasks (lint, test, build, clean)

### Phase 2 — High Priority (Week 2)
5. FastAPI lifespan handler for graceful shutdown
6. Prometheus metrics integration
7. Path traversal protection + upload size limits
8. Structured logging (structlog or JSON)

### Phase 3 — Medium Priority (Week 3)
9. API versioning (`/api/v1/...`)
10. CORS middleware
11. `.env.example` + setup docs
12. Backup documentation/scripts
13. Additional pytest configuration and GPU test markers

### Phase 4 — Nice-to-Have (Week 4)
14. Docker image tagging with git SHA + semver
15. Read-only root filesystem + tmpfs mounts
16. CPU limits in docker-compose.prod.yml
17. Dependency pinning with pip-tools
18. Gradio level parameter for simplified UX

---

## Appendix: Key Files Referenced

| File | Purpose |
|------|---------|
| `video_analysis/__main__.py` | Entry point, CLI parsing, launch |
| `video_analysis/pipeline.py` | Core video processing pipeline (1479 lines) |
| `video_analysis/rag.py` | RAG engine with ChromaDB+BGE-VL (1195 lines) |
| `video_analysis/config.py` | Configuration dataclass (180 lines) |
| `video_analysis/chat.py` | LLM chat with RAG/MLLM backends |
| `ui/app.py` | Gradio UI (1798 lines) |
| `ui/health.py` | FastAPI health + API endpoints (273 lines) |
| `ui/utils.py` | Shared UI utilities (61 lines) |
| `tests/test_basic.py` | Unit tests (1330 lines) |
| `Dockerfile` | Multi-stage Docker build (125 lines) |
| `docker-compose.yml` | Dev compose (106 lines) |
| `docker-compose.prod.yml` | Production compose with DCGM+Caddy (83 lines) |
| `Caddyfile` | Caddy reverse proxy config (33 lines) |
| `docs/production-security-best-practices.md` | Existing security documentation (820 lines) |
