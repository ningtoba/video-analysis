# Engineering Backlog

> Persistent living document tracking all improvement opportunities.
> Updated continuously as new work is discovered.
> Last refreshed: 2026-07-02

---

## Priority Legend

| Priority | Meaning |
|----------|---------|
| P0       | Critical — blocks correctness, security, or deployment |
| P1       | High — significant quality, maintainability, or UX impact |
| P2       | Medium — worthwhile improvement, clear ROI |
| P3       | Low — nice-to-have, cosmetic, or speculative |
| P4       | Backlog — researched but not yet prioritized |

---

## High Priority (P0-P1)

### P1-001: ✅ Python CI Workflow (DONE)

- **Category:** CI/CD
- **Impact:** High — now runs on every push/PR
- **Status:** ✅ **Done** — `.github/workflows/ci.yml` with ruff lint + pytest (non-GPU, non-slow)
- **Completed:** Iteration 1

### P1-003: ✅ classifier.py Dead Code Removal (DONE)

- **Category:** Cleanup / Dead Code
- **Impact:** Removed 1,173 lines of orphaned code + 505 lines of dead tests
- **Status:** ✅ **Done** — entire module was not imported by any production code
- **Completed:** Iteration 3

### P2-001: ✅ rate_limiter.py Removal (DONE)

- **Category:** Dead Code Removal
- **Impact:** Removed 180 lines never wired to any endpoint + 130 lines of tests
- **Status:** ✅ **Done**
- **Completed:** Iteration 3

### P2-003: ✅ Logging Env Var Fallback Fix (DONE)

- **Category:** Bug Fix
- **Impact:** `STRUCTURED_LOGGING_LEVEL` and `STRUCTURED_LOGGING_FORMAT` env vars now work as documented
- **Status:** ✅ **Done**
- **Completed:** Iteration 3

### P2-006: ✅ Lazy CUDA Detection (DONE)

- **Category:** Reliability
- **Impact:** Health checks no longer crash at startup if NVIDIA driver is flaky
- **Status:** ✅ **Done**
- **Completed:** Iteration 3

### P1-002: Dual UI System — Bootstrap + Alpine.js Fragmentation

- **Category:** Architecture
- **Impact:** High — two competing UI paradigms; CDN violations; maintenance burden
- **Effort:** Medium
- **Dependencies:** None
- **Status:** Analyzed
- **Reasoning:**
  - `ui/templates/base.html` — Alpine.js + HTMX SPA shell (tabs, inline styles, no CDN)
  - `ui/templates/index.html` — Bootstrap 5 page (CDN)
  - `ui/templates/settings.html` — Bootstrap 5 page (CDN)
  - `ui/templates/stream.html` — Bootstrap 5 page (CDN)
  - `ui/templates/pages/` — 12 more Bootstrap-5 pages (most unused)
  - Contradicts project principle: "No CDN dependencies, inline CSS, works air-gapped"
  - Should consolidate to a single paradigm (preferably the Alpine/HTMX shell since it's already CDN-free)
- **Files:**
  - `ui/templates/base.html`
  - `ui/templates/index.html`
  - `ui/templates/settings.html`
  - `ui/templates/stream.html`
  - `ui/templates/pages/*.html`
  - `ui/server.py`

### P1-003: classifier.py — 1173 Lines of Dead Code from Old Multi-Model Pipeline

- **Category:** Cleanup / Dead Code
- **Impact:** High — massive file with ML classifier code, MobileNet, ImageNet mapping, scene type heuristics. **Entire module is orphaned** — `classifier.py` is not imported by any production module (verified: zero imports in `video_analysis/` or `ui/`). `VideoPipeline` never references it.
- **Effort:** Low (~20 min to delete + trim test file)
- **Dependencies:** None
- **Status:** Verified — **entire module is dead code**
- **Reasoning:** The refactoring replaced all local vision models with LLM Vision API. `classifier.py` retains 1173 lines including:
  - MobileNet v3 ML classifier singleton (`_ML_CLASSIFIER`, `get_ml_classifier()`) — gated behind `use_ml=False` default, never enabled
  - `classify_frame_with_ml()` — unused
  - `_classify_first_frame()` — unused
  - `_build_scene_mapping()` — ImageNet class-to-label mapping (318 lines)
  - `DEFAULT_STAGE_MAP` — old pipeline stage selection
  - `classify_file()`, `pipeline_skipped_stages()`, `classify_by_extension()`, `sniff_with_ffprobe()` — all defined but **never imported** in any production code path
  - `classifier` not referenced in `pipeline.py` at all
  - Test file `test_classifier.py` (505 lines) tests this dead code
- **Files:**
  - `video_analysis/classifier.py`
  - `tests/test_classifier.py`
### P1-006: No Stream Engine Tests

- **Category:** Testing
- **Impact:** High — stream engine (8 files, 1,221 lines) has zero test coverage
- **Effort:** Medium
- **Status:** Analyzed
- **Reasoning:** Stream engine is the newer subsystem. Core components (source, sampler, motion, analyzer, store, engine, chat, manager) have no tests.
- **Files:**
  - `video_analysis/stream/*.py`
  - `video_analysis/stream_manager.py`
  - `video_analysis/yolo_detector.py`
  - `video_analysis/event_memory.py`

---

## Medium Priority (P2)

### P2-001: rate_limiter.py Never Wired Into Any Endpoint

- **Category:** Dead Code / Instrumentation
- **Impact:** Low — works correctly but does nothing
- **Effort:** Low
- **Status:** Analyzed
- **Reasoning:** `TokenBucketLimiter` is instantiatable but never called from any route handler. The module is imported nowhere. Either wire it into critical endpoints or remove it.
- **Files:** `video_analysis/rate_limiter.py`

### P2-002: CHANGELOG.md is 193KB — Bloated History

- **Category:** Maintainability
- **Impact:** Low — large file in repo but excluded from Docker build
- **Effort:** Low
- **Status:** Analyzed
- **Reasoning:** 193KB changelog with extensive historical entries. Already in `.dockerignore` but still in repo.
- **Files:** `CHANGELOG.md`

### P2-003: _LogConfig.filter_level Never Set — Always INFO

- **Category:** Bug / Dead Configuration
- **Impact:** Low — logging works but filter level configuration is dead
- **Effort:** Very low
- **Status:** Analyzed
- **Reasoning:** `_LogConfig.filter_level` is defined but never written. `_simple_filter_by_level` always compares against the default (0 = DEBUG).
- **Files:** `video_analysis/logging_setup.py`

### P2-004: Video Source Type Helpers Missing from CLI

- **Category:** Developer Experience
- **Impact:** Low
- **Effort:** Low
- **Status:** Analyzed
- **Reasoning:** `--watch` accepts `--source {rtsp,webcam,file}` but no `--help` describes valid source types or defaults.
- **Files:** `video_analysis/__main__.py`

### P2-005: Dockerfile Copies Entire scripts/ Dir but Only init.sh Exists

- **Category:** Build
- **Impact:** Low
- **Effort:** Very low
- **Status:** Analyzed
- **Reasoning:** Minor — copies a dir with one file. Cleanup opportunity.
- **Files:** `Dockerfile`

### P2-006: health.py Inline CUDA Check at Module Load

- **Category:** Reliability
- **Impact:** Low
- **Effort:** Low
- **Status:** Analyzed
- **Reasoning:** `detect_cuda()` called at module import time in `add_health_endpoints()`. If CUDA/NVIDIA driver is flaky, importing the module fails, breaking health endpoints. Should be lazy.
- **Files:** `ui/health.py`

### P2-007: Manual CLI arg parsing instead of e.g. argparse subcommands

- **Category:** Developer Experience
- **Impact:** Low
- **Effort:** Low
- **Status:** Analyzed
- **Reasoning:** Main entry point uses argparse with `--cli`, `--watch`, `--url` modes but the arg parsing is flat (not subcommands). Mode selection is manual string comparison. Minor DX issue.
- **Files:** `video_analysis/__main__.py`

---

## Lower Priority (P3-P4)

### P3-001: Docker Compose Port Mapping Could Be Configurable

- **Category:** Deployment
- **Effort:** Low
- **Status:** Research
- **Reasoning:** Port 7860 hardcoded in compose and Dockerfile. Could expose via env var.

### P3-002: No Health Check in Docker Compose for `restart: unless-stopped`

- **Category:** Deployment
- **Impact:** Low — health check is in Dockerfile, compose doesn't reference it
- **Status:** Verified — Dockerfile has HEALTHCHECK, compose does not override
- **Reasoning:** This is actually fine — Dockerfile HEALTHCHECK works from compose.

### P3-003: .dockerignore Excludes All .md Files Including README.md

- **Category:** Build
- **Impact:** Very low
- **Status:** Analyzed
- **Reasoning:** `*.md` in `.dockerignore` means README.md is not in build context. It's not needed at runtime, but notable.

### P3-004: No Type Hints in Some UI Templates JS

- **Category:** Maintainability
- **Effort:** Low
- **Status:** Research

### P3-005: ffmpeg/ffprobe Subprocess Calls Without Timeout Propagation

- **Category:** Reliability
- **Effort:** Low
- **Status:** Analyzed
- **Reasoning:** Some subprocess calls in pipeline.py use `subprocess.run()` without explicit timeout; they rely on module-level constants but some code paths may bypass.

### P3-006: Consider `ruff check --fix` as Pre-Commit or CI Step

- **Category:** Developer Experience
- **Effort:** Very low
- **Status:** Research

### P3-007: stale `custom_data/` and `mydata/` directories in git

- **Category:** Cleanup
- **Effort:** Very low
- **Status:** Needs verification — already in `.gitignore`

---

## Task Selection History

| Iteration | Selected Task | Category | Outcome |
|-----------|--------------|----------|---------|
| 1         | ✅ P1-001: CI workflow + pyproject deps | CI/CD | Done — `.github/workflows/ci.yml` created |
| 2         | ✅ Fix 7 broken test files | Testing | Done — all test imports fixed |
| 2a        | ✅ Fix test quality/storage/llm/api/jobq/basic | Testing | Done — 124 tests passing |
| 3         | ✅ P1-003: Remove classifier.py | Cleanup | Done — 1,173+505 lines removed |
| 3a        | ✅ P2-001: Remove rate_limiter.py | Cleanup | Done — 180+130 lines removed |
| 3b        | ✅ P2-003: Fix logging env var fallback | Bug | Done — env vars now work |
| 3c        | ✅ P2-006: Lazy CUDA detection | Reliability | Done — per-request health checks |
| 3d        | ✅ Remove orphaned page templates | Cleanup | Done — 12 Bootstrap files removed |
| —         | **Completed in /loop 20** | | |
| 4         | ✅ P2-002: CHANGELOG trimming | Maint | Done — 198KB→5KB |
| 4a        | ✅ P2-005: Dockerfile scripts copy fix | Build | Done — COPY single file |
| 5         | ✅ P1-006: Stream engine tests (6 files) | Testing | Done — 135 new tests (259 total) |
| 6         | ✅ P1-002: UI CDN removal | Architecture | Done — 4 pages converted from Bootstrap→dark-theme.css |
| —         | **Remaining** | | |
| —         | P2-004: CLI help improvements | DX | --watch --source type missing |
| —         | P3-004 through P3-007 | Various | Lower priority |
