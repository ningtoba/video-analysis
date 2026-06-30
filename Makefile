# ─────────────────────────────────────────────────────────────────────────────
# Video Analysis Platform — Makefile
# ─────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash
.ONESHELL:

# Detect OS for GPU flags
UNAME_S := $(shell uname -s)
HAS_CUDA := $(shell python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo False)

# ── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Install ──────────────────────────────────────────────────────────────────
.PHONY: install install-dev install-all install-default

install: ## Install core dependencies only
	pip install --upgrade pip
	pip install -e .

install-default: ## Install core + API + UI + streaming (recommended for most users)
	pip install --upgrade pip
	pip install -e ".[default]"

install-dev: ## Install core + dev extras
	pip install --upgrade pip
	pip install -e ".[dev]"

install-all: ## Install everything (all extras)
	pip install --upgrade pip
	pip install -e ".[all,dev]"

install-gpu: ## Install with GPU-optimized extras (CUDA)
	pip install --upgrade pip
	pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[all,dev]"

# ── Docker ───────────────────────────────────────────────────────────────────
.PHONY: docker-build docker-run

docker-build: ## Build Docker image
	DOCKER_BUILDKIT=1 docker build -t video-analysis:latest .

docker-run: docker-build ## Build and run Docker container with GPU
	docker run --gpus all \
		-p 7860:7860 \
		-v $(CURDIR)/data:/app/data \
		-e CUDA_VISIBLE_DEVICES=0 \
		video-analysis:latest

docker-run-cpu: docker-build ## Build and run Docker container (CPU only)
	docker run \
		-p 7860:7860 \
		-v $(CURDIR)/data:/app/data \
		video-analysis:latest

# ── Type Checking ────────────────────────────────────────────────────────────
.PHONY: typecheck mypy

typecheck: mypy  ## Run static type checking
mypy:  ## Run mypy type checker
	mypy video_analysis --ignore-missing-imports

# ── Linting & Formatting ─────────────────────────────────────────────────────
.PHONY: lint format check

lint: ## Run ruff linter
	ruff check video_analysis/ tests/

format: ## Run ruff formatter
	ruff format video_analysis/ tests/

check: format lint typecheck  ## Run all checks (format + lint + typecheck)

# ── Testing ──────────────────────────────────────────────────────────────────
.PHONY: test test-fast test-cov test-bench test-gpu

test: ## Run all tests (excluding GPU, benchmark, slow)
	python -m pytest tests/ \
		-m "not gpu and not benchmark and not slow" \
		-x --timeout=120

test-fast: ## Run fast unit tests only
	python -m pytest tests/ \
		-m "not gpu and not benchmark and not slow and not integration" \
		-x --timeout=60

test-cov: ## Run tests with coverage report
	python -m pytest tests/ \
		-m "not gpu and not benchmark and not slow" \
		--cov=video_analysis \
		--cov-report=term \
		--cov-report=html \
		-x --timeout=120

test-all: ## Run all tests (including slow, excluding GPU/benchmark)
	python -m pytest tests/ \
		-m "not gpu and not benchmark" \
		-x --timeout=300

test-gpu: ## Run only GPU-requiring tests
	python -m pytest tests/ \
		-m "gpu" \
		-x --timeout=600

test-bench: ## Run benchmark tests
	python -m pytest tests/ \
		-m "benchmark" \
		--benchmark-only \
		--benchmark-save=baseline

# ── Run ──────────────────────────────────────────────────────────────────────
.PHONY: run run-cli run-api run-stream run-live run-mcp

run: ## Launch the web UI (Gradio or FastAPI based on config)
	python -m video_analysis

run-cli: ## Process a video in CLI mode (usage: make run-cli VIDEO=path.mp4 [QUERY="question"])
	python -m video_analysis --cli --video "$(VIDEO)"$(and $(QUERY), --query "$(QUERY)")

run-api: ## Start FastAPI server with uvicorn (usage: make run-api [HOST=0.0.0.0] [PORT=7860])
	uvicorn ui.server:create_app --factory \
		--host $(or $(HOST),0.0.0.0) \
		--port $(or $(PORT),7860) \
		--reload \
		--log-level info

run-stream: ## Stream-process a video (usage: make run-stream VIDEO=path.mp4 [CHUNK=30])
	python -m video_analysis --stream --video "$(VIDEO)" --chunk-duration $(or $(CHUNK),30)

run-live: ## Watch a file being written and process live (usage: make run-live VIDEO=recording.mp4)
	python -m video_analysis --live --video "$(VIDEO)"

run-mcp: ## Start the MCP server (stdio mode — for AI agents / IDE integration)
	python -m video_analysis.mcp_server --stdio

run-mcp-http: ## Start the MCP server (HTTP SSE mode)
	python -m video_analysis.mcp_server --port $(or $(PORT),8081)

# ── Development ──────────────────────────────────────────────────────────────
.PHONY: clean clean-all venv pre-commit

venv: ## Create a Python virtual environment
	python3 -m venv .venv
	@echo "Run: source .venv/bin/activate"

pre-commit: ## Install pre-commit hooks
	pre-commit install

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Clean temporary files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	rm -rf .benchmarks/ 2>/dev/null || true
	@echo "✓ Cleaned project artifacts"

clean-all: clean ## Deep clean including data directory (caution!)
	rm -rf data/ 2>/dev/null || true
	@echo "✓ Cleaned all (including data)"
