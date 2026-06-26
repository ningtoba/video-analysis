# Embedding Model Upgrade Research (v0.5.0)

## Current State
- **Model**: `BAAI/bge-small-en-v1.5` (384-dim, MTEB ~50)
- **Library**: sentence-transformers 2.5+
- **Limitation**: 384-dim limits retrieval quality; bge-small lacks multilingual support

## Upgrade Target: nomic-ai/nomic-embed-text-v1.5

| Feature | Value |
|---------|-------|
| **Dimensions** | 768 |
| **MTEB Score** | ~64 (vs 50 for bge-small) |
| **License** | Apache 2.0 |
| **Size** | ~700MB (sentence-transformers format) |
| **Self-hosted** | Yes — fully local, no API |
| **Prefixes** | `search_document` / `search_query` — requires instruction prefixes |

### Key Findings
1. Model needs `trust_remote_code=True` when loading via sentence-transformers
2. Requires instruction prefixing: encode documents with `search_document: {text}` and queries with `search_query: {text}`
3. The prefixing is critical for optimal retrieval — without it, quality degrades significantly
4. VRAM usage: ~1.2GB (similar to bge-small)
5. Inference speed: ~1000 docs/sec on RTX 4070

## Implementation Plan
1. Add `embedding_trust_remote_code` config flag (default: True)
2. Update RAG._get_embedding() to apply instruction prefixes
3. Update Config.embedding_model default to `nomic-ai/nomic-embed-text-v1.5`

## Alternative Considered
- **BAAI/bge-m3**: 1024-dim, M3-Embedding (multilingual, multi-dense, multi-vec), also Apache 2.0
  - Better multilingual but larger (2GB+), slower inference
  - Overkill for primarily English content
- **Alibaba-NLP/gte-Qwen3-Embedding-2**: Latest 2026 model, very strong MTEB
  - Requires Qwen dependencies (larger memory footprint)
  - More experimental

## Decision
Chose nomic-embed-text-v1.5 for:
- Best quality-per-parameter ratio among Apache 2.0 models
- Direct sentence-transformers support
- Well-documented instruction prefixing
- Proven in production RAG systems
