# Graph-Based Video RAG Research — June 2026

> **Date:** 2026-06-26
> **Context:** Research for video-analysis project roadmap item: "Graph-based video RAG (VGent/ViG-RAG inspired — scene-graph retrieval + K-hop expansion)"
> **Current RAG stack:** ChromaDB + BGE-VL multimodal embeddings + Cross-encoder re-ranking + TV-RAG temporal decay + Temporal context expansion (±1 neighbor scene)

---

## Table of Contents

1. [Key Papers & Their Concrete Contributions](#1-key-papers)
2. [Scene-Graph Retrieval: How It Works in Practice](#2-scene-graph-retrieval)
3. [The K-Hop Expansion Approach](#3-k-hop-expansion)
4. [Concrete Implementations & GitHub Repos](#4-implementations)
5. [Lightweight Alternatives That Work with ChromaDB](#5-lightweight-alternatives)
6. [Hybrid Architecture: ChromaDB + In-Memory Scene Graph](#6-hybrid-architecture)
7. [Implementation Plan for v0.13.0+](#7-implementation-plan)
8. [Graph DB Decision: Neo4j vs FalkorDB vs NetworkX](#8-graph-db-decision)
9. [Key Sources](#9-sources)

---

## 1. Key Papers & Their Concrete Contributions

### VGent — NeurIPS 2025 Spotlight (arXiv:2510.14032)

**Full title:** "VGent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding"

**Key innovation:** Represents videos as **structured graphs** where nodes are video clips/segments with semantic relationships preserved as edges. During retrieval, traverses the graph to find relevant clips and aggregates information across connected nodes via multi-step reasoning.

**Results:**
- +3.0% to +5.4% over base models on MLVU
- **Outperforms SOTA video RAG methods by +8.6%**
- NeurIPS 2025 Spotlight — top-tier paper selection

**How it works concretely:**
1. **Video graph construction:** Segment video into clips → extract entities (objects, people, actions, text) → build nodes per clip → connect nodes with semantic edges (entity co-occurrence, temporal adjacency, visual similarity)
2. **Graph-aware retrieval:** Query → embed → find seed nodes (top-k clips via similarity) → **K-hop expansion** along graph edges → aggregate node information
3. **Graph reasoning:** Multi-step reasoning path across the graph, combining evidence from connected clips
4. **Generation:** Augmented context → LLM for final answer

**Code:** GitHub (public, linked in paper) — cited as having 3.1k stars under the "VideoRAG" name (KDD 2026 paper sharing the same research group — HKU + Baidu).

### ViG-RAG — AAAI 2026 (#6 Ranked Paper)

**Full title:** "ViG-RAG: Video-aware Graph Retrieval-Augmented Generation via Temporal and Semantic Hybrid Reasoning"

**Key contribution:** Combines **temporal relationships** (before/after/overlap) with **semantic relationships** (entity co-occurrence, visual similarity) into a single hybrid graph structure.

**Graph edge types:**
| Edge Type | Description | How to Compute |
|-----------|-------------|----------------|
| **Temporal-before** | Scene A → Scene B (sequential) | Scene order in video |
| **Temporal-overlap** | Scene A ⟷ Scene C (overlapping time) | Shared frame ranges |
| **Entity co-occurrence** | Both scenes mention/detect same entity | YOLO labels, transcript NER, OCR text |
| **Visual similarity** | High CLIP/BGE-VL embedding similarity | Embedding cosine similarity |
| **Action transition** | Connected by action continuity | X-CLIP action labels |

**Concrete takeaway:** The temporal+semantic edge design is directly implementable on top of the existing pipeline outputs (YOLO objects, transcript, OCR, X-CLIP actions, BGE-VL embeddings).

### VideoRAG — KDD 2026 (arXiv:2502.01549)

**Full title:** "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos"

**Key innovations:**
- **Dual-channel architecture:** (1) Graph-based textual knowledge grounding for cross-video semantic relationships + (2) Multi-modal context encoding for visual features
- **Cross-video knowledge graphs** — not just per-video indexing
- **LongerVideos benchmark:** 160+ videos, 134+ hours across lecture/documentary/entertainment

**Relevance:** The graph-based **cross-video indexing** is the most advanced approach found. Builds KGs spanning multiple videos, enabling queries like "find related moments about topic X across my library."

**GitHub:** Links to code available at the paper URL. Paper cites ⭐ 3.1k GitHub stars.

### TV-RAG — ACM Multimedia 2025 (arXiv:2512.23483)

**Already implemented in v0.12.0** — temporal decay weighting, entropy-weighted key-frame selection, temporal window BM25.

### HAVEN — Microsoft Research Asia (arXiv:2601.13719, March 2026)

**SOTA on LVBench:** 84.1% overall, 80.1% reasoning

**Key contribution for graph RAG:**
- **Audiovisual entity cohesion** — cross-modal entity consolidation using speaker identity as "glue"
- **4-level hierarchical indexing:** global summary → scene → segment → entity
- **Entity tracking across scenes** — people, objects, concepts tracked through the video

**Entity-level indexing is the missing layer** in the current architecture. Adding entity nodes (people, key objects, locations) that link back to scenes is the core enabler for graph-based retrieval.

### SceneRAG — Scene-level RAG for Video Understanding (2025)

Scene-level retrieval-augmented generation. Not a full graph approach but validates the scene-as-retrieval-unit pattern.

---

## 2. Scene-Graph Retrieval: How It Works in Practice

### The Core Pattern

```
┌──────────────────────────────────────────────────────────────────┐
│                      GRAPH CONSTRUCTION                          │
│                                                                  │
│  Scenes (existing) → scene_id, start_time, end_time             │
│    ↓                                                             │
│  Entities (extracted) → person:, object:, action:, location:     │
│    ↓                                                             │
│  Build Graph:                                                   │
│    Node types: SCENE, ENTITY                                     │
│    Edges:   SCENE──HAS_ENTITY──→ENTITY                          │
│             SCENE──NEXT──→SCENE (temporal)                       │
│             SCENE──SIMILAR──→SCENE (semantic, score>0.85)        │
│             ENTITY──CO_OCCURS──→ENTITY (shared scene)            │
└──────────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                                    │
│                                                                  │
│  1. Query → Embed (BGE-VL) → ChromaDB similarity search          │
│     → Get top-k seed scenes                                      │
│  2. For each seed scene, expand via graph edges:                  │
│     - K-hop temporal neighbors (K=1: ±1 scene)                   │
│     - Shared entity neighbors (scenes sharing same person/object)│
│     - Semantic neighbors (similarity > threshold)                 │
│  3. Aggregate unique scenes from expansion                       │
│  4. Re-rank expanded set with cross-encoder                      │
│  5. Sort chronologically → LLM context                           │
└──────────────────────────────────────────────────────────────────┘
```

### Concrete Data Model (in ChromaDB metadata + in-memory graph)

**ChromaDB metadata (per chunk) — already has most of this:**
```python
{
    "video_id": "lecture_01",
    "scene_id": 3,
    "start_time": 45.2,
    "end_time": 68.7,
    "chunk_type": "scene",
    # NEW fields for graph:
    "entities": ["person:prof_smith", "object:whiteboard", "action:lecturing"],
    "entity_ids": ["ent_001", "ent_002", "ent_003"],
    "embedding": [...]  # already stored by ChromaDB
}
```

**In-memory scene graph (NetworkX):**
```python
import networkx as nx

G = nx.DiGraph()

# Scene nodes
G.add_node("scene_lecture01_03", type="scene", video_id="lecture_01",
           start_time=45.2, end_time=68.7, chunk_id="lecture_01_scene_0003")

# Entity nodes
G.add_node("ent_prof_smith", type="person", label="Prof. Smith")
G.add_node("ent_whiteboard", type="object", label="whiteboard")

# Edges
G.add_edge("scene_lecture01_03", "ent_prof_smith", relation="has_entity")
G.add_edge("scene_lecture01_03", "ent_whiteboard", relation="has_entity")
G.add_edge("scene_lecture01_02", "scene_lecture01_03", relation="next_scene")
G.add_edge("scene_lecture01_03", "scene_lecture01_04", relation="next_scene")

# Semantic similarity edge (score > threshold)
G.add_edge("scene_lecture01_03", "scene_lecture02_07", relation="semantic_similar", weight=0.91)
```

---

## 3. The K-Hop Expansion Approach

### Definition

K-hop expansion is the graph traversal strategy used by VGent/ViG-RAG: from a **seed node** (retrieved scene), follow graph edges for **K steps** to collect an expanded neighborhood of contextually related scenes.

### Why K-Hop Works for Video

| Hop Level | What It Captures | Example |
|-----------|------------------|---------|
| **1-hop temporal** | Adjacent scenes | "What happened right before/after?" |
| **1-hop entity** | Other scenes with same person/object | "Find all scenes with Prof. Smith" |
| **2-hop temporal** | ±2 scenes | Broader narrative context |
| **2-hop entity→entity** | Scenes with related entities | "Scenes with students (after finding Prof. Smith scene)" |
| **3-hop cross-video** | Related concepts across videos | "Find related moments across my library" |

### Algorithm

```python
def k_hop_expansion(G, seed_nodes, k=2, edge_types=None):
    """
    Expand from seed nodes K hops along specified edge types.
    
    Args:
        G: NetworkX graph
        seed_nodes: List of starting scene node IDs
        k: Number of hops
        edge_types: Filter to specific edge relations (None = all)
    
    Returns:
        Set of expanded scene node IDs
    """
    expanded = set(seed_nodes)
    frontier = set(seed_nodes)
    
    for hop in range(k):
        next_frontier = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in expanded:
                    # Check edge type filter
                    edge_data = G.get_edge_data(node, neighbor)
                    if edge_types is None or edge_data.get('relation') in edge_types:
                        next_frontier.add(neighbor)
        expanded.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    
    return expanded
```

### Practical Config

```python
# Proposed config fields
graph_rag_enabled: bool = False
graph_k_hops: int = 2
graph_edge_types: list = ["next_scene", "has_entity", "semantic_similar"]
graph_similarity_threshold: float = 0.85  # for semantic edges
graph_store_path: str = "data/scene_graph.graphml"  # persistence
```

### K=1 vs K=2 vs K=3 Tradeoff

| K | Context Added | Risk |
|---|---------------|------|
| 1 | Immediate neighbors | Safe, minimal noise |
| 2 | Broader narrative, entity-linked scenes | Moderate. May include loosely related content |
| 3 | Cross-video concepts | High. Risk of topic drift |

**Recommendation:** Start with K=2 for scene-internal queries, K=1 for time-sensitive queries. Make K configurable.

---

## 4. Concrete Implementations & GitHub Repos

### Known Repositories

| Repository | Stars | Paper | Notes |
|------------|-------|-------|-------|
| **VideoRAG** (HKAIR) | ⭐ 3.1k | KDD 2026 | Dual-channel graph + visual. Links from paper |
| **FlagEmbedding** (BAAI) | ⭐ 11.9k | BGE-VL | Already integrated; provides embedding backbone |
| **VideoChat-Flash** (OpenGVLab) | ⭐ 526 | ICLR 2026 | Video MLLM for scene description |
| **Awesome-RAG-Vision** | ⭐ Various | Survey | Comprehensive list of vision RAG papers |
| **Awesome-Scene-Graph-Generation** | ⭐ Various | Survey | Scene graph generation and application papers |

**Note:** Search results were unreliable at query time. Several papers' code links point to GitHub but search APIs may not index them directly. The papers themselves (especially VGent/VideoRAG) confirm code is publicly available.

### What's Available vs What's Missing

| Component | Available? | Source |
|-----------|-----------|--------|
| BGE-VL multimodal embedding | ✅ Integrated in v0.12.0 | FlagEmbedding |
| Scene detection (PySceneDetect) | ✅ Integrated | OSS |
| Object detection (YOLO) | ✅ Integrated | Ultralytics |
| Action recognition (X-CLIP) | ✅ Integrated | v0.12.0 |
| Temporal decay scoring (TV-RAG) | ✅ Integrated | v0.12.0 |
| Cross-encoder re-ranking | ✅ Integrated | sentence-transformers |
| **Scene graph construction** | ❌ New | Build on existing pipeline outputs |
| **Entity extraction** | ❌ New | NER on transcript + aggregate YOLO labels |
| **K-hop expansion retrieval** | ❌ New | NetworkX graph traversal |
| **Graph persistence** | ❌ New | Pickle/GraphML/Neo4j |
| Full VGent/ViG-RAG reproduction | ❌ Not needed | The concepts, not the exact paper |

---

## 5. Lightweight Alternatives That Work with ChromaDB

### Why NOT a Dedicated Graph DB (for v0.13.0-0.14.0)

| DB | RAM | Setup | Use Case |
|----|-----|-------|----------|
| **Neo4j** | ~2-4 GB | Docker, Cypher | Production graph DB, overkill for <100 videos |
| **FalkorDB** | ~1-2 GB | Docker, Redis-like | Fast, but extra dependency |
| **NetworkX** | In-process | pip install | **Perfect for prototyping** — zero infrastructure |
| **ChromaDB metadata only** | In-process | Already have it | Limited: no graph traversal, but works for 1-hop entity expansion |

### Recommended: NetworkX + ChromaDB Metadata (Hybrid)

**Why NetworkX wins for this use case:**
1. **Zero infrastructure** — `pip install networkx` only
2. **In-process** — no Docker, no server, no networking
3. **Serializable** — save/load via `.graphml` or pickle
4. **Scale-appropriate** — for <10K scenes across ~100 videos, NetworkX handles this trivially
5. **Efficient K-hop traversal** — built-in `nx.single_source_shortest_path_length()` and BFS
6. **Works alongside ChromaDB** — ChromaDB remains the primary retrieval engine; NetworkX adds graph-aware expansion on top

### Hybrid Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. BGE-VL embed query                                       │
│ 2. ChromaDB similarity search → top-30 chunks                │
│ 3. Extract scene IDs from results → these are "seed nodes"   │
│ 4. Look up seed nodes in NetworkX scene graph                │
│ 5. K-hop expansion along temporal + entity + semantic edges  │
│ 6. Collect unique scene node IDs → fetch their chunks        │
│    from ChromaDB (by chunk_id)                               │
│ 7. Merge original + expanded chunks                          │
│ 8. Cross-encoder re-rank                                     │
│ 9. Sort chronologically → LLM context                        │
└─────────────────────────────────────────────────────────────┘
```

### When to Graduate to a Dedicated Graph DB

| Scale | Solution |
|-------|----------|
| <10 videos, <500 scenes | NetworkX only |
| <100 videos, <10K scenes | NetworkX + optimized serialization |
| 100-1000 videos, <100K scenes | **FalkorDB** (lightweight, Redis-backed) |
| 1000+ videos, 1M+ nodes | **Neo4j** or **ArangoDB** |

---

## 6. Hybrid Architecture: ChromaDB + In-Memory Scene Graph

### Entity Extraction Pipeline (New Step)

Leverage **existing pipeline outputs** — no new models needed:

| Existing Output | Extracted Entities |
|-----------------|-------------------|
| YOLO detections | `object:truck`, `object:person`, `object:dog` |
| PaddleOCR text | `text:OPEN`, `text:EXIT` |
| X-CLIP actions | `action:walking`, `action:cooking` |
| Faster-Whisper transcript | `person:John` (via NER: spacy or regex) |
| PyAnnote diarization | `speaker:SPEAKER_00` |
| Scene description (OpenCLIP) | `scene_type:classroom` |

### SceneGraph class design

```python
# New file: video_analysis/scene_graph.py

@dataclass
class Entity:
    id: str
    label: str
    entity_type: str  # person, object, action, location, text, scene_type
    aliases: list[str] = None
    embedding: list[float] = None  # optional BGE-VL embedding

@dataclass
class SceneNode:
    scene_id: int
    video_id: str
    start_time: float
    end_time: float
    chunk_id: str
    entities: list[Entity]
    embedding: list[float]

class SceneGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.video_registry: dict[str, list[str]] = {}  # video_id → [node_ids]
    
    def add_scene(self, scene: SceneNode):
        """Add scene node with entity edges."""
        node_id = f"{scene.video_id}_scene_{scene.scene_id:04d}"
        self.graph.add_node(node_id, type="scene", **scene.__dict__)
        self.video_registry.setdefault(scene.video_id, []).append(node_id)
        
        # Entity edges
        for entity in scene.entities:
            ent_id = entity.id
            if not self.graph.has_node(ent_id):
                self.graph.add_node(ent_id, type=entity.entity_type, label=entity.label)
            self.graph.add_edge(node_id, ent_id, relation="has_entity")
    
    def add_temporal_edges(self, video_id: str):
        """Link consecutive scenes."""
        nodes = sorted(self.video_registry.get(video_id, []))
        for i in range(len(nodes) - 1):
            self.graph.add_edge(nodes[i], nodes[i+1], relation="next_scene")
    
    def add_semantic_edges(self, threshold: float = 0.85):
        """Add edges between semantically similar scenes."""
        scene_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get("type") == "scene"]
        for i, n1 in enumerate(scene_nodes):
            for n2 in scene_nodes[i+1:]:
                emb1 = self.graph.nodes[n1].get("embedding")
                emb2 = self.graph.nodes[n2].get("embedding")
                if emb1 is not None and emb2 is not None:
                    sim = cosine_similarity(emb1, emb2)
                    if sim > threshold:
                        self.graph.add_edge(n1, n2, relation="semantic_similar", 
                                           weight=sim)
    
    def k_hop_expand(self, seed_nodes: list[str], k: int = 2) -> set[str]:
        """Get all scene nodes within K hops of seed nodes."""
        expanded = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(k):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in expanded:
                        next_frontier.add(neighbor)
            expanded.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return expanded
    
    def get_scene_chunks(self, expanded_nodes: set[str]) -> list[str]:
        """Get ChromaDB chunk IDs for expanded scene nodes."""
        chunk_ids = []
        for node_id in expanded_nodes:
            data = self.graph.nodes[node_id]
            if data.get("type") == "scene" and data.get("chunk_id"):
                chunk_ids.append(data["chunk_id"])
        return chunk_ids
    
    def save(self, path: str):
        nx.write_graphml(self.graph, path)
    
    @classmethod
    def load(cls, path: str) -> "SceneGraph":
        sg = cls()
        sg.graph = nx.read_graphml(path)
        # Rebuild video_registry from graph
        for n, d in sg.graph.nodes(data=True):
            if d.get("type") == "scene" and "video_id" in d:
                sg.video_registry.setdefault(d["video_id"], []).append(n)
        return sg
```

### Integration into Existing `retrieve()` Flow

```python
def retrieve(self, query, video_id=None, top_k=None, query_time=None):
    # Step 1-2: Existing ChromaDB + TV-RAG (unchanged)
    chunks = self._chroma_retrieve(query, video_id, top_k, query_time)
    
    # Step 3-5: Graph expansion (NEW, gated by config)
    if self.config.graph_rag_enabled and self._scene_graph is not None:
        seed_scene_ids = [c.chunk_id for c in chunks if c.scene_id >= 0]
        expanded_nodes = self._scene_graph.k_hop_expand(
            seed_scene_ids, k=self.config.graph_k_hops
        )
        # Fetch expanded chunks from ChromaDB (not in original results)
        extra_chunk_ids = [
            n for n in expanded_nodes 
            if n not in seed_scene_ids
        ]
        if extra_chunk_ids:
            extra = self.collection.get(
                ids=extra_chunk_ids,
                include=["documents", "metadatas"]
            )
            # Add to chunks list with discounted scores
            for i, cid in enumerate(extra["ids"]):
                meta = extra["metadatas"][i]
                chunks.append(RetrievedChunk(
                    chunk_id=cid, 
                    video_id=meta.get("video_id", ""),
                    text=extra["documents"][i],
                    timestamp=meta.get("start_time", 0),
                    scene_id=meta.get("scene_id", -1),
                    score=0.5,  # graph-expanded, intentionally discounted
                    metadata=meta,
                ))
    
    # Step 6-7: Re-rank + sort (existing)
    chunks = self._rerank(query, chunks, top_k)
    result.sort(key=lambda c: c.timestamp)
    return result
```

---

## 7. Implementation Plan for v0.13.0+

### Phase 1: Scene Graph Construction (New module)

| # | Task | Effort | Dependencies |
|---|------|--------|-------------|
| 1 | Create `video_analysis/scene_graph.py` — SceneGraph class with NetworkX | 1.5h | networkx |
| 2 | Add entity extraction from existing pipeline outputs (YOLO, OCR, X-CLIP, transcript NER) | 1h | spacy (optional) |
| 3 | Build scene graph during `index_video()` | 1h | Phase 1+2 |
| 4 | Persist graph to disk (GraphML) alongside ChromaDB | 15min | Phase 3 |
| 5 | Config fields: `graph_rag_enabled`, `graph_k_hops`, `graph_edge_types` | 5min | — |

### Phase 2: Graph-Aware Retrieval

| # | Task | Effort | Dependencies |
|---|------|--------|-------------|
| 6 | Implement `k_hop_expand()` in SceneGraph | 30min | Phase 1 |
| 7 | Modify `retrieve()` to optionally use graph expansion | 1h | Phase 1+2 |
| 8 | Add semantic similarity edges (BGE-VL embedding comparison) | 30min | Phase 1 |
| 9 | Handle cross-video expansion when query has no `video_id` filter | 30min | Phase 2 |

### Phase 3: Entity-Level Indexing

| # | Task | Effort | Dependencies |
|---|------|--------|-------------|
| 10 | NER on transcript (spacy or regex for people/places) | 30min | spacy |
| 11 | Entity node deduplication across scenes (same person with different names) | 1h | Phase 1 |
| 12 | Entity embedding for direct entity-to-entity similarity edges | 30min | BGE-VL (existing) |

### Config Additions

```python
# In config.py
graph_rag_enabled: bool = False
graph_k_hops: int = 2
graph_similarity_threshold: float = 0.85
graph_store_path: str = "data/scene_graph.graphml"
graph_edge_types: list = None  # None = all types
graph_entity_extraction: bool = True
```

---

## 8. Graph DB Decision

### Recommendation for v0.13.0: NetworkX (In-Process, No Server)

| Criterion | NetworkX | ChromaDB-only | Neo4j | FalkorDB |
|-----------|----------|---------------|-------|----------|
| **Setup time** | 5 min (pip install) | Already there | 30 min (Docker + Cypher) | 15 min (Docker) |
| **RAM overhead** | ~100 MB for 10K nodes | 0 GB | ~2-4 GB | ~1-2 GB |
| **K-hop traversal** | ✅ O(n log n) | ❌ Not possible | ✅ Optimized | ✅ Fast |
| **Persistence** | GraphML/pickle | N/A | Native | Redis |
| **Query language** | Python native | ChromaDB metadata filter | Cypher | Redis-like |
| **Scalability** | <100K nodes | <10K scenes via metadata | 1B+ nodes | 10M+ nodes |
| **Entity dedup** | ✅ In-process de-duplication | ❌ No | ✅ Complex queries | ✅ Basic |
| **Cross-video queries** | ✅ Yes (all in one graph) | ❌ Per-video filter | ✅ Yes | ✅ Yes |

**Verdict:** NetworkX for v0.13.0-0.14.0 (prototyping, <100 videos). Graduate to **FalkorDB** if scaling beyond 100+ videos where the in-memory NetworkX graph becomes unwieldy. Neo4j is overkill for this scale.

---

## 9. Key Sources

1. **VGent** (NeurIPS 2025 Spotlight) — arXiv:2510.14032. Graph-based retrieval-reasoning for long video. +8.6% over SOTA on MLVU.
2. **ViG-RAG** (AAAI 2026, #6) — Hybrid temporal+semantic graph reasoning. Combines temporal and entity-based edges.
3. **VideoRAG** (KDD 2026) — arXiv:2502.01549. Dual-channel graph + visual for extreme long-context video. ⭐ 3.1k stars.
4. **TV-RAG** (ACM Multimedia 2025) — arXiv:2512.23483. Temporal decay scoring (already implemented in v0.12.0).
5. **HAVEN** (Microsoft Research Asia, March 2026) — arXiv:2601.13719. SOTA on LVBench (84.1%). Entity-level indexing + hierarchical retrieval.
6. **SceneRAG** (2025) — Scene-level RAG for video understanding.
7. **NetworkX** — https://networkx.org/ — In-process graph library for Python (BSD license).
8. **FalkorDB** — https://www.falkordb.com/ — Lightweight graph DB (Redis-backed, good for 100-1000 video scale).
9. **Neo4j** — https://neo4j.com/ — Production graph DB (overkill for current scale).
10. **Awesome-RAG-Vision** — https://github.com/zhengxuJosh/Awesome-RAG-Vision — Comprehensive RAG-vision survey.

---

## Summary

**The graph-based video RAG technique is well-understood and implementable:**
- **Scene graph construction** reuses the existing pipeline outputs (YOLO objects, OCR text, X-CLIP actions, transcript) — no new models needed
- **K-hop expansion** is a straightforward BFS graph traversal (implementable in ~30 lines with NetworkX)
- **ChromaDB remains the primary retrieval engine** — the scene graph is a supplemental layer on top, not a replacement
- **NetworkX is the right graph library** for v0.13.0 scale (<100 videos, <10K scenes) — zero infrastructure, in-process, serializable
- **No dedicated graph DB needed yet** — graduate to FalkorDB if/when scaling beyond 100+ videos
- **Entity extraction** is the key enabler: person/object/action NER from existing pipeline outputs

**Estimated total effort for full implementation: ~6-7 hours** across 3 phases (graph construction → graph-aware retrieval → entity-level indexing).
