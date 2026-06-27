# Multi-Modal Knowledge Graphs, Video Scene Graph Generation & Persistent Video Understanding: Advances Through Mid-2026

> **Research compiled:** 2026-06-27
> **Context:** video-analysis v0.54.0 — next-level capabilities for knowledge_graph.py, scene_graph.py
> **Target hardware:** NVIDIA RTX 4070 (12 GB VRAM)
> **Current stack:** SQLite-backed KnowledgeGraph, ChromaDB + NetworkX SceneGraph (VGent/ViG-RAG), BGE-VL embeddings, InternVideo3-8B MLLM, YOLO26, ByteTrack, InsightFace, ColBERT-Att, TV-RAG temporal decay, Robust-TO confidence, HiCrew-inspired multi-agent orchestrator

---

## Executive Summary

This document surveys the latest advances (2025 – mid-2026) across **video scene graph generation (VidSGG), spatio-temporal scene graphs, multi-modal knowledge graphs for video, graph-based video reasoning, entity linking, event segmentation, and persistent video understanding.** Each technique is evaluated for integration into the existing video-analysis project with a hard constraint of RTX 4070 12GB VRAM.

### Priority Recommendations by Capability Area

| Area | Technique | Paper | Effort | VRAM | Impact |
|------|-----------|-------|--------|------|--------|
| **Scene Graph Generation** | DSFlash (panoptic SGG, 56 FPS) | CVPR 2026 | 3-4 days | 4-6 GB | 🔴 Monitor (code TBD) |
| **Scene Graph Generation** | NL-VSGG (weakly supervised VidSGG) | ICLR 2025 | 2-3 days | 0 (LLM-based) | 🟢 Ready |
| **Temporal Scene Evolution** | Temporal scene diff + evolution tracking | Custom | 1-2 days | 0 | 🟢 High |
| **Event Segmentation** | LLM-based narrative event merger | Custom (EVIS-inspired) | 1-2 days | 0 | 🟢 High |
| **Cross-Video Entity Resolution** | BGE embedding dedup + entity graph | VideoRAG-inspired | 1-2 days | ~0.3 GB | 🟢 High |
| **Graph Reasoning** | Leiden clustering + Node2Vec on KG | VideoRAG (KDD 2026) | 2-3 days | 0 | 🟢 Medium |
| **Multi-modal KG** | Vision-augmented KG nodes | Custom | 2-3 days | ~0.5 GB | 🟢 High |
| **Persistent Memory** | Cross-video entity state tracking | LongLive-RAG inspired | 1-2 days | 0 | 🟢 High |
| **Event Detection** | HAVEN hierarchical entity indexing | CVPR 2026 | 3-4 days | 2-4 GB | 🟡 Medium |

---

## 1. Video Scene Graph Generation (VidSGG)

### 1.1 DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**Paper:** arXiv:2603.10538 (CVPR 2026, Mar 2026)
**Authors:** Lorenz, Kovganko, Kohout, Phatak, Kienzle, Lienhart
**GitHub:** TBD (pre-print only as of June 2026)

#### What It Is

DSFlash is the first real-time panoptic scene graph generation model. It processes **video streams at 56 FPS on RTX 3090** and generates *comprehensive* scene graphs (not just salient relationships).

#### Key Innovations

- **Comprehensive vs. salient:** Prior SGG methods only output relationships that pass a saliency threshold. DSFlash outputs all detected (subject, predicate, object) triplets.
- **Panoptic scope:** Objects + stuff (trees, sky, road) — not just things.
- **Lightweight training:** <24 hours on a single GTX 1080 (nine-year-old GPU).
- **RTX 4070 projection:** At 56 FPS on RTX 3090, expect ~25-35 FPS on RTX 4070 — easily real-time for non-streaming analysis.

#### Integration Strategy

```
┌──────────────────────────────┐
│  Frame from Pipeline         │
│  (already decoded)           │
│       ↓                      │
│  DSFlash Inference           │
│  ┌──────────────────────┐   │
│  │ subject-predicate-obj │   │
│  │ triplets per frame    │   │
│  └──────────────────────┘   │
│       ↓                      │
│  SceneGraph.add_edges()     │
│  ┌──────────────────────┐   │
│  │ Typed edges:         │   │
│  │ person-sitting-chair │   │
│  │ car-driving-road     │   │
│  └──────────────────────┘   │
│       ↓                      │
│  KnowledgeGraph:             │
│  export triplets as           │
│  Entity→Relationship→Entity  │
└──────────────────────────────┘
```

#### Status

**DEFER until code release.** The paper describes model architecture but no implementation is yet available. Check https://github.com/lorenzjulian/DSFlash periodically. When released, the model's lightweight nature (~4-6 GB VRAM inference) makes it an ideal replacement for the current post-hoc scene graph (which derives relationships from ChromaDB text metadata).

#### Expected VRAM

~4-6 GB (extrapolated from RTX 3090 performance)

---

### 1.2 NL-VSGG: Weakly Supervised Video Scene Graph Generation

**Paper:** arXiv:2502.15370 (ICLR 2025, Feb 2025)
**Authors:** Kim, Yoon, In, Jeon, Moon, Kim, Park

#### What It Is

NL-VSGG is the first weakly-supervised VidSGG framework that trains a video scene graph generator using **only video captions** (no manual frame-level annotations). It addresses two key challenges:
1. **Temporality** — video captions include temporal markers (before, while, then, after)
2. **Action duration variability** — actions in video unfold over varying time spans

#### Architecture

```
Video Captions
    ↓
Temporality-aware Caption Segmentation (TCS)
    ├── LLM segments captions into temporally ordered sentences
    └── Each sentence maps to a specific frame range
    ↓
Action Duration Variability-aware alignment (ADV)
    ├── Aligns each segmented sentence with appropriate frames
    └── Handles varying action durations
    ↓
VidSGG Model Training (weakly supervised)
    └── Predicts (subject, predicate, object) triplets per frame
```

#### Relevance to This Project

The existing pipeline already has **scene descriptions** from OpenCLIP/InternVideo3 and **captions** from transcripts. These can be used as weak supervision to extract structured (subject, predicate, object) triplets without a dedicated SGG model:

```python
# Current: scene_graph.py derives edges from metadata overlap
# NL-VSGG approach: LLM-based triplet extraction from scene captions
def extract_triplets_from_scene(description: str) -> List[Tuple[str, str, str]]:
    """Use LLM to extract subject-predicate-object from scene caption."""
    prompt = (
        "Extract all (subject, predicate, object) relationships from this scene description. "
        "Output as JSON array: [{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}]\n\n"
        f"Scene: {description}"
    )
    # returns structured triplets
```

#### Integration

**Immediate.** No new model needed — reuses existing LLM call for structured triplet extraction from scene descriptions. Triplets feed directly into both `scene_graph.py` edges and `knowledge_graph.py` entity/relationship records.

---

### 1.3 STAVE: Spatio-Temporal Attention for Video Entity Tracking

**Paper:** arXiv:2510.26027 (NeurIPS 2025)
**Authors:** Rasekh et al.

STAVE focuses on learning spatio-temporal video representations. For integration: the project already has ByteTrack for temporal entity tracking. STAVE's spatio-temporal attention mechanism could refine entity linking by learning appearance-affinity over time, replacing the current simple track_id-based matching. However, this requires a trained model — monitor for released checkpoints.

---

## 2. Spatio-Temporal Scene Graphs & Temporal Scene Evolution

### 2.1 Temporal Scene Diff & Evolution Tracking (Custom)

**Rationale:** The current `scene_graph.py` captures static scene relationships but has no mechanism for tracking how scenes *evolve* over time within or across videos. Adding temporal evolution tracking unlocks:

- "How did entity X change throughout the video?"
- "Track the relationship between person A and object B over time."
- "Show the evolution of [concept] across all videos."

#### Architecture

```python
class TemporalEvolutionTracker:
    """Tracks how entities and relationships evolve over video time."""
    
    def compute_scene_diff(
        self, prev_scene_id: int, curr_scene_id: int, video_id: str
    ) -> SceneDiff:
        """
        Compute the delta between two consecutive scenes:
        - Entities that appeared (new)
        - Entities that disappeared (lost)
        - Relationships that changed
        - Actions that transitioned
        """
        
    def get_entity_trajectory(
        self, entity_name: str, video_id: str
    ) -> List[EntityState]:
        """
        Track an entity's state across all scenes in a video.
        Returns a timeline of (scene_id, timestamp, attributes, relationships).
        """
        
    def get_temporal_subgraph(
        self, video_id: str, start_scene: int, end_scene: int
    ) -> nx.Graph:
        """
        Extract a subgraph spanning a specific time window.
        Useful for "what happened between time X and Y" queries.
        """
```

#### Data Model

```python
@dataclass
class SceneTransition:
    """A transition between two consecutive scenes."""
    video_id: str
    prev_scene: int
    curr_scene: int
    entities_entered: List[str]
    entities_exited: List[str]
    entities_persisted: List[str]
    relationship_deltas: List[Dict]
    action_transition: Optional[str]

@dataclass
class EntityTrajectory:
    """An entity's evolution across a video."""
    entity_name: str
    entity_type: str
    appearances: List[EntityAppearance]
    # scene_id, timestamp, attributes, bounding_box
```

#### Integration into SceneGraph

Add to `scene_graph.py`:
```python
def build_temporal_evolution(self) -> None:
    """Build temporal evolution metadata from scene chronology."""
    for video_id in self._video_ids:
        scenes = sorted(self._get_video_scenes(video_id))
        for i in range(1, len(scenes)):
            diff = self._compute_diff(scenes[i-1], scenes[i])
            self._temporal_diffs[(video_id, scenes[i])] = diff
```

**Effort:** 1-2 days | **VRAM:** 0 | **Value:** High

---

## 3. Multi-Modal Knowledge Graphs for Video

### 3.1 VideoRAG (KDD 2026) — Cross-Video Entity Graph

**Paper:** arXiv:2502.01549 (KDD 2026)
**Authors:** Ren, Xu, Xia, Wang, Yin, Huang
**GitHub:** https://github.com/HKUDS/VideoRAG (⭐ 3.1k stars)

#### Core Architecture

VideoRAG is the **only** paper with explicit cross-video knowledge graph support. Its dual-channel architecture:

**Channel 1: Graph-based Textual Knowledge Grounding**
- Entities extracted from transcripts + captions across ALL videos
- NetworkX-based graph storage (same library as this project)
- **Leiden clustering** for community detection
- **Node2Vec** for node embeddings (graph-based entity similarity)
- GraphML persistence

**Channel 2: Multi-Modal Context Encoding**
- Visual features from video segments
- NanoVectorDB for segment-level storage
- MiniCPM-V-2.6 for captioning

#### What This Project Already Has vs. What's New

| Capability | Current Status | VideoRAG Addition |
|-----------|---------------|-------------------|
| Entity extraction | ✅ YOLO + ByteTrack + transcripts | Leiden clustering on entity graph |
| Cross-video queries | ✅ SQLite KG with entity→video mapping | Node2Vec entity embeddings |
| Scene graph | ✅ Per-video temporal/entity edges | Cross-video entity graph |
| Graph persistence | ✅ SQLite | GraphML + Node2Vec |
| Community detection | ❌ Not implemented | **Leiden clustering of entities** |
| Entity embedding | ❌ Not implemented | **Node2Vec for graph-aware entity similarity** |

#### Priority Integration: Entity Community Detection

```python
import networkx as nx
from community import community_louvain  # python-louvain package

def detect_entity_communities(kg: KnowledgeGraph) -> Dict[int, int]:
    """Discover entity communities via Leiden/Louvain clustering."""
    G = nx.Graph()
    
    # Build entity co-occurrence graph from KG relationships
    for rel in kg.get_top_relationships(limit=10000):
        G.add_edge(rel.source_id, rel.target_id, weight=rel.strength)
    
    # Detect communities
    partition = community_louvain.best_partition(G)
    return partition  # entity_id → community_id
```

This enables:
- "Find all entities related to [entity X]" via community membership
- Entity disambiguation (same community = likely same context)
- Knowledge graph summarization by community

#### VRAM Estimate

- Leiden clustering: ~50-100 MB for 10K entities
- Node2Vec training: ~200-500 MB (CPU-only, no GPU needed)
- **Total: <1 GB, CPU-only**

---

### 3.2 Vision-Augmented Knowledge Graph Nodes

**Rationale:** The current `knowledge_graph.py` stores entities as text records with JSON metadata. Adding visual embeddings as first-class node attributes enables **vision-aware entity matching** and **visual entity retrieval**.

#### Architecture

```python
class VisionKnowledgeGraph(KnowledgeGraph):
    """Extends KnowledgeGraph with vision-augmented entity nodes."""
    
    def add_entity_with_embedding(
        self, name, entity_type, video_id, metadata=None, 
        visual_embedding: Optional[np.ndarray] = None
    ) -> int:
        entity_id = super().add_entity(name, entity_type, video_id, metadata)
        if visual_embedding is not None:
            self._store_visual_embedding(entity_id, visual_embedding)
        return entity_id
    
    def find_visually_similar_entities(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[EntityRecord, float]]:
        """Find entities by visual similarity (not just text)."""
        # Cosine similarity against all stored entity embeddings
```

#### Storage Strategy

Since SQLite doesn't natively support vector search at scale:

**Option A (ChromaDB hybrid):** Store visual embeddings in a separate ChromaDB collection (parallel to scene chunks), keyed by `entity_{entity_id}`.

**Option B (numpy + FAISS):** Keep embeddings in memory as a numpy matrix during active sessions, persist as `.npy` files. FAISS index for similarity search.

**Option C (SQLite FTS5 + fallback):** Store embeddings as BLOBs in a new `entity_embeddings` table, compute similarity in Python for <1K entities, fall back to ChromaDB for larger.

**Recommended:** Option A — reuses existing ChromaDB infrastructure with zero new dependencies.

#### VRAM Estimate

- BGE-VL embedding per entity: ~0.5 MB (1024-dim float32)
- 10K entities: ~5 GB — too much for 12GB VRAM
- **Recommendation:** Store in ChromaDB (disk-backed), not in GPU memory

---

## 4. Event Detection & Segmentation

### 4.1 HAVEN: Hierarchical Long Video Understanding (CVPR 2026)

**Paper:** arXiv:2601.13719 (CVPR 2026, Jan 2026)
**Authors:** Yin, Peng, Li, Xiong, Lu (Microsoft Research Asia)

#### What It Is

HAVEN is a unified framework for long-video understanding that organizes content into a **structured hierarchy** spanning:

```
Global Summary
    ├── Scene Level
    │   ├── Segment Level  
    │   │   └── Entity Level
    │   └── Segment Level
    └── Scene Level
        └── ...
```

Core innovations:
1. **Audiovisual Entity Cohesion** — integrates entity-level representations across visual AND auditory streams (speaker diarization + visual entity tracking)
2. **Hierarchical Video Indexing** — 4-level hierarchy (global → scene → segment → entity)
3. **Agentic Search** — dynamic retrieval and reasoning across hierarchy layers

#### Results

- **LVBench: 84.1%** overall accuracy (state-of-the-art)
- **Reasoning category: 80.1%** (8+% above prior SOTA)
- Accepted at CVPR 2026

#### Relevance & Integration

The existing project already has a similar hierarchy via:
- **HybridTree** (v0.51.0) — temporal-semantic tree
- **KnowledgeGraph** — entity-level tracking
- **SceneGraph** — scene-level connectivity

HAVEN's key addition: **formal 4-level indexing** with **agentic traversal** across levels. The project could adopt:

```python
class HierarchicalVideoIndex:
    """4-level hierarchy: global → scene → segment → entity"""
    
    levels = ["global", "scene", "segment", "entity"]
    
    def query(self, query: str, target_level: str = "entity"):
        """Agentic search across hierarchy levels."""
        # Start at global, drill down to target level
```

**Effort:** 3-4 days | **VRAM:** ~2-4 GB (for entity extraction across streams)

---

### 4.2 EVIS-Inspired Event Segmentation

**Paper:** IEEE TIP 2026 (EVIS — Event-aware Video Segmentation)

Rather than relying on released code (which may not be available), implement a **lightweight event merger** using existing pipeline outputs:

```python
class NarrativeEventMerger:
    """Merge related scenes into narrative events (EVIS-inspired)."""
    
    def merge_scenes_to_events(
        self, video_id: str, scenes: List[SceneInfo]
    ) -> List[NarrativeEvent]:
        """
        Uses three signals to merge scenes into events:
        1. Transcript coherence (topic continuity)
        2. Entity persistence (same people/objects present)
        3. Visual similarity (BGE-VL embedding similarity)
        
        Returns: List of NarrativeEvent with title, scenes, time range
        """
```

This is **zero-VRAM** (CPU-only, LLM-based) and directly enhances both the timeline in `knowledge_graph.py` and the scene structure in `scene_graph.py`.

---

## 5. Entity Linking Across Videos

### 5.1 Cross-Video Entity Resolution with BGE Embeddings

The current `knowledge_graph.py` deduplicates entities by `(name, type)` exact match. This misses:
- "John Smith" in video A vs "Dr. Smith" in video B
- "red car" vs "sports car"
- "running" vs "jogging"

#### Proposed: BGE-Based Entity Resolution

```python
class EntityResolver:
    """Cross-video entity resolution using BGE-VL embeddings."""
    
    def __init__(self, embedding_model="BAAI/BGE-VL-base"):
        self.model = embedding_model  # Already loaded in pipeline!
        self._entity_cache: Dict[int, np.ndarray] = {}
    
    def resolve_entity(
        self, entity: EntityRecord, existing_entities: List[EntityRecord],
        threshold: float = 0.85
    ) -> Tuple[Optional[int], float]:
        """
        Find if 'entity' matches any existing entity.
        Uses BGE embedding of (name + type + metadata).
        
        Returns: (matched_entity_id, similarity) or (None, 0.0)
        """
        query_emb = self._embed(entity)
        best_id, best_sim = None, 0.0
        
        for existing in existing_entities:
            existing_emb = self._get_or_compute(existing)
            sim = cosine_similarity(query_emb, existing_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = existing.id
        
        if best_sim >= threshold:
            return best_id, best_sim
        return None, 0.0
```

#### Integration into KnowledgeGraph

Replace the exact-name dedup in `add_entity()` with a resolution step:

```python
def add_entity(self, name, entity_type, video_id, metadata=None):
    # NEW: Try BGE-based resolution first (when enabled)
    if self._entity_resolver:
        resolved_id, sim = self._entity_resolver.resolve_entity(
            name, entity_type, metadata
        )
        if resolved_id:
            # Merge into existing entity
            return self._merge_entity(resolved_id, name, video_id, metadata)
    
    # FALLBACK: exact (name, type) match (current behavior)
    existing = self._exact_match(name, entity_type)
    ...
```

#### VRAM

- BGE-VL-base is already loaded in the pipeline (reuse existing model)
- **Additional VRAM: 0** (shared with embedding pipeline)

---

## 6. Graph Neural Network Based Reasoning

### 6.1 Node2Vec Entity Embeddings for KG Reasoning

**Paper:** Node2Vec (KDD 2016, widely adopted in 2025-2026 video KG systems)
**Implementation:** `node2vec` pip package

#### What It Adds

Node2Vec learns dense vector representations for each entity in the knowledge graph, capturing:
- **Structural similarity:** entities with similar relationship patterns get similar embeddings
- **Community structure:** entities in the same cluster have close embeddings
- **Relationship roles:** "person who drives car" vs "person who reads book"

```python
from node2vec import Node2Vec

def train_entity_embeddings(kg: KnowledgeGraph) -> Dict[int, np.ndarray]:
    """Train Node2Vec embeddings on the entity relationship graph."""
    G = nx.Graph()
    
    # Build graph from KG relationships
    for rel in kg.get_top_relationships(limit=5000):
        G.add_edge(rel.source_id, rel.target_id, weight=rel.strength)
    
    # Train Node2Vec
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    return {node: model.wv[str(node)] for node in G.nodes()}
```

#### Use Cases

1. **Entity recommendation:** "Given entity A, what other entities are structurally similar?"
2. **Query expansion:** Expand query entities with structurally similar entities
3. **Missing relationship prediction:** Link prediction (two entities are close in embedding space but have no direct relationship)
4. **Graph-based re-ranking:** Score retrieved chunks by entity embedding proximity

#### VRAM

- **CPU-only:** ~200-500 MB RAM
- **No GPU needed**

---

### 6.2 Leiden Community Detection for Entity Clustering

**Paper:** Traag et al. (2019) — From Louvain to Leiden
**Package:** `cdlib` or `python-louvain`

Adds hierarchical community structure to the KG:

```python
def detect_communities(kg: KnowledgeGraph) -> CommunityResult:
    """
    Returns: dict mapping community_id → List[EntityRecord]
    """
    G = nx.Graph()
    for rel in kg.get_top_relationships(limit=10000):
        G.add_edge(rel.source_id, rel.target_id, weight=rel.strength)
    
    from cdlib import algorithms
    communities = algorithms.leiden(G)
    
    result = {}
    for i, community in enumerate(communities.communities):
        result[i] = [kg.get_entity(eid) for eid in community]
    
    return result
```

Integration into `knowledge_graph.py`:
```python
def get_knowledge_context(self, limit_entities=100):
    # ... existing code ...
    
    # NEW: Add community summary
    communities = self._detect_communities()
    lines.append(f"\n### Entity Communities ({len(communities)} groups)")
    for cid, entities in communities.items():
        names = [e.name for e in entities[:5]]
        lines.append(f"- **Community {cid}**: {', '.join(names)}... ({len(entities)} members)")
```

---

## 7. Persistent Video Understanding & Cross-Video Memory

### 7.1 LongLive-RAG: RAG-as-Memory

**Paper:** arXiv:2606.02553 (Jun 2026)
**Authors:** Hu et al.

#### Core Idea

Searchable history of previously retrieved context, growing as conversations progress. Enables reference to earlier scenes without re-retrieval.

#### Integration

The project already has `conversation_memory_enabled` (ChromaDB-backed Q&A memory). Extend this with **entity-aware conversation memory**:

```python
class EntityAwareMemory:
    """Extends conversation memory with entity-state tracking."""
    
    def record_entity_state(
        self, entity_name: str, video_id: str, 
        state_description: str, timestamp: float
    ):
        """Record what we know about an entity at a point in time."""
        
    def get_entity_history(
        self, entity_name: str
    ) -> List[EntityStateRecord]:
        """Get all recorded states for an entity across conversations."""
```

This enhances the `KnowledgeGraph.get_knowledge_context()` to include **what the system has learned** about entities across multiple analysis sessions, not just entity frequencies.

---

### 7.2 Cross-Video Entity State Machine

**New concept:** Track entities as stateful objects that transition between states across videos:

```python
class EntityStateMachine:
    """
    Tracks entity state transitions across videos.
    Example: "car" in video 1 (moving) → video 2 (parked) → video 3 (damaged)
    """
    
    states: Dict[str, List[EntityState]]  # entity_name → state history
    
    def record_state(
        self, entity_name: str, state: str, 
        video_id: str, scene_id: int, confidence: float
    ):
        
    def get_state_sequence(self, entity_name: str) -> List[EntityState]:
        """Chronological state sequence for an entity."""
        
    def detect_state_change(self, entity_name: str) -> Optional[StateChange]:
        """Detect if an entity changed state between videos."""
```

#### Storage

Add an `entity_states` table to `knowledge_graph.py`:
```sql
CREATE TABLE IF NOT EXISTS entity_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL,
    state TEXT NOT NULL,
    video_id TEXT NOT NULL,
    scene_id INTEGER NOT NULL,
    confidence REAL DEFAULT 1.0,
    observed_at REAL NOT NULL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (entity_id) REFERENCES entities(id)
)
```

---

## 8. Integration Blueprint for RTX 4070 12GB

### VRAM Budget

| Component | Current VRAM | Target VRAM | Optimization |
|-----------|-------------|-------------|-------------|
| YOLO26x | ~2.0 GB | ~1.5 GB | FP16 inference |
| BGE-VL-base | ~0.8 GB | ~0.8 GB | Same |
| InternVideo3-8B | ~6.0 GB (FP8) | ~5.5 GB (FP8) | Already FP8 |
| ChromaDB | ~0.2 GB | ~0.2 GB | Same |
| InsightFace | ~1.0 GB | ~0.8 GB | Conditional load |
| **DSFlash** (future) | — | ~4.0 GB | FP16, only when SGG enabled |
| **Total** | ~10.0 GB | ~10.8-11.8 GB | Tight but workable |

**Strategy:** Models that aren't needed simultaneously (e.g., InternVideo3 and DSFlash) can be swapped via `unload()`. The existing pipeline already supports this pattern.

### Implementation Roadmap

#### Sprint 1 (1-2 days): High Impact, Zero VRAM

| Task | Files Modified | Description |
|------|---------------|-------------|
| LLM-based triplet extraction | `scene_graph.py` | Extract (subject, predicate, object) from captions |
| Temporal scene diff tracking | `scene_graph.py` + new module | Scene-to-scene entity evolution |
| Narrative event merger | New: `event_segmentation.py` | Merge related scenes into events |
| Cross-video entity dedup config | `config.py` | Toggle for BGE-based entity resolution |

#### Sprint 2 (2-3 days): KG Enhancement

| Task | Files Modified | Description |
|------|---------------|-------------|
| BGE-based entity resolution | `knowledge_graph.py` | Fuzzy entity merging |
| Leiden community detection | `knowledge_graph.py` | Entity communities in KG |
| Entity state machine | `knowledge_graph.py` + new table | Cross-video stateful tracking |
| Node2Vec entity embeddings | `knowledge_graph.py` + cache | Graph-aware entity similarity |

#### Sprint 3 (2-3 days): Visual Integration

| Task | Files Modified | Description |
|------|---------------|-------------|
| Visual embedding storage in ChromaDB | `knowledge_graph.py` | Vision-augmented entity nodes |
| Entity trajectory viewer | `ui/knowledge_graph.py` | Timeline of entity state changes |
| Community visualization | `ui/knowledge_graph.py` | Entity community graph in KG Explorer |
| Hierarchical video index | New: `video_hierarchy.py` | HAVEN-inspired 4-level hierarchy |

#### Sprint 4 (3-4 days): GNN Reasoning

| Task | Files Modified | Description |
|------|---------------|-------------|
| Graph-based chunk re-ranking | `scene_graph.py` + `rag.py` | Use entity proximity for retrieval |
| Entity recommendation API | `api.py` | "Related entities" endpoint |
| Cross-video story generation | `report.py` | Narrative across video boundaries |
| Link prediction on entity graph | `knowledge_graph.py` | Suggest missing relationships |

---

## 9. Benchmarks & Evaluation

### Benchmarks for VidSGG

| Benchmark | Description | Metric | Relevance |
|-----------|-------------|--------|-----------|
| **Action Genome** | Multi-frame scene graphs with temporal annotations | mR@K, R@K | Standard VidSGG benchmark |
| **VidVRD** | Video Visual Relationship Detection | Recall, mAP | Video-specific relationships |
| **VidSGG benchmark (TRECVID V3C1)** | Large-scale video scene graph evaluation | SGG metrics | Comprehensive evaluation |
| **LongerVideos** (VideoRAG) | 134+ hours across 160 videos | QA accuracy | Cross-video understanding |
| **LVBench** | Long video understanding benchmark | Accuracy | VideoQA + reasoning |
| **Video-MME** | Multi-modal video QA | Accuracy | MLLM video understanding |

### Evaluation Strategy for This Project

For each new capability, define evaluation metrics:

| Capability | Metric | How to Measure |
|-----------|--------|---------------|
| Triplet extraction | Precision/Recall | Manual spot-check vs LLM extracted triplets |
| Event segmentation | Purity, NMI | Compare LLM-merged events vs manual |
| Cross-entity resolution | F1, Accuracy | Test with known aliases across videos |
| Community detection | Modularity | Standard graph metric |
| Node2Vec embeddings | Link prediction AUC | Held-out relationship edges |
| Temporal scene diff | Consistency score | Human evaluation of detected transitions |

---

## 10. Open-Source Projects Referenced

| Project | License | Stars | Use Case |
|---------|---------|-------|----------|
| HKUDS/VideoRAG | Apache 2.0 | ⭐ 3.1k | Cross-video KG, Leiden clustering, Node2Vec |
| xiaoqian-shen/VGent | Apache 2.0 | ⭐ 48 | Per-video scene graphs (already implemented) |
| microsoft/graphrag | MIT | ⭐ 25k+ | Entity-centric text graphs (adaptable for video) |
| node2vec/node2vec | MIT | ⭐ 1.3k | Graph embeddings for entity similarity |
| cdlib (Community Detection Lib) | BSD-2 | ⭐ 1k | Leiden/Louvain clustering |
| NetworkX | BSD-3 | ⭐ 16k+ | Graph infrastructure (already in use) |

---

## 11. Key Papers Reference List

1. **DSFlash** — Lorenz et al. "Comprehensive Panoptic Scene Graph Generation in Realtime." CVPR 2026. arXiv:2603.10538
2. **NL-VSGG** — Kim et al. "Weakly Supervised Video Scene Graph Generation via Natural Language Supervision." ICLR 2025. arXiv:2502.15370
3. **VideoRAG** — Ren et al. "Retrieval-Augmented Generation with Extreme Long-Context Videos." KDD 2026. arXiv:2502.01549
4. **HAVEN** — Yin et al. "Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search." CVPR 2026. arXiv:2601.13719
5. **InternVideo3** — Yan et al. "Agentify Foundation Models with Multimodal Contextual Reasoning." arXiv:2606.12195 (Jun 2026)
6. **Robust-TO** — He, Choi, Yoon. "Confidence-Aware Tool Orchestration for Robust Video Understanding." arXiv:2606.26904 (Jun 2026)
7. **Decoupling Semantics and Logic** — Dai et al. ACL 2026 MAGMAR. arXiv:2606.07924
8. **LongLive-RAG** — Hu et al. "RAG-as-Memory." arXiv:2606.02553 (Jun 2026)
9. **EVIS** — IEEE TIP 2026 (Event-aware segmentation)
10. **VLDB 2026 KG Survey** — "Recent Advances in Knowledge Graph-Enabled Machine Learning." VLDB 2026.

---

## 12. What NOT to Pursue (RTX 4070 Constraints)

| Technique | Reason |
|-----------|--------|
| Full end-to-end VidSGG transformer (e.g., STTran, TRiPD) | >12 GB VRAM, requires Action Genome training data |
| Proprietary API-based entity linking | Project policy: 100% local |
| Training custom Node2Vec at scale (>100K entities) | RAM >12 GB |
| Real-time DSFlash streaming (56 FPS) | Needs RTX 3090 for 56 FPS; 4070 can do offline batch |
| Large-scale GNN training (GCN, RGCN) | Requires >12 GB for meaningful graph sizes |
| Full HAVEN model replication | 8+ A100s for training; only the architecture pattern is portable |

---

## 13. Summary of Highest-Impact Next Steps

### 🟢 Do Now (P0, Zero VRAM Cost)

1. **LLM-based triplet extraction** from scene captions → feeds structured (S, P, O) into both scene graph edges and KG relationships
2. **Temporal scene diff tracking** — entity appearance/disappearance between consecutive scenes
3. **Narrative event merger** — LLM-based merging of related scenes into coherent events
4. **Cross-video entity resolution toggle** in config

### 🟡 Plan This Sprint (P1, <0.5 GB VRAM)

5. **BGE-based fuzzy entity resolution** in `knowledge_graph.py`
6. **Leiden community detection** on entity graph
7. **Entity state machine** — track entity state transitions across videos
8. **Node2Vec entity embeddings** for graph-aware retrieval

### 🟠 Investigate (P2, 0.5-4 GB VRAM)

9. **Vision-augmented KG nodes** — visual embeddings in ChromaDB for entity matching
10. **HAVEN-inspired 4-level hierarchy** — global → scene → segment → entity
11. **Graph-based chunk re-ranking** using entity proximity

### 🔴 Monitor (P3, 4-6 GB VRAM)

12. **DSFlash** — wait for code release; ideal real-time SGG replacement for post-hoc text-derived graphs

---

*Research compiled by Hermes Agent (worker subagent), June 27, 2026*
*Target integration: video-analysis v0.55.0+*
