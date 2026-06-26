"""
Scene Graph for video context — lightweight graph-based video knowledge.

Implements a **VGent/ViG-RAG-inspired** scene-graph retrieval layer on top of
ChromaDB.  Instead of requiring an external graph database, this module builds
and maintains an in-memory graph where:

- **Nodes** = video scenes (identified by `video_id/scene_id`)
- **Edges** = semantic relationships between scenes:
  - ``temporal`` — scenes that are adjacent or nearby in time
  - ``entity_shared`` — scenes sharing the same detected objects / people / actions
  - ``semantic_similar`` — scenes whose descriptions are similar in embedding space

The graph enables **K-hop retrieval**: when a scene matches a query, we also
retrieve scenes that are connected via graph edges (not just temporal neighbors),
finding semantically related content that may appear in different parts of
the video or across different videos.

Architecture follows ViG-RAG (AAAI 2026): hybrid temporal + semantic graph
reasoning.  The graph is built from ChromaDB metadata at query time (or when
``rebuild()`` is called), keeping the storage footprint small.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from video_analysis.config import Config
from video_analysis.rag import VideoRAG, RetrievedChunk

import json  # noqa: E402 — module-level import after package imports

logger = logging.getLogger(__name__)


class SceneGraph:
    """Lightweight in-memory scene graph for video analysis.

    Builds a graph from ChromaDB metadata where each scene is a node and
    edges represent temporal proximity, shared entities, or semantic similarity.

    The graph is built lazily — call ``rebuild()`` after indexing new videos,
    or let it auto-rebuild on the first query.

    Args:
        rag: VideoRAG instance for accessing ChromaDB metadata.
        config: Optional config (uses RAG's config by default).
        k_hop_expansion: Number of hops for graph traversal (default 2).
                         0 = no graph expansion (bypass).
        temporal_edge_window: Max scenes apart for temporal edge (default 3).
        min_shared_entities: Min shared objects/actions for entity edge (default 1).
        entity_similarity_threshold: Min cosine sim for semantic edge (default 0.85).
    """

    def __init__(
        self,
        rag: VideoRAG,
        config: Optional[Config] = None,
        k_hop_expansion: int = 2,
        temporal_edge_window: int = 3,
        min_shared_entities: int = 1,
        entity_similarity_threshold: float = 0.85,
    ):
        self.rag = rag
        self.config = config or rag.config
        self.k_hop_expansion = k_hop_expansion
        self.temporal_edge_window = temporal_edge_window
        self.min_shared_entities = min_shared_entities
        self.entity_similarity_threshold = entity_similarity_threshold

        # Graph structures: adjacency list keyed by (video_id, scene_id)
        self._adjacency: Dict[Tuple[str, int], Set[Tuple[str, int]]] = defaultdict(set)
        # Node metadata: keyed by (video_id, scene_id) -> dict
        self._node_meta: Dict[Tuple[str, int], dict] = {}
        # Whether the graph has been built
        self._built = False

    def rebuild(self):
        """Rebuild the scene graph from ChromaDB metadata.

        Should be called after indexing new videos.
        """
        self._adjacency.clear()
        self._node_meta.clear()

        try:
            all_data = self.rag.collection.get(
                include=["metadatas", "documents"],
            )
        except Exception as e:
            logger.warning(f"Cannot rebuild scene graph: {e}")
            self._built = False
            return

        if not all_data or not all_data["ids"]:
            self._built = True
            return

        # Collect scene-level nodes
        scene_nodes: Dict[Tuple[str, int], dict] = {}
        scene_documents: Dict[Tuple[str, int], str] = {}
        for i, doc_id in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            sid = meta.get("scene_id")
            vid = meta.get("video_id", "unknown")
            ctype = meta.get("chunk_type")
            if sid is not None and sid >= 0 and ctype == "scene":
                key = (vid, sid)
                scene_nodes[key] = meta
                scene_documents[key] = (
                    all_data["documents"][i] if all_data["documents"] else ""
                )

        if not scene_nodes:
            logger.debug("No scene nodes found in ChromaDB")
            self._built = True
            return

        self._node_meta = scene_nodes

        # --- Build edges ---

        # 1. Temporal edges: scenes near each other in the same video
        video_to_scenes: Dict[str, List[int]] = defaultdict(list)
        for vid, sid in scene_nodes:
            video_to_scenes[vid].append(sid)
        for vid, scene_ids in video_to_scenes.items():
            scene_ids.sort()
            for i, sid in enumerate(scene_ids):
                for j in range(
                    i + 1, min(i + self.temporal_edge_window + 1, len(scene_ids))
                ):
                    neighbor = scene_ids[j]
                    self._add_edge((vid, sid), (vid, neighbor))

        # 2. Entity-shared edges: scenes sharing detected objects, people, actions
        # Extract entities from scene metadata (objects, actions)
        scene_entities: Dict[Tuple[str, int], Set[str]] = {}
        for key, meta in scene_nodes.items():
            entities = set()
            doc_text = scene_documents.get(key, "")
            # Extract objects mentioned in the scene document
            if "[Objects detected]" in doc_text:
                for line in doc_text.split("\n"):
                    if line.startswith("[Objects detected]"):
                        objs = line.replace("[Objects detected]:", "").strip()
                        for obj in objs.split(","):
                            obj = obj.strip().lower()
                            if obj:
                                entities.add(f"obj:{obj}")
            # Also extract objects from metadata's objects field (new in v0.19.0+)
            meta_objects = meta.get("objects", "")
            if meta_objects:
                if isinstance(meta_objects, str):
                    obj_list = meta_objects.split(",")
                elif isinstance(meta_objects, list):
                    obj_list = meta_objects
                else:
                    obj_list = []
                for obj_str in obj_list:
                    obj_name = obj_str.strip().lower()
                    if obj_name:
                        entities.add(f"obj:{obj_name}")
            # Extract track IDs from metadata for cross-scene entity matching
            track_ids_raw = meta.get("track_ids", "")
            if track_ids_raw:
                if isinstance(track_ids_raw, str):
                    for tid in track_ids_raw.split(","):
                        tid = tid.strip()
                        if tid:
                            entities.add(f"track:{tid}")
                elif isinstance(track_ids_raw, (list, tuple)):
                    for tid in track_ids_raw:
                        entities.add(f"track:{str(tid)}")
            if "[Action at" in doc_text:
                for line in doc_text.split("\n"):
                    if line.startswith("[Action at") and ":" in line:
                        action_part = line.split("]:", 1)[-1].strip()
                        # Remove confidence percentage
                        if " (" in action_part:
                            action_part = action_part.split(" (")[0]
                        if action_part:
                            entities.add(f"action:{action_part.lower()}")
            # Extract face identities from metadata (InsightFace, v0.26.0+)
            # Faces are stored per-frame; scene-level metadata may include
            # a 'face_ids' field (comma-separated unique face IDs detected
            # in this scene).
            face_ids_raw = meta.get("face_ids", "")
            if face_ids_raw:
                if isinstance(face_ids_raw, str):
                    for fid in face_ids_raw.split(","):
                        fid = fid.strip()
                        if fid:
                            entities.add(f"face:{fid}")
                elif isinstance(face_ids_raw, (list, tuple)):
                    for fid in face_ids_raw:
                        entities.add(f"face:{str(fid)}")
            # Also check the metadata 'faces' field for backward compatibility
            meta_faces = meta.get("faces", "")
            if meta_faces and not face_ids_raw:
                if isinstance(meta_faces, str):
                    try:
                        face_list = json.loads(meta_faces)
                        if isinstance(face_list, list):
                            for face in face_list:
                                fid = face.get("face_id", "")
                                if fid:
                                    entities.add(f"face:{fid.lower()}")
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(meta_faces, list):
                    for face in meta_faces:
                        if isinstance(face, dict):
                            fid = face.get("face_id", "")
                            if fid:
                                entities.add(f"face:{fid.lower()}")
            scene_entities[key] = entities

        # Connect scenes sharing entities
        keys_list = list(scene_entities.keys())
        for i in range(len(keys_list)):
            for j in range(i + 1, len(keys_list)):
                k1, k2 = keys_list[i], keys_list[j]
                shared = scene_entities[k1] & scene_entities[k2]
                if len(shared) >= self.min_shared_entities:
                    self._add_edge(k1, k2)

        # 3. Semantic edges: scenes with similar descriptions (reuse BGE-VL if available)
        # We use a lighter approach: identical or overlapping object sets + transcript keywords
        for i in range(len(keys_list)):
            for j in range(i + 1, len(keys_list)):
                k1, k2 = keys_list[i], keys_list[j]
                if k2 in self._adjacency.get(k1, set()):
                    continue  # already connected
                # Check keyword overlap in scene text
                txt1 = scene_documents.get(k1, "").lower()
                txt2 = scene_documents.get(k2, "").lower()
                # Simple word set overlap as a lightweight semantic proxy
                words1 = set(txt1.split())
                words2 = set(txt2.split())
                if len(words1) > 5 and len(words2) > 5:
                    jaccard = len(words1 & words2) / len(words1 | words2)
                    if jaccard >= 0.3:  # 30% word overlap
                        self._add_edge(k1, k2)

        logger.info(
            f"Scene graph rebuilt: {len(scene_nodes)} nodes, "
            f"{sum(len(v) for v in self._adjacency.values()) // 2} edges"
        )
        self._built = True

    def _add_edge(self, a: Tuple[str, int], b: Tuple[str, int]):
        """Add an undirected edge between two scene nodes."""
        if a != b:
            self._adjacency[a].add(b)
            self._adjacency[b].add(a)

    def k_hop_expand(
        self,
        seed_scene_ids: List[Tuple[str, int]],
    ) -> Set[Tuple[str, int]]:
        """Perform K-hop expansion from seed scene nodes.

        Args:
            seed_scene_ids: List of (video_id, scene_id) tuples to expand from.

        Returns:
            Set of (video_id, scene_id) tuples reachable within K hops.
        """
        if not self._built:
            self.rebuild()
        if not self._built or self.k_hop_expansion <= 0:
            return set(seed_scene_ids)

        visited: Set[Tuple[str, int]] = set()
        frontier: Set[Tuple[str, int]] = set(seed_scene_ids)

        for _ in range(self.k_hop_expansion):
            if not frontier:
                break
            next_frontier: Set[Tuple[str, int]] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in self._adjacency.get(node, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # Add any unvisited seeds
        visited.update(s for s in seed_scene_ids if s not in visited)
        return visited

    def expand_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Expand a list of retrieved chunks with graph-based neighbors.

        For each chunk that has a ``scene_id >= 0``, performs K-hop graph
        traversal to find semantically connected scenes and adds them to the
        result set.

        Args:
            chunks: List of RetrievedChunk from initial retrieval.

        Returns:
            Expanded list with graph-connected scenes appended (deduplicated).
        """
        if self.k_hop_expansion <= 0:
            return chunks

        if not self._built:
            self.rebuild()
        if not self._built:
            return chunks

        # Collect seed scene IDs from chunks
        seed_nodes: List[Tuple[str, int]] = []
        for c in chunks:
            if c.scene_id >= 0 and c.video_id:
                seed_nodes.append((c.video_id, c.scene_id))

        if not seed_nodes:
            return chunks

        expanded_nodes = self.k_hop_expand(seed_nodes)

        # Collect existing chunk IDs to avoid duplicates
        existing_ids = {c.chunk_id for c in chunks}

        # Fetch scene chunks for newly discovered nodes
        new_chunks = []
        for vid, sid in expanded_nodes:
            chunk_id = f"{vid}_scene_{sid:04d}"
            if chunk_id not in existing_ids:
                try:
                    result = self.rag.collection.get(
                        ids=[chunk_id],
                        include=["documents", "metadatas"],
                    )
                    if result["ids"]:
                        meta = result["metadatas"][0]
                        score = 0.5  # graph-expanded chunks get default score
                        new_chunks.append(
                            RetrievedChunk(
                                chunk_id=chunk_id,
                                video_id=vid,
                                text=result["documents"][0],
                                timestamp=meta.get("start_time", 0),
                                scene_id=sid,
                                score=score,
                                metadata=meta,
                                chunk_type="scene",
                            )
                        )
                        existing_ids.add(chunk_id)
                except Exception:
                    pass

        if new_chunks:
            logger.info(
                f"Graph expansion: {len(seed_nodes)} seeds -> "
                f"{len(expanded_nodes)} nodes (+{len(new_chunks)} new chunks)"
            )

        # Merge and sort chronologically
        merged = list(chunks) + new_chunks
        merged.sort(key=lambda c: c.timestamp)
        return merged
