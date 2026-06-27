"""
Persistent Video Knowledge Graph — cross-video entity & relationship store.

Builds and persists a structured knowledge base from all analyzed videos using
SQLite, enabling:

- **Entity extraction** — people, objects, locations, actions, and concepts
  detected across videos, with deduplication and frequency tracking
- **Relationship recording** — co-occurrence, temporal, spatial, and semantic
  relationships between entities
- **Cross-video queries** — find all videos mentioning a person/object/concept
- **Event timeline** — chronologically ordered events across all analyzed videos
- **Persistent storage** — SQLite-backed, survives restarts, auto-VACUUM

Architecture follows the HUME (arXiv:2404.12050) persistent knowledge graph
pattern — entities and relationships are incrementally added as each video is
processed, building a rich cross-video knowledge store over time.

Usage:
    from video_analysis.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(config)
    kg.add_video_index(video_index)
    entities = kg.search_entities("person")
    timeline = kg.get_timeline()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from video_analysis.config import Config

logger = logging.getLogger(__name__)

# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class EntityRecord:
    """A persistent entity record in the knowledge graph.

    Attributes:
        id: Unique entity ID (auto-generated).
        name: Entity name (e.g. "John Doe", "car", "running").
        entity_type: Entity type ('person', 'object', 'action', 'location',
                     'concept', 'event', 'scene_type').
        frequency: Number of times this entity appears across all videos.
        first_seen: Unix timestamp of first appearance.
        last_seen: Unix timestamp of last appearance.
        metadata: Arbitrary JSON metadata (e.g. face embeddings, object class).
        video_ids: Set of video IDs this entity appears in.
    """

    id: int = 0
    name: str = ""
    entity_type: str = "concept"
    frequency: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    video_ids: Set[str] = field(default_factory=set)


@dataclass
class RelationshipRecord:
    """A persistent relationship between two entities.

    Attributes:
        id: Unique relationship ID.
        source_id: Source entity ID.
        target_id: Target entity ID.
        relation_type: Relationship type ('co_occurs', 'appears_with',
                       'temporal_sequence', 'parent', 'child', 'same_as').
        strength: Relationship strength (co-occurrence count).
        last_seen: Unix timestamp of last observation.
        metadata: Arbitrary JSON metadata.
    """

    id: int = 0
    source_id: int = 0
    target_id: int = 0
    relation_type: str = "co_occurs"
    strength: int = 1
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoRecord:
    """Metadata about a video that has been indexed.

    Attributes:
        video_id: Unique video identifier.
        filename: Original filename.
        duration_seconds: Video duration in seconds.
        entity_count: Number of entities extracted.
        indexed_at: Unix timestamp of indexing.
        metadata: Arbitrary JSON metadata.
    """

    video_id: str = ""
    filename: str = ""
    duration_seconds: float = 0.0
    entity_count: int = 0
    indexed_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── SQLite-backed Knowledge Graph ────────────────────────────────────────


class KnowledgeGraph:
    """Persistent cross-video knowledge graph backed by SQLite.

    Thread-safe — uses a per-instance reentrant lock so it can be safely
    called from multiple workers or async contexts.

    Schema:
        - entities: id, name, type, frequency, first_seen, last_seen,
                    metadata (JSON), video_ids (JSON list)
        - relationships: id, source_id, target_id, relation_type, strength,
                         last_seen, metadata (JSON)
        - videos: video_id, filename, duration_seconds, entity_count,
                  indexed_at, metadata (JSON)
    """

    def __init__(self, config=None):
        self._config = config or Config()
        self._db_path: Path = self._config.data_dir / "knowledge_graph.db"
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── Database initialisation ─────────────────────────────────────────

    def _init_db(self) -> None:
        """Create or open the SQLite database and ensure schema exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            cur = self._conn.execute("""CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'concept',
                    frequency INTEGER NOT NULL DEFAULT 1,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    video_ids TEXT DEFAULT '[]'
                )""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL DEFAULT 'co_occurs',
                    strength INTEGER NOT NULL DEFAULT 1,
                    last_seen REAL NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                )""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    filename TEXT DEFAULT '',
                    duration_seconds REAL DEFAULT 0.0,
                    entity_count INTEGER DEFAULT 0,
                    indexed_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )""")
            # Event-Causal RAG event records (v0.58.0)
            self._conn.execute("""CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    start_time REAL DEFAULT 0.0,
                    end_time REAL DEFAULT 0.0,
                    state_before TEXT DEFAULT '',
                    state_after TEXT DEFAULT '',
                    action TEXT DEFAULT '',
                    entities TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 0.0
                )""")
            # Event-Causal RAG causal relations (v0.58.0)
            self._conn.execute("""CREATE TABLE IF NOT EXISTS causal_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_event_id TEXT NOT NULL,
                    target_event_id TEXT NOT NULL,
                    relation_type TEXT DEFAULT 'temporal',
                    strength REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}'
                )""")
            # Indexes for common queries
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_source "
                "ON relationships(source_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_target "
                "ON relationships(target_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_videos_indexed_at "
                "ON videos(indexed_at)"
            )
            self._conn.commit()

    # ── Entity operations ───────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
        video_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add or update an entity.

        If an entity with the same name and type exists, increments its
        frequency and updates last_seen.  Otherwise creates a new record.

        Returns:
            The entity ID.
        """
        now = time.time()
        meta = metadata or {}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, frequency, video_ids FROM entities "
                "WHERE name = ? AND type = ?",
                (name, entity_type),
            ).fetchone()

            if row:
                entity_id = row["id"]
                freq = row["frequency"] + 1
                vids: List[str] = json.loads(row["video_ids"] or "[]")
                if video_id and video_id not in vids:
                    vids.append(video_id)
                self._conn.execute(
                    "UPDATE entities SET frequency = ?, last_seen = ?, "
                    "metadata = ?, video_ids = ? WHERE id = ?",
                    (freq, now, json.dumps(meta), json.dumps(vids), entity_id),
                )
            else:
                vids = [video_id] if video_id else []
                cur = self._conn.execute(
                    "INSERT INTO entities (name, type, frequency, first_seen, "
                    "last_seen, metadata, video_ids) VALUES (?, ?, 1, ?, ?, ?, ?)",
                    (name, entity_type, now, now, json.dumps(meta), json.dumps(vids)),
                )
                entity_id = cur.lastrowid
            self._conn.commit()
            return entity_id

    def add_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        video_id: str = "",
    ) -> List[int]:
        """Add multiple entities in a single transaction.

        Each dict should have keys: name (str), type (str, optional),
        metadata (dict, optional).

        Returns:
            List of entity IDs in the same order as input.
        """
        ids: List[int] = []
        with self._lock:
            for ent in entities:
                eid = self.add_entity(
                    name=ent["name"],
                    entity_type=ent.get("type", "concept"),
                    video_id=video_id,
                    metadata=ent.get("metadata"),
                )
                ids.append(eid)
            self._conn.commit()
        return ids

    def search_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_frequency: int = 1,
        limit: int = 100,
    ) -> List[EntityRecord]:
        """Search entities by name, type, and/or frequency.

        Args:
            name: Substring search against entity name (case-insensitive).
            entity_type: Filter by entity type.
            min_frequency: Minimum appearance frequency.
            limit: Maximum results.

        Returns:
            List of matching EntityRecord instances.
        """
        with self._lock:
            query = "SELECT * FROM entities WHERE frequency >= ?"
            params: List[Any] = [min_frequency]

            if name:
                query += " AND name LIKE ?"
                params.append(f"%{name}%")
            if entity_type:
                query += " AND type = ?"
                params.append(entity_type)

            query += " ORDER BY frequency DESC LIMIT ?"
            params.append(limit)

            rows = self._conn.execute(query, params).fetchall()
            return [self._row_to_entity(r) for r in rows]

    def get_entity(self, entity_id: int) -> Optional[EntityRecord]:
        """Get a single entity by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            return self._row_to_entity(row) if row else None

    def get_top_entities(
        self, entity_type: Optional[str] = None, limit: int = 50
    ) -> List[EntityRecord]:
        """Get most frequent entities, optionally filtered by type."""
        with self._lock:
            query = "SELECT * FROM entities"
            params: List[Any] = []
            if entity_type:
                query += " WHERE type = ?"
                params.append(entity_type)
            query += " ORDER BY frequency DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(query, params).fetchall()
            return [self._row_to_entity(r) for r in rows]

    def entity_count(self) -> int:
        """Total number of unique entities in the knowledge graph."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM entities").fetchone()
            return row["cnt"] if row else 0

    # ── Relationship operations ─────────────────────────────────────────

    def add_relationship(
        self,
        source_id: int,
        target_id: int,
        relation_type: str = "co_occurs",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add or strengthen a relationship between two entities.

        If the relationship already exists, increments its strength and
        updates last_seen.

        Returns:
            The relationship ID.
        """
        now = time.time()
        meta = metadata or {}
        with self._lock:
            row = self._conn.execute(
                "SELECT id, strength FROM relationships "
                "WHERE source_id = ? AND target_id = ? AND relation_type = ?",
                (source_id, target_id, relation_type),
            ).fetchone()

            if row:
                rel_id = row["id"]
                self._conn.execute(
                    "UPDATE relationships SET strength = strength + 1, "
                    "last_seen = ?, metadata = ? WHERE id = ?",
                    (now, json.dumps(meta), rel_id),
                )
            else:
                cur = self._conn.execute(
                    "INSERT INTO relationships (source_id, target_id, "
                    "relation_type, strength, last_seen, metadata) "
                    "VALUES (?, ?, ?, 1, ?, ?)",
                    (source_id, target_id, relation_type, now, json.dumps(meta)),
                )
                rel_id = cur.lastrowid
            self._conn.commit()
            return rel_id

    def get_relationships(
        self,
        entity_id: int,
        relation_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[RelationshipRecord]:
        """Get all relationships involving an entity.

        Returns both incoming and outgoing relationships.
        """
        with self._lock:
            query = (
                "SELECT * FROM relationships WHERE " "(source_id = ? OR target_id = ?)"
            )
            params: List[Any] = [entity_id, entity_id]
            if relation_type:
                query += " AND relation_type = ?"
                params.append(relation_type)
            query += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(query, params).fetchall()
            return [self._row_to_relationship(r) for r in rows]

    def get_top_relationships(
        self, relation_type: Optional[str] = None, limit: int = 50
    ) -> List[RelationshipRecord]:
        """Get strongest relationships across the entire graph."""
        with self._lock:
            query = "SELECT * FROM relationships"
            params: List[Any] = []
            if relation_type:
                query += " WHERE relation_type = ?"
                params.append(relation_type)
            query += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(query, params).fetchall()
            return [self._row_to_relationship(r) for r in rows]

    def relationship_count(self) -> int:
        """Total number of relationships in the knowledge graph."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM relationships"
            ).fetchone()
            return row["cnt"] if row else 0

    # ── Video operations ────────────────────────────────────────────────

    def add_video_record(
        self,
        video_id: str,
        filename: str = "",
        duration_seconds: float = 0.0,
        entity_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record metadata about an indexed video."""
        now = time.time()
        meta = metadata or {}
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO videos
                   (video_id, filename, duration_seconds, entity_count,
                    indexed_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    video_id,
                    filename,
                    duration_seconds,
                    entity_count,
                    now,
                    json.dumps(meta),
                ),
            )
            self._conn.commit()

    def get_video(self, video_id: str) -> Optional[VideoRecord]:
        """Get a video record by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
            return self._row_to_video(row) if row else None

    def list_videos(self, limit: int = 100) -> List[VideoRecord]:
        """List all indexed videos, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM videos ORDER BY indexed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_video(r) for r in rows]

    def video_count(self) -> int:
        """Total number of videos indexed in the knowledge graph."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM videos").fetchone()
            return row["cnt"] if row else 0

    # ── Cross-video queries ─────────────────────────────────────────────

    def get_videos_for_entity(self, name: str) -> List[str]:
        """Get all video IDs mentioning a specific entity."""
        with self._lock:
            row = self._conn.execute(
                "SELECT video_ids FROM entities WHERE name = ? "
                "ORDER BY frequency DESC LIMIT 1",
                (name,),
            ).fetchone()
            if row:
                return json.loads(row["video_ids"] or "[]")
            return []

    def get_entities_for_video(self, video_id: str) -> List[EntityRecord]:
        """Get all entities associated with a specific video."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE video_ids LIKE ? "
                "ORDER BY frequency DESC",
                (f"%{video_id}%",),
            ).fetchall()
            return [self._row_to_entity(r) for r in rows]

    def get_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get a chronological timeline of all indexed videos.

        Returns a chronologically ordered list of events/dates with video
        references, suitable for rendering in the UI or as context for LLM
        queries about "what happened when".
        """
        with self._lock:
            videos = self._conn.execute(
                "SELECT video_id, filename, duration_seconds, entity_count, "
                "indexed_at FROM videos ORDER BY indexed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            timeline = []
            for v in videos:
                # Get top entities for this video
                entities = self.get_entities_for_video(v["video_id"])
                top_entities = [e.name for e in entities[:5]]
                timeline.append(
                    {
                        "video_id": v["video_id"],
                        "filename": v["filename"] or v["video_id"],
                        "duration_seconds": v["duration_seconds"],
                        "entity_count": v["entity_count"],
                        "indexed_at": v["indexed_at"],
                        "top_entities": top_entities,
                    }
                )
            return timeline

    def cross_video_search(self, query: str, limit: int = 20) -> List[EntityRecord]:
        """Search entities across all videos, returning those matching the
        query text in name or type."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE "
                "name LIKE ? OR type LIKE ? "
                "ORDER BY frequency DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [self._row_to_entity(r) for r in rows]

    # ── Event-Causal RAG persistence (v0.58.0) ──────────────────────────

    def add_event_record(
        self,
        event_id: str,
        video_id: str,
        title: str = "",
        description: str = "",
        start_time: float = 0.0,
        end_time: float = 0.0,
        state_before: str = "",
        state_after: str = "",
        action: str = "",
        entities: Optional[List[str]] = None,
        confidence: float = 0.0,
    ) -> None:
        """Persist a single event record into the knowledge graph."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO events
                   (event_id, video_id, title, description, start_time, end_time,
                    state_before, state_after, action, entities, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    video_id,
                    title,
                    description,
                    start_time,
                    end_time,
                    state_before,
                    state_after,
                    action,
                    json.dumps(entities or []),
                    confidence,
                ),
            )
            self._conn.commit()

    def add_causal_relation(
        self,
        source_event_id: str,
        target_event_id: str,
        relation_type: str = "temporal",
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a causal or temporal relationship between two events."""
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO causal_relations
                       (source_event_id, target_event_id, relation_type, strength, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        source_event_id,
                        target_event_id,
                        relation_type,
                        strength,
                        json.dumps(metadata or {}),
                    ),
                )
                self._conn.commit()
            except Exception as exc:
                logger.debug("add_causal_relation skipped: %s", exc)

    def get_events_for_video(self, video_id: str) -> List[Dict[str, Any]]:
        """Get all events for a video, ordered by start_time."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE video_id = ? ORDER BY start_time ASC",
                (video_id,),
            ).fetchall()
            result = []
            for r in rows:
                result.append(
                    {
                        "event_id": r["event_id"],
                        "video_id": r["video_id"],
                        "title": r["title"],
                        "description": r["description"],
                        "start_time": r["start_time"],
                        "end_time": r["end_time"],
                        "state_before": r["state_before"],
                        "state_after": r["state_after"],
                        "action": r["action"],
                        "entities": json.loads(r["entities"] or "[]"),
                        "confidence": r["confidence"],
                    }
                )
            return result

    def get_causal_relations(
        self, video_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get causal/temporal relations between events."""
        with self._lock:
            if video_id:
                rows = self._conn.execute(
                    """SELECT cr.* FROM causal_relations cr
                       JOIN events e ON cr.source_event_id = e.event_id
                       WHERE e.video_id = ?
                       ORDER BY cr.id DESC LIMIT ?""",
                    (video_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM causal_relations ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def delete_events_for_video(self, video_id: str) -> None:
        """Remove all events and causal relations for a video."""
        with self._lock:
            self._conn.execute(
                """DELETE FROM causal_relations WHERE source_event_id IN
                   (SELECT event_id FROM events WHERE video_id = ?)
                   OR target_event_id IN
                   (SELECT event_id FROM events WHERE video_id = ?)""",
                (video_id, video_id),
            )
            self._conn.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
            self._conn.commit()

    def persist_events_from_rag(
        self,
        rag_instance,
        video_id: str,
    ) -> int:
        """Persist all indexed events and causal relations from EventCausalRAG.

        Returns the number of events persisted.
        """
        if not hasattr(rag_instance, "_event_rag_instance"):
            return 0
        event_rag = rag_instance._event_rag_instance
        if event_rag is None:
            return 0
        events = getattr(event_rag, "_events", [])
        if not events:
            return 0

        count = 0
        for evt in events:
            self.add_event_record(
                event_id=evt.event_id,
                video_id=evt.video_id,
                title=evt.title or "",
                description=evt.description or "",
                start_time=evt.start_time or 0.0,
                end_time=evt.end_time or 0.0,
                state_before=evt.state_before or "",
                state_after=evt.state_after or "",
                action=evt.action or "",
                entities=list(evt.entities) if evt.entities else [],
                confidence=evt.confidence or 0.0,
            )
            count += 1

        # Persist causal/temporal edges from SESGraph
        ses = getattr(event_rag, "_ses_graph", None)
        if ses is not None:
            for src_id, edges in getattr(ses, "forward_edges", {}).items():
                for tgt_id, rel_type in edges:
                    self.add_causal_relation(
                        source_event_id=src_id,
                        target_event_id=tgt_id,
                        relation_type=rel_type,
                        strength=1.0 if rel_type == "causal" else 0.5,
                    )

        logger.info(
            "KnowledgeGraph: persisted %d events + causal edges for %s",
            count,
            video_id,
        )
        return count

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the knowledge graph."""
        with self._lock:
            ent_count = self.entity_count()
            rel_count = self.relationship_count()
            vid_count = self.video_count()
            # Type breakdown
            type_rows = self._conn.execute(
                "SELECT type, COUNT(*) AS cnt FROM entities "
                "GROUP BY type ORDER BY cnt DESC"
            ).fetchall()
            type_breakdown = {r["type"]: r["cnt"] for r in type_rows}
            # Storage stats
            db_size = self._db_path.stat().st_size if self._db_path.exists() else 0
            # Last indexed
            last_vid = self._conn.execute(
                "SELECT video_id, indexed_at FROM videos "
                "ORDER BY indexed_at DESC LIMIT 1"
            ).fetchone()

            return {
                "entity_count": ent_count,
                "relationship_count": rel_count,
                "video_count": vid_count,
                "type_breakdown": type_breakdown,
                "database_size_bytes": db_size,
                "last_indexed_video": dict(last_vid) if last_vid else None,
            }

    def get_knowledge_context(self, limit_entities: int = 100) -> str:
        """Generate a compact LLM-friendly text summary of the entire
        knowledge graph for injection into chat context.

        Returns a markdown-formatted string that can be prepended to
        RAG prompts to give the LLM awareness of what's in the system.
        """
        stats = self.stats()
        top_entities = self.get_top_entities(limit=limit_entities)
        top_rels = self.get_top_relationships(limit=30)

        lines = [
            f"## Video Knowledge Graph Summary",
            f"",
            f"- **Videos indexed**: {stats['video_count']}",
            f"- **Unique entities**: {stats['entity_count']}",
            f"- **Relationships**: {stats['relationship_count']}",
        ]

        if stats["type_breakdown"]:
            lines.append(f"- **Entity types**:")
            for t, c in stats["type_breakdown"].items():
                lines.append(f"  - {t}: {c}")

        if top_entities:
            lines.append(f"")
            lines.append(f"### Top entities")
            for e in top_entities[:20]:
                lines.append(
                    f"- **{e.name}** ({e.entity_type}) — seen {e.frequency}× "
                    f"in {len(e.video_ids)} video(s)"
                )

        if top_rels:
            lines.append(f"")
            lines.append(f"### Strongest relationships")
            for r in top_rels[:15]:
                src = self.get_entity(r.source_id)
                tgt = self.get_entity(r.target_id)
                src_name = src.name if src else f"#{r.source_id}"
                tgt_name = tgt.name if tgt else f"#{r.target_id}"
                lines.append(
                    f"- **{src_name}** → **{tgt_name}** "
                    f"({r.relation_type}, strength={r.strength})"
                )

        lines.append("")
        return "\n".join(lines)

    # ── Maintenance ─────────────────────────────────────────────────────

    def vacuum(self) -> None:
        """Reclaim storage — reduces database file size after deletions."""
        with self._lock:
            self._conn.execute("PRAGMA optimize")
            self._conn.execute("VACUUM")
            self._conn.commit()

    def clear(self) -> None:
        """Delete all data from the knowledge graph (irreversible)."""
        with self._lock:
            self._conn.execute("DELETE FROM relationships")
            self._conn.execute("DELETE FROM entities")
            self._conn.execute("DELETE FROM videos")
            self._conn.commit()
            self.vacuum()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.execute("PRAGMA optimize")
                self._conn.close()
                self._conn = None

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> EntityRecord:
        return EntityRecord(
            id=row["id"],
            name=row["name"],
            entity_type=row["type"],
            frequency=row["frequency"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            metadata=json.loads(row["metadata"] or "{}"),
            video_ids=set(json.loads(row["video_ids"] or "[]")),
        )

    @staticmethod
    def _row_to_relationship(row: sqlite3.Row) -> RelationshipRecord:
        return RelationshipRecord(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            strength=row["strength"],
            last_seen=row["last_seen"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_video(row: sqlite3.Row) -> VideoRecord:
        return VideoRecord(
            video_id=row["video_id"],
            filename=row["filename"],
            duration_seconds=row["duration_seconds"],
            entity_count=row["entity_count"],
            indexed_at=row["indexed_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
