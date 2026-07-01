"""
Event memory — stores detected objects and events in SQLite with
time-range and text-based retrieval. Provides LLM RAG for chat queries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StoredEvent:
    id: int
    stream_id: str
    timestamp: float
    objects: List[Dict[str, Any]] = field(default_factory=list)
    object_summary: str = ""
    people_count: int = 0
    motion_score: float = 0.0
    triggered_by: str = ""
    frame_path: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None


class EventMemory:
    """Persistent event memory using SQLite with text search and LLM RAG."""

    def __init__(self, db_path: str = "", retention_days: int = 30):
        self._db_path = db_path or "data/event_memory.db"
        self._retention_days = retention_days
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                objects TEXT DEFAULT '[]',
                object_summary TEXT DEFAULT '',
                people_count INTEGER DEFAULT 0,
                motion_score REAL DEFAULT 0.0,
                triggered_by TEXT DEFAULT '',
                frame_path TEXT DEFAULT NULL,
                description TEXT DEFAULT NULL,
                summary TEXT DEFAULT NULL,
                created_at REAL DEFAULT (strftime('%%s','now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_stream
            ON events(stream_id, timestamp)
        """)
        self._conn.commit()
        self._purge_old()

    def store(self, stream_id: str, timestamp: float, objects: List[Dict],
              motion_score: float = 0.0, triggered_by: str = "",
              frame_path: str = "", description: str = "") -> int:
        """Store a detection event. Returns event ID."""
        object_summary = ", ".join(
            f"{o.get('count', 1)}x {o['label']}"
            for o in objects
        ) if objects else ""
        people_count = sum(
            o.get('count', 1) for o in objects
            if o.get('label') in ('person', 'people')
        )

        cursor = self._conn.execute("""
            INSERT INTO events
            (stream_id, timestamp, objects, object_summary, people_count,
             motion_score, triggered_by, frame_path, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stream_id, timestamp,
            json.dumps(objects), object_summary,
            people_count, motion_score, triggered_by,
            frame_path, description,
        ))
        self._conn.commit()
        return cursor.lastrowid

    def query_time_range(self, stream_id: str,
                         start_ts: float, end_ts: float,
                         limit: int = 100) -> List[StoredEvent]:
        """Get events in a time range."""
        cursor = self._conn.execute("""
            SELECT * FROM events
            WHERE stream_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (stream_id, start_ts, end_ts, limit))
        return [self._row_to_event(r) for r in cursor.fetchall()]

    def query_by_object(self, stream_id: str, object_label: str,
                        limit: int = 100) -> List[StoredEvent]:
        """Get events containing a specific object."""
        cursor = self._conn.execute("""
            SELECT * FROM events
            WHERE stream_id = ? AND object_summary LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (stream_id, f"%{object_label}%", limit))
        return [self._row_to_event(r) for r in cursor.fetchall()]

    def query_natural_language(self, stream_id: str, question: str,
                               llm_chat_fn) -> str:
        """Answer a question using RAG over the event memory.

        1. Parse the question for time/object hints
        2. Query the database
        3. Format events as text context
        4. Send to LLM for answer
        """
        # Default: get recent events
        recent = self.query_time_range(
            stream_id,
            time.time() - 86400,  # last 24 hours
            time.time(),
            limit=50,
        )

        # Format as text
        context_lines = ["Recent CCTV footage analysis (newest first):"]
        for ev in recent[:30]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.timestamp))
            line = f"[{ts}] {ev.object_summary}"
            if ev.people_count:
                line += f" | {ev.people_count} people"
            if ev.description:
                line += f" | {ev.description[:100]}"
            context_lines.append(line)

        context = "\n".join(context_lines)

        prompt = f"""You are a CCTV security analyst. Answer questions based on the detected events.

Event timeline:
{context}

Question: {question}

Provide a concise answer with timestamps where relevant."""

        return llm_chat_fn([{"role": "user", "content": prompt}])

    def _row_to_event(self, row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            id=row["id"],
            stream_id=row["stream_id"],
            timestamp=row["timestamp"],
            objects=json.loads(row["objects"]),
            object_summary=row["object_summary"],
            people_count=row["people_count"],
            motion_score=row["motion_score"],
            triggered_by=row["triggered_by"],
            frame_path=row["frame_path"],
            description=row["description"],
            summary=row["summary"],
        )

    def _purge_old(self):
        cutoff = time.time() - self._retention_days * 86400
        self._conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
