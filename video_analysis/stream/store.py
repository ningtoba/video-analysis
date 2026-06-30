"""
Event store — SQLite-backed persistent timeline of LLM-analyzed events.

Each event has: timestamp, optional frame path, LLM description, metadata.
Supports time-range queries and automatic retention.
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
class TimelineEvent:
    """A single event in the timeline."""
    id: int = 0
    stream_id: str = ""
    timestamp: float = 0.0
    description: str = ""
    frame_path: Optional[str] = None
    motion_score: float = 0.0
    triggered_by: str = "periodic"  # "periodic", "motion", "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


class EventStore:
    """SQLite-backed append-only event timeline with automatic retention."""

    def __init__(self, db_path: str = "", retention_days: int = 30):
        self._db_path = db_path or str(Path.cwd() / "data" / "stream_events.db")
        self._retention_days = retention_days
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                frame_path TEXT,
                motion_score REAL DEFAULT 0.0,
                triggered_by TEXT DEFAULT 'periodic',
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_stream_ts
            ON events(stream_id, timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_created
            ON events(created_at)
        """)
        self._conn.commit()

        # Apply retention on init
        self._apply_retention()

    def _apply_retention(self):
        if self._retention_days <= 0 or not self._conn:
            return
        cutoff = time.time() - self._retention_days * 86400
        deleted = self._conn.execute(
            "DELETE FROM events WHERE created_at < ?", (cutoff,)
        ).rowcount
        self._conn.commit()
        if deleted:
            logger.info("Retention: deleted %d events older than %d days", deleted, self._retention_days)

    def add_event(
        self,
        stream_id: str,
        timestamp: float,
        description: str,
        frame_path: Optional[str] = None,
        motion_score: float = 0.0,
        triggered_by: str = "periodic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add an event to the timeline. Returns event ID."""
        if not self._conn:
            return -1

        cur = self._conn.execute(
            """INSERT INTO events (stream_id, timestamp, description, frame_path,
               motion_score, triggered_by, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                stream_id, timestamp, description, frame_path,
                motion_score, triggered_by,
                json.dumps(metadata or {}),
                time.time(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid or -1

    def get_recent(self, stream_id: str, limit: int = 50) -> List[TimelineEvent]:
        """Get most recent events for a stream."""
        if not self._conn:
            return []
        rows = self._conn.execute(
            """SELECT id, stream_id, timestamp, description, frame_path,
               motion_score, triggered_by, metadata, created_at
               FROM events WHERE stream_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (stream_id, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in reversed(rows)]

    def get_range(
        self,
        stream_id: str,
        start_ts: float,
        end_ts: float,
    ) -> List[TimelineEvent]:
        """Get events within a time range."""
        if not self._conn:
            return []
        rows = self._conn.execute(
            """SELECT id, stream_id, timestamp, description, frame_path,
               motion_score, triggered_by, metadata, created_at
               FROM events WHERE stream_id = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC""",
            (stream_id, start_ts, end_ts),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_latest_event(self, stream_id: str) -> Optional[TimelineEvent]:
        """Get the most recent event."""
        if not self._conn:
            return None
        row = self._conn.execute(
            """SELECT id, stream_id, timestamp, description, frame_path,
               motion_score, triggered_by, metadata, created_at
               FROM events WHERE stream_id = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (stream_id,),
        ).fetchone()
        return self._row_to_event(row) if row else None

    def get_event_count(self, stream_id: str) -> int:
        if not self._conn:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE stream_id = ?", (stream_id,)
        ).fetchone()
        return row[0] if row else 0

    def get_streams(self) -> List[str]:
        """List all stream IDs with events."""
        if not self._conn:
            return []
        rows = self._conn.execute(
            "SELECT DISTINCT stream_id FROM events ORDER BY stream_id"
        ).fetchall()
        return [r[0] for r in rows]

    def _row_to_event(self, row: tuple) -> TimelineEvent:
        return TimelineEvent(
            id=row[0], stream_id=row[1], timestamp=row[2],
            description=row[3] or "", frame_path=row[4],
            motion_score=row[5] or 0.0, triggered_by=row[6] or "periodic",
            metadata=json.loads(row[7] or "{}"), created_at=row[8] or 0.0,
        )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
