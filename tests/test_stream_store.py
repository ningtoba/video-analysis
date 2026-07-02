"""
Tests for the SQLite-backed EventStore timeline.
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from video_analysis.stream.store import EventStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path():
    """Provide a temporary database path (cleaned up after the test)."""
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "test_events.db")


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def test_init_creates_db_and_tables(db_path):
    """EventStore initialisation creates the database file and schema tables."""
    store = EventStore(db_path=db_path, retention_days=0)
    store.close()

    assert Path(db_path).exists(), "Database file was not created"

    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
    ).fetchall()
    assert len(tables) == 1, "events table was not created"

    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    index_names = {r[0] for r in indexes}
    assert "idx_events_stream_ts" in index_names
    assert "idx_events_created" in index_names
    conn.close()


# ---------------------------------------------------------------------------
# add_event
# ---------------------------------------------------------------------------


def test_add_event_returns_positive_id(db_path):
    """add_event returns a positive integer ID."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        event_id = store.add_event(
            stream_id="test_stream",
            timestamp=time.time(),
            description="test event",
        )
        assert event_id > 0, f"Expected positive ID, got {event_id}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# get_recent
# ---------------------------------------------------------------------------


def test_get_recent_returns_events_in_ascending_order(db_path):
    """get_recent returns events ordered by timestamp ascending (oldest first)."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        ts = time.time()
        store.add_event("s1", ts - 10, "oldest")
        store.add_event("s1", ts, "newest")
        store.add_event("s1", ts - 5, "middle")

        recent = store.get_recent("s1", limit=10)
        assert len(recent) == 3

        # Ascending timestamps (oldest first)
        timestamps = [e.timestamp for e in recent]
        assert timestamps == sorted(timestamps), (
            f"Expected ascending order, got {timestamps}"
        )
        assert recent[0].description == "oldest"
        assert recent[-1].description == "newest"
    finally:
        store.close()


def test_get_recent_respects_limit(db_path):
    """get_recent truncates results to the specified limit, oldest-first."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        ts = time.time()
        for i in range(10):
            store.add_event("s1", ts - i, f"event_{i}")

        limited = store.get_recent("s1", limit=3)
        assert len(limited) == 3, f"Expected 3 events, got {len(limited)}"

        # 3 newest (event_0 is newest at ts, event_1 at ts-1, etc.),
        # returned oldest-first
        descriptions = [e.description for e in limited]
        assert descriptions == ["event_2", "event_1", "event_0"]
    finally:
        store.close()


def test_get_recent_empty_stream(db_path):
    """get_recent returns empty list for a stream with no events."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        assert store.get_recent("nonexistent") == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# get_range
# ---------------------------------------------------------------------------


def test_get_range_filters_by_time_boundaries(db_path):
    """get_range returns only events with timestamp in [start_ts, end_ts]."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        base = time.time()
        store.add_event("s1", base - 10, "too_early")
        store.add_event("s1", base, "boundary_low")
        store.add_event("s1", base + 10, "middle")
        store.add_event("s1", base + 20, "boundary_high")
        store.add_event("s1", base + 30, "too_late")

        result = store.get_range("s1", start_ts=base, end_ts=base + 20)
        descriptions = sorted(e.description for e in result)
        assert descriptions == ["boundary_high", "boundary_low", "middle"], (
            f"Expected 3 events in range, got {descriptions}"
        )
    finally:
        store.close()


def test_get_range_empty_result(db_path):
    """get_range returns empty list when nothing matches the time window."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        store.add_event("s1", time.time(), "event")
        result = store.get_range("s1", start_ts=9999999999, end_ts=9999999999 + 1)
        assert result == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# get_latest_event
# ---------------------------------------------------------------------------


def test_get_latest_event_returns_most_recent(db_path):
    """get_latest_event returns the event with the highest timestamp."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        ts = time.time()
        store.add_event("s1", ts - 20, "old")
        store.add_event("s1", ts, "new")
        store.add_event("s1", ts - 10, "middle")

        latest = store.get_latest_event("s1")
        assert latest is not None
        assert latest.description == "new"
        assert latest.timestamp == ts
    finally:
        store.close()


def test_get_latest_event_returns_none_for_empty_stream(db_path):
    """get_latest_event returns None when a stream has no events."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        store.add_event("other_stream", time.time(), "event")
        assert store.get_latest_event("empty_stream") is None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# get_event_count
# ---------------------------------------------------------------------------


def test_get_event_count_returns_correct_count(db_path):
    """get_event_count matches the number of events added."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        assert store.get_event_count("s1") == 0

        store.add_event("s1", time.time(), "e1")
        assert store.get_event_count("s1") == 1

        store.add_event("s1", time.time(), "e2")
        store.add_event("s1", time.time(), "e3")
        assert store.get_event_count("s1") == 3
    finally:
        store.close()


# ---------------------------------------------------------------------------
# get_streams
# ---------------------------------------------------------------------------


def test_get_streams_lists_distinct_stream_ids(db_path):
    """get_streams returns a sorted list of distinct stream IDs."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        assert store.get_streams() == []

        store.add_event("z_stream", time.time(), "z")
        store.add_event("a_stream", time.time(), "a")
        store.add_event("m_stream", time.time(), "m")
        store.add_event("a_stream", time.time(), "a2")  # duplicate

        streams = store.get_streams()
        assert streams == ["a_stream", "m_stream", "z_stream"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_retention_prunes_old_events_on_init(db_path):
    """Events older than retention_days are deleted when a new EventStore opens."""
    # Step 1: Create a store with retention disabled to insert events
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        # Insert a recent event (normal path)
        store.add_event("s1", time.time(), "recent")

        # Insert an old event by writing directly into the database
        # with a created_at far in the past (100 000 seconds ~1.16 days ago)
        old_ts = time.time() - 100_000
        store._conn.execute(
            """INSERT INTO events (stream_id, timestamp, description, created_at)
               VALUES (?, ?, ?, ?)""",
            ("s1", old_ts, "old_event", old_ts),
        )
        store._conn.commit()
    finally:
        store.close()

    # Step 2: Re-open with retention_days=1 — old event should be pruned
    store2 = EventStore(db_path=db_path, retention_days=1)
    try:
        count = store2.get_event_count("s1")
        assert count == 1, (
            f"Expected 1 event (recent) after retention pruning, got {count}"
        )
        remaining = store2.get_recent("s1")
        assert remaining[0].description == "recent"
    finally:
        store2.close()


def test_retention_disabled_with_non_positive_days(db_path):
    """Non-positive retention_days disables pruning."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        store.add_event("s1", time.time(), "e1")
        store.add_event("s1", time.time(), "e2")
    finally:
        store.close()

    # Re-open with retention_days=0 — nothing pruned
    store2 = EventStore(db_path=db_path, retention_days=0)
    try:
        assert store2.get_event_count("s1") == 2
    finally:
        store2.close()

    # Re-open with retention_days=-1 — nothing pruned
    store3 = EventStore(db_path=db_path, retention_days=-1)
    try:
        assert store3.get_event_count("s1") == 2
    finally:
        store3.close()


# ---------------------------------------------------------------------------
# Multiple streams isolation
# ---------------------------------------------------------------------------


def test_multiple_streams_do_not_interfere(db_path):
    """Events for different streams are isolated from each other's queries."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        ts = time.time()
        store.add_event("camera_1", ts, "cam1_event")
        store.add_event("camera_2", ts, "cam2_first")
        store.add_event("camera_2", ts + 1, "cam2_second")

        assert store.get_event_count("camera_1") == 1
        assert store.get_event_count("camera_2") == 2

        assert store.get_latest_event("camera_1").description == "cam1_event"
        assert store.get_latest_event("camera_2").description == "cam2_second"

        assert len(store.get_recent("camera_1", limit=10)) == 1
        assert len(store.get_recent("camera_2", limit=10)) == 2

        assert len(store.get_range("camera_1", ts - 1, ts + 1)) == 1
        assert len(store.get_range("unknown", ts - 1, ts + 1)) == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_is_idempotent(db_path):
    """Calling close() multiple times does not raise."""
    store = EventStore(db_path=db_path, retention_days=0)
    store.add_event("s1", time.time(), "e1")
    store.close()
    store.close()


# ---------------------------------------------------------------------------
# _row_to_event round-trip
# ---------------------------------------------------------------------------


def test_event_metadata_round_trip(db_path):
    """Metadata dict survives a write->read cycle via JSON serialisation."""
    store = EventStore(db_path=db_path, retention_days=0)
    try:
        meta = {"detected": ["person", "car"], "confidence": 0.95, "tags": []}
        event_id = store.add_event(
            stream_id="s1",
            timestamp=42.0,
            description="roundtrip",
            motion_score=0.75,
            triggered_by="motion",
            metadata=meta,
        )
        events = store.get_recent("s1")
        assert len(events) == 1
        assert events[0].id == event_id
        assert events[0].description == "roundtrip"
        assert events[0].motion_score == 0.75
        assert events[0].triggered_by == "motion"
        assert events[0].metadata == meta
    finally:
        store.close()
