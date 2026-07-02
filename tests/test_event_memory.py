"""
Unit tests for the EventMemory (SQLite-backed event store with LLM RAG).
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from video_analysis.event_memory import EventMemory, StoredEvent


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_path():
    """Yield a temporary path (no file created until EventMemory uses it)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    Path(path).unlink()  # delete the file mkstemp created
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def mem(db_path):
    """Return a fresh EventMemory on a temporary DB."""
    em = EventMemory(db_path=db_path, retention_days=30)
    yield em
    em.close()


# ── Init ──────────────────────────────────────────────────────────────────────


def test_init_creates_db_and_tables(db_path):
    """EventMemory creates the SQLite file and the events table."""
    assert not Path(db_path).exists()

    em = EventMemory(db_path=db_path)
    try:
        assert Path(db_path).exists(), "DB file was not created"
        # Verify the table is queryable by inserting a row
        em.store("test_cam", time.time(), [{"label": "cat", "count": 1}])
        events = em.query_time_range("test_cam", 0, time.time() + 10)
        assert len(events) == 1
    finally:
        em.close()


def test_init_retention_days_stored(db_path):
    """retention_days is honoured (tested via purge below)."""
    em = EventMemory(db_path=db_path, retention_days=7)
    assert em._retention_days == 7
    em.close()


# ── store ─────────────────────────────────────────────────────────────────────


def test_store_returns_positive_id(mem):
    """store() returns a positive integer event ID."""
    eid = mem.store("cam1", 1000.0, [{"label": "car", "count": 1}])
    assert isinstance(eid, int)
    assert eid > 0


def test_store_increments_id(mem):
    """Consecutive store() calls return ascending IDs."""
    ids = [
        mem.store("cam1", 2000.0, [{"label": "person", "count": 1}])
        for _ in range(3)
    ]
    assert ids == [1, 2, 3], f"Expected [1, 2, 3], got {ids}"


def test_store_auto_computes_object_summary_and_people_count(mem):
    """store derives object_summary and people_count from the objects list."""
    objects = [
        {"label": "person", "count": 2},
        {"label": "car", "count": 1},
        {"label": "dog", "count": 1},
    ]
    mem.store("cam1", 1500.0, objects)

    events = mem.query_time_range("cam1", 0, 9999)
    assert len(events) == 1
    ev = events[0]
    assert ev.object_summary == "2x person, 1x car, 1x dog"
    assert ev.people_count == 2


def test_store_empty_objects_list(mem):
    """store handles an empty objects list gracefully."""
    eid = mem.store("cam2", 3000.0, [])
    assert eid > 0

    events = mem.query_time_range("cam2", 0, 9999)
    assert len(events) == 1
    assert events[0].object_summary == ""
    assert events[0].people_count == 0
    assert events[0].objects == []


def test_store_defaults(mem):
    """store uses sensible defaults for optional parameters."""
    mem.store("cam1", 100.0, [{"label": "x"}])
    ev = mem.query_time_range("cam1", 0, 9999)[0]
    assert ev.motion_score == 0.0
    assert ev.triggered_by == ""
    assert ev.frame_path == ""
    assert ev.description == ""


# ── query_time_range ──────────────────────────────────────────────────────────


def test_query_time_range_returns_events_in_range(mem):
    """Only events whose timestamp falls between start_ts and end_ts are returned."""
    mem.store("cam1", 100.0, [{"label": "a"}])
    mem.store("cam1", 200.0, [{"label": "b"}])
    mem.store("cam1", 300.0, [{"label": "c"}])

    result = mem.query_time_range("cam1", 150.0, 250.0)
    assert len(result) == 1
    assert result[0].timestamp == 200.0


def test_query_time_range_returns_newest_first(mem):
    """Results are ordered by timestamp descending."""
    mem.store("cam1", 100.0, [{"label": "a"}])
    mem.store("cam1", 200.0, [{"label": "b"}])
    mem.store("cam1", 300.0, [{"label": "c"}])

    result = mem.query_time_range("cam1", 0, 9999)
    assert [ev.timestamp for ev in result] == [300.0, 200.0, 100.0]


def test_query_time_range_respects_limit(mem):
    """The limit parameter caps the number of returned events."""
    for i in range(10):
        mem.store("cam1", float(i * 10), [{"label": "x"}])

    result = mem.query_time_range("cam1", 0, 9999, limit=3)
    assert len(result) == 3


def test_query_time_range_boundary_inclusive(mem):
    """Events exactly at the boundary timestamps are included."""
    mem.store("cam1", 100.0, [{"label": "a"}])
    mem.store("cam1", 200.0, [{"label": "b"}])

    result = mem.query_time_range("cam1", 100.0, 200.0)
    assert len(result) == 2


def test_query_time_range_empty(mem):
    """No events in range returns an empty list."""
    result = mem.query_time_range("cam1", 0, 1)
    assert result == []


# ── query_by_object ───────────────────────────────────────────────────────────


def test_query_by_object_finds_events_with_label(mem):
    """Events whose object_summary contains the label are returned."""
    mem.store("cam1", 100.0, [{"label": "person", "count": 1}])
    mem.store("cam1", 200.0, [{"label": "car", "count": 1}])
    mem.store("cam1", 300.0, [{"label": "person", "count": 2}])

    result = mem.query_by_object("cam1", "person")
    assert len(result) == 2
    assert all("person" in ev.object_summary for ev in result)


def test_query_by_object_returns_newest_first(mem):
    """query_by_object results are ordered by timestamp descending."""
    mem.store("cam1", 100.0, [{"label": "person", "count": 1}])
    mem.store("cam1", 200.0, [{"label": "person", "count": 1}])
    mem.store("cam1", 300.0, [{"label": "person", "count": 1}])

    result = mem.query_by_object("cam1", "person")
    assert [ev.timestamp for ev in result] == [300.0, 200.0, 100.0]


def test_query_by_object_no_match(mem):
    """A label not present in any event returns an empty list."""
    mem.store("cam1", 100.0, [{"label": "Person", "count": 1}])

    result = mem.query_by_object("cam1", "NoSuchLabel")
    assert result == []


def test_query_by_object_respects_limit(mem):
    """query_by_object honours the limit parameter."""
    for _ in range(10):
        mem.store("cam1", 200.0, [{"label": "person", "count": 1}])

    result = mem.query_by_object("cam1", "person", limit=3)
    assert len(result) == 3


def test_query_by_object_other_stream_unaffected(mem):
    """Events from other streams are not returned."""
    mem.store("cam1", 100.0, [{"label": "person", "count": 1}])
    mem.store("cam2", 200.0, [{"label": "person", "count": 1}])

    result = mem.query_by_object("cam1", "person")
    assert len(result) == 1
    assert result[0].stream_id == "cam1"


# ── query_natural_language ────────────────────────────────────────────────────


def test_query_natural_language_calls_llm_with_context(mem):
    """query_natural_language invokes llm_chat_fn with a RAG prompt containing
    recent event context and the user's question."""
    now = time.time()
    mem.store("cam1", now - 100, [{"label": "person", "count": 1}],
              description="person walking")
    mem.store("cam1", now - 50, [{"label": "car", "count": 1}],
              description="car parking")
    mem.store("cam1", now - 10, [{"label": "dog", "count": 1}],
              description="dog running")

    llm_mock = MagicMock(return_value="There was one person, one car, and one dog.")

    with patch("time.time", return_value=now):
        answer = mem.query_natural_language("cam1", "What did you see?", llm_mock)

    assert answer == "There was one person, one car, and one dog."
    llm_mock.assert_called_once()
    call_args = llm_mock.call_args[0][0]
    assert len(call_args) == 1  # single user message
    assert call_args[0]["role"] == "user"
    content = call_args[0]["content"]
    # Context lines should appear, along with the question
    assert "1x person" in content
    assert "1x car" in content
    assert "1x dog" in content
    assert "What did you see?" in content


def test_query_natural_language_empty_store(mem):
    """query_natural_language works with no stored events."""
    llm_mock = MagicMock(return_value="Nothing to report.")

    with patch("time.time", return_value=time.time()):
        answer = mem.query_natural_language("cam1", "Any activity?", llm_mock)

    assert answer == "Nothing to report."
    llm_mock.assert_called_once()


# ── Retention ─────────────────────────────────────────────────────────────────


def test_retention_purges_old_events(db_path):
    """Events older than retention_days are purged on init."""
    em = EventMemory(db_path=db_path, retention_days=30)

    old_ts = time.time() - 100 * 86400  # 100 days ago
    recent_ts = time.time() - 3600  # 1 hour ago
    em.store("cam1", old_ts, [{"label": "old", "count": 1}])
    em.store("cam1", recent_ts, [{"label": "recent", "count": 1}])
    em.close()

    # Reopen with a short retention that should only keep the recent event
    em2 = EventMemory(db_path=db_path, retention_days=2)
    try:
        events = em2.query_time_range("cam1", 0, time.time() + 86400)
        labels = [ev.object_summary for ev in events]
        assert "1x recent" in labels, "Recent event should survive"
        assert "1x old" not in labels, "Old event should have been purged"
    finally:
        em2.close()


# ── Multiple streams ──────────────────────────────────────────────────────────


def test_multiple_streams_independent(mem):
    """Events under different stream_ids are stored independently."""
    mem.store("cam_front", 100.0, [{"label": "person", "count": 1}])
    mem.store("cam_back", 200.0, [{"label": "car", "count": 1}])
    mem.store("cam_front", 300.0, [{"label": "cat", "count": 1}])

    front_events = mem.query_time_range("cam_front", 0, 9999)
    assert len(front_events) == 2
    assert all(e.stream_id == "cam_front" for e in front_events)

    back_events = mem.query_time_range("cam_back", 0, 9999)
    assert len(back_events) == 1
    assert back_events[0].stream_id == "cam_back"


def test_query_time_range_stream_isolation(mem):
    """query_time_range for one stream never returns another stream's events."""
    mem.store("stream_a", 100.0, [{"label": "a"}])
    mem.store("stream_b", 200.0, [{"label": "b"}])

    result = mem.query_time_range("stream_b", 0, 9999)
    assert all(ev.stream_id == "stream_b" for ev in result)


# ── close ─────────────────────────────────────────────────────────────────────


def test_close_is_safe(mem):
    """Calling close() multiple times does not raise."""
    mem.close()
    mem.close()  # second call — should be a no-op


def test_close_allows_reopening(db_path):
    """Closing and reopening the same DB file with recent timestamps works."""
    em = EventMemory(db_path=db_path)
    em.store("cam1", time.time() - 10, [{"label": "a"}])
    em.close()

    em2 = EventMemory(db_path=db_path)
    try:
        events = em2.query_time_range("cam1", 0, time.time() + 86400)
        assert len(events) == 1
        assert events[0].object_summary == "1x a"
    finally:
        em2.close()


# ── StoredEvent dataclass ─────────────────────────────────────────────────────


def test_stored_event_defaults():
    """StoredEvent default values match expectations."""
    ev = StoredEvent(id=1, stream_id="cam", timestamp=100.0)
    assert ev.objects == []
    assert ev.object_summary == ""
    assert ev.people_count == 0
    assert ev.motion_score == 0.0
    assert ev.triggered_by == ""
    assert ev.frame_path is None
    assert ev.description is None
    assert ev.summary is None
