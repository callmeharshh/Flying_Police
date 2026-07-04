import os

import pytest


from query.query_engine import QueryEngine
from storage.event_store import EventStore
from storage.frame_index import FrameIndex


@pytest.fixture
def index(tmp_path):
    return FrameIndex(chroma_dir=str(tmp_path / "chroma"))


@pytest.fixture
def store(tmp_path):
    return EventStore(db_path=str(tmp_path / "events.db"))


@pytest.fixture
def engine(store, index):
    return QueryEngine(store, index)


def _add_person_frame(index, frame_id, description, track_id):
    index.add_frame(
        frame_id=frame_id,
        description=description,
        timestamp=f"2026-06-10T20:0{frame_id}:00+00:00",
        location="main_gate",
        objects=["person"],
        threat_level="high",
        bbox=(100 + frame_id, 200, 50, 80),
        track_id=track_id,
    )


def test_count_query_merges_fragmented_tracks_into_one_individual(engine, index):
    """Same man re-tracked as track_001/002/003 at one location → 1 distinct person."""
    _add_person_frame(index, 14, "a man is seen in this surveillance image", "track_001")
    _add_person_frame(index, 16, "a man is seen in this surveillance image", "track_001")
    _add_person_frame(index, 22, "a man is seen in a surveillance photo", "track_002")
    _add_person_frame(index, 33, "a man is sitting on the floor in front of a door", "track_003")

    results = engine.query("How many men were detected?")
    assert len(results) == 1
    assert results[0]["_type"] == "count"
    assert results[0]["unique_count"] == 1
    assert results[0]["track_count"] == 3
    assert results[0]["frame_count"] == 4


def test_merge_keeps_separate_individuals_when_far_apart_same_frame():
    """Two people visible at once with distant centers stay as 2 individuals."""
    from query.track_merge import merge_tracks_into_individuals

    frames = [
        {"frame_id": 10, "location": "main_gate", "description": "man left",
         "center_x": 125.0, "center_y": 240.0, "track_id": "track_a"},
        {"frame_id": 10, "location": "main_gate", "description": "man right",
         "center_x": 625.0, "center_y": 240.0, "track_id": "track_b"},
    ]
    _, individuals = merge_tracks_into_individuals(frames)
    assert len(individuals) == 2


def test_count_query_formats_clear_answer(engine, index):
    _add_person_frame(index, 14, "a man is seen in this surveillance image", "track_001")
    _add_person_frame(index, 22, "a man is seen in a surveillance photo", "track_002")

    output = engine.format_results(engine.query("how many men were detected?"))
    assert "Answer: 1 distinct person(s) in the video" in output
    assert "track_001" in output
    assert "track_002" in output
    assert "score=" not in output


def test_empty_alerts_query_explains_no_results(engine, store):
    output = engine.format_results(engine.query("any alerts?"))
    assert "No rule alerts were triggered" in output
    assert "No results found" not in output


def test_non_count_query_still_uses_semantic_search(engine, index):
    _add_person_frame(index, 26, "a man is walking down a sidewalk with a black umbrella", "track_002")
    _add_person_frame(index, 18, "surveillance photo of a man who was caught in a car", "track_001")

    results = engine.query("person walking")
    assert results[0].get("_type") != "count"
    assert len(results) >= 1
