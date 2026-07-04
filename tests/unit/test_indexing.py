
import pytest
from storage.frame_index import FrameIndex
from storage.event_store import EventStore
from config import CHROMA_DIR, EVENTS_DB_PATH

TEST_CHROMA = CHROMA_DIR + "_test"
TEST_DB = EVENTS_DB_PATH.replace(".db", "_test.db")


@pytest.fixture
def index(tmp_path):
    return FrameIndex(chroma_dir=str(tmp_path / "chroma"))


@pytest.fixture
def store(tmp_path):
    return EventStore(db_path=str(tmp_path / "events.db"))


def test_frame_indexed(index):
    """TC-04: Frame is added and queryable."""
    index.add_frame(1, "A blue Ford F150 truck entering through the gate",
                    "2026-06-09T08:00:00", "main_gate", ["vehicle"], "low")
    assert index.count() == 1


def test_query_truck_events(index):
    """TC-05: Query 'show all truck events' returns truck frames."""
    index.add_frame(1, "A blue Ford F150 truck entering through the gate",
                    "2026-06-09T08:00:00", "main_gate", ["vehicle"], "low")
    index.add_frame(4, "A person approaching the gate at night",
                    "2026-06-09T23:58:00", "main_gate", ["person"], "high")

    results = index.query("show all truck events")
    assert len(results) > 0
    assert results[0]["frame_id"] == 1


def test_query_person_at_night(index):
    """Person at night query returns correct frame."""
    index.add_frame(4, "A person approaching the gate at night",
                    "2026-06-09T23:58:00", "main_gate", ["person"], "high")
    index.add_frame(1, "A blue Ford F150 truck entering",
                    "2026-06-09T08:00:00", "main_gate", ["vehicle"], "low")

    results = index.query("person at night")
    assert results[0]["frame_id"] == 4


def test_event_store_log(store):
    """TC-01: Event is stored correctly."""
    eid = store.log_event(1, "Blue Ford F150 spotted at main gate", "low")
    events = store.get_all_events()
    assert len(events) == 1
    assert events[0]["frame_id"] == 1
    assert "Ford" in events[0]["message"]


def test_alert_stored(store):
    """Alert is stored and retrievable."""
    store.log_alert(3, "RULE-03", "Vehicle entered 2 times today", "medium")
    alerts = store.get_alerts()
    assert len(alerts) == 1
    assert alerts[0]["rule_id"] == "RULE-03"
    assert alerts[0]["severity"] == "medium"
