
import pytest
from agent.alert_rules import AlertRulesEngine
from storage.event_store import EventStore
from storage.frame_index import FrameIndex
from config import CHROMA_DIR, EVENTS_DB_PATH

TEST_CHROMA = CHROMA_DIR + "_test_agent"
TEST_DB = EVENTS_DB_PATH.replace(".db", "_test_agent.db")


@pytest.fixture
def store(tmp_path):
    return EventStore(db_path=str(tmp_path / "events.db"))


@pytest.fixture
def index(tmp_path):
    return FrameIndex(chroma_dir=str(tmp_path / "chroma"))


def test_tc01_truck_logged(store, index):
    """TC-01: Truck at gate is logged correctly."""
    store.log_event(1, "Blue Ford F150 spotted at main gate, 08:00", "low")
    events = store.get_all_events()
    assert any("Ford" in e["message"] for e in events)


def test_tc07_telemetry_attached():
    """TC-07: Telemetry fields are populated on simulated frames."""
    from data.simulated_frames import get_simulated_frames
    frames = get_simulated_frames()
    for f in frames:
        assert "drone_lat" in f.telemetry
        assert "drone_lon" in f.telemetry
        assert "battery_pct" in f.telemetry
        assert f.telemetry["battery_pct"] > 0


def test_all_scenarios_correct_alerts():
    """All 7 simulated scenarios produce expected alert output."""
    rules = AlertRulesEngine()
    from data.simulated_frames import get_simulated_frames
    frames = get_simulated_frames()

    expected = {
        1: [],
        2: [],
        3: ["RULE-03"],
        4: ["RULE-01"],
        5: ["RULE-04"],  # loitering at night; RULE-01 already fired on frame 4
        6: ["RULE-02"],
        7: [],
    }

    night_frames = {4, 5}

    for f in frames:
        alerts = rules.evaluate(
            f.frame_id, f.timestamp, f.location,
            f.objects, f.activity, f.raw_description,
            is_night=f.frame_id in night_frames,
        )
        fired = [a.rule_id for a in alerts]
        for expected_rule in expected[f.frame_id]:
            assert expected_rule in fired, f"Frame {f.frame_id}: expected {expected_rule}, got {fired}"
        if not expected[f.frame_id]:
            assert len(alerts) == 0, f"Frame {f.frame_id}: expected no alerts, got {fired}"
