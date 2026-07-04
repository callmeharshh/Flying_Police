
import pytest
from agent.alert_rules import AlertRulesEngine, format_dwell_duration, threat_from_alerts


@pytest.fixture
def rules():
    return AlertRulesEngine()


def test_no_alert_daytime_vehicle(rules):
    """TC-06: Vehicle at gate during day — no alert."""
    alerts = rules.evaluate(1, "2026-06-09T10:00:00", "main_gate",
                            ["vehicle"], "entering", "a blue ford f150 truck entering",
                            is_night=False)
    assert len(alerts) == 0


def test_rule01_person_at_night(rules):
    """TC-02: Person detected at night → RULE-01."""
    alerts = rules.evaluate(5, "2026-06-10T00:01:00", "main_gate",
                            ["person"], "loitering", "a person standing at the gate",
                            is_night=True)
    rule_ids = [a.rule_id for a in alerts]
    assert "RULE-01" in rule_ids


def test_rule01_person_daytime_no_alert(rules):
    """Bright scene — person during day should not trigger RULE-01."""
    alerts = rules.evaluate(4, "2026-06-09T14:00:00", "main_gate",
                            ["person"], "entering", "a person approaching the gate",
                            is_night=False)
    assert not any(a.rule_id == "RULE-01" for a in alerts)


def test_rule01_person_at_night_not_repeated_per_track(rules):
    """Same person on consecutive frames → RULE-01 only on first sighting."""
    kwargs = {
        "location": "main_gate",
        "objects": ["person"],
        "activity": "entering",
        "description": "a person approaching the gate",
        "bbox": (100, 200, 50, 80),
        "is_night": True,
    }
    first = rules.evaluate(1, "2026-06-09T23:58:00", **kwargs)
    assert any(a.rule_id == "RULE-01" for a in first)

    second = rules.evaluate(
        2, "2026-06-09T23:59:00",
        bbox=(105, 205, 50, 80),
        **{k: v for k, v in kwargs.items() if k != "bbox"},
    )
    assert not any(a.rule_id == "RULE-01" for a in second)


def test_rule02_unknown_vehicle(rules):
    """Unknown vehicle → RULE-02."""
    alerts = rules.evaluate(6, "2026-06-09T14:00:00", "perimeter",
                            ["vehicle"], "stationary", "an unknown white sedan",
                            is_night=False)
    assert any(a.rule_id == "RULE-02" for a in alerts)


def test_rule02_known_vehicle_no_alert(rules):
    """Known vehicle (blue ford f150) → no RULE-02."""
    alerts = rules.evaluate(1, "2026-06-09T08:00:00", "main_gate",
                            ["vehicle"], "entering", "a blue ford f150 truck entering",
                            is_night=False)
    assert not any(a.rule_id == "RULE-02" for a in alerts)


def test_rule02_police_car_no_alert(rules):
    """Police car → no RULE-02 (authorized vehicle)."""
    alerts = rules.evaluate(1, "2026-06-09T14:00:00", "main_gate",
                            ["police_car"], "entering", "a police car patrolling the lot",
                            is_night=False)
    assert not any(a.rule_id == "RULE-02" for a in alerts)


def test_rule03_repeat_vehicle(rules):
    """TC-03: Same vehicle enters twice → RULE-03 on second visit (not per frame)."""
    rules.evaluate(1, "2026-06-09T08:00:00", "main_gate",
                   ["vehicle"], "entering", "a blue ford f150 truck entering",
                   is_night=False)
    alerts = rules.evaluate(50, "2026-06-09T08:30:00", "main_gate",
                            ["vehicle"], "entering", "a blue ford f150 truck entering again",
                            is_night=False)
    assert any(a.rule_id == "RULE-03" for a in alerts)


def test_rule05_perimeter_at_night(rules):
    """Activity at perimeter at night → RULE-05."""
    alerts = rules.evaluate(9, "2026-06-09T23:00:00", "perimeter",
                            ["person"], "entering", "a person at the perimeter fence",
                            is_night=True)
    assert any(a.rule_id == "RULE-05" for a in alerts)


def test_rule05_perimeter_daytime_no_alert(rules):
    """Activity at perimeter during day → no RULE-05."""
    alerts = rules.evaluate(9, "2026-06-09T14:00:00", "perimeter",
                            ["person"], "entering", "a person at the perimeter fence",
                            is_night=False)
    assert not any(a.rule_id == "RULE-05" for a in alerts)


def test_no_alert_clear_frame(rules):
    """Empty frame → no alerts."""
    alerts = rules.evaluate(7, "2026-06-09T09:00:00", "garage",
                            [], "clear", "no activity detected",
                            is_night=False)
    assert len(alerts) == 0


def test_rule04_stationary_person_fires_immediately(rules):
    """Person sitting/stationary → RULE-04 on first frame (high alert)."""
    alerts = rules.evaluate(
        33, "2026-06-10T20:09:00", "main_gate",
        ["person"], "stationary", "a man is sitting on the floor in front of a door",
        bbox=(566, 493, 221, 138),
        is_night=False,
    )
    assert any(a.rule_id == "RULE-04" and a.severity == "high" for a in alerts)


def test_threat_high_only_when_alert_fires(rules):
    """Daytime person entering → low threat, no high without an alert."""
    alerts = rules.evaluate(
        14, "2026-06-10T20:07:00", "main_gate",
        ["person"], "entering", "a man is seen in this surveillance image",
        bbox=(100, 200, 50, 80),
        is_night=False,
    )
    assert threat_from_alerts(alerts, ["person"]) == "low"
    assert not any(a.severity == "high" for a in alerts)


def test_format_dwell_duration_shows_seconds_under_one_minute():
    assert format_dwell_duration(8) == "8 sec"
    assert format_dwell_duration(59) == "59 sec"


def test_format_dwell_duration_shows_minutes_and_remainder():
    assert format_dwell_duration(60) == "1 min"
    assert format_dwell_duration(90) == "1 min 30 sec"


def test_rule04_prolonged_loiter_message_uses_video_seconds(rules):
    """Dwell duration in alert text must reflect actual elapsed seconds."""
    kwargs = {
        "location": "main_gate",
        "objects": ["person"],
        "activity": "loitering",
        "description": "a person standing at the gate",
        "bbox": (100, 200, 50, 80),
        "is_night": False,
    }
    rules.evaluate(frame_id=19, timestamp="2026-06-10T21:12:00+00:00", **kwargs)
    alerts = rules.evaluate(
        frame_id=22,
        timestamp="2026-06-10T21:12:22+00:00",
        bbox=(102, 202, 50, 80),
        **{k: v for k, v in kwargs.items() if k not in ("bbox",)},
    )

    prolonged = [a for a in alerts if a.message.startswith("Person loitering at")]
    assert len(prolonged) == 1
    assert "22 sec" in prolonged[0].message
    assert " min" not in prolonged[0].message


def test_rule04_prolonged_loiter_not_repeated_every_frame(rules):
    """After dwell threshold, prolonged RULE-04 fires once — not every frame."""
    kwargs = {
        "location": "main_gate",
        "objects": ["person"],
        "activity": "loitering",
        "description": "a person standing at the gate",
        "bbox": (100, 200, 50, 80),
        "is_night": False,
    }
    rules.evaluate(frame_id=19, timestamp="2026-06-10T21:12:00+00:00", **kwargs)
    rules.evaluate(
        frame_id=22,
        timestamp="2026-06-10T21:12:22+00:00",
        bbox=(102, 202, 50, 80),
        **{k: v for k, v in kwargs.items() if k != "bbox"},
    )
    third = rules.evaluate(
        frame_id=23,
        timestamp="2026-06-10T21:12:30+00:00",
        bbox=(103, 203, 50, 80),
        **{k: v for k, v in kwargs.items() if k != "bbox"},
    )
    prolonged = [a for a in third if a.message.startswith("Person loitering at")]
    assert len(prolonged) == 0


def test_threat_high_matches_rule_alert(rules):
    """Night person → high threat and RULE-01 alert together."""
    alerts = rules.evaluate(
        4, "2026-06-09T23:58:00", "main_gate",
        ["person"], "entering", "a person approaching the gate",
        is_night=True,
    )
    assert any(a.rule_id == "RULE-01" for a in alerts)
    assert threat_from_alerts(alerts, ["person"]) == "high"
