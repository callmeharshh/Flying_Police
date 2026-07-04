import os


import pytest
from agent.alert_rules import AlertRulesEngine
from agent.vehicle_tracker import VehicleTracker, build_visit_key

# Bboxes from validation fixtures — same car moving left (frames 271-283)
FIXTURE_BBOXES_TIGHT = [
    (1202, 448, 78, 87),
    (1184, 448, 96, 89),
    (1167, 448, 113, 90),
    (1152, 450, 128, 90),
    (1138, 452, 142, 89),
]

# Frames 298-388 — one car, varying BLIP labels (vehicle → person → police_car)
FIXTURE_BBOXES_298_388 = [
    (1076, 457, 184, 91, 298, "vehicle", "a car is seen in this surveillance image"),
    (925, 468, 196, 97, 328, "vehicle", "a car is seen in this surveillance image"),
    (566, 493, 221, 138, 358, "person", "a man is seen in this image from the video"),
    (0, 526, 205, 112, 388, "police_car", "a police car is seen in this image"),
]


def test_same_vehicle_across_consecutive_bboxes():
    tracker = VehicleTracker()
    results = []
    for frame_id, bbox in zip([271, 274, 277, 280, 283], FIXTURE_BBOXES_TIGHT):
        results.append(tracker.update(
            frame_id=frame_id,
            location="main_gate",
            description="varying blip caption",
            bbox=bbox,
            object_type="vehicle",
        ))

    track_ids = {r.track_id for r in results}
    assert len(track_ids) == 1
    assert results[0].is_new_entry
    assert all(r.is_continuing for r in results[1:])
    assert all(r.entry_count == 1 for r in results)


def test_frames_298_to_388_single_moving_object():
    """One car across frames 298-388 despite vehicle → person → police_car labels."""
    tracker = VehicleTracker()
    results = []
    for bbox, frame_id, obj_type, desc in [
        (b[:4], b[4], b[5], b[6]) for b in FIXTURE_BBOXES_298_388
    ]:
        results.append(tracker.update(
            frame_id=frame_id,
            location="main_gate",
            description=desc,
            bbox=bbox,
            object_type=obj_type,
        ))

    track_ids = {r.track_id for r in results}
    assert len(track_ids) == 1, f"Expected 1 track, got {track_ids}"
    assert results[0].is_new_entry
    assert all(r.is_continuing for r in results[1:])
    assert results[2].label_changed
    assert results[2].prior_object_type == "vehicle"
    assert results[2].object_type == "person"
    assert results[3].label_changed
    assert results[3].object_type == "police_car"
    assert len(results[3].trajectory) == 4


def test_alert_rules_single_track_298_388():
    rules = AlertRulesEngine()
    alerts_total = []
    for bbox, frame_id, obj_type, desc in [
        (b[:4], b[4], b[5], b[6]) for b in FIXTURE_BBOXES_298_388
    ]:
        objects = [obj_type]
        alerts = rules.evaluate(
            frame_id=frame_id,
            timestamp="2026-06-10T19:41:00",
            location="main_gate",
            objects=objects,
            activity="entering",
            description=desc,
            bbox=bbox,
        )
        alerts_total.extend(alerts)

    rule02_count = sum(1 for a in alerts_total if a.rule_id == "RULE-02")
    rule03_count = sum(1 for a in alerts_total if a.rule_id == "RULE-03")
    assert rule02_count <= 1
    assert rule03_count == 0
    assert rules.last_track_update.is_continuing


def test_visit_key_differs_by_type():
    assert build_visit_key("main_gate", "a car", "vehicle") == "main_gate|vehicle"
    assert build_visit_key("main_gate", "police car", "police_car") == "main_gate|police_car"


def test_rule02_skips_police_car():
    rules = AlertRulesEngine()
    alerts = rules.evaluate(
        frame_id=271,
        timestamp="2026-06-10T19:28:00",
        location="main_gate",
        objects=["police_car"],
        activity="entering",
        description="a police car is seen in this und photo",
        bbox=(1202, 448, 78, 87),
    )
    assert not any(a.rule_id == "RULE-02" for a in alerts)


def test_rule03_fires_on_genuine_revisit():
    rules = AlertRulesEngine()
    kwargs = {
        "timestamp": "2026-06-09T08:00:00",
        "location": "main_gate",
        "objects": ["vehicle"],
        "activity": "entering",
        "description": "a blue ford f150 truck entering",
        "bbox": (100, 400, 80, 60),
    }
    rules.evaluate(frame_id=1, **kwargs)
    alerts = rules.evaluate(frame_id=200, **kwargs)
    assert any(a.rule_id == "RULE-03" for a in alerts)


def test_validation_fixture_bboxes_are_spatially_continuous():
    centers = [(x + w / 2, y + h / 2) for x, y, w, h, *_ in FIXTURE_BBOXES_298_388]
    for i in range(1, len(centers)):
        dx = abs(centers[i][0] - centers[i - 1][0])
        dy = abs(centers[i][1] - centers[i - 1][1])
        assert dx < 700
        assert dy < 100
