"""Regression checks on captured fixture data (no model download)."""
from agent.alert_rules import AlertRulesEngine, threat_from_alerts
from agent.vehicle_context import primary_crop_bbox, primary_crop_color
from data.validation_capture import list_capture_dirs, load_capture_record
from query.query_engine import QueryEngine
from storage.event_store import EventStore
from storage.frame_index import FrameIndex
from vlm.blip_analyzer import CropAnalysis, FrameVLMResult


def _load_vlm_result(record):
    crops = [
        CropAnalysis(
            caption=c["caption"],
            object_type=c["object_type"],
            color=c["color"],
            bbox=tuple(c["bbox"]),
        )
        for c in record.blip["crops"]
    ]
    return FrameVLMResult(
        crops=crops,
        objects=record.blip["objects"],
        raw_description=record.blip["raw_description"],
    )


def test_car_frames_298_to_388_share_one_track():
    """Regression: one moving car must not split into multiple tracks."""
    target_dirs = sorted(
        d for d in list_capture_dirs()
        if 298 <= load_capture_record(d).frame_id <= 388
    )
    assert len(target_dirs) >= 2, "Need car sequence fixtures 298-388"

    rules = AlertRulesEngine()
    track_ids = []
    for capture_dir in target_dirs:
        record = load_capture_record(capture_dir)
        if not record.blip:
            continue
        vlm_result = _load_vlm_result(record)
        rules.evaluate(
            frame_id=record.frame_id,
            timestamp=record.timestamp,
            location=record.location,
            objects=record.blip["objects"],
            activity=record.activity,
            description=record.blip["raw_description"],
            bbox=primary_crop_bbox(vlm_result),
            color=primary_crop_color(vlm_result),
        )
        if rules.last_track_update:
            track_ids.append(rules.last_track_update.track_id)

    assert len(set(track_ids)) == 1, f"Expected one track, got {set(track_ids)}"


def test_stored_threat_level_matches_stored_alerts():
    """Captured threat_level must align with alerts saved in each record.json."""
    from agent.alert_rules import Alert

    for capture_dir in list_capture_dirs():
        record = load_capture_record(capture_dir)
        if not record.blip or not record.threat_level:
            continue

        alerts = [
            Alert(
                rule_id=stored.rule_id,
                frame_id=record.frame_id,
                message=stored.message,
                severity=stored.severity,
            )
            for stored in record.alerts
        ]
        expected_threat = threat_from_alerts(alerts, record.blip["objects"])

        if record.threat_level == "high":
            assert any(a.severity == "high" for a in alerts), (
                f"Frame {record.frame_id}: threat_level=high but no high alert in record"
            )
        assert expected_threat == record.threat_level, (
            f"Frame {record.frame_id}: threat mismatch "
            f"(record={record.threat_level}, from_stored_alerts={expected_threat})"
        )


def test_count_query_on_fixture_index(tmp_path):
    """Build index from fixtures and verify count query returns structured answer."""
    index = FrameIndex(chroma_dir=str(tmp_path / "chroma"))
    person_frames = 0

    for capture_dir in list_capture_dirs():
        record = load_capture_record(capture_dir)
        if not record.blip or "person" not in record.blip.get("objects", []):
            continue
        person_frames += 1
        bbox = tuple(record.blip["crops"][0]["bbox"])
        index.add_frame(
            frame_id=record.frame_id,
            description=record.blip["raw_description"],
            timestamp=record.timestamp,
            location=record.location,
            objects=record.blip["objects"],
            threat_level=record.threat_level or "low",
            bbox=bbox,
            track_id=(record.vehicle_track or {}).get("track_id"),
        )

    if person_frames == 0:
        return

    engine = QueryEngine(EventStore(db_path=str(tmp_path / "events.db")), index)
    results = engine.query("how many men were detected?")
    assert results[0]["_type"] == "count"
    assert results[0]["unique_count"] >= 1
    output = engine.format_results(results)
    assert "Answer:" in output
    assert "distinct person" in output
