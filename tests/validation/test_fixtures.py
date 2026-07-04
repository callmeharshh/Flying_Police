"""
Validate captured fixtures: structure, images on disk, and optional BLIP re-run.

Generate fixtures first:
    python scripts/capture_validation_set.py --every 30 --max 10 --skip-agent

Run validation:
    pytest tests/test_validation_fixtures.py -v
    pytest tests/test_validation_fixtures.py -v -m integration
"""
import os

import cv2
import pytest


from config import VALIDATION_FIXTURES_DIR
from data.validation_capture import (
    list_capture_dirs,
    load_capture_record,
    load_manifest,
)
from vlm.constants import OBJECT_TYPES
from vlm.background_subtractor import ObjectCrop
from vlm.blip_analyzer import BLIPAnalyzer


def test_manifest_lists_captures():
    manifest = load_manifest()
    assert manifest["capture_count"] > 0
    assert len(manifest["captures"]) == manifest["capture_count"]
    assert "video" in manifest


def test_each_capture_has_required_files_and_fields():
    manifest = load_manifest()
    for entry in manifest["captures"]:
        capture_dir = os.path.join(VALIDATION_FIXTURES_DIR, entry["dir"])
        record = load_capture_record(capture_dir)

        assert os.path.isfile(os.path.join(capture_dir, record.full_frame_file))
        assert isinstance(record.frame_id, int)
        assert isinstance(record.raw_idx, int)
        assert record.timestamp
        assert record.location

        if entry["has_foreground"]:
            assert record.has_foreground
            assert record.blip is not None
            assert "objects" in record.blip
            assert "raw_description" in record.blip
            assert record.blip["raw_description"]
            for crop in record.blip["crops"]:
                crop_path = os.path.join(capture_dir, crop["image_file"])
                assert os.path.isfile(crop_path), f"Missing crop: {crop_path}"


def test_blip_records_have_parseable_object_types():
    for capture_dir in list_capture_dirs():
        record = load_capture_record(capture_dir)
        if not record.blip:
            continue
        for obj in record.blip["objects"]:
            assert obj in OBJECT_TYPES
        for crop in record.blip["crops"]:
            assert crop["object_type"] in OBJECT_TYPES
            assert isinstance(crop["caption"], str)
            assert len(crop["caption"]) > 0


def test_agent_responses_saved_when_present():
    manifest = load_manifest()
    agent_captures = [c for c in manifest["captures"] if c["has_agent"]]
    if not agent_captures:
        pytest.skip("No agent responses in fixtures (re-run capture without --skip-agent)")

    for entry in agent_captures:
        record = load_capture_record(os.path.join(VALIDATION_FIXTURES_DIR, entry["dir"]))
        assert record.agent_response
        assert len(record.agent_response) > 10


def test_fixture_captures_same_vehicle_track():
    """Captured frames of one moving car should share a track after re-evaluation."""
    from agent.alert_rules import AlertRulesEngine
    from agent.vehicle_context import primary_vehicle_bbox, primary_vehicle_color
    from vlm.blip_analyzer import CropAnalysis, FrameVLMResult

    detection_dirs = sorted(
        (d for d in list_capture_dirs() if load_capture_record(d).has_foreground),
        key=lambda d: load_capture_record(d).frame_id,
    )
    # Frames 298-388: one car moving through scene with varying BLIP labels
    detection_dirs = [
        d for d in detection_dirs
        if 298 <= load_capture_record(d).frame_id <= 388
    ]
    if len(detection_dirs) < 2:
        pytest.skip("Need at least 2 detection fixtures")

    rules = AlertRulesEngine()
    track_ids = []

    for capture_dir in detection_dirs:
        record = load_capture_record(capture_dir)
        if not record.blip:
            continue
        crops = [
            CropAnalysis(
                caption=c["caption"],
                object_type=c["object_type"],
                color=c["color"],
                bbox=tuple(c["bbox"]),
            )
            for c in record.blip["crops"]
        ]
        vlm_result = FrameVLMResult(
            crops=crops,
            objects=record.blip["objects"],
            raw_description=record.blip["raw_description"],
        )
        rules.evaluate(
            frame_id=record.frame_id,
            timestamp=record.timestamp,
            location=record.location,
            objects=record.blip["objects"],
            activity=record.activity,
            description=record.blip["raw_description"],
            bbox=primary_vehicle_bbox(vlm_result),
            color=primary_vehicle_color(vlm_result),
        )
        if rules.last_track_update:
            track_ids.append(rules.last_track_update.track_id)

    if len(track_ids) >= 2:
        assert len(set(track_ids)) == 1, f"Expected one track, got: {set(track_ids)}"


@pytest.mark.integration
def test_blip_rerun_matches_saved_object_types():
    """Re-run BLIP on saved crops and check object types still parse correctly."""
    analyzer = BLIPAnalyzer()
    checked = 0

    for capture_dir in list_capture_dirs():
        record = load_capture_record(capture_dir)
        if not record.blip or not record.blip["crops"]:
            continue

        for crop_meta in record.blip["crops"]:
            crop_path = os.path.join(capture_dir, crop_meta["image_file"])
            frame_bgr = cv2.imread(crop_path)
            assert frame_bgr is not None

            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            dummy = ObjectCrop(
                image=pil_image,
                bbox=tuple(crop_meta["bbox"]),
                bbox_relative=(0, 0, 1, 1),
            )
            result = analyzer.analyze_crop(dummy)

            assert result.object_type in OBJECT_TYPES
            assert isinstance(result.caption, str)
            assert len(result.caption) > 0
            checked += 1

    assert checked > 0, "No crops found to re-run BLIP on"
