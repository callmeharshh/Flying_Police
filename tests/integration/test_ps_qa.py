"""
Integration QA tests for VLM detection, alerts, and indexing.

Maps to QA fixture examples:
  - Truck / vehicle detected and logged
  - Person detected (surveillance pipeline)
  - Security alert at night
  - Indexed frames queryable by object

Run (loads BLIP ~15s first time):
    pytest tests/integration -m integration -v
"""
from __future__ import annotations

import os

import cv2
import pytest
from PIL import Image

from agent.alert_rules import AlertRulesEngine
from config import SAMPLE_IMAGES_DIR
from query.query_engine import QueryEngine
from storage.event_store import EventStore
from storage.frame_index import FrameIndex
from vlm.background_subtractor import BackgroundSubtractor, ObjectCrop
from vlm.blip_analyzer import BLIPAnalyzer
from vlm.constants import OBJECT_TYPES, VEHICLE_KEYWORDS

OUTSIDE_ENTRY_DEBUG_DIR = os.path.join(
    SAMPLE_IMAGES_DIR, "debug_bg", "outside_entry_720p"
)
OUTSIDE_ENTRY_WARMUP_FRAMES = (
    "frame_000.jpg",
    "frame_007.jpg",
    "frame_014.jpg",
    "frame_021.jpg",
    "frame_028.jpg",
)
OUTSIDE_ENTRY_PERSON_FRAME = "frame_049.jpg"

CAR_CROP_FILE = "car_crop.jpg"
VEHICLE_SAMPLE_FILE = "download.jpeg"


def _sample_image_path(filename: str) -> str:
    path = os.path.join(SAMPLE_IMAGES_DIR, filename)
    if not os.path.isfile(path):
        pytest.skip(f"Sample image not found: {path}")
    return path


def _warm_subtractor_and_apply(
    subtractor: BackgroundSubtractor,
    debug_dir: str,
    warmup_frames: tuple[str, ...],
    target_frame: str,
):
    for name in warmup_frames:
        frame_path = os.path.join(debug_dir, name)
        if not os.path.isfile(frame_path):
            pytest.skip(f"Debug frame not found: {frame_path}")
        subtractor.apply(cv2.imread(frame_path))

    target_path = os.path.join(debug_dir, target_frame)
    if not os.path.isfile(target_path):
        pytest.skip(f"Debug frame not found: {target_path}")
    return subtractor.apply(cv2.imread(target_path))


def _caption_suggests_vehicle(caption: str) -> bool:
    lower = caption.lower()
    return any(keyword in lower for keyword in VEHICLE_KEYWORDS)


@pytest.mark.integration
def test_ps_blip_analyzes_car_crop(blip_analyzer: BLIPAnalyzer):
    """PS QA / Correctness: BLIP returns a caption and typed result for a crop image."""
    path = _sample_image_path(CAR_CROP_FILE)
    pil_image = Image.open(path)
    width, height = pil_image.size

    crop = ObjectCrop(
        image=pil_image,
        bbox=(0, 0, width, height),
        bbox_relative=(0, 0, 1, 1),
    )
    result = blip_analyzer.analyze_crop(crop)

    assert isinstance(result.caption, str)
    assert len(result.caption.strip()) > 0
    assert result.object_type in OBJECT_TYPES
    assert result.bbox == (0, 0, width, height)


@pytest.mark.integration
def test_ps_blip_detects_vehicle_in_sample_image(blip_analyzer: BLIPAnalyzer):
    """PS QA: vehicle object identified from a sample image ('Blue truck at gate')."""
    path = _sample_image_path(VEHICLE_SAMPLE_FILE)
    result = blip_analyzer.analyze_full_image(Image.open(path))

    assert len(result.caption.strip()) > 0
    assert result.object_type == "vehicle" or _caption_suggests_vehicle(result.caption)


@pytest.mark.integration
def test_ps_blip_detects_person_via_mog2_pipeline(blip_analyzer: BLIPAnalyzer):
    """PS QA: MOG2 foreground + BLIP identifies a person in surveillance footage."""
    subtractor = BackgroundSubtractor()
    crops = _warm_subtractor_and_apply(
        subtractor,
        OUTSIDE_ENTRY_DEBUG_DIR,
        OUTSIDE_ENTRY_WARMUP_FRAMES,
        OUTSIDE_ENTRY_PERSON_FRAME,
    )
    assert len(crops) >= 1, "Expected foreground motion in sample frame"

    frame_result = blip_analyzer.analyze_frame(crops)
    assert frame_result is not None
    assert "person" in frame_result.objects
    assert len(frame_result.raw_description.strip()) > 0


@pytest.mark.integration
def test_ps_night_person_triggers_rule01():
    """PS QA: security alert when person detected at night."""
    rules = AlertRulesEngine()
    alerts = rules.evaluate(
        frame_id=4,
        timestamp="2026-06-10T00:01:00",
        location="main_gate",
        objects=["person"],
        activity="loitering",
        description="a person standing at the main gate",
        is_night=True,
    )
    assert any(a.rule_id == "RULE-01" for a in alerts)
    assert any("night" in a.message.lower() for a in alerts)


@pytest.mark.integration
def test_ps_truck_event_logged(tmp_path):
    """PS QA: truck event stored in SQLite."""
    store = EventStore(db_path=str(tmp_path / "events.db"))
    store.log_event(
        frame_id=1,
        message="Blue Ford F150 spotted at garage, 12:00.",
        severity="low",
    )
    events = store.get_all_events()
    assert len(events) == 1
    assert "Ford" in events[0]["message"]
    assert "garage" in events[0]["message"]


@pytest.mark.integration
def test_ps_indexed_truck_events_queryable(tmp_path, blip_analyzer: BLIPAnalyzer):
    """PS QA: frame indexed after BLIP and queryable by object ('show all truck events')."""
    path = _sample_image_path(VEHICLE_SAMPLE_FILE)
    blip_result = blip_analyzer.analyze_full_image(Image.open(path))

    index = FrameIndex(chroma_dir=str(tmp_path / "chroma"))
    store = EventStore(db_path=str(tmp_path / "events.db"))
    index.add_frame(
        frame_id=1,
        description=blip_result.caption,
        timestamp="2026-06-09T12:00:00",
        location="garage",
        objects=[blip_result.object_type] if blip_result.object_type != "unknown" else ["vehicle"],
        threat_level="low",
    )

    engine = QueryEngine(store, index)
    results = engine.query("show all truck events")

    assert len(results) >= 1
    assert results[0]["frame_id"] == 1
    assert len(results[0]["description"]) > 0
