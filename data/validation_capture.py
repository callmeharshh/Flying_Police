"""
Capture sampled video frames with BLIP and agent outputs for regression validation.

Usage:
    python scripts/capture_validation_set.py
    python scripts/capture_validation_set.py --every 20 --max 10 --skip-agent
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import cv2
import numpy as np

from agent.alert_rules import Alert, AlertRulesEngine
from agent.security_agent import SecurityAgent
from config import (
    DEMO_VIDEO,
    SAMPLE_VIDEO_DIR,
    VALIDATION_CAPTURE_EVERY_N,
    VALIDATION_FIXTURES_DIR,
    VALIDATION_MAX_CAPTURES,
)
from data.telemetry import battery_for_frame, get_telemetry
from storage.event_store import EventStore
from storage.frame_index import FrameIndex
from vlm.background_subtractor import BackgroundSubtractor, ObjectCrop
from vlm.blip_analyzer import BLIPAnalyzer, FrameVLMResult
from vlm.lighting_detector import analyze_frame_lighting

MANIFEST_FILENAME = "manifest.json"
CAPTURES_SUBDIR = "captures"
RECORD_FILENAME = "record.json"
FULL_FRAME_FILENAME = "full.jpg"
CROP_FILENAME_TEMPLATE = "crop_{:02d}.jpg"


@dataclass
class CropRecord:
    caption: str
    object_type: str
    color: str
    bbox: list
    image_file: str


@dataclass
class AlertRecord:
    rule_id: str
    message: str
    severity: str


@dataclass
class CaptureRecord:
    frame_id: int
    raw_idx: int
    timestamp: str
    location: str
    has_foreground: bool
    full_frame_file: str
    activity: str = ""
    threat_level: str = ""
    blip: Optional[dict] = None
    alerts: List[AlertRecord] = field(default_factory=list)
    agent_response: Optional[str] = None
    telemetry: Optional[dict] = None
    vehicle_track: Optional[dict] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.alerts:
            data["alerts"] = [asdict(a) for a in self.alerts]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> CaptureRecord:
        alerts = [AlertRecord(**a) for a in data.get("alerts", [])]
        payload = {k: v for k, v in data.items() if k != "alerts"}
        return cls(**payload, alerts=alerts)


def _vlm_result_to_dict(vlm_result: FrameVLMResult, crop_files: List[str]) -> dict:
    crops = []
    for analysis, image_file in zip(vlm_result.crops, crop_files):
        crops.append({
            "caption": analysis.caption,
            "object_type": analysis.object_type,
            "color": analysis.color,
            "bbox": list(analysis.bbox),
            "image_file": image_file,
        })
    return {
        "objects": vlm_result.objects,
        "raw_description": vlm_result.raw_description,
        "crops": crops,
    }


def _alerts_to_records(alerts: List[Alert]) -> List[AlertRecord]:
    return [AlertRecord(rule_id=a.rule_id, message=a.message, severity=a.severity) for a in alerts]


class ValidationCaptureRecorder:
    def __init__(
        self,
        output_dir: str = VALIDATION_FIXTURES_DIR,
        every_n: int = VALIDATION_CAPTURE_EVERY_N,
        max_captures: Optional[int] = VALIDATION_MAX_CAPTURES,
    ):
        self._output_dir = output_dir
        self._captures_dir = os.path.join(output_dir, CAPTURES_SUBDIR)
        self._every_n = every_n
        self._max_captures = max_captures
        self._records: List[CaptureRecord] = []
        os.makedirs(self._captures_dir, exist_ok=True)

    @property
    def capture_count(self) -> int:
        return len(self._records)

    def should_capture(self, frame_id: int) -> bool:
        if self._max_captures is not None and self.capture_count >= self._max_captures:
            return False
        return frame_id % self._every_n == 0

    def save_capture(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        raw_idx: int,
        timestamp: str,
        location: str,
        crops: List[ObjectCrop],
        vlm_result: Optional[FrameVLMResult],
        alerts: List[Alert],
        agent_response: Optional[str],
        activity: str,
        threat_level: str,
        telemetry: dict,
        vehicle_track: Optional[dict] = None,
    ) -> CaptureRecord:
        capture_name = f"frame_{frame_id:04d}_raw{raw_idx:04d}"
        capture_dir = os.path.join(self._captures_dir, capture_name)
        os.makedirs(capture_dir, exist_ok=True)

        full_path = os.path.join(capture_dir, FULL_FRAME_FILENAME)
        cv2.imwrite(full_path, frame_bgr)

        crop_files: List[str] = []
        for idx, crop in enumerate(crops):
            crop_file = CROP_FILENAME_TEMPLATE.format(idx)
            crop_path = os.path.join(capture_dir, crop_file)
            crop_bgr = cv2.cvtColor(np.array(crop.image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(crop_path, crop_bgr)
            crop_files.append(crop_file)

        blip_data = _vlm_result_to_dict(vlm_result, crop_files) if vlm_result else None
        record = CaptureRecord(
            frame_id=frame_id,
            raw_idx=raw_idx,
            timestamp=timestamp,
            location=location,
            has_foreground=bool(crops),
            full_frame_file=FULL_FRAME_FILENAME,
            activity=activity,
            threat_level=threat_level,
            blip=blip_data,
            alerts=_alerts_to_records(alerts),
            agent_response=agent_response,
            telemetry=telemetry,
            vehicle_track=vehicle_track,
        )

        record_path = os.path.join(capture_dir, RECORD_FILENAME)
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        self._records.append(record)
        return record

    def write_manifest(self, video_path: str, sample_every_n: int, capture_on: str = "sampled") -> str:
        self._capture_on = capture_on
        manifest = {
            "video": os.path.basename(video_path),
            "video_path": video_path,
            "sample_every_n_frames": sample_every_n,
            "capture_every_n": self._every_n,
            "capture_on": getattr(self, "_capture_on", "sampled"),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "capture_count": len(self._records),
            "captures": [
                {
                    "frame_id": r.frame_id,
                    "raw_idx": r.raw_idx,
                    "dir": f"{CAPTURES_SUBDIR}/frame_{r.frame_id:04d}_raw{r.raw_idx:04d}",
                    "has_foreground": r.has_foreground,
                    "has_blip": r.blip is not None,
                    "has_agent": r.agent_response is not None,
                }
                for r in self._records
            ],
        }
        manifest_path = os.path.join(self._output_dir, MANIFEST_FILENAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path


def discover_capture_dirs(fixtures_dir: str = VALIDATION_FIXTURES_DIR) -> List[str]:
    """Scan captures/ for record.json when manifest is missing."""
    captures_root = os.path.join(fixtures_dir, CAPTURES_SUBDIR)
    if not os.path.isdir(captures_root):
        return []

    dirs = []
    for name in sorted(os.listdir(captures_root)):
        capture_dir = os.path.join(captures_root, name)
        if os.path.isfile(os.path.join(capture_dir, RECORD_FILENAME)):
            dirs.append(capture_dir)
    return dirs


def load_manifest(fixtures_dir: str = VALIDATION_FIXTURES_DIR) -> dict:
    manifest_path = os.path.join(fixtures_dir, MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return rebuild_manifest_from_captures(fixtures_dir, write=False)


def rebuild_manifest_from_captures(
    fixtures_dir: str = VALIDATION_FIXTURES_DIR,
    write: bool = True,
) -> dict:
    """Build manifest.json from on-disk capture directories."""
    capture_dirs = discover_capture_dirs(fixtures_dir)
    captures = []
    for capture_dir in capture_dirs:
        record = load_capture_record(capture_dir)
        rel_dir = os.path.relpath(capture_dir, fixtures_dir)
        captures.append({
            "dir": rel_dir,
            "frame_id": record.frame_id,
            "raw_idx": record.raw_idx,
            "has_foreground": record.has_foreground,
            "has_blip": record.blip is not None,
            "has_agent": record.agent_response is not None,
        })

    manifest = {
        "video": DEMO_VIDEO,
        "capture_count": len(captures),
        "captures": captures,
    }
    if write and captures:
        manifest_path = os.path.join(fixtures_dir, MANIFEST_FILENAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return manifest


def load_capture_record(capture_dir: str) -> CaptureRecord:
    record_path = os.path.join(capture_dir, RECORD_FILENAME)
    with open(record_path, encoding="utf-8") as f:
        return CaptureRecord.from_dict(json.load(f))


def list_capture_dirs(fixtures_dir: str = VALIDATION_FIXTURES_DIR) -> List[str]:
    manifest_path = os.path.join(fixtures_dir, MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        manifest = load_manifest(fixtures_dir)
        return [os.path.join(fixtures_dir, entry["dir"]) for entry in manifest["captures"]]
    return discover_capture_dirs(fixtures_dir)


def fixtures_available(fixtures_dir: str = VALIDATION_FIXTURES_DIR) -> bool:
    manifest_path = os.path.join(fixtures_dir, MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        return True
    return len(discover_capture_dirs(fixtures_dir)) > 0


def run_capture_pipeline(
    video_path: str,
    output_dir: str = VALIDATION_FIXTURES_DIR,
    every_n: int = VALIDATION_CAPTURE_EVERY_N,
    max_captures: Optional[int] = VALIDATION_MAX_CAPTURES,
    skip_agent: bool = False,
    sample_every_n: int = 2,
    location: str = "main_gate",
    capture_on: str = "sampled",
    infer_activity_fn=None,
    infer_threat_fn=None,
) -> str:
    """Process video and save frames with BLIP + agent outputs.

    capture_on:
        "sampled" — every Nth sampled frame (includes no-foreground frames)
        "detection" — every Nth frame that has foreground / BLIP output
    """
    from pipeline.video_processor import infer_activity
    from agent.alert_rules import threat_from_alerts
    from agent.vehicle_context import primary_crop_bbox, primary_crop_color, track_to_context

    activity_fn = infer_activity_fn or infer_activity

    recorder = ValidationCaptureRecorder(output_dir, every_n=every_n, max_captures=max_captures)
    subtractor = BackgroundSubtractor()
    analyzer = BLIPAnalyzer()

    store = EventStore(db_path=os.path.join(output_dir, "_capture_events.db"))
    index = FrameIndex(chroma_dir=os.path.join(output_dir, "_capture_chroma"))
    rules = AlertRulesEngine()
    agent = None if skip_agent else SecurityAgent(store, index)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_start = datetime.now(timezone.utc)

    frame_id = 0
    raw_idx = 0
    detection_count = 0

    print(f"Capturing every {every_n} {capture_on} frames → {output_dir}")
    print(f"Video: {video_path} ({total_frames} frames @ {fps:.1f} fps)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if recorder._max_captures is not None and recorder.capture_count >= recorder._max_captures:
            break

        if raw_idx % sample_every_n != 0:
            raw_idx += 1
            continue

        timestamp = (video_start + timedelta(seconds=raw_idx / fps)).isoformat()
        crops = subtractor.apply(frame)

        vlm_result = None
        alerts: List[Alert] = []
        agent_response = None
        vehicle_context = None
        activity = "clear"
        threat_level = "none"
        telemetry = get_telemetry(timestamp)

        if crops:
            vlm_result = analyzer.analyze_frame(crops)
            if vlm_result:
                detection_count += 1
                activity = activity_fn(vlm_result.objects, vlm_result.raw_description)
                lighting = analyze_frame_lighting(frame)
                alerts = rules.evaluate(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    location=location,
                    objects=vlm_result.objects,
                    activity=activity,
                    description=vlm_result.raw_description,
                    bbox=primary_crop_bbox(vlm_result),
                    color=primary_crop_color(vlm_result),
                    is_night=lighting.is_night,
                )
                threat_level = (
                    threat_fn(alerts, vlm_result.objects)
                    if threat_fn
                    else threat_from_alerts(alerts, vlm_result.objects)
                )
                telemetry = get_telemetry(
                    timestamp,
                    battery_pct=battery_for_frame(frame_id, total_frames // sample_every_n),
                )
                track = rules.last_track_update
                if track is not None:
                    vehicle_context = track_to_context(track)
                if agent is not None:
                    agent_response = agent.process(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        location=location,
                        objects=vlm_result.objects,
                        activity=activity,
                        description=vlm_result.raw_description,
                        pre_alerts=alerts,
                        vehicle_context=vehicle_context,
                        bbox=primary_crop_bbox(vlm_result),
                    )

        should_save = False
        if capture_on == "detection":
            should_save = vlm_result is not None and detection_count % every_n == 0
        else:
            should_save = recorder.should_capture(frame_id)

        if not should_save:
            raw_idx += 1
            frame_id += 1
            continue

        record = recorder.save_capture(
            frame_bgr=frame,
            frame_id=frame_id,
            raw_idx=raw_idx,
            timestamp=timestamp,
            location=location,
            crops=crops,
            vlm_result=vlm_result,
            alerts=alerts,
            agent_response=agent_response,
            activity=activity,
            threat_level=threat_level,
            telemetry=telemetry,
            vehicle_track=vehicle_context,
        )

        blip_summary = record.blip["raw_description"][:60] if record.blip else "no foreground"
        agent_summary = (record.agent_response or "skipped")[:60]
        print(f"  Saved frame {frame_id:04d} (raw {raw_idx:04d}) | BLIP: {blip_summary}")
        print(f"    Agent: {agent_summary}")

        raw_idx += 1
        frame_id += 1

    cap.release()
    store.close()

    manifest_path = recorder.write_manifest(video_path, sample_every_n, capture_on=capture_on)
    print(f"\nCaptured {recorder.capture_count} frames → {manifest_path}")
    return manifest_path
