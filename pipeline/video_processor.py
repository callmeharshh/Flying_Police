"""Frame-by-frame video processing with event stream for CLI and UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional

import cv2

from agent.alert_rules import AlertRulesEngine, threat_from_alerts
from agent.security_agent import SecurityAgent
from notifications.telegram_notifier import send_alert as telegram_alert
from agent.vehicle_context import primary_crop_bbox, primary_crop_color, track_to_context
from blockchain.evidence import build_alert_evidence
from blockchain.monad_client import MonadEvidenceClient
from config import SAMPLE_EVERY_N_FRAMES, MOG2_WARMUP_FRAMES
from data.simulated_frames import FrameAnalysis
from data.telemetry import battery_for_frame, get_telemetry
from pipeline.frame_preview import prepare_frame_preview
from storage.event_store import EventStore
from storage.frame_index import FrameIndex
from vlm.background_subtractor import BackgroundSubtractor
from vlm.blip_analyzer import BLIPAnalyzer
from vlm.lighting_detector import analyze_frame_lighting


def infer_activity(objects: list, caption: str) -> str:
    lower = caption.lower()
    if not objects:
        return "clear"
    if any(word in lower for word in ("walking", "approaching", "entering", "moving")):
        return "entering"
    if any(word in lower for word in ("standing", "loitering", "waiting", "stationary")):
        return "loitering"
    if any(word in lower for word in ("parked", "stopped", "sitting")):
        return "stationary"
    return "entering"


@dataclass
class PipelineEvent:
    kind: str
    message: str
    frame_id: Optional[int] = None
    raw_idx: Optional[int] = None
    severity: Optional[str] = None
    rule_id: Optional[str] = None
    extra: dict = field(default_factory=dict)


class VideoProcessor:
    def __init__(
        self,
        store: EventStore,
        index: FrameIndex,
        rules: AlertRulesEngine,
        agent: Optional[SecurityAgent] = None,
        evidence_client: Optional[MonadEvidenceClient] = None,
        location: str = "main_gate",
        sample_every_n: int = SAMPLE_EVERY_N_FRAMES,
    ):
        self._store = store
        self._index = index
        self._rules = rules
        self._agent = agent
        self._evidence_client = evidence_client or MonadEvidenceClient()
        self._location = location
        self._sample_every_n = sample_every_n

    def _emit_frame_bundle(
        self,
        frame,
        frame_id: int,
        raw_idx: int,
        status: str,
        frame_logs: List[str],
        frame_alerts: List[dict],
        bbox=None,
    ) -> PipelineEvent:
        return PipelineEvent(
            kind="frame",
            message=f"Frame {frame_id:03d} complete",
            frame_id=frame_id,
            raw_idx=raw_idx,
            extra={
                "image_rgb": prepare_frame_preview(frame, bbox=bbox),
                "status": status,
                "logs": frame_logs,
                "alerts": frame_alerts,
            },
        )

    def iter_events(self, video_path: str) -> Iterator[PipelineEvent]:
        subtractor = BackgroundSubtractor()
        analyzer = BLIPAnalyzer()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield PipelineEvent(kind="error", message=f"Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_total = max(1, total_raw_frames // self._sample_every_n)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = round(total_raw_frames / fps, 1)

        yield PipelineEvent(
            kind="info",
            message=f"Video: {video_path.split('/')[-1]} | {total_raw_frames} frames @ {fps:.1f} fps",
        )

        # --- Startup block (real stats from video file only) ---
        yield PipelineEvent(kind="startup", message="=== SESSION INFO ===")
        yield PipelineEvent(kind="startup", message=f"  Monitoring : {self._location.replace('_', ' ').upper()}")
        yield PipelineEvent(kind="startup", message=f"  Resolution : {frame_w}x{frame_h} px")
        yield PipelineEvent(kind="startup", message=f"  Frame rate : {fps:.1f} fps")
        yield PipelineEvent(kind="startup", message=f"  Duration   : {duration_s}s ({total_raw_frames} frames, sampling every {self._sample_every_n})")
        yield PipelineEvent(kind="startup", message=f"  Sampled    : ~{sampled_total} frames to process")
        yield PipelineEvent(kind="startup", message=f"=== BACKGROUND CALIBRATION ({MOG2_WARMUP_FRAMES} warmup frames) ===")

        frame_id = 0
        raw_idx = 0
        processed: List[FrameAnalysis] = []
        video_start = datetime.now(timezone.utc)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % self._sample_every_n != 0:
                raw_idx += 1
                continue

            timestamp = (video_start + timedelta(seconds=raw_idx / fps)).isoformat()
            progress = min(1.0, raw_idx / max(total_raw_frames, 1))
            frame_logs: List[str] = []
            frame_alerts: List[dict] = []

            yield PipelineEvent(
                kind="progress",
                message="Processing",
                frame_id=frame_id,
                raw_idx=raw_idx,
                extra={"progress": progress, "sampled_total": sampled_total},
            )

            crops = subtractor.apply(frame)
            if not crops:
                if not subtractor._warmed_up:
                    msg = f"  Calibrating background model... (frame {subtractor._warmup_count}/{MOG2_WARMUP_FRAMES})"
                    yield PipelineEvent(kind="warmup", message=msg, frame_id=frame_id, raw_idx=raw_idx)
                else:
                    msg = f"Frame {frame_id:03d} (raw {raw_idx:03d}) → no motion, skipped (BLIP not run)"
                    yield PipelineEvent(kind="skip", message=msg, frame_id=frame_id, raw_idx=raw_idx)
                raw_idx += 1
                frame_id += 1
                continue

            detect_msg = f"Frame {frame_id:03d} (raw {raw_idx:03d}) → {len(crops)} object(s) detected"
            frame_logs.append(detect_msg)
            yield PipelineEvent(
                kind="info",
                message=detect_msg,
                frame_id=frame_id,
                raw_idx=raw_idx,
            )

            vlm_result = analyzer.analyze_frame(crops)
            if not vlm_result:
                frame_logs.append("BLIP returned no result for this frame.")
                yield self._emit_frame_bundle(
                    frame, frame_id, raw_idx, "empty", frame_logs, frame_alerts
                )
                raw_idx += 1
                frame_id += 1
                continue

            activity = infer_activity(vlm_result.objects, vlm_result.raw_description)
            telemetry = get_telemetry(
                timestamp,
                battery_pct=battery_for_frame(frame_id, sampled_total),
            )
            crop_bbox = primary_crop_bbox(vlm_result)
            vehicle_color = primary_crop_color(vlm_result)
            lighting = analyze_frame_lighting(frame)
            alerts = self._rules.evaluate(
                frame_id=frame_id,
                timestamp=timestamp,
                location=self._location,
                objects=vlm_result.objects,
                activity=activity,
                description=vlm_result.raw_description,
                bbox=crop_bbox,
                color=vehicle_color,
                is_night=lighting.is_night,
            )
            threat = threat_from_alerts(alerts, vlm_result.objects)

            lighting_line = (
                f"Lighting: {lighting.label} "
                f"(brightness={lighting.brightness}, dark={lighting.dark_pixel_ratio:.0%})"
            )
            frame_logs.append(lighting_line)
            detection_line = (
                f"Objects: {vlm_result.objects} | Activity: {activity} | "
                f"Threat: {threat} | {vlm_result.raw_description[:100]}"
            )
            frame_logs.append(detection_line)
            yield PipelineEvent(
                kind="detection",
                message=vlm_result.raw_description[:120],
                frame_id=frame_id,
                raw_idx=raw_idx,
                severity=threat,
                extra={
                    "objects": vlm_result.objects,
                    "activity": activity,
                    "threat": threat,
                },
            )

            track = self._rules.last_track_update
            for alert in alerts:
                self._store.log_alert(
                    frame_id, alert.rule_id, alert.message, alert.severity
                )
                alert_entry = {
                    "rule_id": alert.rule_id,
                    "message": alert.message,
                    "severity": alert.severity,
                }
                frame_alerts.append(alert_entry)
                alert_line = f"🚨 [{alert.rule_id}] {alert.message}"
                frame_logs.append(alert_line)
                telegram_alert(
                    rule_id=alert.rule_id,
                    message=alert.message,
                    severity=alert.severity,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    location=self._location,
                    frame=frame,
                    bbox=crop_bbox,
                )
                yield PipelineEvent(
                    kind="alert",
                    message=alert.message,
                    frame_id=frame_id,
                    raw_idx=raw_idx,
                    severity=alert.severity,
                    rule_id=alert.rule_id,
                )
                evidence = build_alert_evidence(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    location=self._location,
                    rule_id=alert.rule_id,
                    severity=alert.severity,
                    message=alert.message,
                    description=vlm_result.raw_description,
                    objects=vlm_result.objects,
                    bbox=crop_bbox,
                    track_id=track.track_id if track else None,
                )
                anchor_result = self._evidence_client.anchor(evidence)
                self._store.log_evidence_anchor(
                    frame_id=frame_id,
                    evidence_hash=anchor_result.evidence_hash,
                    tx_hash=anchor_result.tx_hash,
                    status=anchor_result.status,
                    message=anchor_result.message,
                    location=anchor_result.location,
                    alert_message=anchor_result.alert_message,
                    ipfs_cid=anchor_result.ipfs_cid,
                )
                anchor_line = (
                    f"Monad evidence: {anchor_result.status} "
                    f"{anchor_result.evidence_hash[:14]}..."
                )
                if anchor_result.ipfs_cid:
                    anchor_line += f" ipfs={anchor_result.ipfs_cid[:14]}..."
                if anchor_result.tx_hash:
                    anchor_line += f" tx={anchor_result.tx_hash[:14]}..."
                frame_logs.append(anchor_line)

            vehicle_context = None
            if track is not None:
                vehicle_context = track_to_context(track)
                if track.is_continuing:
                    label_note = (
                        f", label {track.prior_object_type}→{track.object_type}"
                        if track.label_changed else ""
                    )
                    track_msg = f"Track: {track.track_id} (same object continuing{label_note})"
                else:
                    track_msg = f"Track: {track.track_id} (new object #{track.entry_count})"
                frame_logs.append(track_msg)
                yield PipelineEvent(
                    kind="track",
                    message=track_msg,
                    frame_id=frame_id,
                    raw_idx=raw_idx,
                )

            if self._agent is not None:
                agent_response = self._agent.process(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    location=self._location,
                    objects=vlm_result.objects,
                    activity=activity,
                    description=vlm_result.raw_description,
                    pre_alerts=alerts,
                    vehicle_context=vehicle_context,
                    bbox=crop_bbox,
                )
                agent_line = f"Agent: {agent_response}"
                frame_logs.append(agent_line)
                yield PipelineEvent(
                    kind="agent",
                    message=agent_response,
                    frame_id=frame_id,
                    raw_idx=raw_idx,
                )

            severity = threat if threat != "none" else "low"
            self._store.log_event(
                frame_id=frame_id,
                message=f"{vlm_result.raw_description} [{self._location}, {timestamp}]",
                severity=severity,
            )

            self._index.add_frame(
                frame_id=frame_id,
                description=vlm_result.raw_description,
                timestamp=timestamp,
                location=self._location,
                objects=vlm_result.objects,
                threat_level=threat,
                bbox=crop_bbox,
                track_id=track.track_id if track else None,
            )

            processed.append(FrameAnalysis(
                frame_id=frame_id,
                timestamp=timestamp,
                location=self._location,
                raw_description=vlm_result.raw_description,
                objects=vlm_result.objects,
                activity=activity,
                threat_level=threat,
                telemetry=telemetry,
            ))

            yield self._emit_frame_bundle(
                frame, frame_id, raw_idx, "detected", frame_logs, frame_alerts, bbox=crop_bbox
            )

            raw_idx += 1
            frame_id += 1

        cap.release()
        yield PipelineEvent(
            kind="done",
            message=f"Session complete — {len(processed)} frames analyzed",
            extra={
                "processed_count": len(processed),
                "alert_count": len(self._store.get_alerts()),
                "event_count": len(self._store.get_all_events()),
                "index_count": self._index.count(),
                "evidence_anchor_count": len(self._store.get_evidence_anchors()),
            },
        )
