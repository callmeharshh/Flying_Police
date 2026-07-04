"""Track moving foreground objects across frames using spatial motion, not BLIP labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    KNOWN_VEHICLES,
    VEHICLE_TRACK_MAX_FRAME_GAP,
    VEHICLE_TRACK_MAX_TRAJECTORY_POINTS,
    VEHICLE_TRACK_MAX_VELOCITY_PX_PER_FRAME,
    VEHICLE_TRACK_VELOCITY_MARGIN,
)
from vlm.constants import MOTOR_VEHICLE_TYPES

BBox = Tuple[int, int, int, int]
Center = Tuple[float, float]


@dataclass
class TrackUpdate:
    track_id: str
    is_continuing: bool
    is_new_entry: bool
    entry_count: int
    object_type: str
    prior_object_type: Optional[str]
    label_changed: bool
    bbox: Optional[BBox]
    center: Optional[Center]
    frame_gap: int
    center_distance: float
    max_match_distance: float
    trajectory: List[dict] = field(default_factory=list)


@dataclass
class _ActiveTrack:
    track_id: str
    location: str
    last_bbox: BBox
    last_frame_id: int
    last_object_type: str
    centers: List[Center] = field(default_factory=list)


def _bbox_center(bbox: BBox) -> Center:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def _center_distance(a: Center, b: Center) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _max_match_distance(frame_gap: int) -> float:
    return (
        VEHICLE_TRACK_MAX_VELOCITY_PX_PER_FRAME
        * max(frame_gap, 1)
        * VEHICLE_TRACK_VELOCITY_MARGIN
    )


def build_visit_key(location: str, description: str, object_type: str) -> str:
    """Key for visit counting when a spatially new track appears."""
    if object_type in MOTOR_VEHICLE_TYPES:
        desc_lower = description.lower()
        for known in KNOWN_VEHICLES:
            if known in desc_lower:
                return f"{location}|{known}"
    return f"{location}|{object_type}"


def format_trajectory(centers: List[Center], labels: List[str]) -> List[dict]:
    trajectory = []
    for center, label in zip(centers, labels):
        trajectory.append({
            "center_x": round(center[0], 1),
            "center_y": round(center[1], 1),
            "object_type": label,
        })
    return trajectory


class VehicleTracker:
    """Match detections by motion path so one moving car stays one track across varying BLIP labels."""

    def __init__(self):
        self._active_tracks: Dict[str, _ActiveTrack] = {}
        self._visit_counts: Dict[str, int] = {}
        self._track_labels: Dict[str, List[str]] = {}
        self._next_id = 1

    def _expire_stale_tracks(self, frame_id: int) -> None:
        stale = [
            track_id
            for track_id, track in self._active_tracks.items()
            if frame_id - track.last_frame_id > VEHICLE_TRACK_MAX_FRAME_GAP
        ]
        for track_id in stale:
            del self._active_tracks[track_id]

    def _match_active_track(
        self,
        frame_id: int,
        location: str,
        bbox: BBox,
    ) -> tuple[Optional[_ActiveTrack], int, float, float]:
        self._expire_stale_tracks(frame_id)

        center = _bbox_center(bbox)
        best_match: Optional[_ActiveTrack] = None
        best_distance = float("inf")
        best_gap = 0
        best_max_dist = 0.0

        for track in self._active_tracks.values():
            if track.location != location:
                continue
            frame_gap = frame_id - track.last_frame_id
            if frame_gap <= 0:
                continue
            max_dist = _max_match_distance(frame_gap)
            distance = _center_distance(center, _bbox_center(track.last_bbox))
            if distance <= max_dist and distance < best_distance:
                best_distance = distance
                best_match = track
                best_gap = frame_gap
                best_max_dist = max_dist

        return best_match, best_gap, best_distance, best_max_dist

    def update(
        self,
        frame_id: int,
        location: str,
        description: str,
        bbox: Optional[BBox] = None,
        color: str = "",
        object_type: str = "unknown",
    ) -> Optional[TrackUpdate]:
        if bbox is None:
            return None

        center = _bbox_center(bbox)
        matched, frame_gap, center_distance, max_match_distance = self._match_active_track(
            frame_id, location, bbox
        )

        if matched is not None:
            prior_type = matched.last_object_type
            label_changed = prior_type != object_type
            matched.last_bbox = bbox
            matched.last_frame_id = frame_id
            matched.last_object_type = object_type
            matched.centers.append(center)
            if len(matched.centers) > VEHICLE_TRACK_MAX_TRAJECTORY_POINTS:
                matched.centers = matched.centers[-VEHICLE_TRACK_MAX_TRAJECTORY_POINTS:]

            labels = self._track_labels.setdefault(matched.track_id, [])
            labels.append(object_type)
            if len(labels) > VEHICLE_TRACK_MAX_TRAJECTORY_POINTS:
                self._track_labels[matched.track_id] = labels[-VEHICLE_TRACK_MAX_TRAJECTORY_POINTS:]

            visit_key = build_visit_key(location, description, object_type)
            return TrackUpdate(
                track_id=matched.track_id,
                is_continuing=True,
                is_new_entry=False,
                entry_count=self._visit_counts.get(visit_key, 1),
                object_type=object_type,
                prior_object_type=prior_type if label_changed else None,
                label_changed=label_changed,
                bbox=bbox,
                center=center,
                frame_gap=frame_gap,
                center_distance=round(center_distance, 1),
                max_match_distance=round(max_match_distance, 1),
                trajectory=format_trajectory(matched.centers, self._track_labels[matched.track_id]),
            )

        track_id = f"track_{self._next_id:03d}"
        self._next_id += 1
        visit_key = build_visit_key(location, description, object_type)
        self._visit_counts[visit_key] = self._visit_counts.get(visit_key, 0) + 1
        entry_count = self._visit_counts[visit_key]

        self._active_tracks[track_id] = _ActiveTrack(
            track_id=track_id,
            location=location,
            last_bbox=bbox,
            last_frame_id=frame_id,
            last_object_type=object_type,
            centers=[center],
        )
        self._track_labels[track_id] = [object_type]

        return TrackUpdate(
            track_id=track_id,
            is_continuing=False,
            is_new_entry=True,
            entry_count=entry_count,
            object_type=object_type,
            prior_object_type=None,
            label_changed=False,
            bbox=bbox,
            center=center,
            frame_gap=0,
            center_distance=0.0,
            max_match_distance=_max_match_distance(1),
            trajectory=format_trajectory([center], [object_type]),
        )

    def reset(self) -> None:
        self._active_tracks.clear()
        self._visit_counts.clear()
        self._track_labels.clear()
        self._next_id = 1
