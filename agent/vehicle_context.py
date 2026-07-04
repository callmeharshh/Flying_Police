"""Helpers for extracting crop metadata from VLM results."""
from typing import Optional, Tuple

from vlm.blip_analyzer import FrameVLMResult

BBox = Tuple[int, int, int, int]


def primary_crop_bbox(vlm_result: FrameVLMResult) -> Optional[BBox]:
    if not vlm_result.crops:
        return None
    return vlm_result.crops[0].bbox


def primary_crop_color(vlm_result: FrameVLMResult) -> str:
    if not vlm_result.crops:
        return ""
    return vlm_result.crops[0].color


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


# Backward-compatible aliases
primary_vehicle_bbox = primary_crop_bbox
primary_vehicle_color = primary_crop_color


def track_to_context(track) -> dict:
    return {
        "track_id": track.track_id,
        "is_continuing": track.is_continuing,
        "is_new_entry": track.is_new_entry,
        "entry_count": track.entry_count,
        "object_type": track.object_type,
        "prior_object_type": track.prior_object_type,
        "label_changed": track.label_changed,
        "frame_gap": track.frame_gap,
        "center_distance": track.center_distance,
        "max_match_distance": track.max_match_distance,
        "trajectory": track.trajectory,
    }
