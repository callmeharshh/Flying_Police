"""Merge fragmented motion tracks into distinct physical individuals."""
from __future__ import annotations

from typing import Dict, List

from config import MIN_DISTINCT_PERSON_SEPARATION_PX, VEHICLE_TRACK_MAX_FRAME_GAP


def _center_distance(a: dict, b: dict) -> float:
    return ((a["center_x"] - b["center_x"]) ** 2 + (a["center_y"] - b["center_y"]) ** 2) ** 0.5


def _build_track_details(frames: List[dict]) -> Dict[str, dict]:
    tracks: Dict[str, dict] = {}
    for frame in frames:
        track_id = frame.get("track_id") or f"untracked_frame_{frame['frame_id']}"
        if track_id not in tracks:
            tracks[track_id] = {
                "track_id": track_id,
                "location": frame["location"],
                "frames": [],
                "sample_description": frame["description"][:80],
            }
        tracks[track_id]["frames"].append({
            "frame_id": frame["frame_id"],
            "center_x": frame.get("center_x"),
            "center_y": frame.get("center_y"),
        })

    for track in tracks.values():
        track["frames"].sort(key=lambda item: item["frame_id"])
        track["first_frame"] = track["frames"][0]["frame_id"]
        track["last_frame"] = track["frames"][-1]["frame_id"]
        track["frame_count"] = len(track["frames"])

    return tracks


def _overlap_min_distance(track_a: dict, track_b: dict) -> float | None:
    frames_a = {
        item["frame_id"]: item
        for item in track_a["frames"]
        if item.get("center_x") is not None
    }
    frames_b = {
        item["frame_id"]: item
        for item in track_b["frames"]
        if item.get("center_x") is not None
    }

    shared_frames = set(frames_a) & set(frames_b)
    if not shared_frames:
        return None

    return min(_center_distance(frames_a[fid], frames_b[fid]) for fid in shared_frames)


def _tracks_should_merge(track_a: dict, track_b: dict) -> bool:
    if track_a["location"] != track_b["location"]:
        return False

    overlap_start = max(track_a["first_frame"], track_b["first_frame"])
    overlap_end = min(track_a["last_frame"], track_b["last_frame"])

    if overlap_start <= overlap_end:
        min_distance = _overlap_min_distance(track_a, track_b)
        if min_distance is None:
            return True
        return min_distance < MIN_DISTINCT_PERSON_SEPARATION_PX

    if track_a["last_frame"] < track_b["first_frame"]:
        gap = track_b["first_frame"] - track_a["last_frame"]
    else:
        gap = track_a["first_frame"] - track_b["last_frame"]

    return gap <= VEHICLE_TRACK_MAX_FRAME_GAP


def _union_find_merge(track_ids: List[str], should_merge) -> Dict[str, str]:
    parent = {track_id: track_id for track_id in track_ids}

    def find(track_id: str) -> str:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def unite(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left_id in enumerate(track_ids):
        for right_id in track_ids[i + 1:]:
            if should_merge(left_id, right_id):
                unite(left_id, right_id)

    return {track_id: find(track_id) for track_id in track_ids}


def merge_tracks_into_individuals(frames: List[dict]) -> tuple[List[dict], List[dict]]:
    """Return (raw_tracks, merged_individuals)."""
    track_details = _build_track_details(frames)
    if not track_details:
        return [], []

    track_ids = list(track_details.keys())
    roots = _union_find_merge(
        track_ids,
        lambda left_id, right_id: _tracks_should_merge(
            track_details[left_id],
            track_details[right_id],
        ),
    )

    raw_tracks = sorted(track_details.values(), key=lambda track: track["first_frame"])

    individuals: Dict[str, dict] = {}
    for track_id, root_id in roots.items():
        track = track_details[track_id]
        if root_id not in individuals:
            individuals[root_id] = {
                "individual_id": root_id,
                "track_ids": [],
                "first_frame": track["first_frame"],
                "last_frame": track["last_frame"],
                "frame_count": 0,
                "sample_description": track["sample_description"],
            }

        person = individuals[root_id]
        person["track_ids"].append(track_id)
        person["first_frame"] = min(person["first_frame"], track["first_frame"])
        person["last_frame"] = max(person["last_frame"], track["last_frame"])
        person["frame_count"] += track["frame_count"]

    merged = sorted(individuals.values(), key=lambda person: person["first_frame"])
    return raw_tracks, merged
