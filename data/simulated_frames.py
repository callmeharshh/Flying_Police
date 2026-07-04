from dataclasses import dataclass, field
from typing import List
from data.telemetry import get_telemetry, battery_for_frame


@dataclass
class FrameAnalysis:
    frame_id: int
    timestamp: str          # ISO 8601
    location: str           # main_gate | garage | perimeter
    raw_description: str    # BLIP captions joined
    objects: List[str]      # e.g. ["vehicle", "person"]
    activity: str           # entering | exiting | loitering | stationary | clear
    threat_level: str       # high | medium | low | none
    telemetry: dict = field(default_factory=dict)
    image_path: str = ""    # optional path to source image/frame


# 7 simulated scenarios from PRD Section 9
SIMULATED_SCENARIOS = [
    {
        "frame_id": 1,
        "timestamp": "2026-06-09T08:00:00",
        "location": "main_gate",
        "raw_description": "A blue Ford F150 truck entering through the gate",
        "objects": ["vehicle"],
        "activity": "entering",
        "threat_level": "low",
    },
    {
        "frame_id": 2,
        "timestamp": "2026-06-09T12:00:00",
        "location": "garage",
        "raw_description": "A blue Ford F150 truck parked in the garage area",
        "objects": ["vehicle"],
        "activity": "stationary",
        "threat_level": "none",
    },
    {
        "frame_id": 3,
        "timestamp": "2026-06-09T08:30:00",
        "location": "main_gate",
        "raw_description": "A blue Ford F150 truck entering through the gate again",
        "objects": ["vehicle"],
        "activity": "entering",
        "threat_level": "medium",
    },
    {
        "frame_id": 4,
        "timestamp": "2026-06-09T23:58:00",
        "location": "main_gate",
        "raw_description": "A person approaching the gate at night",
        "objects": ["person"],
        "activity": "entering",
        "threat_level": "high",
    },
    {
        "frame_id": 5,
        "timestamp": "2026-06-10T00:01:00",
        "location": "main_gate",
        "raw_description": "A person standing at the gate at midnight",
        "objects": ["person"],
        "activity": "loitering",
        "threat_level": "high",
    },
    {
        "frame_id": 6,
        "timestamp": "2026-06-09T14:00:00",
        "location": "perimeter",
        "raw_description": "An unknown white sedan parked near the perimeter fence",
        "objects": ["vehicle"],
        "activity": "stationary",
        "threat_level": "medium",
    },
    {
        "frame_id": 7,
        "timestamp": "2026-06-09T09:00:00",
        "location": "garage",
        "raw_description": "No activity detected in the garage area",
        "objects": [],
        "activity": "clear",
        "threat_level": "none",
    },
]

TOTAL_FRAMES = len(SIMULATED_SCENARIOS)


def get_simulated_frames() -> List[FrameAnalysis]:
    """Return all simulated scenarios as FrameAnalysis objects with telemetry attached."""
    frames = []
    for i, s in enumerate(SIMULATED_SCENARIOS):
        telemetry = get_telemetry(
            timestamp=s["timestamp"],
            battery_pct=battery_for_frame(i, TOTAL_FRAMES),
        )
        frames.append(FrameAnalysis(**s, telemetry=telemetry))
    return frames
