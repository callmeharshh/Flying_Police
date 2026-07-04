import math
from datetime import datetime

# Fixed drone dock position over the property
_DRONE_LAT = 18.5204
_DRONE_LON = 73.8567
_DRONE_ALTITUDE_M = 10.0
_DRONE_H_FOV_DEG = 90.0   # horizontal field of view (typical wide-angle drone camera)


def get_telemetry(timestamp: str, battery_pct: int = 85) -> dict:
    """Return simulated telemetry for a given ISO timestamp."""
    return {
        "drone_lat": _DRONE_LAT,
        "drone_lon": _DRONE_LON,
        "altitude_m": _DRONE_ALTITUDE_M,
        "battery_pct": battery_pct,
        "timestamp": timestamp,
    }


def battery_for_frame(frame_index: int, total_frames: int, start_pct: int = 95) -> int:
    """Simulate battery drain across a session."""
    drain = int((frame_index / max(total_frames, 1)) * 15)
    return max(start_pct - drain, 10)


def estimate_ground_coverage(frame_w: int, frame_h: int,
                              altitude_m: float = _DRONE_ALTITUDE_M,
                              h_fov_deg: float = _DRONE_H_FOV_DEG) -> dict:
    """Estimate ground footprint visible from drone at given altitude."""
    h_fov_rad = math.radians(h_fov_deg)
    v_fov_rad = h_fov_rad * (frame_h / frame_w)   # scale by aspect ratio
    ground_w = round(2 * altitude_m * math.tan(h_fov_rad / 2), 1)
    ground_h = round(2 * altitude_m * math.tan(v_fov_rad / 2), 1)
    area_m2 = round(ground_w * ground_h, 1)
    px_per_meter = round(frame_w / ground_w, 1)
    return {
        "ground_width_m": ground_w,
        "ground_height_m": ground_h,
        "coverage_area_m2": area_m2,
        "px_per_meter": px_per_meter,
    }
