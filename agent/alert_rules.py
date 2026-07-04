from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from config import LOITER_THRESHOLD_SECONDS, REPEAT_ENTRY_LIMIT, KNOWN_VEHICLES
from agent.vehicle_tracker import VehicleTracker, TrackUpdate

BBox = Tuple[int, int, int, int]

SECONDS_PER_MINUTE = 60


def format_dwell_duration(dwell_seconds: float) -> str:
    """Format elapsed dwell time for alert messages."""
    total_seconds = int(dwell_seconds)
    if total_seconds < SECONDS_PER_MINUTE:
        return f"{total_seconds} sec"
    minutes, seconds = divmod(total_seconds, SECONDS_PER_MINUTE)
    if seconds == 0:
        return f"{minutes} min"
    return f"{minutes} min {seconds} sec"


@dataclass
class Alert:
    rule_id: str
    frame_id: int
    message: str
    severity: str


def threat_from_alerts(alerts: List[Alert], objects: List[str]) -> str:
    """Threat level is derived from fired alerts — HIGH only when a rule alert exists."""
    if not objects:
        return "none"
    if any(alert.severity == "high" for alert in alerts):
        return "high"
    if any(alert.severity == "medium" for alert in alerts):
        return "medium"
    return "low"


class AlertRulesEngine:
    def __init__(self):
        self._vehicle_tracker = VehicleTracker()
        self._location_first_seen: dict[str, datetime] = {}
        self._loiter_alerted: set[str] = set()
        self._prolonged_loiter_alerted: set[str] = set()
        self._after_hours_person_alerted: set[str] = set()
        self._after_hours_perimeter_alerted: set[str] = set()
        self._fallback_visit_counts: dict[str, int] = {}
        self._last_track_update: Optional[TrackUpdate] = None

    @property
    def last_track_update(self) -> Optional[TrackUpdate]:
        return self._last_track_update

    def _is_known_vehicle(self, description: str) -> bool:
        desc_lower = description.lower()
        return any(v in desc_lower for v in KNOWN_VEHICLES)

    def _primary_motor_type(self, objects: List[str]) -> Optional[str]:
        if "police_car" in objects:
            return "police_car"
        if "vehicle" in objects:
            return "vehicle"
        return None

    def _is_unrecognized_vehicle(self, objects: List[str], description: str) -> bool:
        if "police_car" in objects:
            return False
        return not self._is_known_vehicle(description)

    def _is_new_after_hours_sighting(
        self,
        location: str,
        track: Optional[TrackUpdate],
        alerted_keys: set[str],
        fallback_suffix: str,
    ) -> bool:
        """Fire at most once per tracked object (or once per location without bbox)."""
        if track is not None:
            if not track.is_new_entry:
                return False
            key = f"{location}|{track.track_id}"
        else:
            key = f"{location}|{fallback_suffix}"
            if key in alerted_keys:
                return False
        if key in alerted_keys:
            return False
        alerted_keys.add(key)
        return True

    def evaluate(
        self,
        frame_id: int,
        timestamp: str,
        location: str,
        objects: List[str],
        activity: str,
        description: str,
        bbox: Optional[BBox] = None,
        color: str = "",
        is_night: bool = False,
    ) -> List[Alert]:
        alerts = []
        dt = datetime.fromisoformat(timestamp)
        self._last_track_update = None

        # Spatial tracking first — rule dedup uses track continuity
        object_type = objects[0] if objects else "unknown"
        if bbox is not None:
            track = self._vehicle_tracker.update(
                frame_id=frame_id,
                location=location,
                description=description,
                bbox=bbox,
                color=color,
                object_type=object_type,
            )
            self._last_track_update = track

        track = self._last_track_update

        # RULE-01: Person detected at night (once per person track)
        if (
            "person" in objects
            and is_night
            and self._is_new_after_hours_sighting(
                location, track, self._after_hours_person_alerted, "person"
            )
        ):
            alerts.append(Alert(
                rule_id="RULE-01",
                frame_id=frame_id,
                message=f"Person detected at night at {location}",
                severity="high",
            ))

        motor_type = self._primary_motor_type(objects)

        if motor_type is not None:
            if track is not None:
                if self._is_unrecognized_vehicle(objects, description) and track.is_new_entry:
                    alerts.append(Alert(
                        rule_id="RULE-02",
                        frame_id=frame_id,
                        message=f"Unrecognized vehicle at {location}, {dt.strftime('%H:%M')}",
                        severity="medium",
                    ))
                if activity == "entering" and track.is_new_entry and track.entry_count > REPEAT_ENTRY_LIMIT:
                    alerts.append(Alert(
                        rule_id="RULE-03",
                        frame_id=frame_id,
                        message=(
                            f"Vehicle re-entered {location} "
                            f"({track.entry_count} visits today, track {track.track_id})"
                        ),
                        severity="medium",
                    ))
            else:
                # Simulated / no-bbox fallback — no spatial data available
                if self._is_unrecognized_vehicle(objects, description):
                    alerts.append(Alert(
                        rule_id="RULE-02",
                        frame_id=frame_id,
                        message=f"Unrecognized vehicle at {location}, {dt.strftime('%H:%M')}",
                        severity="medium",
                    ))
                if activity == "entering":
                    self._fallback_visit_counts[location] = (
                        self._fallback_visit_counts.get(location, 0) + 1
                    )
                    count = self._fallback_visit_counts[location]
                    if count > REPEAT_ENTRY_LIMIT:
                        alerts.append(Alert(
                            rule_id="RULE-03",
                            frame_id=frame_id,
                            message=f"Vehicle entered {location} {count} times today",
                            severity="medium",
                        ))

        # RULE-04: Person loitering / stationary (alert on first sighting + prolonged dwell)
        if "person" in objects and activity in ("loitering", "stationary"):
            loc_key = f"person_{location}"
            if loc_key not in self._location_first_seen:
                self._location_first_seen[loc_key] = dt

            if loc_key not in self._loiter_alerted:
                alerts.append(Alert(
                    rule_id="RULE-04",
                    frame_id=frame_id,
                    message=f"Person {activity} at {location}",
                    severity="high",
                ))
                self._loiter_alerted.add(loc_key)
            else:
                dwell = (dt - self._location_first_seen[loc_key]).total_seconds()
                if (
                    dwell >= LOITER_THRESHOLD_SECONDS
                    and loc_key not in self._prolonged_loiter_alerted
                ):
                    alerts.append(Alert(
                        rule_id="RULE-04",
                        frame_id=frame_id,
                        message=(
                            f"Person loitering at {location} "
                            f"for {format_dwell_duration(dwell)}"
                        ),
                        severity="high",
                    ))
                    self._prolonged_loiter_alerted.add(loc_key)

        # RULE-05: Any activity at perimeter at night (once per track / location)
        if (
            location == "perimeter"
            and objects
            and is_night
            and self._is_new_after_hours_sighting(
                location, track, self._after_hours_perimeter_alerted, "activity"
            )
        ):
            alerts.append(Alert(
                rule_id="RULE-05",
                frame_id=frame_id,
                message=f"Activity detected at perimeter at night",
                severity="high",
            ))

        return alerts

    def reset(self):
        self._vehicle_tracker.reset()
        self._location_first_seen.clear()
        self._loiter_alerted.clear()
        self._prolonged_loiter_alerted.clear()
        self._after_hours_person_alerted.clear()
        self._after_hours_perimeter_alerted.clear()
        self._fallback_visit_counts.clear()
        self._last_track_update = None
