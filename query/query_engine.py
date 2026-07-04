import re
from typing import List, Optional

from query.track_merge import merge_tracks_into_individuals
from storage.event_store import EventStore
from storage.frame_index import FrameIndex
from vlm.constants import PERSON_KEYWORDS

COUNT_QUERY_PATTERN = re.compile(
    r"\b(how many|count|number of|total)\b",
    re.IGNORECASE,
)

OBJECT_TYPE_ALIASES = {
    "person": PERSON_KEYWORDS,
    "vehicle": frozenset({
        "vehicle", "vehicles", "car", "cars", "truck", "trucks", "van", "suv",
    }),
    "police_car": frozenset({"police car", "police_car", "police cars", "patrol car"}),
    "bicycle": frozenset({"bicycle", "bicycles", "bike", "bikes", "cyclist"}),
    "animal": frozenset({"animal", "animals", "dog", "dogs", "cat", "cats"}),
}


class QueryEngine:
    def __init__(self, store: EventStore, index: FrameIndex):
        self._store = store
        self._index = index

    def query(self, text: str) -> List[dict]:
        text_lower = text.lower()

        if COUNT_QUERY_PATTERN.search(text_lower):
            return [self._query_count(text_lower)]

        # Time-based queries → SQLite
        if any(w in text_lower for w in ["midnight", "night", "after hours", "22:", "23:", "00:", "01:"]):
            return self._query_by_time(text_lower)

        # Alert-only queries → SQLite alerts table
        if any(w in text_lower for w in ["alert", "alerts", "flagged", "threat"]):
            return self._query_alerts(text_lower)

        # Location queries → ChromaDB with filter
        location = self._extract_location(text_lower)
        if location:
            results = self._index.query(text, n_results=5, location=location)
            if results:
                return results

        # Default: semantic search in ChromaDB
        return self._index.query(text, n_results=5)

    def _query_count(self, text: str) -> dict:
        object_type = self._extract_object_type(text)
        location = self._extract_location(text)
        frames = self._index.list_all_frames()

        if location:
            frames = [f for f in frames if f["location"] == location]

        matching = [f for f in frames if self._frame_matches_object(f, object_type)]
        raw_tracks, individuals = merge_tracks_into_individuals(matching)

        return {
            "_type": "count",
            "object_type": object_type,
            "location": location or "all locations",
            "unique_count": len(individuals),
            "track_count": len(raw_tracks),
            "frame_count": len(matching),
            "individuals": individuals,
            "tracks": raw_tracks,
        }

    def _frame_matches_object(self, frame: dict, object_type: str) -> bool:
        object_tags = {tag.strip() for tag in frame.get("objects", "").split(",") if tag.strip()}
        if object_type in object_tags:
            return True

        description = frame.get("description", "").lower()
        keywords = OBJECT_TYPE_ALIASES.get(object_type, frozenset({object_type}))
        return any(keyword in description for keyword in keywords)

    def _extract_object_type(self, text: str) -> str:
        for object_type, aliases in OBJECT_TYPE_ALIASES.items():
            if any(alias in text for alias in aliases):
                return object_type
        return "person"

    def _query_by_time(self, text: str) -> List[dict]:
        events = self._store.get_all_events()
        matched = []
        for e in events:
            ts = e.get("timestamp", "")
            if any(k in ts for k in ["T22", "T23", "T00", "T01", "T02"]):
                matched.append({
                    "frame_id": e["frame_id"],
                    "description": e["message"],
                    "timestamp": ts,
                    "location": "",
                    "objects": "",
                    "threat_level": e["severity"],
                    "score": 1.0,
                })
        return matched

    def _query_alerts(self, text: str) -> List[dict]:
        severity = None
        if "high" in text:
            severity = "high"
        elif "medium" in text:
            severity = "medium"
        alerts = self._store.get_alerts(severity=severity)
        if not alerts:
            return [{
                "_type": "alerts_empty",
                "severity_filter": severity,
                "high_threat_observations": self._count_high_threat_observations(),
            }]

        return [{
            "frame_id": a["frame_id"],
            "description": a["message"],
            "timestamp": a["timestamp"],
            "location": "",
            "objects": "",
            "threat_level": a["severity"],
            "score": 1.0,
            "rule_id": a["rule_id"],
        } for a in alerts]

    def _count_high_threat_observations(self) -> int:
        return sum(
            1 for event in self._store.get_all_events()
            if event.get("severity") == "high" and event.get("type") != "alert"
        )

    def _extract_location(self, text: str) -> str:
        if "main gate" in text or "gate" in text:
            return "main_gate"
        if "garage" in text:
            return "garage"
        if "perimeter" in text:
            return "perimeter"
        return ""

    def format_results(self, results: List[dict]) -> str:
        if not results:
            return "  No results found."

        if results[0].get("_type") == "count":
            return self._format_count_result(results[0])

        if results[0].get("_type") == "alerts_empty":
            return self._format_alerts_empty_result(results[0])

        lines = []
        for r in results:
            rule = f" [{r['rule_id']}]" if "rule_id" in r else ""
            score = f" (score={r['score']})" if r.get("score", 1.0) < 1.0 else ""
            lines.append(f"  Frame {r['frame_id']} | {r['timestamp'][:16]}{rule}{score}")
            lines.append(f"    {r['description'][:80]}")
        return "\n".join(lines)

    def _format_alerts_empty_result(self, result: dict) -> str:
        filter_note = ""
        if result.get("severity_filter"):
            filter_note = f" (filter: {result['severity_filter']} severity)"

        lines = [
            f"  No rule alerts were triggered this session{filter_note}.",
            "  Alerts are only created when a rule fires (RULE-01 … RULE-05) or the agent "
            "calls trigger_alert — they are stored in the alerts table in data/events.db.",
        ]

        high_count = result.get("high_threat_observations", 0)
        if high_count:
            lines.append(
                f"  Warning: {high_count} HIGH event-log entries exist without matching alerts "
                f"— threat level and alerts may be out of sync (re-run with latest code to align)."
            )
        else:
            lines.append(
                "  Nothing met the alert thresholds this session "
                "(e.g. RULE-01 after-hours person, RULE-04 loitering/stationary, RULE-02 unknown vehicle)."
            )

        return "\n".join(lines)

    def _format_count_result(self, result: dict) -> str:
        label = result["object_type"].replace("_", " ")
        track_note = ""
        if result["track_count"] > result["unique_count"]:
            track_note = (
                f" Motion tracker split them into {result['track_count']} track IDs "
                f"— merged by position and timing."
            )

        lines = [
            f"  Answer: {result['unique_count']} distinct {label}(s) in the video "
            f"({result['frame_count']} frames with {label} sightings at {result['location']}).{track_note}",
            f"  Note: Per-frame logs show '1 object detected' because each frame captures "
            f"one foreground blob; the count above is across the whole session.",
        ]
        if not result["individuals"]:
            lines.append("  No matching detections found.")
            return "\n".join(lines)

        lines.append("  Individuals:")
        for person in result["individuals"]:
            frame_range = (
                f"frame {person['first_frame']}"
                if person["first_frame"] == person["last_frame"]
                else f"frames {person['first_frame']}-{person['last_frame']}"
            )
            track_ids = ", ".join(sorted(person["track_ids"]))
            lines.append(
                f"    - {frame_range} ({person['frame_count']} sighting(s), "
                f"tracks: {track_ids}) — {person['sample_description']}"
            )
        return "\n".join(lines)

    def run_interactive(self):
        print("\nQuery interface ready. Type a question or 'quit' to exit.")
        print("Examples: 'how many men were detected?' | 'any alerts?' | 'activity at main gate'\n")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            results = self.query(q)
            print(self.format_results(results))
            print()
