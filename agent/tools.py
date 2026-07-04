from langchain.tools import tool
from storage.event_store import EventStore
from storage.frame_index import FrameIndex

_store: EventStore = None
_index: FrameIndex = None


def init_tools(store: EventStore, index: FrameIndex):
    global _store, _index
    _store = store
    _index = index


@tool
def log_event(message: str) -> str:
    """Log a security observation to the event store. Use for all normal activity."""
    _store.log_event(frame_id=0, message=message, severity="low", type="log")
    return f"Logged: {message}"


@tool
def trigger_alert(message: str, severity: str = "medium") -> str:
    """Trigger a security alert. severity: low, medium, or high.
    Use for suspicious or threatening activity beyond pre-fired rule alerts."""
    if severity not in {"low", "medium", "high"}:
        severity = "medium"
    _store.log_alert(frame_id=0, rule_id="AGENT", message=message, severity=severity)
    return f"Alert triggered [{severity}]: {message}"


@tool
def query_history(query: str) -> str:
    """Search past frame descriptions to check if an object or event was seen before.
    Use before logging to provide context-aware observations."""
    results = _index.query(query, n_results=3)
    if not results:
        return "No matching history found."
    return "\n".join(_format_history_line(r) for r in results)


@tool
def query_track_positions(location: str) -> str:
    """Return recent object positions at a location (frame_id, center_x, center_y, track_id, label).
    Use to decide if the current detection is the same moving object continuing from prior frames."""
    results = _index.get_recent_at_location(location, n_results=6)
    if not results:
        return "No prior positions at this location."
    return "\n".join(_format_position_line(r) for r in results)


def _format_history_line(r: dict) -> str:
    pos = ""
    if "center_x" in r:
        pos = f" center=({r['center_x']}, {r['center_y']})"
    track = f" track={r['track_id']}" if r.get("track_id") else ""
    return (
        f"- Frame {r['frame_id']} at {r['location']} ({r['timestamp'][:16]}){pos}{track}: "
        f"{r['description']}"
    )


def _format_position_line(r: dict) -> str:
    if "center_x" not in r:
        return f"- Frame {r['frame_id']}: {r['description'][:60]} (no position data)"
    track = f", track={r['track_id']}" if r.get("track_id") else ""
    return (
        f"- Frame {r['frame_id']}: center=({r['center_x']}, {r['center_y']})"
        f", objects={r.get('objects', '')}{track}"
    )
