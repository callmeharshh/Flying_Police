"""
Flying Police — CLI Entry Point
"""
import os
import shutil

from agent.alert_rules import AlertRulesEngine
from agent.security_agent import SecurityAgent
from config import CHROMA_DIR, DEMO_VIDEO, EVENTS_DB_PATH, SAMPLE_VIDEO_DIR
from pipeline.video_processor import VideoProcessor
from query.query_engine import QueryEngine
from storage.event_store import EventStore
from storage.frame_index import FrameIndex

VIDEO_PATH = os.path.join(SAMPLE_VIDEO_DIR, DEMO_VIDEO)
LOCATION = "main_gate"


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _print_event(event) -> None:
    if event.kind == "startup":
        print(f"  {event.message}")
    elif event.kind == "warmup":
        print(f"  {event.message}")
    elif event.kind == "skip":
        print(f"  {event.message}")
    elif event.kind == "info":
        print(f"  {event.message}")
    elif event.kind == "frame":
        extra = event.extra
        print(f"  Frame {event.frame_id:03d} (raw {event.raw_idx:03d}) — {extra.get('status')}")
        for line in extra.get("logs", []):
            print(f"    {line}")
    elif event.kind == "error":
        print(f"ERROR: {event.message}")
    elif event.kind == "done":
        print(f"\n  {event.message}")


def main():
    print_section("FLYING POLICE")

    for path in [EVENTS_DB_PATH, CHROMA_DIR]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    store = EventStore()
    index = FrameIndex()
    rules = AlertRulesEngine()
    agent = SecurityAgent(store, index)
    processor = VideoProcessor(store, index, rules, agent, location=LOCATION)

    for event in processor.iter_events(VIDEO_PATH):
        if event.kind != "progress":
            _print_event(event)

    all_alerts = store.get_alerts()
    print_section(f"ALERTS ({len(all_alerts)} triggered)")
    if all_alerts:
        for alert in all_alerts:
            print(f"  [{alert['severity'].upper():6}] [{alert['rule_id']}] {alert['message']}")
    else:
        print("  No alerts triggered.")

    all_events = store.get_all_events()
    print_section(f"EVENT LOG ({len(all_events)} events in SQLite)")
    for event in all_events:
        print(f"  [{event['severity'].upper():6}] Frame {event['frame_id']} | {event['message'][:65]}")

    print_section(f"FRAME INDEX ({index.count()} frames in ChromaDB)")
    for query in ("person walking", "vehicle entering", "person at door"):
        results = index.query(query, n_results=2)
        print(f"\n  Query: \"{query}\"")
        if results:
            for result in results:
                print(f"    [score={result['score']}] Frame {result['frame_id']} | {result['description'][:60]}")
        else:
            print("    No results")

    print_section("SESSION COMPLETE")
    QueryEngine(store, index).run_interactive()
    store.close()


if __name__ == "__main__":
    main()
