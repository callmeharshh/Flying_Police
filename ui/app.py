"""
Flying Police — Web UI

Run:
    streamlit run ui/app.py
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv

from agent.alert_rules import AlertRulesEngine
from agent.security_agent import SecurityAgent
from config import (
    EVIDENCE_REGISTRY_ADDRESS,
    LIGHTHOUSE_API_KEY,
    LOCATIONS,
    MONAD_CHAIN_ID,
    MONAD_EXPLORER_TX_URL,
    MONAD_PRIVATE_KEY,
    MONAD_RPC_URL,
)
from pipeline.session import create_session_id, reset_session, save_upload
from pipeline.video_processor import VideoProcessor
from query.query_engine import QueryEngine
from storage.event_store import EventStore
from storage.frame_index import FrameIndex

load_dotenv()

MAX_LOG_LINES = 400
ALERT_SEVERITY_COLORS = {
    "high": "#ff4b4b",
    "medium": "#ffa500",
    "low": "#4b9fff",
}
STATUS_LABELS = {
    "detected": "Motion detected — object analyzed",
    "empty": "Motion detected — BLIP returned no result",
}
ANCHOR_STATUS_COLORS = {
    "anchored": "#16a34a",
    "not_configured": "#ca8a04",
    "ipfs_error": "#ea580c",
    "error": "#dc2626",
}


def _init_state() -> None:
    defaults = {
        "session_id": None,
        "processing_complete": False,
        "log_lines": [],
        "alert_feed": [],
        "chat_history": [],
        "store": None,
        "index": None,
        "stats": {},
        "current_frame": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _severity_badge(severity: str) -> str:
    color = ALERT_SEVERITY_COLORS.get(severity, "#888")
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.75rem;'>{severity.upper()}</span>"
    )


def _render_alert_card(alert: dict) -> None:
    frame_label = f"Frame {alert['frame_id']} · " if alert.get("frame_id") is not None else ""
    st.markdown(
        f"{_severity_badge(alert['severity'])} "
        f"**[{alert['rule_id']}]** {frame_label}"
        f"{alert['message']}",
        unsafe_allow_html=True,
    )


def _render_frame_panel(frame_data: dict, *, title: str = "Current frame") -> None:
    st.markdown(f"**{title}**")
    st.image(
        frame_data["image_rgb"],
        caption=(
            f"Frame {frame_data['frame_id']:03d} (raw {frame_data['raw_idx']:03d}) · "
            f"{STATUS_LABELS.get(frame_data['status'], frame_data['status'])}"
        ),
        use_container_width=True,
    )
    st.markdown("**Logs for this frame**")
    st.code("\n".join(frame_data["logs"]), language=None)
    st.markdown("**Alerts for this frame**")
    if frame_data["alerts"]:
        for alert in frame_data["alerts"]:
            _render_alert_card({
                "frame_id": frame_data["frame_id"],
                "rule_id": alert["rule_id"],
                "message": alert["message"],
                "severity": alert["severity"],
            })
    else:
        st.caption("No alerts on this frame.")


def _render_session_log(log_slot) -> None:
    with log_slot.container():
        st.subheader("Session log")
        if st.session_state.log_lines:
            st.code("\n".join(st.session_state.log_lines[-MAX_LOG_LINES:]))
        else:
            st.text("Full session log builds up as frames are processed.")


def _render_alert_feed(alerts_slot) -> None:
    with alerts_slot.container():
        st.subheader("All alerts")
        if st.session_state.alert_feed:
            for alert in st.session_state.alert_feed[:20]:
                _render_alert_card(alert)
        else:
            st.text("Alerts from the whole session appear here.")


def _short_hash(value: str | None, *, length: int = 10) -> str:
    if not value:
        return "—"
    if len(value) <= length * 2:
        return value
    return f"{value[:length]}…{value[-length:]}"


def _tx_url(tx_hash: str | None) -> str | None:
    if not tx_hash or not MONAD_EXPLORER_TX_URL:
        return None
    return f"{MONAD_EXPLORER_TX_URL.rstrip('/')}/{tx_hash}"


def _render_monad_status() -> None:
    monad_configured = bool(
        MONAD_RPC_URL and MONAD_CHAIN_ID and MONAD_PRIVATE_KEY and EVIDENCE_REGISTRY_ADDRESS
    )
    lighthouse_configured = bool(LIGHTHOUSE_API_KEY)
    monad_label = "Ready for live anchoring" if monad_configured else "Local proof mode"
    ipfs_label = "Lighthouse IPFS ready" if lighthouse_configured else "IPFS key missing"
    st.caption(
        f"Monad: {monad_label} · {ipfs_label} · chain {MONAD_CHAIN_ID or '—'} · "
        f"contract {_short_hash(EVIDENCE_REGISTRY_ADDRESS)}"
    )


def _render_evidence_receipts(store: EventStore | None) -> None:
    st.subheader("Monad Evidence Receipts")
    _render_monad_status()
    if store is None:
        st.caption("Run a session to generate evidence receipts.")
        return

    anchors = store.get_evidence_anchors()
    if not anchors:
        st.caption("No alert evidence has been generated yet.")
        return

    anchored_count = sum(1 for item in anchors if item["status"] == "anchored")
    c1, c2, c3 = st.columns(3)
    c1.metric("Receipts", len(anchors))
    c2.metric("On-chain", anchored_count)
    c3.metric("Pending/local", len(anchors) - anchored_count)

    for anchor in reversed(anchors[-8:]):
        color = ANCHOR_STATUS_COLORS.get(anchor["status"], "#64748b")
        tx_url = _tx_url(anchor.get("tx_hash"))
        location = anchor.get("location") or "—"
        alert_message = anchor.get("alert_message") or "—"
        with st.container(border=True):
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:4px;font-size:0.75rem;'>{anchor['status'].upper()}</span> "
                f"**Frame {anchor['frame_id']}** · **{location}**",
                unsafe_allow_html=True,
            )
            st.caption(alert_message)
            st.code(anchor["evidence_hash"], language=None)
            if tx_url:
                st.link_button("Open transaction", tx_url)
            elif anchor.get("message"):
                st.caption(anchor["message"])


def _run_pipeline(
    video_path: str,
    location: str,
    use_agent: bool,
    live_frame_slot,
    live_meta_slot,
    progress_bar,
    log_slot,
    alerts_slot,
) -> None:
    session_id = create_session_id()
    paths = reset_session(session_id)
    st.session_state.session_id = session_id
    st.session_state.log_lines = []
    st.session_state.alert_feed = []
    st.session_state.chat_history = []
    st.session_state.processing_complete = False
    st.session_state.current_frame = None

    store = EventStore(db_path=paths["events_db"])
    index = FrameIndex(chroma_dir=paths["chroma_dir"])
    rules = AlertRulesEngine()
    agent = SecurityAgent(store, index) if use_agent else None
    processor = VideoProcessor(store, index, rules, agent, location=location)

    for event in processor.iter_events(video_path):
        if event.kind == "progress":
            pct = event.extra.get("progress", 0.0)
            progress_bar.progress(
                pct,
                text=f"Processing frame {event.frame_id} (raw {event.raw_idx})…",
            )
            live_meta_slot.info(f"Scanning frame **{event.frame_id}** (raw {event.raw_idx})…")
            continue

        if event.kind in ("skip", "warmup"):
            if event.message not in st.session_state.log_lines:
                st.session_state.log_lines.append(event.message)
            if event.kind == "skip":
                live_meta_slot.caption(
                    f"Frame **{event.frame_id}** (raw {event.raw_idx}) — no motion, skipped"
                )
            _render_session_log(log_slot)
            continue

        if event.kind == "frame":
            extra = event.extra
            frame_data = {
                "frame_id": event.frame_id,
                "raw_idx": event.raw_idx,
                "image_rgb": extra["image_rgb"],
                "status": extra["status"],
                "logs": extra["logs"],
                "alerts": extra["alerts"],
            }
            st.session_state.current_frame = frame_data

            with live_frame_slot.container():
                _render_frame_panel(frame_data, title="Currently processing")

            for line in extra["logs"]:
                if line not in st.session_state.log_lines:
                    st.session_state.log_lines.append(line)

            for alert in extra["alerts"]:
                alert_record = {
                    "frame_id": event.frame_id,
                    "rule_id": alert["rule_id"],
                    "message": alert["message"],
                    "severity": alert["severity"],
                }
                st.session_state.alert_feed.insert(0, alert_record)
                st.toast(f"{alert['rule_id']}: {alert['message']}", icon="🚨")

            _render_session_log(log_slot)
            _render_alert_feed(alerts_slot)
            continue

        if event.kind == "error":
            st.error(event.message)
            store.close()
            return

        if event.kind == "done":
            st.session_state.stats = event.extra
            st.session_state.processing_complete = True
            st.session_state.store = store
            st.session_state.index = index
            progress_bar.progress(1.0, text="Complete")
            live_meta_slot.success(event.message)
            st.session_state.log_lines.append(event.message)
            continue

        # Legacy stream events (skip duplicate log lines if frame bundle already logged)
        if event.kind == "alert":
            continue

        line = event.message
        if event.kind == "detection":
            extra = event.extra
            line = (
                f"Frame {event.frame_id:03d} | {extra.get('objects')} | "
                f"{extra.get('activity')} | threat={extra.get('threat')} | {event.message[:70]}"
            )
        elif event.kind == "agent":
            line = f"    Agent: {event.message}"

        if line not in st.session_state.log_lines:
            st.session_state.log_lines.append(line)

    if not st.session_state.processing_complete:
        st.session_state.processing_complete = True
        st.session_state.store = store
        st.session_state.index = index


def main() -> None:
    st.set_page_config(
        page_title="Flying Police",
        page_icon="🛸",
        layout="wide",
    )
    _init_state()

    st.title("Flying Police")
    st.caption("Upload surveillance video · live frame analysis · alerts · Q&A")

    with st.sidebar:
        st.header("Settings")
        location = st.selectbox("Location", LOCATIONS, index=0)
        _render_monad_status()
        has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        use_agent = st.checkbox(
            "Enable LangChain agent",
            value=has_api_key,
            disabled=not has_api_key,
            help="Requires OPENAI_API_KEY in .env",
        )
        if not has_api_key:
            st.info("Agent disabled — add OPENAI_API_KEY to enable reasoning.")

    st.subheader("1. Upload video")
    uploaded = st.file_uploader(
        "Choose a video file (MP4, WebM, MOV, AVI)",
        type=["mp4", "webm", "mov", "avi"],
        help="Drag and drop a file here, or click Browse files",
    )
    if uploaded is not None:
        st.success(f"Ready: **{uploaded.name}** ({uploaded.size / 1024 / 1024:.1f} MB)")

    process_clicked = st.button(
        "2. Process video",
        type="primary",
        disabled=uploaded is None,
        use_container_width=False,
    )
    if uploaded is None:
        st.info("Upload a video above, then click **Process video**.")

    st.divider()
    st.subheader("Live analysis")

    progress_placeholder = st.empty()
    meta_placeholder = st.empty()
    live_frame_placeholder = st.empty()

    col_log, col_alerts = st.columns([3, 2])
    log_placeholder = col_log.empty()
    alerts_placeholder = col_alerts.empty()
    _render_session_log(log_placeholder)
    _render_alert_feed(alerts_placeholder)

    if st.session_state.current_frame and not process_clicked:
        st.divider()
        st.subheader("Last processed frame")
        _render_frame_panel(st.session_state.current_frame)

    if process_clicked and uploaded is not None:
        progress_placeholder.progress(0.0, text="Starting…")
        video_path = save_upload(uploaded.getvalue(), uploaded.name)
        _run_pipeline(
            video_path,
            location,
            use_agent,
            live_frame_placeholder,
            meta_placeholder,
            progress_placeholder,
            log_placeholder,
            alerts_placeholder,
        )
        st.success("Processing complete. Ask questions below.")
        st.rerun()

    if st.session_state.processing_complete and st.session_state.stats:
        stats = st.session_state.stats
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Frames analyzed", stats.get("processed_count", 0))
        m2.metric("Alerts", stats.get("alert_count", 0))
        m3.metric("Events logged", stats.get("event_count", 0))
        m4.metric("Indexed frames", stats.get("index_count", 0))
        m5.metric("Monad anchors", stats.get("evidence_anchor_count", 0))

        st.divider()
        _render_evidence_receipts(st.session_state.store)

        st.divider()
        st.subheader("Ask about this session")
        st.caption("Examples: how many men were detected? · any alerts? · person walking")

        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        if prompt := st.chat_input("Your question"):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            store = st.session_state.store
            index = st.session_state.index
            if store is None or index is None:
                answer = "No active session — process a video first."
            else:
                engine = QueryEngine(store, index)
                results = engine.query(prompt)
                answer = engine.format_results(results)

            st.session_state.chat_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(f"```\n{answer}\n```")


if __name__ == "__main__":
    main()
