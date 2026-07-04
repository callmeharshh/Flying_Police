"""Per-run session storage for UI and CLI."""
from __future__ import annotations

import os
import shutil
import uuid

from config import UI_SESSIONS_DIR, UI_UPLOADS_DIR

SESSIONS_DIR = UI_SESSIONS_DIR
UPLOADS_DIR = UI_UPLOADS_DIR


def create_session_id() -> str:
    return uuid.uuid4().hex[:12]


def session_paths(session_id: str) -> dict[str, str]:
    root = os.path.join(SESSIONS_DIR, session_id)
    return {
        "root": root,
        "events_db": os.path.join(root, "events.db"),
        "chroma_dir": os.path.join(root, "chroma"),
    }


def ensure_session_dirs(session_id: str) -> dict[str, str]:
    paths = session_paths(session_id)
    os.makedirs(paths["root"], exist_ok=True)
    return paths


def reset_session(session_id: str) -> dict[str, str]:
    paths = session_paths(session_id)
    if os.path.isdir(paths["root"]):
        shutil.rmtree(paths["root"])
    return ensure_session_dirs(session_id)


def save_upload(uploaded_bytes: bytes, filename: str) -> str:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    safe_name = os.path.basename(filename).replace(" ", "_")
    path = os.path.join(UPLOADS_DIR, f"{create_session_id()}_{safe_name}")
    with open(path, "wb") as handle:
        handle.write(uploaded_bytes)
    return path
