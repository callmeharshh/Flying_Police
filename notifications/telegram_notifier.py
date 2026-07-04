"""Telegram alert notifications via Bot API."""
import logging
import threading
from typing import Optional

import cv2
import numpy as np
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

_SEVERITY_EMOJI = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢",
}

_SEVERITY_COLOR_BGR = {
    "high": (0, 0, 220),
    "medium": (0, 165, 255),
    "low": (0, 200, 0),
}

_API_TIMEOUT = 10  # seconds


def _build_caption(
    rule_id: str,
    message: str,
    severity: str,
    frame_id: int,
    timestamp: str,
    location: str,
) -> str:
    emoji = _SEVERITY_EMOJI.get(severity.lower(), "⚠️")
    ts = timestamp[:19].replace("T", " ")
    return (
        f"{emoji} Security Alert [{rule_id}]\n"
        f"Severity: {severity.upper()}\n"
        f"Frame: {frame_id:03d} | {ts}\n"
        f"Location: {location}\n"
        f"{message}"
    )


def _annotate_frame(
    frame: np.ndarray,
    rule_id: str,
    severity: str,
    bbox: Optional[tuple],
) -> bytes:
    """Draw bbox + label on frame, return JPEG bytes."""
    img = frame.copy()
    color = _SEVERITY_COLOR_BGR.get(severity.lower(), (0, 0, 220))

    if bbox is not None:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        label = f"{rule_id} {severity.upper()}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(img, label, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _send_photo(caption: str, image_bytes: bytes) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        resp = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": ("alert.jpg", image_bytes, "image/jpeg")},
            timeout=_API_TIMEOUT,
        )
        if not resp.ok:
            logger.warning("Telegram photo error %s: %s", resp.status_code, resp.text[:120])
    except requests.RequestException as exc:
        logger.warning("Telegram send failed: %s", exc)


def _send_text(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=_API_TIMEOUT,
        )
        if not resp.ok:
            logger.warning("Telegram API error %s: %s", resp.status_code, resp.text[:120])
    except requests.RequestException as exc:
        logger.warning("Telegram send failed: %s", exc)


def send_alert(
    rule_id: str,
    message: str,
    severity: str,
    frame_id: int,
    timestamp: str,
    location: str,
    frame: Optional[np.ndarray] = None,
    bbox: Optional[tuple] = None,
) -> None:
    """Fire-and-forget Telegram notification. Sends photo if frame provided, text otherwise."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if severity.lower() != "high":
        return

    caption = _build_caption(rule_id, message, severity, frame_id, timestamp, location)

    if frame is not None:
        image_bytes = _annotate_frame(frame, rule_id, severity, bbox)
        threading.Thread(target=_send_photo, args=(caption, image_bytes), daemon=True).start()
    else:
        threading.Thread(target=_send_text, args=(caption,), daemon=True).start()
