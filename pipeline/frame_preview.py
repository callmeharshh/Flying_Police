"""Prepare video frames for live UI preview."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]
MAX_PREVIEW_WIDTH = 960
BBOX_COLOR_BGR = (0, 220, 80)
BBOX_THICKNESS = 2


def prepare_frame_preview(
    bgr_frame: np.ndarray,
    bbox: Optional[BBox] = None,
    max_width: int = MAX_PREVIEW_WIDTH,
) -> np.ndarray:
    display = bgr_frame.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), BBOX_COLOR_BGR, BBOX_THICKNESS)

    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    if width > max_width:
        scale = max_width / width
        rgb = cv2.resize(rgb, (max_width, int(height * scale)))
    return rgb
