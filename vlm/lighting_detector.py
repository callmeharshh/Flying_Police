"""Infer day vs night from frame brightness (surveillance lighting, not wall clock)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    LIGHTING_CENTER_SAMPLE_RATIO,
    LIGHTING_DARK_PIXEL_VALUE,
    LIGHTING_NIGHT_DARK_PIXEL_RATIO,
    LIGHTING_NIGHT_MEAN_THRESHOLD,
)


@dataclass(frozen=True)
class LightingAnalysis:
    is_night: bool
    brightness: float
    dark_pixel_ratio: float
    label: str


def analyze_frame_lighting(frame_bgr: np.ndarray) -> LightingAnalysis:
    """
    Classify scene lighting from the frame image.

    Uses the V (value) channel in a central crop so edge overlays and
    letterboxing do not dominate the score.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return LightingAnalysis(
            is_night=False,
            brightness=255.0,
            dark_pixel_ratio=0.0,
            label="day",
        )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]

    height, width = value_channel.shape
    margin_x = int(width * (1 - LIGHTING_CENTER_SAMPLE_RATIO) / 2)
    margin_y = int(height * (1 - LIGHTING_CENTER_SAMPLE_RATIO) / 2)
    sample = value_channel[margin_y:height - margin_y, margin_x:width - margin_x]
    if sample.size == 0:
        sample = value_channel

    brightness = float(np.mean(sample))
    dark_pixel_ratio = float(np.mean(sample < LIGHTING_DARK_PIXEL_VALUE))

    is_night = (
        brightness < LIGHTING_NIGHT_MEAN_THRESHOLD
        or dark_pixel_ratio >= LIGHTING_NIGHT_DARK_PIXEL_RATIO
    )
    label = "night" if is_night else "day"

    return LightingAnalysis(
        is_night=is_night,
        brightness=round(brightness, 1),
        dark_pixel_ratio=round(dark_pixel_ratio, 3),
        label=label,
    )
