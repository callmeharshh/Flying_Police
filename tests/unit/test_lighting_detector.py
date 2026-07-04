import numpy as np
import pytest

from vlm.lighting_detector import analyze_frame_lighting


def _solid_frame(value: int, size: int = 480) -> np.ndarray:
    gray = np.full((size, size), value, dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def test_bright_frame_classified_as_day():
    result = analyze_frame_lighting(_solid_frame(220))
    assert result.is_night is False
    assert result.label == "day"
    assert result.brightness >= 200


def test_dark_frame_classified_as_night():
    result = analyze_frame_lighting(_solid_frame(25))
    assert result.is_night is True
    assert result.label == "night"
    assert result.dark_pixel_ratio >= 0.9


def test_empty_frame_defaults_to_day():
    result = analyze_frame_lighting(np.array([]))
    assert result.is_night is False
    assert result.label == "day"
