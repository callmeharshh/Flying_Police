import numpy as np

from pipeline.frame_preview import prepare_frame_preview


def test_prepare_frame_preview_resizes_wide_frame():
    bgr = np.zeros((720, 1920, 3), dtype=np.uint8)
    rgb = prepare_frame_preview(bgr, max_width=960)
    assert rgb.shape[1] == 960
    assert rgb.shape[0] == 360


def test_prepare_frame_preview_draws_bbox():
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb = prepare_frame_preview(bgr, bbox=(10, 10, 30, 20))
    assert rgb[10, 10, 1] > 0
