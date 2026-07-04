import pytest

from storage.frame_index import FrameIndex


@pytest.fixture
def index(tmp_path):
    return FrameIndex(chroma_dir=str(tmp_path / "chroma"))


def test_list_all_frames_returns_sorted_entries(index):
    index.add_frame(
        10, "person at gate", "2026-06-10T10:00:00+00:00", "main_gate",
        ["person"], "low", bbox=(100, 200, 50, 80), track_id="track_001",
    )
    index.add_frame(
        5, "earlier frame", "2026-06-10T09:00:00+00:00", "main_gate",
        ["vehicle"], "low", bbox=(300, 200, 60, 40), track_id="track_002",
    )

    frames = index.list_all_frames()
    assert len(frames) == 2
    assert frames[0]["frame_id"] == 5
    assert frames[1]["frame_id"] == 10
    assert frames[1]["center_x"] == 125.0
    assert frames[1]["track_id"] == "track_001"
