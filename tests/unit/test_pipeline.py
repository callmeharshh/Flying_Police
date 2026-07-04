from pipeline.video_processor import PipelineEvent, infer_activity


def test_infer_activity_stationary_from_sitting():
    assert infer_activity(["person"], "a man is sitting on the floor") == "stationary"


def test_infer_activity_clear_when_no_objects():
    assert infer_activity([], "empty scene") == "clear"


def test_pipeline_event_defaults():
    event = PipelineEvent(kind="info", message="test")
    assert event.frame_id is None
    assert event.extra == {}
