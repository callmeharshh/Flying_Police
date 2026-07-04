import os


from query.track_merge import merge_tracks_into_individuals

# Frame centers from logs.txt — same man, fragmented tracks
LOGS_SESSION_FRAMES = [
    {"frame_id": 14, "location": "main_gate", "description": "a man is seen in this surveillance image",
     "center_x": 1168.0, "center_y": 502.5, "track_id": "track_001"},
    {"frame_id": 22, "location": "main_gate", "description": "a man is seen in a surveillance photo",
     "center_x": 226.0, "center_y": 303.0, "track_id": "track_002"},
    {"frame_id": 33, "location": "main_gate", "description": "a man is sitting on the floor in front of a door",
     "center_x": 676.5, "center_y": 562.0, "track_id": "track_003"},
    {"frame_id": 34, "location": "main_gate", "description": "a man in a hat is sitting on the floor",
     "center_x": 514.5, "center_y": 429.0, "track_id": "track_004"},
    {"frame_id": 35, "location": "main_gate", "description": "a man in a hat is seen in a surveillance video",
     "center_x": 507.5, "center_y": 360.0, "track_id": "track_005"},
    {"frame_id": 36, "location": "main_gate", "description": "a man wearing a hat",
     "center_x": 654.0, "center_y": 389.0, "track_id": "track_006"},
]


def test_logs_session_merges_to_one_individual():
    raw_tracks, individuals = merge_tracks_into_individuals(LOGS_SESSION_FRAMES)
    assert len(raw_tracks) == 6
    assert len(individuals) == 1
