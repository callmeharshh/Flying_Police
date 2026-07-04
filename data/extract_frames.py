"""
Utility to extract frames from a real video file into sample_images/.
Usage: python data/extract_frames.py --video data/sample_video/outside_entry_720p.mp4 --every 1
"""
import argparse
import os
import cv2
from config import SAMPLE_IMAGES_DIR


def extract_frames(video_path: str, sample_every_n_seconds: float = 1.0, max_frames: int = 50) -> list[str]:
    """
    Extract frames from video at the given interval.
    Returns list of saved image paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps * sample_every_n_seconds))

    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

    saved_paths = []
    frame_idx = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(SAMPLE_IMAGES_DIR, f"{video_name}_frame{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_count} frames → {SAMPLE_IMAGES_DIR}")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--every", type=float, default=1.0, help="Sample every N seconds")
    parser.add_argument("--max", type=int, default=50, help="Max frames to extract")
    args = parser.parse_args()
    extract_frames(args.video, args.every, args.max)
