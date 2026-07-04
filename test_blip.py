"""
Quick test: run background subtraction + BLIP on the sample video.
Usage: python test_blip.py
"""
import cv2
import sys
import os
from PIL import Image as PILImage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vlm.background_subtractor import BackgroundSubtractor
from vlm.blip_analyzer import BLIPAnalyzer
from config import SAMPLE_VIDEO_DIR

SAMPLE_EVERY_N = 2  # process every 2nd frame


def run_with_bg_subtraction(video_path: str, analyzer: BLIPAnalyzer):
    print(f"\n--- Background Subtraction + BLIP on: {os.path.basename(video_path)} ---")
    subtractor = BackgroundSubtractor()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frames: {total} @ {fps:.1f}fps\n")

    frame_idx = 0
    analyzed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY_N != 0:
            frame_idx += 1
            continue

        crops = subtractor.apply(frame)

        if not crops:
            print(f"  Frame {frame_idx:03d} → no foreground")
        else:
            print(f"  Frame {frame_idx:03d} → {len(crops)} object(s) detected")
            result = analyzer.analyze_frame(crops)
            if result:
                print(f"    Objects    : {result.objects}")
                print(f"    Description: {result.raw_description}")
                for i, crop in enumerate(result.crops):
                    print(f"    Crop {i+1}     : [{crop.object_type}] \"{crop.caption}\" (color: {crop.color or 'n/a'})")
            analyzed += 1

        frame_idx += 1

    cap.release()
    print(f"\n  → {analyzed} frames with objects out of {frame_idx} processed")


def run_direct_blip(video_path: str, analyzer: BLIPAnalyzer, sample_frames: int = 5):
    """Run BLIP directly on full frames — bypasses background subtraction."""
    print(f"\n--- Direct BLIP (full frame, no subtraction) on: {os.path.basename(video_path)} ---")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_frames)

    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(frame_rgb)
        result = analyzer.analyze_full_image(pil_img)
        print(f"  Frame {i * step:03d}: [{result.object_type}] \"{result.caption}\" (color: {result.color or 'n/a'})")

    cap.release()


def run():
    analyzer = BLIPAnalyzer()

    for video_name in ["outside_entry_720p.mp4", "entrance_area_720p.mp4"]:
        video_path = os.path.join(SAMPLE_VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"Skipping {video_name} — not found")
            continue

        run_direct_blip(video_path, analyzer, sample_frames=6)
        run_with_bg_subtraction(video_path, analyzer)


if __name__ == "__main__":
    import sys
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blip_output.txt")
    with open(log_path, "w") as log_file:
        class Tee:
            def write(self, msg):
                sys.__stdout__.write(msg)
                log_file.write(msg)
            def flush(self):
                sys.__stdout__.flush()
                log_file.flush()
        sys.stdout = Tee()
        run()
        sys.stdout = sys.__stdout__
    print(f"\nOutput saved to: {log_path}")
