#!/usr/bin/env python3
"""
Capture validation fixtures from a video: frames, BLIP output, and agent responses.

Examples:
    python scripts/capture_validation_set.py
    python scripts/capture_validation_set.py --every 30 --max 15
    python scripts/capture_validation_set.py --skip-agent
    python scripts/capture_validation_set.py --video data/sample_video/outside_entry_720p.mp4
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DEMO_VIDEO, SAMPLE_VIDEO_DIR, VALIDATION_FIXTURES_DIR
from data.validation_capture import run_capture_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture validation fixtures from video")
    parser.add_argument(
        "--video",
        default=os.path.join(SAMPLE_VIDEO_DIR, DEMO_VIDEO),
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default=VALIDATION_FIXTURES_DIR,
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=None,
        help="Save every N sampled frames (default from config)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of captures (default from config)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=2,
        help="Process every Nth raw video frame",
    )
    parser.add_argument(
        "--capture-on",
        choices=("sampled", "detection"),
        default="detection",
        help="sampled: every Nth frame; detection: every Nth BLIP hit (default)",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Capture BLIP only; skip LangChain agent (no API key needed)",
    )
    args = parser.parse_args()

    kwargs = {
        "video_path": args.video,
        "output_dir": args.output,
        "skip_agent": args.skip_agent,
        "sample_every_n": args.sample_every,
        "capture_on": args.capture_on,
    }
    if args.every is not None:
        kwargs["every_n"] = args.every
    if args.max is not None:
        kwargs["max_captures"] = args.max

    run_capture_pipeline(**kwargs)


if __name__ == "__main__":
    main()
