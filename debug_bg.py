"""Debug background subtraction — saves original + mask + bounding boxes for both videos."""
import cv2
import os
import numpy as np
from config import SAMPLE_VIDEO_DIR, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA_RATIO, MOG2_VAR_THRESHOLD

VIDEOS = ["outside_entry_720p.mp4", "entrance_area_720p.mp4"]


def debug_video(video_name: str):
    video_path = os.path.join(SAMPLE_VIDEO_DIR, video_name)
    out_dir = f"data/sample_images/debug_bg/{os.path.splitext(video_name)[0]}"
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=True)

    print(f"\n{video_name} — {total} frames @ {fps:.1f}fps")

    step = max(1, total // 12)
    saved = 0
    raw_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = mog2.apply(frame)

        if raw_idx % step != 0:
            raw_idx += 1
            continue

        _, fg_binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        vis = frame.copy()
        valid_boxes = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if area / frame_area > MAX_CONTOUR_AREA_RATIO:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)    # red = too large
            elif area < MIN_CONTOUR_AREA:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 100, 0), 1)  # blue = too small
            else:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)    # green = valid
                cv2.putText(vis, f"{int(area)}px", (x, max(y-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                valid_boxes += 1

        label = f"Frame {raw_idx:03d} | valid={valid_boxes} | total={len(contours)}"
        cv2.putText(vis, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        mask_bgr = cv2.cvtColor(fg_binary, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([
            cv2.resize(frame, (480, 270)),
            cv2.resize(mask_bgr, (480, 270)),
            cv2.resize(vis, (480, 270)),
        ])
        cv2.imwrite(f"{out_dir}/frame_{raw_idx:03d}.jpg", combined)
        print(f"  Frame {raw_idx:03d} → valid={valid_boxes}")
        saved += 1
        raw_idx += 1

    cap.release()
    print(f"  Saved {saved} frames → {out_dir}/")


for v in VIDEOS:
    debug_video(v)

print("\nLegend: green=valid object | blue=too small | red=too large (lighting artifact)")
