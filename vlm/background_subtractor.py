from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from config import MIN_CONTOUR_AREA, MAX_CONTOUR_AREA_RATIO, MOG2_VAR_THRESHOLD, MOG2_WARMUP_FRAMES


@dataclass
class ObjectCrop:
    image: Image.Image          # PIL crop of the detected object
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in original frame
    bbox_relative: Tuple[float, float, float, float]  # normalized (0-1)


class BackgroundSubtractor:
    def __init__(self):
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=True,
        )
        self._warmed_up = False
        self._warmup_count = 0

    def warm_up(self, frames: List[np.ndarray]) -> None:
        for frame in frames:
            self._mog2.apply(frame)
        self._warmed_up = True

    def apply(self, frame: np.ndarray) -> List[ObjectCrop]:
        """
        Apply background subtraction to a frame.
        Returns list of ObjectCrop for each detected foreground object.
        Empty list means no new activity — caller should skip VLM.
        """
        # Auto warm-up on first few frames if warm_up() wasn't called explicitly
        if not self._warmed_up:
            self._mog2.apply(frame)
            self._warmup_count += 1
            if self._warmup_count >= MOG2_WARMUP_FRAMES:
                self._warmed_up = True
            return []

        fg_mask = self._mog2.apply(frame)

        # Remove shadows (grey pixels = 127) — keep only definite foreground (white = 255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological close to fill holes in blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_frame, w_frame = frame.shape[:2]
        crops = []

        frame_area = h_frame * w_frame
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            if area / frame_area > MAX_CONTOUR_AREA_RATIO:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Add padding around the crop
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_frame, x + w + pad)
            y2 = min(h_frame, y + h + pad)

            crop_bgr = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)

            bbox_rel = (x1 / w_frame, y1 / h_frame, x2 / w_frame, y2 / h_frame)

            crops.append(ObjectCrop(
                image=pil_image,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                bbox_relative=bbox_rel,
            ))

        return crops

    def reset(self) -> None:
        self.__init__()
