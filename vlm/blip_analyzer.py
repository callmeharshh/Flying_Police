from dataclasses import dataclass
from typing import List, Optional
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import BLIP_MODEL
from vlm.background_subtractor import ObjectCrop
from vlm.constants import extract_color, extract_object_type


@dataclass
class CropAnalysis:
    caption: str
    object_type: str    # person | police_car | bicycle | animal | vehicle | unknown
    color: str          # dominant color keyword or ""
    bbox: tuple


@dataclass
class FrameVLMResult:
    crops: List[CropAnalysis]
    objects: List[str]          # deduplicated object types
    raw_description: str        # all captions joined


class BLIPAnalyzer:
    def __init__(self):
        print(f"Loading BLIP model: {BLIP_MODEL}")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        self._model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(self._device)
        self._model.eval()
        print(f"BLIP ready on {self._device}")

    def analyze_crop(self, crop: ObjectCrop) -> CropAnalysis:
        inputs = self._processor(images=crop.image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=50)
        caption = self._processor.decode(out[0], skip_special_tokens=True)

        return CropAnalysis(
            caption=caption,
            object_type=extract_object_type(caption),
            color=extract_color(caption),
            bbox=crop.bbox,
        )

    def analyze_frame(self, crops: List[ObjectCrop]) -> Optional[FrameVLMResult]:
        """Analyze all crops from a frame and merge into one result."""
        if not crops:
            return None

        analyses = [self.analyze_crop(c) for c in crops]
        objects = list({a.object_type for a in analyses if a.object_type != "unknown"})
        raw_description = "; ".join(a.caption for a in analyses)

        return FrameVLMResult(
            crops=analyses,
            objects=objects,
            raw_description=raw_description,
        )

    def analyze_full_image(self, image: Image.Image) -> CropAnalysis:
        """Analyze a full image directly (no background subtraction)."""
        dummy_crop = ObjectCrop(image=image, bbox=(0, 0, image.width, image.height), bbox_relative=(0, 0, 1, 1))
        return self.analyze_crop(dummy_crop)
