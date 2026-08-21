"""
RapidOCR (PP-OCRv4 ONNX) standalone ANPR pipeline wrapper.

Key: rapid_ocr
UI Label: RapidOCR (PP-OCRv4 ONNX)

Evaluates the ONNX Runtime RapidOCR engine (PP-OCRv4 mobile det/cls/rec) as a
swappable OCR backend alternative to EasyOCR for ANPR plate crops.
"""

from models.anpr.anpr import ANPRDetector
from models.anpr._ocr import DEFAULT_MIN_PLATE_WIDTH


class RapidOCRDetector(ANPRDetector):
    name = "rapid_ocr"

    def __init__(self, conf_threshold: float = 0.35,
                 plate_conf: float = 0.5,
                 min_plate_width: int = DEFAULT_MIN_PLATE_WIDTH,
                 read_every_n_frames: int = 3,
                 gallery_dir: str = None, video_name: str = "run",
                 save_gallery: bool = True, device=None):
        super().__init__(
            conf_threshold=conf_threshold, plate_conf=plate_conf,
            min_plate_width=min_plate_width, read_every_n_frames=read_every_n_frames,
            gallery_dir=gallery_dir, video_name=video_name, save_gallery=save_gallery,
            ocr_backend="rapidocr", device=device,
        )
