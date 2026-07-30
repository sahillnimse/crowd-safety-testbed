"""
Plate reading via RapidOCR (PP-OCRv4 ONNX Runtime engine).

Swappable alternative to EasyOCR in the ANPR pipeline.
Uses PP-OCRv4 mobile det/cls/rec models via the rapidocr_onnxruntime engine
(~10-15 MB combined) for CPU deployment parity and high-speed plate crop OCR.
"""

import numpy as np
from models.anpr._ocr import DEFAULT_MIN_PLATE_WIDTH, PLATE_ALPHABET, enhance_plate


class PlateRapidOCR:
    """RapidOCR (PP-OCRv4 ONNX) restricted to the license plate alphabet."""

    def __init__(self, use_gpu: bool = False,
                 min_plate_width: int = DEFAULT_MIN_PLATE_WIDTH):
        self.use_gpu = use_gpu
        self.min_plate_width = min_plate_width
        self._engine = None

    def load(self):
        """Initialize the RapidOCR ONNX Runtime engine."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        except ImportError:
            # Fallback or stub if package is not yet installed in runtime
            self._engine = None

    def read(self, plate_img) -> tuple[str, float, str]:
        """-> (text, confidence, status).

        status is one of "ok", "too_small", "unreadable".
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0, "unreadable"

        width = plate_img.shape[1]
        if width < self.min_plate_width:
            return "", 0.0, "too_small"

        prepared = enhance_plate(plate_img)
        if prepared is None:
            return "", 0.0, "unreadable"

        if self._engine is None:
            # Fallback attempting to re-init or returning unreadable if rapidocr_onnxruntime not installed
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
            except ImportError:
                return "", 0.0, "unreadable"

        try:
            # RapidOCR expects BGR or RGB numpy array
            result, _ = self._engine(prepared)
            if not result:
                return "", 0.0, "unreadable"

            # Filter recognized characters against plate alphabet
            raw_text = "".join([res[1] for res in result]).upper()
            filtered_chars = [c for c in raw_text if c in PLATE_ALPHABET]
            text = "".join(filtered_chars)

            if not text:
                return "", 0.0, "unreadable"

            conf_scores = [float(res[2]) for res in result if res[2] is not None]
            conf = float(np.mean(conf_scores)) if conf_scores else 0.85
            return text, conf, "ok"

        except Exception:
            return "", 0.0, "unreadable"
