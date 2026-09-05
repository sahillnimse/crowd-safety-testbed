"""
Plate reading: detection of the plate region, image conditioning, and OCR.

Two components, both deliberately swappable:

  - **Plate detector** — a DETR fine-tuned on licence plates, run on the
    *vehicle crop* rather than the whole frame. Searching inside a known
    vehicle box is both faster and far more precise than scanning the full
    image, and it gives the plate an owner without any extra association
    step.
  - **OCR** — EasyOCR restricted to the plate alphabet. Constraining the
    character set removes a whole class of errors immediately: without an
    allowlist, OCR happily returns punctuation and lowercase that no plate
    contains.

**The size gate is the important part of this file.** ANPR only works when
the plate is physically large enough in the image to carry the characters.
Measured on this repo's traffic clip, plates top out at 60x18 px and OCR
returns nothing at all from them — the information is simply not in the
pixels. Rather than emit blanks or noise that look like model failure, a
plate below `min_plate_width` is reported as `too_small`, with its measured
width, so it is obvious that the footage is the limit and not the code.
"""

import cv2
import numpy as np

# Below this the characters are too few pixels tall to survive OCR.
# Synthetic clean plates in this repo's tests started decoding at ~60 px
# wide; real plates are blurred, angled and low-contrast, so the practical
# floor is well above that. 90 px is a deliberately permissive default —
# it lets marginal reads through to the voting stage rather than discarding
# them, since voting can recover from individual bad frames.
DEFAULT_MIN_PLATE_WIDTH = 90

PLATE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# DETR fine-tuned for plates. Chosen over the YOLO-format alternatives
# because it ships safetensors — transformers 5.x refuses to load the .bin
# checkpoints the others publish when torch < 2.6.
DEFAULT_PLATE_MODEL = "nickmuchi/detr-resnet50-license-plate-detection"


class PlateDetector:
    """Locates plate regions inside a vehicle crop."""

    def __init__(self, model_id: str = DEFAULT_PLATE_MODEL,
                 conf_threshold: float = 0.5, device: str = "cpu"):
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.device = device
        self._proc = None
        self._model = None

    def load(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForObjectDetection.from_pretrained(self.model_id)
        self._model.to(self.device).eval()

    def detect(self, crop) -> list[tuple]:
        """-> [(x1, y1, x2, y2, score), ...] in crop-local coordinates."""
        import torch

        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 16:
            return []

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = self._proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target = torch.tensor([rgb.shape[:2]]).to(self.device)
        res = self._proc.post_process_object_detection(
            outputs, threshold=self.conf_threshold, target_sizes=target)[0]

        out = []
        for score, box in zip(res["scores"].tolist(), res["boxes"].tolist(), strict=False):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            out.append((x1, y1, x2, y2, float(score)))
        # Largest first: on a vehicle crop the biggest plate-like region is
        # nearly always the actual plate.
        out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return out


def enhance_plate(plate_img, target_width: int = 320):
    """Condition a plate crop for OCR.

    Upscaling matters more than anything else here: EasyOCR's recognition
    network expects characters far taller than a distant plate provides, so
    a cubic upscale to a fixed working width recovers a surprising amount.
    CLAHE then pulls the characters out of the plate background, which is
    often washed out by headlights or sun.
    """
    if plate_img is None or plate_img.size == 0:
        return None

    h, w = plate_img.shape[:2]
    if w < 8 or h < 4:
        return None

    scale = max(1.0, target_width / w)
    up = cv2.resize(plate_img, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    # Light denoise; plates at distance are grainy and OCR reads grain as
    # punctuation.
    return cv2.bilateralFilter(gray, 5, 50, 50)


class PlateOCR:
    """EasyOCR restricted to the plate alphabet."""

    def __init__(self, use_gpu: bool = True,
                 min_plate_width: int = DEFAULT_MIN_PLATE_WIDTH):
        self.use_gpu = use_gpu
        self.min_plate_width = min_plate_width
        self._reader = None

    def load(self):
        import easyocr
        self._reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)

    def read(self, plate_img) -> tuple[str, float, str]:
        """-> (text, confidence, status).

        status is one of "ok", "too_small", "unreadable" — carried through
        to the output so a run can distinguish "no plate was legible in this
        footage" from "the model found nothing", which look identical if
        you only ever report the text.
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0, "unreadable"

        width = plate_img.shape[1]
        if width < self.min_plate_width:
            return "", 0.0, "too_small"

        prepared = enhance_plate(plate_img)
        if prepared is None:
            return "", 0.0, "unreadable"

        results = self._reader.readtext(prepared, allowlist=PLATE_ALPHABET, detail=1)
        if not results:
            return "", 0.0, "unreadable"

        # A plate may be split across two boxes (state code above, number
        # below on square plates), so join in reading order.
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        text = "".join(r[1] for r in results)
        conf = float(np.mean([r[2] for r in results]))
        return text, conf, "ok"


def get_ocr_engine(backend: str = "easyocr", use_gpu: bool = True,
                   min_plate_width: int = DEFAULT_MIN_PLATE_WIDTH):
    """Factory for swappable ANPR OCR engines ('easyocr' or 'rapidocr')."""
    backend_norm = str(backend).lower().replace("_", "").replace("-", "")
    if "rapid" in backend_norm:
        from models.anpr._rapid_ocr import PlateRapidOCR
        return PlateRapidOCR(use_gpu=use_gpu, min_plate_width=min_plate_width)
    return PlateOCR(use_gpu=use_gpu, min_plate_width=min_plate_width)


def dominant_colour(crop) -> str:
    """Coarse colour name for a vehicle crop.

    Deliberately coarse — "silver" vs "grey" is not reliably separable on
    CCTV, and a confident-sounding wrong colour is worse than a vague right
    one. Sampled from the centre of the crop to avoid road and background
    bleeding in at the edges.
    """
    if crop is None or crop.size == 0:
        return "unknown"

    h, w = crop.shape[:2]
    core = crop[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
    if core.size == 0:
        core = crop

    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[..., 0])) * 2      # OpenCV hue is 0-179
    sat = float(np.median(hsv[..., 1])) / 255.0
    val = float(np.median(hsv[..., 2])) / 255.0

    if val < 0.20:
        return "black"
    if sat < 0.18:
        return "white" if val > 0.65 else "silver/grey"
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange/brown"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 260:
        return "blue"
    return "red" if hue > 320 else "purple"
