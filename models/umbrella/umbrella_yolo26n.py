"""
Umbrella detection via Ultralytics YOLO26-Nano.

Key: umbrella_yolo26n
UI Label: YOLO26-Nano

YOLO26 is Ultralytics' newest nano detector: **NMS-free** (end-to-end, no
post-hoc suppression), with a distribution-focal-loss-free head aimed at
edge deployment. That makes it a genuinely different architecture from the
YOLO11 wrapper, not a size variant of it.

Weight resolution, in order:

  1. A fine-tuned umbrella checkpoint (`weights/yolo26n_umbrella.pt`) if one
     exists — that is the ideal, and it will be reported as `finetuned`.
  2. Otherwise **the real `yolo26n.pt` COCO checkpoint**, detecting COCO's
     `umbrella` class (index 25).

Point 2 matters. This wrapper used to fall back to `yolo11n.pt` when no
fine-tuned checkpoint was present, which made it a byte-for-byte duplicate
of `umbrella_yolo` — same weights, same boxes, same confidences — while
advertising itself as YOLO26. Using the actual YOLO26 COCO weights makes it
a real, independent data point in a comparison, even without fine-tuning.

Whatever ends up loading is reported per detection in
`extra["architecture"]`, so a log can always be traced to the network that
produced it.
"""

import os

from models.base import BaseModelWrapper, Detection
from models._weights import resolve as _resolve_weight_path
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas
from models.umbrella.umbrella_yolo import UMBRELLA_COCO_CLASS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# A fine-tuned umbrella checkpoint, if one is ever produced.
_FINETUNED_PATHS = [
    "weights/yolo26n_umbrella.pt",
    "weights/yolo26n_umbrella.onnx",
    "yolo26n_umbrella.pt",
]

# The real YOLO26-nano COCO checkpoint. Downloaded on first use by
# ultralytics if absent.
_COCO_WEIGHTS = "yolo26n.pt"


class YOLO26NanoUmbrellaDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "umbrella_yolo26n"
    gpu_accelerated = True

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.35,
                 min_area_frac: float = DEFAULT_MIN_AREA_FRAC, track: bool = True,
                 device=None):
        super().__init__(device=device)
        self.weights_path = weights_path or self._resolve_weights()
        self.conf_threshold = conf_threshold
        self.min_area_frac = min_area_frac
        self.track = track
        self._model = None
        self._class_index = UMBRELLA_COCO_CLASS
        self._is_finetuned = False
        self._actual_arch = "unknown"

    def _resolve_weights(self) -> str:
        for p in _FINETUNED_PATHS:
            abs_p = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
            if os.path.exists(abs_p):
                return abs_p
        return _COCO_WEIGHTS

    def load(self):
        from ultralytics import YOLO

        self._is_finetuned = (
            os.path.exists(self.weights_path)
            and os.path.basename(self.weights_path).startswith("yolo26n_umbrella")
        )
        target = self.weights_path if self._is_finetuned else _COCO_WEIGHTS

        self._model = YOLO(_resolve_weight_path(target))
        self._model.to(self.device)
        self._actual_arch = ("YOLO26-Nano (umbrella fine-tuned)"
                             if self._is_finetuned else "YOLO26-Nano (COCO)")

        # Class index comes from the checkpoint's own name map: a fine-tuned
        # single-class model puts umbrella at 0, COCO puts it at 25.
        names = getattr(self._model, "names", {}) or {}
        found = [i for i, n in names.items() if str(n).lower() == "umbrella"]
        if found:
            self._class_index = int(found[0])
        elif names:
            raise ValueError(
                f"{target} has no 'umbrella' class (classes: "
                f"{sorted(set(names.values()))[:8]}...)."
            )

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        if self._model is None:
            return []

        if self.track:
            results = self._model.track(
                frame, persist=True, classes=[self._class_index],
                conf=self.conf_threshold, tracker="bytetrack.yaml",
                verbose=False, device=self.device,
            )
        else:
            results = self._model.predict(
                frame, classes=[self._class_index], conf=self.conf_threshold,
                verbose=False, device=self.device,
            )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        ids = r.boxes.id.tolist() if getattr(r.boxes, "id", None) is not None else None

        return emit_umbrellas(
            r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids,
            frame.shape, self.name, frame_index, timestamp_sec,
            min_area_frac=self.min_area_frac,
            extra_fields={
                "architecture": self._actual_arch,
                "nms_free": True,          # true of YOLO26 either way
                "finetuned": self._is_finetuned,
            },
        )
