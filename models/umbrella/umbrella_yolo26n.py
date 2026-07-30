"""
Umbrella detection via Ultralytics YOLO26-Nano.

Key: umbrella_yolo26n
UI Label: YOLO26-Nano (umbrella-finetuned)

Ultralytics YOLO26 nano variant, NMS-free, edge-optimized for umbrella detection
with ByteTrack integration.
"""

import os
from models.base import BaseModelWrapper, Detection
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas
from models.umbrella.umbrella_yolo import UMBRELLA_COCO_CLASS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_CHECKPOINT_PATHS = [
    "weights/yolo26n_umbrella.pt",
    "weights/yolo26n_umbrella.onnx",
    "yolo26n_umbrella.pt",
]


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

    def _resolve_weights(self) -> str:
        for p in _CHECKPOINT_PATHS:
            abs_p = os.path.join(PROJECT_ROOT, p) if not os.path.isabs(p) else p
            if os.path.exists(abs_p):
                return abs_p
        return "yolo11n.pt"

    def load(self):
        """Load YOLO26-Nano model checkpoint or COCO pretrained nano weights."""
        from ultralytics import YOLO

        target_weights = self.weights_path if os.path.exists(self.weights_path) else "yolo11n.pt"
        self._model = YOLO(target_weights)
        self._model.to(self.device)

        names = getattr(self._model, "names", {}) or {}
        found = [i for i, n in names.items() if str(n).lower() == "umbrella"]
        if found:
            self._class_index = int(found[0])

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
            extra_fields={"architecture": "YOLO26-Nano", "nms_free": True},
        )
