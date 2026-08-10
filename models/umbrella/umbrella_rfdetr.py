"""
Umbrella detection via Roboflow RF-DETR Nano.

Key: umbrella_rfdetr
UI Label: RF-DETR Nano

RF-DETR is Roboflow's real-time DETR variant built on a DINOv2 backbone —
a genuinely different lineage from both the YOLO family and Baidu's
RT-DETR, and the reason it earns a slot here. DINOv2 features tend to hold
up better on small and partially occluded objects, which is exactly the
failure mode in dense umbrella crowds.

Runs through the official `rfdetr` package rather than ultralytics, so the
weights are the real RF-DETR ones (`rf-detr-nano.pth`, fetched to
`~/.roboflow/models/` on first use and MD5-verified by the library).

Weight resolution, in order:

  1. A fine-tuned umbrella checkpoint (`weights/rfdetr_nano_umbrella.pt`),
     reported as `finetuned`.
  2. Otherwise the stock RF-DETR Nano COCO checkpoint, filtered to COCO's
     `umbrella` class.

This previously loaded `rtdetr-l.pt` through ultralytics when no fine-tuned
checkpoint existed — a different architecture entirely, with no DINOv2
backbone, while still reporting itself as RF-DETR. It also had a second
silent fallback to `yolo11n.pt` if that failed, so a run could produce YOLO
results labelled RF-DETR with nothing in the logs to say so. Both are gone:
the real model loads, and a genuine failure raises.
"""

import os

from models.base import BaseModelWrapper, Detection
from models._tracker import IoUTracker
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas
from models.umbrella._common import UMBRELLA_COCO_CLASS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_FINETUNED_PATHS = [
    "weights/rfdetr_nano_umbrella.pt",
    "weights/rfdetr_nano_umbrella.pth",
    "rfdetr_nano_umbrella.pt",
]

# RF-DETR reports COCO ids against the **91-entry** list (the one including
# the N/A placeholders), where umbrella is 28 — not the 80-entry list YOLO
# uses, where it is 25. Verified by dumping class_id on a frame of
# Umbrellas.mp4: the umbrella detections come back as 28. Filtering on 25
# matched nothing at all and made the model look broken.
_COCO91_UMBRELLA = 28


class RFDETRNanoUmbrellaDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "umbrella_rfdetr"
    gpu_accelerated = True

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.35,
                 min_area_frac: float = DEFAULT_MIN_AREA_FRAC, track: bool = True,
                 iou_match_threshold: float = 0.3, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path or self._resolve_weights()
        self.conf_threshold = conf_threshold
        self.min_area_frac = min_area_frac
        self.track = track
        self._model = None
        self._class_index = UMBRELLA_COCO_CLASS
        self._is_finetuned = False
        self._actual_arch = "unknown"
        # RF-DETR returns detections without persistent IDs, so tracking is
        # supplied here — same IoU tracker the other umbrella models use.
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold, max_age=30)

    def _resolve_weights(self) -> str | None:
        for p in _FINETUNED_PATHS:
            abs_p = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
            if os.path.exists(abs_p):
                return abs_p
        return None

    def load(self):
        try:
            from rfdetr import RFDETRNano
        except ImportError as e:
            raise ImportError(
                "umbrella_rfdetr needs the RF-DETR package (`pip install "
                "rfdetr`). Refusing to substitute a different architecture - "
                "that is what made this wrapper report YOLO/RT-DETR results "
                "under an RF-DETR name."
            ) from e

        self._is_finetuned = bool(self.weights_path)
        if self._is_finetuned:
            self._model = RFDETRNano(pretrain_weights=self.weights_path)
            self._actual_arch = "RF-DETR Nano (umbrella fine-tuned)"
            # A single-class fine-tune puts umbrella at index 0.
            self._class_index = 0
        else:
            self._model = RFDETRNano()
            self._actual_arch = "RF-DETR Nano (COCO)"
            self._class_index = _COCO91_UMBRELLA

        self._tracker.reset()

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        if self._model is None:
            return []

        import cv2
        import numpy as np
        from PIL import Image

        # RF-DETR takes RGB; OpenCV hands us BGR.
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det = self._model.predict(pil, threshold=self.conf_threshold)

        xyxy = np.asarray(getattr(det, "xyxy", []))
        conf = np.asarray(getattr(det, "confidence", []))
        cls = np.asarray(getattr(det, "class_id", []))
        if xyxy.size == 0:
            return []

        keep = cls == self._class_index
        if not keep.any():
            return []

        boxes = xyxy[keep].tolist()
        scores = conf[keep].tolist() if conf.size else [1.0] * len(boxes)

        ids = self._tracker.update(boxes, frame_index) if self.track else None

        return emit_umbrellas(
            boxes, scores, ids, frame.shape, self.name, frame_index,
            timestamp_sec, min_area_frac=self.min_area_frac,
            extra_fields={
                "architecture": self._actual_arch,
                "backbone": "DINOv2",
                "finetuned": self._is_finetuned,
            },
        )
