"""
Umbrella detection via SSDLite320 + MobileNetV3 (torchvision).

The architectural counterweight to the YOLO wrapper. Where `umbrella_yolo`
is a modern one-stage anchor-free detector, this is an SSD head on a
MobileNetV3 backbone — an older, lighter design built for mobile/CPU
inference. Including it answers a question the YOLO models alone cannot:
how much of the detection quality comes from the *architecture* versus the
training data, given both are COCO-trained on the same `umbrella` class.

  params      3.44 M
  weights     ~13.8 MB   (well inside the 30 MB budget)
  input       320x320, so it is cheap enough to be genuinely usable on CPU

**COCO indexing differs from YOLO's.** torchvision's COCO categories are
the 91-entry list including the `N/A` placeholders, so `umbrella` sits at
index **28**, not the 25 the YOLO 80-class list uses. Hardcoding one index
across both would silently detect the wrong class in one of them, so the
index is resolved from the weights' own category metadata.

SSD has no built-in tracker, so the shared IoU tracker from the fall models
supplies persistent IDs — without them a single umbrella held for 100
frames is indistinguishable from 100 umbrellas.
"""

from models.base import BaseModelWrapper, Detection
from models.fall._tracker import IoUTracker
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas


class UmbrellaSSDDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "umbrella_ssd"
    gpu_accelerated = True

    def __init__(self, conf_threshold: float = 0.30,
                 min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
                 track: bool = True, iou_match_threshold: float = 0.3,
                 device=None):
        """
        conf_threshold: lower than the YOLO wrapper's 0.35 on purpose —
            SSDLite is a weaker detector and its scores run lower, so the
            same threshold would not be the same operating point.
        """
        super().__init__(device=device)
        self.conf_threshold = conf_threshold
        self.min_area_frac = min_area_frac
        self.track = track
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold, max_age=30)
        self._class_index = None

    def load(self):
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        categories = weights.meta["categories"]

        # Resolved from the weights' own metadata rather than hardcoded:
        # torchvision's 91-entry COCO list puts umbrella at 28, while the
        # YOLO 80-entry list puts it at 25.
        matches = [i for i, c in enumerate(categories) if str(c).lower() == "umbrella"]
        if not matches:
            raise ValueError(
                "SSDLite COCO weights have no 'umbrella' category; "
                f"got {len(categories)} categories."
            )
        self._class_index = matches[0]

        self._model = ssdlite320_mobilenet_v3_large(weights=weights)
        self._model.to(self.device).eval()
        self._preprocess = weights.transforms()
        self._tracker.reset()

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import cv2
        import torch

        # torchvision detectors take RGB float tensors in [0, 1]; OpenCV
        # hands us BGR uint8.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        batch = [self._preprocess(tensor).to(self.device)]

        with torch.no_grad():
            output = self._model(batch)[0]

        keep = (output["labels"] == self._class_index) & (output["scores"] >= self.conf_threshold)
        boxes = output["boxes"][keep].cpu().tolist()
        scores = output["scores"][keep].cpu().tolist()
        if not boxes:
            return []

        ids = self._tracker.update(boxes, frame_index) if self.track else None

        return emit_umbrellas(
            boxes, scores, ids, frame.shape, self.name, frame_index, timestamp_sec,
            min_area_frac=self.min_area_frac,
            extra_fields={"backbone": "ssdlite320_mobilenet_v3_large"},
        )
