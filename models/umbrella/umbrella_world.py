"""
Umbrella detection via YOLO-World (open-vocabulary).

The other two umbrella models are locked to COCO's single fixed `umbrella`
class. That class was trained mostly on handheld fabric umbrellas, and the
boundary shows: on a photo of thatched beach shelters on poles this repo's
YOLO wrapper found nothing at all — not a threshold problem, since nothing
was detected across all 80 classes even at confidence 0.05. Those simply
are not the object COCO calls an umbrella.

YOLO-World takes text prompts instead of a fixed class list, so the target
vocabulary is a parameter.

  weights   yolov8s-worldv2.pt, ~25.9 MB (inside the 30 MB budget)

**Measured behaviour on this repo's test images**, because the intuition
here is easy to get wrong:

  - On crowd photos it found 15 umbrellas where the fixed-class YOLO found
    6-8 and SSDLite found 1-4. Substantially higher recall.
  - On the thatched-shelter photo, the umbrella-family prompts found
    **nothing** — same as the fixed-class models. Adding "canopy",
    "beach hut" or "straw roof" also found nothing. Only the explicit
    prompt "thatched roof shelter" reached them, and then it found 35.

So open-vocabulary does not automatically generalise to "things that give
shade". It finds what you *name*, and near-miss synonyms fail. If the
footage contains shade structures you care about, prompt for them
literally; do not assume "umbrella" plus a loose term covers it.

Higher recall also means looser precision, so which prompt fired is
recorded per detection in `extra["matched_class"]` — a run can then be
audited for whether the extra detections came from genuine umbrellas or
from a broad term pulling in awnings and market stalls.
"""

from models.base import BaseModelWrapper, Detection
from models.fall._tracker import IoUTracker
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas

# Ordered widest-to-narrowest in meaning. "umbrella" first so it wins ties
# in the reported class name for the common case.
DEFAULT_PROMPTS = ("umbrella", "parasol", "beach umbrella", "sun umbrella")


class UmbrellaWorldDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "umbrella_world"
    gpu_accelerated = True

    def __init__(self, weights: str = "yolov8s-worldv2.pt",
                 prompts: tuple = DEFAULT_PROMPTS,
                 conf_threshold: float = 0.25,
                 min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
                 track: bool = True, iou_match_threshold: float = 0.3,
                 device=None):
        """
        prompts: text classes to detect. Add "canopy" or "awning" to catch
            fixed shade structures, at the cost of pulling in market stalls
            and shopfronts.
        conf_threshold: 0.25 rather than the YOLO wrapper's 0.35 — open-vocab
            scores are calibrated differently and run lower for the same
            visual evidence, so matching thresholds numerically would not
            match operating points.
        """
        super().__init__(device=device)
        self.weights = weights
        self.prompts = tuple(prompts)
        self.conf_threshold = conf_threshold
        self.min_area_frac = min_area_frac
        self.track = track
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold, max_age=30)

    def load(self):
        from ultralytics import YOLO

        self._model = YOLO(self.weights)
        if not hasattr(self._model, "set_classes"):
            raise ValueError(
                f"{self.weights} is not an open-vocabulary YOLO-World "
                f"checkpoint (no set_classes). Use a *-world*.pt checkpoint, "
                f"or run umbrella_yolo for the fixed-vocabulary detector."
            )
        # The prompt list becomes the model's entire class vocabulary, so
        # indices below are positions within self.prompts.
        self._model.set_classes(list(self.prompts))
        self._model.to(self.device)
        self._tracker.reset()

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self.conf_threshold, verbose=False, device=self.device,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.tolist()
        confs = r.boxes.conf.tolist()
        cls_ids = r.boxes.cls.tolist()

        ids = self._tracker.update(boxes, frame_index) if self.track else None

        detections = emit_umbrellas(
            boxes, confs, ids, frame.shape, self.name, frame_index, timestamp_sec,
            min_area_frac=self.min_area_frac,
            extra_fields={"prompts": list(self.prompts)},
        )

        # emit_umbrellas drops boxes below the area gate, so the class list
        # is re-walked in the same order to keep matched_class aligned with
        # the detections that actually survived.
        surviving = [i for i, b in enumerate(boxes)
                     if _survives(b, frame.shape, self.min_area_frac)]
        for det, src_i in zip(detections, surviving):
            idx = int(cls_ids[src_i])
            det.extra["matched_class"] = (
                self.prompts[idx] if 0 <= idx < len(self.prompts) else str(idx)
            )
        return detections


def _survives(box, frame_shape, min_area_frac: float) -> bool:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    area = (x2 - x1) * (y2 - y1)
    return area > 0 and area / float(h * w) >= min_area_frac
