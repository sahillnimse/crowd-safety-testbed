"""
Umbrella detection via YOLO's COCO `umbrella` class.

**No new model download.** `umbrella` is COCO class 25, so the YOLO
checkpoints already in this repo detect it natively — `yolo11n.pt` is
5.6 MB and already on disk. Fine-tuning or sourcing a dedicated umbrella
model would add tens of megabytes to detect a class the existing weights
already cover.

Only the `n` and `s` sizes are offered because of the 30 MB budget:

    yolo11n   ~5.6 MB   fast, weaker on small/distant umbrellas
    yolo11s   ~19 MB    noticeably better recall, still inside budget
    yolo11m   ~40 MB    over budget, not offered

Detection is filtered to class 25 at the predictor, so the network never
spends time post-processing the other 79 classes.

Tracking (ByteTrack) is used rather than bare per-frame detection for the
same reason the traffic models use it: without persistent IDs you cannot
tell one umbrella seen for 100 frames from 100 separate umbrellas, so any
count is meaningless. `umbrellas_in_frame` is the live per-frame count and
`track_id` identifies the individual, so downstream code can count either
concurrent umbrellas or unique ones over the clip.

Note on what this is good for in a crowd-safety context: a rising umbrella
count is a proxy for rain, which changes crowd behaviour (people bunch
under shelter, walking speeds drop). Umbrellas also occlude overhead
cameras, so a high count is a useful confidence caveat on the pose-based
fall detectors, which degrade badly when torsos are hidden from above.
"""

from models.base import BaseModelWrapper, Detection
from models._weights import resolve as _resolve_weight_path
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas

# COCO class index for "umbrella". Verified against the bundled
# yolo11n/yolov8n checkpoints' own name maps rather than hardcoded blind.
UMBRELLA_COCO_CLASS = 25

# Capped at the 30 MB budget — yolo11m is ~40 MB and deliberately absent.
_MODEL_SIZES = {
    "n": "yolo11n.pt",
    "s": "yolo11s.pt",
}


class UmbrellaDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "umbrella_yolo"
    gpu_accelerated = True

    def __init__(self, model_size: str = "n", conf_threshold: float = 0.35,
                 min_area_frac: float = DEFAULT_MIN_AREA_FRAC, track: bool = True,
                 device=None):
        """
        conf_threshold: COCO umbrella AP is moderate, and the class is easily
            confused with parasols, awnings and tent canopies. 0.35 keeps
            recall usable; raise it if you see canopies being counted.
        min_area_frac: drop boxes smaller than this fraction of the frame.
            Filters the speck-sized false positives that appear in busy
            backgrounds without discarding genuinely distant umbrellas.
        track: keep persistent IDs across frames. Without them a single
            umbrella held for 100 frames is indistinguishable from 100
            umbrellas, so any unique count is nonsense.
        """
        super().__init__(device=device)
        if model_size not in _MODEL_SIZES:
            raise ValueError(
                f"model_size must be one of {list(_MODEL_SIZES)} "
                f"(larger sizes exceed the 30 MB model budget)"
            )
        self.model_size = model_size
        self.weights = _MODEL_SIZES[model_size]
        self.conf_threshold = conf_threshold
        self.min_area_frac = min_area_frac
        self.track = track
        self._class_index = UMBRELLA_COCO_CLASS

    def load(self):
        from ultralytics import YOLO
        self._model = YOLO(_resolve_weight_path(self.weights))
        self._model.to(self.device)

        # Resolve the class index from the checkpoint's own names rather than
        # trusting the constant: a custom or fine-tuned .pt dropped in here
        # would have a different mapping, and silently filtering on 25 would
        # then detect whatever class happened to sit at that index.
        names = getattr(self._model, "names", {}) or {}
        found = [i for i, n in names.items() if str(n).lower() == "umbrella"]
        if found:
            self._class_index = int(found[0])
        elif names:
            raise ValueError(
                f"{self.weights} has no 'umbrella' class (classes: "
                f"{sorted(set(names.values()))[:10]}...). Use a COCO-pretrained "
                f"checkpoint or a model fine-tuned with an 'umbrella' class."
            )

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
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

        # Filtering/counting/emission is shared with the other two umbrella
        # models (models/umbrella/_common.py) so a difference between them
        # reflects the detector, not the bookkeeping around it.
        return emit_umbrellas(
            r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids,
            frame.shape, self.name, frame_index, timestamp_sec,
            min_area_frac=self.min_area_frac,
            extra_fields={"model_size": self.model_size},
        )
