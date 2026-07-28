"""
YOLOv8/v11 vehicle detector + built-in ByteTrack, feeding into the shared
ParkedMovingClassifier (see _tracker.py) for moving/parked classification.

Uses ultralytics' .track() (not .predict()) so ByteTrack ID assignment is
handled by the library itself — same COCO classes as fire_smoke_yolo.py's
YOLO base, but filtered to vehicle classes only:
  2: car, 3: motorcycle, 5: bus, 7: truck   (COCO class indices)
"""

from models.base import BaseModelWrapper, Detection
from models.traffic._tracker import ParkedMovingClassifier

_VEHICLE_COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloTrafficDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "yolo_traffic"
    gpu_accelerated = True

    def __init__(self, weights: str = "yolo11n.pt", conf_threshold: float = 0.35,
                 parked_window_sec: float = 3.0, parked_radius_px: float = 15.0,
                 device=None):
        super().__init__(device=device)
        self.weights = weights
        self.conf_threshold = conf_threshold
        # No fps argument: the classifier works off the runner's
        # timestamp_sec, so it needs no assumption about frame rate or how
        # densely the video is being sampled.
        self._classifier = ParkedMovingClassifier(
            parked_window_sec=parked_window_sec,
            parked_radius_px=parked_radius_px,
            model_name=self.name,
        )

    def load(self):
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        self._model.to(self.device)
        self._classifier.reset()

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,           # keep track IDs across calls
            classes=list(_VEHICLE_COCO_CLASSES.keys()),
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )

        raw_tracks = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return []  # no tracks yet (first frame, or nothing detected)

        for box, track_id, cls_id, conf in zip(
            r.boxes.xyxy.tolist(),
            r.boxes.id.tolist(),
            r.boxes.cls.tolist(),
            r.boxes.conf.tolist(),
        ):
            vehicle_class = _VEHICLE_COCO_CLASSES.get(int(cls_id), "vehicle")
            raw_tracks.append({
                "track_id": int(track_id),
                "bbox": box,
                "vehicle_class": vehicle_class,
                "confidence": float(conf),
            })

        classified = self._classifier.update(timestamp_sec, raw_tracks)

        detections = []
        for t in classified:
            label = f"vehicle_{t['status']}"  # "vehicle_moving" or "vehicle_parked"
            detections.append(Detection(
                model_name=self.name,
                label=label,
                confidence=t["confidence"],
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=t["bbox"],
                extra={"vehicle_class": t["vehicle_class"], "track_id": t["track_id"]},
            ))
        return detections