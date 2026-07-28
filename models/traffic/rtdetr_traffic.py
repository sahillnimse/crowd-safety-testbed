"""
RT-DETR (Baidu, via ultralytics) vehicle detector.

Same interface and vehicle-class filtering as yolo_traffic.py, but RT-DETR
is a transformer-based detector — generally stronger on small/occluded
objects (distant vehicles, motorcycles partially hidden behind a bus) at
similar real-time speed. Used as a second-opinion comparison model, same
role X3D/SlowFast play relative to the primary violence classifier.

NOTE: ultralytics' RTDETR does not ship a built-in .track() the same way
YOLO does in older versions, but current ultralytics supports .track() for
any detection model including RTDETR — if track IDs come back None on your
installed version, fall back to feeding raw boxes through DeepSORT instead
(see deepsort_wrapper.py) rather than ByteTrack directly.
"""

from models.base import BaseModelWrapper, Detection
from models.traffic._tracker import ParkedMovingClassifier

# RT-DETR ships pretrained on COCO — same class indices as YOLO/COCO.
_VEHICLE_COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class RtdetrTrafficDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "rtdetr_traffic"
    gpu_accelerated = True

    def __init__(self, weights: str = "rtdetr-l.pt", conf_threshold: float = 0.35,
                 fps: float = 30.0, device=None):
        super().__init__(device=device)
        self.weights = weights
        self.conf_threshold = conf_threshold
        self._classifier = ParkedMovingClassifier(fps=fps)
        self._track_fallback = None  # lazily created only if .track() IDs come back None

    def load(self):
        from ultralytics import RTDETR
        self._model = RTDETR(self.weights)

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,
            classes=list(_VEHICLE_COCO_CLASSES.keys()),
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )

        r = results[0]
        if r.boxes is None:
            return []

        # RT-DETR + ByteTrack: if track IDs didn't come through, fall back
        # to DeepSORT so we still get persistent IDs for moving/parked logic.
        if r.boxes.id is None:
            return self._predict_with_deepsort_fallback(r, frame_index, timestamp_sec)

        raw_tracks = []
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

        return self._finalize(raw_tracks, frame_index, timestamp_sec)

    def _predict_with_deepsort_fallback(self, r, frame_index, timestamp_sec):
        if self._track_fallback is None:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._track_fallback = DeepSort(max_age=30)

        # DeepSort expects [([x,y,w,h], conf, class_name), ...]
        ds_input = []
        boxes_xyxy = r.boxes.xyxy.tolist()
        classes = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, classes, confs):
            w, h = x2 - x1, y2 - y1
            vehicle_class = _VEHICLE_COCO_CLASSES.get(int(cls_id), "vehicle")
            ds_input.append(([x1, y1, w, h], conf, vehicle_class))

        tracks = self._track_fallback.update_tracks(ds_input, frame=None)

        raw_tracks = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            raw_tracks.append({
                "track_id": int(t.track_id) if str(t.track_id).isdigit() else hash(t.track_id) % 100000,
                "bbox": [x1, y1, x2, y2],
                "vehicle_class": t.get_det_class() or "vehicle",
                "confidence": t.get_det_conf() or 0.5,
            })

        return self._finalize(raw_tracks, frame_index, timestamp_sec)

    def _finalize(self, raw_tracks, frame_index, timestamp_sec) -> list[Detection]:
        classified = self._classifier.update(frame_index, raw_tracks)
        detections = []
        for t in classified:
            detections.append(Detection(
                model_name=self.name,
                label=f"vehicle_{t['status']}",
                confidence=t["confidence"],
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=t["bbox"],
                extra={"vehicle_class": t["vehicle_class"], "track_id": t["track_id"]},
            ))
        return detections