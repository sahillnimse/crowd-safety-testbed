"""
Roboflow-hosted vehicle detector, following the exact same pattern as
models/roboflow_combined.py (fall/violence) — hosted inference API, no
local GPU/training needed, model pretrained specifically on traffic-camera
footage rather than general COCO images.

Requires the same ROBOFLOW_API_KEY as roboflow_combined.py (reuses the
same key/account — no separate signup needed).

Model used: a Roboflow Universe vehicle-detection model trained on
CCTV/traffic-cam angles — replace model_id below with whichever specific
Universe model you pick (search "vehicle detection" or "traffic counting"
on universe.roboflow.com; many are trained on exactly this top-down/
roadside camera angle, unlike general COCO-pretrained YOLO).
"""

import os

from models.base import BaseModelWrapper, Detection
from models.traffic._tracker import ParkedMovingClassifier


class RoboflowTrafficDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "roboflow_traffic"
    gpu_accelerated = False  # inference runs on Roboflow's servers, not locally

    def __init__(self, model_id: str = "vehicle-detection-3mmwj/1",
                 api_key: str = None, conf_threshold: float = 0.4,
                 fps: float = 30.0, device=None):
        super().__init__(device=device)
        self.model_id = model_id
        # Same fallback pattern as roboflow_combined.py — reuses the same
        # hardcoded key so this works without any extra env-var setup.
        self.api_key = (
            api_key
            or os.environ.get("ROBOFLOW_API_KEY")
            or "c9KEmh1NFvhY8WFH9Iq5"
        )
        self.conf_threshold = conf_threshold
        self._classifier = ParkedMovingClassifier(fps=fps)
        self._track_fallback = None  # DeepSORT — Roboflow gives boxes only, no track IDs

    def load(self):
        if not self.api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY not set. Get a key at roboflow.com "
                "(workspace settings -> Private API Key) and either set the "
                "ROBOFLOW_API_KEY environment variable or pass api_key=."
            )
        from inference_sdk import InferenceHTTPClient
        self._model = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.api_key,
        )

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import cv2
        # Same BGR->RGB fix called out in roboflow_combined.py — don't repeat
        # that bug here.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self._model.infer(rgb_frame, model_id=self.model_id)

        # Roboflow object-detection gives boxes but no persistent track ID —
        # feed through DeepSORT (same fallback used in rtdetr_traffic.py)
        # so moving/parked classification has IDs to work with.
        if self._track_fallback is None:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._track_fallback = DeepSort(max_age=30)

        ds_input = []
        for pred in result.get("predictions", []):
            conf = float(pred.get("confidence", 0.0))
            if conf < self.conf_threshold:
                continue
            label = pred.get("class", "vehicle")
            cx, cy = pred.get("x", 0), pred.get("y", 0)
            w, h = pred.get("width", 0), pred.get("height", 0)
            x1, y1 = cx - w / 2, cy - h / 2
            ds_input.append(([x1, y1, w, h], conf, label))

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

        classified = self._classifier.update(frame_index, raw_tracks)

        detections = []
        for tr in classified:
            detections.append(Detection(
                model_name=self.name,
                label=f"vehicle_{tr['status']}",
                confidence=tr["confidence"],
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=tr["bbox"],
                extra={"vehicle_class": tr["vehicle_class"], "track_id": tr["track_id"]},
            ))
        return detections