"""
RT-DETRv2 (Real-Time Detection Transformer v2) vehicle detector for traffic monitoring.

Key: rtdetrv2_traffic
UI Label: RT-DETRv2 Traffic (Moving / Parked)

Detects vehicles (car, motorcycle, bus, truck) using RT-DETRv2-S (Apache 2.0)
and classifies each tracked vehicle as MOVING or PARKED from centroid drift.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from models.base import BaseModelWrapper, Detection
from models.traffic._tracker import ParkedMovingClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

HF_MODEL_ID = "PekingU/rtdetr_v2_r18vd"

_LOCAL_CHECKPOINTS = [
    "weights/rtdetrv2_r18vd_coco.pth",
    "ML Models/ultralytics/rtdetrv2_r18vd_coco.pth",
]

_VEHICLE_COCO_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _find_local_checkpoint() -> Optional[str]:
    for rel in _LOCAL_CHECKPOINTS:
        abs_p = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(abs_p):
            return abs_p
    return None


class RTDetrV2TrafficDetector(BaseModelWrapper):
    """
    Traffic detector powered by RT-DETRv2-S + moving/parked status classifier.
    """

    consumption_type = "frame"
    name = "rtdetrv2_traffic"
    gpu_accelerated = True

    def __init__(
        self,
        conf_threshold: float = 0.35,
        parked_window_sec: float = 3.0,
        parked_radius_px: float = 15.0,
        local_checkpoint: Optional[str] = None,
        device=None,
    ):
        super().__init__(device=device)
        self.conf_threshold = conf_threshold
        self._local_checkpoint = local_checkpoint or _find_local_checkpoint()

        self._processor = None
        self._model = None
        self._classifier = ParkedMovingClassifier(
            parked_window_sec=parked_window_sec,
            parked_radius_px=parked_radius_px,
            model_name=self.name,
        )

        self._tracks: list[dict] = []
        self._next_track_id: int = 1

    def load(self):
        """Load RT-DETRv2 weights and initialize moving/parked classifier."""
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        if self._local_checkpoint and os.path.isfile(self._local_checkpoint):
            self._processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_ID)
            self._model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_ID)
            state = torch.load(self._local_checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self._model.load_state_dict(state, strict=False)
        else:
            self._processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_ID)
            self._model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_ID)

        self._model = self._model.to(self.device)
        self._model.eval()

        self._classifier.reset()
        self._tracks = []
        self._next_track_id = 1

    def predict(
        self, frame: np.ndarray, frame_index: int, timestamp_sec: float
    ) -> list[Detection]:
        if self._model is None:
            return []

        import torch
        from PIL import Image

        h, w = frame.shape[:2]

        pil_img = Image.fromarray(frame[:, :, ::-1])
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=self.conf_threshold, target_sizes=target_sizes
        )[0]

        boxes_all = results["boxes"].cpu().tolist()
        scores_all = results["scores"].cpu().tolist()
        labels_all = results["labels"].cpu().tolist()

        v_boxes, v_scores, v_classes = [], [], []
        for box, score, label in zip(boxes_all, scores_all, labels_all):
            if label in _VEHICLE_COCO_CLASSES:
                v_boxes.append(box)
                v_scores.append(score)
                v_classes.append(_VEHICLE_COCO_CLASSES[label])

        if not v_boxes:
            return []

        track_ids = self._assign_track_ids(v_boxes, frame.shape)

        raw_tracks = []
        for box, conf, v_cls, tid in zip(v_boxes, v_scores, v_classes, track_ids):
            raw_tracks.append(
                {
                    "track_id": tid,
                    "bbox": box,
                    "vehicle_class": v_cls,
                    "confidence": float(conf),
                }
            )

        classified = self._classifier.update(timestamp_sec, raw_tracks)
        detections = []
        for t in classified:
            detections.append(
                Detection(
                    model_name=self.name,
                    label=f"vehicle_{t['status']}",
                    confidence=t["confidence"],
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    bbox=t["bbox"],
                    extra={
                        "vehicle_class": t["vehicle_class"],
                        "track_id": t["track_id"],
                        "architecture": "RT-DETRv2-S",
                    },
                )
            )

        return detections

    def _assign_track_ids(self, boxes: list, frame_shape: tuple) -> list[int]:
        h, w = frame_shape[:2]
        diag = (h ** 2 + w ** 2) ** 0.5
        match_dist = diag * 0.10
        max_age = 30

        centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        n_det = len(centroids)
        n_trk = len(self._tracks)

        assigned: list[Optional[int]] = [None] * n_det
        used_trk: set[int] = set()

        if n_trk > 0 and n_det > 0:
            dists = np.full((n_trk, n_det), fill_value=1e9, dtype=np.float32)
            for ti, t in enumerate(self._tracks):
                for di, (cx, cy) in enumerate(centroids):
                    dx, dy = cx - t["cx"], cy - t["cy"]
                    dists[ti, di] = (dx * dx + dy * dy) ** 0.5

            flat_order = np.argsort(dists.ravel())
            for idx in flat_order:
                ti, di = divmod(int(idx), n_det)
                if dists[ti, di] > match_dist:
                    break
                if ti in used_trk or assigned[di] is not None:
                    continue
                assigned[di] = self._tracks[ti]["id"]
                self._tracks[ti]["cx"] = centroids[di][0]
                self._tracks[ti]["cy"] = centroids[di][1]
                self._tracks[ti]["age"] = 0
                used_trk.add(ti)

        for di in range(n_det):
            if assigned[di] is None:
                cx, cy = centroids[di]
                tid = self._next_track_id
                assigned[di] = tid
                self._tracks.append({"id": tid, "cx": cx, "cy": cy, "age": 0})
                self._next_track_id += 1

        for ti in range(n_trk - 1, -1, -1):
            if ti not in used_trk:
                self._tracks[ti]["age"] += 1
                if self._tracks[ti]["age"] > max_age:
                    self._tracks.pop(ti)

        return assigned
