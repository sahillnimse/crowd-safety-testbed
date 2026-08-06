"""
Fire/Smoke detection via YOLO.

Expects a YOLO model checkpoint fine-tuned on a fire/smoke dataset
(e.g. classes: ["fire", "smoke"]). Standard Ultralytics YOLO — swap
`weights_path` to whichever fine-tuned .pt file you're testing.

If local weights are missing, it automatically falls back to Roboflow
hosted fire/smoke inference (model `smoke-fire-detection-fpxa0/1`), so it
works out-of-the-box without manual weights downloads.
"""

import os
from models.base import BaseModelWrapper, Detection


class FireSmokeYOLO(BaseModelWrapper):
    consumption_type = "frame"
    name = "fire_smoke_yolo"

    def __init__(self, weights_path: str = "weights/fire_smoke_yolov8.pt",
                 conf_threshold: float = 0.35, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self._uses_roboflow = False

    def load(self):
        if os.path.exists(self.weights_path):
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
            self._model.to(self.device)
            self._uses_roboflow = False
        else:
            # Fall back to Roboflow hosted inference for fire and smoke
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ROBOFLOW_API_KEY environment variable not set. "
                    "Please set ROBOFLOW_API_KEY to run FireSmokeYOLO via Roboflow API."
                )
            from inference_sdk import InferenceHTTPClient
            self._model = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            )
            self._uses_roboflow = True
            self._rf_model_id = "smoke-fire-detection-fpxa0/1"

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        detections = []
        if getattr(self, "_uses_roboflow", False):
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = self._model.infer(rgb_frame, model_id=self._rf_model_id)
                for pred in res.get("predictions", []):
                    conf = float(pred.get("confidence", 0.0))
                    if conf < self.conf_threshold:
                        continue
                    label = pred.get("class", "fire")
                    cx, cy = pred.get("x", 0), pred.get("y", 0)
                    w, h = pred.get("width", 0), pred.get("height", 0)
                    bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                    detections.append(Detection(
                        model_name=self.name,
                        label=label,
                        confidence=conf,
                        timestamp_sec=timestamp_sec,
                        frame_index=frame_index,
                        bbox=bbox,
                    ))
            except Exception:
                pass
        else:
            results = self._model.predict(
                frame, conf=self.conf_threshold, device=self.device, verbose=False
            )
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self._model.names[cls_id]  # "fire" / "smoke"
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    detections.append(Detection(
                        model_name=self.name,
                        label=label,
                        confidence=conf,
                        timestamp_sec=timestamp_sec,
                        frame_index=frame_index,
                        bbox=xyxy,
                    ))
        return detections
