"""
Fire/Smoke detection via YOLO.

Expects a YOLO model checkpoint fine-tuned on a fire/smoke dataset
(e.g. classes: ["fire", "smoke"]). Standard Ultralytics YOLO — swap
`weights_path` to whichever fine-tuned .pt file you're testing.

If you don't have a fine-tuned checkpoint yet, public fire/smoke YOLO
weights exist on Roboflow Universe / HuggingFace — plug the path in below
once sourced. Architecture/wrapper code doesn't change.
"""

from models.base import BaseModelWrapper, Detection


class FireSmokeYOLO(BaseModelWrapper):
    consumption_type = "frame"
    name = "fire_smoke_yolo"

    def __init__(self, weights_path: str = "weights/fire_smoke_yolov8.pt",
                 conf_threshold: float = 0.4, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold

    def load(self):
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)
        self._model.to(self.device)

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )
        detections = []
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
