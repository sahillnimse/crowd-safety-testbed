"""
Roboflow-hosted combined violence/fall/non-violence detector.

Calls Roboflow's hosted inference API (no local GPU/training needed) using
a model trained specifically on violence/fall/non-violence classes — unlike
the Kinetics-pretrained clip classifiers elsewhere in this repo, which have
no violence-specific label and score zero-shot over unrelated action
classes (see webapp/registry.py's `_zero_shot()` notes).

Requires an API key: set the ROBOFLOW_API_KEY environment variable, or
pass api_key= directly. Get a key at https://roboflow.com (workspace
settings -> Private API Key).

Model used: "violence-ftjyp/1" on Roboflow Universe — classes include
fall, nonfall, "nonviolence person", violence.
https://universe.roboflow.com/yolo-ff7xm/violence-ftjyp
"""

import os

from models.base import BaseModelWrapper, Detection


class RoboflowCombinedDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "roboflow_combined"
    gpu_accelerated = False  # inference runs on Roboflow's servers, not locally

    def __init__(self, model_id: str = "violence-ftjyp/1",
                 api_key: str = None, conf_threshold: float = 0.4, device=None):
        super().__init__(device=device)
        self.model_id = model_id
        self.api_key = (
            api_key
            or os.environ.get("ROBOFLOW_API_KEY")
        )
        self.conf_threshold = conf_threshold

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
        # inference_sdk expects an image path OR a numpy array in RGB.
        # cv2 gives BGR — convert (this exact bug is called out in the
        # audit for the other violence classifiers; don't repeat it here).
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self._model.infer(rgb_frame, model_id=self.model_id)

        detections = []
        for pred in result.get("predictions", []):
            conf = float(pred.get("confidence", 0.0))
            if conf < self.conf_threshold:
                continue
            label = pred.get("class", "unknown")
            # Roboflow object-detection predictions are center x/y + width/height
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
        return detections