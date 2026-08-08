"""
Fall detection via YOLOv8-pose keypoints + geometric heuristic.

Extracts per-person keypoints with Ultralytics YOLOv8-pose, scores posture
with the shared torso-angle + bbox-aspect heuristic (models/fall/_geometry.py),
and confirms a fall only once the posture holds across several consecutive
frames on the same tracked person.

Fast, well-supported, and the easiest of the pose-based models to get
running — good baseline against which the other pose backbones are compared.

Model size matters a lot in crowd scenes: "n" (nano) is fast but loses
keypoint accuracy at distance/in dense crowds; "s" (small) is the default
accuracy/speed tradeoff; bump to "m"/"l"/"x" if GPU budget allows.

Three things this wrapper does that a naive angle threshold does not, each
of which was a real source of bad detections here:

  - Low-confidence keypoints are dropped rather than averaged in. YOLO-pose
    emits occluded joints as (0, 0, 0); averaging those pulls the torso line
    toward the image origin and fabricates falls out of occluded people.
  - The reported `confidence` describes the fall, not the person detector.
  - A fall must persist for `confirm_frames` frames on one track, which is
    what separates a fall from a crouch, a bend, or one bad pose estimate.
"""

from models.base import BaseModelWrapper, Detection
from models._weights import resolve as _resolve_weight_path
from models.fall._geometry import (
    DEFAULT_MIN_KP_CONF,
    angle_threshold_to_score,
    bbox_aspect_ratio,
    fall_confidence,
    posture_score,
    torso_angle_deg,
)
from models.fall._tracker import IoUTracker, sustained

_MODEL_SIZES = {
    "n": "yolov8n-pose.pt",
    "s": "yolov8s-pose.pt",
    "m": "yolov8m-pose.pt",
    "l": "yolov8l-pose.pt",
    "x": "yolov8x-pose.pt",
}


class YOLOPoseFallDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "fall_yolo_pose"

    def __init__(self, model_size: str = "s", conf_threshold: float = 0.4,
                 horizontal_angle_threshold_deg: float = 45.0,
                 min_kp_conf: float = DEFAULT_MIN_KP_CONF,
                 confirm_frames: int = 5, iou_match_threshold: float = 0.3,
                 device=None):
        super().__init__(device=device)
        if model_size not in _MODEL_SIZES:
            raise ValueError(f"model_size must be one of {list(_MODEL_SIZES)}")
        self.model_size = model_size
        self.weights_path = _MODEL_SIZES[model_size]
        self.conf_threshold = conf_threshold
        self.horizontal_angle_threshold_deg = horizontal_angle_threshold_deg
        self.score_threshold = angle_threshold_to_score(horizontal_angle_threshold_deg)
        self.min_kp_conf = min_kp_conf
        self.confirm_frames = confirm_frames
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold,
                                   history_len=max(confirm_frames * 2, 8))

    def load(self):
        from ultralytics import YOLO
        self._model = YOLO(_resolve_weight_path(self.weights_path))
        self._model.to(self.device)
        self._tracker.reset()

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )
        detections = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            boxes = [r.boxes[i].xyxy[0].tolist() for i in range(len(r.boxes))]
            track_ids = self._tracker.update(boxes, frame_index)

            for i, kp in enumerate(r.keypoints.data):
                if i >= len(boxes):
                    break
                kp_np = kp.cpu().numpy()
                # (V, 3) = x, y, score. Older exports omit the score column.
                kp_xy = kp_np[:, :2]
                kp_scores = kp_np[:, 2] if kp_np.shape[1] > 2 else None

                xyxy = boxes[i]
                det_conf = float(r.boxes[i].conf[0])
                tid = track_ids[i]

                angle = torso_angle_deg(kp_xy, kp_scores, min_kp_conf=self.min_kp_conf)
                score = posture_score(angle, bbox_aspect_ratio(xyxy))
                if score is None:
                    # No usable posture reading this frame (fully occluded
                    # torso). Record the gap so it breaks any run of
                    # positives rather than silently extending one.
                    self._tracker.append_state(tid, False)
                    continue

                frame_is_down = score >= self.score_threshold
                self._tracker.append_state(tid, frame_is_down)
                is_fall = sustained(self._tracker.states(tid), self.confirm_frames)

                detections.append(Detection(
                    model_name=self.name,
                    label="fall" if is_fall else "standing",
                    confidence=fall_confidence(score, self.score_threshold),
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    bbox=xyxy,
                    keypoints=kp_np.tolist(),
                    extra={
                        "torso_angle_deg": angle,
                        "posture_score": score,
                        "frame_is_down": frame_is_down,
                        "track_id": tid,
                        "detector_confidence": det_conf,
                    },
                ))
        return detections
