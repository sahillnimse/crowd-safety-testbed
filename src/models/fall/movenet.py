"""
Fall detection via Google MoveNet.

MoveNet (TF-Hub / TFLite) is optimized for speed — the "Lightning"
variant targets real-time inference on modest hardware, making it a
useful low-latency comparison point alongside MediaPipe. Like
MediaPipe's default pose solution, the single-person variant needs an
upstream person detector for crowd scenes; MoveNet also ships a
"MultiPose" variant (Lightning-multipose) that detects up to 6 people
directly without a separate detector — used here when `multipose=True`,
which is generally a better fit for this testbed's crowd scenes than
the single-person variant.

Shares the posture scoring, keypoint-confidence gating, tracking and
temporal confirmation in models/fall/_geometry.py and _tracker.py with the
other pose wrappers, so a difference in results here is a difference in the
*backbone* rather than in the heuristic bolted on top of it.

Note MoveNet's native input resolutions differ per variant (192 for
Lightning, 256 for Thunder and MultiPose) and it expects the image
letterboxed to a square rather than squashed — squashing distorts the
aspect ratio, which is exactly the signal the posture score reads.
"""

from models.base import BaseModelWrapper, Detection
from models.fall._geometry import (
    DEFAULT_MIN_KP_CONF,
    angle_threshold_to_score,
    bbox_aspect_ratio,
    fall_confidence,
    posture_score,
    torso_angle_deg,
)
from models._tracker import IoUTracker, sustained

# MoveNet emits COCO-17 keypoints, same layout as YOLO-pose; the shared
# geometry helpers default to those indices.

_INPUT_SIZES = {"lightning": 192, "thunder": 256, "multipose": 256}


class MoveNetFallDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "fall_movenet"
    gpu_accelerated = False  # CPU by default (TF-Hub MoveNet needs tensorflow-gpu configured separately to use CUDA)

    _MODEL_URLS = {
        "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        "multipose": "https://tfhub.dev/google/movenet/multipose/lightning/1",
    }

    def __init__(self, variant: str = "multipose", conf_threshold: float = 0.2,
                 horizontal_angle_threshold_deg: float = 45.0,
                 min_kp_conf: float = DEFAULT_MIN_KP_CONF,
                 confirm_frames: int = 5, iou_match_threshold: float = 0.3,
                 device=None):
        super().__init__(device=device)  # TF-based; device kept for interface consistency
        if variant not in self._MODEL_URLS:
            raise ValueError(f"variant must be one of {list(self._MODEL_URLS)}")
        self.variant = variant
        self.multipose = variant == "multipose"
        self.conf_threshold = conf_threshold
        self.horizontal_angle_threshold_deg = horizontal_angle_threshold_deg
        self.score_threshold = angle_threshold_to_score(horizontal_angle_threshold_deg)
        self.min_kp_conf = min_kp_conf
        self.confirm_frames = confirm_frames
        self.input_size = _INPUT_SIZES[variant]
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold,
                                   history_len=max(confirm_frames * 2, 8))

    def load(self):
        import tensorflow_hub as hub
        self._model = hub.load(self._MODEL_URLS[self.variant])
        self._infer = self._model.signatures["serving_default"]
        self._tracker.reset()

    def _letterbox(self, frame):
        """Pad to square, then resize — preserves aspect ratio.

        Returns the padded image plus the offsets/scale needed to map
        keypoints back into original frame coordinates.
        """
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        side = max(h, w)
        pad_y, pad_x = (side - h) // 2, (side - w) // 2
        canvas = np.zeros((side, side, 3), dtype=frame.dtype)
        canvas[pad_y:pad_y + h, pad_x:pad_x + w] = frame
        resized = cv2.resize(canvas, (self.input_size, self.input_size))
        return resized, pad_x, pad_y, side

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import cv2
        import numpy as np
        import tensorflow as tf

        padded, pad_x, pad_y, side = self._letterbox(frame)
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_tensor = tf.cast(tf.expand_dims(img_rgb, axis=0), dtype=tf.int32)

        outputs = self._infer(input_tensor)

        def to_frame_coords(kps_raw):
            """MoveNet returns (y, x, score) normalized to the padded square;
            undo the letterbox to get original-frame pixels."""
            ys = kps_raw[:, 0] * side - pad_y
            xs = kps_raw[:, 1] * side - pad_x
            return np.stack([xs, ys], axis=1), kps_raw[:, 2]

        people = []  # (keypoints_xy, keypoint_scores, person_conf)
        if self.multipose:
            # output shape: (1, 6, 56) -> 17 kps * 3 (y, x, score) + bbox(4) + conf(1)
            raw_people = outputs["output_0"].numpy()[0]
            for person in raw_people:
                person_conf = float(person[55])
                if person_conf < self.conf_threshold:
                    continue
                kps_xy, kp_scores = to_frame_coords(person[:51].reshape(17, 3))
                people.append((kps_xy, kp_scores, person_conf))
        else:
            kps_raw = outputs["output_0"].numpy()[0, 0]  # (17, 3): (y, x, score)
            person_conf = float(np.mean(kps_raw[:, 2]))
            if person_conf >= self.conf_threshold:
                kps_xy, kp_scores = to_frame_coords(kps_raw)
                people.append((kps_xy, kp_scores, person_conf))

        if not people:
            return []

        def kp_bbox(kps_xy, kp_scores):
            """Bbox over confident keypoints only — including unobserved
            joints (which sit at the padded origin) would stretch the box
            and corrupt the aspect-ratio posture cue."""
            mask = kp_scores >= self.min_kp_conf
            if not mask.any():
                return None
            xs, ys = kps_xy[mask, 0], kps_xy[mask, 1]
            return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

        boxes, valid = [], []
        for kps_xy, kp_scores, person_conf in people:
            box = kp_bbox(kps_xy, kp_scores)
            if box is None:
                continue
            boxes.append(box)
            valid.append((kps_xy, kp_scores, person_conf))

        if not boxes:
            return []

        track_ids = self._tracker.update(boxes, frame_index)
        detections = []

        for i, (kps_xy, kp_scores, person_conf) in enumerate(valid):
            bbox = boxes[i]
            tid = track_ids[i]

            angle = torso_angle_deg(kps_xy, kp_scores, min_kp_conf=self.min_kp_conf)
            score = posture_score(angle, bbox_aspect_ratio(bbox))
            if score is None:
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
                bbox=bbox,
                keypoints=np.concatenate(
                    [kps_xy, kp_scores[:, None]], axis=1).tolist(),
                extra={
                    "torso_angle_deg": angle,
                    "posture_score": score,
                    "frame_is_down": frame_is_down,
                    "track_id": tid,
                    "detector_confidence": person_conf,
                    "variant": self.variant,
                },
            ))
        return detections
