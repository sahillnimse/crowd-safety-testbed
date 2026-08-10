"""
Fall detection via MediaPipe BlazePose.

Google's lightweight single-person pose estimator. Cheap enough to run
comfortably on CPU/edge devices and a useful "lightweight baseline"
comparison point — but it's fundamentally single-person per detector
instance, so multi-person/crowd frames need an upstream person detector +
crop-and-run-per-person, which adds latency and compounds detection error.
Expect this to degrade faster than MoveNet as crowd density increases.

The upstream person detector is the shared RT-DETRv2 in
models/_detectors.py.

Uses the same posture scoring, keypoint gating, tracking and temporal
confirmation as the other pose wrappers (models/fall/_geometry.py,
models/_tracker.py) for apples-to-apples comparison between backbones.

BlazePose landmarks carry a `visibility` score which is used here as the
per-keypoint confidence — landmarks for occluded joints are still
*emitted* with plausible-looking coordinates (BlazePose hallucinates
out-of-view joints by design), so trusting them unfiltered is what makes
this backbone look far worse than it is on crowd footage.
"""

import os
import urllib.request

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

# BlazePose landmark indices (33-point model)
LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER = 11, 12
LM_LEFT_HIP, LM_RIGHT_HIP = 23, 24

# mediapipe>=0.10.13 removed the old mp.solutions.pose API entirely — the
# new Tasks API (mp.tasks.python.vision.PoseLandmarker) requires a .task
# model file, which isn't bundled with the pip package and must be
# downloaded once and cached locally.
_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
_MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model_weights", "mediapipe")


def _get_model_path(model_complexity: int) -> str:
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    url = _MODEL_URLS[model_complexity]
    filename = os.path.basename(url)
    path = os.path.abspath(os.path.join(_MODEL_CACHE_DIR, filename))
    if not os.path.exists(path):
        print(f"Downloading MediaPipe pose model to {path} ...")
        urllib.request.urlretrieve(url, path)
    return path


class MediaPipeFallDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "fall_mediapipe_pose"
    # CPU-only: MediaPipe/BlazePose has no first-class GPU path in this stack.
    # Only the upstream RT-DETRv2 person detector uses self.device.
    gpu_accelerated = False

    def __init__(self, model_complexity: int = 1, conf_threshold: float = 0.5,
                 horizontal_angle_threshold_deg: float = 45.0,
                 min_kp_conf: float = DEFAULT_MIN_KP_CONF,
                 confirm_frames: int = 5, iou_match_threshold: float = 0.3,
                 person_detector_conf: float = 0.4, device=None):
        super().__init__(device=device)  # BlazePose runs on CPU via mediapipe; device kept for interface consistency
        self.model_complexity = model_complexity  # 0=lite, 1=full, 2=heavy
        self.conf_threshold = conf_threshold
        self.horizontal_angle_threshold_deg = horizontal_angle_threshold_deg
        self.score_threshold = angle_threshold_to_score(horizontal_angle_threshold_deg)
        self.min_kp_conf = min_kp_conf
        self.confirm_frames = confirm_frames
        self.person_detector_conf = person_detector_conf
        self._person_detector = None
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold,
                                   history_len=max(confirm_frames * 2, 8))

    def load(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = _get_model_path(self.model_complexity)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.conf_threshold,
            # min_tracking_confidence has no effect in IMAGE running mode —
            # each crop is an independent detection, there is no track to
            # carry forward. Temporal continuity is handled by our own
            # IoUTracker on the person-detector boxes instead.
        )
        self._mp = mp
        self._model = vision.PoseLandmarker.create_from_options(options)
        # BlazePose is single-person; a person detector runs upstream to crop
        # each person before pose is applied on crowd frames.
        from models._detectors import get_detector
        self._person_detector = get_detector(device=self.device)
        self._person_detector.load()
        self._tracker.reset()

    def _landmarks_to_arrays(self, landmarks, crop_w: int, crop_h: int,
                             offset_x: int, offset_y: int):
        """BlazePose landmarks -> (xy in original-frame pixels, visibility)."""
        import numpy as np
        xy = np.array([[lm.x * crop_w + offset_x, lm.y * crop_h + offset_y]
                       for lm in landmarks], dtype=np.float64)
        vis = np.array([getattr(lm, "visibility", 1.0) for lm in landmarks],
                       dtype=np.float64)
        return xy, vis

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import cv2
        import numpy as np

        h, w = frame.shape[:2]

        from models._detectors import COCO_PERSON
        raw = self._person_detector.detect_with_labels(
            frame, classes=(COCO_PERSON,),
            conf_threshold=self.person_detector_conf,
        )

        boxes, det_confs = [], []
        for (x1, y1, x2, y2), _label, score in raw:
            # Clamp to the frame on all four sides — an unclamped x2/y2 makes
            # the crop silently smaller than the box, which then misplaces
            # every landmark mapped back through it.
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append([x1, y1, x2, y2])
            det_confs.append(float(score))

        if not boxes:
            return []

        track_ids = self._tracker.update(boxes, frame_index)
        detections = []

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            tid = track_ids[i]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_crop)
            result = self._model.detect(mp_image)
            if not result.pose_landmarks:
                self._tracker.append_state(tid, False)
                continue

            crop_h, crop_w = crop.shape[:2]
            # New Tasks API returns a list of per-person landmark lists;
            # take the first (BlazePose is single-person per call anyway).
            kp_xy, kp_vis = self._landmarks_to_arrays(
                result.pose_landmarks[0], crop_w, crop_h, x1, y1
            )

            angle = torso_angle_deg(
                kp_xy, kp_vis, min_kp_conf=self.min_kp_conf,
                shoulder_idx=(LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER),
                hip_idx=(LM_LEFT_HIP, LM_RIGHT_HIP),
            )
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
                keypoints=np.concatenate([kp_xy, kp_vis[:, None]], axis=1).tolist(),
                extra={
                    "torso_angle_deg": angle,
                    "posture_score": score,
                    "frame_is_down": frame_is_down,
                    "track_id": tid,
                    "detector_confidence": det_confs[i],
                },
            ))
        return detections
