"""
Fall detection via PoseC3D (MMAction2).

Instead of treating keypoints as a sparse graph (like ST-GCN), PoseC3D
renders pose sequences as stacked 2D heatmaps per joint and runs a 3D-CNN
over the resulting (T, H, W) heatmap volume. This tends to be more robust
to noisy/partial keypoint estimates than pure skeleton-GCN approaches
(a jittery or briefly-missing joint blurs one heatmap channel rather than
breaking a graph edge), which matters on real CCTV footage where pose
estimation quality dips with distance, motion blur, and occlusion.

Heavier than ST-GCN or the single-frame heuristics — budget more
inference time per clip. Uses YOLOv8-pose upstream for 2D keypoints,
consistent with the ST-GCN wrapper, so the two are comparable on the
same pose-detection quality.

**The heatmap rendering is the whole model's input and it has to be right.**
Two properties, both handled in models/fall/_skeleton.py:

  - Joints are rendered as actual gaussians, not single lit pixels. One
    pixel in a 56x56 map is essentially gone after the network's stem
    pools it; the gaussian's spatial support is what the 3D convolutions
    have to work with.
  - Coordinates are normalized against the *clip's* bounding box, never
    each frame's own. Per-frame normalization rescales a descending body
    back to full extent every frame — it deletes the vertical motion,
    which is the entire fall signal.

Needs an MMAction2 PoseC3D checkpoint trained on fall-relevant classes
(NTU RGB+D fall subset is the common starting point). Without one the
wrapper falls back to the shared geometric posture verdict and tags
`extra["scoring"] = "geometric_fallback"` rather than reporting untrained
output as a PoseC3D result.
"""

from models.base import BaseModelWrapper, Detection
from models._weights import resolve as _resolve_weight_path
from models.fall._geometry import (
    DEFAULT_MIN_KP_CONF,
    angle_threshold_to_score,
    bbox_aspect_ratio,
    posture_score,
    torso_angle_deg,
    verdict_from_scores,
)
from models.fall._skeleton import render_heatmap_volume, resample_sequence
from models.fall._tracker import IoUTracker


class PoseC3DFallDetector(BaseModelWrapper):
    consumption_type = "clip"
    name = "fall_posec3d"

    def __init__(self, sequence_len: int = 32, heatmap_size: int = 56,
                 heatmap_sigma: float = 2.0,
                 posec3d_config: str = "configs/posec3d/posec3d_fall.py",
                 posec3d_checkpoint: str = None, pose_conf_threshold: float = 0.4, conf_threshold: float = 0.4,
                 iou_match_threshold: float = 0.3,
                 min_kp_conf: float = DEFAULT_MIN_KP_CONF,
                 horizontal_angle_threshold_deg: float = 45.0,
                 confirm_frames: int = 5, device=None):
        super().__init__(device=device)
        self.sequence_len = sequence_len
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.posec3d_config = posec3d_config
        self.posec3d_checkpoint = posec3d_checkpoint
        self.conf_threshold = conf_threshold
        self.pose_conf_threshold = conf_threshold
        self.iou_match_threshold = iou_match_threshold
        self.min_kp_conf = min_kp_conf
        self.score_threshold = angle_threshold_to_score(horizontal_angle_threshold_deg)
        self.confirm_frames = confirm_frames
        self.clip_len = sequence_len
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold,
                                   history_len=sequence_len * 2)
        self._trained = False

    @property
    def min_clip_frames(self) -> int:
        return 1  # keeps its own per-track history

    @property
    def clip_stride(self) -> int:
        return 1  # every sampled frame, or per-track sequences get holes

    def load(self):
        from ultralytics import YOLO
        self._pose_model = YOLO(_resolve_weight_path("yolov8s-pose.pt"))
        self._pose_model.to(self.device)
        self._tracker.reset()

        if not self.posec3d_checkpoint:
            print(f"[{self.name}] No posec3d_checkpoint given - falling back to "
                  "the geometric posture verdict. Detections are tagged "
                  "extra.scoring='geometric_fallback'.")
            self._model = None
            self._trained = False
            return

        from mmaction.apis import init_recognizer
        self._model = init_recognizer(
            self.posec3d_config, self.posec3d_checkpoint, device=self.device
        )
        self._trained = True

    def _classify(self, keypoint_seq, score_seq):
        import numpy as np
        import torch

        volume = render_heatmap_volume(
            keypoint_seq,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
            keypoint_scores=score_seq,
            min_kp_conf=self.min_kp_conf,
        )
        # (T, J, H, W) -> (1, J, T, H, W) — PoseC3D's expected clip format,
        # joints as channels and time as the depth axis.
        tensor = torch.from_numpy(np.ascontiguousarray(volume))
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # MMAction2 1.x dropped `return_loss=`; call the backbone+head
            # directly and fall back to the 0.x signature if that's what's
            # installed.
            try:
                logits = self._model(tensor, mode="tensor")
            except TypeError:
                logits = self._model(tensor, return_loss=False)

        probs = torch.softmax(torch.as_tensor(logits).float().reshape(1, -1), dim=-1)[0]
        if probs.numel() < 2:
            return False, float(probs[0])
        return bool(probs[1] > probs[0]), float(probs[1])

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        current_frame = clip_frames[-1]
        results = self._pose_model.predict(
            current_frame, conf=self.pose_conf_threshold, device=self.device, verbose=False
        )
        detections = []
        if not results or results[0].keypoints is None or results[0].boxes is None:
            return detections

        r = results[0]
        boxes = [r.boxes[i].xyxy[0].tolist() for i in range(len(r.boxes))]
        track_ids = self._tracker.update(boxes, frame_index)

        for i, kp in enumerate(r.keypoints.data):
            if i >= len(boxes):
                break
            kp_np = kp.cpu().numpy()
            kp_xy = kp_np[:, :2]
            kp_scores = kp_np[:, 2] if kp_np.shape[1] > 2 else None
            xyxy = boxes[i]
            tid = track_ids[i]

            angle = torso_angle_deg(kp_xy, kp_scores, min_kp_conf=self.min_kp_conf)
            frame_score = posture_score(angle, bbox_aspect_ratio(xyxy))

            self._tracker.append_state(tid, {
                "keypoints": kp_xy,
                "scores": kp_scores,
                "posture_score": frame_score,
            })
            states = self._tracker.states(tid)
            if len(states) < self.confirm_frames:
                continue

            if self._trained:
                window = resample_sequence(states, self.sequence_len)
                keypoint_seq = [s["keypoints"] for s in window]
                score_seq = ([s["scores"] for s in window]
                             if all(s["scores"] is not None for s in window) else None)
                is_fall, conf = self._classify(keypoint_seq, score_seq)
                scoring = "posec3d"
            else:
                is_fall, conf = verdict_from_scores(
                    [s["posture_score"] for s in states],
                    self.score_threshold, self.confirm_frames,
                )
                scoring = "geometric_fallback"

            detections.append(Detection(
                model_name=self.name,
                label="fall" if is_fall else "standing",
                confidence=conf,
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=xyxy,
                keypoints=kp_np.tolist(),
                extra={
                    "track_id": tid,
                    "sequence_len": len(states),
                    "torso_angle_deg": angle,
                    "posture_score": frame_score,
                    "scoring": scoring,
                },
            ))
        return detections
