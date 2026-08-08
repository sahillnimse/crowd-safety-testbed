"""
Fall detection via ST-GCN (Spatial Temporal Graph Convolutional Network).

Purpose-built for skeleton-based action recognition: treats each person's
keypoint sequence as a spatio-temporal graph (joints as nodes, bones as
spatial edges, frame-to-frame joint motion as temporal edges) and learns
fall vs not-fall directly from that structure, rather than a hand-tuned
angle threshold. Generally more accurate at distinguishing genuine falls
from fast-but-benign motion (sitting quickly, bending to pick something up)
than the geometric-heuristic wrappers.

Needs an upstream 2D pose estimator to produce the keypoints ST-GCN
consumes — this wrapper uses YOLOv8-pose for that (swap for AlphaPose if
preferred; the graph-conv stage is agnostic to which detector fed it).

**Weights.** ST-GCN only classifies falls once it has been trained to;
point `stgcn_weights_path` at a checkpoint trained on a fall dataset
(NTU RGB+D fall-relevant classes, or Le2i/UR Fall converted to skeleton
sequences). Without one, this wrapper does *not* score with randomly
initialized weights — that produced coin-flip labels that looked like real
detections in the comparison tables. It falls back to the shared geometric
posture verdict instead, and marks every detection
`extra["scoring"] = "geometric_fallback"` so the fallback can never be
mistaken for a trained ST-GCN result.

Keypoints are normalized per clip (mid-hip origin, median torso length
scale) before being fed to the network. Feeding raw pixel coordinates —
as this wrapper previously did — teaches the model *where in the frame*
a person is rather than what shape they are.
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
    verdict_from_scores,
)
from models.fall._skeleton import normalize_sequence, resample_sequence
from models.fall._tracker import IoUTracker


class STGCNFallDetector(BaseModelWrapper):
    consumption_type = "clip"
    name = "fall_stgcn"

    def __init__(self, sequence_len: int = 30, stgcn_weights_path: str = None,
                 pose_conf_threshold: float = 0.4, conf_threshold: float = 0.4,
                 iou_match_threshold: float = 0.3,
                 min_kp_conf: float = DEFAULT_MIN_KP_CONF,
                 horizontal_angle_threshold_deg: float = 45.0,
                 confirm_frames: int = 5, device=None):
        super().__init__(device=device)
        self.sequence_len = sequence_len
        self.stgcn_weights_path = stgcn_weights_path
        self.conf_threshold = conf_threshold
        self.pose_conf_threshold = conf_threshold
        self.iou_match_threshold = iou_match_threshold
        self.min_kp_conf = min_kp_conf
        self.score_threshold = angle_threshold_to_score(horizontal_angle_threshold_deg)
        self.confirm_frames = confirm_frames
        self.clip_len = sequence_len
        # Enough history for the network window, with headroom so a track
        # that briefly drops out doesn't lose its whole sequence.
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold,
                                   history_len=sequence_len * 2)
        self._trained = False

    @property
    def min_clip_frames(self) -> int:
        # The wrapper keeps its own per-track history, so it only needs the
        # current frame from the runner's buffer.
        return 1

    @property
    def clip_stride(self) -> int:
        # Must see every sampled frame: skipping frames would break track
        # continuity and leave the per-track sequences full of holes.
        return 1

    def load(self):
        from ultralytics import YOLO
        self._pose_model = YOLO(_resolve_weight_path("yolov8s-pose.pt"))
        self._pose_model.to(self.device)
        self._tracker.reset()

        if not self.stgcn_weights_path:
            print(f"[{self.name}] No stgcn_weights_path given - an untrained "
                  "ST-GCN would emit random labels, so falling back to the "
                  "geometric posture verdict. Detections are tagged "
                  "extra.scoring='geometric_fallback'.")
            self._model = None
            self._trained = False
            return

        # pyskl/mmskeleton-style checkpoint when the framework is installed;
        # otherwise a minimal in-repo ST-GCN with the same input contract.
        try:
            from pyskl.models import build_model
            from pyskl.utils import Config
            cfg = Config.fromfile("configs/stgcn/stgcn_fall.py")
            self._model = build_model(cfg.model)
        except ImportError:
            self._model = self._build_fallback_stgcn()

        import torch
        state = torch.load(self.stgcn_weights_path, map_location=self.device)
        self._model.load_state_dict(state)
        self._model.to(self.device).eval()
        self._trained = True

    def _build_fallback_stgcn(self):
        """Minimal ST-GCN stand-in (single graph-conv + temporal-conv block)
        used when pyskl isn't installed but a compatible checkpoint is."""
        import torch.nn as nn

        class MinimalSTGCN(nn.Module):
            def __init__(self, num_joints=17, in_channels=2, hidden=64, num_classes=2):
                super().__init__()
                self.spatial_conv = nn.Conv2d(in_channels, hidden, kernel_size=1)
                self.temporal_conv = nn.Conv2d(hidden, hidden, kernel_size=(9, 1), padding=(4, 0))
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(hidden, num_classes)

            def forward(self, x):  # x: (B, C, T, V)
                x = self.spatial_conv(x)
                x = self.temporal_conv(x)
                x = self.pool(x).flatten(1)
                return self.fc(x)

        return MinimalSTGCN()

    def _classify(self, keypoint_seq):
        """Run the trained ST-GCN over one track's normalized keypoint clip."""
        import numpy as np
        import torch

        seq = normalize_sequence(keypoint_seq)  # (T, V, 2), clip-normalized
        # (T, V, C) -> (C, T, V) -> batch dim
        tensor = torch.from_numpy(np.ascontiguousarray(seq)).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        is_fall = bool(probs[1] > probs[0])
        return is_fall, float(probs[1])

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
                "posture_score": frame_score,
            })
            states = self._tracker.states(tid)
            if len(states) < self.confirm_frames:
                continue  # not enough history on this track yet

            if self._trained:
                seq = resample_sequence([s["keypoints"] for s in states], self.sequence_len)
                is_fall, conf = self._classify(seq)
                scoring = "stgcn"
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
