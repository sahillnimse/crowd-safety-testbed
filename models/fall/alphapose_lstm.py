"""
Fall detection via AlphaPose + temporal LSTM classifier.

AlphaPose generally handles occlusion and crowd density better than
YOLO-pose/BlazePose (its top-down pipeline with a dedicated pose-NMS step
is built for multi-person scenes), making it a strong candidate for the
"crowd fall detection" side of this testbed specifically.

Unlike the single-frame heuristic wrappers, this one is temporal: pose
keypoints are tracked per-person across a short window and fed to an
LSTM that classifies the sequence as fall / not-fall.

**Two dependencies this wrapper degrades around, loudly rather than
silently**, because previously it did neither and reported the result as
if both were satisfied:

  - *AlphaPose itself* ships as a research repo, not a pip package. When
    it isn't importable the wrapper falls back to YOLOv8-pose for
    keypoints and tags detections `extra["pose_source"] = "yolov8_pose"`.
    Note that makes it a pose-source duplicate of the ST-GCN wrapper —
    the AlphaPose comparison is only meaningful with AlphaPose installed.
  - *The LSTM head* needs a checkpoint trained on keypoint sequences
    (UR Fall / Le2i). Without `lstm_weights_path` an untrained LSTM emits
    random labels, so the wrapper falls back to the shared geometric
    posture verdict and tags `extra["scoring"] = "geometric_fallback"`.

Keypoint sequences are normalized per clip (mid-hip origin, median torso
length scale) before hitting the LSTM — raw pixel coordinates, which is
what this fed previously, make the model a function of screen position
rather than body shape.
"""

from models.base import BaseModelWrapper, Detection
from models.fall._geometry import (
    DEFAULT_MIN_KP_CONF,
    angle_threshold_to_score,
    bbox_aspect_ratio,
    posture_score,
    torso_angle_deg,
    verdict_from_scores,
)
from models.fall._skeleton import normalize_sequence, resample_sequence
from models.fall._tracker import IoUTracker


class AlphaPoseFallDetector(BaseModelWrapper):
    consumption_type = "clip"  # needs a short temporal window per track
    name = "fall_alphapose_lstm"

    def __init__(self, sequence_len: int = 16, lstm_weights_path: str = None,
                 conf_threshold: float = 0.4, iou_match_threshold: float = 0.3,
                 min_kp_conf: float = DEFAULT_MIN_KP_CONF,
                 horizontal_angle_threshold_deg: float = 45.0,
                 confirm_frames: int = 5,
                 alphapose_config: str = "configs/alphapose/256x192_res50_lr1e-3_1x.yaml",
                 device=None):
        super().__init__(device=device)
        self.sequence_len = sequence_len
        self.lstm_weights_path = lstm_weights_path
        self.conf_threshold = conf_threshold
        self.iou_match_threshold = iou_match_threshold
        self.min_kp_conf = min_kp_conf
        self.score_threshold = angle_threshold_to_score(horizontal_angle_threshold_deg)
        self.confirm_frames = confirm_frames
        self.alphapose_config = alphapose_config
        self.clip_len = sequence_len
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold,
                                   history_len=sequence_len * 2)
        self._pose_source = None
        self._trained = False

    @property
    def min_clip_frames(self) -> int:
        return 1  # keeps its own per-track history

    @property
    def clip_stride(self) -> int:
        return 1  # every sampled frame, or per-track sequences get holes

    def load(self):
        self._load_pose_backbone()
        self._load_lstm()

    def _load_pose_backbone(self):
        try:
            from alphapose.models import builder
            from alphapose.utils.config import update_config

            cfg = update_config(self.alphapose_config)
            self._pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
            self._pose_model.to(self.device).eval()
            self._pose_source = "alphapose"
        except (ImportError, FileNotFoundError) as e:
            print(f"[{self.name}] AlphaPose unavailable ({e.__class__.__name__}: {e}) - "
                  "using YOLOv8-pose for keypoints instead. Detections are tagged "
                  "extra.pose_source='yolov8_pose'; install AlphaPose for a true "
                  "AlphaPose-vs-others comparison.")
            from ultralytics import YOLO
            self._pose_model = YOLO("yolov8s-pose.pt")
            self._pose_model.to(self.device)
            self._pose_source = "yolov8_pose"
        self._tracker.reset()

    def _load_lstm(self):
        if not self.lstm_weights_path:
            print(f"[{self.name}] No lstm_weights_path given - an untrained LSTM "
                  "would emit random labels, so falling back to the geometric "
                  "posture verdict. Detections are tagged "
                  "extra.scoring='geometric_fallback'.")
            self._lstm = None
            self._trained = False
            return

        import torch
        import torch.nn as nn

        num_joints = 26 if self._pose_source == "alphapose" else 17  # Halpe-26 vs COCO-17

        class FallLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, 2)  # [not_fall, fall]

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self._lstm = FallLSTM(input_dim=num_joints * 2)
        self._lstm.load_state_dict(
            torch.load(self.lstm_weights_path, map_location=self.device)
        )
        self._lstm.to(self.device).eval()
        self._trained = True

    def _estimate_pose(self, frame):
        """-> list of {bbox, keypoints (V,2), scores (V,), conf}, whichever
        backbone is active."""
        import numpy as np

        if self._pose_source == "alphapose":
            people = self._pose_model.detect_and_estimate(frame)
            out = []
            for p in people:
                kps = np.asarray(p["keypoints"], dtype=np.float64)
                out.append({
                    "bbox": list(p["bbox"]),
                    "keypoints": kps[:, :2],
                    "scores": kps[:, 2] if kps.shape[1] > 2 else None,
                    "conf": float(p["conf"]),
                })
            return out

        results = self._pose_model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )
        if not results or results[0].keypoints is None or results[0].boxes is None:
            return []
        r = results[0]
        out = []
        for i, kp in enumerate(r.keypoints.data):
            if i >= len(r.boxes):
                break
            kp_np = kp.cpu().numpy()
            out.append({
                "bbox": r.boxes[i].xyxy[0].tolist(),
                "keypoints": kp_np[:, :2],
                "scores": kp_np[:, 2] if kp_np.shape[1] > 2 else None,
                "conf": float(r.boxes[i].conf[0]),
            })
        return out

    def _classify(self, keypoint_seq):
        import numpy as np
        import torch

        seq = normalize_sequence(keypoint_seq)          # (T, V, 2)
        flat = seq.reshape(seq.shape[0], -1)            # (T, V*2)
        tensor = torch.from_numpy(np.ascontiguousarray(flat)).float()
        tensor = tensor.unsqueeze(0).to(self.device)    # (1, T, V*2)

        with torch.no_grad():
            logits = self._lstm(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        return bool(probs[1] > probs[0]), float(probs[1])

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        current_frame = clip_frames[-1]
        people = [p for p in self._estimate_pose(current_frame)
                  if p["conf"] >= self.conf_threshold]
        if not people:
            return []

        boxes = [p["bbox"] for p in people]
        track_ids = self._tracker.update(boxes, frame_index)
        detections = []

        for i, person in enumerate(people):
            tid = track_ids[i]
            angle = torso_angle_deg(person["keypoints"], person["scores"],
                                    min_kp_conf=self.min_kp_conf)
            frame_score = posture_score(angle, bbox_aspect_ratio(person["bbox"]))

            self._tracker.append_state(tid, {
                "keypoints": person["keypoints"],
                "posture_score": frame_score,
            })
            states = self._tracker.states(tid)
            if len(states) < self.confirm_frames:
                continue

            if self._trained:
                seq = resample_sequence([s["keypoints"] for s in states], self.sequence_len)
                is_fall, conf = self._classify(seq)
                scoring = "lstm"
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
                bbox=person["bbox"],
                keypoints=person["keypoints"].tolist(),
                extra={
                    "track_id": tid,
                    "sequence_len": len(states),
                    "torso_angle_deg": angle,
                    "posture_score": frame_score,
                    "scoring": scoring,
                    "pose_source": self._pose_source,
                },
            ))
        return detections
