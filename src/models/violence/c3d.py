"""
Violence/altercation detection via C3D.

An older, simpler 3D-CNN baseline (plain 3D convolutions throughout,
no dual-pathway or transformer machinery).
"""

import os
from models.base import BaseModelWrapper, Detection
from models.violence._common import (
    ViolenceScoringMixin,
    clip_to_tensor,
    infer_num_classes,
)

CLIP_LEN = 16
INPUT_SIZE = 112


class C3DViolenceClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_c3d"

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.5,
                 use_person_roi: bool = True, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.clip_len = CLIP_LEN
        self.input_size = INPUT_SIZE
        self._init_roi(use_person_roi)

    @property
    def min_clip_frames(self) -> int:
        return self.clip_len

    def _build_c3d(self, num_classes: int = 2):
        """Standard C3D architecture (Tran et al. 2015) — 3D-conv stack."""
        import torch.nn as nn

        class C3D(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv3d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool3d((1, 2, 2)),
                    nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                    nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                    nn.Conv3d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                    nn.AdaptiveAvgPool3d(1),
                )
                self.fc = nn.Linear(512, num_classes)

            def forward(self, x):
                x = self.features(x).flatten(1)
                return self.fc(x)

        return C3D(num_classes=num_classes)

    def load(self):
        import torch

        if self.weights_path and os.path.exists(self.weights_path):
            state = torch.load(self.weights_path, map_location=self.device)
            num_classes = infer_num_classes(state, default=2)
            self._model = self._build_c3d(num_classes=num_classes)
            self._model.load_state_dict(state)
        else:
            # No pretrained C3D exists to fall back on, so the whole network
            # is random here — not just the head. It still loads and runs,
            # but is flagged so its output can't be mistaken for detections.
            num_classes = 2
            self._model = self._build_c3d(num_classes=num_classes)
            self._mark_untrained(
                "No weights_path given and no pretrained C3D exists, so the "
                "entire network is randomly initialized. Measured output was "
                "a constant 0.510 on every clip. Fine-tune on Hockey Fight / "
                "RLVS / RWF-2000 for real results."
            )

        self._model.to(self.device).eval()
        self._resolve_head(num_classes)
        self._load_roi()

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import torch

        if len(clip_frames) < self.min_clip_frames:
            return []

        roi = self._clip_roi(clip_frames)
        tensor = clip_to_tensor(clip_frames, self.clip_len, self.input_size,
                                self.device, roi=roi)

        with torch.no_grad():
            logits = self._model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        label, score, extras = self._score(probs)
        return [Detection(
            model_name=self.name,
            label=label,
            confidence=score,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            extra={"person_roi": list(roi) if roi else None, "clip_len": self.clip_len, "input_size": self.input_size, **extras},
        )]
