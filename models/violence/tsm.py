"""
Violence/altercation detection via TSM (Temporal Shift Module).

Adds temporal reasoning to an otherwise-standard 2D-CNN (ResNet) by
shifting a slice of each layer's channels forward/backward along the
time axis before convolving.
"""

import os
from models.base import BaseModelWrapper, Detection
from models.violence._common import (
    KINETICS_MEAN,
    KINETICS_STD,
    ViolenceScoringMixin,
    infer_num_classes,
    preprocess_clip,
)

NUM_SEGMENTS = 8
INPUT_SIZE = 224


class TSMViolenceClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_tsm"

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.5,
                 shift_div: int = 8, use_person_roi: bool = True, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.shift_div = shift_div
        self.num_segments = NUM_SEGMENTS
        self.clip_len = NUM_SEGMENTS
        self.input_size = INPUT_SIZE
        self._init_roi(use_person_roi)

    @property
    def min_clip_frames(self) -> int:
        return self.num_segments

    def _build_tsm_resnet(self, num_classes: int = 2, pretrained_backbone: bool = True):
        """ResNet-50 backbone with temporal shift applied before each residual block."""
        import torch
        import torch.nn as nn
        import torchvision

        shift_div = self.shift_div
        num_segments = self.num_segments

        class TemporalShift(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                nt, c, h, w = x.shape
                n = nt // num_segments
                x = x.view(n, num_segments, c, h, w)
                fold = c // shift_div
                out = torch.zeros_like(x)
                out[:, :-1, :fold] = x[:, 1:, :fold]
                out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
                out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
                out = out.view(nt, c, h, w)
                return self.net(out)

        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = torchvision.models.resnet50(weights=weights)
        for name, module in backbone.named_children():
            if name.startswith("layer"):
                for block in module:
                    block.conv1 = TemporalShift(block.conv1)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone

    def load(self):
        import torch

        if self.weights_path and os.path.exists(self.weights_path):
            state = torch.load(self.weights_path, map_location=self.device)
            num_classes = infer_num_classes(state, default=2)
            self._model = self._build_tsm_resnet(num_classes=num_classes, pretrained_backbone=False)
            self._model.load_state_dict(state)
        else:
            # Safe ImageNet ResNet50 + TSM backbone initialization
            num_classes = 2
            self._model = self._build_tsm_resnet(num_classes=num_classes, pretrained_backbone=True)

        self._model.to(self.device).eval()
        self._resolve_head(num_classes)
        self._load_roi()

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import numpy as np
        import torch

        if len(clip_frames) < self.min_clip_frames:
            return []

        roi = self._clip_roi(clip_frames)
        arr = preprocess_clip(clip_frames, self.num_segments, self.input_size,
                              mean=KINETICS_MEAN, std=KINETICS_STD, roi=roi)
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(1, 0, 2, 3)))
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            logits = logits.mean(dim=0, keepdim=True)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        label, score, extras = self._score(probs)
        return [Detection(
            model_name=self.name,
            label=label,
            confidence=score,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            extra={"person_roi": list(roi) if roi else None, "num_segments": self.num_segments, "shift_div": self.shift_div, **extras},
        )]
