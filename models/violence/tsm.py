"""
Violence/altercation detection via TSM (Temporal Shift Module).

Adds temporal reasoning to an otherwise-standard 2D-CNN (ResNet) by
shifting a slice of each layer's channels forward/backward along the
time axis before convolving — this gets much of the temporal modeling
benefit of a full 3D-CNN (X3D, SlowFast, C3D) at close to 2D-CNN compute
cost, since the shift operation itself is free (no extra FLOPs/params).
Included as the "efficient temporal reasoning without 3D convs" option —
a useful comparison point if inference budget is tight but the single-
frame/optical-flow approaches aren't giving enough temporal context.

Backbone is a standard ResNet-50 with shift modules inserted, consistent
with the original TSM paper's setup. ImageNet weights initialize the
backbone (TSM is designed to be fine-tuned from 2D ImageNet init, not
trained from scratch).

**This wrapper requires `weights_path`.** There is no pretrained
TSM-on-Kinetics checkpoint wired up here, so the classification head is
randomly initialized. With a 2-class head that yields ~0.5 confidence on
every clip and labels roughly half of them "violence" — noise that reads
as the most sensitive detector in the comparison tables. It raises at
load() rather than producing that. Fine-tune on RWF-2000 / Hockey Fight /
RLVS and pass the checkpoint.
"""

from models.base import BaseModelWrapper, Detection
from models.violence._common import (
    KINETICS_MEAN,
    KINETICS_STD,
    ViolenceScoringMixin,
    infer_num_classes,
    preprocess_clip,
)

NUM_SEGMENTS = 8  # TSM uses sparse segment-based sampling rather than a dense clip
INPUT_SIZE = 224


class TSMViolenceClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_tsm"

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.5,
                 shift_div: int = 8, use_person_roi: bool = True, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.shift_div = shift_div  # fraction of channels shifted (1/shift_div), per TSM paper default
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
                # x: (N*T, C, H, W) -> reshape to (N, T, C, H, W), shift along T, reshape back
                nt, c, h, w = x.shape
                n = nt // num_segments
                x = x.view(n, num_segments, c, h, w)
                fold = c // shift_div
                out = torch.zeros_like(x)
                out[:, :-1, :fold] = x[:, 1:, :fold]         # shift left (future -> now)
                out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right (past -> now)
                out[:, :, 2 * fold:] = x[:, :, 2 * fold:]     # unshifted remainder
                out = out.view(nt, c, h, w)
                return self.net(out)

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = torchvision.models.resnet50(weights=weights)
        for name, module in backbone.named_children():
            if name.startswith("layer"):
                for block in module:
                    block.conv1 = TemporalShift(block.conv1)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone

    def load(self):
        import torch

        if not self.weights_path:
            raise ValueError(
                "violence_tsm requires weights_path: the classification head "
                "here is randomly initialized, so without a fine-tuned "
                "checkpoint every clip scores ~0.5 and about half get labeled "
                "'violence' - noise that is indistinguishable from real "
                "detections downstream. Fine-tune on RWF-2000 / Hockey Fight / "
                "RLVS first, or drop violence_tsm from the model list."
            )

        state = torch.load(self.weights_path, map_location=self.device)
        num_classes = infer_num_classes(state, default=2)
        # The checkpoint supplies every weight, so skip the ImageNet download.
        self._model = self._build_tsm_resnet(num_classes=num_classes,
                                             pretrained_backbone=False)
        self._model.load_state_dict(state)
        self._model.to(self.device).eval()
        self._resolve_head(num_classes)
        self._load_roi()

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import numpy as np
        import torch

        if len(clip_frames) < self.min_clip_frames:
            return []

        # (C, T, H, W) -> (T, C, H, W); for TSM the segment axis *is* the
        # batch axis (N*T with N=1), which is what the shift modules reshape.
        roi = self._clip_roi(clip_frames)
        arr = preprocess_clip(clip_frames, self.num_segments, self.input_size,
                              mean=KINETICS_MEAN, std=KINETICS_STD, roi=roi)
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(1, 0, 2, 3)))
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            logits = logits.mean(dim=0, keepdim=True)  # average segment predictions
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
