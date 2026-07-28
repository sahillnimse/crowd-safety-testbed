"""
Violence/altercation detection via C3D.

An older, simpler 3D-CNN baseline (plain 3D convolutions throughout,
no dual-pathway or transformer machinery) — still commonly used as a
reference baseline in violence-detection papers (Hockey Fight, RLVS),
so it's included here mainly as the "oldest/simplest" comparison point:
if the newer architectures (X3D, SlowFast, VideoMAE) aren't clearly
beating C3D on your specific footage, that's a useful signal that the
domain gap or fine-tuning data matters more than architecture choice.

Fixed 16-frame clip length (C3D's original/standard input), fixed
112x112 spatial input (much smaller than the 224x224 used by the other
classifiers here — cheap to run, but loses more spatial detail).

**This wrapper requires `weights_path`.** Unlike X3D/SlowFast/I3D/VideoMAE
there is no pretrained C3D checkpoint to fall back on: the architecture is
built from scratch here, so with no weights every prediction is a coin
flip from randomly-initialized layers. Because the head is binary, those
coin flips land near 0.5 and read as confident violence roughly half the
time — which made this look like the most sensitive detector in the
comparison tables while being pure noise. It now raises at load() instead.
Train on Hockey Fight / RLVS / RWF-2000 and pass the checkpoint.
"""

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

        if not self.weights_path:
            raise ValueError(
                "violence_c3d requires weights_path: there is no pretrained C3D "
                "to fall back on, and an untrained binary head produces "
                "coin-flip 'violence' labels at ~0.5 confidence that are "
                "indistinguishable from real detections in the comparison "
                "tables. Fine-tune on Hockey Fight / RLVS / RWF-2000 first, or "
                "drop violence_c3d from the model list for this run."
            )

        state = torch.load(self.weights_path, map_location=self.device)
        num_classes = infer_num_classes(state, default=2)
        self._model = self._build_c3d(num_classes=num_classes)
        self._model.load_state_dict(state)
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
