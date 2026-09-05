"""
Violence/altercation detection via X3D.

Lightweight 3D-CNN (Facebook/Meta, via pytorchvideo/torch.hub) — expands
a 2D architecture along space, time, width, and depth axes efficiently,
giving a good accuracy/compute tradeoff. Good default choice for CPU or
budget-GPU deployment; the "lightweight" comparison point among the
violence classifiers here.

**Works without fine-tuning.** With no `weights_path` this runs the
Kinetics-pretrained checkpoint and scores violence zero-shot by summing
probability mass over Kinetics' fighting classes (`punching person`,
`wrestling`, `slapping`, `headbutting`, ...) — see models/violence/_common.py.
That is a real signal, not a placeholder, though it will also fire on
sparring and martial arts. Fine-tuning on RWF-2000 (real CCTV/surveillance
footage, so the closest domain match for this testbed's clips), Hockey
Fight, or RLVS and passing the checkpoint via `weights_path` gives a
binary head that is used directly instead.

Each X3D variant has its own native input resolution and clip length;
feeding x3d_s at 224x224/16 (as this wrapper previously did) is off-spec
for the checkpoint and costs accuracy for no benefit.
"""

from models.base import BaseModelWrapper, Detection
from models.violence._common import (
    ViolenceScoringMixin,
    clip_to_tensor,
    infer_num_classes,
)

# (clip_len, spatial size) per variant, from the pytorchvideo model zoo.
_VARIANT_SPECS = {
    "x3d_xs": (4, 182),
    "x3d_s": (13, 182),
    "x3d_m": (16, 224),
    "x3d_l": (16, 312),
}


class X3DViolenceClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_x3d"

    def __init__(self, variant: str = "x3d_s", weights_path: str = None,
                 conf_threshold: float = 0.5, use_person_roi: bool = True,
                 device=None):
        super().__init__(device=device)
        if variant not in _VARIANT_SPECS:
            raise ValueError(f"variant must be one of {list(_VARIANT_SPECS)}")
        self.variant = variant  # x3d_xs / x3d_s / x3d_m — smaller = faster, less accurate
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.clip_len, self.input_size = _VARIANT_SPECS[variant]
        self._init_roi(use_person_roi)

    @property
    def min_clip_frames(self) -> int:
        return self.clip_len

    def load(self):
        import torch
        self._model = torch.hub.load(
            "facebookresearch/pytorchvideo", model=self.variant, pretrained=True
        )
        num_classes = 400
        if self.weights_path:
            state = torch.load(self.weights_path, map_location=self.device)
            self._model.load_state_dict(state)
            num_classes = infer_num_classes(state, default=2)
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
            extra={"person_roi": list(roi) if roi else None, "variant": self.variant, "clip_len": self.clip_len,
                   "input_size": self.input_size, **extras},
        )]
