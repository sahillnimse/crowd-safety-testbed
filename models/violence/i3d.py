"""
Violence/altercation detection via I3D (Inflated 3D ConvNet).

"Inflates" a 2D image classifier into 3D by extending its filters and
pooling kernels along the time axis, then initializes from the 2D ImageNet
weights before training on video — a proven, widely benchmarked
architecture for action recognition. Included here mainly because I3D
fine-tuned on RWF-2000 is one of the most common reference points in the
violence-detection literature, making it a useful sanity check against
published numbers, alongside the newer/lighter architectures (X3D, TSM)
also in this testbed.

Uses the RGB-only I3D stream (no optical-flow stream) to keep inference
cost comparable to the other single-stream classifiers here.

**Works without fine-tuning**, via pytorchvideo's Kinetics-pretrained
`i3d_r50` scored zero-shot over Kinetics' fighting classes. If
pytorchvideo isn't installed there is no pretrained I3D to fall back to —
the wrapper raises rather than quietly substituting a randomly-initialized
r3d_18, which previously produced confident-looking labels from noise.
Pass `weights_path` for a fine-tuned binary checkpoint (RWF-2000 is the
standard choice for I3D specifically).
"""

from models.base import BaseModelWrapper, Detection
from models.violence._common import (
    ViolenceScoringMixin,
    clip_to_tensor,
    infer_num_classes,
)

CLIP_LEN = 32  # what the pytorchvideo i3d_r50 Kinetics checkpoint was trained at
INPUT_SIZE = 224


class I3DViolenceClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_i3d"

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

    def load(self):
        import torch

        try:
            from pytorchvideo.models.hub import i3d_r50
        except ImportError as e:
            raise ImportError(
                "violence_i3d needs pytorchvideo for the Kinetics-pretrained "
                "i3d_r50 checkpoint (`pip install pytorchvideo`). Refusing to "
                "substitute a randomly-initialized network - its output would "
                "look like detections but be noise."
            ) from e

        self._model = i3d_r50(pretrained=True)
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
            extra={"person_roi": list(roi) if roi else None, "clip_len": self.clip_len, **extras},
        )]
