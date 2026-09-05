"""
Violence/altercation detection via SlowFast Networks.

Dual-pathway 3D-CNN (Facebook/Meta, via pytorchvideo): a "slow" pathway
samples frames sparsely to capture spatial/semantic detail, a "fast"
pathway samples densely at low channel capacity to capture motion —
the combination is specifically strong on fast, high-frequency motion
like punches, shoving, and sudden altercation onset, which is why it's
included here alongside X3D rather than as a straight substitute.

Heavier than X3D — budget more inference time per clip.

**Works without fine-tuning.** With no `weights_path` the Kinetics
checkpoint is scored zero-shot over Kinetics' fighting classes (see
models/violence/_common.py). Fine-tuning on RWF-2000 / Hockey Fight / RLVS
and passing `weights_path` switches to the binary head. Given SlowFast's
motion sensitivity, this is the wrapper most likely to be worth the
fine-tuning effort for altercation onset specifically.
"""

from models.base import BaseModelWrapper, Detection
from models.violence._common import (
    ViolenceScoringMixin,
    clip_to_tensor,
    infer_num_classes,
)

FAST_PATHWAY_CLIP_LEN = 32  # slow pathway is subsampled from this internally by the model
INPUT_SIZE = 256


class SlowFastViolenceClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_slowfast"

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.5,
                 alpha: int = 4, use_person_roi: bool = True, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.alpha = alpha  # slow:fast frame sampling ratio (standard SlowFast default)
        self.clip_len = FAST_PATHWAY_CLIP_LEN
        self.input_size = INPUT_SIZE
        self._init_roi(use_person_roi)

    @property
    def min_clip_frames(self) -> int:
        return self.clip_len

    def load(self):
        import torch
        self._model = torch.hub.load(
            "facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True
        )
        num_classes = 400
        if self.weights_path:
            state = torch.load(self.weights_path, map_location=self.device)
            self._model.load_state_dict(state)
            num_classes = infer_num_classes(state, default=2)
        self._model.to(self.device).eval()
        self._resolve_head(num_classes)
        self._load_roi()

    def _build_pathways(self, tensor):
        """SlowFast expects [slow_pathway, fast_pathway] as separate tensors,
        the slow one temporally subsampled by self.alpha from the fast one."""
        import torch
        fast = tensor
        num_slow = max(1, fast.shape[2] // self.alpha)
        slow_idx = torch.linspace(0, fast.shape[2] - 1, num_slow).long()
        slow = torch.index_select(fast, 2, slow_idx.to(fast.device))
        return [slow, fast]

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import torch

        if len(clip_frames) < self.min_clip_frames:
            return []

        roi = self._clip_roi(clip_frames)
        tensor = clip_to_tensor(clip_frames, self.clip_len, self.input_size,
                                self.device, roi=roi)
        pathways = self._build_pathways(tensor)

        with torch.no_grad():
            logits = self._model(pathways)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        label, score, extras = self._score(probs)
        return [Detection(
            model_name=self.name,
            label=label,
            confidence=score,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            extra={"person_roi": list(roi) if roi else None, "alpha": self.alpha, "clip_len": self.clip_len, **extras},
        )]
