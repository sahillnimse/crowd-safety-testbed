"""
Violence/altercation detection via MMAction2's SlowOnly.

The single-pathway counterpart to SlowFast — sparse temporal sampling,
no fast/motion pathway. Included here as the "off-the-shelf framework"
baseline: rather than a from-scratch or torch.hub architecture like the
other wrappers in this testbed, this one goes through the MMAction2
config/checkpoint system directly, which is how a lot of published
violence-detection results in the literature are actually produced —
useful if you want a comparison point that matches published benchmarks
config-for-config rather than a reimplementation.

**Requires a config + checkpoint.** MMAction2 has no meaningful default
here, so `checkpoint_path` is mandatory: MMAction2 ships example SlowOnly
configs for RWF-2000 specifically that are a reasonable starting point
before further fine-tuning on this testbed's own footage. If the
checkpoint's head is 400-class Kinetics rather than binary, it's scored
zero-shot over Kinetics' fighting classes like the other pretrained
wrappers; a 2-class head is used directly.
"""

from models.base import BaseModelWrapper, Detection
from models.violence._common import ViolenceScoringMixin, clip_to_tensor

CLIP_LEN = 8  # SlowOnly's typical sparse sampling (vs SlowFast's fast-pathway 32)
INPUT_SIZE = 224


class MMActionSlowOnlyClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_mmaction_slowonly"

    def __init__(self, config_path: str = "configs/mmaction/slowonly_violence.py",
                 checkpoint_path: str = None, conf_threshold: float = 0.5,
                 num_classes: int = None, use_person_roi: bool = True,
                 device=None):
        super().__init__(device=device)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.conf_threshold = conf_threshold
        # Optional override when the config's head size can't be introspected.
        self.num_classes = num_classes
        self.clip_len = CLIP_LEN
        self.input_size = INPUT_SIZE
        self._init_roi(use_person_roi)

    @property
    def min_clip_frames(self) -> int:
        return self.clip_len

    def load(self):
        if not self.checkpoint_path:
            raise ValueError(
                "violence_mmaction_slowonly requires checkpoint_path - an "
                "MMAction2 recognizer built from config alone has an untrained "
                "head and would emit meaningless labels. Point it at an "
                "MMAction2 SlowOnly checkpoint (their RWF-2000 configs are a "
                "reasonable starting point), or drop this model from the run."
            )

        from mmaction.apis import init_recognizer
        self._model = init_recognizer(self.config_path, self.checkpoint_path,
                                      device=self.device)
        self._resolve_head(self.num_classes or self._head_num_classes())
        self._load_roi()

    def _head_num_classes(self) -> int:
        """Read the class count off the loaded recognizer's head."""
        head = getattr(self._model, "cls_head", None)
        for attr in ("num_classes", "num_class"):
            value = getattr(head, attr, None)
            if isinstance(value, int):
                return value
        fc = getattr(head, "fc_cls", None)
        if fc is not None and hasattr(fc, "out_features"):
            return int(fc.out_features)
        raise ValueError(
            "Could not determine the SlowOnly head's class count; pass "
            "num_classes= explicitly so the violence score maps correctly."
        )

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import torch

        if len(clip_frames) < self.min_clip_frames:
            return []

        # MMAction2 recognizers expect (N, C, T, H, W)
        roi = self._clip_roi(clip_frames)
        tensor = clip_to_tensor(clip_frames, self.clip_len, self.input_size,
                                self.device, roi=roi)

        with torch.no_grad():
            # MMAction2 1.x uses mode=; 0.x used return_loss=.
            try:
                logits = self._model(tensor, mode="tensor")
            except TypeError:
                logits = self._model(tensor, return_loss=False)

        probs = torch.softmax(
            torch.as_tensor(logits).float().reshape(1, -1), dim=-1
        )[0].cpu().numpy()

        label, score, extras = self._score(probs)
        return [Detection(
            model_name=self.name,
            label=label,
            confidence=score,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            extra={"person_roi": list(roi) if roi else None, "clip_len": self.clip_len, **extras},
        )]
