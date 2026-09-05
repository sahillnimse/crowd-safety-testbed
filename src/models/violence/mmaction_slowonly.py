"""
Violence/altercation detection via MMAction2 / SlowOnly.
"""

import os
from models.base import BaseModelWrapper, Detection
from models.violence._common import ViolenceScoringMixin, clip_to_tensor

CLIP_LEN = 8
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
        self.num_classes = num_classes
        self.clip_len = CLIP_LEN
        self.input_size = INPUT_SIZE
        self._uses_fallback = False
        self._init_roi(use_person_roi)

    @property
    def min_clip_frames(self) -> int:
        return self.clip_len

    def load(self):
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                from mmaction.apis import init_recognizer
                self._model = init_recognizer(self.config_path, self.checkpoint_path,
                                              device=self.device)
                self._resolve_head(self.num_classes or self._head_num_classes())
                self._uses_fallback = False
            except Exception:
                self._init_fallback()
        else:
            self._init_fallback()

        self._load_roi()

    def _init_fallback(self):
        import torch
        import torchvision
        self._uses_fallback = True
        # Use torchvision SlowFast r50 or X3D as reliable video model fallback
        try:
            self._model = torchvision.models.video.slowfast_r50(weights=torchvision.models.video.SlowFast_R50_Weights.DEFAULT)
        except Exception:
            self._model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        self._model.to(self.device).eval()
        self._resolve_head(400)

    def _head_num_classes(self) -> int:
        head = getattr(self._model, "cls_head", None)
        for attr in ("num_classes", "num_class"):
            value = getattr(head, attr, None)
            if isinstance(value, int):
                return value
        fc = getattr(head, "fc_cls", None)
        if fc is not None and hasattr(fc, "out_features"):
            return int(fc.out_features)
        return 400

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import torch

        if len(clip_frames) < self.min_clip_frames:
            return []

        roi = self._clip_roi(clip_frames)
        tensor = clip_to_tensor(clip_frames, self.clip_len, self.input_size,
                                self.device, roi=roi)

        with torch.no_grad():
            if self._uses_fallback:
                logits = self._model(tensor)
            else:
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
