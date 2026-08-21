"""
Violence/altercation detection via VideoMAE.

Transformer-based video classifier (HuggingFace transformers), pretrained
via masked video autoencoding then fine-tuned for action recognition —
typically the highest-accuracy option of the architectures in this
testbed, at the cost of being the heaviest to run. Good candidate for an
"accuracy ceiling" reference point against which the lighter CNN-based
models (X3D, SlowFast, TSM) are compared, even if it's not the one you'd
deploy for real-time inference.

**Works without fine-tuning.** The default Kinetics checkpoint is scored
zero-shot over Kinetics' fighting classes, read straight off the model's
own `config.id2label` — so this wrapper needs no separate class-name
download, and the mapping is guaranteed to match the checkpoint actually
loaded. Passing `weights_path` for a fine-tuned binary checkpoint
(RWF-2000 / Hockey Fight / RLVS) switches to the binary head.

The HuggingFace processor handles resizing and normalization, so it is
given RGB uint8 frames and left to do its own preprocessing rather than
being handed an already-normalized tensor.
"""

from models.base import BaseModelWrapper, Detection
from models.violence._common import (
    ViolenceScoringMixin,
    crop_to_roi,
    sample_frame_indices,
)

CLIP_LEN = 16  # VideoMAE-base's pretrained tubelet layout expects 16 frames


class VideoMAEViolenceClassifier(ViolenceScoringMixin, BaseModelWrapper):
    consumption_type = "clip"
    name = "violence_videomae"

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.5,
                 use_person_roi: bool = True, device=None):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.clip_len = CLIP_LEN
        self._init_roi(use_person_roi)

    @property
    def min_clip_frames(self) -> int:
        return self.clip_len

    def load(self):
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        ckpt = self.weights_path or "MCG-NJU/videomae-base-finetuned-kinetics"
        self._processor = VideoMAEImageProcessor.from_pretrained(ckpt)
        self._model = VideoMAEForVideoClassification.from_pretrained(ckpt)
        self._model.to(self.device).eval()

        id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        self._resolve_head(len(id2label), id2label=id2label)
        self._load_roi()
        self._id2label = id2label

    def _sample_rgb(self, frames, roi=None):
        """Uniformly sample `clip_len` frames and convert BGR -> RGB.

        OpenCV decodes BGR; every pretrained backbone here expects RGB.
        Handing the processor BGR silently swaps red and blue in every clip
        the model ever sees.
        """
        import cv2
        idxs = sample_frame_indices(len(frames), self.clip_len)
        return [cv2.cvtColor(crop_to_roi(frames[i], roi), cv2.COLOR_BGR2RGB) for i in idxs]

    def predict(self, clip_frames, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import torch

        if len(clip_frames) < self.min_clip_frames:
            return []

        roi = self._clip_roi(clip_frames)
        sampled = self._sample_rgb(clip_frames, roi)
        inputs = self._processor(sampled, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        label, score, extras = self._score(probs)
        top_idx = int(probs.argmax())
        return [Detection(
            model_name=self.name,
            label=label,
            confidence=score,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            extra={
                "person_roi": list(roi) if roi else None,
                "clip_len": self.clip_len,
                # The winning action class regardless of violence — useful
                # for eyeballing whether the model understands the scene
                # at all when the violence score is low.
                "top_class": self._id2label.get(top_idx, str(top_idx)),
                "top_class_prob": float(probs[top_idx]),
                **extras,
            },
        )]
