"""
Umbrella detection via the project's own fine-tuned RT-DETRv2.

Key:      umbrella_trained
UI Label: RT-DETRv2 (trained)

This is the only umbrella model here trained on umbrella data rather than
borrowing COCO's generic `umbrella` class. Weights live in
`ML Models/umbrella_trained/` (HuggingFace layout: `model.safetensors`,
`config.json`, `preprocessor_config.json`), single class `{0: "umbrella"}`,
640x640 input, 42.7 M parameters.

Reported validation from the bundled `history.json`, epoch 29 of 30:

    precision 0.769   recall 0.661   F1 0.711   (n_gt = 413)

Measured against the COCO-class baseline on test_videos/Umbrellas.mp4, it
finds roughly 2-3x more umbrellas at noticeably higher confidence (0.85-0.90
vs 0.63-0.75), which is what training on the actual target class buys.

**NMS is applied, and that is not optional.** RT-DETR is advertised as
NMS-free — Hungarian one-to-one matching during training is supposed to
make each object claim exactly one query. This checkpoint does not achieve
that: on a single frame at threshold 0.5 it emitted 250 boxes with a
maximum pairwise IoU of 0.98 and 4,585 pairs overlapping by more than half.
NMS collapses that to 92. Reporting the raw count would inflate umbrella
numbers by about 2.7x, so `nms_iou` defaults to on. Set it to None only if
you specifically want the raw query output.

That duplication is likely downstream of an incomplete save: the checkpoint
is missing `class_embed`/`bbox_embed` for decoder layers 1-5 (transformers
warns "checkpoint seems corrupted" on load and re-initialises them). Layer 0
carries the weights that matter, and detections are good, but the auxiliary
heads that sharpen one-to-one assignment are gone. Re-exporting the full
state dict would likely remove the need for NMS here.
"""

import json
import os

from models.base import BaseModelWrapper, Detection
from models._tracker import IoUTracker
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Searched in order; first hit wins. The ML Models copy is the canonical one.
_MODEL_DIRS = [
    os.path.join(PROJECT_ROOT, "ML Models", "umbrella_trained"),
    os.path.join(PROJECT_ROOT, "weights", "umbrella_trained"),
    os.path.join(PROJECT_ROOT, "umbrella_trained"),
]


def find_trained_dir() -> str | None:
    """Directory holding the fine-tuned checkpoint, or None if absent."""
    for d in _MODEL_DIRS:
        if os.path.exists(os.path.join(d, "model.safetensors")) and \
           os.path.exists(os.path.join(d, "config.json")):
            return d
    return None


class TrainedUmbrellaDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "umbrella_trained"
    gpu_accelerated = True

    def __init__(self, model_dir: str = None, conf_threshold: float = 0.5,
                 nms_iou: float | None = 0.5,
                 min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
                 track: bool = True, iou_match_threshold: float = 0.3,
                 device=None):
        """
        conf_threshold: 0.5. RT-DETR emits a fixed 300 queries per frame with
            no objectness gate, so a low threshold does not mean "more
            sensitive" — it means the tail of the query set leaks through.
        nms_iou: IoU for duplicate suppression, or None to disable. See the
            module docstring: this checkpoint genuinely needs it.
        """
        super().__init__(device=device)
        self.model_dir = model_dir or find_trained_dir()
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.min_area_frac = min_area_frac
        self.track = track
        self._proc = None
        self._val_metrics = None
        self._tracker = IoUTracker(iou_threshold=iou_match_threshold, max_age=30)

    def load(self):
        import torch  # noqa: F401  (ensures a clear error if torch is missing)
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if not self.model_dir:
            raise FileNotFoundError(
                "umbrella_trained: no fine-tuned checkpoint found. Expected "
                "model.safetensors + config.json in one of: "
                + ", ".join(_MODEL_DIRS)
            )

        self._proc = AutoImageProcessor.from_pretrained(self.model_dir)
        self._model = AutoModelForObjectDetection.from_pretrained(self.model_dir)
        self._model.to(self.device).eval()

        labels = getattr(self._model.config, "id2label", {}) or {}
        if not any(str(v).lower() == "umbrella" for v in labels.values()):
            raise ValueError(
                f"{self.model_dir} does not look like an umbrella model "
                f"(id2label={labels}). Point model_dir at the right checkpoint."
            )

        # Surface the training result so a run can be traced back to the
        # checkpoint that produced it, rather than just "the trained one".
        history = os.path.join(self.model_dir, "history.json")
        if os.path.exists(history):
            try:
                with open(history, encoding="utf-8") as f:
                    entries = json.load(f)
                scored = [e for e in entries if isinstance(e, dict) and "val" in e]
                if scored:
                    m = scored[-1]["val"].get("umbrella", {})
                    p, r = m.get("precision"), m.get("recall")
                    if p and r:
                        self._val_metrics = {
                            "epoch": scored[-1].get("epoch"),
                            "precision": p,
                            "recall": r,
                            "f1": round(2 * p * r / (p + r), 3),
                        }
                        print(f"[{self.name}] fine-tuned checkpoint, epoch "
                              f"{self._val_metrics['epoch']}: precision "
                              f"{p:.3f} recall {r:.3f} F1 {self._val_metrics['f1']:.3f}")
            except (json.JSONDecodeError, OSError, TypeError):
                pass  # history is a convenience, never a requirement

        self._tracker.reset()

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        import cv2
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        inputs = self._proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # post_process mutates the outputs it is handed, so it is called
        # exactly once per forward pass. Calling it twice on one `outputs`
        # silently returns different counts the second time.
        result = self._proc.post_process_object_detection(
            outputs,
            threshold=self.conf_threshold,
            target_sizes=torch.tensor([[h, w]]).to(self.device),
        )[0]

        boxes, scores = result["boxes"], result["scores"]
        if len(scores) == 0:
            return []

        if self.nms_iou is not None:
            from torchvision.ops import nms
            keep = nms(boxes, scores, self.nms_iou)
            boxes, scores = boxes[keep], scores[keep]

        boxes = boxes.cpu().tolist()
        scores = scores.cpu().tolist()

        ids = self._tracker.update(boxes, frame_index) if self.track else None

        extra = {
            "architecture": "RT-DETRv2 (fine-tuned)",
            "finetuned": True,
            "nms_iou": self.nms_iou,
        }
        if self._val_metrics:
            extra["val_f1"] = self._val_metrics["f1"]

        return emit_umbrellas(
            boxes, scores, ids, frame.shape, self.name, frame_index,
            timestamp_sec, min_area_frac=self.min_area_frac,
            extra_fields=extra,
        )
