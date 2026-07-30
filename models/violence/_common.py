"""
Shared clip preprocessing and violence scoring for the video classifiers.

Three problems this module exists to fix, all of which were present in
every one of the seven wrappers and each of which alone was enough to make
their output meaningless:

1. **Colour order.** OpenCV decodes BGR. Every backbone here is pretrained
   on RGB. The wrappers fed BGR straight in, so red and blue were swapped
   in every clip the models ever saw.

2. **Normalization.** The wrappers divided by 255 and stopped. Kinetics
   pretrained models expect channel normalization on top of that
   (mean 0.45, std 0.225); without it the input distribution is nothing
   like anything the network was trained on.

3. **The label mapping.** The wrappers did
   `label = "violence" if argmax == 1 else "non_violence"` against a
   **400-class Kinetics head**. Index 1 is an arbitrary Kinetics class, so
   the "violence" label was uncorrelated with violence, and a >0.5
   confidence gate over 400 softmax classes meant they almost never fired
   at all.

The fix for (3) is `violence_score()`. A Kinetics-pretrained model has
genuinely learned to recognize fighting — Kinetics-400 contains
`punching person (boxing)`, `wrestling`, `slapping`, `headbutting`,
`side kick` and friends. Summing the probability mass over that class
subset gives a real zero-shot violence score with no fine-tuning at all.
It is weaker than a model fine-tuned on RWF-2000, and it will fire on
sparring and martial arts, but it is a true signal rather than noise.

When a fine-tuned binary checkpoint *is* supplied, the wrappers pass
`num_classes=2` and index 1 is used directly, as intended.
"""

import json
import os
from typing import Optional, Sequence

import cv2
import numpy as np

# Kinetics normalization constants used by pytorchvideo / SlowFast / X3D.
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)

# Canonical Kinetics-400 class names in model-output order (alphabetical:
# index 0 == "abseiling"), bundled in-repo so scoring never depends on a
# network fetch at run time. Regenerate from any Kinetics-400 checkpoint's
# id2label if it ever needs updating.
_CLASSNAMES_PATH = os.path.join(os.path.dirname(__file__), "kinetics_400_classes.json")

# Kinetics-400 classes that depict interpersonal physical aggression.
# Matched as substrings against the real class names, so exact punctuation
# and parenthetical suffixes in the published list don't matter.
VIOLENCE_CLASS_PATTERNS = (
    "punching person",
    "wrestling",
    "slapping",
    "headbutting",
    "side kick",
    "drop kicking",
    "high kick",
    "sword fighting",
    "capoeira",
)

# Superstrings of the patterns above that are *not* interpersonal violence
# and would otherwise be swept in.
VIOLENCE_CLASS_EXCLUDES = (
    "arm wrestling",
    "punching bag",
    "punching person (boxing) bag",
)


def load_kinetics_id2label() -> dict[int, str]:
    """Kinetics-400 index -> class name, read from the bundled list."""
    with open(_CLASSNAMES_PATH, encoding="utf-8") as f:
        names = json.load(f)
    return {i: str(name) for i, name in enumerate(names)}


def violence_class_indices(id2label: dict[int, str]) -> set[int]:
    """Indices of the classes counted as violence, for any label mapping.

    Works for the Kinetics list and equally for a HuggingFace model's
    `config.id2label`, so VideoMAE and the pytorchvideo models score the
    same way.
    """
    indices = set()
    for idx, name in id2label.items():
        lowered = str(name).lower()
        if any(bad in lowered for bad in VIOLENCE_CLASS_EXCLUDES):
            continue
        if any(pat in lowered for pat in VIOLENCE_CLASS_PATTERNS):
            indices.add(int(idx))
    return indices


def violence_score(probs, violence_indices: Optional[set[int]],
                   binary_head: bool) -> tuple[float, dict]:
    """Probability that the clip contains violence, plus diagnostic extras.

    `binary_head=True`  -> a fine-tuned violence/non-violence head; index 1.
    `binary_head=False` -> a Kinetics head; sum the violence-class mass.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)

    if binary_head:
        if probs.shape[0] < 2:
            raise ValueError(f"binary head expected >=2 classes, got {probs.shape[0]}")
        return float(probs[1]), {"scoring": "finetuned_binary"}

    if not violence_indices:
        raise ValueError(
            "No Kinetics violence classes matched the model's label set. "
            "Either supply a fine-tuned binary checkpoint via weights_path, "
            "or check that the model's id2label is a Kinetics-400 mapping."
        )

    idxs = sorted(i for i in violence_indices if i < probs.shape[0])
    score = float(probs[idxs].sum())
    top_i = int(idxs[int(np.argmax(probs[idxs]))]) if idxs else -1
    return score, {
        "scoring": "kinetics_zeroshot",
        "violence_classes_summed": len(idxs),
        "top_violence_class_index": top_i,
    }


def infer_num_classes(state_dict, default: int = 2) -> int:
    """Read the class count off a checkpoint's final linear layer.

    Lets a fine-tuned binary head be detected rather than assumed —
    assuming binary is how "index 1 == violence" ended up being applied to
    a 400-class Kinetics head in the first place.
    """
    for key in reversed(list(state_dict.keys())):
        tensor = state_dict[key]
        if key.endswith("weight") and getattr(tensor, "ndim", 0) >= 1:
            return int(tensor.shape[0])
    return default


def sample_frame_indices(available: int, clip_len: int) -> np.ndarray:
    """Uniformly pick `clip_len` frame indices from `available` frames."""
    return np.linspace(0, max(available - 1, 0), clip_len).round().astype(int)


class PersonROI:
    """Crops each clip to the region containing people before classification.

    This is the difference between these models working on surveillance
    footage and not working at all. Measured on this testbed's own CCTV
    clip, X3D scored a known fight at **0.001-0.007** on the full frame and
    **0.68-0.85** ("wrestling", "punching person (boxing)") once cropped to
    the people — while a quiet window stayed at 0.001, so the crop finds
    violence rather than inflating everything.

    Two compounding reasons the uncropped frame fails:

      - *Position.* Kinetics preprocessing centre-crops, which on a 16:9
        frame keeps only the middle ~56% of the width. Surveillance action
        is wherever it happens to be; here 3 of the 4 fighters sat outside
        that crop entirely.
      - *Scale.* People occupied ~4.5% of frame area — about 47x47 px after
        downscaling to the network's input. Kinetics models are trained on
        clips where the action fills the frame, so a body that small
        carries almost no signal.

    The ROI is the padded union of person boxes over the clip, smoothed
    across calls so the framing doesn't jitter between inferences. With no
    people detected it returns None and the caller uses the whole frame.
    """

    def __init__(self, conf: float = 0.3, pad: float = 0.25,
                 sample_frames: int = 8, smoothing: float = 0.6,
                 min_area_frac: float = 0.01, weights: str = "yolov8n.pt"):
        self.conf = conf
        self.pad = pad
        self.sample_frames = sample_frames
        self.smoothing = smoothing        # 0 = snap instantly, ->1 = very sticky
        self.min_area_frac = min_area_frac
        self.weights = weights
        self._detector = None
        self._last: Optional[tuple] = None

    def load(self, device: Optional[str] = None):
        from ultralytics import YOLO
        self._detector = YOLO(self.weights)
        if device:
            self._detector.to(device)
        self._last = None

    def reset(self):
        self._last = None

    def compute(self, frames, device: Optional[str] = None) -> Optional[tuple]:
        """-> (x1, y1, x2, y2) in source-frame pixels, or None for full frame."""
        if self._detector is None or not frames:
            return None

        h, w = frames[0].shape[:2]
        step = max(1, len(frames) // self.sample_frames)
        xs1, ys1, xs2, ys2 = [], [], [], []

        for frame in frames[::step]:
            result = self._detector.predict(frame, conf=self.conf, classes=[0],
                                            device=device, verbose=False)
            if not result:
                continue
            for box in result[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)

        if not xs1:
            # Nobody visible: keep the previous ROI rather than snapping back
            # to full frame, which would make the score jump for one clip.
            return self._last

        box = (min(xs1), min(ys1), max(xs2), max(ys2))
        box = self._pad_to_square(box, w, h)

        if self._last is not None:
            a = self.smoothing
            box = tuple(a * old + (1 - a) * new for old, new in zip(self._last, box))

        self._last = box
        return tuple(int(round(v)) for v in box)

    def _pad_to_square(self, box, w: int, h: int) -> tuple:
        """Pad the union box out to a square, so cropping doesn't distort the
        aspect ratio that the network's spatial filters expect."""
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2

        side = max(bw, bh) * (1 + self.pad)
        # Don't zoom so far that the crop is mostly a single body — some
        # scene context helps the classifier.
        side = max(side, (self.min_area_frac * w * h) ** 0.5)
        side = min(side, float(min(w, h)))

        x1 = min(max(0.0, cx - side / 2), w - side)
        y1 = min(max(0.0, cy - side / 2), h - side)
        return (x1, y1, x1 + side, y1 + side)


def crop_to_roi(frame: np.ndarray, roi: Optional[tuple]) -> np.ndarray:
    if roi is None:
        return frame
    x1, y1, x2, y2 = (int(v) for v in roi)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return frame
    return frame[y1:y2, x1:x2]


def preprocess_clip(frames: Sequence[np.ndarray], clip_len: int, size: int,
                    mean: Sequence[float] = KINETICS_MEAN,
                    std: Sequence[float] = KINETICS_STD,
                    resize_shortest: bool = True,
                    roi: Optional[tuple] = None) -> "np.ndarray":
    """BGR frame list -> normalized (C, T, H, W) float32 array.

    `roi` crops each frame to a region of interest first (see PersonROI) —
    on surveillance footage this is what makes the difference between a
    usable score and nothing at all.

    Without an ROI the short side is resized and the frame centre-cropped
    rather than squashed to a square, since squashing distorts the aspect
    ratio of every body in frame. With an ROI the crop is already square,
    so it's resized directly and nothing is discarded.
    """
    idxs = sample_frame_indices(len(frames), clip_len)
    sampled = [crop_to_roi(frames[i], roi) for i in idxs]

    out = []
    for f in sampled:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if resize_shortest and abs(h - w) > 2:
            scale = size / min(h, w)
            resized = cv2.resize(rgb, (max(size, int(round(w * scale))),
                                       max(size, int(round(h * scale)))))
            rh, rw = resized.shape[:2]
            y0, x0 = (rh - size) // 2, (rw - size) // 2
            rgb = resized[y0:y0 + size, x0:x0 + size]
        else:
            rgb = cv2.resize(rgb, (size, size))
        out.append(rgb)

    arr = np.stack(out).astype(np.float32) / 255.0        # (T, H, W, C)
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(3, 0, 1, 2))  # (C, T, H, W)


def clip_to_tensor(frames: Sequence[np.ndarray], clip_len: int, size: int,
                   device: str, mean: Sequence[float] = KINETICS_MEAN,
                   std: Sequence[float] = KINETICS_STD,
                   roi: Optional[tuple] = None):
    """`preprocess_clip` plus a batch dimension, on the target device."""
    import torch
    arr = preprocess_clip(frames, clip_len, size, mean=mean, std=std, roi=roi)
    return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, T, H, W)


class ViolenceScoringMixin:
    """Mixin holding the head-resolution and scoring every wrapper shares.

    Wrappers combine this with BaseModelWrapper; it deliberately doesn't
    subclass it so each wrapper keeps its own `load()`/`predict()` shape.
    """

    #: set by the wrapper's load() once it knows the head it ended up with
    _binary_head: bool = False
    _violence_indices: Optional[set[int]] = None

    #: True when this wrapper is running with a randomly-initialized
    #: classification head because no fine-tuned checkpoint was supplied.
    _untrained: bool = False
    _untrained_reason: str = ""
    _roi: Optional[PersonROI] = None

    def _init_roi(self, use_person_roi: bool):
        """Call from __init__. On by default: full-frame scoring simply does
        not work on wide surveillance shots (see PersonROI)."""
        self.use_person_roi = use_person_roi
        self._roi = PersonROI() if use_person_roi else None

    def _load_roi(self):
        """Call from load(), after the main model is on the device."""
        if self._roi is not None:
            self._roi.load(self.device)

    def _clip_roi(self, clip_frames) -> Optional[tuple]:
        if self._roi is None:
            return None
        return self._roi.compute(clip_frames, device=self.device)

    def _resolve_head(self, num_classes: int,
                      id2label: Optional[dict[int, str]] = None) -> None:
        """Decide how this model's output maps onto a violence score."""
        self._binary_head = num_classes == 2
        if self._binary_head:
            self._violence_indices = None
            return

        labels = id2label or load_kinetics_id2label()
        self._violence_indices = violence_class_indices(labels)
        matched = sorted(labels[i] for i in self._violence_indices if i in labels)
        # ASCII only: Windows consoles default to cp1252 and mangle non-ASCII.
        print(f"[{self.name}] No fine-tuned binary head - scoring violence "
              f"zero-shot from {len(matched)} Kinetics classes: {matched}")

    def _mark_untrained(self, reason: str) -> None:
        """Declare that this model's head is randomly initialized.

        Call from load() when no fine-tuned checkpoint was found. The model
        still runs and still emits rows, but its output is labelled so it
        can never be counted as a real detection.
        """
        self._untrained = True
        self._untrained_reason = reason
        print(f"[{self.name}] WARNING: {reason} Output is labelled "
              f"'violence_untrained' and excluded from event counts.")

    def _score(self, probs) -> tuple[str, float, dict]:
        """probs -> (label, confidence, extras) using the resolved head."""
        score, extras = violence_score(probs, self._violence_indices, self._binary_head)

        if self._untrained:
            # A randomly-initialized head does not respond to its input: on
            # this repo's test footage C3D returned 0.510 for every single
            # clip (std 0.000) and TSM 0.494-0.564, both sitting above the
            # 0.5 threshold. Labelling that "violence" made them look like
            # the most sensitive detectors in the comparison tables while
            # carrying no information at all. `violence_untrained` is
            # deliberately absent from POSITIVE_LABELS, so these rows are
            # still written and inspectable but never counted as events.
            return "violence_untrained", score, {
                **extras,
                "scoring": "untrained_random_head",
                "untrained_reason": self._untrained_reason,
            }

        label = "violence" if score >= self.conf_threshold else "non_violence"
        return label, score, extras
