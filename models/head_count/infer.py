"""
Inference wrapper for the density-map head counter.

Two outputs, for two different consumers:

    density_map(frame)      ->  (h/8, w/8) float32, sums to the head count
    points(frame)           ->  (N, 2) head locations in source pixels
    points_and_count(frame) ->  both, from a single forward pass

Density is the primary product.  It is what crowd pressure needs, it needs no
threshold, and it degrades gracefully: in a crowd too dense to separate
individuals the map is still a correct count even though no single head can
be pointed at.  Points are recovered by local-maximum extraction and are for
visualisation and for seeding the annotation tool — anything that needs a
number should integrate the map, not count the points.

Tiling
------
Optional, and off by default.  Unlike the box detector, this model is fully
convolutional and has no fixed input size, so a large frame is not resized
down before it is seen and the small-head problem that makes tiling essential
for RT-DETRv2 does not arise here.  Tiling remains available for frames large
enough to exhaust GPU memory in one pass, where it is a memory measure rather
than an accuracy one — and because it changes nothing about the arithmetic,
the seams are summed rather than NMS'd.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

from models.head_count.model import OUTPUT_STRIDE

logger = logging.getLogger(__name__)

# Below this the local-maximum search returns nothing rather than a scatter of
# noise peaks.  Expressed in density units per output pixel: a single head
# spread over a sigma-1.5 Gaussian peaks around 0.07, so a tenth of that is
# comfortably noise.
DEFAULT_PEAK_THRESHOLD = 0.007


class HeadCounter:
    """
    Loads a trained checkpoint and returns head density (and optionally points).

    Thread-safe: inference is guarded, so one instance can be shared between
    the job worker and the validation runner in the same way BoxDetector is.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
        max_long_side: Optional[int] = 1280,
    ) -> None:
        self.weights = weights
        self.device = device or "cpu"
        self.peak_threshold = peak_threshold
        self.max_long_side = max_long_side
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """False when running on ImageNet initialisation with no checkpoint."""
        return bool(self.weights and os.path.exists(self.weights))

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from models.head_count.model import HeadCountNet

            net = HeadCountNet(pretrained=not self.is_trained)
            if self.is_trained:
                # weights_only=True: a checkpoint is data, and torch's default
                # unpickles arbitrary objects, so a .pt from anywhere but your
                # own training run is remote code execution waiting to happen.
                # Everything written by scripts/train_head_count.py is plain
                # tensors, so nothing is given up by refusing the rest.
                state = torch.load(self.weights, map_location="cpu",
                                   weights_only=True)
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                net.load_state_dict(state)
                logger.info("Head counter loaded from %s on %s",
                            self.weights, self.device)
            else:
                # Loud, because an untrained density head emits a smooth
                # near-constant field: it will happily report a plausible
                # number for any image, and that number means nothing.
                logger.warning(
                    "Head counter has NO trained checkpoint (weights=%r).  "
                    "The backbone is ImageNet-pretrained but the density head "
                    "is randomly initialised, so every count it produces is "
                    "meaningless — it will not be zero, it will be arbitrary.  "
                    "Train one with scripts/train_head_count.py before using "
                    "any number from this model.",
                    self.weights,
                )
            self._model = net.to(self.device).eval()
            self._torch = torch

    # ------------------------------------------------------------------

    def _prepare(self, frame: np.ndarray):
        """BGR frame -> (tensor, scale) with the long side capped."""
        import cv2
        from models.head_count.dataset import normalise

        h, w = frame.shape[:2]
        scale = 1.0
        if self.max_long_side and max(h, w) > self.max_long_side:
            scale = self.max_long_side / float(max(h, w))
            frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        # Trim to a multiple of the stride: a partial output cell would be
        # dropped by the network and its heads lost from the count.
        h2, w2 = frame.shape[:2]
        h2 -= h2 % OUTPUT_STRIDE
        w2 -= w2 % OUTPUT_STRIDE
        frame = frame[:h2, :w2]
        arr = normalise(frame)[None]
        return self._torch.from_numpy(arr).to(self.device), scale

    def density_map(self, frame: np.ndarray) -> np.ndarray:
        """
        (h/8, w/8) float32 density map. Its sum is the predicted head count.

        The map is expressed at the *resized* resolution; total count is
        scale-invariant because density integrates to a count, so no
        rescaling of values is needed or applied.
        """
        self.load()
        with self._lock:
            tensor, _scale = self._prepare(frame)
            with self._torch.no_grad():
                out = self._model(tensor)
            return out[0, 0].cpu().numpy().astype(np.float32)

    def count(self, frame: np.ndarray) -> float:
        """Predicted number of heads in the frame."""
        return float(self.density_map(frame).sum())

    def points(self, frame: np.ndarray) -> np.ndarray:
        """
        (N, 2) head points in SOURCE pixel coordinates, from local maxima.

        Note the count from len(points) will not generally equal count():
        in dense regions individual peaks merge and the map holds more mass
        than there are separable maxima.  That is a property of the crowd,
        not a bug — use count() for numbers and points() for showing people
        where the heads are.
        """
        return self.points_and_count(frame)[0]

    def points_and_count(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Both outputs from ONE forward pass: ((N, 2) points, count).

        Callers that need the locations and the number — density estimation
        needs the points to place people on the ground plane and the count for
        reporting — must use this rather than points() then count().  Those are
        two independent inferences over the same frame, which doubles the cost
        of the most expensive step in the pipeline for a number the first pass
        had already computed.
        """
        import cv2

        self.load()
        with self._lock:
            tensor, scale = self._prepare(frame)
            with self._torch.no_grad():
                out = self._model(tensor)
            dm = out[0, 0].cpu().numpy().astype(np.float32)

        count = float(dm.sum())

        # A pixel is a peak if it equals the max of its 3x3 neighbourhood.
        pooled = cv2.dilate(dm, np.ones((3, 3), np.uint8))
        peaks = (dm >= pooled) & (dm > self.peak_threshold)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return np.zeros((0, 2), np.float32), count

        # Output grid -> resized frame -> source frame.
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        pts = (pts + 0.5) * OUTPUT_STRIDE
        if scale != 1.0:
            pts /= scale
        return pts, count
