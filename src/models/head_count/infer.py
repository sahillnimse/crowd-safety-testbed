"""
Inference wrapper for the APGCC point-based head counter.

    points(frame)            ->  (N, 2) head locations in source pixels
    scores(frame)             -> (N,) confidence per point, same order
    points_and_count(frame)  ->  ((N, 2) points, count) from one forward pass
    count(frame)             ->  int, len(points above threshold)

Thread-safe: inference is guarded, so one instance can be shared between
the job worker and crowd_motion_monitor in the same way BoxDetector is.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

from models.head_count.model import (
    DEFAULT_SCORE_THRESHOLD,
    INPUT_DIVISOR,
    points_and_scores,
    preprocess,
)

logger = logging.getLogger(__name__)


class HeadCounter:
    """
    Loads APGCC and returns head points (and count), thresholded by score.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        config: str = "shha",
    ) -> None:
        self.weights = weights
        self.device = device or "cpu"
        self.score_threshold = score_threshold
        self.config = config
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return bool(self.weights and os.path.exists(self.weights))

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from models.head_count.model import load_model

            net, info = load_model(
                weights=self.weights, device=self.device, config=self.config,
            )
            self._model = net
            self._torch = torch
            logger.info(
                "APGCC head counter loaded on %s (weights=%s, tensors=%d)",
                self.device, self.weights, info["n_tensors"],
            )

    # ------------------------------------------------------------------

    def _prepare(self, frame: np.ndarray):
        """BGR frame -> ImageNet-normalised tensor, padded to a multiple of 16."""
        import cv2

        h, w = frame.shape[:2]
        pad_h = (-h) % INPUT_DIVISOR
        pad_w = (-w) % INPUT_DIVISOR
        if pad_h or pad_w:
            frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w,
                                        cv2.BORDER_REPLICATE)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device)
        tensor = preprocess(tensor)
        return tensor

    def points_and_count(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Both outputs from ONE forward pass: ((N, 2) points, count)."""
        self.load()
        with self._lock:
            tensor = self._prepare(frame)
            with self._torch.no_grad():
                pts, scores = points_and_scores(self._model, tensor)
            keep = scores > self.score_threshold
            pts = pts[keep].detach().cpu().numpy().astype(np.float32)
            n = int(keep.sum().item())
        return pts, float(n)

    def points(self, frame: np.ndarray) -> np.ndarray:
        """(N, 2) head points in source pixel coordinates, above score_threshold."""
        return self.points_and_count(frame)[0]

    def points_with_scores(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        (N, 2) points and (N,) scores, both thresholded — used by
        crowd_motion_monitor's RT-DETRv2/APGCC fusion, which needs the score
        to log/inspect what it's turning into a synthetic box.
        """
        self.load()
        with self._lock:
            tensor = self._prepare(frame)
            with self._torch.no_grad():
                pts, scores = points_and_scores(self._model, tensor)
            keep = scores > self.score_threshold
            pts = pts[keep].detach().cpu().numpy().astype(np.float32)
            scores = scores[keep].detach().cpu().numpy().astype(np.float32)
        return pts, scores

    def count(self, frame: np.ndarray) -> int:
        """Predicted number of heads in the frame (points above threshold)."""
        return int(self.points_and_count(frame)[1])
