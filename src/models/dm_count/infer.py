"""
Inference wrapper for the DM-Count density-map head counter.

Ported from ``Ujwal/__CMS__``: the predictor chain reproduces
``DM-Count-Kit/zoo/predictors.py::DMCountPredictor`` exactly (resize band,
padding rule, ImageNet normalisation, stride-8 output crop) plus the peak
extraction from ``__Dashboard__/core/dmcount_backend.py`` and the CPU
long-side cap of ``__Dashboard__/core/model_kit.py::Prescaled``.

    predict(frame_bgr) -> DensityMap

DensityMap carries three things at once:
  - points  : (N, 3) head locations in SOURCE pixels (x, y, density value),
              extracted as local maxima of the density map — this is what
              the tracker consumes,
  - count   : float, the integral of the density map (the model's own count;
              note it is NOT len(points)),
  - density : the native coarse grid (h/8, w/8), kept at model resolution
              with its grid-to-source scale so overlays can be rendered
              without anyone re-deriving block sizes.

Count-vs-prescale caveat, inherited deliberately from the source workspace:
when ``max_long_side`` forces a downscale, the count is computed on the
downscaled frame. The source workspace accepted this as its primary CPU
speed knob; we keep the behaviour (and expose the knob) rather than silently
changing the numbers a checkpoint was validated to produce.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

#: Default checkpoint, relative to PROJECT_ROOT. Searched in order.
WEIGHT_SEARCH_PATHS = (
    os.path.join("src", "models", "dm_count", "weights", "model_sh_A.pth"),
    os.path.join("ML Models", "dm_count", "model_sh_A.pth"),
    os.path.join("ML Models", "DM-Count_pretrained_models", "model_sh_A.pth"),
)

# ImageNet stats — DM-Count's VGG19 backbone was trained under these. Not
# interchangeable with any other pair (see zoo/base.py's warning).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: ShanghaiTech A checkpoints were trained on native-resolution images, so no
#: shorter-side resize band applies (zoo/registry.py pins min_size=max_size=None).
DEFAULT_MIN_SIZE: Optional[int] = None
DEFAULT_MAX_SIZE: Optional[int] = None


def find_weights() -> Optional[str]:
    """Absolute path of the first DM-Count checkpoint on disk, or None."""
    for rel in WEIGHT_SEARCH_PATHS:
        candidate = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(candidate):
            return candidate
    return None


def _subcell_shift(density: np.ndarray, y: int, x: int, axis: int,
                   limit: int) -> float:
    """Parabolic sub-cell offset along one axis, clipped to [-0.5, 0.5].

    Fits a parabola through (v-1, v, v+1) and returns its vertex offset from
    the centre sample. Border cells and non-concave neighbourhoods get 0.
    """
    if axis == 0:
        lo = density[y - 1, x] if y > 0 else None
        hi = density[y + 1, x] if y + 1 < limit else None
    else:
        lo = density[y, x - 1] if x > 0 else None
        hi = density[y, x + 1] if x + 1 < limit else None
    if lo is None or hi is None:
        return 0.0
    denom = float(2.0 * density[y, x] - lo - hi)
    if denom <= 1e-9:
        return 0.0
    shift = 0.5 * (float(hi) - float(lo)) / denom
    return float(np.clip(shift, -0.5, 0.5))


@dataclass
class DensityMap:
    """One frame's counter output (see module docstring)."""
    points: np.ndarray          # (N, 3) x, y, density-value in source pixels
    count: float                # density-map integral
    density: np.ndarray         # native coarse (h', w') grid
    grid_scale: tuple[float, float]   # source-px per grid cell (sx, sy)
    elapsed_s: float
    input_hw: tuple[int, int]         # padded network input
    source_hw: tuple[int, int]        # original frame


class DMCountCounter:
    """
    Loads DM-Count once and answers per-frame head points + counts.

    Thread-safe in the same way models/head_count/infer.py::HeadCounter is:
    inference is guarded by a lock so one instance can be shared.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        peak_min_distance_px: int = 6,
        peak_value_thresh: float = 0.06,
        max_long_side: int = 960,
        min_size: Optional[int] = DEFAULT_MIN_SIZE,
        max_size: Optional[int] = DEFAULT_MAX_SIZE,
    ) -> None:
        self.weights = weights or find_weights()
        self.device = device or "cpu"
        self.peak_min_distance_px = peak_min_distance_px
        self.peak_value_thresh = peak_value_thresh
        # Primary CPU speed knob (source: dense_flow_config.json used 960).
        self.max_long_side = max_long_side
        self.min_size = min_size
        self.max_size = max_size
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return bool(self.weights and os.path.exists(self.weights))

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available:
            raise FileNotFoundError(
                "DM-Count checkpoint not found. Looked for: "
                + ", ".join(WEIGHT_SEARCH_PATHS)
            )
        with self._lock:
            if self._model is not None:
                return
            import torch
            from models.dm_count.model import vgg19

            model = vgg19(pretrained=False)
            state = torch.load(self.weights, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            # strict=True on purpose: these are published checkpoints whose key
            # set exactly matches the vendored architecture. A silent partial
            # load would produce confident nonsense counts.
            model.load_state_dict(state, strict=True)
            self._model = model.to(self.device).eval()
            self._torch = torch
            logger.info(
                "DM-Count counter loaded on %s (weights=%s)",
                self.device, self.weights,
            )

    # ------------------------------------------------------------------

    def _extract_peaks(self, density: np.ndarray) -> np.ndarray:
        """Local maxima of the density map above the value floor.

        Returns (M, 3) rows of (grid_x, grid_y, value). scipy's
        maximum_filter is the source implementation; cv2.dilate gives an
        identical result when scipy is absent.

        Each peak is refined to sub-cell precision by parabolic interpolation
        through its axis neighbours — the standard keypoint trick. Without it,
        peak coordinates snap to whole density cells, which are ~8 source px
        wide here, and every tracked velocity inherits that quantisation: a
        person drifting half a cell reads as repeated 8 px jumps. The
        refinement is what brings tracked speeds down to pedestrian scale;
        the source workspace lived with the quantised version and simply set
        its speed thresholds around it.
        """
        k = max(1, int(self.peak_min_distance_px))
        try:
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(density, size=k)
        except ImportError:
            kernel = np.ones((k, k), dtype=np.float32)
            local_max = cv2.dilate(density, kernel)
        mask = (density == local_max) & (density > self.peak_value_thresh)
        ys, xs = np.nonzero(mask)

        h, w = density.shape
        pts = []
        for gx, gy in zip(xs.astype(np.float64), ys.astype(np.float64)):
            v = float(density[int(gy), int(gx)])
            dx = _subcell_shift(density, int(gy), int(gx), 1, w)
            dy = _subcell_shift(density, int(gy), int(gx), 0, h)
            pts.append((gx + dx, gy + dy, v))
        return np.array(pts, dtype=np.float32).reshape(-1, 3)

    def predict(self, frame_bgr: np.ndarray) -> DensityMap:
        self.load()
        h0, w0 = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Prescale cap (source: Prescaled in core/model_kit.py): shrink the
        # long side before inference; peaks get mapped back afterwards.
        longest = max(h0, w0)
        ratio = 1.0
        if self.max_long_side and longest > self.max_long_side:
            ratio = self.max_long_side / float(longest)
        if ratio != 1.0:
            rgb = cv2.resize(
                rgb,
                (max(1, int(round(w0 * ratio))), max(1, int(round(h0 * ratio)))),
                interpolation=cv2.INTER_AREA,
            )
        hr, wr = rgb.shape[:2]

        # Shorter-side band (no-op for the ShanghaiTech-A checkpoint).
        resized, _ = self._resize_shorter_side(rgb, self.min_size, self.max_size)
        hp, wp = resized.shape[:2]

        # Zero-pad to a multiple of 16 so every pooling stage keeps whole
        # cells; the pad margin is cropped out of the density map below.
        ph = ((hp + 15) // 16) * 16
        pw = ((wp + 15) // 16) * 16
        if (ph, pw) != (hp, wp):
            canvas = np.zeros((ph, pw, 3), dtype=resized.dtype)
            canvas[:hp, :wp] = resized
            padded = canvas
        else:
            padded = resized

        t0 = time.perf_counter()
        with self._lock:
            x = self._to_tensor(padded)
            with self._torch.no_grad():
                density, _ = self._model(x)
            vh, vw = hp // 8, wp // 8
            # Drop the padded margin before anything reads the grid: the ReLU
            # head emits small positive values along the seam.
            dens = density[0, 0, :vh, :vw].float().cpu().numpy()
        elapsed = time.perf_counter() - t0

        # Grid cell -> resized-frame px, then back through the prescale.
        sx_small = wr / float(max(1, vw))
        sy_small = hr / float(max(1, vh))
        sx_src = sx_small / ratio
        sy_src = sy_small / ratio

        peaks = self._extract_peaks(dens)
        if len(peaks):
            pts = peaks.copy()
            pts[:, 0] *= sx_src
            pts[:, 1] *= sy_src
        else:
            pts = np.zeros((0, 3), dtype=np.float32)

        return DensityMap(
            points=pts.astype(np.float32),
            count=float(dens.sum(dtype=np.float64)),
            density=dens,
            grid_scale=(float(sx_src), float(sy_src)),
            elapsed_s=elapsed,
            input_hw=(ph, pw),
            source_hw=(h0, w0),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resize_shorter_side(image_rgb: np.ndarray, min_size: Optional[int],
                             max_size: Optional[int]):
        """Clamp the SHORTER side into [min_size, max_size].

        Upstream DM-Count baked this into offline preprocessing; skipping it
        is the usual reason a reimplementation miscounts on large images.
        No-op by default because ShanghaiTech A was trained unresized.
        """
        if min_size is None and max_size is None:
            return image_rgb, 1.0
        h, w = image_rgb.shape[:2]
        short = min(h, w)
        ratio = 1.0
        if min_size is not None and short < min_size:
            ratio = min_size / float(short)
        elif max_size is not None and short > max_size:
            ratio = max_size / float(short)
        if ratio == 1.0:
            return image_rgb, 1.0
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        interp = cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA
        return cv2.resize(image_rgb, (new_w, new_h), interpolation=interp), ratio

    def _to_tensor(self, image_rgb: np.ndarray):
        """uint8 HWC RGB -> normalised (1, 3, H, W) tensor on self.device."""
        import torch

        x = torch.from_numpy(np.ascontiguousarray(image_rgb)).float().div_(255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        return ((x - mean) / std).to(self.device)
