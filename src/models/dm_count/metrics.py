"""
Frame-level crowd risk metrics for tracked head points.

Port of ``Ujwal/__CMS__/__Dashboard__/core/metrics.py::MetricsEngine`` with
two deliberate substitutions, made per-function rather than wholesale:

KEPT FROM THE SOURCE WORKSPACE
  * density / velocity_variance / pressure = density x Var(v)  (Helbing)
  * magnitude-weighted directional entropy over an 8-bin heading histogram
  * counter-flow as a percentage opposing the dominant direction
  * cell_density / cell_pressure grids (8x12 by default)
  * occupancy ratio against a safe capacity

TAKEN FROM CROWD-SAFETY-TESTBED INSTEAD (its versions are stronger)
  * stop-and-go: the testbed's negative temporal autocorrelation of mean
    speed at short lags replaces the source's FFT band-power ratio. Both
    detect periodic halting; autocorrelation is stable on the short buffers
    a per-frame series actually has, doesn't need windowing, and matches
    what models/crowd_flow/crowd_metrics.py already reports elsewhere in
    this project.
  * oscillation: same reasoning, applied to the mean velocity VECTOR
    (direction reversals), which the source's scalar-speed FFT cannot see.

CHANGED INPUT DOMAIN
  The source computed entropy/counter-flow/dominant-direction over raw
  optical-flow pixels. Here they are computed over TRACKED head-point
  velocities: flow pixels include background and camera shake, while tracks
  are actual people. Divergence stays a flow-field quantity (it is a spatial
  property of the field, not of individuals).

Units are whatever the caller feeds in: px/frame + heads/frame when the
camera is uncalibrated (the default), m/s + heads/m^2 when the monitor was
given a homography. ``speed_unit``/``density_unit`` travel with every row so
downstream code never has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class FrameMetrics:
    """All frame-level metrics, one instance per processed frame."""
    head_count: int
    density: float
    density_unit: str                 # "heads/frame" | "heads/m2"
    mean_speed: float
    speed_unit: str                   # "px/frame" | "m/s"
    velocity_variance: float
    pressure: float                   # density x velocity_variance
    divergence: float                 # mean of the flow-field divergence
    directional_entropy: float        # [0, 3] bits over 8 bins
    dominant_dir_deg: float | None    # None when nothing is moving
    counter_flow_pct: float           # [0, 100]
    stop_and_go: float                # [0, 1] - higher = periodic halting
    oscillation: float                # [0, 1] - direction-reversal energy
    occupancy_ratio: float | None     # count / safe_capacity, None if unset
    n_moving_tracks: int
    n_stopped_tracks: int
    cell_density: np.ndarray = field(default=None)   # (gh, gw) share per cell
    cell_pressure: np.ndarray = field(default=None)  # (gh, gw)


class CrowdMetricsEngine:
    def __init__(
        self,
        grid: tuple[int, int] = (8, 12),
        safe_capacity: int | None = None,
        calibrated: bool = False,
        stop_go_lags: list[int] | None = None,
        roi_area_m2: float | None = None,
    ):
        self.grid = grid
        self.safe_capacity = safe_capacity
        self.calibrated = calibrated
        # When a homography gives the view's ground footprint, density is
        # heads/m^2 and pressure lands in proper Helbing s^-2-ish units;
        # otherwise both are per-frame quantities.
        self.roi_area_m2 = roi_area_m2
        self.stop_go_lags = stop_go_lags or [5, 10, 15]
        self.speed_hist: deque[float] = deque(maxlen=64)
        self.vec_hist: deque[tuple[float, float]] = deque(maxlen=64)

    # ------------------------------------------------------------------

    def _stop_and_go(self) -> float:
        """Max negative autocorrelation of mean speed at short lags.

        Kept from the testbed's crowd_metrics.py: periodic stop-and-go waves
        make speed at lag k anti-correlated with current speed. Requires a
        minimum buffer so early frames report 0 instead of noise.
        """
        buf = self.speed_hist
        if len(buf) < max(self.stop_go_lags) + 5:
            return 0.0
        x = np.asarray(buf, dtype=np.float64)
        variance = float(np.var(x))
        if variance <= 1e-9:
            return 0.0
        centred = x - x.mean()
        scores = []
        for lag in self.stop_go_lags:
            ac = float(np.sum(centred[:-lag] * centred[lag:]) / (
                (len(x) - lag) * variance))
            scores.append(max(0.0, -ac))   # negative autocorr -> positive score
        return float(np.clip(max(scores), 0.0, 1.0))

    def _oscillation(self) -> float:
        """Negative autocorrelation of the mean velocity vector."""
        buf = self.vec_hist
        if len(buf) < max(self.stop_go_lags) + 5:
            return 0.0
        arr = np.asarray(buf, dtype=np.float64)          # (n, 2)
        centred = arr - arr.mean(axis=0)
        denom = float((centred ** 2).sum(axis=1).mean())
        if denom <= 1e-9:
            return 0.0
        scores = []
        for lag in self.stop_go_lags:
            ac = float((centred[:-lag] * centred[lag:]).sum() / ((len(arr) - lag) * denom))
            scores.append(max(0.0, -ac))
        return float(np.clip(max(scores), 0.0, 1.0))

    @staticmethod
    def _entropy(vxs: np.ndarray, vys: np.ndarray,
                 weights: np.ndarray) -> tuple[float, float | None]:
        """Magnitude-weighted heading entropy over 8 bins + circular-mean dir."""
        mag = np.hypot(vxs, vys)
        moving = mag > 1e-6
        if int(moving.sum()) < 4:
            return 0.0, None
        ang = np.degrees(np.arctan2(vys[moving], vxs[moving]))
        w = weights[moving]
        bins = np.linspace(-180.0, 180.0, 9)
        hist, _ = np.histogram(ang, bins=bins, weights=w)
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        ent = float(-np.sum(p * np.log2(p)))
        dom = float(np.degrees(np.arctan2(float(np.mean(vys[moving])),
                                          float(np.mean(vxs[moving])))))
        return ent, dom

    # ------------------------------------------------------------------

    def update(
        self,
        speeds: list[float],
        vxs: list[float],
        vys: list[float],
        divergence_mean: float,
        positions: list[tuple[float, float]],
        frame_shape: tuple[int, int],
    ) -> FrameMetrics:
        """
        speeds/vxs/vys : per-track velocity, already converted to the run's
                         units by the monitor (m/s or px/frame).
        positions      : per-track source-pixel positions, for the cell grids.
        """
        count = len(speeds)
        arr = np.array(list(zip(vxs, vys)), dtype=np.float64) if count else np.zeros((0, 2))
        moving = np.hypot(arr[:, 0], arr[:, 1]) > 1e-6 if count else np.zeros(0, dtype=bool)
        n_moving = int(moving.sum())

        if count:
            mean_v = arr.mean(axis=0)
            vel_var = float(np.mean(np.sum((arr - mean_v) ** 2, axis=1)))
            mean_speed = float(np.hypot(*mean_v))
        else:
            vel_var = 0.0
            mean_speed = 0.0

        if count:
            entropy, dom_deg = self._entropy(arr[:, 0], arr[:, 1],
                                             np.hypot(arr[:, 0], arr[:, 1]))
        else:
            entropy, dom_deg = 0.0, None

        # Counter-flow: share of moving tracks opposing the dominant stream.
        counter_pct = 0.0
        if count and dom_deg is not None and n_moving:
            dom_rad = np.deg2rad(dom_deg)
            dots = arr[:, 0] * np.cos(dom_rad) + arr[:, 1] * np.sin(dom_rad)
            counter_pct = float((dots[moving] < 0).mean() * 100.0)

        # Cell grids: normalised head-count share per cell (source used
        # absolute counts divided by a nominal cell area; shares stay
        # meaningful whether or not a ground footprint is configured).
        gh, gw = self.grid
        h, w = frame_shape
        cell_d = np.zeros((gh, gw), dtype=np.float32)
        for (px, py) in positions:
            i = int(np.clip(py / max(h, 1) * gh, 0, gh - 1))
            j = int(np.clip(px / max(w, 1) * gw, 0, gw - 1))
            cell_d[i, j] += 1.0
        cell_density = cell_d / max(cell_d.sum(), 1.0)
        cell_pressure = cell_density * vel_var

        self.speed_hist.append(mean_speed)
        self.vec_hist.append((float(mean_v[0]), float(mean_v[1])) if count else (0.0, 0.0))

        occupancy = (count / self.safe_capacity) if self.safe_capacity else None

        if self.roi_area_m2:
            density = count / max(self.roi_area_m2, 1e-3)
        else:
            density = float(count)

        return FrameMetrics(
            head_count=count,
            density=round(density, 4),
            density_unit="heads/m2" if self.calibrated else "heads/frame",
            mean_speed=round(mean_speed, 4),
            speed_unit="m/s" if self.calibrated else "px/frame",
            velocity_variance=round(vel_var, 5),
            pressure=round(density * vel_var, 5),
            divergence=round(float(divergence_mean), 6),
            directional_entropy=round(entropy, 4),
            dominant_dir_deg=round(dom_deg, 2) if dom_deg is not None else None,
            counter_flow_pct=round(counter_pct, 2),
            stop_and_go=self._stop_and_go(),
            oscillation=self._oscillation(),
            occupancy_ratio=round(occupancy, 4) if occupancy is not None else None,
            n_moving_tracks=n_moving,
            n_stopped_tracks=count - n_moving,
            cell_density=cell_density,
            cell_pressure=cell_pressure,
        )
