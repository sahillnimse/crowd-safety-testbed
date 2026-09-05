"""
Point tracker for density-map head peaks.

Ported from ``Ujwal/__CMS__/__Dashboard__/core`` and adapted: it merges

  * ``point_tracker.py::PointTracker``   — Hungarian assignment on point
    distance with smoothed per-track velocity (this is the tracker that
    actually pairs with DM-Count peaks in the source workspace), and
  * ``tracking.py::ByteTrackerLite``     — BYTE-style two-stage association
    with globally-optimal (scipy) assignment.

Why not ByteTrackerLite verbatim? Its two stages match on IoU, which is
undefined for bare points — a density peak has a location and a confidence
value, no extent. The port keeps what makes BYTE work (strong detections
matched first, weak ones recovered in a second pass against leftover tracks)
and swaps the cost from IoU to Euclidean distance. Stage two uses a tighter
gate than stage one, mirroring BYTE's lower match threshold for low-score
detections.

Peak values are density mass, NOT calibrated probabilities: thresholds here
are relative to the counter's own ``peak_value_thresh`` floor, not to a
[0, 1] detector score range.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class PointTrack:
    track_id: int
    x: float
    y: float
    value: float = 0.0        # density-mass of the most recent matched peak
    vx: float = 0.0           # px/frame between consecutive observations,
    vy: float = 0.0           # smoothed over `smooth_frames` samples
    age: int = 1
    time_since_update: int = 0
    _vel_hist: deque = field(default_factory=lambda: deque(maxlen=5))
    history: list = field(default_factory=list)

    @property
    def pos(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def speed_px(self) -> float:
        return float(np.hypot(self.vx, self.vy))


class PointTracker:
    """Two-stage nearest-point tracker with velocity smoothing."""

    def __init__(
        self,
        max_dist_px: float = 60.0,
        high_thresh: float = 0.25,
        low_thresh: float = 0.06,
        max_age: int = 15,
        smooth_frames: int = 5,
    ):
        self.max_dist_px = max_dist_px
        # Stage split on peak density-mass. Defaults bracket the counter's
        # default peak floor (0.06); tune together with it.
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.smooth_frames = smooth_frames
        self.tracks: list[PointTrack] = []
        self._next_id = 1

    # ------------------------------------------------------------------

    def _match(self, tracks: list[PointTrack], pts: np.ndarray,
               gate: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Globally optimal one-to-one assignment under a distance gate."""
        if not tracks or len(pts) == 0:
            return [], list(range(len(tracks))), list(range(len(pts)))
        prev = np.array([t.pos for t in tracks], dtype=np.float32)
        cost = np.linalg.norm(prev[:, None, :] - pts[None, :, :2], axis=2)
        ri, ci = linear_sum_assignment(cost)
        matches, u_tr, u_pt = [], set(range(len(tracks))), set(range(len(pts)))
        for r, c in zip(ri, ci):
            if cost[r, c] <= gate:
                matches.append((int(r), int(c)))
                u_tr.discard(r)
                u_pt.discard(c)
        return matches, sorted(u_tr), sorted(u_pt)

    def _apply(self, tr: PointTrack, px: float, py: float, val: float) -> None:
        raw_vx, raw_vy = px - tr.x, py - tr.y
        tr._vel_hist.append((raw_vx, raw_vy))
        # Median, not mean: a peak briefly matching its neighbour produces one
        # huge outlier displacement, which a mean smears across every
        # following smoothed velocity while a median discards it. With dense
        # head fields this was the difference between pedestrian-scale speeds
        # (~1-3 px/frame) and double-digit phantom speeds.
        sm = np.median(np.array(tr._vel_hist), axis=0)
        tr.vx, tr.vy = float(sm[0]), float(sm[1])
        tr.x, tr.y, tr.value = float(px), float(py), float(val)
        tr.time_since_update = 0
        tr.history.append((tr.x, tr.y))
        if len(tr.history) > 40:
            tr.history = tr.history[-40:]

    def update(self, points_with_values) -> list[PointTrack]:
        """
        points_with_values: iterable of (x, y, density_value) in source pixels.
        Returns the tracks matched this frame (time_since_update == 0).
        """
        pts = np.asarray(list(points_with_values), dtype=np.float32).reshape(-1, 3)

        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        consumed: set[int] = set()
        if self.tracks and len(pts):
            strong = [i for i in range(len(pts)) if pts[i, 2] >= self.high_thresh]
            weak = [i for i in range(len(pts)) if self.low_thresh <= pts[i, 2] < self.high_thresh]

            # Stage 1: confident peaks first.
            m1, u_tr, _ = self._match(self.tracks, pts[strong], self.max_dist_px)
            for ti, si in m1:
                gi = strong[si]
                consumed.add(gi)
                self._apply(self.tracks[ti], pts[gi, 0], pts[gi, 1], pts[gi, 2])

            # Stage 2: weak peaks recover still-unmatched tracks under a
            # tighter gate (BYTE lowers its match threshold here; distance
            # gates tighten instead of loosen).
            remain = [self.tracks[i] for i in u_tr]
            if remain and weak:
                m2, _, _ = self._match(remain, pts[weak], self.max_dist_px * 0.75)
                for ti, si in m2:
                    gi = weak[si]
                    consumed.add(gi)
                    self._apply(remain[ti], pts[gi, 0], pts[gi, 1], pts[gi, 2])

        for i in range(len(pts)):
            if i not in consumed:
                self.tracks.append(PointTrack(
                    track_id=self._next_id,
                    x=float(pts[i, 0]), y=float(pts[i, 1]),
                    value=float(pts[i, 2]),
                    history=[(float(pts[i, 0]), float(pts[i, 1]))],
                ))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.time_since_update == 0]
