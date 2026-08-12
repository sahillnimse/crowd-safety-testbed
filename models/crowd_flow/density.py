"""
Per-cell crowd density, and the crowd pressure built on it.

Why this exists
---------------
Every metric in this module up to now has been derived from the flow field
alone: divergence, curl, counterflow, turbulence.  Those describe how the
crowd is *moving*.  None of them know how many people are there, and the
single most validated predictor of a crowd disaster needs exactly that.

Helbing's crowd pressure is

    P = rho * Var(v)

local density times local velocity variance, established from video analysis
of the 2006 Hajj disaster.  It comes with empirical thresholds that were
calibrated against a real crowd collapse rather than tuned by eye:

    crowd turbulence begins        P ~ 0.02 s^-2
    stampede conditions            P ~ 0.04 s^-2

The engine already computes the Var(v) half.  This module supplies rho, so
the project can report pressure against those published numbers instead of
against thresholds someone picked because the output looked reasonable.

Where the count comes from
--------------------------
Either of two sources, selected by ``source``:

``"boxes"`` (default)
    The shared RT-DETRv2 detector in models/_detectors.py, with tiling.  No
    extra dependency, and tiling takes it from ~35 to ~100 people on a dense
    frame of this project's own footage.  It still counts *bodies*, so it
    degrades exactly where density matters most: at the back of a crowd,
    bodies are almost entirely occluded and boxes merge.

``"heads"``
    models/head_count — a density-map counter trained on point labels from
    your own footage.  Heads stay visible where bodies do not, and the count
    is an integral over a density map rather than a number of successful
    detections, so it does not collapse when individuals stop being
    separable.  This is the better source, and it requires you to label
    patches and train (see models/head_count/__init__.py).

Head points are NOT ground contacts
-----------------------------------
This is the one thing that makes the head source harder than the box source
rather than simply better.  A ground-plane homography maps image points to
where they touch the ground; a head is roughly a person's height above that.
Mapping a head point directly puts them metres further from the camera —
several grid cells at the back of a crowd, in the direction that inflates
density where it is already highest.

So a head point is projected down the image by an assumed stature before it
is mapped.  ``person_height_m`` is that assumption, and it is an assumption:
it is exposed, documented and applied consistently rather than buried.  The
box source does not need this because a box already has a foot edge.

The detector runs every ``stride`` frames, not every frame.  Density is a
property of where the crowd *is*, which changes over seconds; velocity
variance changes frame to frame and is what carries the fast signal.

Two things that are easy to get wrong here
------------------------------------------
**Foot point, not box centre.**  A person is located on the ground plane by
where they touch it.  Mapping a box centre through a ground-plane homography
places them somewhere behind themselves — roughly half a body height further
from the camera, which at the back of a crowd is several metres and several
cells.

**Cell ground area is not constant.**  Perspective means a grid cell near the
horizon covers far more ground than one in the foreground.  Dividing every
cell's count by the same area reports the far field as several times denser
than it is — the exact region where a real crush would be least visible.  So
each cell's area is computed through the homography individually.

Uncalibrated cameras
--------------------
Without a homography there is no m^2, so there is no persons/m^2 and no
comparison to 0.02 or 0.04 s^-2 that means anything.  Density is still
reported, in persons per 1000 px^2, and pressure thresholds are refused for
that camera in the same way speed thresholds already are.  A number in the
wrong units checked against a physical constant is worse than no number.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Helbing et al., PNAS 2011 / Adv. Complex Syst. 2008.  Physical units
# (persons/m^2 * (m/s)^2), so they apply only to a calibrated camera.
PRESSURE_TURBULENCE = 0.02
PRESSURE_STAMPEDE = 0.04

# Uncalibrated density is reported per this many px^2 rather than per px^2,
# which would be a number with five leading zeros in every log line.
PX_AREA_UNIT = 1000.0

DENSITY_UNITS_CALIBRATED = "persons/m^2"
DENSITY_UNITS_UNCALIBRATED = "persons/1000px^2"

# A cell whose ground area comes out at or below this is degenerate: at or
# above the horizon, or behind the camera, where the homography maps a finite
# image region to an unbounded or inverted patch of ground.  Dividing a count
# by it produces an enormous density from one detection.
_MIN_CELL_AREA_M2 = 1e-3

# Ground area beyond which a cell is treated as unmeasurable rather than
# merely large.  Near the vanishing point one cell can span hundreds of
# square metres, and one person in it yields a density near zero that would
# quietly drag a zone mean down.
_MAX_CELL_AREA_M2 = 400.0

# Densities above this are not dense crowds, they are wrong calibrations.
#
# The physical ceiling for standing adults is around 8-10 persons/m^2 — that
# is already the lethal-packing regime documented in crush post-mortems, and
# nobody has measured a sustained figure above it because bodies do not
# occupy less space than that.  So a cell reporting 30, or 160, is not
# reporting a crowd; it is reporting a homography that maps a grid cell to a
# far smaller patch of ground than it really covers.
#
# This matters more than the usual sanity check because density enters
# pressure linearly: a calibration wrong by 16x produces a pressure alert
# wrong by 16x, comfortably past a threshold calibrated against an actual
# disaster, on an empty pavement.  Excluding the cell and saying so is the
# only honest option — the alternative is a CRITICAL alert with a number
# behind it that cannot be true.
_MAX_PLAUSIBLE_DENSITY = 12.0

# Head -> foot search. The expanding scan bounds the answer, the bisection
# refines it; 24 bisections resolve any image height to well under a pixel.
_FOOT_SEARCH_STEPS = 40
_FOOT_BISECT_STEPS = 24


class DensityEstimator:
    """
    Per-cell person density, refreshed every ``stride`` frames.

    Usage mirrors the rest of the module: construct once, call ``update``
    with each frame, read ``cell_density`` when building metrics.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        stride: int = 5,
        tile_grid: Optional[tuple[int, int]] = (2, 2),
        smoothing: float = 0.5,
        enabled: bool = True,
        source: str = "boxes",
        head_weights: Optional[str] = None,
        person_height_m: float = 1.65,
    ) -> None:
        if source not in ("boxes", "heads"):
            raise ValueError(f"source must be 'boxes' or 'heads', got {source!r}")
        self.device = device
        self.conf_threshold = conf_threshold
        self.stride = max(1, int(stride))
        self.tile_grid = tile_grid
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.enabled = enabled
        self.source = source
        self.head_weights = head_weights
        # Mean standing stature. 1.65 m is a reasonable mixed-population
        # figure; it only has to be right on average, because it is used to
        # find the ground point under a head, and an error of 10 cm moves
        # that point by well under one grid cell at any usable range.
        self.person_height_m = person_height_m

        self._detector = None
        self._counts: Optional[np.ndarray] = None      # smoothed persons per cell
        self._areas: Optional[np.ndarray] = None       # m^2 (or px^2) per cell
        self._areas_key: Optional[tuple] = None
        self._last_n_persons: int = 0
        self._frames_since_detect: int = 0
        self._implausible_warned: bool = False
        # update() calls seen, which is what `stride` counts.  See update().
        self._calls: int = 0

    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.enabled or self._detector is not None:
            return
        if self.source == "heads":
            from models.head_count import HeadCounter
            self._detector = HeadCounter(weights=self.head_weights,
                                         device=self.device)
        else:
            from models._detectors import get_detector
            self._detector = get_detector(device=self.device)
        self._detector.load()

    def reset(self) -> None:
        self._counts = None
        self._areas = None
        self._areas_key = None
        self._last_n_persons = 0
        self._frames_since_detect = 0
        self._calls = 0

    @property
    def n_persons(self) -> int:
        """People detected in the most recent detector pass."""
        return self._last_n_persons

    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        grid_cell_px: int,
        calibration=None,
    ) -> None:
        """
        Refresh the density estimate if this call is a detector call.

        ``stride`` counts CALLS, not source frame indices.  It used to test
        ``frame_index % stride``, and frame_index is the index in the source
        video while update() is invoked once per SAMPLED frame — so whenever
        the sampling interval was a multiple of the stride the test was true
        every time and the detector ran on every frame.  That is the default
        configuration: the web UI samples every 5th frame and density_stride is
        5, giving frame indices 0, 5, 10, 15 ... every one divisible by 5.
        """
        if not self.enabled:
            return
        due = (self._calls % self.stride) == 0
        self._calls += 1
        if not due and self._counts is not None:
            self._frames_since_detect += 1
            return

        self.load()

        h, w = frame.shape[:2]
        n_y, n_x = h // grid_cell_px, w // grid_cell_px
        if n_y < 1 or n_x < 1:
            return

        feet = self._foot_points(frame, calibration)
        self._frames_since_detect = 0

        counts = np.zeros((n_y, n_x), dtype=np.float32)
        for fx, fy in feet:
            cx, cy = int(fx // grid_cell_px), int(fy // grid_cell_px)
            if 0 <= cy < n_y and 0 <= cx < n_x:
                counts[cy, cx] += 1.0

        # Temporal smoothing.  A detector is noisy frame to frame — a person
        # at a cell boundary flips between two cells — and density is not
        # supposed to be a fast signal.
        if self._counts is None or self._counts.shape != counts.shape:
            self._counts = counts
        else:
            a = self.smoothing
            self._counts = (1.0 - a) * self._counts + a * counts

        self._ensure_areas(h, w, grid_cell_px, calibration)

    # ------------------------------------------------------------------

    def _foot_points(self, frame: np.ndarray, calibration) -> np.ndarray:
        """
        (N, 2) ground-contact points in image pixels, whichever source is in use.

        Boxes already carry a foot edge.  Head points do not, and are pushed
        down the image by one assumed stature — see the module docstring for
        why mapping a head straight through a ground homography is wrong.
        """
        if self.source == "heads":
            # One forward pass for both.  Calling points() and then count() ran
            # the network twice over the same frame, doubling the cost of the
            # most expensive step in the pipeline to recover a number the first
            # pass had already produced.
            pts, count = self._detector.points_and_count(frame)
            self._last_n_persons = int(round(count))
            if len(pts) == 0:
                return pts
            return self._heads_to_feet(pts, calibration)

        from models._detectors import COCO_PERSON
        boxes = self._detector.detect(
            frame, classes=(COCO_PERSON,),
            conf_threshold=self.conf_threshold,
            tile_grid=self.tile_grid,
        )
        self._last_n_persons = len(boxes)
        return np.array([[(x1 + x2) * 0.5, y2] for x1, y1, x2, y2 in boxes],
                        dtype=np.float32).reshape(-1, 2)

    def _heads_to_feet(self, heads: np.ndarray, calibration) -> np.ndarray:
        """
        Move each head point down to its ground contact.

        Uncalibrated, there is no metric scale, so the only available shift is
        none — the head point is used as-is and the resulting density is a
        px-based figure that is already not comparable to anything physical.

        Calibrated, the shift is found by bisection on image row: the ground
        point directly below a head is the row whose world position is one
        stature away from the world position the head row maps to.  Solving it
        this way rather than with a fixed pixel offset is what makes it work
        at every depth — near the camera a person is hundreds of pixels tall
        and at the back of the crowd a dozen.
        """
        if calibration is None or not calibration.is_calibrated:
            return heads

        out = np.empty_like(heads)
        for i, (hx, hy) in enumerate(heads):
            lo, hi = float(hy), float(hy)
            # Expand downwards until the ground distance exceeds a stature.
            step = 4.0
            hx_w, hy_w = calibration.pixel_to_world(float(hx), float(hy))
            for _ in range(_FOOT_SEARCH_STEPS):
                hi += step
                gx, gy = calibration.pixel_to_world(float(hx), hi)
                if np.hypot(gx - hx_w, gy - hy_w) >= self.person_height_m:
                    break
                step *= 1.6
            else:
                out[i] = (hx, hy)      # never reached a stature; leave it
                continue
            for _ in range(_FOOT_BISECT_STEPS):
                mid = (lo + hi) * 0.5
                gx, gy = calibration.pixel_to_world(float(hx), mid)
                if np.hypot(gx - hx_w, gy - hy_w) < self.person_height_m:
                    lo = mid
                else:
                    hi = mid
            out[i] = (hx, (lo + hi) * 0.5)
        return out

    def _ensure_areas(self, h: int, w: int, grid_cell_px: int, calibration) -> None:
        """Ground area of every cell, computed once per geometry."""
        calibrated = bool(calibration is not None and calibration.is_calibrated)
        key = (h, w, grid_cell_px, calibrated)
        if self._areas_key == key and self._areas is not None:
            return

        n_y, n_x = h // grid_cell_px, w // grid_cell_px
        if not calibrated:
            # Uniform image area; the units say so.
            self._areas = np.full((n_y, n_x),
                                  (grid_cell_px ** 2) / PX_AREA_UNIT,
                                  dtype=np.float32)
            self._areas_key = key
            return

        areas = np.zeros((n_y, n_x), dtype=np.float32)
        g = grid_cell_px
        for iy in range(n_y):
            for ix in range(n_x):
                # Shoelace area of the cell's four corners on the ground.
                # The cell is a rectangle in the image and a general
                # quadrilateral on the ground, so its area is not
                # width x height of anything.
                pts = [
                    calibration.pixel_to_world(ix * g,       iy * g),
                    calibration.pixel_to_world((ix + 1) * g, iy * g),
                    calibration.pixel_to_world((ix + 1) * g, (iy + 1) * g),
                    calibration.pixel_to_world(ix * g,       (iy + 1) * g),
                ]
                area = 0.0
                for i in range(4):
                    x0, y0 = pts[i]
                    x1, y1 = pts[(i + 1) % 4]
                    area += x0 * y1 - x1 * y0
                areas[iy, ix] = abs(area) * 0.5

        n_bad = int(((areas < _MIN_CELL_AREA_M2) |
                     (areas > _MAX_CELL_AREA_M2)).sum())
        if n_bad:
            logger.info(
                "Density: %d of %d cells have an unusable ground area "
                "(<%.3f or >%.0f m^2) and are excluded — these sit at or "
                "beyond the horizon, where a finite image region maps to an "
                "unbounded patch of ground.",
                n_bad, areas.size, _MIN_CELL_AREA_M2, _MAX_CELL_AREA_M2,
            )
        self._areas = areas
        self._areas_key = key

    # ------------------------------------------------------------------

    def cell_density(self) -> Optional[np.ndarray]:
        """
        Persons per m^2 (calibrated) or per 1000 px^2 (uncalibrated).

        NaN in cells whose ground area is unusable, so downstream means skip
        them rather than being dominated by them.
        """
        if self._counts is None or self._areas is None:
            return None
        areas = self._areas
        out = np.full(self._counts.shape, np.nan, dtype=np.float32)
        ok = (areas >= _MIN_CELL_AREA_M2) & (areas <= _MAX_CELL_AREA_M2)
        out[ok] = self._counts[ok] / areas[ok]

        # Physical plausibility.  Only meaningful in persons/m^2 — the
        # uncalibrated unit has no such ceiling because it is not a physical
        # density at all.
        if self._areas_key is not None and self._areas_key[3]:
            impossible = np.isfinite(out) & (out > _MAX_PLAUSIBLE_DENSITY)
            n_bad = int(impossible.sum())
            if n_bad:
                out[impossible] = np.nan
                if not self._implausible_warned:
                    logger.warning(
                        "Density: %d cell(s) exceeded %.0f persons/m^2 and "
                        "were discarded.  Standing adults top out near 8-10, "
                        "so this is a homography that under-measures ground "
                        "area rather than a very dense crowd.  Density feeds "
                        "crowd pressure linearly, so leaving these in would "
                        "raise a CRITICAL pressure alert on a number that "
                        "cannot be true.  Re-run "
                        "scripts/calibrate_ground_plane.py.  (Logged once.)",
                        n_bad, _MAX_PLAUSIBLE_DENSITY,
                    )
                    self._implausible_warned = True
        return out


def crowd_pressure(
    cell_density: Optional[np.ndarray],
    cell_variance: np.ndarray,
) -> Optional[np.ndarray]:
    """
    P = rho * Var(v), per cell.

    ``cell_variance`` must already be in the same length units as the
    density: (m/s)^2 against persons/m^2.  The engine converts the flow field
    to m/s before building cell arrays whenever the camera is calibrated, so
    passing a calibrated density with an uncalibrated variance cannot happen
    through the normal path — but it is the one mistake that would produce a
    plausible number in meaningless units, so it is worth stating.
    """
    if cell_density is None:
        return None
    if cell_density.shape != cell_variance.shape:
        logger.warning(
            "Density grid %s does not match the metrics grid %s; skipping "
            "crowd pressure for this frame.",
            cell_density.shape, cell_variance.shape,
        )
        return None
    return cell_density * cell_variance
