"""
Grid-cell flow field aggregation and crowd-safety metrics.

All statistics are computed over a regular grid of cells (default 16×16 px at
source resolution).  Per-cell arrays are returned in a MetricsFrame; per-zone
scalars are computed by masking the cell arrays against each zone's polygon.

Units — divergence and curl
---------------------------
Divergence and curl are reported **per grid cell**, not per pixel:

    divergence = (∂vx/∂x + ∂vy/∂y) × grid_cell_px

i.e. the net velocity difference (px/frame) accumulated across one cell
width.  This is the unit the configured thresholds are written in
("divergence_critical: -1.5" = 1.5 px/frame of convergence across a cell).

Reporting the raw per-pixel derivative instead makes the numbers depend on
the flow compute resolution: the field is computed downsampled and then
bilinearly upsampled back to source, which divides every per-pixel spatial
derivative by the downsample factor.  At the default 320-px compute width on
a 1280-px source that is a factor of 4, and the resulting values (order 0.1)
never come close to thresholds of order 1 — so no divergence alert can ever
fire, and the heatmap stays uniformly neutral.  Multiplying by the cell size
removes that dependence.

Sign convention — divergence
----------------------------
Divergence ∇·v = ∂vx/∂x + ∂vy/∂y.

A converging crowd (people entering a region faster than they leave, the
compression signature that precedes a crush) has NEGATIVE divergence.
NEGATIVE divergence = HIGHER risk.

This is the standard physical sign and matches the OpenCV coordinate frame
(+x right, +y down).  The sign convention is verified by a synthetic test
every time CrowdMetricsEngine is instantiated: a radially converging field
must give negative divergence everywhere, or a RuntimeError is raised before
any frame is processed.  Getting this wrong inverts every alert.

Sign convention — curl
----------------------
Curl (2D): ∂vy/∂x - ∂vx/∂y.  Positive = counter-clockwise rotation.
Healthy laminar flow has curl ≈ 0.  Rotational flow (positive or negative)
means people are being pushed sideways rather than walking.  AlertEngine
checks |curl| against the threshold.

Metrics computed
----------------
Per cell:
  mean_speed        px/frame (or m/s if calibrated)
  divergence        ∇·v (central differences within cell)
  curl              ∂vy/∂x − ∂vx/∂y
  coherence         1 − circular_variance  (0=random, 1=perfectly aligned)
  velocity_variance spatial variance of speed within the cell

Per zone (polygon-masked aggregation of cells):
  mean_speed        mean of cell mean_speeds
  mean_divergence   mean of cell divergences
  mean_curl         mean of |cell curls|
  mean_coherence    mean of cell coherences
  mean_variance     mean of cell velocity_variances
  stop_go_score     temporal autocorrelation at short lags of zone mean speed
  counterflow_score fraction of cells with direction opposing zone dominant
  turbulence_index  Helbing et al.: velocity_variance / mean_speed²

Stop-and-go wave detection
--------------------------
A ring buffer (50 frames) of zone mean speed is maintained per zone.  At each
frame, the temporal autocorrelation is computed at lags of [5, 10, 15] frames.
High negative autocorrelation at short lags is the signature of periodic
halting propagating backwards through a queue (documented crowd crush precursor).
stop_go_score = max of −autocorr at each tested lag, clipped to [0, 1].

Counterflow detection
---------------------
Counterflow (opposing movement within the same corridor) is the direct
signature of entry/exit separation breaking down.  For each zone, the dominant
direction is computed from the magnitude-weighted mean flow vector.  Cells
whose mean direction deviates by > 90° from the dominant are counted.
counterflow_score = count / total_active_cells.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZoneMetrics:
    """Aggregated metrics for one named zone on one frame."""
    zone_name:       str
    mean_speed:      float    # px/frame or m/s depending on calibration
    mean_divergence: float    # negative = compression
    mean_curl:       float    # |curl|, positive = rotation present
    mean_coherence:  float    # 1 = perfectly aligned, 0 = random
    mean_variance:   float    # spatial speed variance
    stop_go_score:   float    # [0, 1]; higher = periodic halting detected
    counterflow_score: float  # [0, 1]; fraction of cells opposing dominant dir
    oscillation_symmetry_score: float  # [0, 1]; halted dense-crowd rocking
    turbulence_index: float   # velocity_variance / mean_speed²; Helbing et al.
    active_cells:    int      # cells with motion above min_magnitude
    umbrella_occluded_frac: float = 0.0  # fraction of zone under umbrella canopy
    # Crowd pressure and the density it is built from.  None when no density
    # estimate is available (detector disabled or not yet run) — distinct
    # from 0.0, which would read as "measured, and the crowd is empty".
    mean_density:    Optional[float] = None  # persons/m² or persons/1000px²
    crowd_pressure:  Optional[float] = None  # ρ·Var(v); s⁻² when calibrated


@dataclass
class MetricsFrame:
    """Complete per-frame output from CrowdMetricsEngine."""
    # Per-cell arrays (shape: n_cells_y × n_cells_x)
    cell_speed:      np.ndarray   # float32
    cell_divergence: np.ndarray   # float32
    cell_curl:       np.ndarray   # float32  (signed)
    cell_coherence:  np.ndarray   # float32  [0, 1]
    cell_variance:   np.ndarray   # float32
    cell_oscillation: np.ndarray  # float32 [0, 1]

    # Cell-grid layout info
    cell_size_px: int
    grid_origin:  tuple[int, int]   # (x0, y0) in source frame

    # Per-zone aggregates
    zones: dict[str, ZoneMetrics]

    # Metadata
    is_calibrated:  bool
    speed_units:    str
    frame_index:    int
    timestamp_sec:  float
    is_rain_flagged:     bool
    is_lowlight_flagged: bool

    # Density / pressure.  Optional because both require a person detector;
    # every other array here comes from the flow field alone.
    cell_density:   Optional[np.ndarray] = None
    cell_pressure:  Optional[np.ndarray] = None
    density_units:  Optional[str] = None
    n_persons:      Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class CrowdMetricsEngine:
    """
    Aggregates a FlowResult into a MetricsFrame.

    Parameters
    ----------
    grid_cell_px:
        Size of each grid cell in source pixels.
    min_magnitude:
        Cells with mean speed below this are treated as static and excluded
        from zone aggregates (prevents near-zero-motion cells from dominating
        averages in large zones where only part of the frame has crowd).
    stop_go_lag_frames:
        Frame lags at which temporal autocorrelation is evaluated for stop-go
        detection.  Must be shorter than the ring buffer size (50 frames).
    """

    _SPEED_BUFFER_LEN = 50   # frames of zone-speed history for stop-go

    def __init__(
        self,
        grid_cell_px: int = 16,
        min_magnitude: float = 0.3,
        stop_go_lag_frames: Optional[list[int]] = None,
        oscillation_density_threshold: float = 2.5,
        oscillation_stationary_speed: float = 0.5,
    ) -> None:
        self.grid_cell_px = grid_cell_px
        self.min_magnitude = min_magnitude
        # zone name -> already warned that nothing clears its motion floor.
        self._empty_zone_warned: dict[str, bool] = {}
        self.stop_go_lags  = stop_go_lag_frames or [5, 10, 15]
        self.oscillation_density_threshold = oscillation_density_threshold
        self.oscillation_stationary_speed = oscillation_stationary_speed

        # Per-zone speed history for stop-go detection
        self._speed_history: dict[str, deque] = {}
        self._vector_history: dict[str, deque] = {}

        # Rasterised zone→cell masks, keyed by (zone name, grid shape).
        # The polygons are fixed for a run, so this is computed once each.
        self._zone_mask_cache: dict[tuple, np.ndarray] = {}

        # Verify sign convention before any real data is processed
        self._verify_sign_convention()

    # ------------------------------------------------------------------
    # Sign convention self-test
    # ------------------------------------------------------------------

    def _verify_sign_convention(self) -> None:
        """
        Synthesize a radially converging flow field and assert that the
        divergence is negative everywhere.

        A converging crowd has negative divergence by the physical convention
        (∇·v < 0 when flow compresses).  If this test fails, every compression
        alert would fire when it should not — and never fire when it should.

        RuntimeError is raised immediately so the misconfiguration is caught
        at engine startup, not during a live monitoring session.
        """
        H, W = 32, 32
        cx, cy = W / 2.0, H / 2.0
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        fx = cx - xs   # point inward (toward centre)
        fy = cy - ys

        # Normalise to unit vectors so the magnitude doesn't affect the sign test
        mag = np.sqrt(fx ** 2 + fy ** 2) + 1e-6
        fx /= mag
        fy /= mag

        # Divergence via central differences: ∂fx/∂x + ∂fy/∂y
        # On the interior pixels (excluding 1-pixel border where central
        # differences wrap around), divergence of inward-pointing radial field
        # must be negative.
        div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
        interior_div = div[2:-2, 2:-2]

        if interior_div.max() > 0.05:
            raise RuntimeError(
                "CrowdMetricsEngine sign-convention self-test FAILED.  "
                "Divergence of a converging synthetic field is positive "
                "(max interior div = %.4f).  "
                "This means every compression alert would fire incorrectly.  "
                "Do not proceed — investigate the divergence computation." % interior_div.max()
            )
        logger.debug(
            "Sign-convention self-test passed: converging field gives "
            "max interior divergence = %.4f (must be < 0.05).",
            interior_div.max(),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        flow_result,                           # FlowResult from flow_field.py
        zones: list,                           # list[Zone] from zones.py
        calibration,                           # CameraCalibration from ground_plane.py
        ped_mask: Optional[np.ndarray] = None, # 1 = include, 0 = exclude
        umbrella_layer=None,                   # DetectorMaskLayer for occlusion fraction
        fps: float = 30.0,
        density=None,                          # DensityEstimator, or None
    ) -> MetricsFrame:
        """
        Compute per-cell and per-zone metrics.

        Parameters
        ----------
        flow_result:
            Output of FlowField.compute().  field_xy must be in source px / frame.
        zones:
            Zones to aggregate metrics for.
        calibration:
            Used to convert speed to m/s when is_calibrated is True.
        ped_mask:
            Binary uint8 (H, W).  Cells entirely outside the mask are skipped.
            Pass None to use the entire frame.
        umbrella_layer:
            DetectorMaskLayer.  Used to compute per-zone umbrella occlusion
            fraction for density-estimate annotation.
        density:
            DensityEstimator, or None.  Supplies the persons-per-cell half of
            Helbing crowd pressure; without it every pressure field is None
            rather than zero, because "no detector ran" and "nobody is here"
            must not read the same downstream.
        fps:
            Frames per second of the source video.
        """
        field_xy = flow_result.field_xy
        h, w = field_xy.shape[:2]

        # Optionally convert to m/s
        if calibration.is_calibrated:
            try:
                field_ms = calibration.flow_field_to_ms(
                    field_xy, fps=fps, grid_cell_px=self.grid_cell_px
                )
            except Exception as exc:
                logger.warning(
                    "flow_field_to_ms failed (%s); falling back to px/frame.", exc
                )
                field_ms = field_xy
                calibration_effective = False
            else:
                calibration_effective = True
        else:
            field_ms = field_xy
            calibration_effective = False

        speed_units = calibration.speed_units

        # Build cell arrays
        (cell_speed, cell_div, cell_curl, cell_coh, cell_var,
         cell_mx, cell_my) = self._build_cell_arrays(field_ms, ped_mask, h, w)

        # Density and crowd pressure.  cell_var is already in (m/s)^2 when
        # the camera is calibrated, which is the unit the published pressure
        # thresholds assume.
        from models.crowd_flow.density import (
            crowd_pressure, DENSITY_UNITS_CALIBRATED, DENSITY_UNITS_UNCALIBRATED,
        )
        cell_density = density.cell_density() if density is not None else None
        cell_pressure = crowd_pressure(cell_density, cell_var)
        density_units = None
        if cell_density is not None:
            density_units = (DENSITY_UNITS_CALIBRATED if calibration_effective
                             else DENSITY_UNITS_UNCALIBRATED)

        # Aggregate per zone
        zone_results: dict[str, ZoneMetrics] = {}
        cell_oscillation = np.zeros_like(cell_speed, dtype=np.float32)
        for zone in zones:
            zm = self._aggregate_zone(
                zone, cell_speed, cell_div, cell_curl, cell_coh, cell_var,
                cell_mx, cell_my, h, w, umbrella_layer,
                cell_density, cell_pressure,
            )
            zone_results[zone.name] = zm
            mask = self._zone_cell_mask(zone, cell_speed.shape[0], cell_speed.shape[1])
            cell_oscillation[mask] = zm.oscillation_symmetry_score

        return MetricsFrame(
            cell_speed      = cell_speed,
            cell_divergence = cell_div,
            cell_curl       = cell_curl,
            cell_coherence  = cell_coh,
            cell_variance   = cell_var,
            cell_oscillation = cell_oscillation,
            cell_size_px    = self.grid_cell_px,
            grid_origin     = (0, 0),
            zones           = zone_results,
            is_calibrated   = calibration_effective,
            speed_units     = speed_units,
            frame_index     = getattr(flow_result, "frame_index", 0),
            timestamp_sec   = getattr(flow_result, "timestamp_sec", 0.0),
            is_rain_flagged      = flow_result.is_rain_flagged,
            is_lowlight_flagged  = flow_result.is_lowlight_flagged,
            cell_density    = cell_density,
            cell_pressure   = cell_pressure,
            density_units   = density_units,
            n_persons       = density.n_persons if density is not None else None,
        )

    # ------------------------------------------------------------------
    # Per-cell computation
    # ------------------------------------------------------------------

    def _build_cell_arrays(
        self,
        field_ms: np.ndarray,
        ped_mask: Optional[np.ndarray],
        h: int,
        w: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        Aggregate flow field into grid cells.  Returns per-cell arrays:
        (speed, divergence, curl, coherence, variance, mean_vx, mean_vy).

        Every aggregate is a block reduction over the (g × g) pixels of each
        cell, expressed as reshape-and-sum over the whole field at once.  A
        per-cell Python loop calling NumPy on 16×16 slices spends nearly all
        of its time in call overhead — at 1080p with 16-px cells that is 8040
        cells × several reductions each, per frame, which measured at ~1.3 s
        per frame against a stated budget of 1-3 ms.
        """
        g = self.grid_cell_px
        n_y = h // g
        n_x = w // g

        if n_y == 0 or n_x == 0:
            z = np.zeros((max(n_y, 0), max(n_x, 0)), dtype=np.float32)
            return z, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy()

        # Pre-compute field-wide gradient arrays (cheaper than per-cell gradients)
        fx, fy = field_ms[..., 0], field_ms[..., 1]
        # Central differences for ∂fx/∂x and ∂fy/∂y (divergence components)
        grad_fx_x = np.gradient(fx, axis=1)
        grad_fy_y = np.gradient(fy, axis=0)
        # Curl components: ∂fy/∂x and ∂fx/∂y
        grad_fy_x = np.gradient(fy, axis=1)
        grad_fx_y = np.gradient(fx, axis=0)

        # Scale per-pixel derivatives to per-cell so the values are independent
        # of the flow compute resolution and commensurate with the configured
        # thresholds.  See the "Units" section of the module docstring.
        div_field  = (grad_fx_x + grad_fy_y) * g    # (∂vx/∂x + ∂vy/∂y) × cell
        curl_field = (grad_fy_x - grad_fx_y) * g    # (∂vy/∂x − ∂vx/∂y) × cell

        # Crop to a whole number of cells; the remainder strip on the right and
        # bottom edges has no complete cell and was ignored by the per-cell
        # loop too.
        H, W = n_y * g, n_x * g
        fx_c   = fx[:H, :W].astype(np.float32)
        fy_c   = fy[:H, :W].astype(np.float32)
        div_c  = div_field[:H, :W].astype(np.float32)
        curl_c = curl_field[:H, :W].astype(np.float32)
        mag_c  = np.sqrt(fx_c ** 2 + fy_c ** 2)

        def block_mean(a: np.ndarray) -> np.ndarray:
            """
            Mean of each (g × g) block → (n_y, n_x).

            INTER_AREA downscaling by an exact integer factor *is* the box
            average, and OpenCV's implementation is markedly faster than the
            equivalent reshape-and-reduce over a full-resolution array.
            """
            return cv2.resize(a, (n_x, n_y), interpolation=cv2.INTER_AREA)

        if ped_mask is not None:
            m = (ped_mask[:H, :W] > 0).astype(np.float32)
            # Masked cell mean = mean(a·m) / mean(m); the g² block sizes cancel.
            frac = block_mean(m)
            valid = frac > 0
            safe = np.where(valid, frac, 1.0).astype(np.float32)
            mean_of = lambda a: block_mean(a * m) / safe   # noqa: E731
        else:
            valid = np.ones((n_y, n_x), dtype=bool)
            mean_of = block_mean

        cell_speed = mean_of(mag_c).astype(np.float32)
        cell_div   = mean_of(div_c).astype(np.float32)
        cell_curl  = mean_of(curl_c).astype(np.float32)
        cell_mx    = mean_of(fx_c).astype(np.float32)
        cell_my    = mean_of(fy_c).astype(np.float32)

        # Variance of magnitude within the cell: E[m²] − E[m]²
        mean_sq  = mean_of(mag_c ** 2)
        cell_var = np.maximum(mean_sq - cell_speed ** 2, 0.0).astype(np.float32)

        # Magnitude-weighted directional coherence = |Σv| / Σ|v|, matching
        # _circular_coherence.  Sums differ from means by the same per-cell
        # factor in numerator and denominator, so the means can be used
        # directly.  A cell with no motion is coherent by convention.
        sum_mag = cell_speed
        sum_fx  = cell_mx
        sum_fy  = cell_my
        denom   = np.where(sum_mag > 1e-9, sum_mag, 1.0)
        cell_coh = np.minimum(np.hypot(sum_fx, sum_fy) / denom, 1.0)
        cell_coh = np.where(sum_mag > 1e-9, cell_coh, 1.0).astype(np.float32)

        # Fully-masked cells carry no measurement.
        for arr in (cell_speed, cell_div, cell_curl,
                    cell_var, cell_coh, cell_mx, cell_my):
            arr *= valid

        return (cell_speed, cell_div, cell_curl, cell_coh, cell_var,
                cell_mx, cell_my)

    # ------------------------------------------------------------------
    # Per-zone aggregation
    # ------------------------------------------------------------------

    def _zone_cell_mask(self, zone, n_y: int, n_x: int) -> np.ndarray:
        """
        Boolean (n_y, n_x) mask of cells whose centre lies inside zone.polygon.

        Cached per (zone name, grid shape): the polygon is fixed for the whole
        run, so testing every cell centre against it on every frame repeats
        identical work — 8040 pointPolygonTest calls per zone per frame at
        1080p.  Rasterising the polygon once with fillPoly in cell coordinates
        gives the same answer.
        """
        key = (zone.name, n_y, n_x, self.grid_cell_px)
        cached = self._zone_mask_cache.get(key)
        if cached is not None:
            return cached

        g = self.grid_cell_px
        # Rasterise at PIXEL resolution, then sample at the cell centres.
        # Scaling the polygon into cell coordinates first and rounding there
        # would quantise each vertex to the nearest whole cell, moving zone
        # boundaries by up to half a cell and mis-classifying whole rows of
        # cells along any edge that does not fall on a cell boundary.
        poly_px = np.round(
            np.array(zone.polygon, dtype=np.float32)
        ).astype(np.int32)
        full = np.zeros((n_y * g, n_x * g), dtype=np.uint8)
        cv2.fillPoly(full, [poly_px], 1)
        mask = full[g // 2::g, g // 2::g][:n_y, :n_x].astype(bool)

        self._zone_mask_cache[key] = mask
        return mask

    def _aggregate_zone(
        self,
        zone,
        cell_speed, cell_div, cell_curl, cell_coh, cell_var,
        cell_mx, cell_my, h, w,
        umbrella_layer,
        cell_density=None,
        cell_pressure=None,
    ) -> ZoneMetrics:
        """Aggregate cell arrays over the cells whose centres fall in zone.polygon."""
        n_y = cell_speed.shape[0]
        n_x = cell_speed.shape[1]

        in_zone_mask = self._zone_cell_mask(zone, n_y, n_x)

        # Per-zone motion floor.  The engine's min_magnitude is the fallback
        # for zones that predate zone_type; a zone carrying its own floor
        # wins, because a pedestrian concourse and a carriageway do not share
        # a scale and one global number is wrong for at least one of them.
        floor = getattr(zone, "motion_floor", self.min_magnitude)
        active = in_zone_mask & (cell_speed > floor)
        n_active = int(active.sum())

        # Density over the zone regardless of motion.  A stationary packed
        # crowd has no active cells at all, and that is the case this metric
        # exists for — computing density only on the moving path would report
        # the most dangerous state as an empty zone.
        def _zone_mean(arr):
            if arr is None:
                return None
            vals = arr[in_zone_mask]
            if not vals.size or np.all(np.isnan(vals)):
                return None
            return float(np.nanmean(vals))

        mean_density = _zone_mean(cell_density)
        mean_pressure = _zone_mean(cell_pressure)

        # Per-zone physical ceiling.  DensityEstimator applies the pedestrian
        # limit globally because it has no zones; a vehicle zone's real limit
        # is ~40x lower (a car occupies ~7 m² even bumper to bumper), so a
        # gross overcount there would clear the global check untouched.
        max_density = getattr(zone, "max_density", None)
        if (mean_density is not None and max_density is not None
                and mean_density > max_density):
            logger.warning(
                "Zone '%s' (%s): density %.2f exceeds the %.2f/m² physically "
                "achievable for this subject; discarding density and pressure "
                "for this frame.  Either the zone_type is wrong or the "
                "homography under-measures ground area.  (Per frame.)",
                zone.name, getattr(zone, "zone_type", "?"),
                mean_density, max_density,
            )
            mean_density = None
            mean_pressure = None

        if n_active == 0:
            # Zero active cells is ambiguous on its own: a gridlocked zone, an
            # empty one, and a motion floor set too high all report the same
            # zeros.  active_cells=0 in the output is the discriminator, and
            # this says so once per zone so a mis-set floor is findable rather
            # than looking like a permanently stationary crowd.
            if not self._empty_zone_warned.get(zone.name):
                logger.warning(
                    "Zone '%s' (%s): no cell exceeded its %.2f px/frame motion "
                    "floor, so every flow metric for it reads zero.  That is "
                    "correct for a genuinely still zone, but if there is "
                    "visible movement the floor is too high for this scene — "
                    "lower it with `min_magnitude:` on the zone.  "
                    "(Logged once per zone.)",
                    zone.name, getattr(zone, "zone_type", "?"), floor,
                )
                self._empty_zone_warned[zone.name] = True
            zm = ZoneMetrics(
                zone_name=zone.name,
                mean_speed=0.0, mean_divergence=0.0, mean_curl=0.0,
                mean_coherence=1.0, mean_variance=0.0,
                stop_go_score=0.0, counterflow_score=0.0,
                oscillation_symmetry_score=0.0,
                turbulence_index=0.0, active_cells=0,
                mean_density=mean_density, crowd_pressure=mean_pressure,
            )
            self._update_speed_history(zone.name, 0.0)
            self._update_vector_history(zone.name, 0.0, 0.0)
            return zm

        mean_speed      = float(cell_speed[active].mean())
        mean_divergence = float(cell_div[active].mean())
        mean_curl       = float(np.abs(cell_curl[active]).mean())
        mean_coherence  = float(cell_coh[active].mean())
        mean_variance   = float(cell_var[active].mean())

        stop_go  = self._compute_stop_go(zone.name, mean_speed)
        cf_score = self._compute_counterflow(cell_mx, cell_my, in_zone_mask,
                                             floor)
        mean_vx, mean_vy = float(cell_mx[active].mean()), float(cell_my[active].mean())
        oscillation = self._compute_oscillation_symmetry(
            zone.name, mean_vx, mean_vy, mean_speed, mean_density,
        )

        # Turbulence index (Helbing et al.): σ²(v) / <v>²
        turb = (mean_variance / (mean_speed ** 2)) if mean_speed > 1e-3 else 0.0
        turb = float(np.clip(turb, 0.0, 10.0))

        # Umbrella occlusion fraction within this zone
        umb_frac = 0.0
        if umbrella_layer is not None:
            zone_bb = zone.bbox()
            zone_h  = zone_bb[3] - zone_bb[1]
            zone_w  = zone_bb[2] - zone_bb[0]
            if zone_h > 0 and zone_w > 0:
                u_mask_full = umbrella_layer.umbrella_mask((h, w))
                umb_in_zone = u_mask_full[zone_bb[1]:zone_bb[3], zone_bb[0]:zone_bb[2]]
                umb_frac = float(umb_in_zone.sum()) / (zone_h * zone_w)

        self._update_speed_history(zone.name, mean_speed)
        self._update_vector_history(zone.name, mean_vx, mean_vy)

        return ZoneMetrics(
            zone_name=zone.name,
            mean_speed=mean_speed,
            mean_divergence=mean_divergence,
            mean_curl=mean_curl,
            mean_coherence=mean_coherence,
            mean_variance=mean_variance,
            stop_go_score=stop_go,
            counterflow_score=cf_score,
            oscillation_symmetry_score=oscillation,
            turbulence_index=turb,
            active_cells=n_active,
            umbrella_occluded_frac=umb_frac,
            mean_density=mean_density,
            crowd_pressure=mean_pressure,
        )

    # ------------------------------------------------------------------
    # Stop-and-go wave detection
    # ------------------------------------------------------------------

    def _update_speed_history(self, zone_name: str, speed: float) -> None:
        if zone_name not in self._speed_history:
            self._speed_history[zone_name] = deque(maxlen=self._SPEED_BUFFER_LEN)
        self._speed_history[zone_name].append(speed)

    def _compute_stop_go(self, zone_name: str, current_speed: float) -> float:
        """
        Stop-and-go score: maximum negative temporal autocorrelation of zone
        mean speed at configured lags.  High negative autocorrelation at short
        lags = periodic halting propagating backwards through a queue.

        Returns a value in [0, 1]; 0 = no stop-go pattern detected.
        """
        buf = self._speed_history.get(zone_name)
        if buf is None or len(buf) < max(self.stop_go_lags) + 5:
            return 0.0

        speeds = np.array(buf, dtype=np.float32)
        s_mean = speeds.mean()
        if s_mean < 1e-6:
            return 0.0

        s_norm = speeds - s_mean
        var    = float(np.dot(s_norm, s_norm))
        if var < 1e-9:
            return 0.0

        # Full correlation for all needed lags
        corr = np.correlate(s_norm, s_norm, mode='full')
        mid  = len(s_norm) - 1
        # Normalized autocorrelation coefficients at each lag
        scores = []
        for lag in self.stop_go_lags:
            if lag < len(s_norm):
                ac = corr[mid - lag] / var
                scores.append(max(0.0, -float(ac)))   # negative autocorr → positive score

        return float(np.clip(max(scores) if scores else 0.0, 0.0, 1.0))

    def _update_vector_history(self, zone_name: str, vx: float, vy: float) -> None:
        if zone_name not in self._vector_history:
            self._vector_history[zone_name] = deque(maxlen=self._SPEED_BUFFER_LEN)
        self._vector_history[zone_name].append((vx, vy))

    def _compute_oscillation_symmetry(
        self, zone_name: str, vx: float, vy: float, mean_speed: float,
        mean_density: Optional[float],
    ) -> float:
        """Negative vector autocorrelation for dense, nearly halted zones."""
        if (mean_density is None
                or mean_density < self.oscillation_density_threshold
                or mean_speed > self.oscillation_stationary_speed):
            return 0.0
        buf = self._vector_history.get(zone_name)
        if buf is None or len(buf) < max(self.stop_go_lags) + 5:
            return 0.0
        vectors = np.asarray(buf, dtype=np.float32)
        centred = vectors - vectors.mean(axis=0, keepdims=True)
        variance = float(np.sum(centred * centred))
        if variance < 1e-9:
            return 0.0
        scores = []
        for lag in self.stop_go_lags:
            if lag < len(centred):
                autocorr = float(np.sum(centred[:-lag] * centred[lag:]) / variance)
                scores.append(max(0.0, -autocorr))
        return float(np.clip(max(scores) if scores else 0.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Counterflow detection
    # ------------------------------------------------------------------

    def _compute_counterflow(
        self,
        cell_mx: np.ndarray,
        cell_my: np.ndarray,
        in_zone_mask: np.ndarray,
        floor: Optional[float] = None,
    ) -> float:
        """
        Fraction of zone cells whose mean direction deviates by > 90° from
        the zone dominant direction.

        Directly measures whether the planned entry/exit separation is holding:
        if the corridor has two streams moving in opposing directions, counterflow
        will be high even if overall zone speed looks normal.

        Takes the per-cell mean vectors computed once in _build_cell_arrays
        rather than re-reducing the full-resolution field per zone.
        """
        in_z    = in_zone_mask
        n_cells = int(in_z.sum())
        if n_cells == 0:
            return 0.0

        # Dominant direction: magnitude-weighted mean vector
        mags = np.sqrt(cell_mx ** 2 + cell_my ** 2)
        total_mag = float(mags[in_z].sum())
        if total_mag < 1e-6:
            return 0.0

        dom_x = float((cell_mx[in_z] * mags[in_z]).sum()) / total_mag
        dom_y = float((cell_my[in_z] * mags[in_z]).sum()) / total_mag
        dom_norm = float(np.hypot(dom_x, dom_y))
        if dom_norm < 1e-6:
            # Equal opposing flows: pick direction of max-magnitude cell vector as reference axis
            max_idx = mags[in_z].argmax()
            dom_x = float(cell_mx[in_z][max_idx])
            dom_y = float(cell_my[in_z][max_idx])
            dom_norm = float(np.hypot(dom_x, dom_y))
            if dom_norm < 1e-6:
                return 0.0

        dom_x /= dom_norm
        dom_y /= dom_norm

        # Dot product with dominant: negative dot → > 90° deviation
        dots  = cell_mx[in_z] * dom_x + cell_my[in_z] * dom_y
        # Only count cells with meaningful motion
        active = mags[in_z] > (self.min_magnitude if floor is None else floor)
        if active.sum() == 0:
            return 0.0

        opposing = (dots[active] < 0).sum()
        return float(opposing) / float(active.sum())

    # ------------------------------------------------------------------
    # Utility: circular variance → coherence
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_coherence(fx: np.ndarray, fy: np.ndarray) -> float:
        """
        Magnitude-weighted directional coherence of a set of flow vectors.

        Returns 1 − circular_variance, in [0, 1].
          1 = all vectors point the same way
          0 = vectors cancel out entirely (maximum incoherence)

        Uses the same formula as the existing OpticalFlowCrushDetector to
        keep results comparable.
        """
        mag   = np.sqrt(fx ** 2 + fy ** 2)
        total = float(mag.sum())
        if total <= 1e-9:
            return 1.0   # static cell → coherent by convention

        mean_x = float(fx.sum()) / total
        mean_y = float(fy.sum()) / total
        resultant = min(1.0, float(np.hypot(mean_x, mean_y)))
        return resultant   # = 1 - circular_variance
