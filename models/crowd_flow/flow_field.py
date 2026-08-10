"""
Dense optical flow computation layer.

Frame pair → dense flow field, computed on a downsampled grayscale frame
(target ~320 px longest side, configurable) for speed, with the field scaled
back to source coordinates.

Global motion compensation (GMC) counteracts apparent whole-frame motion
caused by camera sway on pole-mounted CCTV.  Dominant global translation is
estimated from ORB feature matches on background (non-masked) regions, with a
modal-flow-vector fallback when fewer than MIN_ORB_INLIERS inliers are
available (feature-poor conditions: night, heavy rain, fog).  The applied
correction and its magnitude are logged; a large persistent correction
(> gmc_warn_threshold_px for > 5 seconds) is logged as WARNING because it
indicates a camera mount requiring physical maintenance, not a crowd event.

GMC estimates are validated before they are applied.  On a scene where moving
objects (vehicles, a dense crowd) supply most of the ORB features, RANSAC
happily returns the *objects'* motion as the "global" transform; subtracting
it injects a large uniform velocity into the static background and destroys
the flow field.  Two guards reject such estimates:

  Inlier ratio
    The RANSAC inlier set must be at least GMC_MIN_INLIER_RATIO of all
    matches.  A genuine background transform is supported by most features;
    an object transform is supported by a minority.

  Magnitude cap
    A correction larger than ``gmc_max_correction_px`` (source pixels) is not
    physically plausible camera sway and is rejected outright.

  Does it actually help?
    The decisive test, applied last.  Global motion compensation exists to
    quiet the background: after subtracting a correct estimate, the field's
    median magnitude must go DOWN.  A correction that leaves the field noisier
    than it found it is wrong by definition, whatever the feature matcher
    thought.

    This catches the failure the other two guards cannot see.  On a static
    camera, ORB happily returns a sub-pixel translation with a 0.96 inlier
    ratio — internally consistent, small enough to look plausible, and pure
    noise, because keypoint localisation at the downsampled compute
    resolution is not accurate to a quarter of a pixel.  Subtracting it turns
    a still frame into one where every pixel is moving in the same direction:
    the whole field lights up, and every downstream metric reads motion in a
    scene where nothing moved.

A rejected estimate yields gmc_method == "rejected" and no correction — the
raw flow is used unchanged, which is always safer than subtracting a wrong
global vector.

Temporal smoothing uses an exponential moving average (EMA) over the flow
field at compute resolution.  alpha ≈ 0.4 at 30 fps gives roughly a 2.5-frame
effective smoothing window.

Known limitations (flagged, not silently elided):

  Rain streaks
    Near-vertical, spatially-uniform, high-magnitude flow.  The heuristic also
    fires on genuine rapid downward crowd movement.  The flag is advisory; do
    not suppress flow on this flag alone — log it and let operators decide.

  Low light
    Brightness-constancy degrades when gradient magnitudes are small.  The flag
    propagates to downstream zone-metric outputs as a data-quality annotation.

  Brightness jumps
    Clouds, floodlight switching → single-frame pair that violates brightness
    constancy.  The affected frame is suppressed: the prior smoothed field is
    returned unchanged rather than emitting a corrupted update.

    Detected as a shift in the frame's *mean level*, not as the mean absolute
    inter-frame difference.  Mean absolute difference grows with scene motion,
    so on a busy low-frame-rate camera it crosses any useful threshold on
    ordinary traffic and suppresses a large fraction of frames.  A global
    illumination change moves the mean level; a moving crowd does not.

  Temporal discontinuity (scene cuts, time-lapse, dropped frames)
    Brightness constancy assumes the two frames are adjacent in time and show
    the same scene.  Compilation footage, time-lapsed CCTV exports, and hard
    cuts violate that: DIS still returns a large, smooth, entirely fictitious
    field, and every downstream metric is computed on it without complaint.

    Detected by warping the previous frame with the computed flow and
    measuring how much of the inter-frame difference the flow actually
    explains (flow_reliability, 0-1).  Genuine motion is largely explained;
    unrelated frames are not.  Low reliability is FLAGGED, not suppressed —
    suppressing would freeze the field permanently on a source that is
    discontinuous throughout, which hides the problem instead of surfacing it.

  CCTV compression artifacts
    DIS is substantially less susceptible than Farneback because it works on
    image pyramids rather than raw pixel differences.  No specific heuristic is
    applied; this is documented, not assumed away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DIS_PRESETS: dict[str, int] = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast":      cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium":    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}

# Minimum ORB inliers needed to trust the affine estimate.
_MIN_ORB_INLIERS: int = 10

# Fraction of ORB matches that must be RANSAC inliers for the estimated
# transform to be accepted as *background* motion.  Below this, the dominant
# consistent motion belongs to objects in the scene, not to the camera.
_GMC_MIN_INLIER_RATIO: float = 0.55

# Continuous seconds of large GMC correction before a mount WARNING is logged.
_GMC_WARN_DURATION_SEC: float = 5.0

# Mean absolute inter-frame difference (grey levels, 0-255) below which the
# reliability check is not meaningful.  On a still or near-still scene the
# frame difference is sensor noise and compression, which no flow field can
# or should explain — reliability is legitimately ~0 there, and flagging it
# as a scene cut would mark every quiet camera as broken.  A real cut moves
# this figure into the tens.
_DISCONTINUITY_MIN_BASELINE: float = 8.0

# Far-field refinement: an ROI smaller than this on either side is not worth a
# second flow pass, and DIS is unreliable on tiny inputs regardless.
_FAR_FIELD_MIN_ROI_PX: int = 64

# Minimum resolution gain over the base pass before the refinement earns its
# cost.  Below this the second pass is recomputing what the first already
# resolved, at roughly the price of the first.
_FAR_FIELD_MIN_GAIN: float = 1.5

# Mean absolute inter-frame difference below which the two frames are treated
# as duplicates (a frozen source, or a container padded to a higher nominal
# frame rate than the content).
_FROZEN_FRAME_BASELINE: float = 0.05

# Consecutive duplicate frames before the source is reported as frozen.  One
# duplicate is an ordinary dropped frame and not worth a warning.
_FROZEN_RUN_FRAMES: int = 10


@dataclass
class FlowResult:
    """
    Output of FlowField.compute().

    field_xy is in SOURCE pixel coordinates / frame — the field has already
    been scaled back from the compute (downsampled) resolution.  Downstream
    consumers do not need to know at what resolution flow was computed.
    """
    field_xy: np.ndarray                    # (H_src, W_src, 2), float32, px/frame
    scale_x: float                          # src_W / compute_W
    scale_y: float                          # src_H / compute_H
    gmc_applied: bool
    gmc_translation_px: tuple[float, float] # (dx, dy) subtracted (source px)
    gmc_method: str                         # "orb" | "modal" | "rejected" | "none"
    is_rain_flagged: bool
    is_lowlight_flagged: bool
    is_brightness_suppressed: bool          # True → prior smoothed field returned
    compute_hw: tuple[int, int]             # (H, W) at which flow was computed
    frame_mean_gradient: float              # diagnostic; 0.0 when suppressed
    frame_mean_magnitude: float             # diagnostic
    flow_reliability: float = 1.0           # [0,1]; fraction of the inter-frame
                                            # difference the flow explains
    is_discontinuity_flagged: bool = False  # True → frames are not temporally
                                            # adjacent (cut / time-lapse);
                                            # the field is not trustworthy


class FlowField:
    """
    Manages dense optical flow computation for one camera stream.

    Not thread-safe — each stream needs its own FlowField instance.

    Usage::

        ff = FlowField(backend="dis", dis_preset="medium")
        result = ff.compute(prev_frame, curr_frame, exclusion_mask=vehicle_mask)
        # result.field_xy: (H, W, 2) in source px / frame
    """

    def __init__(
        self,
        backend: str = "dis",
        dis_preset: str = "medium",
        target_px: int = 320,
        temporal_smooth_alpha: float = 0.4,
        global_motion_compensation: bool = True,
        gmc_warn_threshold_px: float = 3.0,
        gmc_max_correction_px: float = 8.0,
        gmc_min_improvement: float = 0.9,
        rain_mag_threshold: float = 8.0,
        lowlight_gradient_threshold: float = 12.0,
        brightness_jump_threshold: float = 30.0,
        min_flow_reliability: float = 0.35,
        far_field: bool = True,
        far_field_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.45),
        far_field_target_px: int = 960,
        far_field_blend_px: int = 48,
    ) -> None:
        if backend not in ("dis", "farneback"):
            raise ValueError(
                f"backend must be 'dis' or 'farneback', got {backend!r}"
            )
        if dis_preset not in _DIS_PRESETS:
            raise ValueError(
                f"dis_preset must be one of {list(_DIS_PRESETS)}, got {dis_preset!r}"
            )

        self.backend = backend
        self.dis_preset = dis_preset
        self.target_px = target_px
        self.alpha = temporal_smooth_alpha
        self.gmc = global_motion_compensation
        self.gmc_warn_threshold_px = gmc_warn_threshold_px
        self.gmc_max_correction_px = gmc_max_correction_px
        self.gmc_min_improvement = gmc_min_improvement
        self.rain_mag_threshold = rain_mag_threshold
        self.lowlight_gradient_threshold = lowlight_gradient_threshold
        self.brightness_jump_threshold = brightness_jump_threshold
        self.min_flow_reliability = min_flow_reliability
        self.far_field = far_field
        self.far_field_roi = tuple(far_field_roi)
        self.far_field_target_px = far_field_target_px
        self.far_field_blend_px = far_field_blend_px

        if not (0.0 <= self.far_field_roi[0] < self.far_field_roi[2] <= 1.0
                and 0.0 <= self.far_field_roi[1] < self.far_field_roi[3] <= 1.0):
            raise ValueError(
                "far_field_roi must be normalised (x1, y1, x2, y2) with "
                f"x1 < x2 and y1 < y2 inside [0, 1]; got {self.far_field_roi!r}"
            )

        # Source-quality warnings are logged once per run, not once per frame.
        self._discontinuity_warned: bool = False
        self._frozen_warned: bool = False
        self._frozen_run: int = 0
        self._far_field_skipped_warned: bool = False
        self._far_field_logged: bool = False

        # Blend-weight cache, keyed by (shape, blend_px, feather flags).
        self._blend_cache: dict[tuple, np.ndarray] = {}

        # Lazily initialised
        self._dis: Optional[cv2.DISOpticalFlow] = None
        self._orb: Optional[cv2.ORB] = None
        self._bf:  Optional[cv2.BFMatcher] = None

        # Smoothing state
        self._smoothed_field: Optional[np.ndarray] = None  # at compute resolution

        # GMC warning state
        self._gmc_large_since: Optional[float] = None
        self._gmc_warned: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        exclusion_mask: Optional[np.ndarray] = None,
        timestamp_sec: float = 0.0,
    ) -> FlowResult:
        """
        Compute dense optical flow from prev_frame to curr_frame.

        Parameters
        ----------
        prev_frame, curr_frame:
            BGR, uint8, (H, W, 3).  Must be the same shape.
        exclusion_mask:
            uint8 (H, W).  1 = exclude this pixel from flow analysis (vehicles,
            hard-masked regions).  Flow vectors are zeroed here, and these
            pixels are excluded from ORB feature detection for GMC.
            Pass None to use every pixel.
        timestamp_sec:
            Video timestamp of curr_frame, used for GMC mount-warning timing.

        Returns
        -------
        FlowResult with field_xy in source pixel coordinates / frame.
        """
        self._ensure_built()
        h_src, w_src = prev_frame.shape[:2]

        # Convert to grayscale -----------------------------------------------
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Brightness-jump suppression ----------------------------------------
        # Measured as a shift in the frame's mean LEVEL, which only a global
        # illumination change produces.  Mean *absolute* difference — the
        # obvious alternative — also grows with scene motion, so on a busy or
        # low-frame-rate camera it suppresses a large fraction of perfectly
        # good frames and freezes the flow field at its last value.
        level_shift = abs(float(curr_gray.mean()) - float(prev_gray.mean()))
        if level_shift > self.brightness_jump_threshold:
            logger.debug(
                "Brightness jump detected (mean level shift=%.1f > %.1f). "
                "Suppressing flow update; returning prior smoothed field.",
                level_shift, self.brightness_jump_threshold,
            )
            return self._prior_or_zero(h_src, w_src)

        # Gradient-based diagnostics (on curr frame) -------------------------
        sob_x = cv2.Sobel(curr_gray, cv2.CV_32F, 1, 0, ksize=3)
        sob_y = cv2.Sobel(curr_gray, cv2.CV_32F, 0, 1, ksize=3)
        frame_mean_gradient = float(np.mean(np.sqrt(sob_x ** 2 + sob_y ** 2)))
        is_lowlight = frame_mean_gradient < self.lowlight_gradient_threshold
        if is_lowlight:
            logger.debug(
                "Low-light condition (mean_gradient=%.2f < %.2f); "
                "flow angular noise will be elevated.",
                frame_mean_gradient, self.lowlight_gradient_threshold,
            )

        # Downsample ---------------------------------------------------------
        prev_small, sx, sy = self._downsample(prev_gray)
        curr_small, _, _   = self._downsample(curr_gray)
        h_c, w_c = prev_small.shape

        # Downsample exclusion mask to compute resolution --------------------
        bg_mask_small: Optional[np.ndarray] = None
        if exclusion_mask is not None:
            bg_mask_small = cv2.resize(
                exclusion_mask.astype(np.uint8),
                (w_c, h_c),
                interpolation=cv2.INTER_NEAREST,
            )

        # Raw flow -----------------------------------------------------------
        raw_flow = self._compute_raw_flow(prev_small, curr_small)  # (h_c, w_c, 2)

        # Second pass at higher resolution over the far field.  Composited into
        # the raw field before anything else reads it, so reliability, GMC and
        # smoothing all see one coherent field rather than two.
        raw_flow = self._refine_far_field(
            prev_gray, curr_gray, raw_flow, sx, sy
        )

        # Validity check on the RAW field, before GMC and smoothing muddy it.
        # A low reliability only means something when there was a substantial
        # difference to explain in the first place — see _flow_reliability and
        # _DISCONTINUITY_MIN_BASELINE.
        reliability, baseline = self._flow_reliability(
            prev_small, curr_small, raw_flow
        )
        is_discontinuous = (
            baseline >= _DISCONTINUITY_MIN_BASELINE
            and reliability < self.min_flow_reliability
        )
        if is_discontinuous and not self._discontinuity_warned:
            logger.warning(
                "Flow reliability %.2f is below %.2f while the frames differ "
                "substantially (mean |diff| = %.1f grey levels): the computed "
                "field explains almost none of that difference.  These two "
                "frames are most likely not temporally adjacent — a scene "
                "cut, a time-lapsed or compilation video, or heavily dropped "
                "frames.  Optical flow is undefined on such input and every "
                "downstream metric for these frames is meaningless.  Use "
                "continuously-recorded footage.  (Logged once per run.)",
                reliability, self.min_flow_reliability, baseline,
            )
            self._discontinuity_warned = True

        # A single duplicate frame is an ordinary dropped frame, not a fault.
        # Only a sustained run of them means the source is actually stalled,
        # so the counter must reach _FROZEN_RUN_FRAMES before warning.
        if baseline < _FROZEN_FRAME_BASELINE:
            self._frozen_run += 1
            if self._frozen_run >= _FROZEN_RUN_FRAMES and not self._frozen_warned:
                logger.warning(
                    "%d consecutive identical frames (mean |diff| = %.4f).  "
                    "The source is frozen, or its container is padded to a "
                    "higher nominal frame rate than the actual content.  "
                    "There is no motion to measure and every flow metric will "
                    "read zero for these frames.  (Logged once per run.)",
                    self._frozen_run, baseline,
                )
                self._frozen_warned = True
        else:
            self._frozen_run = 0

        # Global motion compensation -----------------------------------------
        gmc_applied = False
        gmc_tx, gmc_ty = 0.0, 0.0
        gmc_method = "none"
        if self.gmc:
            tx_orb, ty_orb, n_inliers, inlier_ratio = self._estimate_gmc_orb(
                prev_small, curr_small, bg_mask_small
            )
            if n_inliers >= _MIN_ORB_INLIERS and inlier_ratio >= _GMC_MIN_INLIER_RATIO:
                gmc_tx, gmc_ty = tx_orb, ty_orb
                gmc_method = "orb"
                logger.debug(
                    "GMC(ORB): tx=%.2f ty=%.2f inliers=%d ratio=%.2f",
                    gmc_tx, gmc_ty, n_inliers, inlier_ratio,
                )
            elif n_inliers >= _MIN_ORB_INLIERS:
                # Enough inliers, but they are a minority of the matches: the
                # consistent motion belongs to objects in the scene, not the
                # camera.  Subtracting it would smear the objects' velocity
                # across the static background.
                gmc_method = "rejected"
                logger.debug(
                    "GMC(ORB) rejected: inlier ratio %.2f < %.2f "
                    "(%d/%d matches).  Dominant consistent motion is scene "
                    "objects, not camera sway; no correction applied.",
                    inlier_ratio, _GMC_MIN_INLIER_RATIO,
                    n_inliers, int(round(n_inliers / max(inlier_ratio, 1e-6))),
                )
            else:
                gmc_tx, gmc_ty = self._estimate_gmc_modal(raw_flow, bg_mask_small)
                gmc_method = "modal"
                logger.debug(
                    "GMC fallback to modal (ORB inliers=%d < %d). tx=%.2f ty=%.2f",
                    n_inliers, _MIN_ORB_INLIERS, gmc_tx, gmc_ty,
                )

            # Plausibility cap.  Camera sway of more than a few source pixels
            # per frame is a mount failure, not something to silently subtract.
            # The cap is expressed in source px; convert to compute px.
            max_corr_compute = self.gmc_max_correction_px / max(sx, sy, 1e-6)
            if (gmc_method != "rejected"
                    and float(np.hypot(gmc_tx, gmc_ty)) > max_corr_compute):
                logger.debug(
                    "GMC(%s) rejected: correction %.2f compute-px exceeds the "
                    "%.2f cap (gmc_max_correction_px=%.1f source px).  "
                    "Implausible as camera sway; no correction applied.",
                    gmc_method, float(np.hypot(gmc_tx, gmc_ty)),
                    max_corr_compute, self.gmc_max_correction_px,
                )
                gmc_method = "rejected"

            # Decisive guard: does subtracting this estimate actually quiet
            # the field?  See the module docstring.  Cheap — one median over
            # the compute-resolution field, which is ~320x180.
            if gmc_method != "rejected" and (abs(gmc_tx) > 0.01
                                             or abs(gmc_ty) > 0.01):
                before, after = self._gmc_median_magnitudes(
                    raw_flow, gmc_tx, gmc_ty, bg_mask_small
                )
                if after > before * self.gmc_min_improvement:
                    logger.debug(
                        "GMC(%s) rejected: correction (%.3f, %.3f) would raise "
                        "the field's median magnitude from %.3f to %.3f "
                        "compute-px.  Compensation must reduce background "
                        "motion; this estimate is noise, and applying it would "
                        "make every pixel appear to move.",
                        gmc_method, gmc_tx, gmc_ty, before, after,
                    )
                    gmc_method = "rejected"

            if gmc_method == "rejected":
                gmc_tx, gmc_ty = 0.0, 0.0
            elif abs(gmc_tx) > 0.01 or abs(gmc_ty) > 0.01:
                raw_flow[..., 0] -= gmc_tx
                raw_flow[..., 1] -= gmc_ty
                gmc_applied = True
                # Warn on the SOURCE-pixel magnitude: gmc_warn_threshold_px is
                # documented in source pixels, and the estimate is in compute
                # pixels, so comparing them directly under-reports by the
                # downsample factor.
                self._check_gmc_magnitude(
                    gmc_tx * sx, gmc_ty * sy, timestamp_sec
                )

        # Zero out excluded pixels -------------------------------------------
        if bg_mask_small is not None:
            excl = bg_mask_small.astype(bool)
            raw_flow[excl] = 0.0

        # Temporal smoothing (EMA at compute resolution) ---------------------
        if self._smoothed_field is None or self._smoothed_field.shape != raw_flow.shape:
            self._smoothed_field = raw_flow.copy()
        else:
            cv2.addWeighted(
                raw_flow, self.alpha,
                self._smoothed_field, 1.0 - self.alpha,
                0.0, self._smoothed_field,
            )

        # Scale field back to source resolution ------------------------------
        if sx != 1.0 or sy != 1.0:
            field_src = cv2.resize(
                self._smoothed_field, (w_src, h_src),
                interpolation=cv2.INTER_LINEAR,
            )
            field_src[..., 0] *= sx
            field_src[..., 1] *= sy
        else:
            field_src = self._smoothed_field.copy()

        # Mean magnitude (for rain heuristic and diagnostics) ----------------
        magnitudes = np.sqrt(field_src[..., 0] ** 2 + field_src[..., 1] ** 2)
        frame_mean_magnitude = float(magnitudes.mean())

        is_rain = self._check_rain(field_src, magnitudes, frame_mean_magnitude)

        return FlowResult(
            field_xy=field_src,
            scale_x=sx,
            scale_y=sy,
            gmc_applied=gmc_applied,
            gmc_translation_px=(gmc_tx * sx, gmc_ty * sy),
            gmc_method=gmc_method,
            is_rain_flagged=is_rain,
            is_lowlight_flagged=is_lowlight,
            is_brightness_suppressed=False,
            compute_hw=(h_c, w_c),
            frame_mean_gradient=frame_mean_gradient,
            frame_mean_magnitude=frame_mean_magnitude,
            flow_reliability=reliability,
            is_discontinuity_flagged=is_discontinuous,
        )

    def reset(self) -> None:
        """Reset internal state.  Call when switching to a new video/stream."""
        self._smoothed_field = None
        self._gmc_large_since = None
        self._gmc_warned = False
        self._discontinuity_warned = False
        self._frozen_warned = False
        self._frozen_run = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        if self.backend == "dis" and self._dis is None:
            self._dis = cv2.DISOpticalFlow_create(_DIS_PRESETS[self.dis_preset])
        if self.gmc and self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=500)
            self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _downsample(
        self, frame_gray: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Resize to target_px longest side.  Returns (small, scale_x, scale_y)."""
        h, w = frame_gray.shape
        scale = self.target_px / max(h, w)
        if scale >= 1.0:
            return frame_gray, 1.0, 1.0
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        small = cv2.resize(frame_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return small, w / new_w, h / new_h

    # ------------------------------------------------------------------
    # Far-field refinement
    # ------------------------------------------------------------------

    def _refine_far_field(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        raw_flow: np.ndarray,
        sx: float,
        sy: float,
    ) -> np.ndarray:
        """
        Recompute flow over the far field at higher resolution and blend it in.

        Why this exists
        ---------------
        The whole frame is downsampled to ``target_px`` before flow is
        computed, which sets one pixel budget for a scene that does not have
        one scale.  Perspective means a pedestrian near the camera spans a
        few hundred source pixels while one at the back of the same crowd
        spans a dozen.  At a 480 px compute width the distant pedestrian is
        two or three pixels across — below the size of the patch DIS matches
        on — so the far field returns flow interpolated from its
        surroundings rather than measured from the people actually there.

        Raising ``target_px`` for the whole frame fixes it at quadratic cost,
        almost all of which is spent on the near field that was already
        adequately resolved.  This pass spends the resolution only where the
        deficit is.

        Units
        -----
        The returned field stays in **compute-resolution px/frame**, the same
        units as ``raw_flow``, so everything downstream is unchanged.  The
        crop is measured in its own pixels, converted to source px/frame, then
        into compute px/frame.

        Seams
        -----
        Divergence and curl are spatial derivatives taken by central
        differences.  A hard edge between the two passes would differentiate
        into a line of large fake divergence — on the metric that drives
        crush alerts.  The two fields are therefore feathered together over
        ``far_field_blend_px``, and only on ROI edges that fall inside the
        frame; an edge lying on the frame border has nothing to blend with.
        """
        if not self.far_field:
            return raw_flow

        h_src, w_src = prev_gray.shape[:2]
        h_c, w_c = raw_flow.shape[:2]

        rx1, ry1, rx2, ry2 = self.far_field_roi
        x1 = max(0, min(w_src, int(round(rx1 * w_src))))
        x2 = max(0, min(w_src, int(round(rx2 * w_src))))
        y1 = max(0, min(h_src, int(round(ry1 * h_src))))
        y2 = max(0, min(h_src, int(round(ry2 * h_src))))
        roi_w, roi_h = x2 - x1, y2 - y1
        if roi_w < _FAR_FIELD_MIN_ROI_PX or roi_h < _FAR_FIELD_MIN_ROI_PX:
            return raw_flow

        # Never upsample past the source: interpolated pixels carry no motion
        # information the original frame did not already have.
        scale = min(1.0, self.far_field_target_px / max(roi_w, roi_h))
        rw = max(1, int(round(roi_w * scale)))
        rh = max(1, int(round(roi_h * scale)))

        # How much of this ROI the base pass already resolved.  If the second
        # pass would not be meaningfully finer, it is pure cost.
        base_w = roi_w / sx
        base_h = roi_h / sy
        if rw <= base_w * _FAR_FIELD_MIN_GAIN or rh <= base_h * _FAR_FIELD_MIN_GAIN:
            if not self._far_field_skipped_warned:
                logger.info(
                    "Far-field refinement skipped: the ROI is already resolved "
                    "at %.0fx%.0f by the base pass and the refinement would "
                    "give %dx%d, less than a %.2fx gain.  Raise "
                    "far_field_target_px, shrink far_field_roi, or lower "
                    "downsample_target_px if you want it to engage.",
                    base_w, base_h, rw, rh, _FAR_FIELD_MIN_GAIN,
                )
                self._far_field_skipped_warned = True
            return raw_flow

        crop_prev = prev_gray[y1:y2, x1:x2]
        crop_curr = curr_gray[y1:y2, x1:x2]
        if (rw, rh) != (roi_w, roi_h):
            crop_prev = cv2.resize(crop_prev, (rw, rh), interpolation=cv2.INTER_AREA)
            crop_curr = cv2.resize(crop_curr, (rw, rh), interpolation=cv2.INTER_AREA)

        fine = self._compute_raw_flow(crop_prev, crop_curr)   # crop px/frame

        # crop px/frame → source px/frame → compute px/frame
        fine[..., 0] *= (roi_w / rw) / sx
        fine[..., 1] *= (roi_h / rh) / sy

        # Footprint of the ROI in compute coordinates.
        cx1 = max(0, min(w_c, int(round(x1 / sx))))
        cx2 = max(0, min(w_c, int(round(x2 / sx))))
        cy1 = max(0, min(h_c, int(round(y1 / sy))))
        cy2 = max(0, min(h_c, int(round(y2 / sy))))
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < 2 or ch < 2:
            return raw_flow

        fine = cv2.resize(fine, (cw, ch), interpolation=cv2.INTER_LINEAR)

        # The blend margin is configured in source px; the blend happens at
        # compute resolution.
        blend_c = max(0, int(round(self.far_field_blend_px / max(sx, sy, 1e-6))))
        weights = self._blend_weights(
            (ch, cw),
            blend_px=blend_c,
            feather_top=cy1 > 0,
            feather_bottom=cy2 < h_c,
            feather_left=cx1 > 0,
            feather_right=cx2 < w_c,
        )[..., None]

        patch = raw_flow[cy1:cy2, cx1:cx2]
        raw_flow[cy1:cy2, cx1:cx2] = patch * (1.0 - weights) + fine * weights

        if not self._far_field_logged:
            logger.info(
                "Far-field refinement active: ROI %dx%d source px recomputed "
                "at %dx%d (base pass gave it %.0fx%.0f), blended over %d "
                "compute px.",
                roi_w, roi_h, rw, rh, base_w, base_h, blend_c,
            )
            self._far_field_logged = True

        return raw_flow

    def _blend_weights(
        self,
        shape: tuple[int, int],
        blend_px: int,
        feather_top: bool,
        feather_bottom: bool,
        feather_left: bool,
        feather_right: bool,
    ) -> np.ndarray:
        """
        Weight map for the refined patch: 1 in the interior, ramping to 0 over
        ``blend_px`` on each feathered edge.  Cached — it depends only on the
        geometry, which does not change between frames.
        """
        key = (shape, blend_px, feather_top, feather_bottom,
               feather_left, feather_right)
        cached = self._blend_cache.get(key)
        if cached is not None:
            return cached

        h, w = shape
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)
        if blend_px > 0:
            ramp_y = np.clip(np.arange(h, dtype=np.float32) / blend_px, 0.0, 1.0)
            ramp_x = np.clip(np.arange(w, dtype=np.float32) / blend_px, 0.0, 1.0)
            if feather_top:
                wy = np.minimum(wy, ramp_y)
            if feather_bottom:
                wy = np.minimum(wy, ramp_y[::-1])
            if feather_left:
                wx = np.minimum(wx, ramp_x)
            if feather_right:
                wx = np.minimum(wx, ramp_x[::-1])

        weights = np.outer(wy, wx).astype(np.float32)
        self._blend_cache[key] = weights
        return weights

    def _compute_raw_flow(
        self, prev_small: np.ndarray, curr_small: np.ndarray
    ) -> np.ndarray:
        if self.backend == "dis":
            return self._dis.calc(prev_small, curr_small, None)
        return cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    @staticmethod
    def _gmc_median_magnitudes(
        raw_flow: np.ndarray,
        tx: float,
        ty: float,
        bg_mask_small: Optional[np.ndarray],
    ) -> tuple[float, float]:
        """
        Median field magnitude before and after subtracting (tx, ty).

        The MEDIAN, not the mean: on a scene with a few moving objects against
        a static background, the median tracks the background — which is
        exactly what compensation is supposed to flatten — while the mean is
        pulled around by the objects themselves, so a correction that wrecked
        the background could still lower it.
        """
        fx, fy = raw_flow[..., 0], raw_flow[..., 1]
        if bg_mask_small is not None:
            keep = bg_mask_small == 0
            if keep.any():
                fx, fy = fx[keep], fy[keep]
        before = float(np.median(np.hypot(fx, fy)))
        after = float(np.median(np.hypot(fx - tx, fy - ty)))
        return before, after

    @staticmethod
    def _flow_reliability(
        prev_small: np.ndarray,
        curr_small: np.ndarray,
        flow: np.ndarray,
    ) -> tuple[float, float]:
        """
        Fraction of the inter-frame difference that the computed flow explains.

        Warps prev_small forward by the flow and compares the result to
        curr_small.  If the flow is a correct motion estimate, the warped
        frame closely matches and the residual is small.  If the two frames
        show unrelated content, no field can explain the difference and the
        residual stays close to the unwarped baseline.

            reliability = 1 − residual / baseline,  clipped to [0, 1]

        ~1.0 = flow accounts for the change (normal motion)
        ~0.0 = flow accounts for nothing

        Returns (reliability, baseline).  The caller needs the baseline
        because reliability alone does not distinguish a scene cut from a
        still scene: on a static camera the frame difference is noise, the
        flow is correctly ~0, and it explains none of that noise — giving a
        reliability of ~0 for a perfectly healthy stream.  Only a large
        baseline makes a low reliability meaningful.

        Costs one remap at compute resolution (~0.2 ms at 320×180).
        """
        h, w = prev_small.shape[:2]
        prev_f = prev_small.astype(np.float32)
        curr_f = curr_small.astype(np.float32)

        baseline = float(np.abs(curr_f - prev_f).mean())
        if baseline < 1e-3:
            return 1.0, baseline   # identical frames: nothing to explain

        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        warped = cv2.remap(
            prev_f, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        residual = float(np.abs(curr_f - warped).mean())
        return float(np.clip(1.0 - residual / baseline, 0.0, 1.0)), baseline

    def _estimate_gmc_orb(
        self,
        prev_small: np.ndarray,
        curr_small: np.ndarray,
        bg_mask_small: Optional[np.ndarray],
    ) -> tuple[float, float, int, float]:
        """
        Estimate global translation via ORB feature matching + RANSAC.

        Returns (tx, ty, n_inliers, inlier_ratio).  n_inliers == 0 means ORB
        found nothing usable.

        inlier_ratio is n_inliers / n_matches.  The caller uses it to decide
        whether the estimated transform describes the background (most
        features agree) or merely the largest moving object in the scene (a
        minority of features agree) — see _GMC_MIN_INLIER_RATIO.

        bg_mask_small: 1 = excluded pixel (vehicle/hard-mask).  ORB features
        are detected only in the *background* (unmasked) pixels.
        """
        orb_mask: Optional[np.ndarray] = None
        if bg_mask_small is not None:
            # ORB detectAndCompute mask: 255 = detect here, 0 = skip
            orb_mask = ((1 - bg_mask_small) * 255).astype(np.uint8)

        kp_p, desc_p = self._orb.detectAndCompute(prev_small, orb_mask)
        kp_c, desc_c = self._orb.detectAndCompute(curr_small, orb_mask)

        if (desc_p is None or desc_c is None
                or len(kp_p) < 4 or len(kp_c) < 4):
            return 0.0, 0.0, 0, 0.0

        matches = self._bf.match(desc_p, desc_c)
        if len(matches) < 4:
            return 0.0, 0.0, 0, 0.0

        pts_p = np.float32([kp_p[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_c = np.float32([kp_c[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, inlier_mask = cv2.estimateAffinePartial2D(
            pts_p, pts_c,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )

        if M is None or inlier_mask is None:
            return 0.0, 0.0, 0, 0.0

        n_inliers = int(inlier_mask.sum())
        inlier_ratio = n_inliers / float(len(matches))
        if n_inliers < _MIN_ORB_INLIERS:
            return 0.0, 0.0, n_inliers, inlier_ratio

        return float(M[0, 2]), float(M[1, 2]), n_inliers, inlier_ratio

    def _estimate_gmc_modal(
        self,
        raw_flow: np.ndarray,
        bg_mask_small: Optional[np.ndarray],
    ) -> tuple[float, float]:
        """
        Estimate global translation as the modal flow vector.

        Bins flow into 0.5-px histogram buckets over the range ±20 px.
        Operates only on unmasked pixels.

        The bin edges are offset by half a bin width so that one bin is
        *centred* on zero.  With edges on the integers-and-halves, zero sits
        on a bin boundary and a perfectly static camera yields a modal
        estimate of 0.25 px/frame — a bias that is then subtracted from every
        vector on every frame, and multiplied by the downsample factor on the
        way back to source resolution.
        """
        fx, fy = raw_flow[..., 0], raw_flow[..., 1]
        if bg_mask_small is not None:
            valid = bg_mask_small == 0
        else:
            valid = np.ones(fx.shape, dtype=bool)

        if not valid.any():
            return 0.0, 0.0

        # Edges offset by half a bin so that [-0.25, 0.25) — centred on zero —
        # is a bin.  See the docstring.
        bins = np.arange(-20.25, 20.75 + 1e-9, 0.5)
        hist_x, _ = np.histogram(fx[valid], bins=bins)
        hist_y, _ = np.histogram(fy[valid], bins=bins)
        # Bin centre = left edge + half-width
        tx = float(bins[hist_x.argmax()]) + 0.25
        ty = float(bins[hist_y.argmax()]) + 0.25
        return tx, ty

    def _check_gmc_magnitude(
        self, tx: float, ty: float, timestamp_sec: float
    ) -> None:
        mag = float(np.hypot(tx, ty))
        if mag > self.gmc_warn_threshold_px:
            if self._gmc_large_since is None:
                self._gmc_large_since = timestamp_sec
            duration = timestamp_sec - self._gmc_large_since
            if duration > _GMC_WARN_DURATION_SEC and not self._gmc_warned:
                logger.warning(
                    "Camera sway: GMC correction has been %.1f px/frame for %.0f s "
                    "(threshold %.1f px).  A large persistent correction indicates "
                    "camera mount instability requiring physical attention — "
                    "this is not a crowd event.",
                    mag, duration, self.gmc_warn_threshold_px,
                )
                self._gmc_warned = True
        else:
            self._gmc_large_since = None
            self._gmc_warned = False

    def _check_rain(
        self,
        field_src: np.ndarray,
        magnitudes: np.ndarray,
        mean_mag: float,
    ) -> bool:
        """
        Rain-streak heuristic: high-magnitude, spatially-uniform, vertical flow.

        Fires on genuine rapid downward crowd movement too.  Advisory only.
        """
        if mean_mag <= self.rain_mag_threshold:
            return False

        mag_var    = float(np.var(magnitudes))
        mean_vy    = float(np.mean(np.abs(field_src[..., 1])))
        mean_vx    = float(np.mean(np.abs(field_src[..., 0])))
        is_uniform = mag_var < (mean_mag * 0.5)
        is_vertical = (mean_vy / max(mean_vx, 1e-6)) > 2.0

        if is_uniform and is_vertical:
            logger.warning(
                "Rain-streak heuristic triggered: mean_mag=%.1f px/frame, "
                "mag_var=%.2f, mean_vy/mean_vx=%.1f.  Flow may be corrupted by "
                "rain streaks (Nashik monsoon tail).  This also fires on genuine "
                "rapid downward crowd movement — treat as advisory only.",
                mean_mag, mag_var, mean_vy / max(mean_vx, 1e-6),
            )
            return True
        return False

    def _prior_or_zero(self, h_src: int, w_src: int) -> FlowResult:
        """Return the prior smoothed field (or zeros if not yet computed)."""
        if self._smoothed_field is not None:
            sx = float(w_src) / self._smoothed_field.shape[1]
            sy = float(h_src) / self._smoothed_field.shape[0]
            field_src = cv2.resize(
                self._smoothed_field, (w_src, h_src),
                interpolation=cv2.INTER_LINEAR,
            )
            field_src[..., 0] *= sx
            field_src[..., 1] *= sy
        else:
            field_src = np.zeros((h_src, w_src, 2), dtype=np.float32)

        # Report the magnitude of the field actually returned.  Hard-coding
        # 0.0 here writes "no motion" into the metrics log for every
        # suppressed frame, which is indistinguishable from a genuinely still
        # scene when reading the CSV back.
        mean_mag = float(np.sqrt(
            field_src[..., 0] ** 2 + field_src[..., 1] ** 2
        ).mean())

        return FlowResult(
            field_xy=field_src,
            scale_x=1.0, scale_y=1.0,
            gmc_applied=False,
            gmc_translation_px=(0.0, 0.0),
            gmc_method="none",
            is_rain_flagged=False,
            is_lowlight_flagged=False,
            is_brightness_suppressed=True,
            compute_hw=(h_src, w_src),
            frame_mean_gradient=0.0,
            frame_mean_magnitude=mean_mag,
        )
