"""
Flow field visualisation helpers.

All functions take an existing BGR frame and return an annotated copy — they
never modify the input in place.  Rendering is a separate concern from metric
computation; disable it entirely (pass visualise=False to DenseFlowAnalyser)
for headless / multi-stream deployments to save ~5-12 ms/frame.

HSV flow overlay
----------------
Hue        = direction (0° = right, 90° = down, 180° = left, 270° = up)
Saturation = full
Value      = full

Magnitude is carried by the per-pixel ALPHA (and by arrow length), not by
Value.  Encoding it in Value as well faded slow and distant movement twice
over — dim colour, then transparent — so the far half of a traffic scene
rendered as a barely-visible haze.  Colour now says direction, opacity says
speed, and each says one thing.

The overlay is blended over the source frame so the scene stays visible for
spatial reference.  When PeopleOverlay is running, the tint is additionally
restricted to confirmed moving people's boxes — the field measures real
apparent motion on bystanders' sway and structure texture too, and without
the restriction "coloured" does not mean "going somewhere".

Divergence heatmap
------------------
Red  = negative divergence (convergence → compression → crush risk)
Blue = positive divergence (expansion → people spreading out)
Zero = white (neutral)

The colour scale is zero-centred and symmetric around ±div_scale.  The sign
convention (red = bad = compression = negative div) is documented here and
matches crowd_metrics.py exactly.  If you see the heatmap predominantly red
in a static scene with no crowd, the sign convention is wrong.

When PeopleOverlay is running, the tint is additionally restricted to cells
under confirmed moving people's boxes; a confirmed stationary person gets a
flat grey marker instead of flow colour.  Divergence about nobody in
particular — textured floor tiles clearing the motion floor — is not a
crowd signal and no longer paints.

Colour choices are rendering decisions only; the pipeline's Detection and Alert
objects carry computed values, not colours.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Matches pipeline/annotate.py COLOR_MAP convention (BGR)
_ZONE_OUTLINE_COLOR   = (0, 230, 255)   # amber
_ZONE_TEXT_COLOR      = (255, 255, 255)
_VEHICLE_BOX_COLOR    = (0, 191, 255)   # goldenrod
_UMBRELLA_BOX_COLOR   = (200, 65, 203)  # magenta

# Divergence heatmap colour stops (BGR)
_DIV_COLOR_NEG  = (0,   0,   220)   # red   (compression)
_DIV_COLOR_ZERO = (240, 240, 240)   # near-white (neutral)
_DIV_COLOR_POS  = (220, 50,  0  )   # blue  (expansion)

# Flat indicator for a confirmed-but-stationary person inside the heatmap
# (BGR muted grey). Deliberately NOT the crush/compression red and not any hue
# the flow colour wheel produces: it must read as "a detected person who is
# standing still", distinct both from "no detection here" (untouched frame)
# and from "moving person with flow colour". The box outline on top of this
# fill still comes from PeopleOverlay (green when seen, tan while coasting).
_PERSON_STATIONARY_FILL = (150, 150, 150)

# Opacity of that stationary fill. Low enough to keep the person visible,
# high enough to read as a deliberate marker rather than sensor noise.
_PERSON_STATIONARY_ALPHA = 0.30


class FlowVisualiser:
    """
    Produces annotated frames from flow fields, metrics, and detections.

    All methods return a uint8 BGR ndarray the same shape as the input frame.
    """

    def __init__(
        self,
        flow_alpha: float = 0.85,
        flow_min_alpha: float = 0.5,
        heatmap_alpha: float = 0.55,
        arrow_step: int = 20,
        max_magnitude_px: float = 20.0,
        div_scale: float = 3.0,
        auto_scale_overlay: bool = True,
        min_draw_magnitude_px: float = 0.4,
        background_floor_factor: float = 3.0,
        overlay_work_px: int = 480,
    ) -> None:
        """
        Parameters
        ----------
        flow_alpha:
            Opacity of the HSV overlay at reference magnitude (0 = invisible,
            1 = opaque).
        flow_min_alpha:
            Opacity for motion that only just clears the display floor.  The
            alpha ramp runs between this and flow_alpha rather than from
            zero, so slow or distant movement is still plainly coloured
            instead of a barely-there tint.
        heatmap_alpha:
            Opacity of the divergence heatmap at ±div_scale, applied per cell.
        arrow_step:
            Grid step (pixels) for the sparse arrow overlay.
        max_magnitude_px:
            Reference flow magnitude for colour/opacity scaling.  Used as a
            fixed reference when auto_scale_overlay is False, and as a lower
            bound on the automatic reference when it is True.
        div_scale:
            Divergence values at ±div_scale are mapped to full red/blue.
        auto_scale_overlay:
            Scale the overlay to each frame's own 95th-percentile magnitude
            rather than to a fixed constant.  Pedestrian flow at 30 fps runs
            about 1-3 px/frame, an order of magnitude below the fixed default
            that suits vehicle traffic; against a fixed reference a crowd
            renders at a few percent opacity and is effectively invisible.
            The reference is floored at max_magnitude_px / 8 so that a still
            scene does not have its sensor noise amplified into a full-scale
            display.
        min_draw_magnitude_px:
            Absolute floor, in source px/frame, below which nothing is drawn.
        background_floor_factor:
            The display floor is also required to be this multiple of the
            frame's MEDIAN magnitude.  A fixed absolute floor cannot work
            across sources: dense flow on compressed CCTV has a per-frame
            noise level that varies with bitrate, texture and lighting, and a
            threshold tuned on clean footage lights up every pixel on noisy
            footage.  The median over the whole frame is a robust estimate of
            "what not moving looks like right now" — most of a fixed camera's
            view is background — so requiring a multiple of it adapts the
            floor to each frame instead of assuming one constant fits all.
        overlay_work_px:
            Longest side at which the translucent overlays are rendered before
            being resized onto the frame.  The flow field is computed
            downsampled and upsampled, so full-resolution rendering costs
            arithmetic without adding detail.  Only the final composite runs
            at full resolution, so the source frame stays sharp.
        """
        self.flow_alpha       = flow_alpha
        self.flow_min_alpha   = flow_min_alpha
        self.heatmap_alpha    = heatmap_alpha
        self.arrow_step       = arrow_step
        self.max_magnitude_px = max_magnitude_px
        self.div_scale        = div_scale
        self.auto_scale_overlay    = auto_scale_overlay
        self.min_draw_magnitude_px = min_draw_magnitude_px
        self.background_floor_factor = background_floor_factor
        self.overlay_work_px       = overlay_work_px

    def motion_floor(self, mag: np.ndarray) -> float:
        """
        Magnitude below which motion is treated as background for DISPLAY.

        Display only — metrics are unaffected.  Combines an absolute floor
        with a multiple of this frame's median magnitude, so the overlay
        adapts to each frame's own noise level rather than to a constant
        chosen on some other footage.
        """
        background = float(np.median(mag))
        return max(self.min_draw_magnitude_px,
                   background * self.background_floor_factor)

    def _reference_magnitude(self, mag: np.ndarray) -> float:
        """
        Magnitude that maps to full colour / full opacity for this frame.

        With auto_scale_overlay the reference follows the scene (95th
        percentile of the current field), floored so a static frame's noise is
        not stretched to full scale.  Otherwise the configured constant.
        """
        if not self.auto_scale_overlay:
            return max(self.max_magnitude_px, 1e-6)
        floor = max(self.max_magnitude_px / 8.0, self.min_draw_magnitude_px)
        return float(max(np.percentile(mag, 95), floor))

    # ------------------------------------------------------------------
    # Flow overlays
    # ------------------------------------------------------------------

    @staticmethod
    def _alpha_blend(
        frame: np.ndarray, overlay: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """
        Per-pixel alpha composite of overlay onto frame.

        out = frame + (overlay − frame) × alpha, evaluated with OpenCV's
        SIMD/threaded primitives.  The equivalent NumPy expression allocates
        several full-resolution float32 temporaries per call and measured
        several times slower at 1080p.
        """
        alpha3 = cv2.merge([alpha, alpha, alpha])
        diff = cv2.subtract(overlay, frame, dtype=cv2.CV_32F)
        cv2.multiply(diff, alpha3, dst=diff)
        return cv2.add(frame, diff, dtype=cv2.CV_8U)

    def hsv_flow_overlay(
        self, frame: np.ndarray, flow_xy: np.ndarray,
        person_boxes: Optional[list] = None,
    ) -> np.ndarray:
        """
        HSV colour-wheel overlay blended over the source frame.

        Hue = direction.  Saturation and Value are held at full above the
        motion floor, so direction reads as a vivid colour rather than a wash.

        Magnitude is deliberately NOT encoded in Value.  It used to be, and
        combined with a magnitude-proportional alpha that meant slow or
        distant traffic faded twice over — a dim colour, then made
        transparent on top of it — so a vehicle at half the reference speed
        rendered at roughly a quarter of the intended strength and vanished
        into the road.  Magnitude is already carried by the alpha ramp and by
        arrow length; spending Value on it as well just makes the slower half
        of every scene unreadable.

        The overlay is still blended with a **per-pixel** alpha, so static
        regions keep the source frame unchanged.  A uniform alpha blends the
        HSV image over the whole frame, and a scene that is 90% static comes
        out 90% tinted with the moving subjects no easier to see.

        ``person_boxes`` ([x1, y1, x2, y2] per MOVING confirmed person,
        source px — the same list the divergence heatmap already masks to)
        restricts the tint to those boxes.  The flow field legitimately
        measures sub-pixel apparent motion on swaying bystanders, clothing
        and structure texture, and this overlay painted all of it; scoping
        the alpha to tracked people is what makes "colour = someone is
        actually going somewhere" true.  Pass None (no people tracker
        configured) for the previous scene-wide behaviour, unchanged.
        """
        h, w = frame.shape[:2]

        # Render the overlay at reduced resolution.  The flow field was
        # computed downsampled (typically 320 px) and bilinearly upsampled, so
        # it carries no detail at full resolution — doing the trigonometry and
        # colour conversion on every 1080p pixel is ~36× the arithmetic for an
        # identical picture.  Magnitudes are in source px/frame either way, so
        # resampling the field does not change the values.
        small = self._downscale_field(flow_xy, h, w)
        fx, fy = small[..., 0], small[..., 1]

        # Hue: angle in [0, 179] (OpenCV uint8 hue is 0-179 for 360°)
        angle = np.arctan2(fy, fx)                           # [-π, π]
        hue   = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

        mag   = np.sqrt(fx ** 2 + fy ** 2)

        # Person scoping mask, built BEFORE the normalisation statistics so
        # those statistics describe the region that will actually be painted.
        #
        # Built at SOURCE resolution (pixel-exact), then area-downscaled to
        # the overlay's working resolution -- INTER_AREA yields fractional
        # coverage on box edges, so boxes fade at their border instead of
        # strobing a hard pixel stair.
        keep = None
        if person_boxes is not None:
            keep = np.zeros((h, w), dtype=np.float32)
            for x1, y1, x2, y2 in person_boxes:
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(w, int(x2)), min(h, int(y2))
                if ix2 > ix1 and iy2 > iy1:
                    keep[iy1:iy2, ix1:ix2] = 1.0
            if keep.shape != mag.shape:
                keep = cv2.resize(keep, (mag.shape[1], mag.shape[0]),
                                  interpolation=cv2.INTER_AREA)

        # The reference magnitude is taken over the KEPT pixels when scoping
        # is active. It is the 95th percentile -- "what counts as fast in
        # this frame" -- and computing it over the whole field while painting
        # only people let traffic set the scale for pedestrians: on a road
        # scene the vehicles dominated the percentile, every pedestrian
        # normalised to near zero, and the boxes that were kept rendered
        # almost colourless. Scoped, colour reads as speed relative to the
        # people on screen, which is what the box overlay is claiming.
        #
        # _reference_magnitude keeps its own lower bound, so a frame where
        # everyone is nearly still cannot stretch noise to full scale.
        #
        # An empty selection (no boxes, or all degenerate) falls back to the
        # whole field: alpha is multiplied by an all-zero mask below anyway,
        # so the value cannot reach the output, and this avoids percentile
        # over an empty array.
        stat_sample = mag
        if keep is not None:
            selected = mag[keep > 0.0]
            if selected.size:
                stat_sample = selected

        ref   = self._reference_magnitude(stat_sample)
        norm  = np.clip(mag / ref, 0.0, 1.0)
        # Square-root ramp: perceptually, a linear ramp spends most of its
        # range on the fastest few percent of pixels and leaves everything
        # slower bunched at the bottom.
        norm  = np.sqrt(norm)

        # Full saturation and value: the hue is the message, and dimming it
        # by speed is what made slower traffic disappear.
        sat   = np.full_like(hue, 255)
        val   = np.full_like(hue, 255)

        hsv_img = np.stack([hue, sat, val], axis=-1)
        bgr_flow = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # Per-pixel alpha: 0 below the background floor, then a ramp from
        # flow_min_alpha to flow_alpha.  The floor keeps dense-flow noise on
        # compressed footage from tinting the whole frame; the *minimum*
        # keeps anything that clears the floor clearly coloured rather than
        # a hint of a tint.  Ramping from zero instead meant a vehicle just
        # above the threshold was drawn at nearly nothing, which is how the
        # far carriageway ended up looking static.
        # NOTE the asymmetry with `ref` above: the floor stays SCENE-WIDE on
        # purpose. It is a background-noise estimate (median x factor), and
        # the whole field is the only sample that actually contains
        # background. Restricted to person boxes the median becomes the
        # signal rather than the noise, the floor lands above the magnitudes
        # it exists to admit, and every kept box is suppressed to nothing.
        floor = self.motion_floor(mag)
        span  = max(self.flow_alpha - self.flow_min_alpha, 0.0)
        alpha = (self.flow_min_alpha + norm * span).astype(np.float32)
        alpha[mag < floor] = 0.0

        if keep is not None:
            alpha = alpha * keep

        if bgr_flow.shape[:2] != (h, w):
            bgr_flow = cv2.resize(bgr_flow, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha    = cv2.resize(alpha,    (w, h), interpolation=cv2.INTER_LINEAR)

        return self._alpha_blend(frame, bgr_flow, alpha)

    def _downscale_field(
        self, flow_xy: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Resample the flow field down to overlay_work_px on its longest side."""
        scale = self.overlay_work_px / float(max(h, w))
        if scale >= 1.0:
            return flow_xy
        sw = max(1, int(round(w * scale)))
        sh = max(1, int(round(h * scale)))
        return cv2.resize(flow_xy, (sw, sh), interpolation=cv2.INTER_AREA)

    def sparse_arrow_grid(
        self,
        frame: np.ndarray,
        flow_xy: np.ndarray,
        scale: Optional[float] = None,
        in_place: bool = False,
    ) -> np.ndarray:
        """
        Sparse arrow grid for human-readable direction display.

        Arrows are drawn at regular step intervals.  Length is proportional to
        flow magnitude, normalised so that a vector at the frame's reference
        magnitude spans roughly one grid step — which keeps arrows legible
        whether the scene is a 1 px/frame crowd or 20 px/frame traffic.  Pass
        an explicit ``scale`` (px of arrow per px/frame of flow) to override.

        Vectors below min_draw_magnitude_px are skipped: their direction is
        noise, and drawing them fills the frame with random needles that read
        as texture rather than as motion.
        """
        # in_place=True skips this defensive copy. Only the caller that
        # already owns a private buffer may pass it -- in this module's own
        # pipeline every stage after the first alpha blend does, since the
        # blend returns a fresh array. Default stays False so an outside
        # caller's frame is never mutated under it.
        out = frame if in_place else frame.copy()
        h, w  = out.shape[:2]
        step  = self.arrow_step
        fx, fy = flow_xy[..., 0], flow_xy[..., 1]

        mag = np.sqrt(fx ** 2 + fy ** 2)
        if scale is None:
            scale = float(step) / self._reference_magnitude(mag)
        floor = self.motion_floor(mag)

        # Sample the grid points in one shot rather than indexing the
        # full-resolution arrays once per point inside the loop.
        ys = np.arange(step // 2, h, step)
        xs = np.arange(step // 2, w, step)
        sel = np.ix_(ys, xs)
        g_mag = mag[sel]
        g_dx  = fx[sel] * scale
        g_dy  = fy[sel] * scale
        g_len = np.hypot(g_dx, g_dy)

        # Sub-pixel arrows render as single dots; not worth drawing.
        draw = (g_mag >= floor) & (g_len >= 2.0)

        # Cap length to avoid arrows that cross multiple cells
        max_len = step * 2.0
        shrink  = np.where(g_len > max_len, max_len / np.maximum(g_len, 1e-6), 1.0)
        g_dx = g_dx * shrink
        g_dy = g_dy * shrink

        end_x = np.clip(xs[None, :] + g_dx, 0, w - 1).astype(np.int32)
        end_y = np.clip(ys[:, None] + g_dy, 0, h - 1).astype(np.int32)

        for iy, ix in zip(*np.nonzero(draw)):
            cv2.arrowedLine(
                out,
                (int(xs[ix]), int(ys[iy])),
                (int(end_x[iy, ix]), int(end_y[iy, ix])),
                color=(255, 255, 255), thickness=1,
                tipLength=0.3,
            )
        return out

    # ------------------------------------------------------------------
    # Divergence heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def _cells_overlapped(
        boxes: list, n_y: int, n_x: int, g: int,
    ) -> np.ndarray:
        """
        Binary (n_y, n_x) mask of cells overlapped by any box.

        Cell (r, c) spans rows [r·g, (r+1)·g) × cols [c·g, (c+1)·g), the same
        tiling crowd_metrics.py uses to build its grids. Overlap marking, not
        centre-in-box: a small or distant person's box can be smaller than one
        cell and contain no cell centre at all, and the failure this mask
        exists to prevent is colour appearing where nobody was detected —
        erring towards including a boundary cell over a person's own box is
        the safe direction.
        """
        mask = np.zeros((n_y, n_x), dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            c1 = max(0, int(x1) // g)
            c2 = min(n_x, int(np.ceil(x2 / g)))
            r1 = max(0, int(y1) // g)
            r2 = min(n_y, int(np.ceil(y2 / g)))
            if c2 > c1 and r2 > r1:
                mask[r1:r2, c1:c2] = 1.0
        return mask

    def divergence_heatmap(
        self, frame: np.ndarray, cell_divergence: np.ndarray, cell_size_px: int,
        cell_speed: Optional[np.ndarray] = None,
        person_boxes: Optional[list] = None,
        stationary_boxes: Optional[list] = None,
    ) -> np.ndarray:
        """
        Diverging colour heatmap over the frame.

        Red  = negative divergence (compression → crush risk).
        Blue = positive divergence (expansion).
        Near-white = neutral.

        Cells near zero divergence are left **untouched** rather than painted
        neutral.  Blending a near-white neutral colour across the whole frame
        at heatmap_alpha washes out the entire image — and since most cells in
        any real scene sit near zero, that is almost every pixel.  Only cells
        with meaningful divergence are tinted, with opacity proportional to
        |divergence| / div_scale.

        A static camera with no crowd therefore produces essentially no
        overlay at all.  Persistent red in a static scene indicates a
        sign-convention error or GMC failure.

        ``cell_speed`` (same shape as cell_divergence, source px/frame) gates
        the overlay on there being motion to have divergence ABOUT.  Spatial
        derivatives of a noisy near-zero field are themselves noisy, and
        without this gate every textured edge in a still scene accumulates
        enough apparent divergence to paint itself red — which reads as
        compression risk covering the whole frame.  Divergence in a region
        that is not moving is not a crowd signal.

        ``person_boxes`` ([x1,y1,x2,y2] per MOVING confirmed person, source
        px) restricts the tint further, to cells those boxes overlap.  The
        motion floor above cannot tell a textured floor tile from a person:
        both produce small flow noise that clears it, so tiles picked up
        colour that belongs to nobody.  Divergence is only meaningful where a
        person actually is, and PeopleOverlay already knows where that is.
        Pass None (no people tracker configured) for the previous whole-grid
        behaviour.

        ``stationary_boxes`` ([x1,y1,x2,y2] per confirmed but stopped person)
        gets NO flow colour at all — divergence about someone standing still
        is differentiated noise, not crowd dynamics — and instead a flat grey
        marker (_PERSON_STATIONARY_FILL), visually distinct from the
        crush/compression red so it cannot be misread as risk.
        """
        h, w     = frame.shape[:2]
        n_y, n_x = cell_divergence.shape
        g        = cell_size_px

        # Signed, normalised divergence per cell in [-1, 1].
        t_cell = np.clip(
            cell_divergence.astype(np.float32) / max(self.div_scale, 1e-6),
            -1.0, 1.0,
        )

        # Per-cell colour, vectorised: lerp neutral→red for t<0, neutral→blue
        # for t>0.  Building this at cell resolution and upsampling with
        # INTER_NEAREST reproduces the blocky per-cell look at a fraction of
        # the cost of the previous per-cell Python loop.
        w_neg = np.clip(-t_cell, 0.0, 1.0)[..., None]
        w_pos = np.clip( t_cell, 0.0, 1.0)[..., None]
        c_zero = np.array(_DIV_COLOR_ZERO, dtype=np.float32)
        c_neg  = np.array(_DIV_COLOR_NEG,  dtype=np.float32)
        c_pos  = np.array(_DIV_COLOR_POS,  dtype=np.float32)
        heat_cell = (c_zero
                     + (c_neg - c_zero) * w_neg
                     + (c_pos - c_zero) * w_pos)

        # Opacity follows |t|, so neutral cells contribute nothing.
        alpha_cell = np.abs(t_cell) * self.heatmap_alpha

        # Gate on motion: divergence where nothing moves is differentiated
        # noise, not a crowd signal.
        if cell_speed is not None and cell_speed.shape == cell_divergence.shape:
            alpha_cell = np.where(
                cell_speed >= self.motion_floor(cell_speed), alpha_cell, 0.0
            ).astype(np.float32)

        # Gate on people: with a tracker running, only cells under a MOVING
        # confirmed person's box keep their colour; every other cell — tiles,
        # floor, background whose texture noise cleared the motion floor —
        # drops to zero no matter what the flow field says.
        if person_boxes is not None:
            alpha_cell = alpha_cell * self._cells_overlapped(
                person_boxes, n_y, n_x, g,
            ).astype(np.float32)

        heat  = cv2.resize(heat_cell, (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        alpha = cv2.resize(alpha_cell.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_NEAREST)

        out = self._alpha_blend(frame, heat, alpha)

        # Confirmed-but-stationary people get a flat neutral fill instead of
        # flow colour, so "detected, standing still" stays visible without
        # borrowing the heatmap's risk semantics. Drawn after the blend so it
        # is not washed out by it; the tracker's own box outline lands on top
        # later via PeopleOverlay.draw().
        if stationary_boxes:
            for x1, y1, x2, y2 in stationary_boxes:
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(w, int(x2)), min(h, int(y2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                roi = out[iy1:iy2, ix1:ix2]
                flat = np.empty_like(roi)
                flat[:] = _PERSON_STATIONARY_FILL
                cv2.addWeighted(
                    roi, 1.0 - _PERSON_STATIONARY_ALPHA,
                    flat, _PERSON_STATIONARY_ALPHA,
                    0.0, dst=roi,
                )
        return out

    # ------------------------------------------------------------------
    # Zone overlay
    # ------------------------------------------------------------------

    def zone_overlay(
        self,
        frame: np.ndarray,
        zones: list,                    # list[Zone]
        zone_metrics: dict,             # zone_name → ZoneMetrics
        alerts: Optional[list] = None,  # list[Alert]
        in_place: bool = False,
    ) -> np.ndarray:
        """
        Draw zone polygons, metric text, and alert severity indicators.

        Zones with active CRITICAL alerts are outlined in red; WARNING in
        orange; no alerts in amber.
        """
        # in_place=True skips this defensive copy. Only the caller that
        # already owns a private buffer may pass it -- in this module's own
        # pipeline every stage after the first alpha blend does, since the
        # blend returns a fresh array. Default stays False so an outside
        # caller's frame is never mutated under it.
        out = frame if in_place else frame.copy()
        active_alerts: dict[str, int] = {}   # zone_name → max severity

        if alerts:
            for alert in alerts:
                prev = active_alerts.get(alert.zone_name, 0)
                active_alerts[alert.zone_name] = max(prev, int(alert.severity))

        for zone in zones:
            pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            sev = active_alerts.get(zone.name, 0)

            if sev >= 3:   # CRITICAL
                colour = (0, 0, 220)
                thick  = 3
            elif sev >= 2: # WARNING
                colour = (0, 140, 255)
                thick  = 2
            else:
                colour = _ZONE_OUTLINE_COLOR
                thick  = 1

            cv2.polylines(out, [pts], isClosed=True, color=colour, thickness=thick)

            # Zone name label
            x1, y1, _, _ = zone.bbox()
            cv2.putText(
                out, zone.name, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ZONE_TEXT_COLOR, 1,
                cv2.LINE_AA,
            )

            zm = zone_metrics.get(zone.name)
            if zm is None:
                continue

            # Compact per-zone metric readout
            lines = [
                f"spd:{zm.mean_speed:.2f}  div:{zm.mean_divergence:.2f}",
                f"cf:{zm.counterflow_score:.2f}  sg:{zm.stop_go_score:.2f}",
            ]
            for li, line in enumerate(lines):
                cv2.putText(
                    out, line,
                    (x1, min(y1 + 14 + li * 16, frame.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _ZONE_TEXT_COLOR, 1,
                    cv2.LINE_AA,
                )

        return out

    # ------------------------------------------------------------------
    # Detector boxes
    # ------------------------------------------------------------------

    def detector_boxes(
        self,
        frame: np.ndarray,
        vehicle_boxes: list,    # [[x1,y1,x2,y2], ...]
        umbrella_boxes: list,
        in_place: bool = False,
    ) -> np.ndarray:
        """
        Draw vehicle and umbrella detection boxes.

        Uses distinct colours consistent with pipeline/annotate.py COLOR_MAP
        (cyan-blue for vehicles, magenta for umbrellas) so both overlays look
        the same whether rendered by this module or the standard annotator.
        """
        # in_place=True skips this defensive copy. Only the caller that
        # already owns a private buffer may pass it -- in this module's own
        # pipeline every stage after the first alpha blend does, since the
        # blend returns a fresh array. Default stays False so an outside
        # caller's frame is never mutated under it.
        out = frame if in_place else frame.copy()
        for (x1, y1, x2, y2) in vehicle_boxes:
            cv2.rectangle(
                out, (int(x1), int(y1)), (int(x2), int(y2)),
                _VEHICLE_BOX_COLOR, 2,
            )
            cv2.putText(
                out, "vehicle",
                (int(x1), max(int(y1) - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _VEHICLE_BOX_COLOR, 1,
            )
        for (x1, y1, x2, y2) in umbrella_boxes:
            cv2.rectangle(
                out, (int(x1), int(y1)), (int(x2), int(y2)),
                _UMBRELLA_BOX_COLOR, 1,
            )
        return out

    # ------------------------------------------------------------------
    # Info banner
    # ------------------------------------------------------------------

    def info_banner(
        self,
        frame: np.ndarray,
        flags: dict,
        frame_index: int,
        timestamp_sec: float,
        in_place: bool = False,
    ) -> np.ndarray:
        """
        Small status strip at the top of the frame with robustness flags.

        flags: dict with optional bool keys:
          rain_flag, lowlight_flag, brightness_suppressed, gmc_applied, calibrated
        """
        # in_place=True skips this defensive copy. Only the caller that
        # already owns a private buffer may pass it -- in this module's own
        # pipeline every stage after the first alpha blend does, since the
        # blend returns a fresh array. Default stays False so an outside
        # caller's frame is never mutated under it.
        out = frame if in_place else frame.copy()
        parts = [f"f={frame_index} t={timestamp_sec:.1f}s"]

        if flags.get("rain_flag"):
            parts.append("[RAIN?]")
        if flags.get("lowlight_flag"):
            parts.append("[LOWLIGHT]")
        if flags.get("brightness_suppressed"):
            parts.append("[BRTJUMP-SUPPRESSED]")
        if flags.get("discontinuity"):
            parts.append(
                f"[SCENE-CUT? rel={flags.get('flow_reliability', 0.0):.2f} "
                f"FLOW UNRELIABLE]"
            )
        if not flags.get("calibrated", True):
            parts.append("[UNCALIBRATED]")
        if flags.get("gmc_applied"):
            parts.append(f"[GMC:{flags.get('gmc_method','?')}]")

        text = "  ".join(parts)
        cv2.rectangle(out, (0, 0), (frame.shape[1], 18), (30, 30, 30), -1)
        cv2.putText(
            out, text, (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 200), 1, cv2.LINE_AA,
        )
        return out

    # RGB (not BGR — this panel is composited via PIL), matched to the
    # Overview KPI card's hex accents in app.js so an operator sees the same
    # color for the same metric on the dashboard and on the video.
    _HUD_COLORS = {
        "Divergence":    (251, 146, 60),   # #fb923c
        "Counter-Flow":  (245, 158, 11),   # #f59e0b
        "Turbulence":    (34, 211, 238),   # #22d3ee
        "Entropy":       (167, 127, 250),  # #a78bfa
        "Oscillation":   (244, 114, 182),  # #f472b6
    }

    _HUD_FONT_PATHS = {
        "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "bold":    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    }

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _hud_font(weight: str, size: int):
        """Cached so the TTF isn't re-parsed from disk every frame."""
        from PIL import ImageFont
        path = FlowVisualiser._HUD_FONT_PATHS.get(weight)
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()

    def metrics_hud(
        self,
        frame: np.ndarray,
        mf: "MetricsFrame",
        specific_flow_latest: Optional[dict] = None,
        in_place: bool = False,
    ) -> np.ndarray:
        """
        Compact live-numbers panel, top-right corner: the 5 headline crowd
        metrics plus the current specific-flow rate if configured.  Worst
        zone wins for the 4 zone-scored metrics, matching the "safety-first,
        show the number that would trigger action" convention used
        elsewhere (peak values in the KPI card, not averages).

        Rendered with PIL (rounded rect, anti-aliased mono type) rather than
        cv2.putText so it matches the reference HUD mockup instead of
        looking like a raw debug overlay.
        """
        from PIL import Image, ImageDraw

        zones = list(mf.zones.values())

        def _max(attr: str) -> float:
            return max((getattr(z, attr) for z in zones), default=0.0)

        min_coherence = min((z.mean_coherence for z in zones), default=1.0)
        rows = [
            ("Divergence",   _max("mean_divergence")),
            ("Counter-Flow", _max("counterflow_score")),
            ("Turbulence",   _max("turbulence_index")),
            ("Entropy",      1.0 - min_coherence),
            ("Oscillation",  _max("oscillation_symmetry_score")),
        ]

        flow_rate, flow_units = None, ""
        if specific_flow_latest:
            flow_rate = max((d.get("rate", 0.0) for d in specific_flow_latest.values()), default=0.0)
            flow_units = next((d.get("units") for d in specific_flow_latest.values()), "")

        panel_w   = 176
        row_h     = 20
        top_pad   = 30
        bottom_pad = 10
        panel_h   = top_pad + row_h * len(rows) + bottom_pad
        if flow_rate is not None:
            panel_h += row_h + 6

        x0 = frame.shape[1] - panel_w - 10
        y0 = 10

        # Build the panel on its own RGBA canvas so the rounded corners and
        # translucency anti-alias cleanly, then composite onto the frame.
        panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            [0, 0, panel_w - 1, panel_h - 1], radius=9,
            fill=(10, 12, 18, 217), outline=(255, 255, 255, 26), width=1,
        )

        title_font = self._hud_font("bold", 10)
        label_font = self._hud_font("regular", 11)
        value_font = self._hud_font("bold", 12)

        draw.text((12, 9), "LIVE METRICS", font=title_font, fill=(131, 139, 163, 255))

        y = top_pad
        for label, value in rows:
            color = self._HUD_COLORS[label] + (255,)
            draw.ellipse([12, y + 5, 18, y + 11], fill=color)
            draw.text((24, y), label, font=label_font, fill=(207, 211, 224, 255))
            text = f"{value:.2f}"
            tw = draw.textlength(text, font=value_font)
            draw.text((panel_w - 12 - tw, y - 1), text, font=value_font, fill=color)
            y += row_h

        if flow_rate is not None:
            draw.line([12, y + 2, panel_w - 12, y + 2], fill=(255, 255, 255, 26), width=1)
            flow_color = self._HUD_COLORS["Turbulence"] + (255,)
            draw.text((12, y + 9), f"\u2192 {flow_rate:.2f} {flow_units}",
                       font=label_font, fill=flow_color)

        # Composite the RGBA panel over the region of the frame it covers.
        region = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        base = Image.fromarray(region_rgb).convert("RGBA")
        composited = Image.alpha_composite(base, panel).convert("RGB")
        blended_bgr = cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)

        # in_place=True skips this defensive copy. Only the caller that
        # already owns a private buffer may pass it -- in this module's own
        # pipeline every stage after the first alpha blend does, since the
        # blend returns a fresh array. Default stays False so an outside
        # caller's frame is never mutated under it.
        out = frame if in_place else frame.copy()
        out[y0:y0 + panel_h, x0:x0 + panel_w] = blended_bgr
        return out

    # ------------------------------------------------------------------
    # Time-series plot export (offline, post-run)
    # ------------------------------------------------------------------

    @staticmethod
    def save_timeseries_plots(
        metrics_log: list[dict],   # list of per-frame zone-metrics dicts
        output_dir: str,
        camera_id: str,
    ) -> list[str]:
        """
        Write one PNG per zone showing speed, divergence, counterflow, and
        stop-go score over time.

        Requires matplotlib.  If not installed, logs a WARNING and returns [].
        Written to output_dir/<camera_id>_<zone_name>_timeseries.png.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")          # headless
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib is not installed; time-series plots will not be "
                "generated.  Install it with: pip install matplotlib"
            )
            return []

        if not metrics_log:
            return []

        os.makedirs(output_dir, exist_ok=True)
        # Collect zone names from first entry that has zones
        zone_names: set[str] = set()
        for entry in metrics_log:
            zone_names.update(entry.get("zones", {}).keys())

        written: list[str] = []
        for zone_name in sorted(zone_names):
            timestamps = []
            speeds, divergences, counterflows, stop_gos = [], [], [], []

            for entry in metrics_log:
                zm = entry.get("zones", {}).get(zone_name)
                if zm is None:
                    continue
                timestamps.append(entry.get("timestamp_sec", 0.0))
                speeds.append(zm.get("mean_speed", 0.0))
                divergences.append(zm.get("mean_divergence", 0.0))
                counterflows.append(zm.get("counterflow_score", 0.0))
                stop_gos.append(zm.get("stop_go_score", 0.0))

            if not timestamps:
                continue

            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f"{camera_id} / {zone_name}", fontsize=12, fontweight="bold")

            t = timestamps
            axes[0].plot(t, speeds, color="steelblue", linewidth=1.2)
            axes[0].set_ylabel("Mean speed")
            axes[0].axhline(0, color="gray", linewidth=0.5)
            axes[0].set_title("Speed (m/s or px/frame)")

            axes[1].plot(t, divergences, color="firebrick", linewidth=1.2)
            axes[1].set_ylabel("Divergence")
            axes[1].axhline(0, color="gray", linewidth=0.5)
            axes[1].set_title("Divergence (negative = compression)")
            axes[1].fill_between(t, divergences, 0,
                                 where=[d < 0 for d in divergences],
                                 alpha=0.25, color="firebrick", label="compression")

            axes[2].plot(t, counterflows, color="darkorange", linewidth=1.2)
            axes[2].set_ylabel("Counterflow")
            axes[2].set_ylim(0, 1)
            axes[2].set_title("Counterflow score (entry/exit separation check)")

            axes[3].plot(t, stop_gos, color="purple", linewidth=1.2)
            axes[3].set_ylabel("Stop-go")
            axes[3].set_ylim(0, 1)
            axes[3].set_xlabel("Time (s)")
            axes[3].set_title("Stop-and-go score")

            plt.tight_layout()
            safe_zone = zone_name.replace(" ", "_").replace("/", "_")
            out_path  = os.path.join(output_dir, f"{camera_id}_{safe_zone}_timeseries.png")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            written.append(out_path)
            logger.info("Time-series plot written to %s", out_path)

        return written
