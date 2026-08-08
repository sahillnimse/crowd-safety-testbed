"""
Flow field visualisation helpers.

All functions take an existing BGR frame and return an annotated copy — they
never modify the input in place.  Rendering is a separate concern from metric
computation; disable it entirely (pass visualise=False to DenseFlowAnalyser)
for headless / multi-stream deployments to save ~5-12 ms/frame.

HSV flow overlay
----------------
Hue   = direction (0° = right, 90° = down, 180° = left, 270° = up)
Value = magnitude, clipped to max_magnitude_px and rescaled to [0, 255]
Saturation = 1.0 (full saturation — avoids the "dark at zero" artefact that
             makes low-motion regions invisible in the overlay)

The flow overlay is blended over the source frame at configurable alpha so
the original scene is still visible for spatial reference.

Divergence heatmap
------------------
Red  = negative divergence (convergence → compression → crush risk)
Blue = positive divergence (expansion → people spreading out)
Zero = white (neutral)

The colour scale is zero-centred and symmetric around ±div_scale.  The sign
convention (red = bad = compression = negative div) is documented here and
matches crowd_metrics.py exactly.  If you see the heatmap predominantly red
in a static scene with no crowd, the sign convention is wrong.

Colour choices are rendering decisions only; the pipeline's Detection and Alert
objects carry computed values, not colours.
"""

from __future__ import annotations

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


class FlowVisualiser:
    """
    Produces annotated frames from flow fields, metrics, and detections.

    All methods return a uint8 BGR ndarray the same shape as the input frame.
    """

    def __init__(
        self,
        flow_alpha: float = 0.5,
        heatmap_alpha: float = 0.55,
        arrow_step: int = 20,
        max_magnitude_px: float = 20.0,
        div_scale: float = 3.0,
        auto_scale_overlay: bool = True,
        min_draw_magnitude_px: float = 0.4,
        overlay_work_px: int = 480,
    ) -> None:
        """
        Parameters
        ----------
        flow_alpha:
            Opacity of the HSV overlay at reference magnitude (0 = invisible,
            1 = opaque).  Applied per pixel, scaled by local magnitude.
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
            Arrows shorter than this (in source px/frame) are not drawn; below
            it, direction is noise rather than measurement.
        overlay_work_px:
            Longest side at which the translucent overlays are rendered before
            being resized onto the frame.  The flow field is computed
            downsampled and upsampled, so full-resolution rendering costs
            arithmetic without adding detail.  Only the final composite runs
            at full resolution, so the source frame stays sharp.
        """
        self.flow_alpha       = flow_alpha
        self.heatmap_alpha    = heatmap_alpha
        self.arrow_step       = arrow_step
        self.max_magnitude_px = max_magnitude_px
        self.div_scale        = div_scale
        self.auto_scale_overlay    = auto_scale_overlay
        self.min_draw_magnitude_px = min_draw_magnitude_px
        self.overlay_work_px       = overlay_work_px

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
        self, frame: np.ndarray, flow_xy: np.ndarray
    ) -> np.ndarray:
        """
        HSV colour-wheel overlay blended over the source frame.

        Hue = direction, Value = magnitude (clipped to max_magnitude_px),
        Saturation = 1 everywhere.

        The overlay is blended with a **per-pixel** alpha proportional to flow
        magnitude, so static regions keep the source frame unchanged and
        moving regions are tinted.  A uniform alpha blends the HSV image —
        which is black wherever magnitude is zero — over the whole frame, so
        a scene that is 90% static comes out 90% darkened and the moving
        subjects are no easier to see than in the raw video.
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

        # Value: magnitude scaled to [0, 255] against the frame's reference.
        mag   = np.sqrt(fx ** 2 + fy ** 2)
        ref   = self._reference_magnitude(mag)
        norm  = np.clip(mag / ref, 0.0, 1.0)
        # Square-root ramp: perceptually, a linear ramp spends most of its
        # range on the fastest few percent of pixels and leaves the bulk of a
        # crowd — which moves at a fraction of the peak — near-black.
        norm  = np.sqrt(norm)
        val   = (norm * 255).astype(np.uint8)

        sat   = np.full_like(hue, 255)

        hsv_img = np.stack([hue, sat, val], axis=-1)
        bgr_flow = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # Per-pixel alpha: 0 where there is no motion, flow_alpha at full scale.
        alpha = (norm * self.flow_alpha).astype(np.float32)

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
        out   = frame.copy()
        h, w  = out.shape[:2]
        step  = self.arrow_step
        fx, fy = flow_xy[..., 0], flow_xy[..., 1]

        mag = np.sqrt(fx ** 2 + fy ** 2)
        if scale is None:
            scale = float(step) / self._reference_magnitude(mag)

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
        draw = (g_mag >= self.min_draw_magnitude_px) & (g_len >= 2.0)

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

    def divergence_heatmap(
        self, frame: np.ndarray, cell_divergence: np.ndarray, cell_size_px: int
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

        heat  = cv2.resize(heat_cell, (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        alpha = cv2.resize(alpha_cell.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_NEAREST)

        return self._alpha_blend(frame, heat, alpha)

    # ------------------------------------------------------------------
    # Zone overlay
    # ------------------------------------------------------------------

    def zone_overlay(
        self,
        frame: np.ndarray,
        zones: list,                    # list[Zone]
        zone_metrics: dict,             # zone_name → ZoneMetrics
        alerts: Optional[list] = None,  # list[Alert]
    ) -> np.ndarray:
        """
        Draw zone polygons, metric text, and alert severity indicators.

        Zones with active CRITICAL alerts are outlined in red; WARNING in
        orange; no alerts in amber.
        """
        out = frame.copy()
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
    ) -> np.ndarray:
        """
        Draw vehicle and umbrella detection boxes.

        Uses distinct colours consistent with pipeline/annotate.py COLOR_MAP
        (cyan-blue for vehicles, magenta for umbrellas) so both overlays look
        the same whether rendered by this module or the standard annotator.
        """
        out = frame.copy()
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
    ) -> np.ndarray:
        """
        Small status strip at the top of the frame with robustness flags.

        flags: dict with optional bool keys:
          rain_flag, lowlight_flag, brightness_suppressed, gmc_applied, calibrated
        """
        out   = frame.copy()
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
