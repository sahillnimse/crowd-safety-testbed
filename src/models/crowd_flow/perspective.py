"""
Perspective-map calibration: metres per pixel, without measuring the ground.

Why a second calibration method
-------------------------------
ground_plane.py calibrates by homography: pick >= 4 ground points, measure the
real distances between them, solve. That is the more rigorous method and it
stays the preferred one — it produces a true planar metric map.

It also requires somebody to stand on the ghat with a tape measure. On this
project that has never happened: all four Nashik cameras ship with no
homography, so speed is px/frame, density is not persons/m2, and crowd
pressure is disabled. A perfect method with no data loses to an approximate
method with data.

This module is the approximate method with data. It models person SIZE in
pixels as a bilinear function of image position:

    height_px(x, y) = ah*x + bh*y + ch
    width_px (x, y) = aw*x + bw*y + cw

Assume a standing adult is `person_height_m` tall (1.65-1.7 m) and the scale
at any pixel follows:

    metres_per_pixel(x, y) = person_height_m / height_px(x, y)

Fitting needs no tape measure — only boxes drawn round people already visible
in the frame, which can be done from a still, after the fact, by someone who
was never on site. That is the whole point: it is retrofittable.

Indoor and outdoor
------------------
The model is the same for both; only the fit data differs. SAIVT ships fitted
coefficients for ten indoor cameras (from ~59 annotated boxes each), and
``fit_from_boxes`` produces the same thing for an outdoor ghat from boxes you
draw yourself. A Nashik camera needs roughly 30-60 people spread across the
full depth of the scene.

Where this is weaker than a homography
--------------------------------------
1. It assumes everyone is the same height. Real crowds vary ~+/-8%, and this
   dataset's crowd is students -- a Kumbh crowd includes children and seated
   pilgrims, for whom the assumption fails outright.
2. It gives a SCALE at each pixel, not a full ground-plane mapping, so it
   cannot rectify a polygon into true ground area as accurately as a
   homography can.
3. Bilinear is a first-order approximation of true perspective. It holds well
   across a normal camera's field of view and degrades near the horizon,
   where person height tends to zero and metres-per-pixel diverges -- which
   is why `mpp` clamps.

So: use a homography where the site can be measured, and this where it
cannot. Both feed the same downstream metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Below this many pixels a "person" is too small to trust as a scale
#: reference; the far field of an oblique camera otherwise drives
#: metres-per-pixel towards infinity.
_MIN_PERSON_PX = 8.0

#: Hard bounds on metres-per-pixel. Wide enough for any plausible mount,
#: narrow enough that a degenerate fit cannot produce a crowd moving at
#: 400 m/s.
_MPP_MIN = 0.0025
_MPP_MAX = 0.04


@dataclass
class PerspectiveMap:
    """Bilinear person-size model over the image plane."""
    aw: float = 0.0
    bw: float = 0.0
    cw: float = 0.0
    ah: float = 0.0
    bh: float = 0.0
    ch: float = 0.0
    image_width: int = 0
    image_height: int = 0
    person_height_m: float = 1.70
    #: Provenance, so a reader can tell a fitted map from a guessed one.
    source: str = "unknown"
    #: Fit quality, when this came from fit_from_boxes.
    fit_rmse_px: Optional[float] = None
    n_fit_samples: int = 0

    # ------------------------------------------------------------------

    @classmethod
    def from_saivt(cls, coeffs: dict, person_height_m: float = 1.70) -> "PerspectiveMap":
        """Build from a parsed SAIVT perspectivemap.xml."""
        return cls(
            aw=coeffs.get("aw", 0.0), bw=coeffs.get("bw", 0.0),
            cw=coeffs.get("cw", 0.0), ah=coeffs.get("ah", 0.0),
            bh=coeffs.get("bh", 0.0), ch=coeffs.get("ch", 0.0),
            image_width=coeffs.get("image_width", 0),
            image_height=coeffs.get("image_height", 0),
            person_height_m=person_height_m,
            source="saivt_perspectivemap.xml",
        )

    @classmethod
    def fit_from_boxes(cls, boxes, image_size: tuple,
                       person_height_m: float = 1.70) -> "PerspectiveMap":
        """
        Least-squares fit from annotated person boxes.

        ``boxes``: iterable of (left, top, right, bottom) in pixels, each
        around one standing person. Spread them across the FULL depth of the
        scene — a fit from boxes that all sit at one distance is a constant,
        and constant scale is exactly what perspective is not.

        This is the path for a Nashik ghat camera: draw 30-60 boxes on a
        still, fit, done. No site visit.
        """
        arr = np.asarray(list(boxes), dtype=np.float64).reshape(-1, 4)
        if len(arr) < 3:
            raise ValueError(
                f"need at least 3 annotated people to fit a perspective map, "
                f"got {len(arr)}")

        cx = (arr[:, 0] + arr[:, 2]) / 2.0
        # CENTRE y, not feet. This is the SAIVT convention, and matching it is
        # what makes a map fitted here interchangeable with the ten shipped
        # ones. Verified against P_Lev_4_Entry_Way_ip_107: evaluating SAIVT's
        # own published coefficients against its own 59 annotated boxes gives
        # RMSE 14.6 px using centre-y, versus 43.3 (feet) and 47.0 (top), and
        # only centre-y is unbiased (mean predicted == mean actual == 96.2 px).
        #
        # Fitting against feet instead looked more physical -- ground contact
        # is what perspective is a function of -- but it produced a systematic
        # ~40% scale error against the shipped maps, which would have gone
        # straight into every speed in m/s and every density in persons/m2.
        # See person_size_at_feet() for the conversion back to a ground point.
        cy = (arr[:, 1] + arr[:, 3]) / 2.0
        w_px = arr[:, 2] - arr[:, 0]
        h_px = arr[:, 3] - arr[:, 1]

        keep = (h_px > _MIN_PERSON_PX) & (w_px > 1.0)
        if keep.sum() < 3:
            raise ValueError("too few usable boxes after filtering tiny ones")
        cx, cy, w_px, h_px = cx[keep], cy[keep], w_px[keep], h_px[keep]

        design = np.column_stack([cx, cy, np.ones_like(cx)])
        (ah, bh, ch), *_ = np.linalg.lstsq(design, h_px, rcond=None)
        (aw, bw, cw), *_ = np.linalg.lstsq(design, w_px, rcond=None)

        pred_h = design @ np.array([ah, bh, ch])
        rmse = float(np.sqrt(np.mean((pred_h - h_px) ** 2)))

        pm = cls(aw=float(aw), bw=float(bw), cw=float(cw),
                 ah=float(ah), bh=float(bh), ch=float(ch),
                 image_width=int(image_size[0]), image_height=int(image_size[1]),
                 person_height_m=person_height_m,
                 source="fit_from_boxes", fit_rmse_px=rmse,
                 n_fit_samples=int(keep.sum()))

        # A fit whose residual rivals the thing being measured is not a
        # calibration. Mean person height here is typically 60-200 px; an RMSE
        # near that means the boxes did not span the scene's depth.
        mean_h = float(np.mean(h_px))
        if rmse > 0.25 * mean_h:
            logger.warning(
                "Perspective fit is poor: RMSE %.1f px against a mean person "
                "height of %.1f px (%.0f%%). The annotated people probably do "
                "not span the full depth of the scene. Speeds and densities "
                "derived from this will carry that error.",
                rmse, mean_h, 100 * rmse / max(mean_h, 1e-6))
        return pm

    # ------------------------------------------------------------------

    def person_size_px(self, x: float, y_centre: float) -> tuple:
        """
        (width, height) in pixels of a person whose BODY CENTRE is at (x, y).

        Note the reference point: the model is fitted against box centres (see
        fit_from_boxes). Passing a ground/feet coordinate here under-reads the
        person's height and inflates metres-per-pixel — use
        ``person_size_at_feet`` when what you have is a ground contact, which
        is the usual case for a detected head point projected down or a track
        centroid on the floor.
        """
        w = self.aw * x + self.bw * y_centre + self.cw
        h = self.ah * x + self.bh * y_centre + self.ch
        return float(max(w, 1.0)), float(max(h, _MIN_PERSON_PX))

    def person_size_at_feet(self, x: float, y_feet: float) -> tuple:
        """
        (width, height) in pixels for a person STANDING at (x, y_feet).

        Solves the implicit form. Height is modelled against the body centre,
        and the centre of a person standing at y_feet is y_feet - h/2, so:

            h = ah*x + bh*(y_feet - h/2) + ch
        =>  h * (1 + bh/2) = ah*x + bh*y_feet + ch
        =>  h = (ah*x + bh*y_feet + ch) / (1 + bh/2)

        Closed form, no iteration. The denominator is ~1.39 on this dataset's
        cameras and is guarded below in case a degenerate fit drives bh to -2.
        """
        denom = 1.0 + self.bh / 2.0
        if abs(denom) < 1e-6:
            # Degenerate fit; fall back to treating the input as a centre.
            return self.person_size_px(x, y_feet)
        h = (self.ah * x + self.bh * y_feet + self.ch) / denom
        h = max(h, _MIN_PERSON_PX)
        y_centre = y_feet - h / 2.0
        w = self.aw * x + self.bw * y_centre + self.cw
        return float(max(w, 1.0)), float(h)

    def metres_per_pixel(self, x: float, y: float, at_feet: bool = True) -> float:
        """
        Local scale at (x, y), clamped to a physically plausible band.

        ``at_feet`` defaults True because callers almost always hold a ground
        point — a foot position, a track on the floor, a grid cell centre on
        the walking surface. Pass False when (x, y) is genuinely a body centre.
        """
        _, h_px = (self.person_size_at_feet(x, y) if at_feet
                   else self.person_size_px(x, y))
        return float(np.clip(self.person_height_m / h_px, _MPP_MIN, _MPP_MAX))

    def pixel_area_to_m2(self, x: float, y: float, area_px: float) -> float:
        """Convert a pixel area near (x, y) to m2. Scale is squared."""
        s = self.metres_per_pixel(x, y)
        return float(area_px * s * s)

    def polygon_area_m2(self, polygon) -> float:
        """
        Ground area of a pixel polygon, in m2.

        Integrated over a grid rather than computed from a single scale: the
        whole point of a perspective map is that scale varies across the
        region, so multiplying the shoelace area by one m/px value is wrong
        by however much the polygon spans in depth — which for a ghat
        approach is a lot.
        """
        import cv2
        poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(poly) < 3:
            return 0.0
        x0, y0 = np.floor(poly.min(axis=0)).astype(int)
        x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [(poly - [x0, y0]).astype(np.int32)], 255)

        step = max(1, int(min(w, h) / 64))       # ~64 samples across, plenty
        total = 0.0
        for yy in range(0, h, step):
            for xx in range(0, w, step):
                if mask[yy, xx]:
                    s = self.metres_per_pixel(x0 + xx, y0 + yy)
                    total += (step * step) * s * s
        return float(total)

    def speed_px_to_mps(self, x: float, y: float, speed_px_per_frame: float,
                        fps: float) -> float:
        """Convert a local pixel speed to m/s at (x, y)."""
        return float(speed_px_per_frame * self.metres_per_pixel(x, y) * max(fps, 1e-6))

    def describe(self) -> str:
        base = (f"PerspectiveMap({self.image_width}x{self.image_height}, "
                f"person={self.person_height_m} m, source={self.source}")
        if self.fit_rmse_px is not None:
            base += f", fit RMSE {self.fit_rmse_px:.1f} px over {self.n_fit_samples} people"
        return base + ")"

    def to_dict(self) -> dict:
        return {
            "aw": self.aw, "bw": self.bw, "cw": self.cw,
            "ah": self.ah, "bh": self.bh, "ch": self.ch,
            "image_width": self.image_width, "image_height": self.image_height,
            "person_height_m": self.person_height_m,
            "source": self.source,
            "fit_rmse_px": self.fit_rmse_px,
            "n_fit_samples": self.n_fit_samples,
        }
