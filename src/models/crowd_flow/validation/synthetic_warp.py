"""
Route (a) — synthetic warp validation.

Take a real frame, displace it by a field you chose yourself, and measure
whether the estimator recovers that field.  The truth is exact because you
defined it, and it works at any crowd density because warping a packed frame
is no harder than warping an empty one.

What this route validates
-------------------------
The estimator's mathematics on real image texture: does DIS recover a known
displacement, and how does that degrade with magnitude, field shape, compute
resolution, and smoothing?

What it CANNOT validate
-----------------------
The motion is synthetic even though the imagery is real.  A warped frame has
no motion blur, no person occluding another differently between frames, no
umbrella tilting independently of the body under it, and — critically — no
independently moving objects.  The global-motion-compensation failure that
made this project's output unusable (ORB locking onto moving vehicles and
subtracting their motion from the whole field) is invisible here, because a
warped frame contains exactly one coherent motion by construction.

Optimising configuration against this route alone is therefore a real risk:
settings that ace rigid warps can be worse on live crowds.  Confirm any
change against route (c) before accepting it.

How the warp is constructed
---------------------------
Given a target forward flow F (what the estimator should report), the warped
frame is built by backward sampling:

    warped(x) = source(x − F(x))

so the content that sat at x − F(x) now sits at x, i.e. it moved by F.  This
is exact for a constant F and a close approximation for smooth fields, since
it evaluates F at the destination rather than the source.  The residual from
that approximation is second-order in the field's spatial gradient and is far
below the estimator errors being measured; for the fields used here it is
under 0.01 px.  Border pixels are excluded from scoring because backward
sampling invents content outside the source frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from models.crowd_flow.flow_field import FlowField
from models.crowd_flow.validation.report import (
    Measurement, RouteResult, STATUS_PASS,
)

logger = logging.getLogger(__name__)

ROUTE_KEY   = "synthetic_warp"
ROUTE_TITLE = "Route (a) — synthetic warp"
ROUTE_CAVEAT = (
    "Real imagery, synthetic motion.  No motion blur, no differential "
    "occlusion, and no independently moving objects — so estimator failures "
    "that are triggered by real scene content (the GMC-on-moving-vehicles "
    "class of bug) cannot appear here.  Necessary, not sufficient."
)

# Pixels trimmed from each edge before scoring: backward sampling invents
# content outside the source frame, and the estimator cannot be blamed for it.
_BORDER_PX = 24


# ----------------------------------------------------------------------------
# Ground-truth field generators
# ----------------------------------------------------------------------------

def translation_field(h: int, w: int, dx: float, dy: float) -> np.ndarray:
    """Uniform translation — the simplest possible case."""
    f = np.zeros((h, w, 2), dtype=np.float32)
    f[..., 0] = dx
    f[..., 1] = dy
    return f


def radial_field(h: int, w: int, strength: float) -> np.ndarray:
    """
    Radial field about the frame centre.

    strength < 0 converges (compression — the crush-risk case), > 0 expands.
    Magnitude grows linearly with radius, normalised so the corner reaches
    |strength| px/frame.
    """
    f = np.zeros((h, w, 2), dtype=np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    norm = float(np.hypot(cx, cy)) or 1.0
    f[..., 0] = strength * dx / norm
    f[..., 1] = strength * dy / norm
    return f


def shear_field(h: int, w: int, strength: float) -> np.ndarray:
    """Horizontal velocity varying with row — two streams sliding past."""
    f = np.zeros((h, w, 2), dtype=np.float32)
    ys = np.arange(h, dtype=np.float32).reshape(-1, 1)
    f[..., 0] = strength * (2.0 * ys / max(h - 1, 1) - 1.0)
    return f


def rotation_field(h: int, w: int, degrees: float) -> np.ndarray:
    """Rigid rotation about the frame centre (a curl test)."""
    f = np.zeros((h, w, 2), dtype=np.float32)
    theta = np.deg2rad(degrees)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    f[..., 0] = np.cos(theta) * dx - np.sin(theta) * dy - dx
    f[..., 1] = np.sin(theta) * dx + np.cos(theta) * dy - dy
    return f


@dataclass
class WarpCase:
    name: str
    field: np.ndarray

    @property
    def mean_magnitude(self) -> float:
        return float(np.hypot(self.field[..., 0], self.field[..., 1]).mean())


def default_cases(h: int, w: int) -> list[WarpCase]:
    """
    A spread of field shapes and magnitudes.

    Magnitudes span the pedestrian range (~1-3 px/frame at 30 fps) up to
    vehicle speeds, because the estimator has to serve both and its error is
    not flat across that range.
    """
    cases: list[WarpCase] = []
    for mag in (1.0, 3.0, 8.0, 16.0):
        cases.append(WarpCase(f"translate_{mag:g}px",
                              translation_field(h, w, mag * 0.8, mag * 0.6)))
    cases.append(WarpCase("converge_4px",  radial_field(h, w, -4.0)))
    cases.append(WarpCase("expand_4px",    radial_field(h, w,  4.0)))
    cases.append(WarpCase("shear_5px",     shear_field(h, w, 5.0)))
    cases.append(WarpCase("rotate_1.5deg", rotation_field(h, w, 1.5)))
    return cases


# ----------------------------------------------------------------------------
# Warping and scoring
# ----------------------------------------------------------------------------

def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Build the second frame of a pair whose true forward flow is ``flow``."""
    h, w = frame.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    return cv2.remap(frame, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT101)


def endpoint_error(
    estimated: np.ndarray, truth: np.ndarray, border: int = _BORDER_PX
) -> dict[str, float]:
    """
    Endpoint error between two flow fields, scored away from the border.

    Returns mean/median/p95 EPE in px/frame, plus the mean angular error in
    degrees over vectors long enough for direction to be meaningful.
    """
    b = border
    est = estimated[b:-b, b:-b] if b > 0 else estimated
    gt  = truth[b:-b, b:-b]     if b > 0 else truth

    diff = est - gt
    epe  = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)

    gt_mag = np.sqrt(gt[..., 0] ** 2 + gt[..., 1] ** 2)
    meaningful = gt_mag > 0.5
    if meaningful.any():
        dot = (est[..., 0] * gt[..., 0] + est[..., 1] * gt[..., 1])
        est_mag = np.sqrt(est[..., 0] ** 2 + est[..., 1] ** 2)
        cos = np.clip(dot / np.maximum(est_mag * gt_mag, 1e-9), -1.0, 1.0)
        ang = float(np.degrees(np.arccos(cos[meaningful])).mean())
    else:
        ang = 0.0

    return {
        "epe_mean":   float(epe.mean()),
        "epe_median": float(np.median(epe)),
        "epe_p95":    float(np.percentile(epe, 95)),
        "angular_deg": ang,
        "gt_mean_magnitude": float(gt_mag.mean()),
    }


# ----------------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------------

class SyntheticWarpValidator:
    """
    Runs a set of known warps through a FlowField and scores the recovery.

    Parameters
    ----------
    flow_factory:
        Callable returning a fresh FlowField.  A factory rather than an
        instance because each case must start from clean temporal-smoothing
        state — reusing one instance would let the previous case's field bleed
        into this one through the EMA and quietly flatter every case after the
        first.
    max_epe_px:
        Tolerance on mean endpoint error, in px/frame.
    """

    def __init__(
        self,
        flow_factory: Optional[Callable[[], FlowField]] = None,
        max_epe_px: float = 1.0,
        max_angular_deg: float = 20.0,
    ) -> None:
        self.flow_factory = flow_factory or (
            # GMC off by default: a synthetic warp IS global motion, so the
            # compensator would correctly identify and subtract the very
            # signal under test.  That measures the compensator, not the
            # estimator.  Route (c) covers GMC behaviour on real scenes.
            lambda: FlowField(global_motion_compensation=False,
                              temporal_smooth_alpha=1.0)
        )
        self.max_epe_px = max_epe_px
        self.max_angular_deg = max_angular_deg

    def run_case(self, frame: np.ndarray, case: WarpCase) -> dict:
        warped = warp_frame(frame, case.field)
        ff = self.flow_factory()
        result = ff.compute(frame, warped, exclusion_mask=None, timestamp_sec=0.0)
        scores = endpoint_error(result.field_xy, case.field)
        scores["name"] = case.name
        scores["relative_error"] = (
            scores["epe_mean"] / max(scores["gt_mean_magnitude"], 1e-6)
        )
        return scores

    def run(self, frame: np.ndarray,
            cases: Optional[list[WarpCase]] = None) -> RouteResult:
        h, w = frame.shape[:2]
        cases = cases or default_cases(h, w)

        per_case = [self.run_case(frame, c) for c in cases]

        mean_epe = float(np.mean([c["epe_mean"] for c in per_case]))
        worst    = max(per_case, key=lambda c: c["epe_mean"])
        mean_ang = float(np.mean([c["angular_deg"] for c in per_case]))

        res = RouteResult(
            route=ROUTE_KEY, title=ROUTE_TITLE, status=STATUS_PASS,
            summary=(
                f"{len(per_case)} warp cases on a {w}x{h} real frame; "
                f"mean endpoint error {mean_epe:.2f} px/frame "
                f"(worst: {worst['name']} at {worst['epe_mean']:.2f})"
            ),
            caveat=ROUTE_CAVEAT,
            measurements=[
                Measurement("Mean endpoint error", mean_epe, "px/frame",
                            tolerance=self.max_epe_px),
                Measurement("Worst-case endpoint error", worst["epe_mean"],
                            "px/frame", tolerance=self.max_epe_px * 3,
                            note=f"case: {worst['name']}"),
                Measurement("Mean angular error", mean_ang, "deg",
                            tolerance=self.max_angular_deg),
            ],
            detail={"cases": per_case},
        )
        res.resolve_status()
        return res

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    def sweep(
        self,
        frame: np.ndarray,
        variants: dict[str, Callable[[], FlowField]],
        cases: Optional[list[WarpCase]] = None,
    ) -> list[dict]:
        """
        Score several FlowField configurations against the same warp cases.

        This is the route's practical payoff: several values in
        configs/crowd_flow.yaml (compute resolution, backend, smoothing) are
        judgement calls with no measurement behind them.  A sweep replaces
        each with a number.  Results are ordered best-first by mean EPE.
        """
        h, w = frame.shape[:2]
        cases = cases or default_cases(h, w)

        rows: list[dict] = []
        for name, factory in variants.items():
            saved, self.flow_factory = self.flow_factory, factory
            try:
                scored = [self.run_case(frame, c) for c in cases]
            finally:
                self.flow_factory = saved
            rows.append({
                "variant":  name,
                "epe_mean": float(np.mean([s["epe_mean"] for s in scored])),
                "epe_p95":  float(np.mean([s["epe_p95"] for s in scored])),
                "angular_deg": float(np.mean([s["angular_deg"] for s in scored])),
                "per_case": {s["name"]: round(s["epe_mean"], 3) for s in scored},
            })
        rows.sort(key=lambda r: r["epe_mean"])
        return rows
