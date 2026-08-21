"""
Route (b) — cross-camera agreement.

Two cameras watching the same patch of ground from different angles must
describe the same physical motion once both are projected onto the ground
plane.  Where they disagree, at least one is wrong.

Why the disagreement is a real bound
------------------------------------
The two views differ in angle, perspective compression, occlusion pattern and
sensor noise, so their errors are largely independent — they do not fail the
same way at the same place.  That independence is what turns "they disagree
by 0.4 m/s" into a genuine lower bound on the error, rather than a mere
consistency check between two runs of the same method.

What this route validates
-------------------------
Real crowd motion, at real density, with real occlusion — the regime routes
(a) and (c) cannot both reach at once.  It is the only one of the three that
is simultaneously real-motion and high-density, which makes it the load-
bearing route for crush conditions.

What it CANNOT tell you
-----------------------
Which camera is wrong.  A disagreement localises the problem to a ground
patch, not to a cause.

It is also only as good as the homographies: a calibration error is
indistinguishable from a flow error here, and both cameras being wrong in the
same way (a shared systematic bias) cancels out and reads as agreement.  The
route therefore reports the calibration residual alongside the flow
disagreement, so a reader can see whether the geometry was trustworthy before
reading the velocity numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.crowd_flow.ground_plane import CameraCalibration, UncalibratedCamera
from models.crowd_flow.validation.report import (
    Measurement, RouteResult, STATUS_PASS,
)

logger = logging.getLogger(__name__)

ROUTE_KEY   = "cross_camera"
ROUTE_TITLE = "Route (b) — cross-camera agreement"
ROUTE_CAVEAT = (
    "Bounds the error without locating it: a disagreement says at least one "
    "camera is wrong, never which.  Depends entirely on the homographies — a "
    "calibration error reads as a flow error — and a bias shared by both "
    "cameras cancels out and looks like agreement."
)


@dataclass
class CameraView:
    """One camera's contribution to a cross-camera comparison."""
    calibration: CameraCalibration
    field_xy: np.ndarray          # (H, W, 2) px/frame, source resolution
    fps: float

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.field_xy.shape[0], self.field_xy.shape[1]


class CrossCameraValidator:
    """
    Compares two calibrated views of the same ground plane.

    Parameters
    ----------
    grid_spacing_m:
        Sample spacing on the ground, in metres.
    max_disagreement_ms:
        Tolerance on median ground-plane velocity disagreement.
    min_speed_ms:
        Ground points slower than this in BOTH views are skipped.  Two
        cameras agreeing that nothing is moving is not evidence that either
        measures motion correctly, and including those points would dilute
        the disagreement statistic towards zero with samples that carry no
        information.
    """

    def __init__(
        self,
        grid_spacing_m: float = 0.5,
        max_disagreement_ms: float = 0.35,
        min_speed_ms: float = 0.15,
    ) -> None:
        self.grid_spacing_m = grid_spacing_m
        self.max_disagreement_ms = max_disagreement_ms
        self.min_speed_ms = min_speed_ms

    # ------------------------------------------------------------------

    def _overlap_bounds(
        self, a: CameraView, b: CameraView
    ) -> Optional[tuple[float, float, float, float]]:
        """Intersection of the two cameras' ground footprints, or None."""
        ah, aw = a.shape_hw
        bh, bw = b.shape_hw
        ax0, ay0, ax1, ay1 = a.calibration.image_footprint_world(aw, ah)
        bx0, by0, bx1, by1 = b.calibration.image_footprint_world(bw, bh)

        x0, y0 = max(ax0, bx0), max(ay0, by0)
        x1, y1 = min(ax1, bx1), min(ay1, by1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _sample_ground_velocity(
        view: CameraView, pts_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ground-plane velocity (m/s) at the given world points, as seen by one
        camera.

        Returns (velocities (N,2), valid mask (N,)).  A point is invalid when
        it falls outside this camera's image.

        The conversion is exact rather than linearised: the image point and
        its flow-displaced neighbour are each mapped to the ground and
        differenced there, so perspective is handled correctly even far from
        the calibration points.
        """
        h, w = view.shape_hw
        px = view.calibration.world_to_pixel_many(pts_world)

        inside = (
            (px[:, 0] >= 0) & (px[:, 0] < w - 1) &
            (px[:, 1] >= 0) & (px[:, 1] < h - 1)
        )

        vel = np.zeros((len(pts_world), 2), dtype=np.float64)
        if not inside.any():
            return vel, inside

        xi = np.clip(px[inside, 0].astype(np.int32), 0, w - 1)
        yi = np.clip(px[inside, 1].astype(np.int32), 0, h - 1)
        dx = view.field_xy[yi, xi, 0].astype(np.float64)
        dy = view.field_xy[yi, xi, 1].astype(np.float64)

        src = np.stack([px[inside, 0], px[inside, 1]], axis=1)
        dst = np.stack([px[inside, 0] + dx, px[inside, 1] + dy], axis=1)

        import cv2
        w0 = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), view.calibration.H).reshape(-1, 2)
        w1 = cv2.perspectiveTransform(
            dst.reshape(-1, 1, 2), view.calibration.H).reshape(-1, 2)

        vel[inside] = (w1 - w0) * view.fps
        return vel, inside

    @staticmethod
    def _calibration_residual(cal: CameraCalibration) -> float:
        """
        Mean reprojection error of the calibration's own control points, in
        metres.  Reported so a reader can tell a geometry problem from a flow
        problem before interpreting the disagreement.
        """
        if not cal.is_calibrated or not cal.image_points:
            return float("nan")
        img = np.asarray(cal.image_points, dtype=np.float64)
        world = np.asarray(cal.world_points_m, dtype=np.float64)
        import cv2
        pred = cv2.perspectiveTransform(img.reshape(-1, 1, 2), cal.H).reshape(-1, 2)
        return float(np.linalg.norm(pred - world, axis=1).mean())

    # ------------------------------------------------------------------

    def run(self, a: CameraView, b: CameraView) -> RouteResult:
        if not a.calibration.is_calibrated or not b.calibration.is_calibrated:
            return RouteResult.skipped(
                ROUTE_KEY, ROUTE_TITLE,
                "Both cameras must have a homography configured; at least one "
                "is uncalibrated.  Run scripts/calibrate_ground_plane.py for "
                "each camera.",
                ROUTE_CAVEAT,
            )

        bounds = self._overlap_bounds(a, b)
        if bounds is None:
            return RouteResult.skipped(
                ROUTE_KEY, ROUTE_TITLE,
                "The two cameras' ground footprints do not overlap, so there "
                "is no common region to compare.",
                ROUTE_CAVEAT,
            )

        x0, y0, x1, y1 = bounds
        step = self.grid_spacing_m
        gx = np.arange(x0, x1, step)
        gy = np.arange(y0, y1, step)
        if len(gx) < 2 or len(gy) < 2:
            return RouteResult.skipped(
                ROUTE_KEY, ROUTE_TITLE,
                f"Overlap region ({x1-x0:.1f} x {y1-y0:.1f} m) is smaller than "
                f"the {step} m sample spacing.",
                ROUTE_CAVEAT,
            )

        mesh = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)

        vel_a, ok_a = self._sample_ground_velocity(a, mesh)
        vel_b, ok_b = self._sample_ground_velocity(b, mesh)
        both = ok_a & ok_b

        if not both.any():
            return RouteResult.skipped(
                ROUTE_KEY, ROUTE_TITLE,
                "No ground point in the overlap region is visible in both "
                "camera images.",
                ROUTE_CAVEAT,
            )

        speed_a = np.linalg.norm(vel_a, axis=1)
        speed_b = np.linalg.norm(vel_b, axis=1)
        moving = both & ((speed_a > self.min_speed_ms) |
                         (speed_b > self.min_speed_ms))

        if not moving.any():
            return RouteResult.skipped(
                ROUTE_KEY, ROUTE_TITLE,
                f"No ground point in the overlap exceeds {self.min_speed_ms} "
                f"m/s in either view — nothing is moving, so agreement here "
                f"would carry no information.",
                ROUTE_CAVEAT,
            )

        disagreement = np.linalg.norm(vel_a[moving] - vel_b[moving], axis=1)
        mean_speed = 0.5 * (speed_a[moving] + speed_b[moving])
        relative = disagreement / np.maximum(mean_speed, 1e-6)

        res_a = self._calibration_residual(a.calibration)
        res_b = self._calibration_residual(b.calibration)

        result = RouteResult(
            route=ROUTE_KEY, title=ROUTE_TITLE, status=STATUS_PASS,
            summary=(
                f"{int(moving.sum())} moving ground points compared over a "
                f"{x1-x0:.1f} x {y1-y0:.1f} m overlap; median disagreement "
                f"{float(np.median(disagreement)):.3f} m/s"
            ),
            caveat=ROUTE_CAVEAT,
            measurements=[
                Measurement("Median disagreement", float(np.median(disagreement)),
                            "m/s", tolerance=self.max_disagreement_ms),
                Measurement("95th-percentile disagreement",
                            float(np.percentile(disagreement, 95)), "m/s",
                            tolerance=self.max_disagreement_ms * 3),
                Measurement("Median relative disagreement",
                            float(np.median(relative)), "fraction"),
                Measurement("Comparable moving points", float(moving.sum()),
                            "points"),
                Measurement(f"Calibration residual ({a.calibration.camera_id})",
                            res_a, "m",
                            note="geometry error floor; flow cannot beat this"),
                Measurement(f"Calibration residual ({b.calibration.camera_id})",
                            res_b, "m",
                            note="geometry error floor; flow cannot beat this"),
            ],
            detail={
                "overlap_m": [float(x0), float(y0), float(x1), float(y1)],
                "grid_spacing_m": step,
                "points_in_both_views": int(both.sum()),
                "mean_ground_speed_ms": float(mean_speed.mean()),
            },
        )
        result.resolve_status()
        return result
