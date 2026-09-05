"""
Perspective correction: pixel velocity → ground-plane velocity.

A person 60 m away on a ghat approach moves a fraction of the pixels a person
10 m away does at the same ground speed.  Without a per-camera homography,
speed values are in pixels/frame and are neither comparable across cameras nor
interpretable as m/s — making threshold-based alerting meaningless.

CameraCalibration
-----------------
Holds a homography H (3×3) mapping image coordinates to ground-plane
coordinates (in metres).  Call pixel_velocity_to_ms() to convert a flow
vector at a given image position into a (vx_ms, vy_ms) pair.

The conversion uses the local Jacobian of the homography:

    world_pt  = H  ·  [x,  y,  1]ᵀ   (normalised)
    world_pt' = H  ·  [x+dx, y+dy, 1]ᵀ
    world_velocity = (world_pt' − world_pt) × fps

This is exact (non-linearised) and valid for any projective homography.

Uncalibrated cameras
--------------------
If no calibration is present, CameraCalibration.is_calibrated is False.
Calling pixel_velocity_to_ms() on an uncalibrated instance raises
UncalibratedCamera.  Callers MUST check is_calibrated first; speed values in
uncalibrated mode are returned as px/frame from a separate code path that
annotates them with 'UNCALIBRATED' in the units string.

Speed-unit contract:
  calibrated   → units string "m/s"
  uncalibrated → units string "px/frame (UNCALIBRATED)"

Alert thresholds that rely on absolute speed (speed_low_ms, speed_high_ms)
are disabled for uncalibrated cameras at AlertEngine startup — they emit a
single WARNING then skip those checks for the whole run.  They do not silently
pretend px/frame values are m/s.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Central-difference step for the image->ground Jacobian, in pixels.  Large
# enough that the difference is well clear of float64 cancellation, small
# enough that the linearisation holds across it on a normal mount.
_JACOBIAN_DELTA_PX = 0.5

SPEED_UNITS_CALIBRATED   = "m/s"
SPEED_UNITS_UNCALIBRATED = "px/frame (UNCALIBRATED)"


class UncalibratedCamera(RuntimeError):
    """
    Raised when a calibration-dependent operation is requested on a camera
    that has no homography configured.

    Catch this at the call site and report pixel-space values instead.
    """


@dataclass
class CameraCalibration:
    """
    Per-camera homography + helpers for converting pixel flow to m/s.

    Parameters
    ----------
    camera_id : str
        Human-readable camera name (matches the key in crowd_flow.yaml).
    image_points : list of [x, y] pixel coordinates (≥ 4).
    world_points_m : list of [X, Y] ground-plane coordinates in metres (≥ 4).
        The coordinate system is arbitrary but must be metric and consistent
        within a camera.  A convenient choice: origin at one corner of the
        zone, Y pointing away from the camera, X to the right.
    H : np.ndarray (3, 3)
        Computed homography.  Built from image/world points via from_points();
        can also be supplied directly for advanced use.
    """
    camera_id:    str
    image_points: list                    # [[x,y], ...], stored for reference
    world_points_m: list                  # [[X,Y], ...], stored for reference
    H: Optional[np.ndarray] = field(default=None, repr=False)
    perspective_map: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.H is not None:
            self.H = np.asarray(self.H, dtype=np.float64)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_points(
        cls,
        camera_id: str,
        image_points: list,
        world_points_m: list,
    ) -> "CameraCalibration":
        """
        Build a CameraCalibration by computing the homography from ≥ 4
        matched image-to-ground-plane point pairs.

        Raises ValueError if fewer than 4 points are provided or if
        cv2.findHomography fails to find a valid solution.
        """
        if len(image_points) < 4 or len(world_points_m) < 4:
            raise ValueError(
                f"At least 4 point pairs are required for homography estimation "
                f"(got {len(image_points)} image and {len(world_points_m)} world points)."
            )
        if len(image_points) != len(world_points_m):
            raise ValueError(
                f"image_points and world_points_m must have the same length "
                f"({len(image_points)} vs {len(world_points_m)})."
            )

        src = np.array(image_points, dtype=np.float64)
        dst = np.array(world_points_m, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

        if H is None:
            raise ValueError(
                "cv2.findHomography failed to find a valid homography.  "
                "Check that your control points are not collinear and that "
                "the image-to-world correspondence is correct."
            )

        inliers = int(mask.sum()) if mask is not None else 0
        logger.info(
            "Homography computed for camera '%s' with %d/%d inliers.",
            camera_id, inliers, len(image_points),
        )

        return cls(
            camera_id=camera_id,
            image_points=image_points,
            world_points_m=world_points_m,
            H=H,
        )

    @classmethod
    def from_yaml_block(cls, camera_id: str, block: dict) -> "CameraCalibration":
        """
        Build a CameraCalibration from the ``homography:`` or ``perspective_map:``
        block in configs/crowd_flow.yaml.  Returns an *uncalibrated* instance if
        neither is configured or valid.
        """
        # 1. Check perspective_map (bilinear person-height scale model)
        pm_block = block.get("perspective_map")
        if pm_block and isinstance(pm_block, dict):
            from models.crowd_flow.perspective import PerspectiveMap
            pm = PerspectiveMap(
                aw=float(pm_block.get("aw", 0.0)),
                bw=float(pm_block.get("bw", 0.0)),
                cw=float(pm_block.get("cw", 0.0)),
                ah=float(pm_block.get("ah", 0.0)),
                bh=float(pm_block.get("bh", 0.0)),
                ch=float(pm_block.get("ch", 0.0)),
                image_width=int(pm_block.get("image_width", 0)),
                image_height=int(pm_block.get("image_height", 0)),
                person_height_m=float(pm_block.get("person_height_m", 1.70)),
                source=str(pm_block.get("source", "perspective_map")),
                fit_rmse_px=float(pm_block["fit_rmse_px"]) if pm_block.get("fit_rmse_px") is not None else None,
                n_fit_samples=int(pm_block.get("n_fit_samples", 0)),
            )
            logger.info("Camera '%s': loaded perspective map (%s)", camera_id, pm.describe())
            return cls(camera_id=camera_id, image_points=[], world_points_m=[], H=None, perspective_map=pm)

        # 2. Check homography
        hom = block.get("homography")
        if not hom:
            logger.warning(
                "Camera '%s' has no homography or perspective map configured.  "
                "Speed values will be in px/frame.  "
                "Run scripts/calibrate_ground_plane.py or PerspectiveMap.fit_from_boxes.",
                camera_id,
            )
            return cls(camera_id=camera_id, image_points=[], world_points_m=[])

        img_pts   = hom.get("image_points", [])
        world_pts = hom.get("world_points_m", [])

        if not img_pts or not world_pts:
            logger.warning(
                "Camera '%s': homography block present but image_points or "
                "world_points_m is empty.  Treating as uncalibrated.",
                camera_id,
            )
            return cls(camera_id=camera_id, image_points=[], world_points_m=[])

        return cls.from_points(camera_id, img_pts, world_pts)

    def to_yaml_block(self) -> dict:
        """Serialise this calibration to a dict suitable for YAML output."""
        if not self.is_calibrated:
            return {}
        if self.perspective_map is not None:
            return {"perspective_map": self.perspective_map.to_dict()}
        return {
            "image_points":   self.image_points,
            "world_points_m": self.world_points_m,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None or self.perspective_map is not None

    @property
    def speed_units(self) -> str:
        return SPEED_UNITS_CALIBRATED if self.is_calibrated else SPEED_UNITS_UNCALIBRATED

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        """
        Map a single image point (x, y) to ground-plane coordinates (X, Y) in
        metres.  Raises UncalibratedCamera if no calibration is configured.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no calibration; "
                "cannot convert pixel coordinates to world coordinates."
            )
        if self.perspective_map is not None:
            mpp = self.perspective_map.metres_per_pixel(x, y)
            return float(x * mpp), float(y * mpp)
        p_img = np.array([[[x, y]]], dtype=np.float64)
        p_world = cv2.perspectiveTransform(p_img, self.H)
        return float(p_world[0, 0, 0]), float(p_world[0, 0, 1])

    def world_to_pixel(self, X: float, Y: float) -> tuple[float, float]:
        """
        Map a ground-plane point (X, Y) in metres back to image coordinates.

        The inverse of pixel_to_world.  Needed to ask "where does this patch of
        ground appear in this camera?", which is how two cameras are brought
        into correspondence for cross-camera agreement checks.

        Raises UncalibratedCamera if no calibration is configured.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no calibration; "
                "cannot convert world coordinates to pixel coordinates."
            )
        if self.perspective_map is not None:
            cx, cy = self.perspective_map.image_width / 2.0, self.perspective_map.image_height / 2.0
            mpp = self.perspective_map.metres_per_pixel(cx, cy)
            return float(X / max(mpp, 1e-6)), float(Y / max(mpp, 1e-6))
        H_inv = np.linalg.inv(self.H)
        p_world = np.array([[[X, Y]]], dtype=np.float64)
        p_img = cv2.perspectiveTransform(p_world, H_inv)
        return float(p_img[0, 0, 0]), float(p_img[0, 0, 1])

    def world_to_pixel_many(self, pts_world: np.ndarray) -> np.ndarray:
        """
        Batch form of world_to_pixel.

        pts_world: (N, 2) in metres.  Returns (N, 2) image coordinates.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no homography."
            )
        H_inv = np.linalg.inv(self.H)
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H_inv)
        return out.reshape(-1, 2)

    def image_footprint_world(
        self, width: int, height: int
    ) -> tuple[float, float, float, float]:
        """
        Axis-aligned ground-plane bounds (min_X, min_Y, max_X, max_Y) of this
        camera's image rectangle, in metres.

        Used to find the ground region two cameras have in common.  The bound
        is the corners' bounding box, which over-covers a perspective view --
        it is a cheap superset, and points inside it are re-checked against
        each camera's image bounds before use.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no calibration."
            )
        if self.perspective_map is not None:
            x0, y0 = self.pixel_to_world(0.0, 0.0)
            x1, y1 = self.pixel_to_world(float(width), float(height))
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        corners = np.array(
            [[[0.0, 0.0]], [[width, 0.0]], [[width, height]], [[0.0, height]]],
            dtype=np.float64,
        )
        world = cv2.perspectiveTransform(corners, self.H).reshape(-1, 2)
        return (float(world[:, 0].min()), float(world[:, 1].min()),
                float(world[:, 0].max()), float(world[:, 1].max()))

    def pixel_velocity_to_ms(
        self,
        x: float,
        y: float,
        dx_px: float,
        dy_px: float,
        fps: float,
    ) -> tuple[float, float]:
        """
        Convert a pixel flow vector (dx_px, dy_px) at image position (x, y) to
        a ground-plane velocity (vx_ms, vy_ms) in m/s.

        Uses the exact (non-linearised) homography when available, or the local
        metres-per-pixel scale when calibrated via a perspective map.

        Raises UncalibratedCamera if no calibration is configured.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no calibration; "
                "cannot convert pixel velocity to m/s."
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        if self.perspective_map is not None:
            mpp = self.perspective_map.metres_per_pixel(x, y)
            return float(dx_px * mpp * fps), float(dy_px * mpp * fps)

        X0, Y0 = self.pixel_to_world(x,          y)
        X1, Y1 = self.pixel_to_world(x + dx_px,  y + dy_px)

        # World displacement in metres → divide by frame duration for m/s
        vx_ms = (X1 - X0) * fps
        vy_ms = (Y1 - Y0) * fps
        return vx_ms, vy_ms

    def flow_field_to_ms(
        self,
        field_xy: np.ndarray,
        fps: float,
        grid_cell_px: int = 1,
    ) -> np.ndarray:
        """
        Convert an entire flow field (H, W, 2) from px/frame to m/s.

        The conversion factor — not the velocity — is what gets sampled at
        ``grid_cell_px`` spacing.  The local Jacobian of the homography (or
        the perspective scale) is computed at each sample point, bilinearly
        upsampled, and applied to the field **per pixel**, so every pixel
        keeps its own velocity.

        Returns an (H, W, 2) float32 array in m/s, or raises UncalibratedCamera.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no calibration."
            )

        h, w = field_xy.shape[:2]
        step = max(1, grid_cell_px)

        # Compute at sampled positions
        xs = np.arange(step // 2, w, step)
        ys = np.arange(step // 2, h, step)
        n_y, n_x = len(ys), len(xs)

        if self.perspective_map is not None:
            mpp_grid = np.empty((n_y, n_x), dtype=np.float32)
            for i, yy in enumerate(ys):
                for j, xx in enumerate(xs):
                    mpp_grid[i, j] = self.perspective_map.metres_per_pixel(float(xx), float(yy)) * fps
            if (n_y, n_x) != (h, w):
                scale_full = cv2.resize(mpp_grid, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                scale_full = mpp_grid
            out = np.empty((h, w, 2), dtype=np.float32)
            out[..., 0] = field_xy[..., 0].astype(np.float32) * scale_full
            out[..., 1] = field_xy[..., 1].astype(np.float32) * scale_full
            return out

        # Local Jacobian of the image -> ground map at each sample point,
        # by central difference over _JACOBIAN_DELTA_PX.
        d = _JACOBIAN_DELTA_PX
        base = np.array(
            [[[float(x), float(y)] for x in xs] for y in ys],
            dtype=np.float64,
        ).reshape(-1, 1, 2)

        def _world(offset_x: float, offset_y: float) -> np.ndarray:
            pts = base.copy()
            pts[:, 0, 0] += offset_x
            pts[:, 0, 1] += offset_y
            return cv2.perspectiveTransform(pts, self.H)[:, 0, :]

        wx_p, wx_m = _world(d, 0.0), _world(-d, 0.0)
        wy_p, wy_m = _world(0.0, d), _world(0.0, -d)

        # J = [[dX/dx, dX/dy], [dY/dx, dY/dy]], metres per pixel.
        jac = np.empty((n_y, n_x, 4), dtype=np.float32)
        jac[..., 0] = ((wx_p[:, 0] - wx_m[:, 0]) / (2 * d)).reshape(n_y, n_x)
        jac[..., 1] = ((wy_p[:, 0] - wy_m[:, 0]) / (2 * d)).reshape(n_y, n_x)
        jac[..., 2] = ((wx_p[:, 1] - wx_m[:, 1]) / (2 * d)).reshape(n_y, n_x)
        jac[..., 3] = ((wy_p[:, 1] - wy_m[:, 1]) / (2 * d)).reshape(n_y, n_x)

        if (n_y, n_x) != (h, w):
            # Bilinear, not nearest: the conversion factor varies smoothly
            # with perspective, and a piecewise-constant one puts a visible
            # step in the speed field at every cell boundary.
            jac_full = cv2.resize(jac, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            jac_full = jac

        dx = field_xy[..., 0].astype(np.float32)
        dy = field_xy[..., 1].astype(np.float32)

        out = np.empty((h, w, 2), dtype=np.float32)
        out[..., 0] = (jac_full[..., 0] * dx + jac_full[..., 1] * dy) * fps
        out[..., 1] = (jac_full[..., 2] * dx + jac_full[..., 3] * dy) * fps
        return out
