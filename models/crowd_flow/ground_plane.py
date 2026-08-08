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
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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
        Build a CameraCalibration from the ``homography:`` block in
        configs/crowd_flow.yaml.  Returns an *uncalibrated* instance if the
        block is absent or empty.
        """
        hom = block.get("homography")
        if not hom:
            logger.warning(
                "Camera '%s' has no homography configured.  "
                "Speed values will be in px/frame.  "
                "Run scripts/calibrate_ground_plane.py to generate this block.",
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
        return {
            "image_points":   self.image_points,
            "world_points_m": self.world_points_m,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    @property
    def speed_units(self) -> str:
        return SPEED_UNITS_CALIBRATED if self.is_calibrated else SPEED_UNITS_UNCALIBRATED

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        """
        Map a single image point (x, y) to ground-plane coordinates (X, Y) in
        metres.  Raises UncalibratedCamera if no homography is configured.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no homography; "
                "cannot convert pixel coordinates to world coordinates."
            )
        p_img = np.array([[[x, y]]], dtype=np.float64)
        p_world = cv2.perspectiveTransform(p_img, self.H)
        return float(p_world[0, 0, 0]), float(p_world[0, 0, 1])

    def world_to_pixel(self, X: float, Y: float) -> tuple[float, float]:
        """
        Map a ground-plane point (X, Y) in metres back to image coordinates.

        The inverse of pixel_to_world.  Needed to ask "where does this patch of
        ground appear in this camera?", which is how two cameras are brought
        into correspondence for cross-camera agreement checks.

        Raises UncalibratedCamera if no homography is configured.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no homography; "
                "cannot convert world coordinates to pixel coordinates."
            )
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
                f"Camera '{self.camera_id}' has no homography."
            )
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

        Uses the exact (non-linearised) homography: map both the origin point
        and the displaced point to world coordinates, then divide by 1/fps.

        Raises UncalibratedCamera if no homography is configured.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no homography; "
                "cannot convert pixel velocity to m/s."
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

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

        For efficiency, conversion is computed at grid_cell_px spacing (i.e.
        at cell centres when grid_cell_px matches the metrics grid) rather than
        per-pixel.  The result is then nearest-neighbour upsampled to the
        original field shape.

        Returns an (H, W, 2) float32 array in m/s, or raises UncalibratedCamera.
        """
        if not self.is_calibrated:
            raise UncalibratedCamera(
                f"Camera '{self.camera_id}' has no homography."
            )

        h, w = field_xy.shape[:2]
        step = max(1, grid_cell_px)

        # Compute at sampled positions
        xs = np.arange(step // 2, w, step)
        ys = np.arange(step // 2, h, step)

        n_y, n_x = len(ys), len(xs)
        result_small = np.zeros((n_y, n_x, 2), dtype=np.float32)

        # Build arrays of image points and displaced points for batch transform
        pts_orig = np.array(
            [[[float(x), float(y)] for x in xs] for y in ys],
            dtype=np.float64,
        ).reshape(-1, 1, 2)

        dx_arr = field_xy[
            np.clip(ys[:, None], 0, h-1),
            np.clip(xs[None, :], 0, w-1),
            0,
        ].ravel()
        dy_arr = field_xy[
            np.clip(ys[:, None], 0, h-1),
            np.clip(xs[None, :], 0, w-1),
            1,
        ].ravel()

        pts_disp = pts_orig.copy()
        pts_disp[:, 0, 0] += dx_arr
        pts_disp[:, 0, 1] += dy_arr

        world_orig = cv2.perspectiveTransform(pts_orig, self.H)  # (N,1,2)
        world_disp = cv2.perspectiveTransform(pts_disp, self.H)

        vx_ms = ((world_disp[:, 0, 0] - world_orig[:, 0, 0]) * fps).reshape(n_y, n_x)
        vy_ms = ((world_disp[:, 0, 1] - world_orig[:, 0, 1]) * fps).reshape(n_y, n_x)

        result_small[..., 0] = vx_ms.astype(np.float32)
        result_small[..., 1] = vy_ms.astype(np.float32)

        if step == 1:
            return result_small

        # Upsample back to source resolution
        return cv2.resize(result_small, (w, h), interpolation=cv2.INTER_NEAREST)
