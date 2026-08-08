"""
Interactive ground-plane calibration helper.

Opens a resized OpenCV window over a single video frame (or image).  Click
≥ 4 image points that correspond to known ground-plane distances.  After
clicking, enter the real-world coordinates (in metres) at the prompt.  The
script prints the homography YAML block to stdout -- paste it into the
``cameras.<camera_id>.homography:`` section of configs/crowd_flow.yaml.

Usage
-----
  # From a video (uses the first frame):
  python scripts/calibrate_ground_plane.py \\
      --source test_videos/ram_kund.mp4 \\
      --camera ram_kund_approach \\
      --output-config configs/crowd_flow.yaml

  # From an image:
  python scripts/calibrate_ground_plane.py \\
      --source frame.jpg \\
      --camera ram_kund_approach

Instructions
------------
Choose points that are:
  - Spread across the full depth of the scene (near and far).
  - On the ground plane -- NOT on elevated surfaces (steps, walls).
  - At locations where you can measure or know the real-world distance
    (road markings, known tile sizes, surveyor stakes).

A minimum of 4 points is required.  6-8 improves robustness.

The resulting YAML block example:

  homography:
    image_points:
      - [142, 387]
      - [501, 352]
      - [638, 471]
      - [78,  482]
    world_points_m:
      - [0.0, 0.0]
      - [8.0, 0.0]
      - [8.0, 12.0]
      - [0.0, 12.0]

Verification
------------
After pasting the block and restarting the analyser, check that pedestrian
walking speed in an uncrowded segment reads as 1.0–1.4 m/s.  If it reads
~10× higher or lower, your world points are in cm or km, not metres.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Maximum display size for the calibration window
_MAX_DISPLAY_W = 1280
_MAX_DISPLAY_H = 720

_clicked_points: list[tuple[int, int]] = []
_display_scale: float = 1.0
_display_frame: np.ndarray | None = None


def _mouse_callback(event, x, y, flags, param):
    global _clicked_points, _display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_points.append((x, y))
        # Draw the clicked point
        cv2.circle(_display_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            _display_frame,
            f"P{len(_clicked_points)}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        cv2.imshow("Calibration -- click ground-plane points (Q to finish)", _display_frame)
        logger.info("Point %d clicked: display=(%d, %d)", len(_clicked_points), x, y)


def _load_frame(source: str) -> np.ndarray:
    """Load the first frame of a video or an image file."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")
        return frame

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from: {source}")
    return frame


def _scale_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale frame to fit the display window.  Returns (scaled, scale)."""
    h, w = frame.shape[:2]
    scale = min(_MAX_DISPLAY_W / w, _MAX_DISPLAY_H / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h)), scale
    return frame.copy(), 1.0


def _collect_points(frame: np.ndarray) -> tuple[list, float]:
    """Open the OpenCV window and collect clicked points."""
    global _clicked_points, _display_frame, _display_scale
    _clicked_points = []
    _display_frame, _display_scale = _scale_frame(frame)

    window_name = "Calibration -- click ground-plane points (Q to finish)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_callback)

    print(
        "\n+---------------------------------------------------------+\n"
        "|  Click ≥ 4 ground-level points in the image.            |\n"
        "|  Points should be spread across the full scene depth.   |\n"
        "|  Press Q or close the window when done.                 |\n"
        "+---------------------------------------------------------+"
    )
    cv2.imshow(window_name, _display_frame)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return _clicked_points, _display_scale


def _scale_to_source(
    display_pts: list[tuple[int, int]], scale: float
) -> list[list[float]]:
    """Convert display-resolution clicks back to source-resolution coordinates."""
    return [[x / scale, y / scale] for (x, y) in display_pts]


def _collect_world_points(n: int) -> list[list[float]]:
    """Prompt the user to enter real-world ground coordinates for each clicked point."""
    print(
        f"\nEnter real-world ground-plane coordinates (in METRES) for each of "
        f"the {n} clicked points.\n"
        "Format: X Y  (e.g.  0 0  or  8.5 12.3)\n"
        "Tip: choose a convenient origin (e.g. a survey stake or tile corner).\n"
    )
    world_pts: list[list[float]] = []
    for i in range(n):
        while True:
            raw = input(f"  Point {i + 1} world coords (X Y): ").strip()
            parts = raw.split()
            if len(parts) == 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    world_pts.append([x, y])
                    break
                except ValueError:
                    pass
            print("  [FAIL]  Enter two numbers separated by a space.")
    return world_pts


def _verify_homography(
    image_pts: list[list[float]], world_pts: list[list[float]]
) -> None:
    """
    Quick sanity-check: compute the homography and print reprojection errors.
    Large errors (> 0.5 m) suggest mis-clicked points or measurement errors.
    """
    src = np.array(image_pts, dtype=np.float64)
    dst = np.array(world_pts, dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

    if H is None:
        logger.warning("Homography could not be computed -- collinear points?")
        return

    # Reproject and measure error in world coordinates
    src_h = np.concatenate([src, np.ones((len(src), 1))], axis=1)  # (N, 3)
    projected = (H @ src_h.T).T                                    # (N, 3)
    projected /= projected[:, 2:3]

    errors = np.sqrt(
        (projected[:, 0] - dst[:, 0]) ** 2 + (projected[:, 1] - dst[:, 1]) ** 2
    )
    print("\nReprojection errors (world metres):")
    for i, (e, m) in enumerate(zip(errors, mask.ravel())):
        outlier = "" if m else " ← RANSAC outlier"
        print(f"  Point {i + 1}: {e:.3f} m{outlier}")

    inlier_errs = errors[mask.ravel().astype(bool)]
    if len(inlier_errs):
        print(f"\nMean inlier error: {inlier_errs.mean():.3f} m")
        if inlier_errs.mean() > 0.5:
            logger.warning(
                "Mean reprojection error %.3f m is large.  "
                "Check that world coordinates are in metres, not cm or other units, "
                "and that all points are on the ground plane (not steps or walls).",
                inlier_errs.mean(),
            )


def _build_yaml_block(
    camera_id: str, image_pts: list[list[float]], world_pts: list[list[float]]
) -> str:
    """Format the calibration block as a YAML string."""
    block = {
        "image_points":   [[round(v, 1) for v in p] for p in image_pts],
        "world_points_m": [[round(v, 4) for v in p] for p in world_pts],
    }
    inner = yaml.dump({"homography": block}, default_flow_style=False, indent=2)
    # Indent for insertion under cameras.<camera_id>:
    indented = "\n".join("      " + line for line in inner.splitlines())
    return f"    # Paste under cameras.{camera_id}: in configs/crowd_flow.yaml\n{indented}"


def _patch_config(config_path: str, camera_id: str,
                  image_pts: list, world_pts: list) -> None:
    """
    Write the homography block into an existing configs/crowd_flow.yaml in-place.

    This is a best-effort patch; if the YAML structure doesn't match, it falls
    back to printing the block for manual insertion.
    """
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        cameras = data.setdefault("crowd_flow", {}).setdefault("cameras", {})
        cam = cameras.setdefault(camera_id, {})
        cam["homography"] = {
            "image_points":   image_pts,
            "world_points_m": world_pts,
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Homography written to %s under cameras.%s", config_path, camera_id)

    except Exception as exc:
        logger.warning(
            "Could not patch %s: %s.  "
            "Copy the YAML block above into the file manually.", config_path, exc,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive ground-plane calibration for crowd_flow cameras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source",        required=True,
                    help="Video file or image to calibrate from.")
    ap.add_argument("--camera",        required=True,
                    help="Camera ID (must match a key in configs/crowd_flow.yaml cameras:).")
    ap.add_argument("--output-config", default="",
                    help="Path to crowd_flow.yaml to patch in-place (optional).")
    args = ap.parse_args()

    # Load frame
    logger.info("Loading source: %s", args.source)
    frame = _load_frame(args.source)
    h, w = frame.shape[:2]
    logger.info("Frame size: %dx%d", w, h)

    # Collect clicks
    display_pts, scale = _collect_points(frame)

    if len(display_pts) < 4:
        print(f"\n[FAIL]  Need at least 4 points; got {len(display_pts)}.  Exiting.")
        sys.exit(1)

    image_pts = _scale_to_source(display_pts, scale)
    logger.info("Collected %d image points.", len(image_pts))

    # Collect world coordinates
    world_pts = _collect_world_points(len(image_pts))

    # Verify
    _verify_homography(image_pts, world_pts)

    # Print YAML block
    yaml_block = _build_yaml_block(args.camera, image_pts, world_pts)
    print(f"\n{'-' * 60}")
    print("YAML block (paste into configs/crowd_flow.yaml):")
    print(f"{'-' * 60}")
    print(yaml_block)
    print(f"{'-' * 60}\n")

    # Optionally patch the config file
    if args.output_config:
        _patch_config(args.output_config, args.camera, image_pts, world_pts)


if __name__ == "__main__":
    main()
