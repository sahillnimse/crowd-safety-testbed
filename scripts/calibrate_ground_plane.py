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


def _read_config_text(config_path: str) -> str:
    """
    Read the config as text with EXPLICIT utf-8 decoding.

    Text-mode open() without encoding= uses the platform codec (cp1252 on
    Windows), and this file is full of non-ASCII characters — that is the
    "'charmap' codec can't decode byte 0x90" failure. newline="" keeps every
    line ending exactly as it is on disk so the surgical patch below can
    write bytes back without newline translation.
    """
    with open(config_path, encoding="utf-8", newline="") as f:
        return f.read()


def _homography_lines(
    image_pts: list, world_pts: list, newline: str,
) -> list[str]:
    """
    Render the homography block exactly in house style (matching the example
    in this module's docstring and the rest of configs/crowd_flow.yaml):

          homography:
            image_points:
              - [142.0, 387.0]
            ...
    """
    img = [[round(float(v), 1) for v in p] for p in image_pts]
    wrd = [[round(float(v), 4) for v in p] for p in world_pts]
    lines = ["      homography:", "        image_points:"]
    lines += [f"          - [{x:.1f}, {y:.1f}]" for x, y in img]
    lines.append("        world_points_m:")
    lines += [f"          - [{x:.4f}, {y:.4f}]" for x, y in wrd]
    return [ln + newline for ln in lines]


def _patched_text(
    original: str, camera_id: str, image_pts: list, world_pts: list,
) -> str | None:
    """
    Return the config text with ONLY the target camera's homography block
    added/replaced — every other byte of the file, comments included, passes
    through untouched. Returns None if the structure can't be matched.

    Why not yaml.safe_load -> yaml.dump: PyYAML does not preserve comments,
    so a round-trip silently deletes ~600 lines of documented rationale from
    configs/crowd_flow.yaml. This patcher edits lines instead.
    """
    import re

    nl = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines(keepends=True)

    def strip_end(s: str) -> str:
        return s.rstrip("\r\n")

    def indent_of(s: str) -> int:
        return len(strip_end(s)) - len(strip_end(s).lstrip(" "))

    new_block = _homography_lines(image_pts, world_pts, nl)
    cam_re = re.compile(r"^ {4}" + re.escape(camera_id) + r":(\s.*)?$")

    cam_idx = next(
        (i for i, ln in enumerate(lines) if cam_re.match(strip_end(ln))), None
    )

    if cam_idx is None:
        # Unknown camera: insert a fresh block at the end of `cameras:`.
        cams_idx = next(
            (i for i, ln in enumerate(lines)
             if re.match(r"^ {2}cameras:\s*$", strip_end(ln))),
            None,
        )
        if cams_idx is None:
            return None
        end = len(lines)
        for i in range(cams_idx + 1, len(lines)):
            s = strip_end(lines[i])
            if s.strip() and indent_of(lines[i]) <= 2:
                end = i
                break
        # Keep a blank separator line between camera blocks, matching the
        # file's existing style.
        prefix = [] if (end == 0 or not strip_end(lines[end - 1]).strip()) else [nl]
        lines[end:end] = prefix + [f"    {camera_id}:{nl}"] + new_block + [nl]
        return "".join(lines)

    # Camera exists. Find its extent (first non-blank line indented <= 4).
    cam_end = len(lines)
    for i in range(cam_idx + 1, len(lines)):
        s = strip_end(lines[i])
        if s.strip() and indent_of(lines[i]) <= 4:
            cam_end = i
            break

    # Find an existing `      homography:` inside the block and its extent
    # (first non-blank line indented <= 6, e.g. the `      zones:` sibling).
    homo_start = next(
        (i for i in range(cam_idx + 1, cam_end)
         if re.match(r"^ {6}homography:\s*$", strip_end(lines[i]))),
        None,
    )
    if homo_start is not None:
        homo_end = cam_end
        for i in range(homo_start + 1, cam_end):
            s = strip_end(lines[i])
            if s.strip() and indent_of(lines[i]) <= 6:
                homo_end = i
                break
        lines[homo_start:homo_end] = new_block
    else:
        lines[cam_idx + 1:cam_idx + 1] = new_block
    return "".join(lines)


def _patch_config(config_path: str, camera_id: str,
                  image_pts: list, world_pts: list) -> None:
    """
    Write the homography block into configs/crowd_flow.yaml IN PLACE without
    disturbing any other content: comments, formatting and unrelated cameras
    pass through untouched (a load->dump round-trip would strip every comment
    in the file).

    Safety rails:
      - read/write with explicit utf-8 (platform codec crashed on this file);
      - the patched text must re-parse AND contain the expected points before
        it is written;
      - on ANY mismatch the original file is left untouched and the block is
        printed for manual insertion instead.
    """
    try:
        original = _read_config_text(config_path)
        patched = _patched_text(original, camera_id, image_pts, world_pts)
        if patched is None:
            raise ValueError(
                f"could not locate cameras.{camera_id}: (or cameras:) to patch"
            )

        # Verify BEFORE writing: must parse, and the new points must be there.
        check = yaml.safe_load(patched)
        got = check["crowd_flow"]["cameras"][camera_id]["homography"]
        want_img = [[round(float(v), 1) for v in p] for p in image_pts]
        want_wrd = [[round(float(v), 4) for v in p] for p in world_pts]
        if ([list(map(float, p)) for p in got["image_points"]] != want_img
                or [list(map(float, p)) for p in got["world_points_m"]] != want_wrd):
            raise ValueError("patched content failed verification")

        with open(config_path, "w", encoding="utf-8", newline="") as f:
            f.write(patched)

        logger.info("Homography written to %s under cameras.%s "
                    "(all other content preserved)", config_path, camera_id)

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
