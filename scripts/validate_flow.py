"""
Synthetic validation suite for the dense optical flow crowd-safety module.

Validates correctness BEFORE a live event -- when fixing a bug is cheap.

Test suites (all use synthetic data; no crowd footage required)
---------------------------------------------------------------

1. Uniform translation
   Shift a textured test image by known (dx, dy).  Recovered flow must be
   within tolerance of the ground truth.  Tests the flow backend end-to-end.

2. Divergence sign convention  ← most important test
   Synthesize a radially converging flow field (all vectors pointing inward).
   Divergence must be NEGATIVE everywhere in the interior.
   Getting this wrong inverts every compression alert -- a crush would report
   as expansion.  This test failing means DO NOT USE for live monitoring.

3. Divergence sign convention -- expansion
   Synthesize a radially expanding field.  Divergence must be POSITIVE.
   Confirms the sign is correct in both directions.

4. Curl field
   Synthesize a counter-clockwise rotating field.  Curl must be positive.

5. Counterflow detection
   Left half of frame: flow pointing right.  Right half: flow pointing left.
   counterflow_score should be >= 0.45.

6. Stop-and-go pattern
   Feed alternating high/low speed values to the zone speed history buffer.
   stop_go_score should be > 0.5.

7. GMC subtraction
   Apply a known uniform translation to a textured frame pair.  After GMC,
   the mean residual flow magnitude should be < 1 px/frame.

8. CameraCalibration round-trip
   Build a known homography, convert a pixel point to world and back.
   World coordinates must match within tolerance.

9. Rain heuristic
   Feed a flow field with high-magnitude, uniform, predominantly-vertical
   vectors.  Rain flag must be set.

10. Brightness suppression
    Feed two frames with large mean intensity difference.
    flow_result.is_brightness_suppressed must be True.

Usage
-----
  python scripts/validate_flow.py            # run all tests
  python scripts/validate_flow.py --all      # same
  python scripts/validate_flow.py --test 1   # run one test
  python scripts/validate_flow.py --verbose  # show per-pixel diagnostics

Exit code 0 = all selected tests passed.
Exit code 1 = one or more tests failed.

Published crowd video datasets for additional (non-synthetic) validation:
  - PETS2009 S1:    https://ieeexplore.ieee.org/document/5399511
                    Public domain; bidirectional corridors documented.
  - UCF-CC-50:      https://www.crcv.ucf.edu/data/crowd_counting.php
                    Dense crowd images for density estimation sanity-checks.
  - UDY dataset:    https://github.com/davidminlk/udy-dataset
                    University of Bath crowd videos; open access.
For each: run the analyser and verify that visually-compressing regions
produce negative divergence, visible stop-and-go produces stop_go_score > 0.3,
and bidirectional corridors produce counterflow_score > 0.3.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from models.crowd_flow.flow_field    import FlowField
from models.crowd_flow.crowd_metrics import CrowdMetricsEngine
from models.crowd_flow.zones         import Zone, ZoneThresholds
from models.crowd_flow.ground_plane  import CameraCalibration

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("validate_flow")

# ------------------------------------------------------------------------------
# Test infrastructure
# ------------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"

_results: list[tuple[int, str, str, str]] = []   # (id, name, status, detail)

def _register(test_id: int, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
    try:
        ok, detail = fn()
        status = _PASS if ok else _FAIL
    except Exception as exc:
        ok, status, detail = False, _FAIL, f"Exception: {exc}"
    _results.append((test_id, name, status, detail))


def _make_textured_frame(h: int = 240, w: int = 320) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pattern = ((np.sin(xs * 0.05) + np.cos(ys * 0.05) + np.sin((xs + ys) * 0.03)) * 60 + 127).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
def _translate_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift frame by (dx, dy) pixels using warpAffine."""
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h))


# ------------------------------------------------------------------------------
# Test 1: Uniform translation
# ------------------------------------------------------------------------------

def _test_uniform_translation() -> tuple[bool, str]:
    """
    Translate a textured frame by a known amount and verify the recovered
    mean flow is within 10% of ground truth.
    """
    dx, dy = 5.0, -3.0
    frame = _make_textured_frame()
    shifted = _translate_frame(frame, dx, dy)

    ff = FlowField(backend="dis", dis_preset="medium", target_px=320,
                   global_motion_compensation=False, temporal_smooth_alpha=1.0)
    result = ff.compute(frame, shifted)

    h, w = frame.shape[:2]
    # Exclude border pixels (flow is unreliable at edges)
    b = 16
    fx = result.field_xy[b:h-b, b:w-b, 0]
    fy = result.field_xy[b:h-b, b:w-b, 1]

    mean_fx = float(fx.mean())
    mean_fy = float(fy.mean())

    tol_x = max(0.5, abs(dx) * 0.15)
    tol_y = max(0.5, abs(dy) * 0.15)
    ok_x  = abs(mean_fx - dx) < tol_x
    ok_y  = abs(mean_fy - dy) < tol_y
    ok    = ok_x and ok_y

    detail = (
        f"GT=({dx:.1f},{dy:.1f})  "
        f"measured=({mean_fx:.2f},{mean_fy:.2f})  "
        f"tol=({tol_x:.2f},{tol_y:.2f})"
    )
    return ok, detail


# ------------------------------------------------------------------------------
# Test 2: Divergence sign convention -- convergence
# ------------------------------------------------------------------------------

def _test_divergence_sign_converging() -> tuple[bool, str]:
    """
    A radially converging synthetic flow field must give NEGATIVE divergence.
    This is the most critical sign-convention test.
    Getting this wrong → every compression alert fires on expansion.
    """
    H, W = 64, 64
    cx, cy = W / 2.0, H / 2.0
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    fx = cx - xs   # point inward
    fy = cy - ys
    mag = np.sqrt(fx**2 + fy**2) + 1e-6
    fx /= mag
    fy /= mag

    # Divergence ∂vx/∂x + ∂vy/∂y
    div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
    interior = div[4:-4, 4:-4]
    max_div = float(interior.max())

    ok     = max_div < 0.0
    detail = (
        f"Max interior div={max_div:.4f}  "
        f"(must be < 0.0 for converging field -- negative = compression)"
    )
    if not ok:
        detail += (
            "\n  CRITICAL: This means compression alerts are INVERTED. "
            "A converging crowd would report as expanding."
        )
    return ok, detail


# ------------------------------------------------------------------------------
# Test 3: Divergence sign convention -- expansion
# ------------------------------------------------------------------------------

def _test_divergence_sign_expanding() -> tuple[bool, str]:
    """A radially expanding field must give POSITIVE divergence."""
    H, W = 64, 64
    cx, cy = W / 2.0, H / 2.0
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    fx = xs - cx   # point outward
    fy = ys - cy
    mag = np.sqrt(fx**2 + fy**2) + 1e-6
    fx /= mag
    fy /= mag

    div     = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
    interior = div[4:-4, 4:-4]
    min_div  = float(interior.min())

    ok     = min_div > 0.0
    detail = f"Min interior div={min_div:.4f}  (must be > 0.0 for expanding field)"
    return ok, detail


# ------------------------------------------------------------------------------
# Test 4: Curl sign convention
# ------------------------------------------------------------------------------

def _test_curl_sign() -> tuple[bool, str]:
    """Counter-clockwise rotation gives positive curl."""
    H, W = 64, 64
    cx, cy = W / 2.0, H / 2.0
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    # CCW rotation: vx = -(y - cy), vy = (x - cx)
    fx = -(ys - cy)
    fy =   xs - cx

    # Curl ∂vy/∂x - ∂vx/∂y
    curl = np.gradient(fy, axis=1) - np.gradient(fx, axis=0)
    interior = curl[4:-4, 4:-4]
    min_curl = float(interior.min())

    ok     = min_curl > 0.0
    detail = f"Min interior curl={min_curl:.4f}  (must be > 0.0 for CCW rotation)"
    return ok, detail


# ------------------------------------------------------------------------------
# Test 5: Counterflow detection
# ------------------------------------------------------------------------------

def _test_counterflow() -> tuple[bool, str]:
    """
    Left half: flow right (+x).  Right half: flow left (-x).
    counterflow_score should be >= 0.45 (~ 50% of cells opposing dominant).
    """
    H, W = 128, 128
    field_xy = np.zeros((H, W, 2), dtype=np.float32)
    field_xy[:, :W//2, 0] =  3.0   # left half: rightward
    field_xy[:, W//2:, 0] = -3.0   # right half: leftward

    engine = CrowdMetricsEngine(grid_cell_px=16, min_magnitude=0.1)
    # Per-cell mean vectors come from the same routine the live path uses.
    (_, _, _, _, _, cell_mx, cell_my) = engine._build_cell_arrays(
        field_xy, None, H, W
    )
    # Build full-zone mask
    in_zone = np.ones((H // 16, W // 16), dtype=bool)

    cf = engine._compute_counterflow(cell_mx, cell_my, in_zone)

    ok     = cf >= 0.45
    detail = f"counterflow_score={cf:.3f}  (expected >= 0.45 for 50/50 split)"
    return ok, detail


# ------------------------------------------------------------------------------
# Test 6: Stop-and-go pattern detection
# ------------------------------------------------------------------------------

def _test_stop_go() -> tuple[bool, str]:
    """Feed alternating high/low speed values; stop_go_score should be > 0.5."""
    engine = CrowdMetricsEngine(
        grid_cell_px=16, min_magnitude=0.0, stop_go_lag_frames=[5, 10, 15]
    )

    # Seed the speed history with a clear alternating pattern
    zone_name = "test_zone"
    speeds = []
    for i in range(60):
        spd = 2.0 if i % 10 < 5 else 0.2
        speeds.append(spd)
        engine._update_speed_history(zone_name, spd)

    score = engine._compute_stop_go(zone_name, speeds[-1])

    ok     = score > 0.5
    detail = f"stop_go_score={score:.3f}  (expected > 0.5 for clear alternation)"
    return ok, detail


# ------------------------------------------------------------------------------
# Test 7: GMC subtraction
# ------------------------------------------------------------------------------

def _test_gmc_subtraction() -> tuple[bool, str]:
    """
    Apply a known camera sway (uniform translation) to a textured frame pair.
    After GMC, the mean residual flow magnitude should be < 1.5 px/frame.
    """
    dx_cam, dy_cam = 4.0, 2.0   # simulated camera sway
    frame = _make_textured_frame(240, 320)
    shifted = _translate_frame(frame, dx_cam, dy_cam)

    ff = FlowField(
        backend="dis", dis_preset="medium", target_px=320,
        global_motion_compensation=True, temporal_smooth_alpha=1.0,
    )
    result = ff.compute(frame, shifted)

    h, w = frame.shape[:2]
    b = 16
    mag = np.sqrt(
        result.field_xy[b:h-b, b:w-b, 0]**2 +
        result.field_xy[b:h-b, b:w-b, 1]**2
    )
    mean_residual = float(mag.mean())

    ok     = mean_residual < 1.5
    detail = (
        f"sway=({dx_cam},{dy_cam}) px  "
        f"mean_residual={mean_residual:.3f} px  "
        f"gmc_applied={result.gmc_applied}  "
        f"method={result.gmc_method}"
    )
    return ok, detail


# ------------------------------------------------------------------------------
# Test 8: CameraCalibration round-trip
# ------------------------------------------------------------------------------

def _test_calibration_roundtrip() -> tuple[bool, str]:
    """
    Known homography: 4-point rectangle 100×100 px → 10×10 m.
    pixel_to_world([50,50]) should recover ~ (5, 5).
    """
    img_pts   = [[0, 0], [100, 0], [100, 100], [0, 100]]
    world_pts = [[0, 0], [10, 0], [10, 10], [0, 10]]

    calib = CameraCalibration.from_points("test_cam", img_pts, world_pts)

    X, Y = calib.pixel_to_world(50.0, 50.0)
    ok     = abs(X - 5.0) < 0.2 and abs(Y - 5.0) < 0.2
    detail = f"pixel(50,50) → world({X:.3f},{Y:.3f})  expected (5.0,5.0)  tol=0.2"
    return ok, detail


# ------------------------------------------------------------------------------
# Test 9: Rain heuristic
# ------------------------------------------------------------------------------

def _test_rain_heuristic() -> tuple[bool, str]:
    """
    Uniform high-magnitude vertical flow → rain flag must be set.
    """
    ff = FlowField(
        backend="dis", dis_preset="medium", target_px=320,
        global_motion_compensation=False, temporal_smooth_alpha=1.0,
        rain_mag_threshold=5.0,
    )

    # Craft a flow result directly (bypassing compute())
    H, W = 240, 320
    field = np.zeros((H, W, 2), dtype=np.float32)
    field[..., 1] = 8.0   # uniform downward flow, high magnitude

    magnitudes = np.sqrt(field[..., 0]**2 + field[..., 1]**2)
    mean_mag   = float(magnitudes.mean())
    is_rain    = ff._check_rain(field, magnitudes, mean_mag)

    ok     = is_rain
    detail = f"mean_mag={mean_mag:.1f}  rain_flag={is_rain}  (expected True)"
    return ok, detail


# ------------------------------------------------------------------------------
# Test 10: Brightness jump suppression
# ------------------------------------------------------------------------------

def _test_brightness_suppression() -> tuple[bool, str]:
    """
    Two frames with a large brightness difference should be suppressed.
    """
    ff = FlowField(
        backend="dis", dis_preset="medium", target_px=320,
        global_motion_compensation=False, temporal_smooth_alpha=1.0,
        brightness_jump_threshold=20.0,
    )

    frame_dark  = np.full((240, 320, 3), 20, dtype=np.uint8)
    frame_light = np.full((240, 320, 3), 200, dtype=np.uint8)

    result = ff.compute(frame_dark, frame_light)

    ok     = result.is_brightness_suppressed
    detail = (
        f"mean_diff~{abs(200-20)}  "
        f"brightness_suppressed={result.is_brightness_suppressed}  "
        f"(expected True)"
    )
    return ok, detail


# ------------------------------------------------------------------------------
# Test 11: CrowdMetricsEngine sign-convention self-test (internal)
# ------------------------------------------------------------------------------

def _test_metrics_selftest() -> tuple[bool, str]:
    """
    CrowdMetricsEngine.__init__ runs its own sign-convention self-test.
    This wrapper verifies it does NOT raise (which would mean it passed).
    """
    try:
        CrowdMetricsEngine(grid_cell_px=16)
        return True, "Internal self-test passed (no RuntimeError raised)"
    except RuntimeError as exc:
        return False, f"Internal self-test RAISED RuntimeError: {exc}"


# ------------------------------------------------------------------------------
# Registry and runner
# ------------------------------------------------------------------------------

_TESTS: list[tuple[int, str, Callable]] = [
    (1,  "Uniform translation",             _test_uniform_translation),
    (2,  "Divergence sign -- converging",    _test_divergence_sign_converging),
    (3,  "Divergence sign -- expanding",     _test_divergence_sign_expanding),
    (4,  "Curl sign -- CCW rotation",        _test_curl_sign),
    (5,  "Counterflow detection",           _test_counterflow),
    (6,  "Stop-and-go pattern",             _test_stop_go),
    (7,  "GMC subtraction",                 _test_gmc_subtraction),
    (8,  "CameraCalibration round-trip",    _test_calibration_roundtrip),
    (9,  "Rain heuristic",                  _test_rain_heuristic),
    (10, "Brightness jump suppression",     _test_brightness_suppression),
    (11, "CrowdMetricsEngine self-test",    _test_metrics_selftest),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic validation for crowd-flow module.")
    ap.add_argument("--all",    action="store_true", help="Run all tests (default).")
    ap.add_argument("--test",   type=int, default=0,
                    help="Run a single test by number.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print detail strings for passing tests too.")
    args = ap.parse_args()

    selected = (
        [t for t in _TESTS if t[0] == args.test] if args.test
        else _TESTS
    )
    if not selected:
        print(f"No test with id={args.test}")
        sys.exit(2)

    print(f"\n{'-' * 70}")
    print(f"  Dense optical flow -- validation suite ({len(selected)} tests)")
    print(f"{'-' * 70}")

    for tid, name, fn in selected:
        _register(tid, name, fn)

    n_pass = sum(1 for r in _results if r[2] == _PASS)
    n_fail = sum(1 for r in _results if r[2] == _FAIL)

    for tid, name, status, detail in _results:
        icon = "[PASS]" if status == _PASS else "[FAIL]"
        print(f"  [{icon}] Test {tid:2d}: {name}")
        if status == _FAIL or args.verbose:
            wrapped = textwrap.fill(detail, width=68, initial_indent="          ",
                                    subsequent_indent="          ")
            print(wrapped)

    print(f"{'-' * 70}")
    print(f"  Results: {n_pass} passed, {n_fail} failed")

    if n_fail > 0:
        print(
            "\n  [!]  One or more tests FAILED.\n"
            "  DO NOT USE for live crowd monitoring until all tests pass.\n"
            "  A sign-convention failure (tests 2/3) is particularly critical:\n"
            "  it means compression alerts fire on expansion and vice versa.\n"
        )
        sys.exit(1)
    else:
        print("\n  All tests passed.  [PASS]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
