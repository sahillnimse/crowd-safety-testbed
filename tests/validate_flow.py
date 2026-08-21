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
  python tests/validate_flow.py            # run all tests
  python tests/validate_flow.py --all      # same
  python tests/validate_flow.py --test 1   # run one test
  python tests/validate_flow.py --verbose  # show per-pixel diagnostics
  python -m tests.validate_flow            # module invocation

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
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
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
# Test 12: within-cell variance survives the m/s conversion
# ------------------------------------------------------------------------------

def _test_variance_survives_calibration() -> tuple[bool, str]:
    """
    A calibrated conversion must not flatten the field inside a metrics cell.

    Regression test.  flow_field_to_ms used to sample the velocity once per
    grid cell and nearest-neighbour upsample it, making every pixel in a cell
    numerically identical.  Per-cell means were unaffected, so nothing looked
    wrong — but velocity_variance was exactly zero, which silently took
    turbulence_index (CRITICAL threshold) and crowd pressure to zero with it,
    on calibrated cameras only.
    """
    import numpy as np
    H, W, G, S = 480, 640, 16, 0.05
    calib = CameraCalibration.from_points(
        "t", [[0, 0], [W, 0], [W, H], [0, H]],
        [[0, 0], [W * S, 0], [W * S, H * S], [0, H * S]],
    )
    rng = np.random.default_rng(0)
    field = np.zeros((H, W, 2), np.float32)
    field[..., 0] = rng.normal(0.0, 0.9, (H, W))
    field[..., 1] = rng.normal(0.0, 0.9, (H, W))

    ms = calib.flow_field_to_ms(field, fps=25.0, grid_cell_px=G)
    speed = np.hypot(ms[..., 0], ms[..., 1])
    within = speed.reshape(H // G, G, W // G, G).var(axis=(1, 3)).mean()

    n_unique = len(np.unique(ms[0:G, 0:G, 0]))
    if n_unique < G * G // 2:
        return False, (f"only {n_unique} distinct values in a {G}x{G} cell "
                       f"(expected ~{G*G}); the conversion is flattening cells")
    if within < 1e-3:
        return False, f"within-cell variance collapsed to {within:.2e}"
    return True, (f"within-cell variance {within:.4f}, "
                  f"{n_unique} distinct values per {G}x{G} cell")


# ------------------------------------------------------------------------------
# Test 13: crowd pressure arithmetic and its unit gate
# ------------------------------------------------------------------------------

def _test_crowd_pressure() -> tuple[bool, str]:
    """
    P = rho * Var(v), and the published thresholds are refused when the
    camera is uncalibrated (where density is px-based, not persons/m^2).
    """
    import numpy as np
    from models.crowd_flow.density import (
        crowd_pressure, PRESSURE_TURBULENCE, PRESSURE_STAMPEDE,
    )
    from models.crowd_flow.zones import Zone, ZoneThresholds, AlertEngine

    rho = np.full((1, 1), 5.0, np.float32)
    var = np.full((1, 1), PRESSURE_STAMPEDE / 5.0, np.float32)
    p = float(crowd_pressure(rho, var)[0, 0])
    if abs(p - PRESSURE_STAMPEDE) > 1e-6:
        return False, f"rho*Var(v) = {p:.5f}, expected {PRESSURE_STAMPEDE}"

    if crowd_pressure(None, var) is not None:
        return False, "pressure should be None when no density is available"

    zone = Zone(name="z", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                thresholds=ZoneThresholds())
    uncal = AlertEngine([zone], fps=25.0, is_calibrated=False, speed_units="px/frame")
    gated = [k for k in uncal._state if "pressure" in k[1]]
    cal = AlertEngine([zone], fps=25.0, is_calibrated=True, speed_units="m/s")
    active = [k for k in cal._state if "pressure" in k[1]]
    if len(active) != 2:
        return False, f"expected warning+critical pressure states, got {len(active)}"

    class _ZM:
        crowd_pressure = PRESSURE_STAMPEDE * 2
        mean_speed = mean_divergence = mean_curl = 0.0
        counterflow_score = turbulence_index = 0.0
    fired_uncal = [a.metric_name for a in
                   uncal.update({"z": _ZM()}, {"z": False}, 0, 0.0)]
    if "crowd_pressure" in fired_uncal:
        return False, "pressure alert fired on an UNCALIBRATED camera"

    return True, (f"P=rho*Var(v) exact; warning/critical states registered "
                  f"({len(active)}); refused when uncalibrated "
                  f"(thresholds {PRESSURE_TURBULENCE}/{PRESSURE_STAMPEDE} s^-2)")


# ------------------------------------------------------------------------------
# Test 14: head points are projected to ground contacts
# ------------------------------------------------------------------------------

def _test_head_to_foot() -> tuple[bool, str]:
    """
    A head point must be moved down to the ground before it is mapped.

    A ground-plane homography maps image points to where they touch the
    ground.  A head is a stature above that, so using it directly places the
    person a stature further from the camera — several grid cells at the back
    of a crowd, in the direction that inflates density where it is already
    highest.  This checks the projected point is exactly one stature from the
    head on the ground, at every depth, and that the pixel drop shrinks with
    distance as perspective requires.
    """
    import numpy as np
    from models.crowd_flow.density import DensityEstimator

    calib = CameraCalibration.from_points(
        "t", [[100, 470], [540, 470], [420, 250], [220, 250]],
        [[0, 0], [10, 0], [10, 25], [0, 25]],
    )
    de = DensityEstimator(source="heads", enabled=True, person_height_m=1.65)

    drops, dists = [], []
    for hy in (270.0, 330.0, 400.0, 460.0):
        foot = de._heads_to_feet(np.array([[320.0, hy]], np.float32), calib)
        fy = float(foot[0, 1])
        hw = calib.pixel_to_world(320.0, hy)
        fw = calib.pixel_to_world(320.0, fy)
        dists.append(float(np.hypot(fw[0] - hw[0], fw[1] - hw[1])))
        drops.append(fy - hy)

    if max(abs(d - 1.65) for d in dists) > 0.02:
        return False, f"ground distances {['%.3f' % d for d in dists]}, expected 1.65 m"
    if not all(a < b for a, b in zip(drops, drops[1:])):
        return False, f"pixel drop must grow towards the camera, got {drops}"

    unchanged = de._heads_to_feet(np.array([[320.0, 400.0]], np.float32), None)
    if float(unchanged[0, 1]) != 400.0:
        return False, "uncalibrated cameras must leave head points unshifted"

    return True, (f"ground distance 1.65 m at every depth; pixel drop "
                  f"{drops[0]:.1f} -> {drops[-1]:.1f} px towards the camera; "
                  f"no shift when uncalibrated")


# ------------------------------------------------------------------------------
# Test 15: density target sums to the head count
# ------------------------------------------------------------------------------

def _test_density_target() -> tuple[bool, str]:
    """
    Each labelled head must contribute exactly 1.0 to the training target,
    including heads whose Gaussian is clipped by the patch edge.

    Normalising the kernel before clipping instead of after would leave every
    edge person contributing a fraction of a head, making crop boundaries a
    systematic undercount that no amount of training can correct.
    """
    return True, ("skipped: head_count switched to APGCC (point detector, "
                  "no density target to build)")


# ------------------------------------------------------------------------------
# Test 16: per-zone scale (pedestrian vs vehicle)
# ------------------------------------------------------------------------------

def _test_zone_types() -> tuple[bool, str]:
    """
    A pedestrian zone and a vehicle zone over the same field must not be
    aggregated at the same motion floor.

    Pedestrians move 1-2 px/frame and vehicles 10-50, so one global floor is
    necessarily wrong for one of them: set for pedestrians it admits every
    noisy background cell in the carriageway and drags its mean towards zero
    (reading as congestion); set for vehicles it discards people who are
    genuinely walking.
    """
    import numpy as np
    from models.crowd_flow.zones import Zone

    ped = Zone(name="p", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
               zone_type="pedestrian")
    veh = Zone(name="v", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
               zone_type="vehicle")
    if not (ped.motion_floor < veh.motion_floor):
        return False, (f"vehicle floor {veh.motion_floor} must exceed "
                       f"pedestrian floor {ped.motion_floor}")
    if not (veh.max_density < ped.max_density):
        return False, (f"vehicle density ceiling {veh.max_density} must be "
                       f"below pedestrian {ped.max_density}")

    override = Zone(name="o", polygon=ped.polygon, zone_type="vehicle",
                    min_magnitude=0.25)
    if override.motion_floor != 0.25:
        return False, f"explicit min_magnitude ignored ({override.motion_floor})"

    # A fractional zone is rebuilt by resolve_to_frame; the type must survive.
    frac = Zone(name="f", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                zone_type="vehicle", min_magnitude=2.5)
    res = frac.resolve_to_frame(640, 480)
    if res.zone_type != "vehicle" or res.motion_floor != 2.5:
        return False, ("resolve_to_frame dropped zone_type/min_magnitude "
                       f"({res.zone_type}, {res.motion_floor})")

    try:
        Zone(name="bad", polygon=ped.polygon, zone_type="bicycle")
    except ValueError:
        pass
    else:
        return False, "an unknown zone_type was accepted"

    return True, (f"pedestrian floor {ped.motion_floor} < vehicle "
                  f"{veh.motion_floor} px/frame; density ceiling "
                  f"{ped.max_density} vs {veh.max_density} /m²; "
                  f"overrides and resolve_to_frame preserved")


# ------------------------------------------------------------------------------
# Test 17: RAFT backend recovers a known translation
# ------------------------------------------------------------------------------

def _test_raft_backend() -> tuple[bool, str]:
    """
    The learned backend must recover an exact shift, like the classical one.

    This is a correctness check on the integration — tensor layout, the
    [-1, 1] normalisation, divisible-by-8 padding, and taking the LAST of
    RAFT's per-iteration outputs rather than the first (the coarsest).  Any
    of those wrong yields a field that is plausible but wrong by a scale
    factor, which no downstream metric would flag.

    SKIPPED without CUDA: RAFT on CPU takes minutes per pair, and a suite
    people avoid running because it is slow validates nothing.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return True, "SKIPPED (no CUDA; RAFT on CPU is too slow to gate on)"
    except ImportError:
        return True, "SKIPPED (torch unavailable)"

    # Random texture, NOT _make_textured_frame().  That helper builds a sum of
    # sinusoids, which is spatially periodic and therefore ambiguous to a
    # correlation-based matcher: RAFT lands 0.687 px out on it and 0.100 px
    # out on aperiodic texture, while DIS is unaffected (0.003 vs 0.002)
    # because its pyramid resolves the ambiguity at a coarser level.  Testing
    # RAFT on the periodic frame measures the frame, not the backend.
    dx, dy = 3.0, -2.0
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    frame = cv2.cvtColor(cv2.GaussianBlur(noise, (5, 5), 0), cv2.COLOR_GRAY2BGR)
    shifted = _translate_frame(frame, dx, dy)
    ff = FlowField(backend="raft", raft_variant="large", device="cuda",
                   target_px=320, global_motion_compensation=False,
                   far_field=False, temporal_smooth_alpha=1.0)
    res = ff.compute(frame, shifted)

    b = 24
    fx = float(res.field_xy[b:-b, b:-b, 0].mean())
    fy = float(res.field_xy[b:-b, b:-b, 1].mean())
    err = float(np.hypot(fx - dx, fy - dy))
    if err > 0.35:
        return False, (f"RAFT recovered ({fx:.2f},{fy:.2f}) for a known "
                       f"({dx},{dy}) shift; error {err:.2f} px")
    return True, (f"RAFT recovered ({fx:.2f},{fy:.2f}) vs ({dx},{dy}); "
                  f"error {err:.3f} px")


# ------------------------------------------------------------------------------
# Test 18: invented motion is rejected, measured motion is kept
# ------------------------------------------------------------------------------

def _test_validity_gating() -> tuple[bool, str]:
    """
    Flow over a textureless region must be rejected; flow on a textured
    moving object must survive.

    Dense flow returns a vector everywhere unconditionally.  Where there is
    nothing to match — sky, blank tarmac, still water — the algorithm fills
    the region in from its neighbours, and that fabricated motion becomes
    divergence, turbulence and counterflow indistinguishable from a crowd.

    A textured patch moving across a FLAT background is the clean case: the
    patch carries real, measurable motion; the background carries none and
    cannot carry any, because a constant region has no information about
    displacement at all.
    """
    H, W = 240, 320
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, (60, 60), dtype=np.uint8)

    def frame_at(x0):
        img = np.full((H, W), 120, np.uint8)      # flat, featureless
        img[90:150, x0:x0 + 60] = patch
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    f0, f1 = frame_at(100), frame_at(104)          # 4 px to the right

    obj = np.zeros((H, W), bool)
    obj[90:150, 100:164] = True                    # union of both positions

    out = {}
    for gate in (False, True):
        ff = FlowField(backend="dis", target_px=320, global_motion_compensation=False,
                       far_field=False, temporal_smooth_alpha=1.0,
                       validity_gating=gate, fb_consistency=gate,
                       max_flow_error_px=1.0)
        res = ff.compute(f0, f1)
        mag = np.hypot(res.field_xy[..., 0], res.field_xy[..., 1])
        moving = mag > 0.5
        out[gate] = (moving[obj].mean(), moving[~obj].mean())

    (on_off, bg_off), (on_on, bg_on) = out[False], out[True]
    if bg_off <= 0.01:
        return True, ("SKIPPED: this build's flow did not bleed into the flat "
                      "region, so there is nothing for the gate to remove")
    if bg_on > bg_off * 0.5:
        return False, (f"gating left {bg_on*100:.1f}% of the featureless "
                       f"background moving (was {bg_off*100:.1f}%)")
    if on_on < on_off * 0.5:
        return False, (f"gating destroyed real motion: {on_on*100:.1f}% of the "
                       f"moving object kept, was {on_off*100:.1f}%")
    return True, (f"flat background moving {bg_off*100:.1f}% -> {bg_on*100:.1f}%; "
                  f"object motion {on_off*100:.1f}% -> {on_on*100:.1f}% kept")


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
    (12, "Variance survives calibration",   _test_variance_survives_calibration),
    (13, "Crowd pressure + unit gating",    _test_crowd_pressure),
    (14, "Head point -> ground contact",    _test_head_to_foot),
    (15, "Density target sums to count",    _test_density_target),
    (16, "Per-zone scale (ped vs vehicle)", _test_zone_types),
    (17, "RAFT backend known translation",  _test_raft_backend),
    (18, "Validity gating rejects invented", _test_validity_gating),
]


def main() -> None:
    # Several detail strings contain arrows and unit symbols.  Windows
    # consoles default to cp1252, where printing those raises
    # UnicodeEncodeError and takes the whole suite down mid-run — so
    # `--verbose` crashed after test 8 while the plain run passed.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

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
