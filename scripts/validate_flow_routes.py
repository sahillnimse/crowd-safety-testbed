"""
Run the three annotation-free validation routes against the dense flow module.

This is the empirical counterpart to scripts/validate_flow.py.  That suite
checks sign conventions and internal contracts on synthetic data; this one
measures how wrong the estimator is on real imagery, and reports what each
measurement cannot tell you.

Usage
-----
  # All available routes against a video:
  python scripts/validate_flow_routes.py --source test_videos/Umbrellas.mp4

  # Route (a) only, plus a configuration sweep:
  python scripts/validate_flow_routes.py --source test_videos/Umbrellas.mp4 \\
      --routes a --sweep

  # Route (b) machinery self-test (no camera pair needed):
  python scripts/validate_flow_routes.py --routes b --selftest-cross-camera

  # Write the report where the web UI will pick it up:
  python scripts/validate_flow_routes.py --source test_videos/Umbrellas.mp4 \\
      --output outputs/validation/flow_validation.json

Route (b) needs two calibrated cameras viewing overlapping ground.  With no
such pair configured it reports "skipped", never "pass" — an unrun route must
not read as a satisfied one.  --selftest-cross-camera exercises the route's
machinery end to end by synthesising two virtual cameras over a known ground
motion, which validates the code without validating any real deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from models.crowd_flow.flow_field import FlowField
from models.crowd_flow.ground_plane import CameraCalibration
from models.crowd_flow.validation import (
    CameraView, CrossCameraValidator, CrossFamilyValidator,
    SyntheticWarpValidator, ValidationReport,
    STATUS_PASS, STATUS_FAIL, STATUS_SKIPPED, STATUS_ERROR,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("validate_flow_routes")

DEFAULT_OUTPUT = os.path.join(
    str(_PROJECT_ROOT), "outputs", "validation", "flow_validation.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_frame(video_path: str, index: int = 30) -> np.ndarray:
    """Read one frame from a video, for the synthetic-warp route."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path!r}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path!r}")
    return frame


def _load_cfg(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return (yaml.safe_load(f) or {}).get("crowd_flow", {})


def _flow_factory_from_cfg(cfg: dict, **overrides):
    """FlowField factory using the project config, with optional overrides."""
    kw = dict(
        backend=cfg.get("flow_backend", "dis"),
        dis_preset=cfg.get("dis_preset", "medium"),
        target_px=cfg.get("downsample_target_px", 320),
        temporal_smooth_alpha=cfg.get("temporal_smooth_alpha", 0.4),
        global_motion_compensation=cfg.get("global_motion_compensation", True),
        gmc_max_correction_px=cfg.get("gmc_max_correction_px", 8.0),
    )
    kw.update(overrides)
    return lambda: FlowField(**kw)


# ---------------------------------------------------------------------------
# Route (b) machinery self-test
# ---------------------------------------------------------------------------

def _synthetic_camera_pair(frame: np.ndarray, shift_m: tuple[float, float],
                           fps: float = 30.0):
    """
    Build two virtual cameras viewing the same ground plane, both seeing a
    known uniform ground motion.

    A ground texture is rendered into two different perspective views.  The
    second time-step is the same texture translated by a known distance in
    METRES on the ground, so the true ground-plane velocity is known exactly
    and is identical for both cameras — while the pixel motion each camera
    sees differs, because their viewing geometry differs.

    This exercises the whole chain (flow → homography → ground projection →
    comparison) without needing real camera pairs.  It validates the code, not
    any deployment.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    texture = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Ground extent covered by the texture, in metres.
    ground_w, ground_h = 20.0, 12.0
    world_corners = np.array(
        [[0.0, 0.0], [ground_w, 0.0], [ground_w, ground_h], [0.0, ground_h]],
        dtype=np.float64,
    )

    # Two camera image-plane quadrilaterals over that ground patch: different
    # perspective distortions stand in for different mounting angles.
    view_a = np.array([[120, 90], [w - 100, 40],
                       [w - 40, h - 60], [60, h - 30]], dtype=np.float64)
    view_b = np.array([[40, 60], [w - 150, 100],
                       [w - 90, h - 40], [130, h - 80]], dtype=np.float64)

    def build(view_quad):
        cal = CameraCalibration.from_points(
            "synthetic", view_quad.tolist(), world_corners.tolist()
        )
        # world → image, to render the ground texture into this camera
        H_inv = np.linalg.inv(cal.H)
        # texture pixel → world
        tex_to_world = cv2.getPerspectiveTransform(
            np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32),
            world_corners.astype(np.float32),
        )
        return cal, H_inv @ tex_to_world

    cal_a, M_a = build(view_a)
    cal_b, M_b = build(view_b)

    # Second time-step: shift the texture by shift_m on the GROUND.
    shift_world = np.array(
        [[1.0, 0.0, shift_m[0]], [0.0, 1.0, shift_m[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tex_to_world = cv2.getPerspectiveTransform(
        np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32),
        world_corners.astype(np.float32),
    )

    views = []
    for cal in (cal_a, cal_b):
        H_inv = np.linalg.inv(cal.H)
        M0 = H_inv @ tex_to_world
        M1 = H_inv @ shift_world @ tex_to_world
        f0 = cv2.warpPerspective(texture, M0, (w, h), flags=cv2.INTER_LINEAR)
        f1 = cv2.warpPerspective(texture, M1, (w, h), flags=cv2.INTER_LINEAR)
        ff = FlowField(global_motion_compensation=False,
                       temporal_smooth_alpha=1.0, target_px=480)
        res = ff.compute(f0, f1, None, 0.0)
        views.append(CameraView(calibration=cal, field_xy=res.field_xy, fps=fps))

    true_speed = float(np.hypot(*shift_m) * fps)
    return views[0], views[1], true_speed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_ICON = {
    STATUS_PASS: "[PASS]", STATUS_FAIL: "[FAIL]",
    STATUS_SKIPPED: "[SKIP]", STATUS_ERROR: "[ERR ]",
}


def _print_report(report: ValidationReport, verbose: bool) -> None:
    print(f"\n{'=' * 74}")
    print("  Dense optical flow — annotation-free validation")
    if report.source:
        print(f"  Source: {report.source}")
    print(f"{'=' * 74}")

    for r in report.routes:
        print(f"\n  {_ICON.get(r.status, '[?]')}  {r.title}")
        print(textwrap.fill(r.summary, width=70,
                            initial_indent="        ", subsequent_indent="        "))
        for m in r.measurements:
            mark = "" if m.passed is None else ("  ok" if m.passed else "  <-- OVER")
            tol = "" if m.tolerance is None else (
                f" (limit {'>=' if m.higher_is_better else '<='} {m.tolerance:g})"
            )
            print(f"          {m.label:38s} {m.value:9.3f} {m.units}{tol}{mark}")
            if m.note and verbose:
                print(f"            note: {m.note}")
        if r.caveat:
            print(textwrap.fill(
                "CANNOT TELL YOU: " + r.caveat, width=70,
                initial_indent="        ! ", subsequent_indent="          "))

    print(f"\n{'-' * 74}")
    print(f"  Overall: {report.status.upper()}")
    n_skip = sum(1 for r in report.routes if r.status == STATUS_SKIPPED)
    if n_skip:
        print(f"  {n_skip} route(s) skipped — the picture is incomplete, not clean.")
    print(
        "  Passing is necessary, not sufficient: these routes measure the\n"
        "  velocity field, not whether its derived metrics predict crush risk."
    )
    print(f"{'-' * 74}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotation-free validation routes for dense optical flow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--source", default="",
                    help="Video for routes (a) and (c).")
    ap.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "crowd_flow.yaml"))
    ap.add_argument("--routes", default="abc",
                    help="Which routes to run; any subset of 'abc'.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Where to write the JSON report.")
    ap.add_argument("--frame-index", type=int, default=30,
                    help="Frame used for the synthetic warp route.")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="Frames processed by route (c).")
    ap.add_argument("--sweep", action="store_true",
                    help="Route (a): sweep configurations and report the best.")
    ap.add_argument("--selftest-cross-camera", action="store_true",
                    help="Route (b): run the synthetic two-camera self-test.")
    ap.add_argument("--comparison-video", action="store_true",
                    help="Route (c): write a video showing both instruments' "
                         "arrows on each tracked person.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    report = ValidationReport(source=args.source or "(no video)")
    routes = args.routes.lower()

    # ---- Route (a) -------------------------------------------------------
    if "a" in routes:
        if not args.source:
            from models.crowd_flow.validation.synthetic_warp import (
                ROUTE_KEY, ROUTE_TITLE, ROUTE_CAVEAT,
            )
            from models.crowd_flow.validation.report import RouteResult
            report.routes.append(RouteResult.skipped(
                ROUTE_KEY, ROUTE_TITLE,
                "No --source given; this route needs a real frame to warp.",
                ROUTE_CAVEAT,
            ))
        else:
            frame = _pick_frame(args.source, args.frame_index)
            validator = SyntheticWarpValidator()
            result = validator.run(frame)

            if args.sweep:
                variants = {
                    "target_px=320, dis medium":
                        _flow_factory_from_cfg(cfg, target_px=320,
                                               global_motion_compensation=False,
                                               temporal_smooth_alpha=1.0),
                    "target_px=480, dis medium":
                        _flow_factory_from_cfg(cfg, target_px=480,
                                               global_motion_compensation=False,
                                               temporal_smooth_alpha=1.0),
                    "target_px=640, dis medium":
                        _flow_factory_from_cfg(cfg, target_px=640,
                                               global_motion_compensation=False,
                                               temporal_smooth_alpha=1.0),
                    "target_px=320, dis fast":
                        _flow_factory_from_cfg(cfg, target_px=320,
                                               dis_preset="fast",
                                               global_motion_compensation=False,
                                               temporal_smooth_alpha=1.0),
                    "target_px=320, farneback":
                        _flow_factory_from_cfg(cfg, target_px=320,
                                               backend="farneback",
                                               global_motion_compensation=False,
                                               temporal_smooth_alpha=1.0),
                }
                rows = validator.sweep(frame, variants)
                result.detail["sweep"] = rows
                print("\n  Configuration sweep (best first, mean endpoint error):")
                for row in rows:
                    print(f"    {row['epe_mean']:7.3f} px  {row['variant']}")
                print()

            report.routes.append(result)

    # ---- Route (b) -------------------------------------------------------
    if "b" in routes:
        from models.crowd_flow.validation.cross_camera import (
            ROUTE_KEY as B_KEY, ROUTE_TITLE as B_TITLE, ROUTE_CAVEAT as B_CAVEAT,
        )
        from models.crowd_flow.validation.report import RouteResult

        if args.selftest_cross_camera:
            if not args.source:
                report.routes.append(RouteResult.skipped(
                    B_KEY, B_TITLE,
                    "The self-test needs --source for its ground texture.",
                    B_CAVEAT,
                ))
            else:
                frame = _pick_frame(args.source, args.frame_index)
                # 0.05 m/frame at 30 fps = 1.5 m/s, a brisk walk.
                view_a, view_b, true_speed = _synthetic_camera_pair(
                    frame, shift_m=(0.05, 0.0)
                )
                result = CrossCameraValidator(
                    grid_spacing_m=0.5, max_disagreement_ms=0.35
                ).run(view_a, view_b)
                result.detail["selftest"] = True
                result.detail["true_ground_speed_ms"] = true_speed
                result.summary += (
                    f"  [SELF-TEST on synthetic camera pair; true ground speed "
                    f"{true_speed:.2f} m/s.  Validates the code path, not any "
                    f"real deployment.]"
                )
                report.routes.append(result)
        else:
            report.routes.append(RouteResult.skipped(
                B_KEY, B_TITLE,
                "No calibrated camera pair configured.  This route needs two "
                "cameras with homographies viewing overlapping ground; none of "
                "the current test footage is multi-camera.  Run with "
                "--selftest-cross-camera to exercise the machinery on a "
                "synthetic pair.",
                B_CAVEAT,
            ))

    # ---- Route (c) -------------------------------------------------------
    if "c" in routes:
        from models.crowd_flow.validation.cross_family import (
            ROUTE_KEY as C_KEY, ROUTE_TITLE as C_TITLE, ROUTE_CAVEAT as C_CAVEAT,
        )
        from models.crowd_flow.validation.report import RouteResult

        if not args.source:
            report.routes.append(RouteResult.skipped(
                C_KEY, C_TITLE, "No --source given.", C_CAVEAT,
            ))
        else:
            validator = CrossFamilyValidator()
            video_out = None
            if args.comparison_video:
                video_out = os.path.join(
                    os.path.dirname(os.path.abspath(args.output)),
                    "cross_family_comparison.mp4",
                )
                os.makedirs(os.path.dirname(video_out), exist_ok=True)
            report.routes.append(validator.run(
                args.source,
                flow_factory=_flow_factory_from_cfg(cfg),
                max_frames=args.max_frames,
                annotate_path=video_out,
            ))

    _print_report(report, args.verbose)
    path = report.write_json(args.output)
    print(f"  Report written to {path}\n")

    sys.exit(1 if report.status in (STATUS_FAIL, STATUS_ERROR) else 0)


if __name__ == "__main__":
    main()
