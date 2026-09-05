"""
Deployment preflight: does this camera's configuration actually measure what
the operator will assume it measures?

Why this exists
---------------
Every gate below could be shut while the system ran, produced numbers, drew
overlays and reported "done". Nothing lied, but nothing said the most
important metrics were switched off either, and a dashboard full of green
numbers is read as "this location is safe" rather than "this location is
partially instrumented".

The specific ways that happened, all of which were live in the shipped
config:

1. Not one of the four Nashik cameras had a `homography` block, so the
   AlertEngine silently disabled `speed_low_ms`, `pressure_warning` and
   `pressure_critical` -- Helbing crowd pressure, the best-supported crush
   precursor available here, could never fire.

2. `density_enabled: false` is the default, so `crowd_pressure` is None on
   every frame. That is a SECOND independent gate on the same metric:
   calibrating the camera alone does not turn pressure back on.

3. `ram_kund_approach`'s zones carry a comment saying they are "illustrative
   placeholder values (640x480 frame assumed)", and
   `kushavarta_kund_approach`'s polygon is the entire 1920x1080 frame, which
   is not a zone at all.

What this module does NOT do
----------------------------
It cannot tell you whether a threshold is *correct*. Nothing can, without
annotated footage of real incidents from these cameras -- and there is none
in this repo (`test_videos.yaml` has four `ground_truth` keys, all empty).
Preflight checks that the instrument is switched on and pointed somewhere
plausible. Whether its readings are trustworthy is a separate question that
only measurement against ground truth can answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Severities, worst first.
BLOCKER = "BLOCKER"     # a headline safety metric cannot fire at all
WARNING = "WARNING"     # measurable, but the reading is suspect
NOTE = "NOTE"           # worth knowing, not dangerous

# Polygons shipped as examples. A deployment that still carries one of these
# is running on documentation, not on its own camera geometry.
_PLACEHOLDER_POLYGONS = {
    ((0, 240), (640, 240), (640, 480), (0, 480)),
    ((160, 360), (480, 360), (480, 480), (160, 480)),
}

# A zone covering essentially the whole frame is not a zone. Per-zone metrics
# exist to separate the ghat steps from the approach road; one polygon over
# everything averages them together and hides the local signal that matters.
_WHOLE_FRAME_COVERAGE = 0.98


@dataclass
class Finding:
    severity: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


#: A homography needs at least four point correspondences to define a plane
#: mapping. Fewer is not a degraded calibration, it is not a calibration.
_MIN_HOMOGRAPHY_POINTS = 4


def _is_really_calibrated(cam_cfg: dict) -> bool:
    """
    True only when the camera has a USABLE calibration, not merely a key.

    ``bool(cam_cfg.get("homography"))`` is not sufficient, and testing it that
    way produced the exact false assurance this module exists to prevent: the
    shipped `crowd_ralley` camera carries

        homography:
          image_points: []
          world_points_m: []

    which is a placeholder left behind for someone to fill in. The dict is
    truthy, so preflight reported that camera as FULLY INSTRUMENTED while
    CameraCalibration correctly treated it as uncalibrated and disabled every
    speed and pressure threshold on it. Preflight was contradicting the engine
    it is supposed to be reporting on.

    Accepts either calibration route: a homography with enough matched points,
    or a fitted bilinear perspective map (see
    models/crowd_flow/perspective.py), which needs no site measurement.
    """
    homography = cam_cfg.get("homography") or {}
    img_pts = homography.get("image_points") or []
    world_pts = homography.get("world_points_m") or []
    if (len(img_pts) >= _MIN_HOMOGRAPHY_POINTS
            and len(world_pts) >= _MIN_HOMOGRAPHY_POINTS
            and len(img_pts) == len(world_pts)):
        return True

    # A perspective map is a valid alternative: it yields metres-per-pixel
    # without a ground survey, which is the practical route for a camera
    # nobody can stand in front of with a tape measure.
    persp = cam_cfg.get("perspective_map") or cam_cfg.get("perspective") or {}
    if persp and any(persp.get(k) for k in ("ah", "bh", "ch")):
        return True

    return False


def check_camera(camera_id: str, cam_cfg: dict, cfg: dict,
                 frame_wh: Optional[tuple] = None) -> list:
    """
    Inspect one camera block. Returns findings, worst first.

    ``frame_wh`` (width, height) enables the whole-frame-zone check; omit it
    before the first frame is seen and that check is skipped rather than
    guessed at.
    """
    findings: list[Finding] = []
    zones = cam_cfg.get("zones") or []

    calibrated = _is_really_calibrated(cam_cfg)
    density_on = bool(cfg.get("density_enabled", False))

    # --- 1. Calibration -------------------------------------------------
    if not calibrated:
        findings.append(Finding(
            BLOCKER, "NO_HOMOGRAPHY",
            f"Camera '{camera_id}' has no homography. Speed is px/frame (not "
            f"m/s) and density is not persons/m2, so speed_low_ms, "
            f"pressure_warning and pressure_critical are DISABLED. Crowd "
            f"pressure is the best-supported crush precursor here and it "
            f"cannot fire. Run scripts/calibrate_ground_plane.py."))

    # --- 2. Density gate on pressure ------------------------------------
    wants_pressure = any(
        (z.get("thresholds") or {}).get(k) is not None
        for z in zones for k in ("pressure_warning", "pressure_critical"))
    if wants_pressure and not density_on:
        findings.append(Finding(
            BLOCKER, "PRESSURE_WITHOUT_DENSITY",
            f"Camera '{camera_id}' configures pressure thresholds, but "
            f"density_enabled is false so crowd_pressure is never computed "
            f"and those alerts can NEVER fire. This is a second gate, "
            f"independent of calibration: fixing the homography alone will "
            f"not turn them on. Set density_enabled: true."))
    elif not density_on:
        findings.append(Finding(
            WARNING, "DENSITY_OFF",
            f"density_enabled is false, so density and crowd pressure are "
            f"not measured for '{camera_id}'. Flow metrics (divergence, "
            f"counterflow, turbulence) still work."))

    # --- 3. Zones exist -------------------------------------------------
    if not zones:
        findings.append(Finding(
            BLOCKER, "NO_ZONES",
            f"Camera '{camera_id}' has no zones. No per-zone metric and no "
            f"alert of any kind can be produced for it."))
        return _sorted(findings)

    # --- 4. Placeholder and whole-frame zones ---------------------------
    for z in zones:
        name = z.get("name", "?")
        poly = z.get("polygon") or []
        key = tuple(tuple(p) for p in poly)
        if key in _PLACEHOLDER_POLYGONS:
            findings.append(Finding(
                BLOCKER, "PLACEHOLDER_ZONE",
                f"Zone '{name}' on '{camera_id}' is still the shipped "
                f"EXAMPLE polygon {poly} (authored for a 640x480 frame). It "
                f"does not describe this camera's view, so every metric for "
                f"it is computed over the wrong region."))
            continue
        if frame_wh and _covers_whole_frame(poly, *frame_wh):
            findings.append(Finding(
                WARNING, "WHOLE_FRAME_ZONE",
                f"Zone '{name}' on '{camera_id}' covers the entire frame. "
                f"Per-zone metrics exist to separate distinct areas; one "
                f"zone over everything averages the crowded part with the "
                f"empty part and hides the local signal."))

    # --- 5. Thresholds present ------------------------------------------
    for z in zones:
        if not (z.get("thresholds") or {}):
            findings.append(Finding(
                WARNING, "NO_THRESHOLDS",
                f"Zone '{z.get('name','?')}' on '{camera_id}' has no "
                f"thresholds, so it is measured but can never alert."))

    return _sorted(findings)


def _covers_whole_frame(polygon: list, width: int, height: int) -> bool:
    if len(polygon) < 3 or width <= 0 or height <= 0:
        return False
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    # Fractional polygons are in 0..1; scale before comparing.
    if max(xs) <= 1.0 and max(ys) <= 1.0:
        xs = [x * width for x in xs]
        ys = [y * height for y in ys]
    span = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return span >= _WHOLE_FRAME_COVERAGE * width * height


def _sorted(findings: list) -> list:
    order = {BLOCKER: 0, WARNING: 1, NOTE: 2}
    return sorted(findings, key=lambda f: order.get(f.severity, 9))


def report(camera_id: str, findings: list) -> dict:
    """
    Log findings at a level matching their severity, and return a summary
    dict for summary.json.

    BLOCKERs are logged as errors and repeated in a banner: the WARNING that
    scrolled past at startup is long gone by the time anyone reads the
    numbers, and these particular facts change what the numbers MEAN.
    """
    blockers = [f for f in findings if f.severity == BLOCKER]
    warnings = [f for f in findings if f.severity == WARNING]

    for f in findings:
        (logger.error if f.severity == BLOCKER else logger.warning)("%s", f)

    if blockers:
        logger.error(
            "\n"
            "=================================================================\n"
            " PREFLIGHT: %d BLOCKER(S) on camera '%s'\n"
            " One or more headline safety metrics CANNOT FIRE on this camera.\n"
            " Readings that do appear are still real, but this camera is NOT\n"
            " fully instrumented and must not be treated as full coverage.\n"
            "=================================================================",
            len(blockers), camera_id)

    return {
        "camera_id": camera_id,
        "blockers": [f.code for f in blockers],
        "warnings": [f.code for f in warnings],
        "findings": [str(f) for f in findings],
        "fully_instrumented": not blockers,
    }
