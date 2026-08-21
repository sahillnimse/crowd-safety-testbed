"""
Background runner for the dense-flow validation routes.

Keeps the API thin: the endpoints in webapp/app.py start a run and poll for
the report, and everything heavyweight (torch, transformers) is imported
inside the worker thread so the page still loads instantly.

Only one validation run is allowed at a time.  Routes (a) and (b) are pure
CPU; route (c) runs a detector on the GPU and takes JobManager's GPU lock to
do it, so it queues behind a running model job rather than competing with one
for the card.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "validation")
REPORT_JSON = "flow_validation.json"
REPORT_VIDEO = "cross_family_comparison.mp4"

# Legacy single-slot location, still read so a report written before the
# per-video split is not orphaned.
REPORT_PATH = os.path.join(REPORT_DIR, REPORT_JSON)


def video_report_dir(source: str, create: bool = False) -> str:
    """
    outputs/validation/<video>/ — one directory per source.

    Validation measures a SPECIFIC video: routes (a) and (c) both run against
    that footage, and the verdict is about that footage.  Writing every run to
    one shared slot meant validating clip B silently destroyed clip A's
    report, and the UI could show a green result beside a video it was never
    computed from.  A directory per source removes both.

    This mirrors outputs/runs/<video>/<model>/ and is deliberately NOT the
    same directory: a run is the product, a validation is the measuring stick
    held against it.  Mixing them is what made a validation artifact get
    mistaken for the dense-flow output before.
    """
    stem = os.path.splitext(os.path.basename(source or "unknown"))[0]
    path = os.path.join(REPORT_DIR, stem)
    if create:
        os.makedirs(path, exist_ok=True)
    return path

# Route (c) runs a detector on every frame, so the budget is a direct cost.
# It is spent as DEFAULT_SEGMENTS bursts spread through the source rather
# than one run from frame 0: reading the first N frames validates the opening
# seconds and reports the verdict as if it covered the clip.
#
# max_frames <= 0 validates the WHOLE video in one continuous pass, so the
# comparison video spans the source exactly.  Cost is linear in frames:
# measured on Song.mp4 (3207 frames, 128 s) at roughly 1.1 s per validated
# frame on a free GPU —
#
#     max_frames   coverage   video     GPU time
#            240       7.5%     9.6s      ~4.5 min
#            600      18.7%    24.0s     ~11.2 min
#           1500      46.8%    60.0s     ~28 min
#           3207 (0)  100.0%   128.3s    ~60 min
#
# The budget is spread across the WHOLE source as a stride, so the default
# already covers the full clip and the comparison video spans its full
# duration.  Raising max_frames buys temporal resolution within that coverage,
# not more of the video.
DEFAULT_MAX_FRAMES = 240


class ValidationRunner:
    """Runs the three routes in a worker thread and holds the latest report."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status = "idle"        # idle | running | done | error
        self._message = ""
        self._report: Optional[dict] = None
        self._started_at: float = 0.0
        self._finished_at: float = 0.0
        self._load_persisted()

    # ------------------------------------------------------------------

    def _load_persisted(self) -> None:
        """Restore the most recent report from disk so it survives a restart."""
        candidates = []
        if os.path.isdir(REPORT_DIR):
            for name in os.listdir(REPORT_DIR):
                p = os.path.join(REPORT_DIR, name, REPORT_JSON)
                if os.path.isfile(p):
                    candidates.append(p)
        if os.path.isfile(REPORT_PATH):        # pre-split location
            candidates.append(REPORT_PATH)
        if not candidates:
            return
        newest = max(candidates, key=os.path.getmtime)
        try:
            import json
            with open(newest, encoding="utf-8") as f:
                self._report = json.load(f)
            self._status = "done"
            self._message = (f"Loaded the previous report for "
                             f"{self._report.get('source', 'unknown')}.")
            self._finished_at = os.path.getmtime(newest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", newest, exc)

    def state(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "message": self._message,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "report": self._report,
            }

    def clear(self) -> tuple[int, str]:
        """
        Delete the stored report and any files a run produced.

        Returns (n_removed, message); n_removed is -1 when the request was
        refused.  Deleting mid-run is refused rather than raced: the worker
        thread rewrites both the report and the comparison video when it
        finishes, so a delete that lands first is silently undone a minute
        later and looks like the button did nothing.
        """
        with self._lock:
            if self._status == "running":
                return -1, ("A validation run is in progress. Wait for it to "
                            "finish before deleting, or the run will just "
                            "write the files again.")

        # Reports now live in per-video subdirectories; loose files from
        # before that split are still cleaned so a wipe means a wipe.
        import shutil
        removed = 0
        if os.path.isdir(REPORT_DIR):
            for name in os.listdir(REPORT_DIR):
                path = os.path.join(REPORT_DIR, name)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", path, exc)

        with self._lock:
            self._report = None
            self._status = "idle"
            self._message = ""
            self._started_at = 0.0
            self._finished_at = 0.0

        return removed, f"Deleted {removed} validation file(s)."

    def start(self, source: str, routes: str = "abc",
              max_frames: int = DEFAULT_MAX_FRAMES) -> tuple[bool, str]:
        """Begin a run.  Returns (accepted, message)."""
        with self._lock:
            if self._status == "running":
                return False, "A validation run is already in progress."
            self._status = "running"
            self._message = f"Running routes {routes} on {os.path.basename(source)}…"
            self._started_at = time.time()
            self._finished_at = 0.0

        self._thread = threading.Thread(
            target=self._run, args=(source, routes, max_frames),
            daemon=True, name="flow-validation",
        )
        self._thread.start()
        return True, self._message

    # ------------------------------------------------------------------

    def _run(self, source: str, routes: str, max_frames: int) -> None:
        try:
            report = self._build_report(source, routes, max_frames)
            out_dir = video_report_dir(source, create=True)
            report.write_json(os.path.join(out_dir, REPORT_JSON))
            with self._lock:
                self._report = report.to_dict()
                self._status = "done"
                self._message = f"Validation complete: {report.status}."
                self._finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Validation run failed")
            with self._lock:
                self._status = "error"
                self._message = f"{exc.__class__.__name__}: {exc}"
                self._finished_at = time.time()

    def _set_message(self, message: str) -> None:
        """Update the status line the UI polls, without touching run state."""
        with self._lock:
            self._message = message

    def _build_report(self, source: str, routes: str, max_frames: int):
        import cv2
        import yaml

        from models.crowd_flow.flow_field import FlowField
        from models.crowd_flow.validation import (
            CrossCameraValidator, CrossFamilyValidator, SyntheticWarpValidator,
            ValidationReport,
        )
        from models.crowd_flow.validation.report import RouteResult

        cfg_path = os.path.join(PROJECT_ROOT, "configs", "crowd_flow.yaml")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = (yaml.safe_load(f) or {}).get("crowd_flow", {})

        # Validation has to measure the field the analyser actually produces.
        # Any setting that changes the flow and is not mirrored here means the
        # routes are grading a different field than the one that ships — which
        # is worse than not measuring at all, because it still reports a
        # number.  The far-field ROI is resolved per-camera exactly as
        # DenseFlowAnalyser.load() resolves it.
        cam_cfg = cfg.get("cameras", {}).get("default", {})

        def flow_factory():
            # Every keyword DenseFlowAnalyser.load() passes to FlowField is
            # mirrored here, with the same defaults.  The set used to be
            # partial: per-pixel validity, forward-backward consistency, the
            # structure-tensor settings, the reliability floor and the RAFT
            # variant were all left at FlowField's own defaults.  That silently
            # agreed with the shipped config while it kept those defaults and
            # would have diverged the moment anyone turned validity gating off
            # or raised max_flow_error_px — leaving the routes grading a field
            # the analyser does not produce, which still reports a number.
            return FlowField(
                backend=cfg.get("flow_backend", "dis"),
                dis_preset=cfg.get("dis_preset", "medium"),
                raft_variant=cfg.get("raft_variant", "large"),
                validity_gating=cfg.get("validity_gating", True),
                fb_consistency=cfg.get("fb_consistency", True),
                max_flow_error_px=cfg.get("max_flow_error_px", 1.0),
                structure_window=cfg.get("structure_window", 7),
                target_px=cfg.get("downsample_target_px", 320),
                temporal_smooth_alpha=cfg.get("temporal_smooth_alpha", 0.4),
                global_motion_compensation=cfg.get(
                    "global_motion_compensation", True),
                gmc_warn_threshold_px=cfg.get("gmc_warn_threshold_px", 3.0),
                gmc_max_correction_px=cfg.get("gmc_max_correction_px", 8.0),
                gmc_min_improvement=cfg.get("gmc_min_improvement", 0.9),
                rain_mag_threshold=cfg.get("rain_mag_threshold", 8.0),
                lowlight_gradient_threshold=cfg.get(
                    "lowlight_gradient_threshold", 12.0),
                brightness_jump_threshold=cfg.get(
                    "brightness_jump_threshold", 30.0),
                min_flow_reliability=cfg.get("min_flow_reliability", 0.35),
                far_field=cfg.get("far_field_refinement", True),
                far_field_roi=tuple(
                    cam_cfg.get("far_field_roi")
                    or cfg.get("far_field_roi", (0.0, 0.0, 1.0, 0.45))
                ),
                far_field_target_px=cfg.get("far_field_target_px", 960),
                far_field_blend_px=cfg.get("far_field_blend_px", 48),
            )

        report = ValidationReport(source=os.path.basename(source))

        # ---- route (a) ----
        if "a" in routes:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            cap.release()
            if not ok:
                from models.crowd_flow.validation.synthetic_warp import (
                    ROUTE_KEY, ROUTE_TITLE, ROUTE_CAVEAT)
                report.routes.append(RouteResult.skipped(
                    ROUTE_KEY, ROUTE_TITLE,
                    f"Could not read a frame from {os.path.basename(source)}.",
                    ROUTE_CAVEAT))
            else:
                report.routes.append(SyntheticWarpValidator().run(frame))

        # ---- route (b) ----
        if "b" in routes:
            from models.crowd_flow.validation.cross_camera import (
                ROUTE_KEY as B_KEY, ROUTE_TITLE as B_TITLE,
                ROUTE_CAVEAT as B_CAVEAT)
            report.routes.append(RouteResult.skipped(
                B_KEY, B_TITLE,
                "No calibrated camera pair is configured.  This route needs "
                "two cameras with homographies viewing overlapping ground; "
                "the uploaded footage is single-camera.  Run "
                "tests/validate_flow_routes.py --selftest-cross-camera to "
                "exercise the machinery on a synthetic pair.",
                B_CAVEAT))

        # ---- route (c) ----
        if "c" in routes:
            out_dir = video_report_dir(source, create=True)
            video_out = os.path.join(out_dir, REPORT_VIDEO)
            # Route (c) is the only route that runs a network, so it is the
            # only one that takes the card — and it takes it under
            # JobManager's own GPU lock rather than alongside it.
            #
            # This used to pin the detector to CPU instead.  That was the
            # right call when it ran one inference per frame; it stops being
            # affordable now that the detector is worth tiling (a 3x3 grid is
            # nine inferences per frame, and CPU turns a two-minute check into
            # a twenty-minute one).  Holding the lock costs a wait when a job
            # is already running, which is visible in the UI, instead of
            # costing an OOM that kills both.
            from pipeline.device import resolve_device
            from webapp.jobs import MANAGER

            device = resolve_device(None)
            with MANAGER.gpu_guard(
                on_wait=lambda: self._set_message(
                    "Queued — route (c) is waiting for the running model job "
                    "to release the GPU."
                )
            ):
                self._set_message(f"Running route (c) on {device}…")
                report.routes.append(
                    CrossFamilyValidator(device=device).run(
                        source, flow_factory=flow_factory,
                        max_frames=max_frames, annotate_path=video_out)
                )

        return report


RUNNER = ValidationRunner()
