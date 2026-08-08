"""
Background runner for the dense-flow validation routes.

Keeps the API thin: the endpoints in webapp/app.py start a run and poll for
the report, and everything heavyweight (torch, ultralytics) is imported
inside the worker thread so the page still loads instantly.

Only one validation run is allowed at a time.  The routes are CPU-bound and
route (c) loads a detector; letting the UI queue several would starve any
model job running alongside them.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "validation")
REPORT_PATH = os.path.join(REPORT_DIR, "flow_validation.json")

# Route (c) runs a detector per frame, so the default is kept modest; the
# statistic stabilises well before the frame budget is exhausted.
DEFAULT_MAX_FRAMES = 120


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
        """Restore the last report from disk so it survives a restart."""
        if not os.path.exists(REPORT_PATH):
            return
        try:
            import json
            with open(REPORT_PATH, encoding="utf-8") as f:
                self._report = json.load(f)
            self._status = "done"
            self._message = "Loaded the previous report from disk."
            self._finished_at = os.path.getmtime(REPORT_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", REPORT_PATH, exc)

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

        removed = 0
        if os.path.isdir(REPORT_DIR):
            for name in os.listdir(REPORT_DIR):
                path = os.path.join(REPORT_DIR, name)
                if not os.path.isfile(path):
                    continue
                try:
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
            os.makedirs(REPORT_DIR, exist_ok=True)
            report.write_json(REPORT_PATH)
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

    @staticmethod
    def _build_report(source: str, routes: str, max_frames: int):
        import cv2
        import yaml

        from models.crowd_flow.flow_field import FlowField
        from models.crowd_flow.validation import (
            CrossCameraValidator, CrossFamilyValidator, SyntheticWarpValidator,
            ValidationReport, find_person_weights,
        )
        from models.crowd_flow.validation.report import RouteResult

        cfg_path = os.path.join(PROJECT_ROOT, "configs", "crowd_flow.yaml")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = (yaml.safe_load(f) or {}).get("crowd_flow", {})

        def flow_factory():
            return FlowField(
                backend=cfg.get("flow_backend", "dis"),
                dis_preset=cfg.get("dis_preset", "medium"),
                target_px=cfg.get("downsample_target_px", 320),
                temporal_smooth_alpha=cfg.get("temporal_smooth_alpha", 0.4),
                global_motion_compensation=cfg.get(
                    "global_motion_compensation", True),
                gmc_max_correction_px=cfg.get("gmc_max_correction_px", 8.0),
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
                "scripts/validate_flow_routes.py --selftest-cross-camera to "
                "exercise the machinery on a synthetic pair.",
                B_CAVEAT))

        # ---- route (c) ----
        if "c" in routes:
            weights = find_person_weights(PROJECT_ROOT)
            os.makedirs(REPORT_DIR, exist_ok=True)
            video_out = os.path.join(REPORT_DIR, "cross_family_comparison.mp4")
            report.routes.append(
                CrossFamilyValidator(
                    weights_path=weights,
                    # Pinned to CPU on purpose.  JobManager serialises model
                    # runs behind a GPU lock because two networks on a 4 GB
                    # card will OOM both; this runner is a separate thread and
                    # does not hold that lock, so putting its detector on the
                    # GPU would race any job the user has going.  The person
                    # detector is a few megabytes — CPU costs a little speed
                    # and removes the interaction entirely, which also means
                    # validation never has to queue behind a long model run.
                    device="cpu",
                ).run(
                    source, flow_factory=flow_factory, max_frames=max_frames,
                    annotate_path=video_out)
            )

        return report


RUNNER = ValidationRunner()
