"""
Route Session Job Engine: runs multi-camera route sessions and manages their lifecycle.

Isolates all session outputs under `outputs/sessions/<session_name>/` separate
from single-camera runs under `outputs/runs/`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pipeline.alert_sink import dispatch_detections
from pipeline.runner import PipelineRunner
from topology.graph import TOPOLOGY
from topology.metric_store import METRIC_STORE
from webapp.jobs import (PROJECT_ROOT, SESSIONS_DIR,
                         positive_labels)
from webapp.registry import build_model
from webapp.session_report import build_session_report

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Normalize session names to safe filesystem directory names."""
    s = re.sub(r"[^\w\-_.]", "_", name.strip())
    return s or "route_session"


@dataclass
class CameraSlotConfig:
    camera_id: str
    video_source: str
    camera_name: str = ""
    include_in_session: bool = True


@dataclass
class RouteSessionRequest:
    session_name: str
    slots: list[CameraSlotConfig]
    models: list[str] = field(default_factory=lambda: ["crowd_motion_monitor"])
    sample_every_n_frames: int = 5
    device: Optional[str] = None
    export_video: bool = True
    threshold: Optional[float] = None


@dataclass
class CameraRunState:
    camera_id: str
    camera_name: str
    video_source: str
    video_name: str = ""
    status: str = "pending"  # pending | running | done | failed
    progress: float = 0.0
    frames_done: int = 0
    frames_total: int = 0
    detections: int = 0
    positives: int = 0
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    run_dir: str = ""


@dataclass
class RouteSessionState:
    session_name: str
    created_at: str
    status: str = "pending"  # pending | running | generating_report | done | failed | partial
    models: list[str] = field(default_factory=list)
    cameras: dict[str, CameraRunState] = field(default_factory=dict)
    error: Optional[str] = None
    session_dir: str = ""
    report_html: Optional[str] = None
    summary_json: Optional[str] = None
    # One common time base for every camera in the session.
    #
    # The clips mapped onto a route are asserted by the operator to cover the
    # SAME wall-clock window; cross-camera fusion correlates an upstream
    # reading at t - travel_time_sec with a downstream reading at t, and
    # travel times are tens of seconds.
    #
    # Cameras are processed sequentially, so stamping each camera with the
    # moment ITS processing began separated the timelines by however long the
    # preceding cameras took to run -- minutes, on a 9-camera session. Every
    # cross-camera lookup then fell outside the retained history and the
    # fusion engine could never correlate two cameras from a route session.
    #
    # Per-clip start skew belongs in the camera's `clock_offset_sec`, which is
    # surveyed and applies on top of this shared base.
    session_epoch_ms: Optional[int] = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "session_name": self.session_name,
            "created_at": self.created_at,
            "status": self.status,
            "models": self.models,
            "error": self.error,
            "report_html": self.report_html,
            "summary_json": self.summary_json,
            "session_epoch_ms": self.session_epoch_ms,
            "cameras": {
                cid: asdict(c) for cid, c in self.cameras.items()
            },
        }


class SessionManager:
    """Manages execution and persistence of multi-camera Route Sessions."""

    def __init__(self, sessions_dir: str = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        os.makedirs(self.sessions_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._active_sessions: dict[str, RouteSessionState] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start_session(self, req: RouteSessionRequest) -> RouteSessionState:
        """Start an asynchronous multi-camera route session."""
        with self._lock:
            safe_name = _slugify(req.session_name)
            active_slots = [s for s in req.slots if s.include_in_session]
            if not active_slots:
                raise ValueError("Route session must include at least one camera slot.")

            session_dir = os.path.join(self.sessions_dir, safe_name)
            os.makedirs(session_dir, exist_ok=True)

            cam_states: dict[str, CameraRunState] = {}
            for s in active_slots:
                node = TOPOLOGY.get_camera(s.camera_id)
                c_name = s.camera_name or (node.name if node else s.camera_id)
                v_name = os.path.basename(s.video_source)
                c_dir = os.path.join(session_dir, s.camera_id)
                os.makedirs(c_dir, exist_ok=True)

                cam_states[s.camera_id] = CameraRunState(
                    camera_id=s.camera_id,
                    camera_name=c_name,
                    video_source=s.video_source,
                    video_name=v_name,
                    run_dir=c_dir,
                )

            session = RouteSessionState(
                session_name=safe_name,
                created_at=datetime.now(timezone.utc).isoformat(),
                status="pending",
                models=req.models,
                cameras=cam_states,
                session_dir=session_dir,
                session_epoch_ms=int(time.time() * 1000),
            )

            self._active_sessions[safe_name] = session
            self._save_manifest(session)

            t = threading.Thread(
                target=self._run_session_worker,
                args=(session, req),
                daemon=True,
                name=f"SessionWorker-{safe_name}",
            )
            self._threads[safe_name] = t
            t.start()
            return session

    def _save_manifest(self, session: RouteSessionState) -> None:
        """Persist session_manifest.json atomically to disk if session has not been deleted."""
        with self._lock:
            if session.session_name not in self._active_sessions:
                return
        if not os.path.isdir(session.session_dir):
            return
        m_path = os.path.join(session.session_dir, "session_manifest.json")
        tmp_path = m_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(session.to_manifest(), f, indent=2)
            os.replace(tmp_path, m_path)
        except Exception as e:
            logger.warning("Could not write session manifest: %s", e)

    def _run_session_worker(self, session: RouteSessionState, req: RouteSessionRequest) -> None:
        """Execute each camera in the session, then build fused report."""
        session.status = "running"
        self._save_manifest(session)

        # Execute camera pipelines
        for cam_id, cam_state in session.cameras.items():
            cam_state.status = "running"
            cam_state.started_at = time.time()
            self._save_manifest(session)

            try:
                self._execute_camera_pipeline(session, cam_state, req)
                cam_state.status = "done"
                cam_state.progress = 1.0
                cam_state.finished_at = time.time()
            except Exception as e:
                logger.error("Camera %s failed in session %s: %s", cam_id, session.session_name, e, exc_info=True)
                cam_state.status = "failed"
                cam_state.error = str(e)
                cam_state.finished_at = time.time()

            self._save_manifest(session)

        # Check overall status
        failed_count = sum(1 for c in session.cameras.values() if c.status == "failed")
        done_count = sum(1 for c in session.cameras.values() if c.status == "done")

        if done_count == 0:
            session.status = "failed"
            session.error = "All cameras in session failed to execute."
        elif failed_count > 0:
            session.status = "partial"
        else:
            session.status = "generating_report"

        self._save_manifest(session)

        # Generate fused report if at least one camera succeeded
        if done_count > 0:
            try:
                manifest_dict = session.to_manifest()
                sum_path, rep_path = build_session_report(
                    session_dir=session.session_dir,
                    session_name=session.session_name,
                    topology=TOPOLOGY,
                    manifest=manifest_dict,
                )
                session.summary_json = "session_summary.json"
                session.report_html = "session_report.html"
                if session.status != "partial":
                    session.status = "done"
            except Exception as e:
                logger.error("Failed to build fused session report: %s", e, exc_info=True)
                session.error = f"Failed to generate session report: {e}"
                session.status = "failed"

        self._save_manifest(session)

    def _execute_camera_pipeline(
        self,
        session: RouteSessionState,
        cam_state: CameraRunState,
        req: RouteSessionRequest,
    ) -> None:
        """Run requested models on a camera's video source."""
        # Resolve source video
        src = cam_state.video_source
        if not os.path.isabs(src):
            test_v_path = os.path.join(PROJECT_ROOT, "test_videos", src)
            if os.path.exists(test_v_path):
                src = test_v_path

        if not os.path.exists(src):
            raise FileNotFoundError(f"Video source not found: {src}")

        cam_state.video_name = os.path.basename(src)

        for model_key in req.models:
            model = build_model(
                model_key,
                device=req.device,
                video_name=f"{session.session_name}_{cam_state.camera_id}",
                threshold=req.threshold,
                camera_id=cam_state.camera_id,
            )

            # Give flow models video meta
            if getattr(model, "consumption_type", "") == "flow_pair":
                from pipeline.video_meta import source_fps
                _src_fps = source_fps(src)
                stride = max(1, int(req.sample_every_n_frames or 1))
                model._fps = float(_src_fps)
                model._frame_stride = stride
                model.output_fps = float(_src_fps) / stride

            model.load()

            runner = PipelineRunner(
                models=[model],
                sample_every_n_frames=req.sample_every_n_frames,
            )

            cam_node = TOPOLOGY.get_camera(cam_state.camera_id)
            cam_clock_offset = cam_node.clock_offset_sec if cam_node else 0.0
            session_epoch_ms = session.session_epoch_ms

            def on_progress(done, total, n_dets):
                cam_state.frames_done = int(done)
                cam_state.frames_total = int(total)
                cam_state.progress = (done / total) if total else 0.0
                cam_state.detections = int(n_dets)

            # Latch so a per-frame telemetry failure is reported once, at a
            # level that is actually visible. This block previously failed on
            # EVERY frame -- it called METRIC_STORE.update() with the wrong
            # keyword names -- and logged the TypeError at DEBUG, so the
            # multi-camera session path never wrote a single metric and the
            # fusion engine saw no data from the feature built to feed it.
            _telemetry_warned = {"done": False}

            def on_detections(dets):
                if dets:
                    pos_count = sum(1 for d in dets if d.label in positive_labels())
                    cam_state.positives += pos_count

                    # Alerts must leave the process here too. The single-camera
                    # path in jobs.py dispatches; this one imported
                    # dispatch_detections and never called it, so an alert
                    # raised during a route session reached nobody.
                    try:
                        dispatch_detections(
                            dets, camera_id=cam_state.camera_id,
                            source=getattr(cam_state, "video_name", "") or src)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Alert dispatch failed for %s: %s",
                                       cam_state.camera_id, exc)

                    # Extract live telemetry into METRIC_STORE
                    try:
                        raw_ts = float(getattr(dets[0], "timestamp_sec", 0.0))
                        person_dets = [d for d in dets if "person" in d.label or "head" in d.label]
                        moving_dets = [d for d in person_dets if "moving" in d.label or "crush" in d.label]
                        p_count = len(person_dets) if person_dets else len(dets)

                        speed_samples = [
                            float(d.extra.get("speed_px_frame", 1.0))
                            for d in dets if isinstance(d.extra, dict) and "speed_px_frame" in d.extra
                        ]
                        avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 1.0
                        flow_rate = max(0.0, float(len(moving_dets) * max(1.0, avg_speed) * 12.0))

                        crush_scores = [
                            float(d.extra.get("local_crush_risk", 0.0))
                            for d in dets if isinstance(d.extra, dict) and "local_crush_risk" in d.extra
                        ]
                        crush_risk = max(crush_scores) if crush_scores else 0.0

                        # Keyword names must match MetricStore.update exactly:
                        is_cal = bool(getattr(getattr(model, "_calib", None), "is_calibrated", False))
                        METRIC_STORE.update(
                            camera_id=cam_state.camera_id,
                            raw_timestamp_sec=raw_ts,
                            person_count=p_count,
                            flow_rate_pax_min=flow_rate,
                            crush_risk_score=crush_risk,
                            dominant_direction_vector=(0.0, 0.0),
                            clock_offset_sec=cam_clock_offset,
                            # Shared session base, NOT this camera's processing
                            # start -- see RouteSessionState.session_epoch_ms.
                            stream_start_epoch_ms=session_epoch_ms,
                            flow_is_calibrated=is_cal,
                            density_is_calibrated=is_cal,
                            units=("pax/min, pax/m2" if is_cal
                                   else "UNCALIBRATED: flow=relative (scaled count x speed)"),
                        )
                    except Exception as e:
                        # WARNING, not DEBUG, and latched. A silent telemetry
                        # failure here disables cross-camera fusion entirely
                        # while the session still reports success.
                        if not _telemetry_warned["done"]:
                            _telemetry_warned["done"] = True
                            logger.warning(
                                "Telemetry sync FAILED for camera '%s' (%s: %s). "
                                "Cross-camera fusion will have no data from this "
                                "camera for the whole session. Logged once.",
                                cam_state.camera_id, e.__class__.__name__, e)

            # Run pipeline
            det_list = runner.run(
                video_path=src,
                progress_callback=on_progress,
                on_detections=on_detections,
            )

            # Export camera artifacts into cam_state.run_dir
            from pipeline.annotate import (export_annotated_video,
                                           export_detection_csv,
                                           export_detection_log)
            from pipeline.html_report import export_html_report

            # 1. detections.json
            export_detection_log(det_list, os.path.join(cam_state.run_dir, "detections.json"))

            # 2. detections.csv
            export_detection_csv(det_list, os.path.join(cam_state.run_dir, "detections.csv"))

            # 3. summary.json
            summary_dict = {}
            if hasattr(model, "summary") and model.summary:
                summary_dict = dict(model.summary)
            elif hasattr(model, "finalize"):
                summary_dict = model.finalize() or {}
            else:
                from webapp.history import compute_detections_summary
                summary_dict = compute_detections_summary(det_list)

            summary_dict["camera_id"] = cam_state.camera_id
            summary_dict["camera_name"] = cam_state.camera_name
            summary_dict["video_name"] = cam_state.video_name
            summary_dict["total_detections"] = len(det_list)

            with open(os.path.join(cam_state.run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2)

            # 4. report.html (individual camera report)
            try:
                report_out_path = os.path.join(cam_state.run_dir, "report.html")
                export_html_report(
                    output_path=report_out_path,
                    video_name=cam_state.video_name,
                    model_key=model_key,
                    summary=summary_dict,
                    detections=det_list,
                )
            except Exception as e:
                logger.warning("Could not generate individual report for %s: %s", cam_state.camera_id, e)

            # 5. annotated.mp4
            if req.export_video:
                out_video_path = os.path.join(cam_state.run_dir, "annotated.mp4")
                own_video = getattr(model, "annotated_video_path", None)
                if own_video and os.path.exists(own_video):
                    shutil.copyfile(own_video, out_video_path)
                else:
                    try:
                        export_annotated_video(src, det_list, out_video_path)
                    except Exception as e:
                        logger.warning("Could not write annotated video for %s: %s", cam_state.camera_id, e)

    def get_session(self, name: str) -> Optional[dict[str, Any]]:
        """Get session metadata and current state."""
        safe = _slugify(name)
        data = None
        with self._lock:
            if safe in self._active_sessions:
                data = self._active_sessions[safe].to_manifest()

        # Try loading manifest from disk
        if not data:
            m_path = os.path.join(self.sessions_dir, safe, "session_manifest.json")
            if os.path.exists(m_path):
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass

        if data:
            sum_path = os.path.join(self.sessions_dir, safe, "session_summary.json")
            if os.path.exists(sum_path):
                try:
                    with open(sum_path, "r", encoding="utf-8") as f:
                        data["summary"] = json.load(f)
                except Exception:
                    pass
        return data

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved and active route sessions."""
        sessions = []
        seen = set()

        with self._lock:
            for safe, s in self._active_sessions.items():
                m = s.to_manifest()
                sum_path = os.path.join(self.sessions_dir, safe, "session_summary.json")
                if os.path.exists(sum_path):
                    try:
                        with open(sum_path, "r", encoding="utf-8") as sf:
                            m["summary"] = json.load(sf)
                    except Exception:
                        pass
                sessions.append(m)
                seen.add(safe)

        if os.path.isdir(self.sessions_dir):
            for entry in os.listdir(self.sessions_dir):
                if entry in seen:
                    continue
                m_path = os.path.join(self.sessions_dir, entry, "session_manifest.json")
                if os.path.exists(m_path):
                    try:
                        with open(m_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            sum_path = os.path.join(self.sessions_dir, entry, "session_summary.json")
                            if os.path.exists(sum_path):
                                try:
                                    with open(sum_path, "r", encoding="utf-8") as sf:
                                        data["summary"] = json.load(sf)
                                except Exception:
                                    pass
                            sessions.append(data)
                    except Exception as e:
                        logger.debug("Failed reading %s: %s", m_path, e)

        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions

    def delete_session(self, name: str) -> bool:
        """Delete session directory and manifest from disk."""
        safe = _slugify(name)
        with self._lock:
            self._active_sessions.pop(safe, None)
            t = self._threads.pop(safe, None)
        if t and t.is_alive():
            t.join(timeout=1.0)
        target = os.path.join(self.sessions_dir, safe)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            return True
        return False


SESSION_MANAGER = SessionManager()
