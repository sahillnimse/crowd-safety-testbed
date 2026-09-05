"""
Regression tests for the production-readiness fixes.

These cover the paths that had no tests at all and whose failure modes are
silent — which is exactly the combination that makes a safety system dangerous
rather than merely broken. Each test names the specific behaviour that used to
be wrong, so a future change that reintroduces it fails here with an
explanation rather than a bare assertion.

Deliberately free of GPU, model weights and network: everything here runs in
well under a second so it can gate every commit.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# 1. A dying source must never report success
# ======================================================================

class TestSourceIntegrity:
    """A dropped camera used to be indistinguishable from a finished video."""

    def test_completed_source_is_ok(self):
        from pipeline.runner import SourceStatus
        s = SourceStatus(expected_frames=100, frames_read=100)
        assert s.ok
        assert s.outcome == "completed"

    def test_truncated_source_is_not_ok(self):
        from pipeline.runner import SourceStatus
        s = SourceStatus(expected_frames=1000, frames_read=86, outcome="truncated")
        assert not s.ok, "a truncated read must not report ok"
        assert "TRUNCATED" in s.describe()
        assert "86" in s.describe() and "1000" in s.describe(), \
            "the operator needs the actual coverage, not just a flag"

    def test_lost_stream_says_monitoring_stopped(self):
        from pipeline.runner import SourceStatus
        s = SourceStatus(is_stream=True, frames_read=4200, reconnects=5,
                         outcome="stream_lost")
        assert not s.ok
        msg = s.describe()
        # The wording matters: this is what an operator reads at 3am.
        assert "NOT covered" in msg
        assert "STOPPED" in msg

    def test_cancelled_is_distinct_from_lost(self):
        from pipeline.runner import SourceStatus
        s = SourceStatus(outcome="cancelled", frames_read=10)
        assert not s.ok
        assert "cancelled" in s.describe().lower()

    @pytest.mark.parametrize("url,expected", [
        ("rtsp://10.0.0.5:554/stream1", True),
        ("rtmp://host/live", True),
        ("udp://239.0.0.1:1234", True),
        ("http://host/feed.m3u8", True),
        ("test_videos/clip.mp4", False),
    ])
    def test_stream_detection(self, url, expected):
        from pipeline.runner import PipelineRunner
        # total_frames > 0 so only the scheme decides, except for the file.
        assert PipelineRunner._is_stream(url, 500) is expected

    def test_file_with_no_declared_length_counts_as_stream(self):
        from pipeline.runner import PipelineRunner
        assert PipelineRunner._is_stream("weird.mkv", 0) is True


# ======================================================================
# 2. Multi-GPU concurrency
# ======================================================================

class TestDevicePool:
    """One global lock meant N GPUs still ran exactly one job."""

    def _manager(self, devices):
        import webapp.jobs as J
        original = J._available_devices
        J._available_devices = lambda: list(devices)
        try:
            return J.JobManager(), J, original
        finally:
            J._available_devices = original

    def test_capacity_matches_device_count(self):
        m, J, orig = self._manager(["cuda:0", "cuda:1", "cuda:2"])
        try:
            assert m.capacity == 3
        finally:
            J._available_devices = orig

    def test_jobs_run_concurrently_up_to_device_count(self):
        m, J, orig = self._manager(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        try:
            live, peak, lock = [], [0], threading.Lock()

            def worker():
                with m.gpu_guard() as dev:
                    with lock:
                        live.append(dev)
                        peak[0] = max(peak[0], len(live))
                    time.sleep(0.15)
                    with lock:
                        live.remove(dev)

            ts = [threading.Thread(target=worker) for _ in range(8)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            assert peak[0] > 1, "the pool must actually run jobs in parallel"
            assert peak[0] <= 4, "a device must never be double-booked"
        finally:
            J._available_devices = orig

    def test_never_oversubscribes_a_single_device(self):
        m, J, orig = self._manager(["cuda:0"])
        try:
            live, peak, lock = [], [0], threading.Lock()

            def worker():
                with m.gpu_guard() as dev:
                    with lock:
                        live.append(dev)
                        peak[0] = max(peak[0], len(live))
                    time.sleep(0.05)
                    with lock:
                        live.remove(dev)

            ts = [threading.Thread(target=worker) for _ in range(5)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            assert peak[0] == 1, "two networks on one card is the OOM this prevents"
        finally:
            J._available_devices = orig

    def test_cpu_only_machine_still_works(self):
        m, J, orig = self._manager(["cpu"])
        try:
            with m.gpu_guard() as dev:
                assert dev == "cpu"
        finally:
            J._available_devices = orig

    def test_on_wait_fires_only_when_blocked(self):
        m, J, orig = self._manager(["cpu"])
        try:
            calls = []
            with m.gpu_guard():                      # holds the only device
                t = threading.Thread(
                    target=lambda: m.gpu_guard(on_wait=lambda: calls.append(1)).__enter__())
                t.start()
                time.sleep(0.3)
                assert calls, "a queued caller must be told it is queued"
            t.join(timeout=2)
        finally:
            J._available_devices = orig


# ======================================================================
# 3. Job state survives a restart
# ======================================================================

class TestJobPersistence:
    """In-memory-only state erased every record on restart."""

    def _isolated(self, tmp):
        import webapp.jobs as J
        J.STATE_DIR = os.path.join(tmp, "state")
        J._available_devices = lambda: ["cpu"]
        return J

    def test_running_job_comes_back_as_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            J = self._isolated(tmp)
            m = J.JobManager()
            job = J.Job(id="j1", source="rtsp://cam7/live", model_keys=["m"],
                        sample_every_n_frames=5, device="cpu", export_video=False)
            job.stages["m"] = J.Stage(model_key="m", status="running")
            job.status = "running"
            m._jobs[job.id] = job
            m._persist(job)

            restored = J.JobManager().get("j1")
            assert restored is not None, "the job record must survive restart"
            assert restored.status == "interrupted", \
                "a job cannot still be 'running' — its thread died with the process"
            assert restored.stages["m"].degraded is True
            assert "INCOMPLETE" in restored.message

    def test_finished_job_keeps_its_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            J = self._isolated(tmp)
            m = J.JobManager()
            job = J.Job(id="j2", source="clip.mp4", model_keys=["m"],
                        sample_every_n_frames=5, device="cpu", export_video=False)
            job.status = "done"
            job.message = "All models completed."
            m._jobs[job.id] = job
            m._persist(job)

            restored = J.JobManager().get("j2")
            assert restored.status == "done", \
                "a genuinely completed job must not be downgraded on restart"

    def test_corrupt_state_file_does_not_crash_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            J = self._isolated(tmp)
            os.makedirs(J.STATE_DIR, exist_ok=True)
            with open(os.path.join(J.STATE_DIR, "bad.json"), "w") as f:
                f.write("{ this is not json")
            # A damaged record must not stop the server from starting: losing
            # one audit entry is bad, failing to boot is worse.
            m = J.JobManager()
            assert m is not None

    def test_persist_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            J = self._isolated(tmp)
            m = J.JobManager()
            job = J.Job(id="j3", source="s", model_keys=[], device=None,
                        sample_every_n_frames=1, export_video=False)
            m._persist(job)
            files = os.listdir(J.STATE_DIR)
            assert "j3.json" in files
            assert not any(f.endswith(".tmp") for f in files), \
                "the temp file must be renamed, never left behind"


# ======================================================================
# 4. Alerts actually leave the process
# ======================================================================

class TestAlertDelivery:
    """Alerts used to end their life as a row in a JSON file."""

    def _det(self, label, severity="warning", **extra):
        from models.base import Detection
        e = {"severity": severity}
        e.update(extra)
        return Detection("m", label, 0.9, 12.5, 300, None, None, e)

    def test_alerts_are_written_to_the_log_sink(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "alerts.jsonl")
            from pipeline.alert_sink import AlertDispatcher, AlertEvent
            d = AlertDispatcher(webhook=None, log_path=log, min_severity="warning")
            d.dispatch(AlertEvent("cam1", "mean_divergence_critical", "critical",
                                  "mean_divergence", -2.4, -1.5, 12.5, 300))
            d.stop()
            rows = [json.loads(l) for l in open(log, encoding="utf-8")]
            assert len(rows) == 1
            assert rows[0]["camera_id"] == "cam1"
            assert rows[0]["severity"] == "critical"

    def test_telemetry_rows_are_not_alerts(self):
        """flow_analysis is emitted EVERY frame; delivering it buries real alerts."""
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "a.jsonl")
            import importlib, pipeline.alert_sink as A
            os.environ["CROWD_ALERT_LOG"] = log
            importlib.reload(A)
            try:
                n = A.dispatch_detections(
                    [self._det("flow_analysis"),
                     self._det("dm_frame_metrics"),
                     self._det("mean_divergence_critical", "critical")],
                    camera_id="c")
                A.DISPATCHER.stop()
                assert n == 1, "only the real alert should be delivered"
            finally:
                os.environ.pop("CROWD_ALERT_LOG", None)
                importlib.reload(A)

    def test_severity_filter(self):
        from pipeline.alert_sink import AlertDispatcher, AlertEvent
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "a.jsonl")
            d = AlertDispatcher(webhook=None, log_path=log, min_severity="critical")
            assert not d.dispatch(AlertEvent("c", "x", "warning", "m", 1, 2, 0, 0))
            assert d.dispatch(AlertEvent("c", "x", "critical", "m", 1, 2, 0, 0))
            d.stop()

    def test_disabled_dispatcher_is_a_no_op(self):
        from pipeline.alert_sink import AlertDispatcher, AlertEvent
        d = AlertDispatcher(webhook=None, log_path=None)
        assert not d.enabled
        assert not d.dispatch(AlertEvent("c", "x", "critical", "m", 1, 2, 0, 0))

    def test_a_broken_sink_never_raises_into_the_pipeline(self):
        """A DNS failure must not stop a camera being monitored."""
        from pipeline.alert_sink import AlertDispatcher, AlertEvent
        d = AlertDispatcher(webhook="http://127.0.0.1:9/nope",
                            log_path=None, min_severity="warning")
        d.dispatch(AlertEvent("c", "x", "critical", "m", 1, 2, 0, 0))
        d.stop()
        assert d.failed >= 1, "the failure should be counted"
        # Reaching here at all is the assertion: nothing propagated.

    def test_queue_overflow_is_counted_not_hidden(self):
        from pipeline.alert_sink import AlertDispatcher, AlertEvent
        import pipeline.alert_sink as A
        d = AlertDispatcher(webhook=None, log_path=os.devnull)
        d._q = __import__("queue").Queue(maxsize=2)
        for _ in range(6):
            d.dispatch(AlertEvent("c", "x", "critical", "m", 1, 2, 0, 0))
        assert d.dropped > 0, "dropped alerts must be counted, never silent"


# ======================================================================
# 5. API authentication
# ======================================================================

class TestApiAuth:
    def _client(self, token=None):
        import importlib
        if token:
            os.environ["CROWD_API_TOKEN"] = token
        else:
            os.environ.pop("CROWD_API_TOKEN", None)
        import webapp.app as A
        importlib.reload(A)
        from fastapi.testclient import TestClient
        return TestClient(A.app), A

    def teardown_method(self):
        os.environ.pop("CROWD_API_TOKEN", None)

    def test_no_token_configured_leaves_local_dev_open(self):
        c, _ = self._client(None)
        assert c.get("/api/models").status_code == 200

    def test_token_configured_blocks_unauthenticated_calls(self):
        c, _ = self._client("secret")
        assert c.get("/api/models").status_code == 401

    def test_valid_token_is_accepted(self):
        c, _ = self._client("secret")
        r = c.get("/api/models", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200

    def test_destructive_endpoint_is_protected(self):
        c, _ = self._client("secret")
        assert c.delete("/api/outputs").status_code == 401, \
            "DELETE /api/outputs wipes every stored run"

    def test_health_stays_public_for_probes(self):
        c, _ = self._client("secret")
        assert c.get("/api/health").status_code == 200


# ======================================================================
# 6/7. Deployment preflight
# ======================================================================

class TestPreflight:
    def test_uncalibrated_camera_is_a_blocker(self):
        from models.crowd_flow import preflight
        f = preflight.check_camera("cam", {"zones": [{"name": "z",
                                   "polygon": [[0, 0], [10, 0], [10, 10]],
                                   "thresholds": {"divergence_critical": -1.5}}]}, {})
        assert "NO_HOMOGRAPHY" in [x.code for x in f]
        assert any(x.severity == preflight.BLOCKER for x in f)

    def test_pressure_thresholds_without_density_is_a_blocker(self):
        """The second, independent gate: calibration alone does not fix it."""
        from models.crowd_flow import preflight
        cam = {"homography": {"a": 1},
               "zones": [{"name": "z", "polygon": [[0, 0], [9, 0], [9, 9]],
                          "thresholds": {"pressure_critical": 4.0}}]}
        codes = [x.code for x in preflight.check_camera("c", cam, {"density_enabled": False})]
        assert "PRESSURE_WITHOUT_DENSITY" in codes

    def test_enabling_density_clears_that_blocker(self):
        from models.crowd_flow import preflight
        cam = {"homography": {"a": 1},
               "zones": [{"name": "z", "polygon": [[0, 0], [9, 0], [9, 9]],
                          "thresholds": {"pressure_critical": 4.0}}]}
        codes = [x.code for x in preflight.check_camera("c", cam, {"density_enabled": True})]
        assert "PRESSURE_WITHOUT_DENSITY" not in codes

    def test_shipped_placeholder_polygon_is_caught(self):
        from models.crowd_flow import preflight
        cam = {"homography": {"a": 1}, "zones": [{
            "name": "ghat_approach",
            "polygon": [[0, 240], [640, 240], [640, 480], [0, 480]],
            "thresholds": {"divergence_critical": -1.5}}]}
        codes = [x.code for x in preflight.check_camera("c", cam, {"density_enabled": True})]
        assert "PLACEHOLDER_ZONE" in codes

    def test_camera_with_no_zones_is_a_blocker(self):
        from models.crowd_flow import preflight
        codes = [x.code for x in preflight.check_camera("c", {"homography": {}}, {})]
        assert "NO_ZONES" in codes

    def test_whole_frame_zone_is_flagged(self):
        from models.crowd_flow import preflight
        cam = {"homography": {"a": 1}, "zones": [{
            "name": "everything",
            "polygon": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
            "thresholds": {"divergence_critical": -1.5}}]}
        codes = [x.code for x in preflight.check_camera(
            "c", cam, {"density_enabled": True}, frame_wh=(1920, 1080))]
        assert "WHOLE_FRAME_ZONE" in codes

    def test_report_marks_instrumentation_status(self):
        from models.crowd_flow import preflight
        findings = preflight.check_camera("c", {"zones": []}, {})
        rep = preflight.report("c", findings)
        assert rep["fully_instrumented"] is False
        assert rep["blockers"]

    def test_the_shipped_kumbh_cameras_are_not_production_ready(self):
        """
        Documents the real state of configs/crowd_flow.yaml.

        If someone calibrates the cameras and authors real zones, this test
        starts failing — which is the correct signal to update it, and a much
        better outcome than nobody noticing either way.
        """
        from config_io import load_yaml
        from models.crowd_flow import preflight
        cfg = load_yaml("configs/crowd_flow.yaml")["crowd_flow"]
        for cam in ("ram_kund_approach", "kushavarta_kund_approach",
                    "ram_kund_bridge", "procession_route_north"):
            block = cfg["cameras"].get(cam, {})
            findings = preflight.check_camera(cam, block, cfg)
            blockers = [f.code for f in findings if f.severity == preflight.BLOCKER]
            assert "NO_HOMOGRAPHY" in blockers, (
                f"{cam} appears to be calibrated now — update this test and "
                f"re-check the deployment readiness assessment.")


# ======================================================================
# 8. Source-path restriction
# ======================================================================

class TestMediaPathRestriction:
    def test_paths_outside_media_roots_are_rejected(self):
        import webapp.app as A
        assert not A._within_allowed_roots(os.path.abspath(os.sep + "etc/passwd"))

    def test_test_videos_is_allowed(self):
        import webapp.app as A
        p = os.path.join(A.TEST_VIDEOS_DIR, "clip.mp4")
        assert A._within_allowed_roots(p)

    def test_traversal_out_of_media_root_is_rejected(self):
        import webapp.app as A
        p = os.path.join(A.TEST_VIDEOS_DIR, "..", "..", "secrets.txt")
        assert not A._within_allowed_roots(p)


class TestPreflightCalibrationDetection:
    """
    A `homography:` KEY is not a calibration.

    The shipped `crowd_ralley` camera carries a placeholder block with empty
    point lists. `bool(cam_cfg.get("homography"))` is truthy for it, so
    preflight reported that camera FULLY INSTRUMENTED while CameraCalibration
    correctly treated it as uncalibrated and disabled every speed and pressure
    threshold — preflight contradicting the engine it reports on.
    """

    def test_empty_homography_block_is_not_calibrated(self):
        from models.crowd_flow import preflight
        stub = {"homography": {"image_points": [], "world_points_m": []}}
        assert preflight._is_really_calibrated(stub) is False

    def test_too_few_points_is_not_calibrated(self):
        from models.crowd_flow import preflight
        three = {"homography": {"image_points": [[0, 0], [1, 0], [1, 1]],
                                "world_points_m": [[0, 0], [1, 0], [1, 1]]}}
        assert preflight._is_really_calibrated(three) is False, \
            "a plane mapping needs at least four correspondences"

    def test_mismatched_point_counts_is_not_calibrated(self):
        from models.crowd_flow import preflight
        bad = {"homography": {"image_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                              "world_points_m": [[0, 0], [1, 0], [1, 1]]}}
        assert preflight._is_really_calibrated(bad) is False

    def test_four_matched_points_is_calibrated(self):
        from models.crowd_flow import preflight
        ok = {"homography": {"image_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                             "world_points_m": [[0, 0], [8, 0], [8, 12], [0, 12]]}}
        assert preflight._is_really_calibrated(ok) is True

    def test_perspective_map_counts_as_calibrated(self):
        """The no-site-visit route must also open the gate."""
        from models.crowd_flow import preflight
        pm = {"perspective_map": {"ah": -0.05, "bh": 0.78, "ch": -50.4}}
        assert preflight._is_really_calibrated(pm) is True

    def test_preflight_agrees_with_the_calibration_engine(self):
        """
        The two must never disagree: preflight exists to report on the engine,
        so a divergence means one of them is lying to the operator.
        """
        from config_io import load_yaml
        from models.crowd_flow import preflight
        from models.crowd_flow.ground_plane import CameraCalibration
        cfg = load_yaml("configs/crowd_flow.yaml")["crowd_flow"]
        for cam, block in cfg["cameras"].items():
            assert (preflight._is_really_calibrated(block)
                    == CameraCalibration.from_yaml_block(cam, block).is_calibrated), \
                f"preflight and CameraCalibration disagree about '{cam}'"
