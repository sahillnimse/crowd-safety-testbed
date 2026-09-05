"""
Tests for Live Preview Mode:
- Frame drawing (draw_frame)
- LiveStreamHub thread-safe pub/sub
- Job mode serialization
- Live WebSocket endpoint (/ws/live/{job_id})
"""

import asyncio
import numpy as np
from fastapi.testclient import TestClient

from models.base import Detection
from pipeline.annotate import draw_frame
from webapp.app import app
from webapp.jobs import Job, MANAGER
from webapp.live_hub import LiveStreamHub, LIVE_HUB


def test_draw_frame_basic():
    # 640x480 black image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original_copy = frame.copy()

    dets = [
        Detection(
            model_name="yolo",
            label="standing",
            confidence=0.88,
            timestamp_sec=1.5,
            frame_index=45,
            bbox=[50, 50, 200, 300],
            extra={"track_id": 7},
        ),
        Detection(
            model_name="anpr",
            label="vehicle_plate",
            confidence=0.95,
            timestamp_sec=1.5,
            frame_index=45,
            bbox=[250, 100, 350, 150],
            extra={"plate_display": "MH 15 AB 1234"},
        ),
        Detection(
            model_name="violence",
            label="violence",
            confidence=0.72,
            timestamp_sec=1.5,
            frame_index=45,
            bbox=None,  # clip-level banner
        ),
    ]

    annotated = draw_frame(frame, dets)

    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    # Ensure original frame was NOT modified in-place
    assert np.array_equal(frame, original_copy)
    # Annotated image should have drawings on it
    assert not np.array_equal(annotated, frame)


def test_job_mode_serialization():
    job_live = Job(
        id="test_live_123",
        source="test.mp4",
        model_keys=["dense_flow"],
        sample_every_n_frames=1,
        device="cpu",
        export_video=False,
        mode="live",
    )
    d = job_live.to_dict()
    assert d["mode"] == "live"

    job_batch = Job(
        id="test_batch_123",
        source="test.mp4",
        model_keys=["dense_flow"],
        sample_every_n_frames=5,
        device="cpu",
        export_video=True,
    )
    assert job_batch.to_dict()["mode"] == "batch"


def test_live_stream_hub():
    async def _run():
        hub = LiveStreamHub()
        loop = asyncio.get_running_loop()
        hub.set_loop(loop)

        q = asyncio.Queue(maxsize=3)
        hub.register("job_abc", q)

        assert hub.has_subscribers("job_abc") is True

        # Broadcast payload
        test_payload = {"event": "frame", "frame_index": 1, "kpis": {"people": 5}}
        hub.broadcast("job_abc", test_payload)

        # Allow event loop tick to deliver
        await asyncio.sleep(0.05)

        msg = await q.get()
        assert msg["event"] == "frame"
        assert msg["frame_index"] == 1

        # Test bounded queue overflow (drops oldest to keep live stream fresh)
        for i in range(5):
            hub.broadcast("job_abc", {"event": "frame", "frame_index": i + 10})

        await asyncio.sleep(0.05)
        assert q.qsize() <= 3

        hub.unregister("job_abc", q)
        assert hub.has_subscribers("job_abc") is False

    asyncio.run(_run())


def test_ws_live_endpoint():
    # Register a test job in MANAGER
    job = Job(
        id="job_ws_test",
        source="test.mp4",
        model_keys=["dense_flow"],
        sample_every_n_frames=1,
        device="cpu",
        export_video=False,
        mode="live",
    )
    with MANAGER._lock:
        MANAGER._jobs[job.id] = job

    with TestClient(app) as client:
        with client.websocket_connect("/ws/live/job_ws_test") as ws:
            init_msg = ws.receive_json()
            assert init_msg["event"] == "init"
            assert init_msg["job_id"] == "job_ws_test"

            # Broadcast frame
            LIVE_HUB.broadcast("job_ws_test", {
                "event": "frame",
                "frame_index": 12,
                "ts": 0.4,
                "kpis": {"person_count": 3, "crush_risk": 0.05},
            })

            frame_msg = ws.receive_json()
            assert frame_msg["event"] == "frame"
            assert frame_msg["kpis"]["person_count"] == 3

            # Broadcast done
            LIVE_HUB.broadcast("job_ws_test", {
                "event": "done",
                "job_id": "job_ws_test",
                "model_key": "dense_flow",
                "status": "done",
            })

            done_msg = ws.receive_json()
            assert done_msg["event"] == "done"


def test_live_job_full_flow():
    with TestClient(app) as client:
        # Submit live job with dense_flow
        res = client.post("/api/jobs", json={
            "source": "crowd_1.mp4",
            "models": ["dense_flow"],
            "sample_every_n_frames": 1,
            "device": "cpu",
            "mode": "live",
        })
        assert res.status_code == 200, res.text
        job_data = res.json()
        job_id = job_data["id"]
        assert job_data["mode"] == "live"

        with client.websocket_connect(f"/ws/live/{job_id}") as ws:
            init_msg = ws.receive_json()
            assert init_msg["event"] == "init"

            # Wait for at least one frame message from live model
            received_frame = False
            for _ in range(50):
                msg = ws.receive_json()
                if msg.get("event") == "frame":
                    assert len(msg["jpeg_b64"]) > 0
                    assert "kpis" in msg
                    assert "person_count" in msg["kpis"]
                    assert "fps" in msg["kpis"]
                    received_frame = True
                    break
                elif msg.get("event") == "done":
                    break

            assert received_frame is True, "Must receive at least one live frame from model"

        # Cancel the job
        cancel_res = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancel_res.status_code == 200


# ---------------------------------------------------------------------------
# Adaptive stride governor
# ---------------------------------------------------------------------------

def test_live_stride_governor_matches_frame_cost():
    """Stride must cover the video the wall clock passes while a frame runs.

    Regression test.  The governor used to be fed a window that excluded
    model inference — the overwhelming majority of a frame's cost — so it
    chose a stride far too small and the live preview fell steadily behind
    the camera.  A 1 s frame on 24 fps footage has to skip ~24 frames to
    hold real time.
    """
    from webapp.jobs import LiveStrideGovernor

    gov = LiveStrideGovernor(video_fps=24.0)
    assert gov.stride == 1, "no measurements yet: process every frame"

    for _ in range(10):
        gov.observe(frame_period_sec=1.0, pacing_sleep_sec=0.0)
    assert gov.stride == 24

    # A cheap model that keeps ahead of real time needs no skipping.
    cheap = LiveStrideGovernor(video_fps=24.0)
    for _ in range(10):
        cheap.observe(frame_period_sec=0.01, pacing_sleep_sec=0.0)
    assert cheap.stride == 1


def test_live_stride_governor_discounts_pacing_sleep():
    """Time spent sleeping to pace playback is not time spent working."""
    from webapp.jobs import LiveStrideGovernor

    gov = LiveStrideGovernor(video_fps=24.0)
    for _ in range(10):
        # A 40 ms frame that then slept 960 ms waiting for the video clock.
        gov.observe(frame_period_sec=1.0, pacing_sleep_sec=0.96)
    assert gov.stride == 1, "pacing sleep must not inflate the measured cost"


def test_live_stride_governor_is_capped():
    """A pathologically slow frame must not skip the whole video."""
    from webapp.jobs import LiveStrideGovernor, _MAX_LIVE_STRIDE

    gov = LiveStrideGovernor(video_fps=24.0)
    for _ in range(10):
        gov.observe(frame_period_sec=600.0, pacing_sleep_sec=0.0)
    assert gov.stride == _MAX_LIVE_STRIDE


def test_live_stride_governor_tracks_recent_cost_only():
    """Recovery: a model that speeds up must stop skipping."""
    from webapp.jobs import LiveStrideGovernor

    gov = LiveStrideGovernor(video_fps=24.0, window=5)
    for _ in range(5):
        gov.observe(frame_period_sec=2.0)
    assert gov.stride == 48

    for _ in range(5):
        gov.observe(frame_period_sec=0.02)
    assert gov.stride == 1


# ---------------------------------------------------------------------------
# Live source classification
# ---------------------------------------------------------------------------
#
# This one predicate decides four things about a live run: whether frames may
# be dropped to hold real time, whether the model renders its own overlay and
# annotated video, whether detections are retained, and whether the run
# exports artifacts at the end.  Getting it wrong in the "file" direction
# silently discards a run's outputs; in the "camera" direction it leaks
# memory and grows a video file with no end.

def test_is_live_source_network_urls():
    from pipeline.runner import PipelineRunner

    for url in ("rtsp://cam.local/stream", "RTSP://CAM/1", "rtmp://host/live",
                "http://host/feed.mjpg", "https://host/feed", "udp://239.0.0.1:1234"):
        assert PipelineRunner.is_live_source(url) is True, url


def test_is_live_source_video_file():
    import os
    from pipeline.runner import PipelineRunner

    path = os.path.join("test_videos", "CCTV_surveillance_1.mp4")
    if not os.path.exists(path):
        import pytest
        pytest.skip("sample video not present")
    assert PipelineRunner.is_live_source(path) is False


def test_is_live_source_missing_file_is_not_live():
    """An unreadable path is not a camera.

    It is a broken file, and treating it as live would silently switch the
    run into the mode that produces no artifacts instead of failing on the
    source.
    """
    from pipeline.runner import PipelineRunner

    assert PipelineRunner.is_live_source("does/not/exist.mp4") is False


def test_stage_reports_live_source_and_export_error():
    """Both fields survive the round trip the UI reads them through."""
    from webapp.jobs import Stage

    st = Stage(model_key="crowd_motion_monitor")
    assert st.live_source is False
    assert st.export_error is None

    st.live_source = True
    st.export_error = "OSError: disk full"
    d = st.to_dict()
    assert d["live_source"] is True
    assert d["export_error"] == "OSError: disk full"


def test_runner_retains_detections_by_default():
    """Batch behaviour is unchanged; only live opts out."""
    from pipeline.runner import PipelineRunner

    class _Stub:
        name = "stub"
        consumption_type = "frame"
        clip_len = 1

    assert PipelineRunner(models=[_Stub()]).retain_detections is True
    assert PipelineRunner(models=[_Stub()],
                          retain_detections=False).retain_detections is False


def test_detector_oom_classification():
    """Only genuine memory exhaustion may fall back; everything else raises.

    The batched tile pass costs several times the peak memory of a single
    one, so it needs a way down on a small card -- but silently retrying a
    broken model one image at a time, and blaming memory for it, would hide
    a real fault behind a slow run.
    """
    from models._detectors import get_detector

    d = get_detector(device="cpu")
    import torch
    d._torch = torch

    assert d._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 126 MiB"))
    assert d._is_oom(MemoryError("out of memory"))
    assert not d._is_oom(RuntimeError("shape mismatch in forward pass"))
    assert not d._is_oom(ValueError("bad config"))
