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
