"""
Unit tests for Route Session Manager lifecycle and directory isolation.
"""

import json
from webapp.session_jobs import (
    SessionManager,
    RouteSessionRequest,
    CameraSlotConfig,
)


def test_session_manager_start_and_manifest(tmp_path):
    sm = SessionManager(sessions_dir=str(tmp_path))

    req = RouteSessionRequest(
        session_name="Kumbh_Test_Session",
        slots=[
            CameraSlotConfig(
                camera_id="CCTV1",
                camera_name="Gate A",
                video_source="test_videos/sample.mp4",
                include_in_session=True,
            ),
            CameraSlotConfig(
                camera_id="CCTV2",
                camera_name="Gate B",
                video_source="test_videos/sample2.mp4",
                include_in_session=False,  # Excluded
            ),
            CameraSlotConfig(
                camera_id="CCTV3",
                camera_name="Merge Point",
                video_source="test_videos/sample3.mp4",
                include_in_session=True,
            ),
        ],
        models=["crowd_motion_monitor"],
    )

    session = sm.start_session(req)
    assert session.session_name == "Kumbh_Test_Session"
    assert len(session.cameras) == 2
    assert "CCTV1" in session.cameras
    assert "CCTV3" in session.cameras
    assert "CCTV2" not in session.cameras

    # Verify session directory isolation
    sess_dir = tmp_path / "Kumbh_Test_Session"
    assert sess_dir.is_dir()
    assert (sess_dir / "CCTV1").is_dir()
    assert (sess_dir / "CCTV3").is_dir()

    manifest_path = sess_dir / "session_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["session_name"] == "Kumbh_Test_Session"
    assert "CCTV1" in manifest["cameras"]


def test_session_manager_list_and_delete(tmp_path):
    sm = SessionManager(sessions_dir=str(tmp_path))

    req = RouteSessionRequest(
        session_name="Session_To_Delete",
        slots=[
            CameraSlotConfig(
                camera_id="CCTV1",
                video_source="test_videos/sample.mp4",
                include_in_session=True,
            )
        ],
    )
    sm.start_session(req)

    sessions = sm.list_sessions()
    assert any(s["session_name"] == "Session_To_Delete" for s in sessions)

    ok = sm.delete_session("Session_To_Delete")
    assert ok is True
    assert not (tmp_path / "Session_To_Delete").exists()


def test_session_has_one_shared_time_base_for_all_cameras(tmp_path):
    """Every camera in a route session must share one time base.

    Cameras are processed SEQUENTIALLY. Stamping each camera's telemetry with
    the moment its own processing began separated the camera timelines by the
    processing time of everything before it -- minutes on a 9-camera session --
    while cross-camera fusion correlates readings tens of seconds apart. Every
    upstream lookup then missed and the fusion engine could never correlate two
    cameras from a session. The clips are asserted to cover the same wall-clock
    window, so the base is per-SESSION; per-clip skew belongs in
    `clock_offset_sec`.
    """
    sm = SessionManager(sessions_dir=str(tmp_path))
    req = RouteSessionRequest(
        session_name="Shared_Clock_Session",
        slots=[
            CameraSlotConfig(camera_id="CCTV1", video_source="test_videos/a.mp4"),
            CameraSlotConfig(camera_id="CCTV2", video_source="test_videos/b.mp4"),
            CameraSlotConfig(camera_id="CCTV3", video_source="test_videos/c.mp4"),
        ],
    )
    session = sm.start_session(req)

    assert session.session_epoch_ms is not None
    assert isinstance(session.session_epoch_ms, int)

    manifest = session.to_manifest()
    assert manifest["session_epoch_ms"] == session.session_epoch_ms

    # It is a property of the session, not of any camera.
    assert not any("session_epoch_ms" in c for c in manifest["cameras"].values())

    with open(tmp_path / "Shared_Clock_Session" / "session_manifest.json",
              encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["session_epoch_ms"] == session.session_epoch_ms
