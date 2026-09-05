"""
Unit tests for Route Session API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from webapp.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_list_sessions_endpoint(client):
    res = client.get("/api/sessions")
    assert res.status_code == 200
    assert "sessions" in res.json()


def test_create_and_get_session_endpoint(client, tmp_path):
    # Test session creation with synthetic slot
    payload = {
        "session_name": "API_Test_Route",
        "slots": [
            {
                "camera_id": "CCTV1",
                "camera_name": "Gate A",
                "video_source": "test_videos/crowd_sample.mp4",
                "include_in_session": True,
            }
        ],
        "models": ["crowd_motion_monitor"],
        "sample_every_n_frames": 5,
        "export_video": False,
    }

    res = client.post("/api/sessions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["session_name"] == "API_Test_Route"
    assert "CCTV1" in data["cameras"]

    # Get session
    get_res = client.get("/api/sessions/API_Test_Route")
    assert get_res.status_code == 200
    assert get_res.json()["session_name"] == "API_Test_Route"

    # Clean up
    del_res = client.delete("/api/sessions/API_Test_Route")
    assert del_res.status_code == 200
