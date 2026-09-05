"""Tests for FastAPI topology and fusion endpoints."""

import pytest
from fastapi.testclient import TestClient
from webapp.app import app


@pytest.fixture
def client():
    return TestClient(app)



@pytest.fixture
def isolated_generated_topology(tmp_path, monkeypatch):
    """
    Redirect the generated-topology path into a temp dir for the duration of
    a test.

    These tests previously wrote to and DELETED the real
    ``configs/topology.generated.yaml`` — a file tracked in git. Running the
    suite left the working tree dirty with a deletion, and on a machine where
    someone had built a route through the UI it destroyed their topology.
    A test must never mutate project configuration.

    Patched on the module global, which every call site reads at call time
    (``graph.reset_to_default``, ``graph.is_generated``, and the local import
    inside the API handler), so one patch covers all of them.
    """
    import topology.graph as G
    import webapp.app as A
    fake = str(tmp_path / "topology.generated.yaml")
    monkeypatch.setattr(G, "GENERATED_TOPOLOGY_PATH", fake)
    monkeypatch.setattr(A, "GENERATED_TOPOLOGY_PATH", fake, raising=False)
    yield fake
    # Leave the real topology object back on its baseline for later tests.
    G.TOPOLOGY.reset_to_default()


def test_api_topology_get(client):
    res = client.get("/api/topology")
    assert res.status_code == 200
    data = res.json()
    assert "cameras" in data
    assert "edges" in data
    assert "CCTV1" in data["cameras"]
    assert "CCTV2" in data["cameras"]
    assert "CCTV3" in data["cameras"]


def test_api_fusion_alerts(client):
    res = client.get("/api/fusion/alerts")
    assert res.status_code == 200
    data = res.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_api_fusion_metrics(client):
    res = client.get("/api/fusion/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "cameras" in data
    assert "CCTV1" in data["cameras"]


def test_api_fusion_sparklines(client):
    res = client.get("/api/fusion/sparklines")
    assert res.status_code == 200
    data = res.json()
    assert "sparklines" in data
    assert "CCTV1" in data["sparklines"]


def test_api_topology_admin_update(client):
    # Test valid update
    payload = {
        "staleness_threshold_sec": 6.0,
        "fusion_tick_sec": 1.5,
        "density_threshold": 3.0,
        "cameras": {
            "CCTV1": {"name": "Gate A Updated", "corridor_capacity_pax_min": 420.0, "position": {"x": 100, "y": 100}},
            "CCTV2": {"name": "Gate B", "corridor_capacity_pax_min": 350.0, "position": {"x": 100, "y": 300}},
            "CCTV3": {"name": "Merge Point", "corridor_capacity_pax_min": 500.0, "position": {"x": 400, "y": 200}},
        },
        "edges": [
            {"from": "CCTV1", "to": "CCTV3", "travel_time_sec": 24.0},
            {"from": "CCTV2", "to": "CCTV3", "travel_time_sec": 19.0},
        ],
    }
    res = client.post("/api/topology", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["topology"]["cameras"]["CCTV1"]["name"] == "Gate A Updated"


# ======================================================================
# Regression tests for defects found in review
# ======================================================================

def test_websocket_requires_a_token_when_auth_is_enabled():
    """
    The HTTP middleware only gates paths starting with "/api", so "/ws/fusion"
    bypassed authentication entirely and streamed live camera telemetry to any
    client that could reach the port.
    """
    import importlib, os
    os.environ["CROWD_API_TOKEN"] = "secret-ws-token"
    try:
        import webapp.app as A
        importlib.reload(A)
        from fastapi.testclient import TestClient
        c = TestClient(A.app)

        connected = False
        try:
            with c.websocket_connect("/ws/fusion") as ws:
                ws.receive_json()
                connected = True
        except Exception:
            connected = False
        assert not connected, "an unauthenticated websocket must be refused"

        # ...and the correct token still gets through.
        with c.websocket_connect("/ws/fusion?token=secret-ws-token") as ws:
            msg = ws.receive_json()
            assert msg["event"] == "init"
    finally:
        os.environ.pop("CROWD_API_TOKEN", None)
        import webapp.app as A2
        importlib.reload(A2)


def test_metrics_endpoint_exposes_forecast_completeness(client):
    """A partial forecast under-estimates inflow and suppresses alerts, so the
    UI must be able to tell 'quiet' from 'blind'."""
    r = client.get("/api/fusion/metrics")
    assert r.status_code == 200
    for cam in r.json()["cameras"].values():
        assert "forecast_status" in cam
        assert "complete" in cam["forecast_status"]


def test_api_topology_from_route_and_reset(client, isolated_generated_topology):
    import os
    import topology.graph as G
    GENERATED_TOPOLOGY_PATH = G.GENERATED_TOPOLOGY_PATH

    payload = {
        "cameras": [
            {"camera_id": "ROUTE_A", "name": "Start", "corridor_capacity_pax_min": 400.0},
            {"camera_id": "ROUTE_B", "name": "Middle", "corridor_capacity_pax_min": 350.0},
            {"camera_id": "ROUTE_C", "name": "End", "corridor_capacity_pax_min": 300.0},
        ],
        "edges": [
            {"from": "ROUTE_A", "to": "ROUTE_B", "travel_time_sec": 20.0},
            {"from": "ROUTE_B", "to": "ROUTE_C", "travel_time_sec": 25.0},
        ],
    }

    try:
        res = client.post("/api/topology/from-route", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["topology"]["is_generated"] is True
        assert os.path.exists(GENERATED_TOPOLOGY_PATH)

        # GET confirms is_generated is true
        get_res = client.get("/api/topology")
        assert get_res.json()["is_generated"] is True
        assert "ROUTE_A" in get_res.json()["cameras"]

        # Reset reverts to baseline topology.yaml
        reset_res = client.post("/api/topology/reset")
        assert reset_res.status_code == 200
        assert reset_res.json()["topology"]["is_generated"] is False
        assert not os.path.exists(GENERATED_TOPOLOGY_PATH)

        # GET confirms baseline
        get_res2 = client.get("/api/topology")
        assert get_res2.json()["is_generated"] is False
        assert "CCTV1" in get_res2.json()["cameras"]
    finally:
        if os.path.exists(GENERATED_TOPOLOGY_PATH):
            try:
                os.remove(GENERATED_TOPOLOGY_PATH)
            except Exception:
                pass


def test_api_topology_from_route_rejects_invalid(client, isolated_generated_topology):
    # Cycle payload
    payload = {
        "cameras": [{"camera_id": "A"}, {"camera_id": "B"}],
        "edges": [
            {"from": "A", "to": "B", "travel_time_sec": 10.0},
            {"from": "B", "to": "A", "travel_time_sec": 10.0},
        ],
    }
    res = client.post("/api/topology/from-route", json=payload)
    assert res.status_code == 400
    assert "Cycle detected" in res.json()["detail"]
