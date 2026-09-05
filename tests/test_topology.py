"""Unit tests for CameraTopology and graph query operations."""

import pytest
from topology.graph import CameraTopology


def test_topology_load_from_dict():
    data = {
        "staleness_threshold_sec": 4.0,
        "fusion_tick_sec": 0.5,
        "density_threshold": 3.0,
        "cameras": {
            "CAM_A": {"name": "Gate A", "corridor_capacity_pax_min": 450, "position": {"x": 100, "y": 150}, "clock_offset_sec": 1.2},
            "CAM_B": {"name": "Gate B", "corridor_capacity_pax_min": 300, "position": {"x": 100, "y": 350}, "clock_offset_sec": -0.5},
            "CAM_C": {"name": "Junction", "corridor_capacity_pax_min": 600, "position": {"x": 400, "y": 250}},
        },
        "edges": [
            {"from": "CAM_A", "to": "CAM_C", "travel_time_sec": 22.5},
            {"from": "CAM_B", "to": "CAM_C", "travel_time_sec": 18.0},
        ],
    }

    topo = CameraTopology()
    topo.update_from_dict(data)

    assert topo.staleness_threshold_sec == 4.0
    assert topo.fusion_tick_sec == 0.5
    assert topo.density_threshold == 3.0
    assert len(topo.all_cameras()) == 3
    assert len(topo.all_edges()) == 2

    cam_a = topo.get_camera("CAM_A")
    assert cam_a is not None
    assert cam_a.name == "Gate A"
    assert cam_a.corridor_capacity_pax_min == 450.0
    assert cam_a.position.x == 100.0
    assert cam_a.position.y == 150.0
    assert cam_a.clock_offset_sec == 1.2

    # Upstream of CAM_C should be CAM_A (22.5s) and CAM_B (18.0s)
    upstream_c = topo.upstream_of("CAM_C")
    assert len(upstream_c) == 2
    assert ("CAM_A", 22.5) in upstream_c
    assert ("CAM_B", 18.0) in upstream_c

    # Downstream of CAM_A should be CAM_C (22.5s)
    downstream_a = topo.downstream_of("CAM_A")
    assert downstream_a == [("CAM_C", 22.5)]

    # Serialized dict matches structure
    d = topo.to_dict()
    assert "cameras" in d
    assert "edges" in d
    assert d["cameras"]["CAM_C"]["name"] == "Junction"


def test_chain_example_topology():
    import os
    example_path = os.path.join(os.path.dirname(__file__), "..", "configs", "topology.chain.example.yaml")
    assert os.path.exists(example_path), "configs/topology.chain.example.yaml must exist"

    topo = CameraTopology(config_path=example_path)
    assert len(topo.all_cameras()) == 3
    assert len(topo.all_edges()) == 2

    # CCTV1 -> CCTV2 -> CCTV3
    assert topo.upstream_of("CCTV1") == []
    assert topo.downstream_of("CCTV1") == [("CCTV2", 25.0)]

    assert topo.upstream_of("CCTV2") == [("CCTV1", 25.0)]
    assert topo.downstream_of("CCTV2") == [("CCTV3", 30.0)]

    assert topo.upstream_of("CCTV3") == [("CCTV2", 30.0)]
    assert topo.downstream_of("CCTV3") == []


def test_build_topology_from_route_chain():
    from topology.graph import build_topology_from_route
    cams = [
        {"camera_id": "C1", "name": "Start", "corridor_capacity_pax_min": 400.0},
        {"camera_id": "C2", "name": "Middle", "corridor_capacity_pax_min": 350.0},
        {"camera_id": "C3", "name": "End", "corridor_capacity_pax_min": 300.0},
    ]
    edges = [
        {"from": "C1", "to": "C2", "travel_time_sec": 20.0},
        {"from": "C2", "to": "C3", "travel_time_sec": 25.0},
    ]
    res = build_topology_from_route(cams, edges)
    assert len(res["cameras"]) == 3
    assert len(res["edges"]) == 2

    # In a chain, X positions increase monotonically with layer depth
    pos1 = res["cameras"]["C1"]["position"]
    pos2 = res["cameras"]["C2"]["position"]
    pos3 = res["cameras"]["C3"]["position"]
    assert pos1["x"] < pos2["x"] < pos3["x"]
    assert pos1["y"] == pos2["y"] == pos3["y"]  # Centered horizontally


def test_build_topology_from_route_merge():
    from topology.graph import build_topology_from_route
    cams = [
        {"camera_id": "C1", "name": "Gate 1"},
        {"camera_id": "C2", "name": "Gate 2"},
        {"camera_id": "C3", "name": "Merge Point"},
    ]
    edges = [
        {"from": "C1", "to": "C3", "travel_time_sec": 15.0},
        {"from": "C2", "to": "C3", "travel_time_sec": 20.0},
    ]
    res = build_topology_from_route(cams, edges)
    pos1 = res["cameras"]["C1"]["position"]
    pos2 = res["cameras"]["C2"]["position"]
    pos3 = res["cameras"]["C3"]["position"]

    # C1 and C2 are in layer 0 (same X, distinct Y)
    assert pos1["x"] == pos2["x"]
    assert pos1["y"] != pos2["y"]
    # C3 is in layer 1 (greater X, centered Y)
    assert pos3["x"] > pos1["x"]


def test_build_topology_from_route_split():
    from topology.graph import build_topology_from_route
    cams = [
        {"camera_id": "C1", "name": "Ghat Exit"},
        {"camera_id": "C2", "name": "North Path"},
        {"camera_id": "C3", "name": "South Path"},
    ]
    edges = [
        {"from": "C1", "to": "C2", "travel_time_sec": 10.0},
        {"from": "C1", "to": "C3", "travel_time_sec": 12.0},
    ]
    res = build_topology_from_route(cams, edges)
    pos1 = res["cameras"]["C1"]["position"]
    pos2 = res["cameras"]["C2"]["position"]
    pos3 = res["cameras"]["C3"]["position"]

    # C1 is at root layer
    assert pos1["x"] < pos2["x"]
    # C2 and C3 are at depth 1 (same X, vertically separated)
    assert pos2["x"] == pos3["x"]
    assert pos2["y"] != pos3["y"]


def test_build_topology_from_route_validations():
    from topology.graph import build_topology_from_route

    # Duplicate camera ID
    with pytest.raises(ValueError, match="Duplicate camera ID"):
        build_topology_from_route([{"camera_id": "A"}, {"camera_id": "A"}], [])

    # Edge referencing unknown camera
    with pytest.raises(ValueError, match="unknown target"):
        build_topology_from_route([{"camera_id": "A"}], [{"from": "A", "to": "B", "travel_time_sec": 10.0}])

    # Negative / zero travel time
    with pytest.raises(ValueError, match="strictly positive"):
        build_topology_from_route(
            [{"camera_id": "A"}, {"camera_id": "B"}],
            [{"from": "A", "to": "B", "travel_time_sec": 0.0}],
        )

    # Cycle detection
    with pytest.raises(ValueError, match="Cycle detected"):
        build_topology_from_route(
            [{"camera_id": "A"}, {"camera_id": "B"}, {"camera_id": "C"}],
            [
                {"from": "A", "to": "B", "travel_time_sec": 10.0},
                {"from": "B", "to": "C", "travel_time_sec": 10.0},
                {"from": "C", "to": "A", "travel_time_sec": 10.0},
            ],
        )




def test_route_builder_clock_offset_survives_round_trip():
    """Per-clip start skew must reach the metric store, not be dropped in transit.

    `clock_offset_sec` is what corrects clips that do not start together. The
    field was plumbed through CameraNode and MetricStore.update() but the Route
    Builder never sent it, so mapped clips were always treated as perfectly
    simultaneous with no way to say otherwise.
    """
    from topology.graph import build_topology_from_route

    cams = [
        {"camera_id": "C1", "name": "Start", "clock_offset_sec": 0.0},
        {"camera_id": "C2", "name": "Middle", "clock_offset_sec": 12.5},
        # Negative is legitimate: this clip started before the reference.
        {"camera_id": "C3", "name": "End", "clock_offset_sec": -3.25},
    ]
    edges = [
        {"from": "C1", "to": "C2", "travel_time_sec": 20.0},
        {"from": "C2", "to": "C3", "travel_time_sec": 25.0},
    ]
    built = build_topology_from_route(cams, edges)
    assert built["cameras"]["C2"]["clock_offset_sec"] == 12.5
    assert built["cameras"]["C3"]["clock_offset_sec"] == -3.25

    # Survives load into the graph and serialisation back out for the UI.
    topo = CameraTopology(config_path=None)
    topo.update_from_dict(built)
    assert topo.get_camera("C2").clock_offset_sec == 12.5
    assert topo.get_camera("C3").clock_offset_sec == -3.25
    assert topo.to_dict()["cameras"]["C2"]["clock_offset_sec"] == 12.5
