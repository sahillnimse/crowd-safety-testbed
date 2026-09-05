"""
Unit tests for Route Session metric aggregation and fused HTML report generator.
"""

import json
import os
import pytest
from topology.graph import CameraTopology
from webapp.session_report import (
    aggregate_session_metrics,
    generate_session_report_html,
    build_session_report,
)


@pytest.fixture
def mock_topology():
    topo = CameraTopology(config_path="nonexistent.yaml")
    topo.update_from_dict({
        "cameras": {
            "CCTV1": {"name": "Gate A", "corridor_capacity_pax_min": 400.0, "position": {"x": 100, "y": 100}},
            "CCTV2": {"name": "Gate B", "corridor_capacity_pax_min": 350.0, "position": {"x": 100, "y": 300}},
            "CCTV3": {"name": "Merge Point", "corridor_capacity_pax_min": 500.0, "position": {"x": 400, "y": 200}},
        },
        "edges": [
            {"from": "CCTV1", "to": "CCTV3", "travel_time_sec": 25.0},
            {"from": "CCTV2", "to": "CCTV3", "travel_time_sec": 20.0},
        ],
    })
    return topo


@pytest.fixture
def mock_camera_summaries():
    return {
        "CCTV1": {
            "avg_density": 3.0,
            "avg_speed_px_frame": 2.5,
            "peak_speed_px_frame": 4.0,
            "specific_flow_current": 300.0,
            "avg_crowd_pressure": 0.45,
            "peak_crowd_pressure": 0.85,
            "strongest_compression": -1.2,
            "stop_go_score": 0.35,
            "oscillation_symmetry": 0.40,
            "pct_crush_risk": 8.5,
            "crush_event_count": 3,
            "pct_counterflow_people": 12.0,
            "counterflow_events_count": 2,
            "avg_directional_entropy": 1.45,
            "avg_velocity_variance": 0.60,
            "peak_crush_timestamp_sec": 45.0,
            "total_detections": 1200,
            "total_tracks": 150,
        },
        "CCTV2": {
            "avg_density": 2.0,
            "avg_speed_px_frame": 3.0,
            "peak_speed_px_frame": 4.5,
            "specific_flow_current": 250.0,
            "avg_crowd_pressure": 0.30,
            "peak_crowd_pressure": 0.50,
            "strongest_compression": -0.8,
            "stop_go_score": 0.20,
            "oscillation_symmetry": 0.25,
            "pct_crush_risk": 4.0,
            "crush_event_count": 1,
            "pct_counterflow_people": 6.0,
            "counterflow_events_count": 1,
            "avg_directional_entropy": 1.10,
            "avg_velocity_variance": 0.40,
            "peak_crush_timestamp_sec": 50.0,
            "total_detections": 900,
            "total_tracks": 110,
        },
        "CCTV3": {
            "avg_density": 4.5,
            "avg_speed_px_frame": 1.5,
            "peak_speed_px_frame": 3.0,
            "specific_flow_current": 480.0,
            "avg_crowd_pressure": 0.90,
            "peak_crowd_pressure": 1.40,
            "strongest_compression": -2.8,
            "stop_go_score": 0.75,
            "oscillation_symmetry": 0.80,
            "pct_crush_risk": 22.0,
            "crush_event_count": 8,
            "pct_counterflow_people": 18.0,
            "counterflow_events_count": 5,
            "avg_directional_entropy": 2.30,
            "avg_velocity_variance": 1.10,
            "peak_crush_timestamp_sec": 70.0,
            "total_detections": 2500,
            "total_tracks": 280,
        },
    }


def test_aggregate_session_metrics(mock_camera_summaries, mock_topology):
    agg = aggregate_session_metrics(mock_camera_summaries, topology=mock_topology)

    assert agg["camera_count"] == 3
    # Density is weighted avg (capacity weights: 400, 350, 500 = total 1250)
    # (3.0*400 + 2.0*350 + 4.5*500) / 1250 = (1200 + 700 + 2250) / 1250 = 4150 / 1250 = 3.32
    assert agg["avg_density"] == pytest.approx(3.32, 0.05)

    # Worst crowd pressure = max(0.85, 0.50, 1.40) = 1.40
    assert agg["max_crowd_pressure"] == 1.40

    # Worst divergence = min(-1.2, -0.8, -2.8) = -2.8
    assert agg["worst_divergence"] == -2.8

    # Max crush risk = 22.0%
    assert agg["max_crush_risk_pct"] == 22.0
    assert agg["total_crush_events"] == 12  # 3 + 1 + 8

    # Specific flow sum = 300 + 250 + 480 = 1030
    # Renamed and re-defined: route flow is the BOTTLENECK, not a sum.
    # Summing per-camera flow along a route counts the same crowd once per
    # camera it walks past; what limits a route is its narrowest section.
    assert agg["bottleneck_specific_flow"] == min(
        c["specific_flow"] for c in agg["cameras"].values())
    assert "total_specific_flow" not in agg

    # Total detections
    assert agg["total_detections"] == 4600

    # Transit narratives
    assert len(agg["transit_narratives"]) == 2
    narratives = {n["source_cam"]: n for n in agg["transit_narratives"]}
    assert "CCTV1" in narratives
    assert narratives["CCTV1"]["travel_time_sec"] == 25.0
    assert narratives["CCTV1"]["target_cam"] == "CCTV3"


def test_generate_session_report_html(mock_camera_summaries, mock_topology):
    agg = aggregate_session_metrics(mock_camera_summaries, topology=mock_topology)
    manifest = {
        "session_name": "Kumbh_Route_Test",
        "created_at": "2026-09-02T12:00:00Z",
        "status": "done",
        "cameras": {
            "CCTV1": {"name": "Gate A"},
            "CCTV2": {"name": "Gate B"},
            "CCTV3": {"name": "Merge Point"},
        }
    }

    html = generate_session_report_html("Kumbh_Route_Test", agg, manifest)

    assert "<!DOCTYPE html>" in html
    assert "Kumbh_Route_Test" in html
    assert "Gate A" in html
    assert "Merge Point" in html
    assert "Route-Level Aggregated Metrics" in html
    assert "CCTV1" in html
    assert "CCTV3" in html


def test_build_session_report(tmp_path, mock_camera_summaries, mock_topology):
    # Set up synthetic session directory
    session_dir = tmp_path / "test_session"
    session_dir.mkdir()

    for cid, summary in mock_camera_summaries.items():
        cam_dir = session_dir / cid
        cam_dir.mkdir()
        with open(cam_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f)

    sum_path, rep_path = build_session_report(
        session_dir=str(session_dir),
        session_name="test_session",
        topology=mock_topology,
        manifest={"status": "done"},
    )

    assert os.path.exists(sum_path)
    assert os.path.exists(rep_path)

    with open(sum_path, "r", encoding="utf-8") as f:
        loaded_sum = json.load(f)
    assert loaded_sum["camera_count"] == 3
    assert loaded_sum["total_detections"] == 4600


# -----------------------------------------------------------------------------
# Regression tests for the 8 session_report.py defects (see docs/HANDOFF.md)
# -----------------------------------------------------------------------------

def test_uncalibrated_route_suppresses_capacity_utilization(mock_topology):
    """
    Defect 1: Report bypassed calibration gate and compared uncalibrated
    flow against pax/min capacity.
    When is_calibrated is False or omitted, capacity_utilization_pct must be None
    and status must be 'unmeasured'.
    """
    summaries = {
        "CCTV1": {
            "specific_flow_current": 300.0,
            "is_calibrated": False,
        },
        "CCTV3": {
            "specific_flow_current": 400.0,
            "is_calibrated": False,
        },
    }
    agg = aggregate_session_metrics(summaries, topology=mock_topology)
    narratives = {n["source_cam"]: n for n in agg["transit_narratives"]}
    c1_narrative = narratives["CCTV1"]

    assert c1_narrative["flow_is_calibrated"] is False
    assert c1_narrative["capacity_utilization_pct"] is None
    assert c1_narrative["status"] == "unmeasured"
    assert "Capacity utilisation NOT computed" in c1_narrative["summary_text"]
    assert "Corridor flow is UNCALIBRATED" in c1_narrative["summary_text"]
    assert agg["is_calibrated"] is False
    assert "CCTV1" in agg["uncalibrated_cameras"]
    assert "UNCALIBRATED" in agg["units_note"]


def test_calibrated_route_computes_capacity_utilization(mock_topology):
    """
    Defect 1: When calibrated, capacity utilization is legitimately computed.
    CCTV1 flow = 5.0 pax/sec = 300.0 pax/min.
    CCTV3 capacity = 500.0 pax/min.
    Utilization = 300 / 500 * 100 = 60.0% (nominal status).
    """
    summaries = {
        "CCTV1": {
            "specific_flow_gross_per_sec": 5.0,
            "is_calibrated": True,
        },
        "CCTV3": {
            "specific_flow_gross_per_sec": 4.0,
            "is_calibrated": True,
        },
    }
    agg = aggregate_session_metrics(summaries, topology=mock_topology)
    narratives = {n["source_cam"]: n for n in agg["transit_narratives"]}
    c1_narrative = narratives["CCTV1"]

    assert c1_narrative["flow_is_calibrated"] is True
    assert c1_narrative["source_flow_pax_min"] == 300.0
    assert c1_narrative["target_capacity"] == 500.0
    assert c1_narrative["capacity_utilization_pct"] == pytest.approx(60.0, 0.1)
    assert c1_narrative["status"] == "nominal"
    assert "60.0%" in c1_narrative["summary_text"]
    assert agg["is_calibrated"] is True
    assert len(agg["uncalibrated_cameras"]) == 0


def test_unmeasured_camera_does_not_dilute_density_weighted_mean(mock_topology):
    """
    Defect 2: 'or 0.0' erased 'not measured', causing unmeasured cameras to
    contribute 0.0 to a weighted mean and dragging the route density down.
    A camera without density must abstain from weighted mean, not vote zero.
    """
    # CCTV1: capacity 400, density 3.5
    # CCTV2: capacity 350, unmeasured density (None or missing)
    # Mean should be exactly 3.5, NOT (3.5 * 400 + 0 * 350) / 750 = 1.87
    summaries = {
        "CCTV1": {
            "avg_density": 3.5,
        },
        "CCTV2": {
            # avg_density omitted
        },
    }
    agg = aggregate_session_metrics(summaries, topology=mock_topology)
    assert agg["avg_density"] == pytest.approx(3.5, 0.01)
    assert agg["cameras"]["CCTV1"]["density"] == 3.5
    assert agg["cameras"]["CCTV2"]["density"] is None

    # When all cameras have unmeasured density, agg_density should be None
    summaries_all_none = {
        "CCTV1": {"avg_density": None},
        "CCTV2": {},
    }
    agg_none = aggregate_session_metrics(summaries_all_none, topology=mock_topology)
    assert agg_none["avg_density"] is None


def test_bottleneck_specific_flow_takes_minimum_not_sum(mock_topology):
    """
    Defect 3: Route flow was summed across cameras, triple-counting the crowd.
    Now bottleneck_specific_flow takes min() and total_specific_flow is removed.
    """
    summaries = {
        "CCTV1": {"specific_flow_gross_per_sec": 300.0},
        "CCTV2": {"specific_flow_gross_per_sec": 150.0},
        "CCTV3": {"specific_flow_gross_per_sec": 450.0},
    }
    agg = aggregate_session_metrics(summaries, topology=mock_topology)
    assert agg["bottleneck_specific_flow"] == 150.0
    assert "total_specific_flow" not in agg


def test_gross_flow_preferred_over_net(mock_topology):
    """
    Defect 4: abs() on net flow reported an empty corridor when two balanced
    streams cancel out. specific_flow_gross_per_sec must be preferred over
    specific_flow_current and specific_flow_net_per_sec.
    """
    # Case A: gross present alongside net and current
    summaries = {
        "CCTV1": {
            "specific_flow_gross_per_sec": 10.0,
            "specific_flow_current": 5.0,
            "specific_flow_net_per_sec": 0.2,
        },
    }
    agg = aggregate_session_metrics(summaries, topology=mock_topology)
    assert agg["cameras"]["CCTV1"]["specific_flow"] == 10.0

    # Case B: gross absent, fallback to specific_flow_current
    summaries_fallback_curr = {
        "CCTV1": {
            "specific_flow_current": 5.0,
            "specific_flow_net_per_sec": 0.2,
        },
    }
    agg_b = aggregate_session_metrics(summaries_fallback_curr, topology=mock_topology)
    assert agg_b["cameras"]["CCTV1"]["specific_flow"] == 5.0

    # Case C: only net present
    summaries_fallback_net = {
        "CCTV1": {
            "specific_flow_net_per_sec": 0.8,
        },
    }
    agg_c = aggregate_session_metrics(summaries_fallback_net, topology=mock_topology)
    assert agg_c["cameras"]["CCTV1"]["specific_flow"] == 0.8


def test_peak_density_captured_and_greater_equal_avg_density(mock_topology):
    """
    Defect 5: Density reported as mean only, hiding critical local hotspots.
    peak_density must be tracked and satisfy peak_density >= avg_density.
    """
    summaries = {
        "CCTV1": {"avg_density": 1.0},
        "CCTV2": {"avg_density": 1.0},
        "CCTV3": {"avg_density": 6.0},
    }
    agg = aggregate_session_metrics(summaries, topology=mock_topology)
    assert agg["peak_density"] == 6.0
    assert agg["avg_density"] < 6.0
    assert agg["peak_density"] >= agg["avg_density"]


def test_propagation_lag_consistency(mock_topology):
    """
    Defect 6: Propagation check collected timestamps and never compared them.
    observed_lag_sec and lag_consistent_with_topology must validate transit timing.
    Edge CCTV1 -> CCTV3 has travel_time_sec = 25.0.
    """
    # Case A: Consistent lag (30s lag for 25s travel time: -5 <= 30 - 25 <= 30 -> True)
    summaries_ok = {
        "CCTV1": {"peak_crush_timestamp_sec": 100.0},
        "CCTV3": {"peak_crush_timestamp_sec": 130.0},
    }
    agg_ok = aggregate_session_metrics(summaries_ok, topology=mock_topology)
    n_ok = [n for n in agg_ok["transit_narratives"] if n["source_cam"] == "CCTV1"][0]
    assert n_ok["observed_lag_sec"] == 30.0
    assert n_ok["lag_consistent_with_topology"] is True
    assert "consistent" in n_ok["summary_text"]

    # Case B: Inconsistent lag (downstream peaked before upstream: lag = -20s)
    summaries_bad = {
        "CCTV1": {"peak_crush_timestamp_sec": 100.0},
        "CCTV3": {"peak_crush_timestamp_sec": 80.0},
    }
    agg_bad = aggregate_session_metrics(summaries_bad, topology=mock_topology)
    n_bad = [n for n in agg_bad["transit_narratives"] if n["source_cam"] == "CCTV1"][0]
    assert n_bad["observed_lag_sec"] == -20.0
    assert n_bad["lag_consistent_with_topology"] is False
    assert "INCONSISTENT" in n_bad["summary_text"]

    # Case C: Missing timestamps -> lag not computed (None)
    summaries_missing = {
        "CCTV1": {"peak_crush_timestamp_sec": 0.0},
        "CCTV3": {"peak_crush_timestamp_sec": 0.0},
    }
    agg_missing = aggregate_session_metrics(summaries_missing, topology=mock_topology)
    n_missing = [n for n in agg_missing["transit_narratives"] if n["source_cam"] == "CCTV1"][0]
    assert n_missing["observed_lag_sec"] is None
    assert n_missing["lag_consistent_with_topology"] is None


def test_html_report_handles_none_values_safely(mock_topology):
    """
    Defect 7: HTML renderer crashed on None with TypeError in f-string formatting,
    and unmeasured cells were colored green.
    Must generate cleanly with None values, render 'n/a', never use literal 'None',
    and assign neutral styling class.
    """
    # Create session summary where all optional fields are None
    summaries = {
        "CCTV1": {
            "avg_density": None,
            "avg_speed_px_frame": None,
            "specific_flow_current": None,
            "avg_crowd_pressure": None,
            "strongest_compression": None,
            "pct_crush_risk": None,
            "pct_counterflow_people": None,
            "avg_directional_entropy": None,
        },
    }
    agg = aggregate_session_metrics(summaries, topology=mock_topology)
    manifest = {
        "session_name": "Null_Test",
        "status": "done",
        "cameras": {"CCTV1": {"name": "Gate A"}},
    }

    # Must not raise TypeError or ValueError
    html = generate_session_report_html("Null_Test", agg, manifest)

    # Must contain honest 'n/a' placeholders
    assert "n/a" in html
    # Must never render Python 'None' literally to the user
    assert "None" not in html
    # Unmeasured crush risk / divergence must get 'neutral' CSS class, never 'good'
    assert 'class="neutral"' in html


def test_html_report_calibration_banner(mock_topology):
    """
    Defect 8: No calibration banner on uncalibrated report.
    Uncalibrated sessions must display an orange UNCALIBRATED ROUTE banner;
    calibrated sessions must omit it.
    """
    manifest = {"session_name": "Banner_Test", "status": "done", "cameras": {}}

    # Uncalibrated session
    uncal_summaries = {"CCTV1": {"is_calibrated": False}}
    agg_uncal = aggregate_session_metrics(uncal_summaries, topology=mock_topology)
    html_uncal = generate_session_report_html("Banner_Test", agg_uncal, manifest)
    assert '<div class="calib-banner">' in html_uncal
    assert "UNCALIBRATED ROUTE" in html_uncal
    assert "CCTV1" in html_uncal

    # Fully calibrated session
    cal_summaries = {"CCTV1": {"is_calibrated": True}}
    agg_cal = aggregate_session_metrics(cal_summaries, topology=mock_topology)
    html_cal = generate_session_report_html("Banner_Test", agg_cal, manifest)
    assert '<div class="calib-banner">' not in html_cal
    assert "UNCALIBRATED ROUTE" not in html_cal

