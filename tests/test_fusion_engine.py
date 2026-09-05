"""
Synthetic time-series unit tests for FusionEngine rules, lead times, and alerts.
Validates the cross-camera fusion math without any video file dependency.
"""

import time
import pytest
from topology.graph import CameraTopology
from topology.metric_store import MetricStore
from topology.fusion_engine import FusionEngine


@pytest.fixture
def test_setup():
    """Create a configured test topology, metric store, and fusion engine."""
    topo_data = {
        "staleness_threshold_sec": 5.0,
        "fusion_tick_sec": 1.0,
        "density_threshold": 2.5,
        "cameras": {
            "CCTV1": {"name": "Gate A", "corridor_capacity_pax_min": 400.0, "position": {"x": 100, "y": 100}},
            "CCTV2": {"name": "Gate B", "corridor_capacity_pax_min": 350.0, "position": {"x": 100, "y": 300}},
            "CCTV3": {"name": "Merge Point", "corridor_capacity_pax_min": 500.0, "position": {"x": 400, "y": 200}},
        },
        "edges": [
            {"from": "CCTV1", "to": "CCTV3", "travel_time_sec": 25.0},
            {"from": "CCTV2", "to": "CCTV3", "travel_time_sec": 20.0},
        ],
    }

    topo = CameraTopology()
    topo.update_from_dict(topo_data)
    store = MetricStore()
    engine = FusionEngine(topology=topo, metric_store=store)
    return topo, store, engine


def test_bottleneck_predicted_fires_at_correct_lead_time(test_setup):
    topo, store, engine = test_setup
    now_ms = 1700000050000  # t = 50s
    wall_now = time.time()

    # CCTV1: flow 320 pax/min at t = 25s (now_ms - 25000)
    store.update(
        camera_id="CCTV1",
        flow_rate_pax_min=320.0, flow_is_calibrated=True,
        explicit_epoch_ms=now_ms - 25000,
    )
    # CCTV2: flow 260 pax/min at t = 30s (now_ms - 20000)
    store.update(
        camera_id="CCTV2",
        flow_rate_pax_min=260.0, flow_is_calibrated=True,
        explicit_epoch_ms=now_ms - 20000,
    )
    # CCTV3: baseline target metrics
    store.update(
        camera_id="CCTV3",
        density=1.0,
        flow_rate_pax_min=100.0, flow_is_calibrated=True,
        explicit_epoch_ms=now_ms,
    )

    alerts = engine.evaluate_step(current_time_epoch_ms=now_ms, wall_now=wall_now)

    # Combined inflow = 320 + 260 = 580 pax/min > 500 pax/min capacity
    assert len(alerts) >= 1
    bottleneck_alert = next((a for a in alerts if a.level == "BOTTLENECK_PREDICTED"), None)
    assert bottleneck_alert is not None
    assert bottleneck_alert.camera_id == "CCTV3"
    assert bottleneck_alert.target_name == "Merge Point"
    assert bottleneck_alert.predicted_inflow == 580.0
    assert bottleneck_alert.target_capacity == 500.0
    assert bottleneck_alert.lead_time_sec == 20.0  # min(25s, 20s)
    assert set(bottleneck_alert.source_cameras) == {"CCTV1", "CCTV2"}
    assert "Gate A + Gate B" in bottleneck_alert.detail
    assert "580 pax/min" in bottleneck_alert.detail
    assert "500 pax/min" in bottleneck_alert.detail


def test_crush_risk_rising_rule(test_setup):
    topo, store, engine = test_setup
    now_ms = 1700000050000
    wall_now = time.time()

    # Upstream combined flow > 0.7 * 500 = 350 pax/min (e.g. 200 + 190 = 390 pax/min)
    store.update(camera_id="CCTV1", flow_rate_pax_min=200.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 25000)
    store.update(camera_id="CCTV2", flow_rate_pax_min=190.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 20000)

    # Target CCTV3 has critical density >= 2.5 pax/m²
    store.update(camera_id="CCTV3", density=2.8, flow_rate_pax_min=150.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms)

    alerts = engine.evaluate_step(current_time_epoch_ms=now_ms, wall_now=wall_now)

    crush_alert = next((a for a in alerts if a.level == "CRUSH_RISK_RISING"), None)
    assert crush_alert is not None
    assert crush_alert.camera_id == "CCTV3"
    assert crush_alert.current_density == 2.8
    assert crush_alert.predicted_inflow == 390.0
    assert "2.80 pax/m²" in crush_alert.detail


def test_no_alert_when_inflow_below_capacity(test_setup):
    topo, store, engine = test_setup
    now_ms = 1700000050000
    wall_now = time.time()

    # Inflow well below capacity: 100 + 80 = 180 pax/min vs 500 capacity
    store.update(camera_id="CCTV1", flow_rate_pax_min=100.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 25000)
    store.update(camera_id="CCTV2", flow_rate_pax_min=80.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 20000)
    store.update(camera_id="CCTV3", density=0.8, flow_rate_pax_min=150.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms)

    alerts = engine.evaluate_step(current_time_epoch_ms=now_ms, wall_now=wall_now)
    assert len(alerts) == 0
    assert len(engine.get_active_alerts()) == 0


def test_stale_camera_skipped(test_setup):
    topo, store, engine = test_setup
    now_ms = 1700000050000

    # CCTV1 updated freshly
    store.update(camera_id="CCTV1", flow_rate_pax_min=300.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 25000)
    # CCTV2 is stale (> 5s since received)
    store.update(camera_id="CCTV2", flow_rate_pax_min=300.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 20000)
    store._latest["CCTV2"].received_at = time.time() - 10.0  # Force stale

    store.update(camera_id="CCTV3", density=0.5, explicit_epoch_ms=now_ms)

    # CCTV2 is skipped due to staleness, so only CCTV1 contributes (300 < 500 capacity)
    alerts = engine.evaluate_step(current_time_epoch_ms=now_ms)
    assert len(alerts) == 0


def test_alert_deduplication_and_clearing(test_setup):
    topo, store, engine = test_setup
    now_ms = 1700000050000

    # Step 1: High inflow triggers alert
    store.update(camera_id="CCTV1", flow_rate_pax_min=350.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 25000)
    store.update(camera_id="CCTV2", flow_rate_pax_min=250.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms - 20000)
    store.update(camera_id="CCTV3", density=1.0, explicit_epoch_ms=now_ms)

    engine.evaluate_step(current_time_epoch_ms=now_ms, wall_now=100.0)
    assert len(engine.get_active_alerts()) == 1
    initial_id = engine.get_active_alerts()[0].id

    # Step 2: Next tick, still high inflow -> same alert updated, NOT duplicated
    engine.evaluate_step(current_time_epoch_ms=now_ms + 1000, wall_now=101.0)
    active = engine.get_active_alerts()
    assert len(active) == 1
    assert active[0].id == initial_id
    assert active[0].updated_at == 101.0

    # Step 3: Inflow subsides at upstream cameras at t = 52s
    store.update(camera_id="CCTV1", flow_rate_pax_min=100.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms + 2000)
    store.update(camera_id="CCTV2", flow_rate_pax_min=100.0, flow_is_calibrated=True, explicit_epoch_ms=now_ms + 2000)

    # At t = 80s (30s later, after 25s travel time), the reduced inflow reaches Merge Point
    engine.evaluate_step(current_time_epoch_ms=now_ms + 30000, wall_now=130.0)
    assert len(engine.get_active_alerts()) == 0


# ======================================================================
# Regression tests for defects found in review
# ======================================================================

def test_uncalibrated_flow_does_not_produce_a_capacity_alert(test_setup):
    """
    `corridor_capacity_pax_min` is a physical survey figure. An uncalibrated
    camera cannot produce pax/min -- without the real width of the counting
    line its "flow" is a count times an arbitrary constant. Comparing the two
    yields a confident, wrongly-scaled safety alert, which is worse than none
    because it looks authoritative.
    """
    topo, store, engine = test_setup
    now_ms = 1700000050000
    # Same magnitudes as the bottleneck test, but NOT calibrated.
    store.update(camera_id="CCTV1", flow_rate_pax_min=400.0,
                 explicit_epoch_ms=now_ms - 25000, flow_is_calibrated=False)
    store.update(camera_id="CCTV2", flow_rate_pax_min=300.0,
                 explicit_epoch_ms=now_ms - 20000, flow_is_calibrated=False)

    engine.evaluate_step(current_time_epoch_ms=now_ms, wall_now=time.time())

    assert engine.get_active_alerts() == [], \
        "uncalibrated flow must never be compared against a pax/min capacity"
    status = engine.get_forecast_status("CCTV3")
    assert status.get("blocked") == "uncalibrated_flow", \
        "and the refusal must be visible, not silent"


def test_missing_upstream_marks_forecast_incomplete(test_setup):
    """
    A forecast built from only some upstream sources UNDER-estimates inflow
    and therefore SUPPRESSES the alert. That is the dangerous direction, and
    it used to be silent: `has_stale_source` was computed and never read.
    """
    topo, store, engine = test_setup
    now_ms = 1700000050000
    # Only CCTV1 reports; CCTV2 is absent entirely.
    store.update(camera_id="CCTV1", flow_rate_pax_min=300.0,
                 explicit_epoch_ms=now_ms - 25000, flow_is_calibrated=True)

    engine.evaluate_step(current_time_epoch_ms=now_ms, wall_now=time.time())

    status = engine.get_forecast_status("CCTV3")
    assert status["complete"] is False
    assert "Gate B" in status["missing"], \
        "the missing source must be named so an operator knows what is unseen"


def test_no_usable_upstream_reports_none_not_zero(test_setup):
    """
    "We cannot forecast this camera" and "we forecast no inflow" are different
    claims. Zero reads as safe; None reads as unknown, which is the truth.
    """
    topo, store, engine = test_setup
    engine.evaluate_step(current_time_epoch_ms=1700000050000, wall_now=time.time())
    assert engine.get_predicted_inflow("CCTV3") is None


def test_accumulation_fires_on_terminal_camera(test_setup):
    """
    Terminal cameras (e.g. CCTV3 in the shipped merge topology, or a dead-end
    ghat like Ram Kund) have no downstream cameras, but the corridor feeding
    them MUST conserve people.

    Inflow (300) is below capacity (500), so no threshold bottleneck fires,
    yet 200 people/min accumulate between the upstream gates and CCTV3.
    """
    topo, store, engine = test_setup  # default topology: CCTV1 -> CCTV3, CCTV2 -> CCTV3 (terminal)
    base = 1700000050000

    # Seven minutes of 200 (CCTV1) + 100 (CCTV2) = 300 in vs 100 out at CCTV3
    for i in range(7):
        t = base + i * 60000
        store.update(camera_id="CCTV1", flow_rate_pax_min=200.0,
                     explicit_epoch_ms=t - 25000, flow_is_calibrated=True)
        store.update(camera_id="CCTV2", flow_rate_pax_min=100.0,
                     explicit_epoch_ms=t - 20000, flow_is_calibrated=True)
        store.update(camera_id="CCTV3", flow_rate_pax_min=100.0,
                     explicit_epoch_ms=t, flow_is_calibrated=True)
        engine.evaluate_step(current_time_epoch_ms=t, wall_now=time.time())

    alerts = engine.get_active_alerts()
    levels = {a.level for a in alerts}
    assert "BOTTLENECK_PREDICTED" not in levels, \
        "inflow (300) is below capacity (500) - the threshold rule must stay quiet"
    assert "ACCUMULATION_RISING" in levels, \
        "terminal camera must compute accumulation for the feeding segment"
    acc_alert = next(a for a in alerts if a.level == "ACCUMULATION_RISING")
    assert acc_alert.camera_id == "CCTV3"
    assert "accumulated between Gate A + Gate B and Merge Point" in acc_alert.detail


def test_accumulation_not_reported_when_target_metrics_missing(test_setup):
    """When target camera has no metrics, outflow cannot be measured.
    A blind spot must not manufacture an accumulation alert."""
    topo, store, engine = test_setup
    base = 1700000050000
    for i in range(5):
        t = base + i * 60000
        store.update(camera_id="CCTV1", flow_rate_pax_min=300.0,
                     explicit_epoch_ms=t - 25000, flow_is_calibrated=True)
        store.update(camera_id="CCTV2", flow_rate_pax_min=100.0,
                     explicit_epoch_ms=t - 20000, flow_is_calibrated=True)
        # CCTV3 is never updated -> missing target metrics
        engine.evaluate_step(current_time_epoch_ms=t, wall_now=time.time())
    levels = {a.level for a in engine.get_active_alerts()}
    assert "ACCUMULATION_RISING" not in levels
    assert engine.get_forecast_status("CCTV3").get("accumulation") == "missing_target_metrics"


def test_accumulation_not_reported_when_target_uncalibrated(test_setup):
    """When target camera flow is uncalibrated, outflow is not in real pax/min
    and cannot be conserved against physical inflow."""
    topo, store, engine = test_setup
    base = 1700000050000
    for i in range(5):
        t = base + i * 60000
        store.update(camera_id="CCTV1", flow_rate_pax_min=300.0,
                     explicit_epoch_ms=t - 25000, flow_is_calibrated=True)
        store.update(camera_id="CCTV2", flow_rate_pax_min=100.0,
                     explicit_epoch_ms=t - 20000, flow_is_calibrated=True)
        # CCTV3 updated with uncalibrated flow
        store.update(camera_id="CCTV3", flow_rate_pax_min=100.0,
                     explicit_epoch_ms=t, flow_is_calibrated=False)
        engine.evaluate_step(current_time_epoch_ms=t, wall_now=time.time())
    levels = {a.level for a in engine.get_active_alerts()}
    assert "ACCUMULATION_RISING" not in levels
    assert engine.get_forecast_status("CCTV3").get("accumulation") == "uncalibrated_target_flow"
