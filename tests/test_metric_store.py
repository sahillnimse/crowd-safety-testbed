"""Unit tests for MetricStore, clock-offset correction, and time-series history."""

import time
from topology.metric_store import MetricStore


def test_metric_store_update_and_get():
    store = MetricStore()
    snap = store.update(
        camera_id="CCTV1",
        density=1.5,
        flow_rate_pax_min=320.0,
        dominant_direction_vector=(0.8, 0.6),
        crush_risk_score=0.25,
        person_count=42,
        raw_timestamp_sec=10.0,
        stream_start_epoch_ms=1700000000000,
        clock_offset_sec=0.0,
    )

    assert snap.camera_id == "CCTV1"
    assert snap.density == 1.5
    assert snap.flow_rate_pax_min == 320.0
    assert snap.person_count == 42
    # 1700000000000 + 10.0 * 1000 = 1700000010000
    assert snap.timestamp_epoch_ms == 1700000010000

    latest = store.get_latest("CCTV1")
    assert latest is not None
    assert latest.flow_rate_pax_min == 320.0


def test_metric_store_clock_offset_correction():
    store = MetricStore()
    # Camera with +2.5s clock offset
    snap1 = store.update(
        camera_id="CCTV1",
        raw_timestamp_sec=5.0,
        stream_start_epoch_ms=1700000000000,
        clock_offset_sec=2.5,
    )
    # 1700000000000 + 5000 + 2500 = 1700000007500
    assert snap1.timestamp_epoch_ms == 1700000007500

    # Camera with -1.0s clock offset
    snap2 = store.update(
        camera_id="CCTV2",
        raw_timestamp_sec=5.0,
        stream_start_epoch_ms=1700000000000,
        clock_offset_sec=-1.0,
    )
    # 1700000000000 + 5000 - 1000 = 1700000004000
    assert snap2.timestamp_epoch_ms == 1700000004000


def test_metric_store_historical_lookup():
    store = MetricStore()
    base_ms = 1700000000000

    # Simulate sequence of time snapshots
    for t_sec, flow in [(0, 100), (10, 200), (20, 300), (30, 400), (40, 500)]:
        store.update(
            camera_id="CCTV1",
            flow_rate_pax_min=float(flow),
            raw_timestamp_sec=float(t_sec),
            stream_start_epoch_ms=base_ms,
        )

    # Query flow at t = 20s (base_ms + 20000)
    flow_at_20 = store.get_historical_flow_rate("CCTV1", target_epoch_ms=base_ms + 20000)
    assert flow_at_20 == 300.0

    # Query flow at t = 29s (closest is t = 30s)
    flow_at_29 = store.get_historical_flow_rate("CCTV1", target_epoch_ms=base_ms + 29000)
    assert flow_at_29 == 400.0


def test_metric_store_staleness():
    store = MetricStore()
    # Non-existent camera is stale
    assert store.is_stale("UNKNOWN", threshold_sec=5.0) is True

    # Freshly updated camera is not stale
    store.update("CCTV1", flow_rate_pax_min=150.0)
    assert store.is_stale("CCTV1", threshold_sec=5.0) is False


# ======================================================================
# Regression tests for defects found in review
# ======================================================================

def test_historical_lookup_returns_none_rather_than_the_current_value():
    """
    The lookup used to fall through to `self._latest` whenever nothing sat
    within tolerance -- unbounded, despite a comment claiming "slightly out of
    range". A request for "30 seconds ago" silently returned "right now", so
    the fusion engine compared the present against the present while reporting
    a travel-time offset it had never applied.
    """
    from topology.metric_store import MetricStore
    ms = MetricStore()
    now = 1700000000000
    ms.update("C1", flow_rate_pax_min=100.0, explicit_epoch_ms=now - 30000)
    ms.update("C1", flow_rate_pax_min=900.0, explicit_epoch_ms=now)

    assert ms.get_historical_flow_rate("C1", now - 30000) == 100.0
    assert ms.get_historical_flow_rate("C1", now - 600000) is None, \
        "no sample near the requested time must report a miss, not substitute now"


def test_historical_snapshot_carries_calibration_provenance():
    """A flow rate is only comparable to a capacity if it was calibrated when
    recorded, so that fact must travel with the sample."""
    from topology.metric_store import MetricStore
    ms = MetricStore()
    now = 1700000000000
    ms.update("C1", flow_rate_pax_min=100.0, explicit_epoch_ms=now,
              flow_is_calibrated=True, units="pax/min")
    snap = ms.get_historical_snapshot("C1", now)
    assert snap is not None and snap.flow_is_calibrated is True
    assert snap.units == "pax/min"


def test_reference_epoch_tracks_the_source_clock_not_the_wall_clock():
    """
    Recorded footage stamps samples on the VIDEO's timeline. A wall-clock
    "now" then sits hours from every sample in the buffer and misses them all.
    """
    from topology.metric_store import MetricStore
    ms = MetricStore()
    future = int(time.time() * 1000) + 3_600_000       # video an hour ahead
    ms.update("C1", flow_rate_pax_min=10.0, explicit_epoch_ms=future)
    assert ms.reference_epoch_ms() == future


def test_reference_epoch_is_none_when_empty():
    from topology.metric_store import MetricStore
    assert MetricStore().reference_epoch_ms() is None


# ======================================================================
# Persistence — telemetry must survive a restart
# ======================================================================

class TestMetricStorePersistence:
    """
    Job state was persisted but telemetry was not, so a restart erased every
    sample while the jobs that produced them survived. Route View then read
    "telemetry offline" for cameras monitored minutes earlier, and the fusion
    engine had no history for its time-shifted lookups until the buffers
    refilled — minutes of blindness after every restart.
    """

    def test_samples_survive_a_restart(self, tmp_path):
        from topology.metric_store import MetricStore
        path = str(tmp_path / "state.json")
        s1 = MetricStore(persist_path=path)
        for i in range(5):
            s1.update("CCTV1", flow_rate_pax_min=100.0 + i,
                      explicit_epoch_ms=1700000000000 + i * 1000,
                      flow_is_calibrated=True, units="pax/min")
        s1.flush(force=True)

        s2 = MetricStore(persist_path=path)      # fresh process
        assert s2.get_all_camera_ids() == ["CCTV1"]
        assert len(s2.get_history("CCTV1", window_sec=1e9)) == 5
        assert s2.get_latest("CCTV1").flow_rate_pax_min == 104.0

    def test_calibration_provenance_survives(self, tmp_path):
        """Units must not be lost across a restart — a restored sample whose
        provenance was dropped would be treated as uncalibrated (or worse,
        assumed calibrated) by the fusion gate."""
        from topology.metric_store import MetricStore
        path = str(tmp_path / "state.json")
        s1 = MetricStore(persist_path=path)
        s1.update("C", flow_rate_pax_min=1.0, flow_is_calibrated=True,
                  density_is_calibrated=True, units="pax/min")
        s1.flush(force=True)
        snap = MetricStore(persist_path=path).get_latest("C")
        assert snap.flow_is_calibrated is True
        assert snap.units == "pax/min"

    def test_time_shifted_lookup_works_after_restart(self, tmp_path):
        """The whole point of restoring history: the fusion engine's
        flow_rate(Ui, t - travel_time) query must find something."""
        from topology.metric_store import MetricStore
        path = str(tmp_path / "state.json")
        s1 = MetricStore(persist_path=path)
        for i in range(5):
            s1.update("C", flow_rate_pax_min=float(i),
                      explicit_epoch_ms=1700000000000 + i * 1000)
        s1.flush(force=True)
        assert MetricStore(persist_path=path).get_historical_flow_rate(
            "C", 1700000002000) == 2.0

    def test_a_camera_that_stopped_before_restart_is_still_stale(self, tmp_path):
        """
        Restoring history must NOT resurrect a dead camera as live. Samples
        keep their original received_at precisely so staleness stays honest.
        """
        import json
        from topology.metric_store import MetricStore
        path = str(tmp_path / "state.json")
        s1 = MetricStore(persist_path=path)
        s1.update("C", flow_rate_pax_min=1.0)
        s1.flush(force=True)

        d = json.loads(open(path, encoding="utf-8").read())
        for row in d["history"]["C"]:
            row["received_at"] -= 600          # stopped 10 minutes ago
        open(path, "w", encoding="utf-8").write(json.dumps(d))

        s2 = MetricStore(persist_path=path)
        assert len(s2.get_history("C", window_sec=1e9)) == 1, "history restored"
        assert s2.is_stale("C", threshold_sec=5.0) is True, \
            "a camera that stopped feeding must not come back as live"

    def test_corrupt_state_does_not_prevent_startup(self, tmp_path):
        from topology.metric_store import MetricStore
        path = str(tmp_path / "state.json")
        open(path, "w").write("{ not json")
        s = MetricStore(persist_path=path)     # must not raise
        assert s.get_all_camera_ids() == []

    def test_persistence_is_off_unless_asked(self):
        """Tests and library users must not touch shared on-disk state."""
        from topology.metric_store import MetricStore
        s = MetricStore()
        s.update("X", flow_rate_pax_min=1.0)
        assert s.persist_path is None
        assert s.flush(force=True) is False

    def test_write_is_atomic(self, tmp_path):
        """Temp-and-rename, so a crash mid-write cannot leave a truncated file
        that fails to parse on the next start."""
        from topology.metric_store import MetricStore
        path = str(tmp_path / "state.json")
        s = MetricStore(persist_path=path)
        s.update("C", flow_rate_pax_min=1.0)
        s.flush(force=True)
        assert not (tmp_path / "state.json.tmp").exists()
