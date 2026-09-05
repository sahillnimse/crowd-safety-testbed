"""
Unit tests for the calibration loader and the alert threshold engine.

Covers two critical paths that previously had no tests at all:

1. CameraCalibration.from_yaml_block() — the empty/missing homography tiers
   (exactly the shapes configs/crowd_flow.yaml carries today), point-count
   validation, and a pixel->world->pixel round trip on a known homography.

2. AlertEngine / _ThresholdState — hysteresis band arithmetic (including the
   documented positive-threshold-on-fire-low bug this arithmetic exists to
   prevent), min-duration gating, priming reset, the uncalibrated-camera
   gate that disables physical-unit thresholds, per-threshold state keying,
   the vehicle-in-pedestrian-zone shortcut, and NaN handling.

Fabricated data only — no video, no detectors, no torch.

Run:  python -m pytest tests/test_calibration_and_alerts.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.crowd_flow.ground_plane import CameraCalibration
from models.crowd_flow.zones import (
    DENSE_FLOW_ALERT_LABELS,
    AlertEngine,
    AlertSeverity,
    Zone,
    ZoneThresholds,
    _ThresholdState,
)

# Unit square in pixels mapped to itself in metres -> H is ~identity, which
# makes the round-trip assertions exact without inventing survey data.
_SQUARE_PTS = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def make_zone(name="z", **thr_kwargs) -> Zone:
    return Zone(
        name=name,
        polygon=[(0.0, 0.0), (16.0, 0.0), (16.0, 16.0)],
        thresholds=ZoneThresholds(**thr_kwargs),
    )


def metrics(**overrides) -> SimpleNamespace:
    """ZoneMetrics stand-in carrying every attribute _METRIC_DEFS reads."""
    base = dict(
        mean_speed=1.0,            # healthy walking px/frame
        mean_divergence=0.0,       # neutral
        mean_curl=0.0,
        counterflow_score=0.0,
        turbulence_index=0.0,
        crowd_pressure=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ======================================================================
# 1. CameraCalibration.from_yaml_block
# ======================================================================

class TestFromYamlBlock:
    def test_missing_homography_key_is_uncalibrated(self, caplog):
        with caplog.at_level(logging.WARNING, logger="models.crowd_flow.ground_plane"):
            cal = CameraCalibration.from_yaml_block("cam", {"zones": []})
        assert cal.is_calibrated is False
        assert any("no homography" in r.message for r in caplog.records)

    def test_empty_homography_block_is_uncalibrated(self):
        cal = CameraCalibration.from_yaml_block("cam", {"homography": {}})
        assert cal.is_calibrated is False

    def test_empty_point_lists_are_uncalibrated(self, caplog):
        # The exact shape the crowd_ralley block ships today.
        block = {"image_points": [], "world_points_m": []}
        with caplog.at_level(logging.WARNING, logger="models.crowd_flow.ground_plane"):
            cal = CameraCalibration.from_yaml_block("crowd_ralley",
                                                    {"homography": block})
        assert cal.is_calibrated is False
        assert any("Treating as uncalibrated" in r.message for r in caplog.records)

    def test_world_points_only_is_uncalibrated(self):
        block = {"image_points": _SQUARE_PTS, "world_points_m": []}
        assert CameraCalibration.from_yaml_block("cam",
                                                 {"homography": block}).is_calibrated is False

    def test_valid_points_calibrate_and_round_trip(self):
        block = {"image_points": _SQUARE_PTS, "world_points_m": _SQUARE_PTS}
        cal = CameraCalibration.from_yaml_block("cam", {"homography": block})
        assert cal.is_calibrated is True
        assert cal.H is not None
        # Identity mapping: pixel (2, 3) is ground (2 m, 3 m), both ways.
        X, Y = cal.pixel_to_world(2.0, 3.0)
        assert abs(X - 2.0) < 1e-6 and abs(Y - 3.0) < 1e-6
        x, y = cal.world_to_pixel(2.0, 3.0)
        assert abs(x - 2.0) < 1e-6 and abs(y - 3.0) < 1e-6

    def test_units_flip_with_calibration(self):
        uncal = CameraCalibration.from_yaml_block("cam", {})
        block = {"image_points": _SQUARE_PTS, "world_points_m": _SQUARE_PTS}
        cal = CameraCalibration.from_yaml_block("cam", {"homography": block})
        assert cal.speed_units != uncal.speed_units
        assert "px" in uncal.speed_units.lower()

    def test_fewer_than_four_pairs_raises(self):
        with pytest.raises(ValueError, match="At least 4 point pairs"):
            CameraCalibration.from_points("cam", _SQUARE_PTS[:3],
                                          _SQUARE_PTS[:3])

    def test_mismatched_lengths_raise(self):
        # Both sides >= 4 so the count floor passes and the equality guard
        # is what rejects the call.
        with pytest.raises(ValueError, match="same length"):
            CameraCalibration.from_points("cam", _SQUARE_PTS,
                                          _SQUARE_PTS + [[15.0, 15.0]])


# ======================================================================
# 2. _ThresholdState — hysteresis band + duration gating
# ======================================================================

class TestThresholdState:
    def test_band_higher_is_worse(self):
        s = _ThresholdState(0.20, 0.20, 0.0, 10.0, higher_is_worse=True)
        assert s.clear == pytest.approx(0.16)

    def test_band_fire_on_low_negative(self):
        # Divergence -1.5 fires; clear must be NEARER ZERO (-1.2), i.e. away
        # from severity.
        s = _ThresholdState(-1.5, 0.20, 0.0, 10.0, higher_is_worse=False)
        assert s.clear == pytest.approx(-1.2)

    def test_band_positive_threshold_fire_on_low(self):
        # THE regression documented on _ThresholdState: speed_low_ms=0.4 with
        # the old fire*(1-frac) formula produced clear=0.32 < fire, so the
        # whole band sat inside the breach region and hysteresis was dead.
        s = _ThresholdState(0.40, 0.20, 0.0, 10.0, higher_is_worse=False)
        assert s.clear > s.fire
        assert s.clear == pytest.approx(0.48)

    def test_min_duration_gate_exact_frame_count(self):
        # fps=10, min_duration_sec=2.0 -> exactly 20 consecutive breaches.
        s = _ThresholdState(0.20, 0.0, 2.0, 10.0, higher_is_worse=True)
        for _ in range(19):
            assert s.update(0.30) is False
        assert s.update(0.30) is True          # frame 20 emits

    def test_priming_resets_on_single_noisy_frame(self):
        s = _ThresholdState(0.20, 0.0, 2.0, 10.0, higher_is_worse=True)
        for _ in range(19):
            s.update(0.30)
        s.update(0.10)                          # noise dips below fire...
        for _ in range(19):                     # ...and the clock restarts
            assert s.update(0.30) is False
        assert s.update(0.30) is True

    def test_active_holds_between_clear_and_fire(self):
        s = _ThresholdState(0.20, 0.20, 2.0, 10.0, higher_is_worse=True)
        for _ in range(s.min_frames):
            s.update(0.30)
        assert s.active
        # 0.18 is below fire but above clear: no re-breach needed, no clear.
        assert s.update(0.18) is True

    def test_crossing_below_clear_deactivates(self):
        s = _ThresholdState(0.20, 0.20, 2.0, 10.0, higher_is_worse=True)
        for _ in range(s.min_frames):
            s.update(0.30)
        assert s.update(0.15) is False          # strictly under clear=0.16


# ======================================================================
# 3. AlertEngine — gates, keying, shortcuts
# ======================================================================

class TestAlertEngine:
    FPS = 50.0
    MIN_DUR = 0.1                    # -> 5-frame prime, keeps tests fast

    def _engine(self, calibrated, zones) -> AlertEngine:
        return AlertEngine(zones=zones, fps=self.FPS,
                           is_calibrated=calibrated, speed_units="px/frame")

    def _breach_for(self, eng, zone_name, n, **vals):
        alerts = []
        zm = metrics(**vals)
        for f in range(n):
            alerts += eng.update({zone_name: zm}, {}, f, f / self.FPS)
        return alerts

    def test_uncalibrated_skips_physical_keeps_dimensionless(self):
        z = make_zone(min_duration_sec=self.MIN_DUR)
        eng = self._engine(False, [z])
        # Speed far under speed_low_ms would breach if the gate were absent.
        assert self._breach_for(eng, "z", 10, mean_speed=0.0) == []
        alerts = self._breach_for(eng, "z", 10, mean_divergence=-2.0)
        assert [a.label_for_detection() for a in alerts][:1] == \
            ["mean_divergence_critical"]

    def test_calibrated_speed_threshold_fires(self):
        z = make_zone(min_duration_sec=self.MIN_DUR)
        eng = self._engine(True, [z])
        alerts = self._breach_for(eng, "z", 10, mean_speed=0.1)
        kinds = {(a.metric_name, a.severity) for a in alerts}
        assert ("mean_speed", AlertSeverity.WARNING) in kinds
        assert all(a.is_calibrated for a in alerts)

    def test_pressure_warning_and_critical_fire_together(self):
        z = make_zone(speed_low_ms=None, divergence_critical=None,
                      counterflow_critical=None, turbulence_critical=None,
                      pressure_warning=0.02, pressure_critical=0.04,
                      min_duration_sec=self.MIN_DUR)
        eng = self._engine(True, [z])
        alerts = self._breach_for(eng, "z", 10, crowd_pressure=0.05)
        # An active alert re-emits every frame while it holds, so compare
        # severity SETS, not the raw per-frame list.
        sevs = {a.severity.name for a in alerts
                if a.metric_name == "crowd_pressure"}
        assert sevs == {"CRITICAL", "WARNING"}      # BOTH levels, keyed apart

    def test_vehicle_in_ped_zone_needs_no_priming(self):
        z = make_zone(min_duration_sec=60.0)        # long gate: proves bypass
        eng = self._engine(False, [z])
        alerts = eng.update({"z": metrics()}, {"z": True}, 0, 0.0)
        assert len(alerts) == 1
        assert alerts[0].label_for_detection() == "vehicle_in_ped_zone_warning"

    def test_nan_metric_values_are_skipped(self):
        z = make_zone(min_duration_sec=self.MIN_DUR)
        eng = self._engine(False, [z])
        nan = float("nan")
        alerts = self._breach_for(eng, "z", 10,
                                  mean_divergence=nan, counterflow_score=nan)
        assert alerts == []

    def test_emitted_labels_are_registry_consistent(self):
        z = make_zone(min_duration_sec=self.MIN_DUR)
        eng = self._engine(False, [z])
        seen = {a.label_for_detection()
                for a in self._breach_for(eng, "z", 10, mean_divergence=-2.0)}
        assert seen and seen <= DENSE_FLOW_ALERT_LABELS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
