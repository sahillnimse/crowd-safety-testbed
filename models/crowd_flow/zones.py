"""
Named polygonal zones, per-zone thresholds, and the alert engine.

Zone
----
A named polygon in image coordinates with per-metric alert thresholds.  Loaded
from the ``cameras.<camera_id>.zones`` list in configs/crowd_flow.yaml.

AlertEngine
-----------
Threshold-crossing detection with hysteresis and minimum-duration gating:

  Hysteresis
    Each threshold fires at T_fire and clears at T_fire offset by |T_fire| ×
    hysteresis_frac, moved AWAY from severity — lower for a fires-high metric,
    higher for a fires-low one.  This prevents a single noisy frame
    oscillating between ALERT and CLEAR.  The offset form is deliberate: a
    multiplicative (1 − hysteresis_frac) puts the band on the wrong side of
    any fires-low threshold whose value is positive, which is the case for
    speed_low_ms.

  Minimum duration
    An alert must persist for at least ``min_duration_sec`` before being
    emitted.  A sub-threshold frame during the priming window resets the
    counter.  Once emitted, the alert continues each frame until it clears.

  Every Alert carries the numbers that triggered it — zone, metric,
  measured value, threshold, severity — not just a level.  A monitoring
  system that says "CRITICAL" without a number is useless for post-event
  analysis.

Severity tiers
  INFO     : early-warning metric crossing (speed drop, mild divergence)
  WARNING  : actionable condition; supervisors should be notified
  CRITICAL : immediate intervention required
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(IntEnum):
    INFO     = 1
    WARNING  = 2
    CRITICAL = 3


@dataclass
class ZoneThresholds:
    """
    Per-metric thresholds for one zone.  Metrics with None thresholds are
    not checked (useful for zones where a metric has no meaningful normal
    range, e.g. an open plaza where divergence is expected to vary widely).

    speed_low_ms:
        Mean zone speed drops below this → early congestion signal.
        DISABLED for uncalibrated cameras (value present but check skipped).
    divergence_critical:
        Zone mean divergence drops below this (more negative → more compression).
        Negative = converging crowd.  This is the single most important output.
    curl_critical:
        Absolute mean curl exceeds this → rotational / turbulent flow.
    counterflow_critical:
        Fraction of cells with opposing direction exceeds this.
    turbulence_critical:
        Helbing turbulence index (velocity variance / mean_speed²) exceeds this.
    pressure_warning / pressure_critical:
        Helbing crowd pressure, ρ·Var(v), in s⁻².  Unlike every other
        threshold here these are not tuned to this project: 0.02 is the
        onset of crowd turbulence and 0.04 the onset of stampede conditions,
        established from video analysis of the 2006 Hajj disaster.  Change
        them only with a reason better than "it fires too often".

        DISABLED for uncalibrated cameras, like the speed thresholds and for
        the same reason: without a homography the density is persons per
        1000 px² rather than per m², and comparing that to a physical
        constant produces a confident number about nothing.
    min_duration_sec:
        Alert must persist for at least this many seconds.
    hysteresis_frac:
        Clear threshold = fire threshold offset by |fire| × hysteresis_frac,
        away from severity.  See _ThresholdState.
    """
    speed_low_ms:          Optional[float] = 0.4
    speed_high_ms:         Optional[float] = None
    divergence_critical:   Optional[float] = -1.5
    curl_critical:         Optional[float] = None
    counterflow_critical:  Optional[float] = 0.20
    turbulence_critical:   Optional[float] = 0.50
    pressure_warning:      Optional[float] = 0.02
    pressure_critical:     Optional[float] = 0.04
    min_duration_sec:      float = 2.0
    hysteresis_frac:       float = 0.20

    @classmethod
    def from_dict(cls, d: dict) -> "ZoneThresholds":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in valid}
        return cls(**kwargs)


# Motion floor per zone type, in source px/frame: the speed below which a
# cell is treated as static and excluded from that zone's aggregates.
#
# One global floor cannot serve both subjects, because they do not share a
# scale.  Pedestrians move 1-2 px/frame, so a floor of 0.8 sits inside the
# signal and discards genuinely walking people.  Vehicles move 10-50, so the
# same 0.8 sits far below the signal and admits every noisy background cell,
# dragging the zone mean towards zero — which reads as congestion.
#
# These are starting points keyed to the physics, not tuned constants; a zone
# can override with an explicit min_magnitude.
# The vehicle figure was 3.0 on the reasoning that vehicles move 10-50
# px/frame, and measured on this project's own Nashik clip it selected ZERO
# active cells — because the traffic there is crawling, which is precisely
# the state worth monitoring.  Measured over the lower half of that frame:
#
#     floor 0.5  ->  258 active cells, mean speed 1.02 px/frame
#     floor 0.8  ->  116 active cells, mean speed 1.50
#     floor 3.0  ->    0 active cells, mean speed 0.00
#
# 1.5 keeps roughly a 5x margin over dense-flow noise (~0.1-0.3 px/frame)
# while still admitting stationary-to-slow traffic.  A floor tuned to
# free-flowing vehicles goes blind exactly when the road congests.
_ZONE_TYPE_FLOORS: dict[str, float] = {
    "pedestrian": 0.5,
    "vehicle":    1.5,
    "mixed":      0.8,
}

# Physically achievable density per zone type, persons or vehicles per m^2.
# Standing adults pack to 8-10; cars occupy ~7 m^2 each even bumper to
# bumper, so anything above ~0.2/m^2 in a vehicle zone is a measurement
# fault rather than dense traffic.  Applying the pedestrian ceiling to a
# vehicle zone would let a 50x overcount pass unremarked.
_ZONE_TYPE_MAX_DENSITY: dict[str, float] = {
    "pedestrian": 12.0,
    "vehicle":    0.30,
    "mixed":      12.0,
}

ZONE_TYPES = tuple(_ZONE_TYPE_FLOORS)


@dataclass
class Zone:
    """
    A named polygonal region of interest within one camera's frame.

    polygon: list of (x, y) pixel tuples forming a closed convex or concave
        polygon.  The last vertex is implicitly connected back to the first.
    thresholds: per-metric alert thresholds for this zone.
    zone_type: "pedestrian" | "vehicle" | "mixed".  Selects the motion floor
        and density ceiling appropriate to what moves through this zone —
        see _ZONE_TYPE_FLOORS.  Defaults to "pedestrian", which is what every
        zone was implicitly assumed to be before this existed.
    min_magnitude: explicit motion floor in source px/frame, overriding the
        zone_type default.  Set it when you have measured the scene rather
        than when the default merely looks wrong.

    Coordinates may be given either in absolute pixels or as fractions of the
    frame size (every vertex component in [0, 1]).  Fractional polygons are
    resolved against the real frame dimensions by resolve_to_frame() and are
    the safer choice: a pixel polygon authored against one camera resolution
    silently covers the wrong part of the scene on any other, with no error
    and plausible-looking numbers.
    """
    name:       str
    polygon:    list[tuple[float, float]]
    thresholds: ZoneThresholds = field(default_factory=ZoneThresholds)
    zone_type:  str = "pedestrian"
    min_magnitude: Optional[float] = None

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                f"Zone '{self.name}' polygon must have at least 3 vertices "
                f"(got {len(self.polygon)})."
            )
        if self.zone_type not in _ZONE_TYPE_FLOORS:
            raise ValueError(
                f"Zone '{self.name}': zone_type must be one of "
                f"{list(_ZONE_TYPE_FLOORS)}, got {self.zone_type!r}"
            )

    @property
    def motion_floor(self) -> float:
        """Source px/frame below which a cell is static for THIS zone."""
        if self.min_magnitude is not None:
            return float(self.min_magnitude)
        return _ZONE_TYPE_FLOORS[self.zone_type]

    @property
    def max_density(self) -> float:
        """Physically achievable density for this zone's subject, per m²."""
        return _ZONE_TYPE_MAX_DENSITY[self.zone_type]

    @property
    def is_fractional(self) -> bool:
        """True if every vertex component lies in [0, 1] (frame fractions)."""
        return all(0.0 <= c <= 1.0 for pt in self.polygon for c in pt[:2])

    def resolve_to_frame(self, width: int, height: int) -> "Zone":
        """
        Return a copy of this zone in absolute pixel coordinates for a frame
        of the given size.

        Fractional polygons are scaled up.  Pixel polygons are returned as-is;
        the caller is responsible for warning if they do not fit the frame
        (see coverage_fraction).
        """
        if not self.is_fractional:
            return self
        scaled = [(float(x) * width, float(y) * height) for x, y in self.polygon]
        # Every field must be carried, not just the polygon: this returns a
        # NEW Zone that replaces the configured one for the rest of the run,
        # so anything omitted here silently reverts to its default.
        return Zone(name=self.name, polygon=scaled, thresholds=self.thresholds,
                    zone_type=self.zone_type, min_magnitude=self.min_magnitude)

    def coverage_fraction(self, width: int, height: int) -> float:
        """
        Fraction of the polygon's bounding box that lies inside the frame.

        1.0 = fully inside.  A low value means the polygon was authored for a
        different resolution and the zone is measuring the wrong region.
        """
        x1, y1, x2, y2 = self.bbox()
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if area <= 0:
            return 0.0
        ix1, iy1 = max(x1, 0), max(y1, 0)
        ix2, iy2 = min(x2, width), min(y2, height)
        inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        return float(inter) / float(area)

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        name    = d["name"]
        polygon = [tuple(pt) for pt in d["polygon"]]
        thresh  = ZoneThresholds.from_dict(d.get("thresholds", {}))
        return cls(name=name, polygon=polygon, thresholds=thresh,
                   zone_type=d.get("zone_type", "pedestrian"),
                   min_magnitude=d.get("min_magnitude"))

    def contains_point(self, x: float, y: float) -> bool:
        poly = np.array(self.polygon, dtype=np.float32)
        return cv2.pointPolygonTest(poly, (x, y), False) >= 0

    def bbox(self) -> tuple[int, int, int, int]:
        """Axis-aligned bounding box (x1, y1, x2, y2) of the polygon."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


@dataclass
class Alert:
    """
    One threshold-crossing event with the full measurement context.

    Every alert carries the numbers that triggered it so post-event
    analysis has concrete data, not just a severity level.
    """
    zone_name:      str
    metric_name:    str              # e.g. "divergence", "counterflow_score"
    measured_value: float
    threshold_value: float
    severity:       AlertSeverity
    frame_index:    int
    timestamp_sec:  float
    is_calibrated:  bool
    units:          str

    def as_dict(self) -> dict:
        return {
            "zone":          self.zone_name,
            "metric":        self.metric_name,
            "value":         round(self.measured_value, 4),
            "threshold":     round(self.threshold_value, 4),
            "severity":      self.severity.name,
            "frame_index":   self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "calibrated":    self.is_calibrated,
            "units":         self.units,
        }

    def label_for_detection(self) -> str:
        """Compact string label suitable for the Detection.label field."""
        sev = self.severity.name.lower()
        return f"{self.metric_name}_{sev}"


# Metric definitions: (attribute_name, higher_is_worse, severity_at_fire)
# higher_is_worse=True  → fires when value ≥ threshold
# higher_is_worse=False → fires when value ≤ threshold (e.g. negative divergence)
# (zone_metrics_attr, threshold_attr, higher_is_worse, severity,
#  needs_calibration, units)
#
# needs_calibration is explicit rather than inferred.  The gate used to be
# `"speed" in metric_attr`, which happened to be right for the two speed
# thresholds and silently wrong for anything else in physical units — crowd
# pressure is in s⁻² and would have been checked against a literature
# constant on an uncalibrated camera reporting persons per 1000 px².
_METRIC_DEFS: list[tuple[str, str, bool, AlertSeverity, bool, str]] = [
    ("mean_speed",       "speed_low_ms",          False, AlertSeverity.WARNING,  True,  "speed"),
    ("mean_divergence",  "divergence_critical",   False, AlertSeverity.CRITICAL, False, "dimensionless"),
    ("mean_curl",        "curl_critical",         True,  AlertSeverity.WARNING,  False, "dimensionless"),
    ("counterflow_score","counterflow_critical",  True,  AlertSeverity.WARNING,  False, "dimensionless"),
    ("turbulence_index", "turbulence_critical",   True,  AlertSeverity.CRITICAL, False, "dimensionless"),
    ("crowd_pressure",   "pressure_warning",      True,  AlertSeverity.WARNING,  True,  "s^-2"),
    ("crowd_pressure",   "pressure_critical",     True,  AlertSeverity.CRITICAL, True,  "s^-2"),
]

# The synthetic metric name for the vehicle-in-pedestrian-zone check, which
# has no threshold and so no entry in _METRIC_DEFS.
VEHICLE_IN_PED_METRIC = "vehicle_in_ped_zone"

#: Every Detection.label DenseFlowAnalyser can emit for an alert, i.e. exactly
#: what Alert.label_for_detection() produces.  Derived from _METRIC_DEFS rather
#: than written out, because two consumers (webapp.jobs.POSITIVE_LABELS and
#: pipeline.annotate.COLOR_MAP) keep their own copies and both had drifted out
#: of date: they listed "counterflow_warning" and "vehicle_in_ped_zone" while
#: the engine emits "counterflow_score_warning" and
#: "vehicle_in_ped_zone_warning", and neither knew about crowd_pressure at all.
#: Adding a row above now updates this automatically.
DENSE_FLOW_ALERT_LABELS: frozenset[str] = frozenset(
    [f"{metric}_{sev.name.lower()}"
     for metric, _thr, _higher, sev, _cal, _units in _METRIC_DEFS]
    + [f"{VEHICLE_IN_PED_METRIC}_{AlertSeverity.WARNING.name.lower()}"]
)


class _ThresholdState:
    """Tracks one metric × zone combination across frames."""

    def __init__(
        self,
        fire_threshold: float,
        hysteresis_frac: float,
        min_duration_sec: float,
        fps: float,
        higher_is_worse: bool,
    ) -> None:
        self.fire    = fire_threshold
        self.higher  = higher_is_worse

        # Clear threshold: offset from the fire threshold AWAY from severity by
        # |fire| x hysteresis_frac, so the band always sits on the safe side.
        #
        # This has to be expressed as an offset, not as `fire * (1 - frac)`.
        # That form is only correct when the sign of the threshold happens to
        # agree with the direction of severity, and for a fire-on-LOW metric
        # with a POSITIVE threshold it does not: speed_low_ms=0.4 gave a clear
        # value of 0.32, i.e. BELOW the fire value, so the whole band lay
        # inside the breach region.  Any value that stopped breaching also
        # cleared immediately and the metric ran with no hysteresis at all —
        # exactly the single-noisy-frame ALERT/CLEAR oscillation this class
        # exists to prevent.  Divergence worked only because its threshold is
        # negative, which flipped the arithmetic into the right direction by
        # accident.
        band = abs(fire_threshold) * hysteresis_frac
        if higher_is_worse:
            self.clear = fire_threshold - band   # e.g. 0.20 fires, 0.16 clears
        else:
            self.clear = fire_threshold + band   # e.g. -1.5 fires, -1.2 clears
                                                 #      0.4 fires,  0.48 clears

        self.min_frames = int(min_duration_sec * max(fps, 1.0))
        self.active     = False          # currently in alert state
        self.priming    = 0             # frames where threshold is breached but not yet emitted

    def _is_breached(self, value: float) -> bool:
        if self.higher:
            return value >= self.fire
        return value <= self.fire

    def _is_cleared(self, value: float) -> bool:
        if self.higher:
            return value < self.clear
        return value > self.clear

    def update(self, value: float) -> bool:
        """
        Update state machine.  Returns True if an alert should be emitted
        on this frame (i.e. currently active).
        """
        if self._is_breached(value):
            self.priming += 1
            if self.priming >= self.min_frames:
                self.active = True
        else:
            if self.active and self._is_cleared(value):
                self.active  = False
                self.priming = 0
            elif not self.active:
                self.priming = 0

        return self.active


class AlertEngine:
    """
    Evaluates all configured threshold checks across all zones each frame.

    Usage::

        engine = AlertEngine(zones, fps, is_calibrated, speed_units)
        alerts = engine.update(metrics_frame, frame_index, timestamp_sec)
    """

    def __init__(
        self,
        zones: list[Zone],
        fps: float,
        is_calibrated: bool,
        speed_units: str,
    ) -> None:
        self.zones        = zones
        self.fps          = fps
        self.is_calibrated = is_calibrated
        self.speed_units  = speed_units

        # state[(zone_name, metric_attr)] → _ThresholdState
        self._state: dict[tuple[str, str], _ThresholdState] = {}
        self._init_states()

        if not is_calibrated:
            logger.warning(
                "Camera is uncalibrated.  "
                "Alert thresholds in physical units (speed_low_ms, "
                "speed_high_ms, pressure_warning, pressure_critical) are "
                "DISABLED for this run: without a homography, speed is "
                "px/frame rather than m/s and density is persons per 1000 "
                "px² rather than per m², so comparing either against a "
                "physical threshold is meaningless.  Divergence, curl, "
                "counterflow and turbulence are dimensionless and remain "
                "active.  Run scripts/calibrate_ground_plane.py to enable "
                "the rest."
            )

    def _init_states(self) -> None:
        for zone in self.zones:
            thr = zone.thresholds
            for metric_attr, thresh_attr, higher, sev, _cal, _u in _METRIC_DEFS:
                thresh_val = getattr(thr, thresh_attr, None)
                if thresh_val is None:
                    continue  # metric not configured for this zone
                # Keyed by the THRESHOLD, not the metric: crowd_pressure has
                # both a warning and a critical level, and keying by metric
                # would let the second overwrite the first so only one of the
                # two ever fired.
                key = (zone.name, thresh_attr)
                self._state[key] = _ThresholdState(
                    fire_threshold=thresh_val,
                    hysteresis_frac=thr.hysteresis_frac,
                    min_duration_sec=thr.min_duration_sec,
                    fps=self.fps,
                    higher_is_worse=higher,
                )

    def update(
        self,
        zone_metrics: dict,        # zone_name → ZoneMetrics (from crowd_metrics.py)
        vehicle_in_ped: dict,      # zone_name → bool
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Alert]:
        """
        Evaluate all thresholds.  Returns the list of active alerts this frame.
        """
        alerts: list[Alert] = []

        for zone in self.zones:
            zm = zone_metrics.get(zone.name)
            if zm is None:
                continue

            for metric_attr, thresh_attr, higher, sev, needs_cal, units in _METRIC_DEFS:
                if needs_cal and not self.is_calibrated:
                    continue

                key = (zone.name, thresh_attr)
                state = self._state.get(key)
                if state is None:
                    continue

                value = getattr(zm, metric_attr, None)
                if value is None or np.isnan(value):
                    continue

                if state.update(float(value)):
                    thresh_val = getattr(zone.thresholds, thresh_attr)
                    alerts.append(Alert(
                        zone_name=zone.name,
                        metric_name=metric_attr,
                        measured_value=float(value),
                        threshold_value=float(thresh_val),
                        severity=sev,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        is_calibrated=self.is_calibrated,
                        units=self.speed_units if units == "speed" else units,
                    ))

            # Vehicle-in-pedestrian-zone alert (no hysteresis needed)
            if vehicle_in_ped.get(zone.name, False):
                alerts.append(Alert(
                    zone_name=zone.name,
                    metric_name=VEHICLE_IN_PED_METRIC,
                    measured_value=1.0,
                    threshold_value=0.0,
                    severity=AlertSeverity.WARNING,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    is_calibrated=self.is_calibrated,
                    units="boolean",
                ))

        return alerts
