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
    Each threshold fires at value T_fire and clears at T_clear = T_fire ×
    (1 − hysteresis_frac).  This prevents a single noisy frame oscillating
    between ALERT and CLEAR.  For thresholds where the alert fires on
    *negative* values (divergence), the signs are handled correctly.

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
    min_duration_sec:
        Alert must persist for at least this many seconds.
    hysteresis_frac:
        Clear threshold = fire threshold × (1 − hysteresis_frac).
    """
    speed_low_ms:          Optional[float] = 0.4
    speed_high_ms:         Optional[float] = None
    divergence_critical:   Optional[float] = -1.5
    curl_critical:         Optional[float] = None
    counterflow_critical:  Optional[float] = 0.20
    turbulence_critical:   Optional[float] = 0.50
    min_duration_sec:      float = 2.0
    hysteresis_frac:       float = 0.20

    @classmethod
    def from_dict(cls, d: dict) -> "ZoneThresholds":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in valid}
        return cls(**kwargs)


@dataclass
class Zone:
    """
    A named polygonal region of interest within one camera's frame.

    polygon: list of (x, y) pixel tuples forming a closed convex or concave
        polygon.  The last vertex is implicitly connected back to the first.
    thresholds: per-metric alert thresholds for this zone.

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

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                f"Zone '{self.name}' polygon must have at least 3 vertices "
                f"(got {len(self.polygon)})."
            )

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
        return Zone(name=self.name, polygon=scaled, thresholds=self.thresholds)

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
        return cls(name=name, polygon=polygon, thresholds=thresh)

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
_METRIC_DEFS: list[tuple[str, str, bool, AlertSeverity]] = [
    # (zone_metrics_attr, threshold_attr, higher_is_worse, severity)
    ("mean_speed",       "speed_low_ms",         False, AlertSeverity.WARNING),
    ("mean_divergence",  "divergence_critical",   False, AlertSeverity.CRITICAL),
    ("mean_curl",        "curl_critical",         True,  AlertSeverity.WARNING),
    ("counterflow_score","counterflow_critical",  True,  AlertSeverity.WARNING),
    ("turbulence_index", "turbulence_critical",   True,  AlertSeverity.CRITICAL),
]


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

        # Clear threshold: inside the hysteresis band
        if higher_is_worse:
            self.clear = fire_threshold * (1.0 - hysteresis_frac)
        else:
            # Fire on low value: clear when value rises above fire*(1 - hysteresis)
            # e.g. divergence fires at -1.5, clears at -1.5*(1-0.2) = -1.2
            self.clear = fire_threshold * (1.0 - hysteresis_frac)

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
                "Speed-based alert thresholds (speed_low_ms, speed_high_ms) "
                "are DISABLED for this run to prevent px/frame values from "
                "being compared against m/s thresholds.  "
                "All other metrics (divergence, curl, counterflow, turbulence) "
                "remain active."
            )

    def _init_states(self) -> None:
        for zone in self.zones:
            thr = zone.thresholds
            for metric_attr, thresh_attr, higher, sev in _METRIC_DEFS:
                thresh_val = getattr(thr, thresh_attr, None)
                if thresh_val is None:
                    continue  # metric not configured for this zone
                key = (zone.name, metric_attr)
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

            for metric_attr, thresh_attr, higher, sev in _METRIC_DEFS:
                # Skip speed checks if uncalibrated
                if not self.is_calibrated and "speed" in metric_attr:
                    continue

                key = (zone.name, metric_attr)
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
                        units=self.speed_units if "speed" in metric_attr else "dimensionless",
                    ))

            # Vehicle-in-pedestrian-zone alert (no hysteresis needed)
            if vehicle_in_ped.get(zone.name, False):
                alerts.append(Alert(
                    zone_name=zone.name,
                    metric_name="vehicle_in_ped_zone",
                    measured_value=1.0,
                    threshold_value=0.0,
                    severity=AlertSeverity.WARNING,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    is_calibrated=self.is_calibrated,
                    units="boolean",
                ))

        return alerts
