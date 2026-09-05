"""
Rule-based crowd alert engine -> Detection records.

Port of ``Ujwal/__CMS__/__Dashboard__/core/alerts.py::AlertEngine``, adapted
to this project's alert convention instead of its JSONL sidecar:

* The source raised structured Alert objects into a Streamlit feed and a
  JSONL log. This project represents alerts as Detection rows with
  ``<metric>_<severity>`` labels (see pipeline/annotate.py::COLOR_MAP and the
  dense-flow engine's zone alerts), so ported rules emit Detections whose
  ``extra`` carries the full source context: metric name, value, threshold,
  severity, zone, and the source's operator-facing action/diversion text.
* The source deduped an alert until an operator acknowledged it. Offline
  video has no operator loop, so dedup is replaced by hysteresis: a rule is
  re-armed only after its value falls back below warn x ``rearm_ratio``,
  which lets one video contain several distinct episodes without strobing.

Rules kept from the source (feed-health and cross-camera forecast rules are
not portable here - they need live-feed heartbeats and the site graph):
  density warn/critical, pressure elevated/critical with the compression
  condition (divergence below threshold), counter-flow percentage, and
  capacity approach.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.base import Detection
from models.dm_count.metrics import FrameMetrics


@dataclass
class AlertThresholds:
    """Alert rule thresholds, in the monitor's active unit system.

    Two different scales exist and they are NOT interchangeable:

    * CALIBRATED runs (camera homography configured): speeds in m/s, density
      in heads/m^2. There, pressure = rho x Var(v) approaches Helbing units
      and thresholds around 0.1-1.0 are meaningful.
    * UNCALIBRATED runs (default): speeds are px per processed sample-step
      and density is heads/frame, so pressure lands in the hundreds-to-
      thousands range even for calm footage. The defaults below were set
      from the healthy (calm-footage) distribution of such a run: ordinary
      movement sat at median ~270 / p90 ~1750, so 1500/5000 flags sustained
      unusual agitation without firing on every frame. They are scene-
      dependent STARTING POINTS - retune per camera exactly like the
      dense-flow engine's crowd_flow.yaml thresholds.
    """
    # Density (heads/m^2 calibrated, heads/frame otherwise).
    density_warn: float = 40.0
    density_critical: float = 60.0
    # Pressure = density x velocity-variance (see scale note above).
    pressure_warn: float = 1500.0
    pressure_critical: float = 5000.0
    # Mean flow-field divergence below this signals net inward compression.
    compression_divergence: float = -0.02
    # % of moving tracks opposing the dominant direction. 45 rather than the
    # corridor-style 25: on open-area milling footage (airport halls, plazas)
    # 30-40% opposition is the healthy baseline, and 25% flagged nearly every
    # frame of a calm run.
    counter_flow_warn_pct: float = 45.0
    # Counter-flow is only meaningful when a dominant direction EXISTS. The
    # rule is evaluated only while heading entropy (8-bin, bits, max 3.0)
    # sits below this coherence ceiling - above it the crowd is milling in
    # all directions and "opposing the mean" stops being an anomaly.
    counter_flow_max_entropy: float = 2.0
    capacity_fraction: float = 0.8        # warn at fraction of safe_capacity


# label -> (metric attr, severity). Single table drives evaluation order,
# threshold lookup and the annotate.py colour keys.
_DENSITY_RULES = (
    ("crowd_density_critical", "critical"),
    ("crowd_density_warning", "warning"),
)
_COMPRESSION_RULES = (
    ("crowd_compression_critical", "critical"),
    ("crowd_compression_warning", "warning"),
)


class CrowdAlertEngine:
    def __init__(self, thresholds: AlertThresholds | None = None,
                 safe_capacity: int | None = None, rearm_ratio: float = 0.9):
        self.th = thresholds or AlertThresholds()
        self.safe_capacity = safe_capacity
        self.rearm_ratio = rearm_ratio
        # label -> True while above its raise threshold; re-arms below
        # threshold * rearm_ratio.
        self._active: dict[str, bool] = {}

    # ------------------------------------------------------------------

    def _edge(self, label: str, value: float, warn: float) -> bool:
        """Rising-edge with hysteresis re-arm."""
        if value >= warn:
            self._active[label] = True
            return True
        if self._active.get(label) and value <= warn * self.rearm_ratio:
            self._active[label] = False
        return False

    @staticmethod
    def _alert_detection(label: str, severity: str, metric: str, value: float,
                         threshold: float, frame_index: int, timestamp_sec: float,
                         action: str, diversion: str) -> Detection:
        return Detection(
            model_name="dm_count_crowd",
            label=label,
            confidence=min(1.0, abs(value) / max(abs(threshold), 1e-9)),
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            bbox=None,
            extra={
                "alert_severity": severity.upper(),
                "metric": metric,
                "value": round(float(value), 5),
                "threshold": threshold,
                "zone": "FOV",
                "action": action,
                "diversion": diversion,
            },
        )

    # ------------------------------------------------------------------

    def evaluate(self, m: FrameMetrics, frame_index: int,
                 timestamp_sec: float) -> list[Detection]:
        raised: list[Detection] = []

        # --- density -----------------------------------------------------
        if m.density > 0:
            if self._edge("crowd_density_critical", m.density, self.th.density_critical):
                raised.append(self._alert_detection(
                    "crowd_density_critical", "critical", "density", m.density,
                    self.th.density_critical, frame_index, timestamp_sec,
                    "Hold inbound flow; open overflow route immediately.",
                    "Divert arrivals via an alternate corridor away from this view.",
                ))
            elif self._edge("crowd_density_warning", m.density, self.th.density_warn):
                raised.append(self._alert_detection(
                    "crowd_density_warning", "warning", "density", m.density,
                    self.th.density_warn, frame_index, timestamp_sec,
                    "Stage officers at this view and slow inbound movement.",
                    "Prepare diversion before capacity is reached.",
                ))

        # --- pressure / compression --------------------------------------
        if m.pressure >= self.th.pressure_critical and \
                m.divergence < self.th.compression_divergence:
            if self._edge("crowd_compression_critical", m.pressure,
                          self.th.pressure_critical):
                raised.append(self._alert_detection(
                    "crowd_compression_critical", "critical", "pressure",
                    m.pressure, self.th.pressure_critical, frame_index,
                    timestamp_sec,
                    "Crowd pressure rising WITH inward compression - clear this view.",
                    "Push people toward the lowest-pressure adjacent exit.",
                ))
        elif self._edge("crowd_compression_warning", m.pressure, self.th.pressure_warn):
            raised.append(self._alert_detection(
                "crowd_compression_warning", "warning", "pressure",
                m.pressure, self.th.pressure_warn, frame_index, timestamp_sec,
                "Monitor for stop-and-go onset; keep exits clear.",
                "Keep a downstream path open.",
            ))

        # --- counter-flow --------------------------------------------------
        # Gated on coherence: without a dominant direction (entropy below the
        # ceiling) there is no flow to counter. While gated, the rule also
        # re-arms so a genuinely bidirectional episode later in the video can
        # still fire.
        if m.directional_entropy >= self.th.counter_flow_max_entropy:
            self._active["counter_flow_warning"] = False
        elif self._edge("counter_flow_warning", m.counter_flow_pct,
                        self.th.counter_flow_warn_pct):
            raised.append(self._alert_detection(
                "counter_flow_warning", "warning", "counter_flow_pct",
                m.counter_flow_pct, self.th.counter_flow_warn_pct,
                frame_index, timestamp_sec,
                "Impose one-way control. Separate opposing streams.",
                "Hold one direction; release the dominant stream first.",
            ))

        # --- capacity ------------------------------------------------------
        if self.safe_capacity and m.head_count >= \
                self.safe_capacity * self.th.capacity_fraction:
            if self._edge("capacity_warning", m.head_count,
                          self.safe_capacity * self.th.capacity_fraction):
                raised.append(self._alert_detection(
                    "capacity_warning", "warning", "head_count",
                    float(m.head_count),
                    self.safe_capacity * self.th.capacity_fraction,
                    frame_index, timestamp_sec,
                    f"{m.head_count}/{self.safe_capacity} in view. "
                    "Stop additional inflow.",
                    "Redirect arrivals to the alternate route.",
                ))

        return raised

    @property
    def active_labels(self) -> list[str]:
        return [k for k, v in self._active.items() if v]
