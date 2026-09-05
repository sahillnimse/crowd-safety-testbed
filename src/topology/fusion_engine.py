"""
Fusion & Cross-Camera Reasoning Engine.

Periodically processes multi-camera metrics using the topology graph to forecast
downstream corridor inflows, predict bottlenecks, and flag rising crush risks.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from topology.graph import TOPOLOGY, CameraNode, CameraTopology
from topology.metric_store import METRIC_STORE, MetricStore

logger = logging.getLogger(__name__)


@dataclass
class FusionAlert:
    """One cross-camera fusion alert."""
    id: str
    camera_id: str                      # Target camera ID
    target_name: str                    # Human-friendly camera name
    level: str                          # "BOTTLENECK_PREDICTED" | "CRUSH_RISK_RISING" | "ACCUMULATION_RISING"
    lead_time_sec: float                # Predicted lead time in seconds
    source_cameras: List[str]           # Upstream camera IDs
    source_names: List[str]             # Upstream camera human names
    predicted_inflow: float             # Combined predicted inflow (pax/min)
    target_capacity: float              # Target corridor capacity (pax/min)
    current_density: float              # Current measured density at target
    timestamp_epoch_ms: int             # Alert generation timestamp
    detail: str                         # Templated detail string
    active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "target_name": self.target_name,
            "level": self.level,
            "lead_time_sec": round(self.lead_time_sec, 1),
            "source_cameras": self.source_cameras,
            "source_names": self.source_names,
            "predicted_inflow": round(self.predicted_inflow, 1),
            "target_capacity": round(self.target_capacity, 1),
            "current_density": round(self.current_density, 3),
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "detail": self.detail,
            "active": self.active,
            "created_at": round(self.created_at, 2),
            "updated_at": round(self.updated_at, 2),
        }


class FusionEngine:
    """
    Async service evaluating cross-camera inflow predictions and safety rules.
    """

    #: Seconds an alert must go un-triggered before it clears. Prevents the
    #: per-tick flapping a bare "not re-triggered -> clear" produces when the
    #: underlying estimate sits near a threshold.
    CLEAR_HOLD_SEC: float = 10.0

    def __init__(
        self,
        topology: Optional[CameraTopology] = None,
        metric_store: Optional[MetricStore] = None,
    ):
        self.topology = topology or TOPOLOGY
        self.metric_store = metric_store or METRIC_STORE
        self._running = False
        self._alerts: Dict[str, FusionAlert] = {}   # alert_key -> FusionAlert
        # None means "could not forecast", which is NOT the same as 0.0.
        self._predicted_inflows: Dict[str, Optional[float]] = {}
        # cam_id -> {"complete": bool, "reason": str, "missing": [...]}
        self._forecast_status: Dict[str, dict] = {}
        # Cameras already warned about for uncalibrated flow, so the error is
        # logged once rather than at every tick of the loop.
        self._uncalibrated_warned: Set[str] = set()
        # Running people-count imbalance per camera, for the conservation rule.
        # cam_id -> (accumulated_people, last_update_epoch_ms)
        self._accumulation: Dict[str, Tuple[float, int]] = {}
        self._subscribers: Set[asyncio.Queue] = set()

    # ------------------------------------------------------------------
    # Alert Subscribers (WebSocket & Push)
    # ------------------------------------------------------------------

    def register_subscriber(self, q: asyncio.Queue) -> None:
        self._subscribers.add(q)

    def unregister_subscriber(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def _broadcast(self, event_type: str, data: Any) -> None:
        if not self._subscribers:
            return
        payload = {"event": event_type, "data": data, "timestamp_epoch_ms": int(time.time() * 1000)}
        dead_queues = set()
        # Iterate a COPY: a WebSocket connecting or dropping calls
        # register/unregister from another task, and mutating the set while
        # this loop is walking it raises "Set changed size during iteration"
        # — which would kill the tick that was mid-broadcast.
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except Exception:
                # Full or closed queue: a slow client must not be allowed to
                # back-pressure the reasoning loop, so it is dropped.
                dead_queues.add(q)
        for dq in dead_queues:
            self._subscribers.discard(dq)

    # ------------------------------------------------------------------
    # Core Mathematical Reasoning Step
    # ------------------------------------------------------------------

    def evaluate_step(
        self,
        current_time_epoch_ms: Optional[int] = None,
        wall_now: Optional[float] = None,
    ) -> List[FusionAlert]:
        """
        Execute one fusion reasoning cycle across all topology nodes.
        Can be called synchronously by tests or asynchronously by run_loop().
        """
        # "Now" comes from the newest stored sample, not the wall clock.
        #
        # Every stored timestamp lives on the SOURCE's timeline. For live
        # cameras that is the wall clock, but for recorded footage it is the
        # video's own clock, which runs at whatever rate the pipeline decodes.
        # Using time.time() there put the reference hours away from every
        # sample in the buffer, so each historical lookup missed and the
        # engine either saw no data or (before the store was fixed) silently
        # received the current value for every past query.
        if current_time_epoch_ms is not None:
            now_ms = current_time_epoch_ms
        else:
            now_ms = self.metric_store.reference_epoch_ms() or int(time.time() * 1000)
        wall_time = wall_now or time.time()
        staleness_thr = self.topology.staleness_threshold_sec
        density_thr = self.topology.density_threshold
        emitted_alerts: List[FusionAlert] = []
        active_keys_this_step: Set[str] = set()

        for target_cam in self.topology.all_cameras():
            target_id = target_cam.id
            upstream = self.topology.upstream_of(target_id)
            if not upstream:
                # Leaf/source camera with no incoming edges
                self._predicted_inflows[target_id] = 0.0
                continue

            # Check target staleness (if target exists in metrics)
            target_snap = self.metric_store.get_latest(target_id)
            target_density = target_snap.density if target_snap else 0.0

            # Compute predicted inflow: sum(flow_rate(Ui, t - travel_time(Ui, C)))
            total_inflow = 0.0
            valid_sources: List[str] = []
            source_names: List[str] = []
            travel_times: List[float] = []
            _has_stale_source = False

            missing_sources: List[str] = []
            uncalibrated_sources: List[str] = []

            for u_id, travel_time_sec in upstream:
                u_cam = self.topology.get_camera(u_id)
                u_name = u_cam.name if u_cam else u_id

                # Staleness check on upstream camera
                if self.metric_store.is_stale(u_id, threshold_sec=staleness_thr):
                    logger.debug(
                        "Fusion skipped edge %s -> %s: upstream camera '%s' is stale (>%.1fs)",
                        u_id, target_id, u_id, staleness_thr,
                    )
                    _has_stale_source = True
                    missing_sources.append(u_name)
                    continue

                target_time_ms = now_ms - int(travel_time_sec * 1000)

                # Whole snapshot, not just the number: whether the value was
                # calibrated when recorded travels with it, and a flow rate
                # that is not in pax/min must not be summed into a total that
                # is about to be compared against a pax/min capacity.
                snap = self.metric_store.get_historical_snapshot(
                    camera_id=u_id, target_epoch_ms=target_time_ms,
                )
                if snap is None:
                    # No sample near t - travel_time. NOT the same as zero
                    # flow: we simply cannot see what that camera was doing
                    # at the relevant moment, and pretending otherwise is how
                    # a forecast becomes fiction.
                    missing_sources.append(u_name)
                    continue

                if not snap.flow_is_calibrated:
                    uncalibrated_sources.append(u_name)

                total_inflow += max(0.0, snap.flow_rate_pax_min)
                valid_sources.append(u_id)
                source_names.append(u_name)
                travel_times.append(travel_time_sec)

            if not valid_sources:
                # No usable upstream source. Recorded as None rather than 0.0:
                # "we cannot forecast this camera" and "we forecast no inflow"
                # are different claims, and the second one reads as safe.
                self._predicted_inflows[target_id] = None
                self._forecast_status[target_id] = {
                    "complete": False,
                    "reason": "no_usable_upstream",
                    "missing": missing_sources,
                }
                continue

            # A forecast built from only SOME of the upstream sources
            # UNDER-estimates inflow, which suppresses the alert. That is the
            # dangerous direction and must never be silent -- it is the same
            # class of failure as a dead camera reporting a completed run.
            complete = not missing_sources
            self._forecast_status[target_id] = {
                "complete": complete,
                "reason": "" if complete else "incomplete_upstream",
                "missing": missing_sources,
                "uncalibrated": uncalibrated_sources,
            }
            if not complete:
                logger.warning(
                    "Forecast for '%s' is INCOMPLETE: %d of %d upstream sources "
                    "unavailable (%s). Predicted inflow is an UNDER-estimate and "
                    "alerting is suppressed accordingly.",
                    target_cam.name, len(missing_sources), len(upstream),
                    ", ".join(missing_sources),
                )

            self._predicted_inflows[target_id] = total_inflow
            lead_time = min(travel_times) if travel_times else 0.0
            capacity = target_cam.corridor_capacity_pax_min
            src_str = " + ".join(source_names)

            # Capacity comparisons require REAL units on both sides.
            #
            # `corridor_capacity_pax_min` is a physical figure from a site
            # survey. An uncalibrated camera cannot produce pax/min: without
            # the real-world width of the counting line, its "flow" is a count
            # scaled by an arbitrary constant. Comparing the two produces a
            # confident, wrongly-scaled safety alert -- worse than no alert,
            # because it looks authoritative.
            #
            # So the safety rules are SKIPPED, loudly, rather than evaluated on
            # numbers that do not mean what the field name says.
            if uncalibrated_sources:
                self._forecast_status[target_id]["blocked"] = "uncalibrated_flow"
                if target_id not in self._uncalibrated_warned:
                    self._uncalibrated_warned.add(target_id)
                    logger.error(
                        "Fusion rules DISABLED for '%s': upstream flow from %s is "
                        "uncalibrated, so it is not in pax/min and cannot be "
                        "compared against a %.0f pax/min corridor capacity. "
                        "Calibrate those cameras (scripts/calibrate_ground_plane.py, "
                        "or fit a perspective map) to enable bottleneck alerting.",
                        target_cam.name, ", ".join(uncalibrated_sources), capacity,
                    )
                continue

            # --- Rule 1: BOTTLENECK_PREDICTED ---
            # if predicted_inflow(C) > C.corridor_capacity
            if total_inflow > capacity:
                alert_key = f"{target_id}:BOTTLENECK_PREDICTED"
                detail_text = (
                    f"{src_str} combined inflow ({total_inflow:.0f} pax/min) "
                    f"exceeds {target_cam.name} capacity ({capacity:.0f} pax/min) "
                    f"— predicted in {int(lead_time)}s"
                )
                alert = self._raise_or_update_alert(
                    key=alert_key,
                    target_id=target_id,
                    target_name=target_cam.name,
                    level="BOTTLENECK_PREDICTED",
                    lead_time_sec=lead_time,
                    source_cameras=valid_sources,
                    source_names=source_names,
                    predicted_inflow=total_inflow,
                    target_capacity=capacity,
                    current_density=target_density,
                    timestamp_epoch_ms=now_ms,
                    detail=detail_text,
                    wall_time=wall_time,
                )
                emitted_alerts.append(alert)
                active_keys_this_step.add(alert_key)

            # --- Rule 2: CRUSH_RISK_RISING ---
            # if current_density(C) > density_threshold AND predicted_inflow(C) > 0.7 * C.capacity
            if target_density >= density_thr and total_inflow > (0.7 * capacity):
                alert_key = f"{target_id}:CRUSH_RISK_RISING"
                detail_text = (
                    f"{target_cam.name} density ({target_density:.2f} pax/m²) is critical "
                    f"and inbound inflow ({total_inflow:.0f} pax/min, >70% capacity) from {src_str} "
                    f"is rising — predicted in {int(lead_time)}s"
                )
                alert = self._raise_or_update_alert(
                    key=alert_key,
                    target_id=target_id,
                    target_name=target_cam.name,
                    level="CRUSH_RISK_RISING",
                    lead_time_sec=lead_time,
                    source_cameras=valid_sources,
                    source_names=source_names,
                    predicted_inflow=total_inflow,
                    target_capacity=capacity,
                    current_density=target_density,
                    timestamp_epoch_ms=now_ms,
                    detail=detail_text,
                    wall_time=wall_time,
                )
                emitted_alerts.append(alert)
                active_keys_this_step.add(alert_key)

            # --- Rule 3: ACCUMULATION_RISING (conservation of people) ---
            #
            # The rule the other two cannot express. Both of those fire on a
            # THRESHOLD BREACH -- inflow exceeding capacity. But a segment can
            # fill dangerously while inflow stays comfortably below capacity,
            # if outflow is lower still:
            #
            #     300 pax/min in  ->  [ segment, capacity 400 ]  ->  100 out
            #                              200/min ACCUMULATING
            #
            # Inflow (300) never breaches capacity (400), so Rules 1 and 2 stay
            # silent while 200 people per minute pile up in a space no camera
            # covers -- 1,000 extra people after five minutes. That is the
            # crush build-up this whole layer exists to catch, and it is
            # invisible from any single camera: both ends look healthy.
            #
            # Integrated over time rather than compared instantaneously,
            # because accumulation is the INTEGRAL of the imbalance. A brief
            # surplus is normal; a sustained one is a build-up.
            outflow, outflow_unmeasurable_reason = self._measured_outflow(target_id, now_ms)
            if outflow is not None:
                imbalance = total_inflow - outflow          # pax/min
                accumulated, last_ms = self._accumulation.get(target_id, (0.0, now_ms))
                dt_min = max(0.0, (now_ms - last_ms) / 60000.0)
                # Drain toward zero when outflow exceeds inflow: people who
                # left are gone, and a segment that clears must not stay
                # flagged on a debt from ten minutes ago.
                accumulated = max(0.0, accumulated + imbalance * dt_min)
                self._accumulation[target_id] = (accumulated, now_ms)

                holding = self._holding_capacity(target_cam)
                if holding > 0 and accumulated > holding:
                    alert_key = f"{target_id}:ACCUMULATION_RISING"
                    detail_text = (
                        f"{accumulated:.0f} people have accumulated between "
                        f"{src_str} and {target_cam.name} "
                        f"({total_inflow:.0f} in vs {outflow:.0f} out pax/min, "
                        f"net +{imbalance:.0f}/min) — exceeds the {holding:.0f} "
                        f"holding capacity of that segment"
                    )
                    alert = self._raise_or_update_alert(
                        key=alert_key,
                        target_id=target_id,
                        target_name=target_cam.name,
                        level="ACCUMULATION_RISING",
                        lead_time_sec=lead_time,
                        source_cameras=valid_sources,
                        source_names=source_names,
                        predicted_inflow=total_inflow,
                        target_capacity=capacity,
                        current_density=target_density,
                        timestamp_epoch_ms=now_ms,
                        detail=detail_text,
                        wall_time=wall_time,
                    )
                    emitted_alerts.append(alert)
                    active_keys_this_step.add(alert_key)
            else:
                # Outflow is unmeasurable (missing, stale, or uncalibrated), so this
                # segment's accumulation cannot be computed. Recorded so the
                # UI can say "not monitored" rather than implying "safe".
                self._forecast_status[target_id]["accumulation"] = outflow_unmeasurable_reason or "unmeasurable_outflow"

        # Deactivate alerts that did not re-trigger — but only after a hold-down.
        #
        # At a 1 Hz tick with a noisy flow estimate hovering near capacity, a
        # bare "not re-triggered this step -> clear" makes alerts flap on and
        # off every second. In a control room that is worse than a stuck alert:
        # it trains the operator to ignore the panel, and the one that matters
        # arrives looking like all the noise before it.
        #
        # An alert therefore has to stay un-triggered for CLEAR_HOLD_SEC of
        # wall time before it clears. Hysteresis in time, matching what
        # AlertEngine already does for the per-camera thresholds.
        for key, alert in list(self._alerts.items()):
            if key in active_keys_this_step or not alert.active:
                continue
            if (wall_time - alert.updated_at) >= self.CLEAR_HOLD_SEC:
                alert.active = False
                alert.updated_at = wall_time

        return emitted_alerts

    def _measured_outflow(self, cam_id: str, now_ms: int) -> Tuple[Optional[float], Optional[str]]:
        """
        Flow leaving the segment feeding ``cam_id``, measured by ``cam_id`` itself, in pax/min.

        None when it cannot be measured — missing samples, stale, or uncalibrated.
        None is not zero: "we cannot see what is leaving" must not be treated as
        "nothing is leaving", which would manufacture an accumulation alert out of
        a blind spot.

        The segment being conserved is between the upstream cameras and ``cam_id``.
        Inflow was measured upstream at (t - travel_time); outflow is measured
        by ``cam_id`` passing its counting line at t. Whether anything exists
        downstream of ``cam_id`` is irrelevant to that calculation — a terminal
        camera (e.g. at a dead-end ghat or exit) is precisely where accumulation
        is most dangerous and must be computed.
        """
        snap = self.metric_store.get_latest(cam_id)
        if snap is None:
            return None, "missing_target_metrics"
        if self.metric_store.is_stale(cam_id, self.topology.staleness_threshold_sec):
            return None, "stale_target_metrics"
        if not snap.flow_is_calibrated:
            return None, "uncalibrated_target_flow"
        return max(0.0, snap.flow_rate_pax_min), None

    @staticmethod
    def _holding_capacity(cam: CameraNode) -> float:
        """
        How many people the segment feeding this camera can hold before the
        accumulation is dangerous.

        Taken from the node when configured. The fallback is deliberately
        conservative -- one minute of corridor capacity -- because a segment
        holding more than a minute's worth of throughput is, by definition,
        not clearing at the rate it is filling.

        This is a placeholder for a surveyed figure. A real deployment should
        set `holding_capacity_pax` per camera from the physical area of the
        segment and a target density; until then the alert is an indicator,
        not a measurement.
        """
        configured = getattr(cam, "holding_capacity_pax", None)
        if configured:
            return float(configured)
        return float(cam.corridor_capacity_pax_min)

    def get_forecast_status(self, cam_id: str) -> dict:
        """Whether this camera's forecast is complete, and why not."""
        return self._forecast_status.get(
            cam_id, {"complete": False, "reason": "not_evaluated", "missing": []})

    def _raise_or_update_alert(
        self,
        key: str,
        target_id: str,
        target_name: str,
        level: str,
        lead_time_sec: float,
        source_cameras: List[str],
        source_names: List[str],
        predicted_inflow: float,
        target_capacity: float,
        current_density: float,
        timestamp_epoch_ms: int,
        detail: str,
        wall_time: float,
    ) -> FusionAlert:
        existing = self._alerts.get(key)
        if existing is not None:
            existing.lead_time_sec = lead_time_sec
            existing.source_cameras = source_cameras
            existing.source_names = source_names
            existing.predicted_inflow = predicted_inflow
            existing.target_capacity = target_capacity
            existing.current_density = current_density
            existing.timestamp_epoch_ms = timestamp_epoch_ms
            existing.detail = detail
            existing.active = True
            existing.updated_at = wall_time
            return existing

        new_alert = FusionAlert(
            id=uuid.uuid4().hex[:12],
            camera_id=target_id,
            target_name=target_name,
            level=level,
            lead_time_sec=lead_time_sec,
            source_cameras=source_cameras,
            source_names=source_names,
            predicted_inflow=predicted_inflow,
            target_capacity=target_capacity,
            current_density=current_density,
            timestamp_epoch_ms=timestamp_epoch_ms,
            detail=detail,
            active=True,
            created_at=wall_time,
            updated_at=wall_time,
        )
        self._alerts[key] = new_alert
        return new_alert

    def get_active_alerts(self) -> List[FusionAlert]:
        """Return currently active alerts sorted by recency."""
        active = [a for a in self._alerts.values() if a.active]
        active.sort(key=lambda a: a.updated_at, reverse=True)
        return active

    def get_all_alerts(self) -> List[FusionAlert]:
        alerts = list(self._alerts.values())
        alerts.sort(key=lambda a: a.updated_at, reverse=True)
        return alerts

    def get_predicted_inflow(self, camera_id: str) -> Optional[float]:
        """
        Latest forecast inflow in pax/min, or None when it could not be made.

        None, not 0.0. A camera whose upstream sources are all stale has an
        UNKNOWN inflow, and reporting zero would render as a quiet, healthy
        gauge on exactly the camera the system has stopped being able to see.
        """
        return self._predicted_inflows.get(camera_id)

    # ------------------------------------------------------------------
    # Asynchronous background loop
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """Background coroutine ticking at fusion_tick_sec."""
        self._running = True
        logger.info("FusionEngine async reasoning loop started.")
        while self._running:
            try:
                alerts = self.evaluate_step()
                if alerts or self._predicted_inflows:
                    # Broadcast telemetry & alerts to WebSocket subscribers
                    active_payload = [a.to_dict() for a in self.get_active_alerts()]
                    # None survives to the client as null. round(None) is a
                    # TypeError, and coercing it to 0.0 here would erase the
                    # distinction the None exists to carry: "cannot forecast"
                    # must not render as "no inflow", which reads as safe.
                    inflow_payload = {
                        cid: (round(val, 1) if val is not None else None)
                        for cid, val in self._predicted_inflows.items()
                    }
                    status_payload = {
                        cid: self.get_forecast_status(cid)
                        for cid in self._predicted_inflows
                    }
                    await self._broadcast(
                        "fusion_tick",
                        {
                            "alerts": active_payload,
                            "predicted_inflows": inflow_payload,
                            "forecast_status": status_payload,
                        },
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in FusionEngine run_loop: %s", exc, exc_info=True)

            tick_sec = max(0.2, self.topology.fusion_tick_sec)
            await asyncio.sleep(tick_sec)

    def stop(self) -> None:
        self._running = False


FUSION_ENGINE = FusionEngine()
