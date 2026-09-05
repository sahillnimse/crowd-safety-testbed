"""
Thread-safe in-memory metric store for real-time per-camera telemetry.

Stores current snapshots and historical ring buffers with clock offset correction
for cross-camera synchronization.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Max snapshots preserved per camera (e.g. 600 frames at 2Hz = 5 minutes)
_MAX_HISTORY_PER_CAMERA = 1000

# Seconds between disk flushes. Telemetry arrives per frame; writing every one
# would be a disk write per camera per frame. Losing up to this much history to
# an abrupt kill is an acceptable trade for not saturating I/O during a run.
_FLUSH_INTERVAL_SEC = 5.0

# Default location for the persisted buffers. Alongside the job state, which is
# already persisted here for the same reason.
_DEFAULT_PERSIST_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "state", "metric_store.json"))


@dataclass
class CameraMetricSnapshot:
    """Telemetry snapshot from a single camera at a single point in time."""
    camera_id: str
    timestamp_epoch_ms: int              # UTC epoch ms corrected by clock_offset_sec
    density: float = 0.0                 # see `units` - NOT always pax/m²
    flow_rate_pax_min: float = 0.0       # see `units` - NOT always pax/min
    dominant_direction_vector: Tuple[float, float] = (0.0, 0.0)
    crush_risk_score: float = 0.0        # Normalized 0.0-1.0 risk score
    person_count: int = 0
    raw_timestamp_sec: float = 0.0       # Original video timestamp_sec
    received_at: float = field(default_factory=time.time)

    # ---- Provenance of the two physical quantities -------------------
    #
    # These exist because the field names promise physical units that the
    # producer cannot always deliver. Flow in pax/min requires knowing the
    # real-world width of the counting line; density in pax/m² requires a
    # ground-plane homography. Neither is available on an uncalibrated
    # camera, and NONE of the four Nashik cameras is calibrated.
    #
    # Without these flags an uncalibrated estimate in arbitrary units flows
    # straight into `predicted_inflow > corridor_capacity_pax_min`, which
    # compares a made-up number against a real physical capacity and produces
    # a confident, wrongly-scaled safety alert. The fusion engine refuses that
    # comparison when either flag is False.
    #
    # `units` is free text naming what the value actually is, so a reader of
    # the API or a log can tell without tracing the producer.
    flow_is_calibrated: bool = False
    density_is_calibrated: bool = False
    units: str = "uncalibrated"

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "density": round(self.density, 3),
            "flow_rate_pax_min": round(self.flow_rate_pax_min, 1),
            "dominant_direction_vector": [
                round(self.dominant_direction_vector[0], 3),
                round(self.dominant_direction_vector[1], 3),
            ],
            "crush_risk_score": round(self.crush_risk_score, 3),
            "person_count": self.person_count,
            "raw_timestamp_sec": round(self.raw_timestamp_sec, 3),
            "received_at": round(self.received_at, 3),
            "flow_is_calibrated": self.flow_is_calibrated,
            "density_is_calibrated": self.density_is_calibrated,
            "units": self.units,
        }


class MetricStore:
    """Thread-safe store for camera metrics and time-series buffers."""

    def __init__(self, max_history: int = _MAX_HISTORY_PER_CAMERA,
                 persist_path: Optional[str] = None,
                 autoload: bool = True):
        """
        ``persist_path``: JSON file the ring buffers are mirrored to, so
        telemetry survives a process restart. None disables persistence
        entirely (used by tests, which must not touch shared state).

        Why this exists
        ---------------
        The store was memory-only. Job state was persisted but telemetry was
        not, so restarting the server (deploy, crash, power event) erased every
        sample while the jobs that produced them survived. The Route View then
        showed "telemetry offline" for cameras that had been monitored minutes
        earlier, and the fusion engine had no history to do its time-shifted
        upstream lookups against until the buffers refilled — several minutes
        of blindness after every restart, at a multi-day event.
        """
        self._lock = threading.RLock()
        self.max_history = max_history
        self._latest: Dict[str, CameraMetricSnapshot] = {}
        self._history: Dict[str, Deque[CameraMetricSnapshot]] = {}
        self.persist_path = persist_path
        # Writes are batched: flushing on every update() would mean a disk
        # write per camera per frame, which at 25 fps across several cameras
        # is far more I/O than the value of a sub-second-fresh snapshot.
        self._dirty = False
        self._last_flush = 0.0
        if persist_path and autoload:
            self.load()

    def update(
        self,
        camera_id: str,
        density: float = 0.0,
        flow_rate_pax_min: float = 0.0,
        dominant_direction_vector: Tuple[float, float] = (0.0, 0.0),
        crush_risk_score: float = 0.0,
        person_count: int = 0,
        raw_timestamp_sec: float = 0.0,
        stream_start_epoch_ms: Optional[int] = None,
        clock_offset_sec: float = 0.0,
        explicit_epoch_ms: Optional[int] = None,
        flow_is_calibrated: bool = False,
        density_is_calibrated: bool = False,
        units: str = "uncalibrated",
    ) -> CameraMetricSnapshot:
        """
        Record a new metric snapshot for a camera.

        Timestamp synchronization:
        If explicit_epoch_ms is provided, it is used.
        Otherwise, if stream_start_epoch_ms is provided, timestamp is
        stream_start_epoch_ms + int(raw_timestamp_sec * 1000).
        Otherwise, fallback to current system wall-clock (epoch ms).
        Then, clock_offset_sec (in milliseconds) is added to account for camera-specific clock drift.
        """
        now_wall = time.time()
        if explicit_epoch_ms is not None:
            base_ms = explicit_epoch_ms
        elif stream_start_epoch_ms is not None:
            base_ms = stream_start_epoch_ms + int(raw_timestamp_sec * 1000)
        else:
            base_ms = int(now_wall * 1000)

        # Apply calibrated clock offset (sec -> ms)
        corrected_epoch_ms = base_ms + int(clock_offset_sec * 1000)

        snapshot = CameraMetricSnapshot(
            camera_id=camera_id,
            timestamp_epoch_ms=corrected_epoch_ms,
            density=float(density),
            flow_rate_pax_min=float(flow_rate_pax_min),
            dominant_direction_vector=dominant_direction_vector,
            crush_risk_score=float(crush_risk_score),
            person_count=int(person_count),
            raw_timestamp_sec=float(raw_timestamp_sec),
            received_at=now_wall,
            flow_is_calibrated=bool(flow_is_calibrated),
            density_is_calibrated=bool(density_is_calibrated),
            units=str(units),
        )

        with self._lock:
            self._latest[camera_id] = snapshot
            if camera_id not in self._history:
                self._history[camera_id] = deque(maxlen=self.max_history)
            self._history[camera_id].append(snapshot)
            self._dirty = True

        # Outside the lock: the flush takes it again briefly and disk I/O must
        # not be held across a write while producers are blocked on update().
        self.flush()
        return snapshot

    def get_latest(self, camera_id: str) -> Optional[CameraMetricSnapshot]:
        """Return the most recent snapshot for a camera."""
        with self._lock:
            return self._latest.get(camera_id)

    def is_stale(self, camera_id: str, threshold_sec: float = 5.0) -> bool:
        """
        True if this camera has not delivered telemetry recently.

        Deliberately measured on ``received_at`` (wall clock), NOT on
        ``timestamp_epoch_ms``. Staleness is a question about the pipeline —
        "is data still arriving?" — and the answer must not change because a
        video's internal clock runs fast or slow.

        Note the consequence, which callers must respect: staleness and the
        historical lookup use DIFFERENT time bases. When a recorded file is
        processed faster than real time, ``timestamp_epoch_ms`` advances at
        the video's rate (a 60-minute file replayed in 5 minutes ends up an
        hour ahead of the wall clock) while ``received_at`` tracks real time.
        Asking for "30 seconds ago" in wall-clock terms then finds nothing,
        because the stored stamps live on the other clock.

        ``FusionEngine`` resolves this by deriving its own "now" from the
        stored stamps (see ``reference_epoch_ms``) instead of from
        ``time.time()``, so both sides of the comparison share one clock.
        """
        with self._lock:
            latest = self._latest.get(camera_id)
            if latest is None:
                return True
            return (time.time() - latest.received_at) > threshold_sec

    def reference_epoch_ms(self) -> Optional[int]:
        """
        The newest ``timestamp_epoch_ms`` across all cameras, or None.

        This is the "now" the fusion engine must reason in. Using
        ``time.time()`` instead works only when every source is live; on
        recorded footage the stored stamps sit on the video's timeline, and a
        wall-clock "now" misses every sample in the buffer — which then looked
        like "no data" or, before the fix above, silently returned the current
        value for every historical query.

        Newest rather than oldest: a camera that stopped feeding should not
        drag the reference backwards and make live cameras look like the
        future. Staleness handles the stopped camera separately.
        """
        with self._lock:
            if not self._latest:
                return None
            return max(s.timestamp_epoch_ms for s in self._latest.values())

    def get_history(self, camera_id: str, window_sec: float = 300.0) -> List[CameraMetricSnapshot]:
        """Return historical snapshots within the last window_sec seconds."""
        with self._lock:
            buf = self._history.get(camera_id)
            if not buf:
                return []
            cutoff_time = time.time() - window_sec
            return [s for s in buf if s.received_at >= cutoff_time]

    def get_historical_flow_rate(
        self,
        camera_id: str,
        target_epoch_ms: int,
        tolerance_ms: int = 15000,
    ) -> Optional[float]:
        """
        Flow rate at ``target_epoch_ms``, or None if no sample is close enough.

        Used for the delayed upstream lookup, flow_rate(Ui, t - travel_time).

        Returns None rather than substituting the CURRENT value.
        ------------------------------------------------------------------
        This previously fell through to `self._latest` whenever no sample sat
        within tolerance — with no bound, despite the comment claiming the
        buffer was only "slightly out of range". The effect was that a lookup
        for "30 seconds ago" silently returned "right now", so the fusion
        engine compared the present against the present while reporting a
        travel-time offset it had not actually applied. The prediction still
        produced a confident number; it just wasn't a prediction.

        Substituting a value that is not the requested measurement is the
        specific failure this whole module has to avoid, so the miss is
        reported and the caller decides (see FusionEngine, which drops that
        edge and marks the target's forecast incomplete).
        """
        snap = self.get_historical_snapshot(camera_id, target_epoch_ms, tolerance_ms)
        return snap.flow_rate_pax_min if snap is not None else None

    def get_historical_snapshot(
        self,
        camera_id: str,
        target_epoch_ms: int,
        tolerance_ms: int = 15000,
    ) -> Optional[CameraMetricSnapshot]:
        """
        Closest snapshot to ``target_epoch_ms`` within ``tolerance_ms``, else None.

        Returns the whole snapshot so callers can read the calibration flags
        alongside the value — a flow rate is only comparable to a capacity if
        it was calibrated when it was recorded, and that fact travels with the
        sample rather than being assumed at the point of use.
        """
        with self._lock:
            buf = self._history.get(camera_id)
            if not buf:
                return None

            closest: Optional[CameraMetricSnapshot] = None
            min_diff = float("inf")
            for snap in buf:
                diff = abs(snap.timestamp_epoch_ms - target_epoch_ms)
                if diff < min_diff:
                    min_diff = diff
                    closest = snap

            if closest is not None and min_diff <= tolerance_ms:
                return closest
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _snapshot_from_dict(self, d: dict) -> CameraMetricSnapshot:
        dv = d.get("dominant_direction_vector") or [0.0, 0.0]
        return CameraMetricSnapshot(
            camera_id=d["camera_id"],
            timestamp_epoch_ms=int(d.get("timestamp_epoch_ms", 0)),
            density=float(d.get("density", 0.0)),
            flow_rate_pax_min=float(d.get("flow_rate_pax_min", 0.0)),
            dominant_direction_vector=(float(dv[0]), float(dv[1])),
            crush_risk_score=float(d.get("crush_risk_score", 0.0)),
            person_count=int(d.get("person_count", 0)),
            raw_timestamp_sec=float(d.get("raw_timestamp_sec", 0.0)),
            received_at=float(d.get("received_at", 0.0)),
            flow_is_calibrated=bool(d.get("flow_is_calibrated", False)),
            density_is_calibrated=bool(d.get("density_is_calibrated", False)),
            units=str(d.get("units", "uncalibrated")),
        )

    def flush(self, force: bool = False) -> bool:
        """
        Mirror the buffers to disk. Returns True if a write happened.

        Rate-limited to ``_FLUSH_INTERVAL_SEC`` unless ``force``. Written to a
        temp file and renamed, so a crash mid-write cannot leave a truncated
        JSON that fails to parse on the next start.
        """
        if not self.persist_path:
            return False
        now = time.time()
        with self._lock:
            if not force and (not self._dirty
                              or now - self._last_flush < _FLUSH_INTERVAL_SEC):
                return False
            payload = {
                "version": 1,
                "saved_at": now,
                "history": {cid: [s.to_dict() for s in buf]
                            for cid, buf in self._history.items()},
            }
            self._dirty = False
            self._last_flush = now
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)) or ".",
                        exist_ok=True)
            tmp = self.persist_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, self.persist_path)
            return True
        except Exception as exc:  # noqa: BLE001
            # Losing the audit trail is bad; taking the monitoring down with it
            # would be worse. Warned, not raised.
            logger.warning("MetricStore could not persist to %s: %s",
                           self.persist_path, exc)
            return False

    def load(self) -> int:
        """
        Restore buffers from disk. Returns the number of samples loaded.

        Samples keep their ORIGINAL ``received_at``, so ``is_stale`` correctly
        reports a camera that stopped feeding before the restart as stale
        rather than resurrecting it as live. Restoring history is about giving
        the fusion engine something to do its time-shifted lookups against, not
        about pretending the cameras are still streaming.
        """
        if not self.persist_path or not os.path.exists(self.persist_path):
            return 0
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("MetricStore state at %s is unreadable (%s); "
                           "starting empty.", self.persist_path, exc)
            return 0

        loaded = 0
        with self._lock:
            for cid, rows in (payload.get("history") or {}).items():
                buf: Deque[CameraMetricSnapshot] = deque(maxlen=self.max_history)
                for row in rows:
                    try:
                        buf.append(self._snapshot_from_dict(row))
                        loaded += 1
                    except (KeyError, TypeError, ValueError):
                        continue          # skip a damaged row, keep the rest
                if buf:
                    self._history[cid] = buf
                    self._latest[cid] = buf[-1]
        if loaded:
            logger.info("MetricStore restored %d sample(s) for %d camera(s) from %s",
                        loaded, len(self._history), self.persist_path)
        return loaded

    def get_all_camera_ids(self) -> List[str]:
        with self._lock:
            return list(self._latest.keys())

    def clear(self) -> None:
        with self._lock:
            self._latest.clear()
            self._history.clear()


METRIC_STORE = MetricStore(persist_path=_DEFAULT_PERSIST_PATH)
