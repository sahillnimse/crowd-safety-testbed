"""
Getting an alert out of the process and in front of a human.

Why this exists
---------------
Every alert this system raised ended its life as a row in a JSON file under
outputs/. Nothing sent it anywhere. A crush-precursor alert at Ram Kund at
03:14 would sit in `detections.json` until somebody opened the file, which at
a mass gathering is the same as not detecting it at all.

Detection without delivery is not a safety system. This module is the
delivery half.

Design
------
Sinks are configured by environment variable so a deployment can add a
webhook without a code change, and the set is additive: every configured sink
gets every alert.

  CROWD_ALERT_WEBHOOK   POST one JSON object per alert batch
  CROWD_ALERT_LOG       append one JSON line per alert (default: on)
  CROWD_ALERT_MIN_SEV   "warning" (default) or "critical"

Delivery runs on a background thread with a bounded queue. Three properties
matter more than throughput:

1. It must never block the pipeline. A slow or hanging webhook cannot be
   allowed to stall frame processing -- the monitoring is more important
   than the notification.
2. It must never raise into the pipeline. A DNS failure is not a reason to
   kill a running camera.
3. It must not silently drop alerts. When the queue overflows the count is
   reported, because "we sent nothing and told no one" is the failure this
   module exists to prevent.

What this is NOT
----------------
Not a replacement for a control-room protocol, and not a guaranteed-delivery
bus. There is no retry across a process restart and no acknowledgement. A
webhook that is down during an outage loses those alerts. For life-safety use
this must feed something durable (a message broker, a paging service) and the
human protocol must not depend on any single channel.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_WEBHOOK = "CROWD_ALERT_WEBHOOK"
_ENV_LOG = "CROWD_ALERT_LOG"
_ENV_MIN_SEV = "CROWD_ALERT_MIN_SEV"

# Bounded so a wedged sink cannot grow without limit and take the process out
# of memory. Overflow is counted and reported rather than ignored.
_QUEUE_MAX = 1000
_WEBHOOK_TIMEOUT_SEC = 5.0

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class AlertEvent:
    """One alert, flattened to what a downstream consumer needs."""
    camera_id: str
    label: str
    severity: str
    metric: str
    value: float
    threshold: float
    timestamp_sec: float
    frame_index: int
    zone: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "label": self.label,
            "severity": self.severity,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "frame_index": self.frame_index,
            "zone": self.zone,
            "source": self.source,
            "sent_at": time.time(),
            **({"extra": self.extra} if self.extra else {}),
        }


class AlertDispatcher:
    """Fan one alert out to every configured sink, off the hot path."""

    def __init__(self, webhook: Optional[str] = None,
                 log_path: Optional[str] = None,
                 min_severity: Optional[str] = None) -> None:
        self.webhook = webhook if webhook is not None else os.environ.get(_ENV_WEBHOOK)
        self.log_path = log_path if log_path is not None else os.environ.get(_ENV_LOG)
        self.min_severity = (min_severity
                             or os.environ.get(_ENV_MIN_SEV, "warning")).lower()
        self._q: "queue.Queue[AlertEvent]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.dropped = 0
        self.sent = 0
        self.failed = 0

    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.webhook or self.log_path)

    def start(self) -> None:
        if self._thread is not None or not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name="alert-dispatcher")
        self._thread.start()
        logger.info("Alert dispatcher started (webhook=%s, log=%s, min_sev=%s)",
                    bool(self.webhook), self.log_path or "-", self.min_severity)

    def stop(self, timeout: float = 5.0) -> None:
        """Flush and stop. Called at shutdown so queued alerts still go out."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        if self.dropped:
            logger.error("Alert dispatcher dropped %d alert(s) - the queue "
                         "overflowed, so those alerts reached NOBODY.",
                         self.dropped)

    # ------------------------------------------------------------------

    def _passes_severity(self, severity: str) -> bool:
        return (_SEVERITY_ORDER.get(severity.lower(), 1)
                >= _SEVERITY_ORDER.get(self.min_severity, 1))

    def dispatch(self, event: AlertEvent) -> bool:
        """
        Queue one alert. Returns False if it was dropped.

        Never blocks and never raises: called from the frame loop, where both
        would stop the monitoring this alert is about.
        """
        if not self.enabled or not self._passes_severity(event.severity):
            return False
        if self._thread is None:
            self.start()
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self.dropped += 1
            if self.dropped in (1, 10, 100) or self.dropped % 500 == 0:
                logger.error("Alert queue FULL - dropped %d alert(s) so far. "
                             "A sink is not keeping up and alerts are being "
                             "lost.", self.dropped)
            return False

    # ------------------------------------------------------------------

    def _drain(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                event = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            payload = event.to_dict()
            for send in (self._send_log, self._send_webhook):
                try:
                    send(payload)
                except Exception as exc:  # noqa: BLE001 - a sink must not kill the thread
                    self.failed += 1
                    logger.warning("Alert sink %s failed: %s",
                                   send.__name__, exc)
            self.sent += 1

    def _send_log(self, payload: dict) -> None:
        if not self.log_path:
            return
        # One JSON object per line: append-only, greppable, and survives a
        # crash mid-write without corrupting earlier records.
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".",
                    exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _send_webhook(self, payload: dict) -> None:
        if not self.webhook:
            return
        import urllib.request
        req = urllib.request.Request(
            self.webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_WEBHOOK_TIMEOUT_SEC) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"webhook returned HTTP {resp.status}")

    def stats(self) -> dict:
        return {"enabled": self.enabled, "sent": self.sent,
                "failed": self.failed, "dropped": self.dropped,
                "queued": self._q.qsize()}


#: Process-wide dispatcher. One instance so every model shares the queue and
#: the drop counter reflects the whole process rather than one camera.
DISPATCHER = AlertDispatcher()


def dispatch_detections(detections: list, camera_id: str, source: str = "") -> int:
    """
    Send every alert-shaped Detection in ``detections``.

    Rows that are not alerts (per-frame telemetry like "flow_analysis" or
    "dm_frame_metrics") are skipped: they are emitted every frame by design
    and would bury a real alert in noise, which is its own failure mode.

    Returns the number queued.
    """
    if not DISPATCHER.enabled or not detections:
        return 0

    skip = {"flow_analysis", "dm_frame_metrics"}
    queued = 0
    for det in detections:
        label = getattr(det, "label", "") or ""
        if label in skip:
            continue
        extra = getattr(det, "extra", None) or {}
        severity = str(extra.get("severity")
                       or ("critical" if label.endswith("_critical") else "warning"))
        event = AlertEvent(
            camera_id=camera_id,
            label=label,
            severity=severity,
            metric=str(extra.get("metric_name") or label),
            value=float(extra.get("measured_value") or getattr(det, "confidence", 0.0) or 0.0),
            threshold=float(extra.get("threshold_value") or 0.0),
            timestamp_sec=float(getattr(det, "timestamp_sec", 0.0) or 0.0),
            frame_index=int(getattr(det, "frame_index", 0) or 0),
            zone=str(extra.get("zone_name") or extra.get("zone") or ""),
            source=source,
        )
        if DISPATCHER.dispatch(event):
            queued += 1
    return queued
