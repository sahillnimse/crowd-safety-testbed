"""
LiveStreamHub: thread-safe pub/sub bridge for live preview frame and telemetry streaming.

Worker threads produce annotated frames and incremental KPIs, which are delivered
to FastAPI WebSocket subscribers running on the main asyncio event loop.

CRITICAL THREAD-SAFETY RULE:
Worker threads never interact with asyncio.Queue methods directly. They call
LIVE_HUB.broadcast(job_id, payload), which delegates to the captured main event
loop via loop.call_soon_threadsafe().
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class LiveStreamHub:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running event loop at webapp startup (during lifespan)."""
        self._loop = loop

    def register(self, job_id: str, q: asyncio.Queue) -> None:
        """Register a subscriber queue for a specific job_id."""
        if job_id not in self._subscribers:
            self._subscribers[job_id] = set()
        self._subscribers[job_id].add(q)

    def unregister(self, job_id: str, q: asyncio.Queue) -> None:
        """Unregister a subscriber queue."""
        if job_id in self._subscribers:
            self._subscribers[job_id].discard(q)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    def has_subscribers(self, job_id: str) -> bool:
        """True if there is at least one active subscriber for job_id."""
        return bool(self._subscribers.get(job_id))

    def broadcast(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Broadcast a message from ANY thread to all subscribers of job_id.

        Uses loop.call_soon_threadsafe to schedule queue delivery onto the
        main asyncio event loop.
        """
        if self._loop is None or self._loop.is_closed():
            return

        subs = self._subscribers.get(job_id)
        if not subs:
            return

        # Snapshot subscribers to safely iterate
        target_queues = list(subs)

        def _deliver() -> None:
            for q in target_queues:
                try:
                    if q.full():
                        try:
                            q.get_nowait()
                        except (asyncio.QueueEmpty, ValueError):
                            pass
                    q.put_nowait(payload)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to deliver live frame payload: %s", exc)

        try:
            self._loop.call_soon_threadsafe(_deliver)
        except RuntimeError:
            # Loop might be closing during shutdown
            pass


LIVE_HUB = LiveStreamHub()
