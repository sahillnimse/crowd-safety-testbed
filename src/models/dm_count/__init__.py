"""
DM-Count density-map crowd monitoring package.

    DMCountCrowdMonitor   flow_pair wrapper (the registry model)
    get_dm_count_counter  shared cached counter instance

APGCC (models/head_count/) answers "where are individual heads" with point
queries; DM-Count answers "how many heads are here" with a density integral
and hands back local maxima as points. The two are complementary counters,
not duplicates: this package exists because the density-map family was the
one counting model this testbed did not have.
"""

import threading
from typing import Optional

from models.dm_count.infer import DMCountCounter
from models.dm_count.monitor import DMCountCrowdMonitor

__all__ = ["DMCountCounter", "DMCountCrowdMonitor", "get_dm_count_counter"]

_CACHE: dict[tuple, DMCountCounter] = {}
_CACHE_LOCK = threading.Lock()


def get_dm_count_counter(
    weights: Optional[str] = None,
    device: Optional[str] = None,
    peak_min_distance_px: int = 6,
    peak_value_thresh: float = 0.06,
    max_long_side: int = 960,
) -> DMCountCounter:
    """
    Shared DMCountCounter for (weights, device, knobs). Mirrors
    get_head_counter() in models/head_count/__init__.py.
    """
    key = (weights, device or "cpu", peak_min_distance_px,
           peak_value_thresh, max_long_side)
    with _CACHE_LOCK:
        counter = _CACHE.get(key)
        if counter is None:
            counter = DMCountCounter(
                weights=weights, device=device,
                peak_min_distance_px=peak_min_distance_px,
                peak_value_thresh=peak_value_thresh,
                max_long_side=max_long_side,
            )
            _CACHE[key] = counter
    return counter
