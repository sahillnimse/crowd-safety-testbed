"""
Head counting for crowd density and RT-DETRv2/APGCC fusion.

APGCC (VGG16-BN encoder, IFI decoder) replaces the earlier ResNet-18
density-map net entirely. It supplies per-point head detections used two
ways:

    models/crowd_flow/density.py            rho for Helbing crowd pressure
    models/crowd_flow/crowd_motion_monitor  fills gaps RT-DETRv2 misses

    HeadCounter            use it directly (density.py's own instance)
    get_head_counter()     shared cached instance (crowd_motion_monitor)
"""

import threading
from typing import Optional

from models.head_count.infer import HeadCounter

__all__ = ["HeadCounter", "get_head_counter"]

_CACHE: dict[tuple, HeadCounter] = {}
_CACHE_LOCK = threading.Lock()


def get_head_counter(
    weights: Optional[str] = None,
    device: Optional[str] = None,
    score_threshold: float = 0.5,
) -> HeadCounter:
    """
    Shared HeadCounter for (weights, device). Mirrors get_detector() in
    models/_detectors.py.
    """
    key = (weights, device or "cpu", score_threshold)
    with _CACHE_LOCK:
        hc = _CACHE.get(key)
        if hc is None:
            hc = HeadCounter(weights=weights, device=device,
                              score_threshold=score_threshold)
            _CACHE[key] = hc
    return hc
