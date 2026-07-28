"""
Shared tracking-state logic for traffic models.

Any detector (YOLO, RT-DETR, Roboflow) hands its raw per-frame boxes to
this tracker, which assigns persistent IDs and classifies each ID as
"moving" or "parked" based on how much its centroid has drifted over a
recent window of time.

This mirrors models/fall/_tracker.py's role for pose-based fall models —
kept separate from any one detector so all traffic models share identical
moving/parked classification logic instead of re-implementing it per file.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class _TrackHistory:
    positions: deque = field(default_factory=lambda: deque(maxlen=90))  # ~3s @30fps
    last_seen_frame: int = 0
    vehicle_class: str = "vehicle"


class ParkedMovingClassifier:
    """
    Wraps a tracker's raw (track_id -> bbox) output per frame and decides
    moving vs parked per track_id based on centroid displacement.

    parked_window_sec: how far back to look when deciding "parked"
    parked_radius_px: max centroid drift (in pixels) over that window to
                      still count as "parked" rather than "moving"
    """

    def __init__(self, fps: float = 30.0, parked_window_sec: float = 3.0,
                 parked_radius_px: float = 15.0):
        self.fps = fps
        self.window_frames = max(1, int(parked_window_sec * fps))
        self.parked_radius_px = parked_radius_px
        self._tracks: dict[int, _TrackHistory] = {}

    def update(self, frame_index: int, tracked_boxes: list[dict]) -> list[dict]:
        """
        tracked_boxes: list of {"track_id": int, "bbox": [x1,y1,x2,y2],
                                 "vehicle_class": str, "confidence": float}

        Returns the same list with an added "status": "moving"/"parked" key.
        """
        results = []
        seen_ids = set()

        for det in tracked_boxes:
            tid = det["track_id"]
            seen_ids.add(tid)
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            hist = self._tracks.setdefault(tid, _TrackHistory())
            hist.vehicle_class = det.get("vehicle_class", hist.vehicle_class)
            hist.last_seen_frame = frame_index
            hist.positions.append((frame_index, cx, cy))

            status = self._classify(hist)
            results.append({**det, "status": status})

        # prune tracks not seen in a while to bound memory
        stale = [tid for tid, h in self._tracks.items()
                 if frame_index - h.last_seen_frame > self.window_frames * 2]
        for tid in stale:
            del self._tracks[tid]

        return results

    def _classify(self, hist: _TrackHistory) -> str:
        if len(hist.positions) < 2:
            return "moving"  # not enough history yet — default to moving (safer for counting)

        oldest_frame = hist.positions[0][0]
        newest_frame = hist.positions[-1][0]
        if newest_frame - oldest_frame < self.window_frames * 0.5:
            return "moving"  # hasn't been tracked long enough to judge "parked" confidently

        xs = [p[1] for p in hist.positions]
        ys = [p[2] for p in hist.positions]
        drift = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5

        return "parked" if drift <= self.parked_radius_px else "moving"