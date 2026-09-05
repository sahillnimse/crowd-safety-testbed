"""
Shared tracking-state logic for traffic models.

Any detector (YOLO, RT-DETR, Roboflow) hands its raw per-frame boxes to
this tracker, which assigns persistent IDs and classifies each ID as
"moving" or "parked" based on how much its centroid has drifted over a
recent window of time.

This mirrors models/_tracker.py's role for the shared IoU tracker —
kept separate from any one detector so all traffic models share identical
moving/parked classification logic instead of re-implementing it per file.

**Everything here is measured in seconds, never in frames.** The pipeline
runner processes only every Nth frame (`sample_every_n_frames`), so
`frame_index` advances N per call. Timing this in frames made the window
stretch with the sampling stride: at the default stride of 5, a nominal
"3 second" window actually spanned 14.8 seconds of video, and the same
clip reported 33% of vehicles parked at stride 1 but only 10% at stride 5.
Deriving the window from `timestamp_sec` makes the answer independent of
how densely the video was sampled — which is the only way two runs are
comparable.

For the same reason there is no `fps` parameter any more. It was hardcoded
to 30.0 and never read from the video, so every window was silently wrong
on any clip that wasn't 30 fps.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# Below this effective sampling rate, ByteTrack's IoU-based association
# starts failing: vehicles move too far between the frames it actually
# sees, so IDs churn and tracks fragment. Measured on this repo's test
# clip, stride 5 (6 fps effective) more than halved the distinct track
# count versus stride 1 — a tracking failure, not a scene change.
_MIN_HEALTHY_FPS = 10.0


@dataclass
class _TrackHistory:
    # (timestamp_sec, cx, cy) — bounded by time in update(), not by a fixed
    # maxlen, which previously disagreed with the configured window.
    positions: deque = field(default_factory=deque)
    last_seen_sec: float = 0.0
    vehicle_class: str = "vehicle"


class ParkedMovingClassifier:
    """
    Wraps a tracker's raw (track_id -> bbox) output per frame and decides
    moving vs parked per track_id based on centroid displacement.

    parked_window_sec: how far back to look when deciding "parked"
    parked_radius_px: max centroid drift (in pixels) over that window to
                      still count as "parked" rather than "moving"
    min_samples: how many observations are needed before judging at all;
                 guards against calling something parked off two frames
    """

    def __init__(self, parked_window_sec: float = 3.0,
                 parked_radius_px: float = 15.0, min_samples: int = 3,
                 model_name: str = "traffic"):
        self.parked_window_sec = parked_window_sec
        self.parked_radius_px = parked_radius_px
        self.min_samples = min_samples
        self.model_name = model_name
        self._tracks: dict[int, _TrackHistory] = {}
        self._last_update_sec: Optional[float] = None
        self._intervals: deque = deque(maxlen=30)
        self._warned_sampling = False

    def update(self, timestamp_sec: float, tracked_boxes: list[dict]) -> list[dict]:
        """
        timestamp_sec: presentation time of this frame, from the runner.
        tracked_boxes: list of {"track_id": int, "bbox": [x1,y1,x2,y2],
                                 "vehicle_class": str, "confidence": float}

        Returns the same list with an added "status": "moving"/"parked" key.
        """
        self._check_sampling_rate(timestamp_sec)

        results = []
        for det in tracked_boxes:
            tid = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            hist = self._tracks.setdefault(tid, _TrackHistory())
            hist.vehicle_class = det.get("vehicle_class", hist.vehicle_class)
            hist.last_seen_sec = timestamp_sec
            hist.positions.append((timestamp_sec, cx, cy))

            # Drop observations that have fallen out of the time window, so
            # drift is always measured over exactly parked_window_sec.
            cutoff = timestamp_sec - self.parked_window_sec
            while hist.positions and hist.positions[0][0] < cutoff:
                hist.positions.popleft()

            results.append({**det, "status": self._classify(hist)})

        # prune tracks not seen in a while to bound memory
        stale = [tid for tid, h in self._tracks.items()
                 if timestamp_sec - h.last_seen_sec > self.parked_window_sec * 2]
        for tid in stale:
            del self._tracks[tid]

        return results

    def _classify(self, hist: _TrackHistory) -> str:
        if len(hist.positions) < self.min_samples:
            return "moving"  # not enough history yet — default to moving (safer for counting)

        span = hist.positions[-1][0] - hist.positions[0][0]
        if span < self.parked_window_sec * 0.5:
            return "moving"  # hasn't been tracked long enough to judge "parked" confidently

        xs = [p[1] for p in hist.positions]
        ys = [p[2] for p in hist.positions]
        drift = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5

        return "parked" if drift <= self.parked_radius_px else "moving"

    def _check_sampling_rate(self, timestamp_sec: float) -> None:
        """Warn once if the runner is sampling too sparsely for tracking.

        The parked/moving window is stride-independent now, but the
        *detector's* tracker is not: ByteTrack matches boxes between the
        frames it's given, so a large stride breaks ID continuity no matter
        what this class does. Silent track fragmentation looks like vehicles
        entering and leaving, so it's worth saying out loud.
        """
        if self._last_update_sec is not None:
            delta = timestamp_sec - self._last_update_sec
            if delta > 0:
                self._intervals.append(delta)
        self._last_update_sec = timestamp_sec

        if self._warned_sampling or len(self._intervals) < 10:
            return

        median = sorted(self._intervals)[len(self._intervals) // 2]
        effective_fps = 1.0 / median if median > 0 else 0.0
        if effective_fps < _MIN_HEALTHY_FPS:
            self._warned_sampling = True
            print(
                f"[{self.model_name}] WARNING: frames are arriving at only "
                f"~{effective_fps:.1f} fps (sample_every_n_frames is set too "
                f"high). Vehicle tracking degrades badly below "
                f"~{_MIN_HEALTHY_FPS:.0f} fps because ByteTrack matches boxes "
                f"between consecutive processed frames - expect fragmented "
                f"track IDs and unreliable parked/moving counts. Use stride 1-2 "
                f"for traffic."
            )

    def reset(self) -> None:
        self._tracks.clear()
        self._last_update_sec = None
        self._intervals.clear()
        self._warned_sampling = False


def _stable_track_id(track_id) -> int:
    """DeepSORT IDs are strings. Use int() when possible, and a stable hash
    otherwise — Python's built-in hash() is randomized per process, so it
    would give the same vehicle a different ID on every run.

    Lives here rather than beside one detector: it is shared by the traffic
    wrappers, and it previously sat in the RT-DETR module purely because that
    was the first place that needed it.
    """
    text = str(track_id)
    if text.isdigit():
        return int(text)
    import zlib
    return zlib.crc32(text.encode()) % 100000
