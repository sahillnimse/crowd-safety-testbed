"""
Shared greedy IoU tracker.

Used across the project by anything that needs stable per-object identity
between frames without a learned tracker: the pose-based fall models, the
umbrella detectors, ANPR's vehicle capture, and the dense-flow cross-family
validation route.

It lived in ``models/fall/`` until five other families were importing it from
there — ANPR reaching into the fall package to track vehicles is the sort of
dependency that reads as an accident, because it was one: fall detection
simply needed it first.  See models/traffic/_tracker.py for the traffic
family's own centroid/parked classifier, which is a different thing.

Replaces the per-wrapper trackers, which had two bugs that quietly
corrupted every temporal model built on them:

1. **Track IDs were not exclusive.** Each detection was matched against a
   `last_boxes` snapshot that was never consumed, so in a crowd two
   overlapping people could both be assigned the same track ID in the same
   frame — and both appended to that track's sequence. The LSTM/ST-GCN
   downstream then classified a blend of two bodies. Matching here is
   greedy but strictly one-to-one: best IoU pair first, and both the
   detection and the track are consumed on assignment.

2. **Tracks were never evicted.** Every person who ever appeared stayed in
   the dict for the rest of the video, so per-frame matching cost grew
   without bound and long-dead boxes kept competing for matches. Tracks
   unseen for `max_age` frames are dropped.

The tracker also carries each track's recent per-frame payload (keypoints,
posture flags, ...), which is what lets the pose wrappers apply temporal
confirmation instead of firing on a single frame.
"""

from collections import deque
from typing import Any, Optional

import numpy as np


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """IoU of every box in `boxes_a` against every box in `boxes_b`.

    Same value as calling `iou()` on each pair, computed as one numpy
    broadcast instead of a Python double loop.  In a dense crowd the pair
    count is the whole cost of tracking: 620 detections against 650 live
    tracks is ~400k `iou()` calls per frame, which measured at ~1.0 s of the
    ~2.2 s frame budget on 720p footage.  Vectorised, the same matrix takes
    single-digit milliseconds.

    Returns an (len(a), len(b)) float64 array; empty inputs give an
    appropriately shaped empty array rather than raising.
    """
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # Checked rather than reshaped: the scalar iou() unpacks exactly four
    # values and raises on anything else, and silently folding a wrongly
    # shaped array into (-1, 4) here would turn that error into plausible
    # nonsense.
    for name, arr in (("boxes_a", a), ("boxes_b", b)):
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(
                f"{name} must be a sequence of [x1, y1, x2, y2]; got shape "
                f"{arr.shape}"
            )

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = (np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1))
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    # union <= 0 means both boxes are degenerate; the scalar iou() returns
    # 0.0 there, so divide against a safe denominator and mask afterwards.
    safe = np.where(union > 0.0, union, 1.0)
    return np.where(union > 0.0, inter / safe, 0.0)


class _Track:
    __slots__ = ("bbox", "last_seen", "states", "age")

    def __init__(self, bbox, frame_index: int, history_len: int):
        self.bbox = bbox
        self.last_seen = frame_index
        self.age = 1
        self.states: deque = deque(maxlen=history_len)


class IoUTracker:
    """Greedy one-to-one IoU tracker with age-based eviction.

    Good enough for the short temporal windows these models need. If
    identity switches show up in dense crowds, swap this for ByteTrack /
    DeepSORT — the wrappers only depend on `update()` returning one ID per
    input box, in order.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30,
                 history_len: int = 64):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.history_len = history_len
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    def update(self, boxes: list, frame_index: int) -> list[int]:
        """Assign a track ID to each box. Returns IDs aligned with `boxes`."""
        self._evict(frame_index)

        assigned: list[Optional[int]] = [None] * len(boxes)
        if boxes and self._tracks:
            # Score every (detection, track) pair, then take them best-first,
            # skipping any pair whose detection or track is already spoken
            # for. This is what makes the assignment one-to-one.
            #
            # The scoring is one numpy broadcast rather than a Python double
            # loop: at crowd scale (hundreds of boxes against hundreds of
            # live tracks) the pair count runs to six figures per frame and
            # dominated the whole pipeline.  Only the greedy walk below stays
            # in Python, and it visits just the pairs that clear the
            # threshold.
            tids = list(self._tracks.keys())
            scores = iou_matrix(boxes, [self._tracks[t].bbox for t in tids])

            # Row-major flat indices run detection-outer / track-inner, which
            # is the order the old nested loop appended candidates in, and
            # the sort is stable — so equal scores are still broken the same
            # way and the assignment is unchanged.
            flat = np.flatnonzero(scores.ravel() >= self.iou_threshold)
            order = flat[np.argsort(-scores.ravel()[flat], kind="stable")]

            n_tracks = len(tids)
            taken_dets: set[int] = set()
            taken_tracks: set[int] = set()
            for pair in order.tolist():
                det_i, trk_i = divmod(pair, n_tracks)
                if det_i in taken_dets or trk_i in taken_tracks:
                    continue
                assigned[det_i] = tids[trk_i]
                taken_dets.add(det_i)
                taken_tracks.add(trk_i)

        ids: list[int] = []
        for det_i, box in enumerate(boxes):
            tid = assigned[det_i]
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(box, frame_index, self.history_len)
            else:
                track = self._tracks[tid]
                track.bbox = box
                track.last_seen = frame_index
                track.age += 1
            ids.append(tid)
        return ids

    def _evict(self, frame_index: int) -> None:
        stale = [tid for tid, t in self._tracks.items()
                 if frame_index - t.last_seen > self.max_age]
        for tid in stale:
            del self._tracks[tid]

    def append_state(self, track_id: int, state: Any) -> None:
        """Record this frame's payload for a track (keypoints, flags, ...)."""
        track = self._tracks.get(track_id)
        if track is not None:
            track.states.append(state)

    def states(self, track_id: int) -> list:
        track = self._tracks.get(track_id)
        return list(track.states) if track is not None else []

    def age(self, track_id: int) -> int:
        track = self._tracks.get(track_id)
        return track.age if track is not None else 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._tracks)


def sustained(flags, k: int) -> bool:
    """True if the last `k` entries are all truthy (and there are `k` of them).

    Temporal confirmation: a genuine fall stays down, whereas the frame-level
    false positives that dominate crowd footage (a crouch, a bend, one bad
    pose estimate) flicker. Requiring K consecutive positive frames on the
    same track is the single cheapest precision win available to the
    single-frame pose wrappers.
    """
    flags = list(flags)
    if len(flags) < k:
        return False
    return all(flags[-k:])
