"""
Per-person boxes with a per-person motion arrow.

Why this exists alongside the dense overlay
-------------------------------------------
On 720p footage of a deep crowd, the honest per-pixel answer is often "not
measurable" — a person forty metres away is a dozen pixels tall, their frame
to frame displacement is a fraction of a pixel, and after validity gating
most of that region is correctly blank.  A dense colour overlay of that is
mostly empty, which is truthful but not useful to an operator.

What IS reliably recoverable at that resolution is: where the people are, and
roughly which way each is going.  A detector finds a person from their whole
appearance rather than from a sub-pixel brightness change, and averaging the
flow over a whole body integrates many weak measurements into one usable
vector.  So this draws a box per person and one short arrow per person,
instead of asking the viewer to read a per-pixel field that the footage
cannot support.

The three things that make boxes flicker, and what is done about each
--------------------------------------------------------------------
1. Detections toggle across the confidence threshold.  A distant person sits
   near the operating point and appears and disappears frame to frame.
   -> A track is CONFIRMED after being seen `confirm_frames` times and is
      only dropped after `max_age` frames of absence.  One missed detection
      no longer removes a person from the display.

2. Box coordinates jitter by a few pixels even on a stationary subject.
   -> Coordinates are smoothed with an EMA.  Position is a physical quantity
      that cannot change discontinuously; the jitter is measurement noise and
      is treated as such.

3. Running a detector on every frame is unaffordable, so naive code detects
   every Nth frame and the boxes visibly jump when they update.
   -> Between detector frames each box is ADVANCED BY THE FLOW FIELD that
      this module already computes.  The box follows the person continuously
      instead of freezing and then teleporting.  This is nearly free: the
      flow is computed regardless, and sampling it inside a box is a median
      over a few hundred values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Drawing (BGR).
_BOX_CONFIRMED = (120, 230, 120)
_BOX_COASTING = (90, 170, 220)      # detector has not seen it for a frame or two
_ARROW = (255, 220, 0)

# Pedestrian motion is 1-2 px/frame; drawn at true length an arrow would be a
# few pixels and invisible.  The factor is printed on the frame so nobody
# reads arrow length as a speed in pixels.
_ARROW_SCALE = 14
_MIN_ARROW_PX = 6.0

# Fraction each box is shrunk before the flow inside it is sampled, so the
# sample comes from the body rather than the background framed around it.
_BOX_SHRINK = 0.35


@dataclass
class _Track:
    box: np.ndarray                    # smoothed [x1, y1, x2, y2]
    track_id: int
    seen: int = 1                      # detector frames this track was matched
    missing: int = 0                   # frames since the detector last saw it
    vx: float = 0.0                    # smoothed velocity, source px/frame
    vy: float = 0.0
    confirmed: bool = False


class PeopleOverlay:
    """
    Tracks people across frames and renders boxes plus per-person arrows.

    Parameters
    ----------
    detect_every:
        Run the detector every Nth frame.  Between those frames boxes are
        carried by the flow field, so raising this costs tracking accuracy
        rather than visible smoothness.
    tile_grid:
        Detector tiling.  The processor resizes its input to 640x640, so a
        720p frame is halved before the model sees it and a distant person
        arrives at ~10 px.  Measured on this project's Nashik footage: 35
        people full-frame, 96 at 2x2, 109 at 3x3, 145 at 4x4.
    confirm_frames:
        Detector sightings before a track is drawn.  Suppresses one-frame
        false positives without hiding real people for long.
    max_age:
        Frames a track survives without being re-detected.  This is the main
        anti-flicker control: it is what carries a person through the frames
        where the detector loses them.
    box_smoothing:
        EMA factor for box coordinates.  0 = frozen, 1 = no smoothing.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        conf_threshold: float = 0.20,
        detect_every: int = 5,
        tile_grid: Optional[tuple[int, int]] = (3, 3),
        confirm_frames: int = 2,
        max_age: int = 30,
        box_smoothing: float = 0.45,
        iou_threshold: float = 0.3,
        draw_arrows: bool = True,
    ) -> None:
        self.device = device
        self.conf_threshold = conf_threshold
        self.detect_every = max(1, int(detect_every))
        self.tile_grid = tile_grid
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_age = max(1, int(max_age))
        self.box_smoothing = float(np.clip(box_smoothing, 0.05, 1.0))
        self.iou_threshold = iou_threshold
        self.draw_arrows = draw_arrows

        self._detector = None
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        # update() calls seen, which is what `detect_every` counts.  See update().
        self._calls = 0

    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._detector is not None:
            return
        from models._detectors import get_detector
        self._detector = get_detector(device=self.device)
        self._detector.load()

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._calls = 0

    @property
    def n_visible(self) -> int:
        return sum(1 for t in self._tracks.values() if t.confirmed)

    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        field_xy: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Advance every track, and re-detect on detector frames.

        ``detect_every`` counts CALLS, not source frame indices.  It used to
        test ``frame_index % detect_every``, and frame_index is the index in the
        source video while update() is invoked once per SAMPLED frame — so
        whenever the sampling interval was a multiple of detect_every the test
        was true every time and the detector ran on every frame.  That is the
        default configuration: the web UI samples every 5th frame and
        people_detect_every is 5, giving frame indices 0, 5, 10, 15 ... every
        one divisible by 5.
        """
        h, w = frame.shape[:2]
        due = (self._calls % self.detect_every) == 0
        self._calls += 1

        # 1. Carry every track forward on the flow field.  Done EVERY frame,
        #    including detector frames, so a box is never stationary while the
        #    person it belongs to is not.
        if field_xy is not None:
            for t in self._tracks.values():
                vx, vy = self._sample_flow(field_xy, t.box, valid_mask)
                if vx is not None:
                    # Velocity is smoothed as well as position: a single
                    # frame's flow inside one box is noisy, and the arrow
                    # direction is the thing an operator actually reads.
                    a = self.box_smoothing
                    t.vx = (1 - a) * t.vx + a * vx
                    t.vy = (1 - a) * t.vy + a * vy
                t.box = t.box + np.array([t.vx, t.vy, t.vx, t.vy], np.float32)
                np.clip(t.box, [0, 0, 0, 0], [w, h, w, h], out=t.box)

        if not due:
            for t in self._tracks.values():
                t.missing += 1
            self._evict()
            return

        # 2. Detector frame: match detections to tracks.
        self.load()
        from models._detectors import COCO_PERSON
        boxes = self._detector.detect(
            frame, classes=(COCO_PERSON,),
            conf_threshold=self.conf_threshold, tile_grid=self.tile_grid,
        )
        self._associate(np.asarray(boxes, np.float32).reshape(-1, 4))
        self._evict()

    # ------------------------------------------------------------------

    def _associate(self, dets: np.ndarray) -> None:
        """Greedy IoU matching, best pair first, one-to-one."""
        tracks = list(self._tracks.values())
        unmatched_d = set(range(len(dets)))
        unmatched_t = set(range(len(tracks)))

        # Vectorised IoU matrix.  A dense crowd yields several hundred tracks
        # and several hundred detections, and the nested Python loop this
        # replaces was ~130k calls per detector frame — which measured as the
        # dominant cost of the whole overlay, above the detector itself.
        pairs = []
        if tracks and len(dets):
            tb = np.stack([t.box for t in tracks])            # (T, 4)
            ix1 = np.maximum(tb[:, None, 0], dets[None, :, 0])
            iy1 = np.maximum(tb[:, None, 1], dets[None, :, 1])
            ix2 = np.minimum(tb[:, None, 2], dets[None, :, 2])
            iy2 = np.minimum(tb[:, None, 3], dets[None, :, 3])
            inter = (np.clip(ix2 - ix1, 0, None) *
                     np.clip(iy2 - iy1, 0, None))
            at = np.clip(tb[:, 2] - tb[:, 0], 0, None) * np.clip(tb[:, 3] - tb[:, 1], 0, None)
            ad = np.clip(dets[:, 2] - dets[:, 0], 0, None) * np.clip(dets[:, 3] - dets[:, 1], 0, None)
            union = at[:, None] + ad[None, :] - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

            ti_idx, di_idx = np.nonzero(iou >= self.iou_threshold)
            if len(ti_idx):
                vals = iou[ti_idx, di_idx]
                order = np.argsort(-vals)
                pairs = list(zip(vals[order], ti_idx[order], di_idx[order]))

        for _v, ti, di in pairs:
            if ti not in unmatched_t or di not in unmatched_d:
                continue
            t = tracks[ti]
            a = self.box_smoothing
            # EMA towards the detection rather than snapping to it: the
            # detector's box jitters by a few px on a stationary person, and
            # that jitter is what reads as flicker.
            t.box = ((1 - a) * t.box + a * dets[di]).astype(np.float32)
            t.seen += 1
            t.missing = 0
            if t.seen >= self.confirm_frames:
                t.confirmed = True
            unmatched_t.discard(ti)
            unmatched_d.discard(di)

        for ti in unmatched_t:
            tracks[ti].missing += 1

        for di in unmatched_d:
            self._tracks[self._next_id] = _Track(
                box=dets[di].astype(np.float32), track_id=self._next_id,
                confirmed=self.confirm_frames <= 1,
            )
            self._next_id += 1

    def _evict(self) -> None:
        dead = [k for k, t in self._tracks.items() if t.missing > self.max_age]
        for k in dead:
            del self._tracks[k]

    @staticmethod
    def _sample_flow(field_xy, box, valid_mask) -> tuple:
        """
        Median flow inside a shrunk box, over VALID pixels only.

        The median, not the mean: a box contains background around the body,
        and the mean is dragged by whatever that background is doing.  Valid
        pixels only, because averaging in flow the algorithm invented is
        exactly what the validity mask exists to prevent — on a distant person
        most of the box may be unmeasurable, and the honest answer is then to
        leave the velocity where it was rather than invent one.
        """
        h, w = field_xy.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        sx, sy = bw * _BOX_SHRINK * 0.5, bh * _BOX_SHRINK * 0.5
        ix1 = max(0, int(x1 + sx)); ix2 = min(w, int(x2 - sx))
        iy1 = max(0, int(y1 + sy)); iy2 = min(h, int(y2 - sy))
        if ix2 - ix1 < 2 or iy2 - iy1 < 2:
            return None, None
        patch = field_xy[iy1:iy2, ix1:ix2]
        if valid_mask is not None:
            m = valid_mask[iy1:iy2, ix1:ix2].astype(bool)
            if m.sum() < 4:
                return None, None
            return (float(np.median(patch[..., 0][m])),
                    float(np.median(patch[..., 1][m])))
        return (float(np.median(patch[..., 0])), float(np.median(patch[..., 1])))

    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Boxes and per-person arrows on a copy of the frame."""
        out = frame.copy()
        n = 0
        for t in self._tracks.values():
            if not t.confirmed:
                continue
            n += 1
            x1, y1, x2, y2 = (int(round(v)) for v in t.box)
            colour = _BOX_CONFIRMED if t.missing == 0 else _BOX_COASTING
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1, cv2.LINE_AA)

            if not self.draw_arrows:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ex = cx + t.vx * _ARROW_SCALE
            ey = cy + t.vy * _ARROW_SCALE
            if np.hypot(ex - cx, ey - cy) >= _MIN_ARROW_PX:
                cv2.arrowedLine(out, (cx, cy), (int(ex), int(ey)),
                                _ARROW, 1, cv2.LINE_AA, tipLength=0.35)

        cv2.putText(out, f"{n} people   arrows x{_ARROW_SCALE}",
                    (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"{n} people   arrows x{_ARROW_SCALE}",
                    (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out
