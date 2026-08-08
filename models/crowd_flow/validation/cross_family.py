"""
Route (c) — cross-family agreement.

A person detector plus a tracker gives a velocity per individual, computed
from machinery that shares nothing with dense optical flow: boxes and
identity association rather than brightness constancy.  Two instruments built
on different principles agreeing is evidence; two variants of the same method
agreeing is not.

Why this route earns its place
------------------------------
It is the only one of the three that sees real scene content — real motion
blur, real occlusion, and above all *independently moving objects*.  That is
exactly the class of failure the synthetic route is blind to.  This project's
global-motion-compensation bug (ORB locking onto moving vehicles and
subtracting their motion from the entire field, corrupting it by up to
179 px/frame) passed all eleven synthetic tests and would have failed here on
the first frame: a tracked pedestrian walking at 1 px/frame against a flow
field claiming 15 px/frame is an immediate, loud contradiction.

What it CANNOT validate
-----------------------
High density — which is the regime that matters most.  Tracking degrades as
crowds thicken: detections merge, identities swap, tracks fragment.  So this
route is reliable precisely where crush risk is lowest, and its own
disagreement statistic becomes untrustworthy exactly when the crowd gets
interesting.

The route therefore reports the mean person count per frame alongside the
agreement figures, and marks the result as density-limited above
``density_warn_persons``.  Reading a green result from a dense clip as
validation would be a mistake; the number is there to make that visible.

Neither is the tracker ground truth.  Both instruments can be wrong, and
agreement between them is evidence of correctness, not proof.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from models.crowd_flow.flow_field import FlowField
from models.crowd_flow.validation.report import (
    Measurement, RouteResult, STATUS_PASS,
)
from models.fall._tracker import IoUTracker

logger = logging.getLogger(__name__)

ROUTE_KEY   = "cross_family"
ROUTE_TITLE = "Route (c) — cross-family agreement"
ROUTE_CAVEAT = (
    "Works only at low density: tracking degrades as crowds thicken, so this "
    "route is most reliable where crush risk is lowest and cannot validate "
    "the dense regime at all.  The tracker is not ground truth either — "
    "agreement is evidence for both instruments, not proof of either."
)

# COCO class index for 'person'.
_PERSON_CLASS = 0

# Candidate detector weights, in preference order.
_WEIGHT_CANDIDATES = ("yolo11n.pt", "yolov8n.pt", "yolo11s.pt", "yolov8s.pt")

# Spread of tracked speeds (std, px/frame) below which a correlation between
# track speed and flow speed carries no information.
#
# A detection box jitters by roughly a pixel between frames even for someone
# standing still, so on a near-stationary clip the "track velocity" signal is
# mostly that jitter.  Correlating jitter against a flow field that correctly
# reports ~zero yields r ~ 0 — which says the clip was quiet, not that the
# estimator is wrong.  Below this spread the correlation is reported for
# information only and is not allowed to fail the route.
_MIN_SPEED_SPREAD_PX = 1.0

# Comparison-video drawing (BGR).
_TRACKER_COLOUR = (255, 255, 255)   # white  — the reference instrument
_FLOW_COLOUR    = (255, 220, 0)     # cyan   — the instrument under test
_BOX_AGREE      = (120, 220, 120)   # green
_BOX_DISAGREE   = (60, 90, 240)     # red

# Pedestrian motion is 1-2 px/frame; at true length the arrows would be a few
# pixels and unreadable.  Both instruments use the same factor, so the
# comparison stays fair, and the factor is printed on the frame.
_ARROW_SCALE = 12


@dataclass
class _TrackState:
    centroid: tuple[float, float]
    frame_index: int
    seen: int = 1


@dataclass
class Comparison:
    """One track's velocity compared against the flow field at its position."""
    frame_index: int
    track_id: int
    track_vx: float
    track_vy: float
    flow_vx: float
    flow_vy: float

    @property
    def track_speed(self) -> float:
        return float(np.hypot(self.track_vx, self.track_vy))

    @property
    def flow_speed(self) -> float:
        return float(np.hypot(self.flow_vx, self.flow_vy))

    @property
    def speed_error(self) -> float:
        return abs(self.flow_speed - self.track_speed)

    @property
    def angular_error_deg(self) -> float:
        na = self.track_speed
        nb = self.flow_speed
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        cos = (self.track_vx * self.flow_vx + self.track_vy * self.flow_vy) / (na * nb)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def find_person_weights(project_root: str) -> Optional[str]:
    """First available person-capable YOLO checkpoint in the project root."""
    for name in _WEIGHT_CANDIDATES:
        path = os.path.join(project_root, name)
        if os.path.exists(path):
            return path
    return None


class CrossFamilyValidator:
    """
    Compares tracked-person velocities against the dense flow field.

    Parameters
    ----------
    weights_path:
        YOLO checkpoint for person detection.  Auto-discovered if None.
    conf_threshold:
        Detection confidence floor.
    min_track_age:
        Frames a track must have been seen before its velocity is trusted.  A
        one-frame-old track has no velocity, and a two-frame-old one is
        dominated by detection jitter.
    min_track_speed_px:
        Tracks slower than this are skipped — a stationary person's direction
        is undefined, and comparing directions there measures noise.
    box_shrink:
        Fraction each bounding box is shrunk before sampling the flow, so the
        sample comes from the body rather than the background framed around
        it.  The flow is then taken as the MEDIAN over that inner region,
        which survives the background pixels that remain.
    density_warn_persons:
        Mean persons per frame above which the result is marked
        density-limited (tracking, and therefore this route, is no longer
        dependable).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        min_track_age: int = 3,
        min_track_speed_px: float = 0.5,
        box_shrink: float = 0.3,
        density_warn_persons: float = 25.0,
        max_speed_error_px: float = 1.5,
        max_angular_deg: float = 35.0,
        min_comparisons: int = 50,
    ) -> None:
        self.weights_path = weights_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.min_track_age = min_track_age
        self.min_track_speed_px = min_track_speed_px
        self.box_shrink = box_shrink
        self.density_warn_persons = density_warn_persons
        self.max_speed_error_px = max_speed_error_px
        self.max_angular_deg = max_angular_deg
        self.min_comparisons = min_comparisons
        self._model = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        if not self.weights_path or not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Person-detector weights not found: {self.weights_path!r}.  "
                f"Place one of {', '.join(_WEIGHT_CANDIDATES)} in the project "
                f"root, or pass weights_path explicitly."
            )
        self._model = YOLO(self.weights_path)

    def _detect_persons(self, frame: np.ndarray) -> list[list[float]]:
        res = self._model.predict(
            frame, conf=self.conf_threshold, classes=[_PERSON_CLASS],
            verbose=False, device=self.device,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        return [[float(v) for v in b] for b in res.boxes.xyxy.cpu().numpy()]

    def _sample_flow(
        self, field_xy: np.ndarray, box: list[float]
    ) -> Optional[tuple[float, float]]:
        """Median flow inside a shrunk bounding box, or None if degenerate."""
        h, w = field_xy.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        sx, sy = bw * self.box_shrink / 2.0, bh * self.box_shrink / 2.0

        xi1 = max(0, int(x1 + sx))
        yi1 = max(0, int(y1 + sy))
        xi2 = min(w, int(x2 - sx))
        yi2 = min(h, int(y2 - sy))
        if xi2 <= xi1 or yi2 <= yi1:
            return None

        patch = field_xy[yi1:yi2, xi1:xi2]
        return (float(np.median(patch[..., 0])), float(np.median(patch[..., 1])))

    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        flow_factory: Optional[Callable[[], FlowField]] = None,
        max_frames: int = 150,
        frame_stride: int = 1,
        annotate_path: Optional[str] = None,
    ) -> RouteResult:
        """
        Compare tracked-person velocities against the flow field.

        ``annotate_path`` writes a video showing the comparison directly: each
        compared person carries two arrows, one per instrument.  The numbers
        this route produces are medians over hundreds of samples, which say
        how well the two agree overall but never where or when they diverge.
        On the video a disagreement is visible as two arrows splitting apart,
        on a specific person, at a specific moment — which is what tells you
        whether an error is spread evenly or concentrated somewhere.
        """
        try:
            self._load()
        except Exception as exc:
            return RouteResult.errored(ROUTE_KEY, ROUTE_TITLE, exc, ROUTE_CAVEAT)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return RouteResult.errored(
                ROUTE_KEY, ROUTE_TITLE,
                RuntimeError(f"Could not open video: {video_path!r}"),
                ROUTE_CAVEAT,
            )

        ff = (flow_factory or (lambda: FlowField()))()
        tracker = IoUTracker(iou_threshold=0.3, max_age=15)
        states: dict[int, _TrackState] = {}
        comparisons: list[Comparison] = []
        person_counts: list[int] = []

        writer = None
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        prev_frame = None
        idx = 0
        processed = 0
        try:
            while processed < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_stride != 0:
                    idx += 1
                    continue

                if prev_frame is not None:
                    flow = ff.compute(prev_frame, frame, None, idx / 30.0)

                    boxes = self._detect_persons(frame)
                    person_counts.append(len(boxes))
                    ids = tracker.update(boxes, idx)

                    frame_pairs: list[tuple[list[float], Comparison]] = []

                    for box, tid in zip(boxes, ids):
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        prev_state = states.get(tid)
                        states[tid] = _TrackState(
                            (cx, cy), idx,
                            seen=(prev_state.seen + 1) if prev_state else 1,
                        )
                        if prev_state is None:
                            continue

                        dt = idx - prev_state.frame_index
                        if dt <= 0:
                            continue

                        # Track velocity in px/frame, matching the flow's units.
                        tvx = (cx - prev_state.centroid[0]) / dt
                        tvy = (cy - prev_state.centroid[1]) / dt

                        if states[tid].seen < self.min_track_age:
                            continue
                        if np.hypot(tvx, tvy) < self.min_track_speed_px:
                            continue

                        sampled = self._sample_flow(flow.field_xy, box)
                        if sampled is None:
                            continue

                        comp = Comparison(
                            frame_index=idx, track_id=tid,
                            track_vx=tvx, track_vy=tvy,
                            flow_vx=sampled[0], flow_vy=sampled[1],
                        )
                        comparisons.append(comp)
                        frame_pairs.append((box, comp))

                    if annotate_path:
                        annotated = self._draw_comparison(frame, frame_pairs)
                        if writer is None:
                            from models.crowd_flow.dense_flow_analyser import (
                                _AnnotatedVideoWriter)
                            h, w = annotated.shape[:2]
                            writer = _AnnotatedVideoWriter(
                                annotate_path, src_fps / max(1, frame_stride),
                                w, h)
                        writer.write(annotated)

                    processed += 1

                prev_frame = frame
                idx += 1
        finally:
            cap.release()
            if writer is not None:
                written = writer.close()
                if written:
                    logger.info("Cross-family comparison video: %s", written)

        result = self._score(comparisons, person_counts, video_path, processed)
        if annotate_path and writer is not None:
            result.detail["comparison_video"] = os.path.basename(annotate_path)
        return result

    # ------------------------------------------------------------------

    def _draw_comparison(
        self, frame: np.ndarray,
        pairs: list[tuple[list[float], "Comparison"]],
    ) -> np.ndarray:
        """
        Draw both instruments' velocity arrows on each compared person.

        White  = tracker (bounding-box motion)
        Cyan   = dense optical flow (pixel motion)

        Arrows are drawn from the same origin and at the same exaggeration, so
        agreement reads as one thick arrow and disagreement as a visible V.
        Pedestrian velocities are 1-2 px/frame — drawn at true length they
        would be a few pixels long and nothing could be judged from them — so
        both are scaled by the same factor.  The scale is stated on screen,
        because an unlabelled exaggeration invites reading arrow length as
        real speed.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        for box, c in pairs:
            x1, y1, x2, y2 = [int(v) for v in box]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            agree = (c.speed_error <= self.max_speed_error_px
                     and c.angular_error_deg <= self.max_angular_deg)
            box_colour = _BOX_AGREE if agree else _BOX_DISAGREE
            cv2.rectangle(out, (x1, y1), (x2, y2), box_colour, 1)

            # Origin dot: in a crowd, neighbouring boxes overlap and the two
            # arrows of one person sit next to another person's pair.  Marking
            # the shared start point makes it unambiguous which white arrow
            # belongs with which cyan one.
            cv2.circle(out, (cx, cy), 3, box_colour, -1, cv2.LINE_AA)

            for vx, vy, colour in (
                (c.track_vx, c.track_vy, _TRACKER_COLOUR),
                (c.flow_vx,  c.flow_vy,  _FLOW_COLOUR),
            ):
                ex = int(np.clip(cx + vx * _ARROW_SCALE, 0, w - 1))
                ey = int(np.clip(cy + vy * _ARROW_SCALE, 0, h - 1))
                if abs(ex - cx) + abs(ey - cy) >= 2:
                    cv2.arrowedLine(out, (cx, cy), (ex, ey), colour, 2,
                                    tipLength=0.3, line_type=cv2.LINE_AA)

            if not agree:
                cv2.putText(out, f"{c.angular_error_deg:.0f}deg",
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, _BOX_DISAGREE, 1, cv2.LINE_AA)

        n_agree = sum(
            1 for _, c in pairs
            if c.speed_error <= self.max_speed_error_px
            and c.angular_error_deg <= self.max_angular_deg
        )
        self._draw_legend(out, len(pairs), n_agree)
        return out

    @staticmethod
    def _draw_legend(out: np.ndarray, n_pairs: int, n_agree: int) -> None:
        """Fixed key in the top-left, so the video is readable on its own."""
        pad = 8
        lines = [
            ("WHITE arrow = tracker (box motion)", _TRACKER_COLOUR),
            ("CYAN  arrow = dense optical flow",   _FLOW_COLOUR),
            (f"arrows drawn {_ARROW_SCALE}x actual length", (200, 200, 200)),
            (f"compared this frame: {n_pairs}   agreeing: {n_agree}",
             _BOX_AGREE if n_agree == n_pairs else _BOX_DISAGREE),
        ]
        box_h = pad * 2 + 18 * len(lines)
        cv2.rectangle(out, (0, 0), (430, box_h), (25, 25, 25), -1)
        for i, (text, colour) in enumerate(lines):
            cv2.putText(out, text, (pad, pad + 13 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)

    def _score(
        self,
        comparisons: list[Comparison],
        person_counts: list[int],
        video_path: str,
        processed: int,
    ) -> RouteResult:
        mean_density = float(np.mean(person_counts)) if person_counts else 0.0

        # Too little evidence must report SKIPPED, never FAIL.  A handful of
        # samples produces a wild correlation coefficient by chance, and
        # calling that a failure blames the flow estimator for what is
        # actually an absence of data — the exact false alarm that trains
        # people to ignore a validation suite.
        if len(comparisons) < self.min_comparisons:
            if mean_density < 1.0:
                why = (
                    f"The person detector found only {mean_density:.2f} people "
                    f"per frame over {processed} frames, yielding "
                    f"{len(comparisons)} comparisons (need "
                    f"{self.min_comparisons}).  This is a DETECTOR limitation, "
                    f"not a flow result: a COCO-trained person detector expects "
                    f"upright people at ground level and largely fails on "
                    f"overhead or steeply oblique views, small figures, and "
                    f"people hidden under umbrellas.  Use a clip with "
                    f"ground-level, visible pedestrians, or a detector trained "
                    f"for this viewpoint.  Nothing here reflects on the flow "
                    f"estimator either way."
                )
            else:
                why = (
                    f"Only {len(comparisons)} usable comparisons over "
                    f"{processed} frames (need {self.min_comparisons}); "
                    f"{mean_density:.1f} people per frame were detected but too "
                    f"few produced stable, moving tracks.  Too little evidence "
                    f"to judge the estimator."
                )
            res = RouteResult.skipped(ROUTE_KEY, ROUTE_TITLE, why, ROUTE_CAVEAT)
            res.detail = {
                "video": os.path.basename(video_path),
                "frames_processed": processed,
                "comparisons": len(comparisons),
                "min_comparisons": self.min_comparisons,
                "mean_persons_per_frame": mean_density,
            }
            return res

        speed_err = np.array([c.speed_error for c in comparisons])
        ang_err   = np.array([c.angular_error_deg for c in comparisons])
        t_speed   = np.array([c.track_speed for c in comparisons])
        f_speed   = np.array([c.flow_speed for c in comparisons])

        # Correlation over speeds: catches a systematic scale error that a
        # median absolute difference can hide.  Only meaningful when the
        # tracked speeds actually span a range -- see _MIN_SPEED_SPREAD_PX.
        if t_speed.std() > 1e-9 and f_speed.std() > 1e-9:
            corr = float(np.corrcoef(t_speed, f_speed)[0, 1])
        else:
            corr = 0.0

        speed_spread = float(t_speed.std())
        corr_meaningful = speed_spread >= _MIN_SPEED_SPREAD_PX
        corr_note = (
            "catches a scale error a median would hide"
            if corr_meaningful else
            f"NOT JUDGED: tracked speeds span only {speed_spread:.2f} px/frame "
            f"(need {_MIN_SPEED_SPREAD_PX:g}), which is detection-box jitter. "
            f"With no real spread to correlate, r is uninformative here."
        )

        density_limited = mean_density > self.density_warn_persons

        agree = float(np.mean(
            (speed_err <= self.max_speed_error_px) &
            (ang_err <= self.max_angular_deg)
        ))

        summary = (
            f"{len(comparisons)} track/flow comparisons over {processed} "
            f"frames; median speed error {float(np.median(speed_err)):.2f} "
            f"px/frame, median direction error {float(np.median(ang_err)):.0f} deg"
        )
        if density_limited:
            summary += (
                f"  [DENSITY-LIMITED: {mean_density:.0f} persons/frame — "
                f"tracking is unreliable above {self.density_warn_persons:.0f}, "
                f"so this result is not trustworthy]"
            )

        result = RouteResult(
            route=ROUTE_KEY, title=ROUTE_TITLE, status=STATUS_PASS,
            summary=summary, caveat=ROUTE_CAVEAT,
            measurements=[
                Measurement("Median speed error", float(np.median(speed_err)),
                            "px/frame", tolerance=self.max_speed_error_px),
                Measurement("Median direction error", float(np.median(ang_err)),
                            "deg", tolerance=self.max_angular_deg),
                Measurement("Speed correlation", corr, "r",
                            tolerance=0.5 if corr_meaningful else None,
                            higher_is_better=True, note=corr_note),
                Measurement("Agreeing comparisons", agree * 100.0, "%"),
                Measurement("Median tracked speed", float(np.median(t_speed)),
                            "px/frame"),
                Measurement("Median flow speed", float(np.median(f_speed)),
                            "px/frame",
                            note="a persistent gap from the tracked speed "
                                 "indicates a scale error in one instrument"),
                Measurement("Mean persons per frame", mean_density, "persons",
                            note=("above "
                                  f"{self.density_warn_persons:.0f} this route "
                                  "is not dependable")),
                Measurement("Comparisons", float(len(comparisons)), "samples"),
            ],
            detail={
                "video": os.path.basename(video_path),
                "frames_processed": processed,
                "density_limited": density_limited,
                "p95_speed_error": float(np.percentile(speed_err, 95)),
                "p95_angular_error": float(np.percentile(ang_err, 95)),
                "tracked_speed_spread": speed_spread,
                "correlation_judged": corr_meaningful,
            },
        )
        result.resolve_status()
        return result
