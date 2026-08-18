"""
CrowdMotionMonitor — per-person velocity, heading, and crush-risk detector.

What it does
------------
1. Detects people in each frame using the project's shared RT-DETRv2 detector
   (models/_detectors.py :: get_detector, COCO_PERSON).
2. Tracks each person across frames with the shared IoU tracker
   (models/_tracker.py :: IoUTracker).
3. Computes per-person velocity by sampling the median Farneback dense optical
   flow inside a shrunk bounding box around each tracked person (20 % inset on
   every side — avoids sampling background at the body edge).
4. Draws a filled equilateral triangle centred on each person, rotated to point
   in their smoothed direction of travel (full 2-D rotation from a heading
   angle, not an arrow primitive).
5. Colours the triangle by crowd-flow type (5-state scheme):
     TEAL-GREEN   — confirmed, moving rightward  (heading_deg in ±90°)
     ELECTRIC BLUE— confirmed, moving leftward   (heading_deg outside ±90°)
     RED          — personally stationary (speed below threshold for N frames);
                    outranks direction and crush colours.
     ORANGE       — collision / crush zone: crowd flow is converging at this
                    cell (divergence < crush_divergence_threshold) but the
                    person is still moving.  Red outranks orange.
     DARK GREY    — track not yet confirmed (pending).

Speed-history bookkeeping
-------------------------
Per-track speed history lives in a plain ``dict[int, deque]`` on the model
instance — not in the shared tracker's ``append_state`` / ``states`` store.
That keeps this model genuinely standalone: it has no dependency on the
tracker's internal state mechanism and does not write into a store that other
models or future tracker implementations might interpret differently.

Integration contract
--------------------
- consumption_type = "flow_pair"  (runner calls predict((prev, curr), …))
- Emits one Detection per confirmed tracked person per frame.
  label      : "person_moving_right" | "person_moving_left" |
               "person_stopped" | "person_crush_zone"
  confidence : normalised speed (moving) or 0.0 (stopped)
  bbox       : [x1, y1, x2, y2] person box in source pixels
  extra      : {track_id, speed_px_frame, heading_deg, crowd_direction,
                personally_stationary, local_divergence, local_crush_risk}
- finalize() closes the streaming H.264 writer and sets
  self.annotated_video_path (picked up by webapp/jobs.py via getattr).
- Reads self._fps / self.output_fps if set externally by jobs.py after
  construction (flow_pair protocol).

This file must NOT import from people_overlay.py, dense_flow_analyser.py's
model logic, zones.py, or crowd_metrics.py.  The only crowd_flow import is
_AnnotatedVideoWriter from dense_flow_analyser (shared encoder utility, not
model logic).
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter, defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from models.base import BaseModelWrapper, Detection
from models._detectors import get_detector, COCO_PERSON
from models._tracker import IoUTracker, sustained

# Re-use the project's streaming H.264 writer (encoder utility, not model
# logic — same rationale as importing get_detector from _detectors.py).
from models.crowd_flow.dense_flow_analyser import _AnnotatedVideoWriter

logger = logging.getLogger(__name__)

# ── Colour palette (BGR) ───────────────────────────────────────────────────
# Five-state crowd-flow colour scheme.
_COLOUR_PENDING  = ( 80,  80,  80)   # dark grey     — track not yet confirmed
_COLOUR_RIGHT    = (140, 200,   0)   # teal-green    — moving rightward
_COLOUR_LEFT     = (220,  80,   0)   # electric blue — moving leftward
_COLOUR_STOPPED  = (  0,  40, 220)   # red           — personally stationary
_COLOUR_CRUSH    = (  0, 140, 255)   # orange        — collision / crush zone
_COLOUR_TEXT     = (255, 255, 255)   # white         — track-id label

# Direction boundary: |heading_deg| < this → rightward, else → leftward.
# heading_deg = atan2(-vy, vx), so 0° = right on screen, ±180° = left.
_HEADING_RIGHT_THRESH = 90.0

# Triangle geometry: half-height and half-base of an equilateral triangle
# expressed as a fraction of the shorter side of the person's bounding box.
# 0.22 keeps markers readable in the foreground without overlapping neighbors
# in the dense mid-field/background regions where boxes are much smaller.
_TRI_SCALE = 0.22          # fraction of min(box_w, box_h)

# Farneback parameters — same as OpticalFlowCrushDetector for consistency.
_FB_PYR_SCALE  = 0.5
_FB_LEVELS     = 3
_FB_WINSIZE    = 15
_FB_ITERATIONS = 3
_FB_POLY_N     = 5
_FB_POLY_SIGMA = 1.2
_FB_FLAGS      = 0

# Divergence grid for crush-risk: cell size in pixels.
_DIV_CELL_PX = 32


class CrowdMotionMonitor(BaseModelWrapper):
    """
    Real-time crowd movement monitor.

    Parameters
    ----------
    device : str | None
        Torch device ("cuda", "cpu", None → auto).
    output_dir : str
        Where to write the annotated video.
    video_name : str
        Stem used for the output filename:
        ``{output_dir}/{video_name}_crowd_motion_monitor.mp4``.
    stationary_speed_px : float
        Speed floor in px/frame.  A track whose speed stays below this for
        ``stationary_frames`` consecutive frames is flagged as personally
        stationary.
    stationary_frames : int
        Number of consecutive sub-threshold frames before a track is flagged.
    crush_divergence_threshold : float
        Per-cell p10 divergence below this value signals local crowd
        compression (crush risk).  Negative — divergence is negative when
        vectors converge.
    confirm_frames : int
        Minimum track age (frames seen) before a Detection is emitted.
        Suppresses single-frame ghost detections.
    detect_every : int
        Run the person detector every N frames; carry boxes on the others.
    detect_tile_grid : tuple[int, int] | None
        Tiling grid passed to the shared RT-DETRv2 detector.
        ``(2, 2)`` runs 5 overlapping crops (4 tiles + full frame) and
        recovers far more of the small/distant people in the upper portion
        of dense crowd footage — measured at 35→95 people on comparable
        footage.  ``None`` runs a single full-frame pass (faster, lower
        recall on small subjects).  Default: ``(2, 2)``.
    detect_conf_threshold : float
        Confidence threshold for the person detector.  Lowered to 0.28
        (from the shared detector's default 0.35) because distant/small
        people tend to score lower confidence even with tiling; the
        threshold is only applied to this model's detect call and does
        not affect any other consumer of the shared detector.
    """

    consumption_type = "flow_pair"
    name             = "crowd_motion_monitor"
    gpu_accelerated  = False   # flow is CPU; detector uses GPU if available

    def __init__(
        self,
        device: Optional[str] = None,
        output_dir: str = "outputs/annotated",
        video_name: str = "run",
        stationary_speed_px: float = 1.5,
        stationary_frames: int = 10,
        crush_divergence_threshold: float = -0.5,
        confirm_frames: int = 3,
        detect_every: int = 5,
        detect_tile_grid: Optional[tuple] = (2, 2),
        detect_conf_threshold: float = 0.28,
    ) -> None:
        super().__init__(device=device)

        self._output_dir   = output_dir
        self._video_name   = video_name

        self.stationary_speed_px        = stationary_speed_px
        self.stationary_frames          = stationary_frames
        self.crush_divergence_threshold = crush_divergence_threshold
        self.confirm_frames             = confirm_frames
        self.detect_every               = detect_every
        self.detect_tile_grid           = detect_tile_grid
        self.detect_conf_threshold      = detect_conf_threshold

        # jobs.py sets these after construction for any flow_pair model.
        self._fps: float = 25.0
        self._frame_stride: int = 1
        self.output_fps: Optional[float] = None

        # Runtime state — initialised in load().
        self._detector  = None
        self._tracker:  Optional[IoUTracker]           = None

        # Per-track speed history: dict[track_id, deque[float]]
        # Lives here, not in the tracker's state store, so this model is
        # genuinely standalone.  Evicted when the tracker drops the track.
        self._speed_history: dict[int, deque] = {}

        # Per-track smoothed heading as a (cos, sin) unit vector.
        self._heading_vec: dict[int, tuple[float, float]] = {}

        # Last set of detected boxes (carried on non-detect frames).
        self._last_boxes: list[list[float]] = []

        # Streaming video writer.
        self._writer: Optional[_AnnotatedVideoWriter] = None
        self._frames_written: int = 0

        # Set by finalize() / _close_video(); picked up by jobs.py.
        self.annotated_video_path: Optional[str] = None

        # Exposed for the web-UI live preview (same pattern as DenseFlowAnalyser).
        self.latest_annotated_frame: Optional[np.ndarray] = None

        # Run-level summary stats (populated in finalize()).
        self.summary: dict = {}
        self._total_detections_count: int = 0
        self._label_counter: Counter = Counter()
        self._track_directions: dict[int, list[str]] = defaultdict(list)
        self._frame_crush_counts: list[tuple[int, float, int]] = []
        self._speed_records: dict[str, list[float]] = defaultdict(list)
        self._heading_hist_bins: list[int] = [0] * 18
        self._heading_right_count: int = 0
        self._heading_left_count: int = 0
        self._boundary_crush_count: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Obtain the shared detector and create a fresh tracker."""
        self._detector = get_detector(device=self.device)
        self._detector.load()

        self._tracker = IoUTracker(
            iou_threshold=0.3,
            max_age=30,
            history_len=64,
        )

        self._speed_history.clear()
        self._heading_vec.clear()
        self._last_boxes = []
        self._writer = None
        self._frames_written = 0
        self.annotated_video_path = None

        # Reset summary accumulators
        self.summary = {}
        self._total_detections_count = 0
        self._label_counter.clear()
        self._track_directions.clear()
        self._frame_crush_counts.clear()
        self._speed_records.clear()
        self._heading_hist_bins = [0] * 18
        self._heading_right_count = 0
        self._heading_left_count = 0
        self._boundary_crush_count = 0

        os.makedirs(self._output_dir, exist_ok=True)
        self._model = "ready"

        logger.info(
            "CrowdMotionMonitor loaded.  device=%s  stationary_speed_px=%.2f  "
            "stationary_frames=%d  crush_div_thr=%.3f  confirm=%d  "
            "detect_every=%d  tile_grid=%s  conf_thr=%.2f",
            self.device, self.stationary_speed_px, self.stationary_frames,
            self.crush_divergence_threshold, self.confirm_frames, self.detect_every,
            self.detect_tile_grid, self.detect_conf_threshold,
        )

    # ──────────────────────────────────────────────────────────────────────
    # predict — called once per frame pair by the pipeline runner
    # ──────────────────────────────────────────────────────────────────────

    def predict(
        self,
        frame_pair: tuple[np.ndarray, np.ndarray],
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("CrowdMotionMonitor.load() must be called before predict().")

        prev_frame, curr_frame = frame_pair

        # 1. Dense optical flow (Farneback, full frame, CPU).
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            _FB_PYR_SCALE, _FB_LEVELS, _FB_WINSIZE,
            _FB_ITERATIONS, _FB_POLY_N, _FB_POLY_SIGMA, _FB_FLAGS,
        )  # shape (H, W, 2) — per-pixel (dx, dy)

        # 2. Global divergence grid — used for per-person crush-risk lookup.
        div_grid = self._compute_divergence_grid(flow)  # shape (n_rows, n_cols)

        # 3. Person detection (runs every detect_every frames; boxes carried on others).
        # tile_grid=(2,2) runs 5 overlapping crops so small/distant people in the
        # upper crowd region are detected — the full-frame pass alone resizes a
        # 1280×720 source to 640×640 which halves a 20 px person to ~10 px,
        # below what the model resolves.  detect_conf_threshold is intentionally
        # lower than the shared default (0.35) because distant people score lower.
        if frame_index % self.detect_every == 0:
            self._last_boxes = self._detector.detect(
                curr_frame,
                classes=(COCO_PERSON,),
                tile_grid=self.detect_tile_grid,
                conf_threshold=self.detect_conf_threshold,
            )

        boxes = self._last_boxes

        # 4. IoU tracking → one track ID per box, in order.
        track_ids = self._tracker.update(boxes, frame_index)

        # 5. Evict speed/heading history for tracks the tracker dropped.
        live_ids = set(track_ids)
        for stale in [tid for tid in list(self._speed_history) if tid not in live_ids]:
            self._speed_history.pop(stale, None)
            self._heading_vec.pop(stale, None)

        # 6. Annotated frame (copy so we don't mutate the source).
        annotated = curr_frame.copy()
        h_frame, w_frame = curr_frame.shape[:2]

        detections: list[Detection] = []

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # 6a. Sample flow inside a shrunk box (20 % inset) to avoid
            #     background pixels at the body boundary.
            bw, bh = x2 - x1, y2 - y1
            ix = max(1, int(bw * 0.20))
            iy = max(1, int(bh * 0.20))
            sx1, sy1 = x1 + ix, y1 + iy
            sx2, sy2 = x2 - ix, y2 - iy

            if sx2 > sx1 and sy2 > sy1:
                patch = flow[sy1:sy2, sx1:sx2]
                vx = float(np.median(patch[..., 0]))
                vy = float(np.median(patch[..., 1]))
            else:
                vx, vy = 0.0, 0.0

            speed = math.hypot(vx, vy)

            # 6b. Per-track speed deque (owned by this model instance).
            if tid not in self._speed_history:
                self._speed_history[tid] = deque(maxlen=self.stationary_frames)
            self._speed_history[tid].append(speed)

            # 6c. Smoothed heading via EMA on the unit vector (wrap-safe).
            if speed > 1e-3:
                ux, uy = vx / speed, vy / speed
            else:
                ux, uy = 0.0, 0.0

            alpha = 0.35  # EMA smoothing factor
            if tid not in self._heading_vec:
                self._heading_vec[tid] = (ux, uy)
            else:
                ox, oy = self._heading_vec[tid]
                nx = alpha * ux + (1 - alpha) * ox
                ny = alpha * uy + (1 - alpha) * oy
                norm = math.hypot(nx, ny)
                if norm > 1e-6:
                    nx, ny = nx / norm, ny / norm
                self._heading_vec[tid] = (nx, ny)

            hx, hy = self._heading_vec[tid]
            # Screen y-axis is inverted: -vy gives the correct compass heading.
            heading_deg = math.degrees(math.atan2(-hy, hx))

            # 6d. Stationary flag: last N speeds all below threshold.
            personally_stationary = sustained(
                [s < self.stationary_speed_px for s in self._speed_history[tid]],
                self.stationary_frames,
            )

            # 6e. Local crush-risk: look up divergence grid at box centre.
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            gr, gc = cy // _DIV_CELL_PX, cx // _DIV_CELL_PX
            nr, nc = div_grid.shape
            gr = min(gr, nr - 1)
            gc = min(gc, nc - 1)
            local_divergence = float(div_grid[gr, gc])
            local_crush_risk = local_divergence < self.crush_divergence_threshold

            # 6f. Track-age confirmation gate.
            track_age = self._tracker.age(tid)
            confirmed  = track_age >= self.confirm_frames

            # 6g. Crowd-flow direction classification.
            # A person is "rightward" when the x-component of their smoothed
            # heading is positive, i.e. |heading_deg| < 90°.
            moving_right = abs(heading_deg) < _HEADING_RIGHT_THRESH
            crowd_direction = "right" if moving_right else "left"

            # 6h. Draw filled triangle.
            # Priority (high → low): pending → stopped (red) → crush zone
            # (orange) → moving right (teal-green) → moving left (blue).
            # Stopped always outranks direction and crush colours so a halted
            # person in a converging zone is unambiguously red.
            if not confirmed:
                colour = _COLOUR_PENDING
            elif personally_stationary:
                colour = _COLOUR_STOPPED    # red   — this person has stopped
            elif local_crush_risk:
                colour = _COLOUR_CRUSH      # orange — crowd converging here
            elif moving_right:
                colour = _COLOUR_RIGHT      # teal-green — moving right
            else:
                colour = _COLOUR_LEFT       # electric blue — moving left

            self._draw_triangle(annotated, cx, cy, bw, bh, heading_deg, colour)

            # Optionally label with track ID (small, above box).
            cv2.putText(
                annotated, str(tid),
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOUR_TEXT, 1, cv2.LINE_AA,
            )

            # 6h. Emit Detection for confirmed tracks only.
            # 6i. Emit Detection for confirmed tracks only and record summary statistics.
            if not confirmed:
                continue

            if personally_stationary:
                label = "person_stopped"
            elif local_crush_risk:
                label = "person_crush_zone"
            elif moving_right:
                label = "person_moving_right"
            else:
                label = "person_moving_left"
            # Confidence: normalised speed capped at 1, or 0 for stopped.
            conf = 0.0 if personally_stationary else min(1.0, speed / max(self.stationary_speed_px * 5, 1e-6))
            det = Detection(
                model_name=self.name,
                label=label,
                confidence=round(conf, 4),
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=[x1, y1, x2, y2],
                extra={
                    "track_id":              tid,
                    "speed_px_frame":        round(speed, 4),
                    "heading_deg":           round(heading_deg, 2),
                    "crowd_direction":       crowd_direction,
                    "personally_stationary": personally_stationary,
                    "local_divergence":      round(local_divergence, 5),
                    "local_crush_risk":      local_crush_risk,
                },
            )
            detections.append(det)

            # Summary stats accumulation
            self._total_detections_count += 1
            self._label_counter[label] += 1
            self._track_directions[tid].append(crowd_direction)
            self._speed_records[label].append(speed)

            if abs(heading_deg) < _HEADING_RIGHT_THRESH:
                self._heading_right_count += 1
            else:
                self._heading_left_count += 1

            bin_idx = int((heading_deg + 180.0) / 20.0) % 18
            self._heading_hist_bins[bin_idx] += 1

            if local_crush_risk and abs(abs(heading_deg) - _HEADING_RIGHT_THRESH) < 15.0:
                self._boundary_crush_count += 1

        # Track per-frame crush count
        frame_crush_count = sum(1 for d in detections if d.label == "person_crush_zone")
        self._frame_crush_counts.append((frame_index, timestamp_sec, frame_crush_count))

        # 7. Stream annotated frame.
        self.latest_annotated_frame = annotated
        self._write_frame(annotated)

        return detections

    # ──────────────────────────────────────────────────────────────────────
    # finalize — called by the runner after all frames
    # ──────────────────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """Compute run-level summary stats, close video, and publish annotated_video_path."""
        self._compute_summary()
        self._close_video()

    def _compute_summary(self) -> dict:
        """Calculate comprehensive run-level summary metrics."""
        total = self._total_detections_count
        label_counts = dict(self._label_counter)
        n_stopped = label_counts.get("person_stopped", 0)
        n_crush = label_counts.get("person_crush_zone", 0)
        n_moving_right = label_counts.get("person_moving_right", 0)
        n_moving_left = label_counts.get("person_moving_left", 0)
        n_moving = n_moving_right + n_moving_left + n_crush

        pct_moving = round((n_moving / total * 100), 1) if total > 0 else 0.0
        pct_stationary = round((n_stopped / total * 100), 1) if total > 0 else 0.0
        pct_crush_risk = round((n_crush / total * 100), 1) if total > 0 else 0.0
        pct_moving_right = round((n_moving_right / total * 100), 1) if total > 0 else 0.0
        pct_moving_left = round((n_moving_left / total * 100), 1) if total > 0 else 0.0

        h_total = self._heading_right_count + self._heading_left_count
        pct_heading_right = round((self._heading_right_count / h_total * 100), 1) if h_total > 0 else 0.0
        pct_heading_left = round((self._heading_left_count / h_total * 100), 1) if h_total > 0 else 0.0

        # Crush events: distinct periods where 3+ people are simultaneously crush-flagged
        crush_events = 0
        in_event = False
        peak_crush_count = 0
        peak_crush_timestamp_sec = 0.0

        for f_idx, t_sec, c_cnt in self._frame_crush_counts:
            if c_cnt > peak_crush_count:
                peak_crush_count = c_cnt
                peak_crush_timestamp_sec = t_sec
            if c_cnt >= 3:
                if not in_event:
                    crush_events += 1
                    in_event = True
            else:
                in_event = False

        # Per-track stability
        flip_data = []
        for tid, dirs in self._track_directions.items():
            flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
            r_cnt = dirs.count("right")
            l_cnt = dirs.count("left")
            flip_data.append((tid, flips, r_cnt, l_cnt))

        total_tracks = len(flip_data)
        stable_tracks = sum(1 for _, f, _, _ in flip_data if f == 0)
        unstable_tracks = [x for x in flip_data if x[1] > 5]
        avg_flips = (sum(f for _, f, _, _ in flip_data) / total_tracks) if total_tracks else 0.0
        pct_stable_tracks = round((stable_tracks / total_tracks * 100), 1) if total_tracks else 0.0

        worst_tracks = sorted(flip_data, key=lambda x: -x[1])[:10]
        suspicious_list = []
        for tid, f, r, l in worst_tracks:
            if f > 5:
                dom = "right" if r >= l else "left"
                pct_dom = round(max(r, l) / (r + l) * 100) if (r + l) else 0
                suspicious_list.append({
                    "track_id": tid,
                    "flips": f,
                    "right_count": r,
                    "left_count": l,
                    "dominant_direction": dom,
                    "dominant_pct": pct_dom,
                })

        speed_stats = {}
        all_speeds = []
        for lbl, spds in self._speed_records.items():
            if spds:
                speed_stats[lbl] = {
                    "avg_px_frame": round(float(np.mean(spds)), 2),
                    "max_px_frame": round(float(np.max(spds)), 2),
                    "count": len(spds),
                }
                all_speeds.extend(spds)
        overall_avg_speed = round(float(np.mean(all_speeds)), 2) if all_speeds else 0.0

        heading_bins = []
        bin_size = 360.0 / len(self._heading_hist_bins)
        for i, cnt in enumerate(self._heading_hist_bins):
            lo = -180.0 + i * bin_size
            hi = lo + bin_size
            direction = "left" if (lo < -_HEADING_RIGHT_THRESH or hi > _HEADING_RIGHT_THRESH) else "right"
            heading_bins.append({
                "range": [round(lo, 1), round(hi, 1)],
                "count": cnt,
                "direction": direction,
            })

        self.summary = {
            "total_detections": total,
            "total_tracks": total_tracks,
            "pct_moving": pct_moving,
            "pct_stationary": pct_stationary,
            "pct_crush_risk": pct_crush_risk,
            "pct_moving_right": pct_moving_right,
            "pct_moving_left": pct_moving_left,
            "pct_heading_right": pct_heading_right,
            "pct_heading_left": pct_heading_left,
            "crush_event_count": crush_events,
            "peak_crush_timestamp_sec": round(peak_crush_timestamp_sec, 2),
            "peak_crush_people_count": peak_crush_count,
            "boundary_crush_pct": round((self._boundary_crush_count / n_crush * 100), 1) if n_crush > 0 else 0.0,
            "label_counts": label_counts,
            "avg_speed_px_frame": overall_avg_speed,
            "speed_by_label": speed_stats,
            "stable_tracks_count": stable_tracks,
            "stable_tracks_pct": pct_stable_tracks,
            "unstable_tracks_count": len(unstable_tracks),
            "avg_flips_per_track": round(avg_flips, 2),
            "suspicious_tracks": suspicious_list,
            "heading_histogram": heading_bins,
        }

        logger.info(
            "CrowdMotionMonitor summary: total=%d, moving=%.1f%% (R:%.1f%%, L:%.1f%%), "
            "crush=%.1f%% (%d events, peak @ %.2fs), stationary=%.1f%%, tracks=%d (%.1f%% stable)",
            total, pct_moving, pct_moving_right, pct_moving_left, pct_crush_risk,
            crush_events, peak_crush_timestamp_sec, pct_stationary, total_tracks, pct_stable_tracks,
        )
        return self.summary

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_divergence_grid(self, flow: np.ndarray) -> np.ndarray:
        """
        Per-cell p10 divergence of the optical flow field.

        div = ∂fx/∂x + ∂fy/∂y  (discrete: np.gradient)

        Taking the 10th percentile inside each grid cell captures localised
        compression without the mean-cancellation that plagues crowd centres
        (inward vectors on one side cancel outward vectors on the other).
        """
        fx = flow[..., 0]
        fy = flow[..., 1]
        div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)

        h, w = div.shape
        g = _DIV_CELL_PX
        n_rows = max(1, h // g)
        n_cols = max(1, w // g)
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)

        for r in range(n_rows):
            for c in range(n_cols):
                cell = div[r * g: r * g + g, c * g: c * g + g]
                grid[r, c] = float(np.percentile(cell, 10))

        return grid

    @staticmethod
    def _draw_triangle(
        frame: np.ndarray,
        cx: int, cy: int,
        bw: int, bh: int,
        heading_deg: float,
        colour: tuple[int, int, int],
    ) -> None:
        """
        Draw a filled equilateral triangle centred at (cx, cy), rotated to
        point in the direction given by heading_deg.

        The triangle is built in local coordinates (pointing right along +x),
        then rotated by a 2-D rotation matrix and translated to the centre.

        Coordinates:
          apex   = ( r,   0)   — points in the direction of travel
          left   = (-r/2, +r/2 * tan(60°))
          right  = (-r/2, -r/2 * tan(60°))
        where r = _TRI_SCALE * min(bw, bh).
        """
        r = max(4, int(_TRI_SCALE * min(bw, bh)))
        h_half = int(r * math.tan(math.radians(30)))

        # Local coords (pointing right)
        pts_local = np.array([
            [ r,       0],
            [-r // 2,  h_half],
            [-r // 2, -h_half],
        ], dtype=np.float64)

        # Rotation matrix for heading_deg.
        # heading_deg = atan2(-vy, vx) so +x (right) = 0°.
        rad = math.radians(heading_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])

        pts_rot = (R @ pts_local.T).T  # shape (3, 2)
        pts_abs = pts_rot + np.array([cx, cy])
        pts_int = pts_abs.astype(np.int32).reshape(1, 3, 2)

        cv2.fillPoly(frame, pts_int, colour)

    def _write_frame(self, annotated: np.ndarray) -> None:
        """Stream one annotated frame to the encoder, opening it on first use."""
        if self._writer is None:
            h, w = annotated.shape[:2]
            stem = os.path.splitext(os.path.basename(self._video_name))[0]
            out_path = os.path.join(
                self._output_dir,
                f"{stem}_crowd_motion_monitor.mp4",
            )
            fps = self.output_fps or self._fps
            self._writer = _AnnotatedVideoWriter(out_path, fps, w, h)
            logger.info(
                "CrowdMotionMonitor: opening annotated video at %s (%.2f fps %dx%d)",
                out_path, fps, w, h,
            )

        if self._writer.write(annotated):
            self._frames_written += 1

    def _close_video(self) -> None:
        """Close the encoder and set annotated_video_path."""
        if self._writer is None:
            return
        path = self._writer.close()
        self._writer = None
        if path is not None:
            self.annotated_video_path = path
            logger.info(
                "CrowdMotionMonitor: annotated video written: %s (%d frames)",
                path, self._frames_written,
            )
