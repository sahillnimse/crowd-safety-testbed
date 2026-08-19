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
model logic, zones.py, or crowd_metrics.py. Its only shared crowd_flow
dependency is _AnnotatedVideoWriter from video_writer.py (an encoder utility,
not model logic).
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
from models.crowd_flow.video_writer import _AnnotatedVideoWriter

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
        resume_moving_frames: int = 3,
        motion_noise_floor_ratio: float = 0.35,
        crush_divergence_threshold: float = -1.0,
        crush_max_speed_px: float = 6.0,
        counterflow_angle_threshold_deg: float = 120.0,
        counterflow_score_threshold: float = 0.30,
        overlay_mode: str = "markers",
        heatmap_metric: str = "divergence",
        confirm_frames: int = 3,
        detect_every: int = 5,
        detect_tile_grid: Optional[tuple] = (2, 2),
        detect_conf_threshold: float = 0.28,
    ) -> None:
        super().__init__(device=device)

        self._output_dir   = output_dir
        self._video_name   = video_name

        self.stationary_speed_px             = stationary_speed_px
        self.stationary_frames               = stationary_frames
        self.resume_moving_frames            = resume_moving_frames
        self.motion_noise_floor_ratio        = motion_noise_floor_ratio
        self.crush_divergence_threshold      = crush_divergence_threshold
        self.crush_max_speed_px              = crush_max_speed_px
        self.counterflow_angle_threshold_deg = counterflow_angle_threshold_deg
        self.counterflow_score_threshold     = counterflow_score_threshold
        self.overlay_mode                    = overlay_mode
        self.heatmap_metric                  = heatmap_metric
        self.confirm_frames                  = confirm_frames
        self.detect_every                    = detect_every
        self.detect_tile_grid                = detect_tile_grid
        self.detect_conf_threshold           = detect_conf_threshold

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

        # Per-track stationary status and consecutive moving frame streak (hysteresis).
        self._stationary_tracks: set[int] = set()
        self._moving_streak: dict[int, int] = {}

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

        # New metrics accumulators: variance, entropy, counterflow
        self._variance_records: list[float] = []
        self._entropy_records: list[float] = []
        self._counterflow_people_count: int = 0
        self._frame_counterflow_counts: list[tuple[int, float, int, float]] = []

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
        self._stationary_tracks.clear()
        self._moving_streak.clear()
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

        self._variance_records.clear()
        self._entropy_records.clear()
        self._counterflow_people_count = 0
        self._frame_counterflow_counts.clear()

        os.makedirs(self._output_dir, exist_ok=True)
        self._model = "ready"

        logger.info(
            "CrowdMotionMonitor loaded.  device=%s  stationary_speed_px=%.2f  "
            "stationary_frames=%d  crush_div_thr=%.3f  counterflow_ang_thr=%.1f  "
            "overlay_mode=%s  heatmap_metric=%s  confirm=%d  detect_every=%d",
            self.device, self.stationary_speed_px, self.stationary_frames,
            self.crush_divergence_threshold, self.counterflow_angle_threshold_deg,
            self.overlay_mode, self.heatmap_metric, self.confirm_frames, self.detect_every,
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

        # 2. Compute spatial grids (divergence and velocity variance) using shared cell iterator.
        div_grid = self._compute_divergence_grid(flow)  # shape (n_rows, n_cols)
        var_grid = self._compute_variance_grid(flow)    # shape (n_rows, n_cols)

        # 3. Person detection (runs every detect_every frames; boxes carried on others).
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
            self._moving_streak.pop(stale, None)
            self._stationary_tracks.discard(stale)

        # 6. Pass 1: Extract kinematic telemetry for all tracked people.
        h_frame, w_frame = curr_frame.shape[:2]
        track_records = []

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            bw, bh = x2 - x1, y2 - y1
            ix = max(1, int(bw * 0.20))
            sx1, sx2 = x1 + ix, x2 - ix

            sy1 = y1 + int(bh * 0.40)
            sy2 = y1 + int(bh * 0.70)

            # Fallback for small boxes
            if sy2 <= sy1:
                iy = max(1, int(bh * 0.20))
                sy1 = y1 + iy
                sy2 = y2 - iy

            if sx2 > sx1 and sy2 > sy1:
                patch = flow[sy1:sy2, sx1:sx2]
                vx = float(np.median(patch[..., 0]))
                vy = float(np.median(patch[..., 1]))
            else:
                vx, vy = 0.0, 0.0

            speed = math.hypot(vx, vy)

            if tid not in self._speed_history:
                self._speed_history[tid] = deque(maxlen=self.stationary_frames)
            self._speed_history[tid].append(speed)

            is_sub_threshold = sustained(
                [s < self.stationary_speed_px for s in self._speed_history[tid]],
                self.stationary_frames,
            )

            if speed >= self.stationary_speed_px:
                self._moving_streak[tid] = self._moving_streak.get(tid, 0) + 1
            else:
                self._moving_streak[tid] = 0

            if is_sub_threshold:
                self._stationary_tracks.add(tid)
            elif tid in self._stationary_tracks:
                if self._moving_streak.get(tid, 0) >= self.resume_moving_frames:
                    self._stationary_tracks.discard(tid)

            personally_stationary = (tid in self._stationary_tracks)

            noise_floor = max(self.motion_noise_floor_ratio * self.stationary_speed_px, 0.3)

            if not personally_stationary and speed >= noise_floor:
                ux, uy = vx / speed, vy / speed
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
            elif tid not in self._heading_vec:
                self._heading_vec[tid] = (1.0, 0.0)

            hx, hy = self._heading_vec[tid]
            heading_deg = math.degrees(math.atan2(-hy, hx))

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_age = self._tracker.age(tid)
            confirmed = track_age >= self.confirm_frames

            track_records.append({
                "tid": tid,
                "box": [x1, y1, x2, y2],
                "cx": cx,
                "cy": cy,
                "bw": bw,
                "bh": bh,
                "speed": speed,
                "heading_vec": (hx, hy),
                "heading_deg": heading_deg,
                "personally_stationary": personally_stationary,
                "confirmed": confirmed,
            })

        # 7. Pass 2: Calculate local directional entropy and counterflow opposition scores.
        detections: list[Detection] = []
        radius_px = 3 * _DIV_CELL_PX  # ~96 px local neighborhood

        # Prepare grid-level entropy and counterflow maps for optional heatmap rendering
        n_rows, n_cols = div_grid.shape
        entropy_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        counterflow_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

        for tr in track_records:
            cx, cy = tr["cx"], tr["cy"]
            tid = tr["tid"]
            hx, hy = tr["heading_vec"]
            hdeg = tr["heading_deg"]
            p_stat = tr["personally_stationary"]

            # Grid cell indices
            gr = min(cy // _DIV_CELL_PX, n_rows - 1)
            gc = min(cx // _DIV_CELL_PX, n_cols - 1)

            local_div = float(div_grid[gr, gc])
            max_crush_spd = max(self.crush_max_speed_px, 2.5 * self.stationary_speed_px)
            local_crush_risk = (local_div < self.crush_divergence_threshold) and (tr["speed"] <= max_crush_spd) and (not p_stat)
            local_var = float(var_grid[gr, gc])

            # Find neighboring moving tracks
            neighbors = [
                other for other in track_records
                if other["confirmed"] and (not other["personally_stationary"])
                and math.hypot(other["cx"] - cx, other["cy"] - cy) <= radius_px
            ]

            # 7a. Directional entropy: Shannon entropy over 8-bin heading histogram
            if len(neighbors) >= 2:
                bins = [0] * 8
                for nb in neighbors:
                    b_idx = int((nb["heading_deg"] + 180.0) / 45.0) % 8
                    bins[b_idx] += 1
                n_total = len(neighbors)
                ent = 0.0
                for cnt in bins:
                    if cnt > 0:
                        p = cnt / n_total
                        ent -= p * math.log2(p)
                local_entropy = round(ent, 3)
            else:
                local_entropy = 0.0

            entropy_grid[gr, gc] = max(entropy_grid[gr, gc], local_entropy)

            # 7b. Counterflow opposition score
            other_moving = [nb for nb in neighbors if nb["tid"] != tid]
            if (not p_stat) and len(other_moving) >= 1:
                mean_ux = sum(nb["heading_vec"][0] for nb in other_moving) / len(other_moving)
                mean_uy = sum(nb["heading_vec"][1] for nb in other_moving) / len(other_moving)
                mean_norm = math.hypot(mean_ux, mean_uy)

                if mean_norm > 0.15:
                    dom_ux, dom_uy = mean_ux / mean_norm, mean_uy / mean_norm
                    dp = max(-1.0, min(1.0, hx * dom_ux + hy * dom_uy))
                    ang_diff = math.degrees(math.acos(dp))
                    cf_angle_deg = round(ang_diff, 1)
                    is_cf = bool(cf_angle_deg >= self.counterflow_angle_threshold_deg)
                else:
                    cf_angle_deg = 0.0
                    is_cf = False
            else:
                cf_angle_deg = 0.0
                is_cf = False

            if is_cf:
                counterflow_grid[gr, gc] = 1.0

            tr["local_divergence"] = local_div
            tr["local_crush_risk"] = local_crush_risk
            tr["local_velocity_variance"] = local_var
            tr["local_directional_entropy"] = local_entropy
            tr["is_counterflow"] = is_cf
            tr["counterflow_angle_deg"] = cf_angle_deg

            # Direction classification
            moving_right = abs(hdeg) < _HEADING_RIGHT_THRESH
            crowd_direction = "right" if moving_right else "left"
            tr["crowd_direction"] = crowd_direction

            if not tr["confirmed"]:
                continue

            if p_stat:
                label = "person_stopped"
            elif local_crush_risk:
                label = "person_crush_zone"
            elif moving_right:
                label = "person_moving_right"
            else:
                label = "person_moving_left"

            conf = 0.0 if p_stat else min(1.0, tr["speed"] / max(self.stationary_speed_px * 5, 1e-6))
            det = Detection(
                model_name=self.name,
                label=label,
                confidence=round(conf, 4),
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=tr["box"],
                extra={
                    "track_id":                  tid,
                    "speed_px_frame":            round(tr["speed"], 4),
                    "heading_deg":               round(hdeg, 2),
                    "crowd_direction":           crowd_direction,
                    "personally_stationary":     p_stat,
                    "local_divergence":          round(local_div, 5),
                    "local_crush_risk":          local_crush_risk,
                    "local_velocity_variance":   round(local_var, 4),
                    "local_directional_entropy": local_entropy,
                    "is_counterflow":            is_cf,
                    "counterflow_angle_deg":     cf_angle_deg,
                },
            )
            detections.append(det)

            # Summary stats accumulation
            self._total_detections_count += 1
            self._label_counter[label] += 1
            if not p_stat:
                self._track_directions[tid].append(crowd_direction)
            self._speed_records[label].append(tr["speed"])
            self._variance_records.append(local_var)
            self._entropy_records.append(local_entropy)
            if is_cf:
                self._counterflow_people_count += 1

            if not p_stat:
                if abs(hdeg) < _HEADING_RIGHT_THRESH:
                    self._heading_right_count += 1
                else:
                    self._heading_left_count += 1

                bin_idx = int((hdeg + 180.0) / 20.0) % 18
                self._heading_hist_bins[bin_idx] += 1

                if local_crush_risk and abs(abs(hdeg) - _HEADING_RIGHT_THRESH) < 15.0:
                    self._boundary_crush_count += 1

        # Track per-frame crush & counterflow counts
        frame_crush_count = sum(1 for d in detections if d.label == "person_crush_zone")
        self._frame_crush_counts.append((frame_index, timestamp_sec, frame_crush_count))

        frame_cf_count = sum(1 for d in detections if d.extra.get("is_counterflow"))
        frame_cf_rate = frame_cf_count / max(1, len(detections))
        self._frame_counterflow_counts.append((frame_index, timestamp_sec, frame_cf_count, frame_cf_rate))

        # 8. Video overlay rendering
        if self.overlay_mode in ("heatmap", "combined"):
            if self.heatmap_metric == "variance":
                target_grid = var_grid
            elif self.heatmap_metric == "entropy":
                target_grid = entropy_grid
            elif self.heatmap_metric == "counterflow":
                target_grid = counterflow_grid
            else:
                target_grid = div_grid
            annotated = self._render_heatmap_overlay(curr_frame, target_grid, self.heatmap_metric)
        else:
            annotated = curr_frame.copy()

        # Draw markers on top if in markers or combined mode
        if self.overlay_mode in ("markers", "combined"):
            for tr in track_records:
                cx, cy = tr["cx"], tr["cy"]
                bw, bh = tr["bw"], tr["bh"]
                hdeg = tr["heading_deg"]
                p_stat = tr["personally_stationary"]
                confirmed = tr["confirmed"]
                local_crush_risk = tr["local_crush_risk"]
                moving_right = tr["crowd_direction"] == "right"
                tid = tr["tid"]
                x1, y1 = tr["box"][0], tr["box"][1]

                if not confirmed:
                    colour = _COLOUR_PENDING
                elif p_stat:
                    colour = _COLOUR_STOPPED
                elif local_crush_risk:
                    colour = _COLOUR_CRUSH
                elif moving_right:
                    colour = _COLOUR_RIGHT
                else:
                    colour = _COLOUR_LEFT

                self._draw_marker(annotated, cx, cy, bw, bh, hdeg, colour, is_stationary=p_stat)
                cv2.putText(
                    annotated, str(tid),
                    (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOUR_TEXT, 1, cv2.LINE_AA,
                )

        # 9. Stream annotated frame.
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
        in_crush_event = False
        peak_crush_count = 0
        peak_crush_timestamp_sec = 0.0

        for f_idx, t_sec, c_cnt in self._frame_crush_counts:
            if c_cnt > peak_crush_count:
                peak_crush_count = c_cnt
                peak_crush_timestamp_sec = t_sec
            if c_cnt >= 3:
                if not in_crush_event:
                    crush_events += 1
                    in_crush_event = True
            else:
                in_crush_event = False

        # Counterflow events: distinct periods where 2+ people or score > threshold are counterflow
        counterflow_events = 0
        in_cf_event = False
        peak_cf_count = 0
        peak_cf_timestamp_sec = 0.0

        for f_idx, t_sec, cf_cnt, cf_rate in self._frame_counterflow_counts:
            if cf_cnt > peak_cf_count:
                peak_cf_count = cf_cnt
                peak_cf_timestamp_sec = t_sec
            if cf_cnt >= 2 or cf_rate >= self.counterflow_score_threshold:
                if not in_cf_event:
                    counterflow_events += 1
                    in_cf_event = True
            else:
                in_cf_event = False

        pct_cf_people = round((self._counterflow_people_count / total * 100), 1) if total > 0 else 0.0
        avg_var = round(float(np.mean(self._variance_records)), 3) if self._variance_records else 0.0
        peak_var = round(float(np.max(self._variance_records)), 3) if self._variance_records else 0.0
        avg_entropy = round(float(np.mean(self._entropy_records)), 3) if self._entropy_records else 0.0

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
            "counterflow_events_count": counterflow_events,
            "pct_counterflow_people": pct_cf_people,
            "peak_counterflow_timestamp_sec": round(peak_cf_timestamp_sec, 2),
            "peak_counterflow_people_count": peak_cf_count,
            "avg_velocity_variance": avg_var,
            "peak_velocity_variance": peak_var,
            "avg_directional_entropy": avg_entropy,
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
            "crush=%.1f%% (%d events), counterflow=%.1f%% (%d events), entropy=%.3f, var=%.3f",
            total, pct_moving, pct_moving_right, pct_moving_left, pct_crush_risk,
            crush_events, pct_cf_people, counterflow_events, avg_entropy, avg_var,
        )
        return self.summary

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _iterate_grid_cells(h: int, w: int, cell_fn) -> np.ndarray:
        """
        Shared per-cell grid iterator.
        Constructs grid of shape (n_rows, n_cols) based on _DIV_CELL_PX.
        Invokes cell_fn(row_slice, col_slice, r, c) to populate each cell.
        """
        g = _DIV_CELL_PX
        n_rows = max(1, h // g)
        n_cols = max(1, w // g)
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)

        for r in range(n_rows):
            r_slice = slice(r * g, r * g + g)
            for c in range(n_cols):
                c_slice = slice(c * g, c * g + g)
                grid[r, c] = cell_fn(r_slice, c_slice, r, c)

        return grid

    def _compute_divergence_grid(self, flow: np.ndarray) -> np.ndarray:
        """
        Per-cell p10 divergence of the optical flow field.
        div = ∂fx/∂x + ∂fy/∂y (discrete np.gradient).
        """
        fx = flow[..., 0]
        fy = flow[..., 1]
        div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
        h, w = div.shape

        def _div_cell(r_s, c_s, r, c):
            cell = div[r_s, c_s]
            return float(np.percentile(cell, 10)) if cell.size > 0 else 0.0

        return self._iterate_grid_cells(h, w, _div_cell)

    def _compute_variance_grid(self, flow: np.ndarray) -> np.ndarray:
        """
        Per-cell circular/directional variance of optical flow vectors.
        Returns value in [0.0, 1.0], where 0 is perfectly aligned flow
        and 1.0 is maximum directional variance/disorder.
        """
        fx = flow[..., 0]
        fy = flow[..., 1]
        h, w = fx.shape

        def _var_cell(r_s, c_s, r, c):
            cell_fx = fx[r_s, c_s]
            cell_fy = fy[r_s, c_s]
            mags = np.hypot(cell_fx, cell_fy)
            moving = mags > 0.3
            if np.count_nonzero(moving) < 4:
                return 0.0
            ux = cell_fx[moving] / mags[moving]
            uy = cell_fy[moving] / mags[moving]
            mean_ux = np.mean(ux)
            mean_uy = np.mean(uy)
            R = float(np.hypot(mean_ux, mean_uy))
            return float(np.clip(1.0 - R, 0.0, 1.0))

        return self._iterate_grid_cells(h, w, _var_cell)

    def _render_heatmap_overlay(
        self,
        frame: np.ndarray,
        grid: np.ndarray,
        metric: str,
        alpha_val: float = 0.55,
    ) -> np.ndarray:
        """
        Render semi-transparent per-cell heatmap overlay onto frame.
        """
        h, w = frame.shape[:2]
        if metric == "divergence":
            scale = max(abs(self.crush_divergence_threshold * 2), 1e-3)
            t_cell = np.clip(grid / scale, -1.0, 1.0)
            w_neg = np.clip(-t_cell, 0.0, 1.0)[..., None]
            w_pos = np.clip(t_cell, 0.0, 1.0)[..., None]
            c_zero = np.array([240, 240, 240], dtype=np.float32)
            c_neg  = np.array([0, 0, 220], dtype=np.float32)     # BGR Red
            c_pos  = np.array([220, 50, 0], dtype=np.float32)    # BGR Blue
            heat_cell = c_zero + (c_neg - c_zero) * w_neg + (c_pos - c_zero) * w_pos
            alpha_cell = (np.abs(t_cell) * alpha_val).astype(np.float32)
        elif metric == "variance":
            norm = np.clip(grid, 0.0, 1.0)
            heat_8u = (norm * 255).astype(np.uint8)
            heat_cell = cv2.applyColorMap(heat_8u, cv2.COLORMAP_JET)
            alpha_cell = (norm * alpha_val).astype(np.float32)
        elif metric == "entropy":
            norm = np.clip(grid / 3.0, 0.0, 1.0)
            heat_8u = (norm * 255).astype(np.uint8)
            heat_cell = cv2.applyColorMap(heat_8u, cv2.COLORMAP_MAGMA)
            alpha_cell = (norm * alpha_val).astype(np.float32)
        elif metric == "counterflow":
            norm = np.clip(grid, 0.0, 1.0)
            heat_8u = (norm * 255).astype(np.uint8)
            heat_cell = cv2.applyColorMap(heat_8u, cv2.COLORMAP_HOT)
            alpha_cell = (norm * alpha_val).astype(np.float32)
        else:
            return frame.copy()

        heat = cv2.resize(heat_cell, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        alpha = cv2.resize(alpha_cell, (w, h), interpolation=cv2.INTER_NEAREST)

        # Alpha blend using OpenCV primitives
        alpha3 = cv2.merge([alpha, alpha, alpha])
        diff = cv2.subtract(heat, frame, dtype=cv2.CV_32F)
        cv2.multiply(diff, alpha3, dst=diff)
        return cv2.add(frame, diff, dtype=cv2.CV_8U)

    @staticmethod
    def _draw_marker(
        frame: np.ndarray,
        cx: int, cy: int,
        bw: int, bh: int,
        heading_deg: float,
        colour: tuple[int, int, int],
        is_stationary: bool = False,
    ) -> None:
        """
        Draw a visual indicator centred at (cx, cy).

        - Moving / Pending / Crush: Filled equilateral triangle rotated to point in heading_deg.
        - Personally Stationary: Distinct neutral static marker (solid circle with inner highlight)
          with NO rotation, ensuring stationary people do not swing with head/torso motion.
        """
        r = max(4, int(_TRI_SCALE * min(bw, bh)))

        if is_stationary:
            cv2.circle(frame, (cx, cy), r, colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(2, r // 2), (255, 255, 255), 1, cv2.LINE_AA)
            return

        h_half = int(r * math.tan(math.radians(30)))

        pts_local = np.array([
            [ r,       0],
            [-r // 2,  h_half],
            [-r // 2, -h_half],
        ], dtype=np.float64)

        rad = math.radians(heading_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])

        pts_rot = (R @ pts_local.T).T
        pts_abs = pts_rot + np.array([cx, cy])
        pts_int = pts_abs.astype(np.int32).reshape(1, 3, 2)

        cv2.fillPoly(frame, pts_int, colour)

    @classmethod
    def _draw_triangle(
        cls,
        frame: np.ndarray,
        cx: int, cy: int,
        bw: int, bh: int,
        heading_deg: float,
        colour: tuple[int, int, int],
    ) -> None:
        """Backward-compatibility alias for _draw_marker."""
        cls._draw_marker(frame, cx, cy, bw, bh, heading_deg, colour, is_stationary=False)

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
