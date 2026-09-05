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
  label      : "person_moving" | "person_moving_stream_a" |
               "person_moving_stream_b" | "person_stopped" |
               "person_crush_zone"
  confidence : normalised speed (moving) or 0.0 (stopped)
  bbox       : [x1, y1, x2, y2] person box in source pixels
  extra      : {track_id, speed_px_frame, heading_deg, crowd_direction,
                stream_screen_direction, stream_angle_deg,
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
from models.head_count import get_head_counter

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

# Stream modes are inferred from image-space heading vectors (+y is downward).
_STREAM_MIN_TRACKS = 4
_STREAM_MIN_CLUSTER_SHARE = 0.15
_STREAM_MIN_SEPARATION_DEG = 60.0

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

# APGCC has no box, only a point. This is the synthetic box half-size (px)
# built around each surviving point, used for IoU tracking and drawing —
# small enough not to falsely overlap a neighbouring RT-DETRv2 box, big
# enough for the tracker's IoU match to hold across a few frames.
_APGCC_SYNTH_HALF_BOX_PX = 12

# An APGCC point is considered "already claimed" by RT-DETRv2 if it falls
# inside an RT-DETRv2 box, expanded by this fraction on every side — a point
# exactly on a box edge is still that person, not a second one standing
# next to them.
_APGCC_CLAIM_MARGIN = 0.15


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
        apgcc_weights: Optional[str] = None,
        apgcc_score_threshold: float = 0.5,
        apgcc_every: int = 5,
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
        self.apgcc_weights                   = apgcc_weights
        self.apgcc_score_threshold           = apgcc_score_threshold
        self.apgcc_every                     = apgcc_every

        # jobs.py sets these after construction for any flow_pair model.
        self._fps: float = 25.0
        self._frame_stride: int = 1
        self.output_fps: Optional[float] = None

        # Runtime state — initialised in load().
        self._detector  = None
        self._head_counter = None
        self._tracker:  Optional[IoUTracker]           = None
        self._last_apgcc_boxes: list[list[float]] = []

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
        self._stream_counts: Counter = Counter()

        # New metrics accumulators: variance, entropy, counterflow
        self._variance_records: list[float] = []
        self._entropy_records: list[float] = []
        self._counterflow_people_count: int = 0
        self._frame_counterflow_counts: list[tuple[int, float, int, float]] = []

        # ── Remaining crowd-safety metrics ──────────────────────────────
        # Divergence, density, crowd pressure, specific flow, stop-and-go and
        # oscillation symmetry. All are recorded PER FRAME (not per person) so
        # the run-level series can be analysed temporally in _compute_summary:
        # stop-and-go and oscillation are autocorrelations over time and are
        # meaningless on a per-person sample.
        #
        # IMPORTANT — units. This model has no ground-plane homography (see
        # _run_stream_directions' note), so nothing here can be expressed in
        # m/s, persons/m², or s⁻². Density is persons per 1000 px² of IMAGE
        # area and pressure is that density times velocity variance in px²
        # units. Both are comparable across frames of the same camera and
        # NOT comparable between cameras or against any published physical
        # threshold. The summary labels them accordingly.
        self._divergence_records: list[float] = []
        self._frame_density: list[float] = []          # persons / megapixel
        self._frame_person_count: list[int] = []       # raw count per frame
        self._frame_pressure: list[float] = []         # density * variance
        self._frame_mean_speed: list[float] = []       # px/frame, for stop-go
        self._frame_mean_vec: list[tuple[float, float]] = []   # for oscillation
        self._frame_area_px: float = 0.0               # set on first frame
        self._last_timestamp_sec: float = 0.0

        # Specific flow: net people crossing a counting line per second.
        # _track_last_side remembers which side of the line each track was on
        # so a crossing is an EDGE, not a state — counting "is on the right"
        # every frame would report the standing crowd, not the flow through.
        self._line_crossings_pos: int = 0     # left→right / top→bottom
        self._line_crossings_neg: int = 0     # the other way
        self._track_last_side: dict[int, int] = {}

        # Running sum of the two inferred stream centres (unit heading vectors)
        # so finalize() can emit one screen-relative direction per stream.
        self._stream_centre_acc: list[list[float]] = [[0.0, 0.0], [0.0, 0.0]]
        self._stream_centre_n: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Obtain the shared detector and create a fresh tracker."""
        self._detector = get_detector(device=self.device)
        self._detector.load()

        self._head_counter = get_head_counter(
            weights=self.apgcc_weights,
            device=self.device,
            score_threshold=self.apgcc_score_threshold,
        )
        self._head_counter.load()

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
        self._last_apgcc_boxes = []
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
        self._stream_counts.clear()

        self._variance_records.clear()
        self._entropy_records.clear()
        self._counterflow_people_count = 0
        self._frame_counterflow_counts.clear()
        self._divergence_records.clear()
        self._frame_density.clear()
        self._frame_person_count.clear()
        self._frame_pressure.clear()
        self._frame_mean_speed.clear()
        self._frame_mean_vec.clear()
        self._frame_area_px = 0.0
        self._last_timestamp_sec = 0.0
        self._line_crossings_pos = 0
        self._line_crossings_neg = 0
        self._track_last_side.clear()
        self._stream_centre_acc = [[0.0, 0.0], [0.0, 0.0]]
        self._stream_centre_n = 0

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
        var_grid = self._compute_variance_grid(flow)    # shape (n_rows, n_cols)

        rtdetr_boxes = self._last_boxes

        # 3. APGCC fusion — ground truth for head detection, but RT-DETRv2
        # claims a person first. An APGCC point that falls inside an
        # existing RT-DETRv2 box is the same person already covered and is
        # dropped; only points RT-DETRv2 missed become new synthetic boxes.
        # This is what stops the same person getting two triangles.
        #
        # APGCC *points* are only recomputed every apgcc_every frames (the
        # expensive step). But the dedup-against-RT-DETRv2 filter runs on
        # every frame using the *current* frame's RT-DETRv2 boxes, so a
        # person detected mid-interval by RT-DETRv2 is immediately removed
        # from the stale APGCC box list rather than creating a double-track
        # until the next APGCC recompute.
        if frame_index % self.apgcc_every == 0:
            apgcc_points = self._head_counter.points(curr_frame)
            self._last_apgcc_boxes = self._apgcc_points_to_boxes(
                apgcc_points, rtdetr_boxes, curr_frame.shape[:2],
            )

        # Re-filter stale APGCC boxes against the *current* frame's
        # RT-DETRv2 boxes every frame, not only on recompute frames.
        # _apgcc_points_to_boxes works on points, so on skip frames we
        # reconstruct synthetic centre points from the stored boxes and
        # let the same claim/expand logic drop any that are now covered.
        if frame_index % self.apgcc_every != 0 and self._last_apgcc_boxes:
            synth_centres = np.array(
                [[(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5]
                 for b in self._last_apgcc_boxes],
                dtype=np.float32,
            )
            self._last_apgcc_boxes = self._apgcc_points_to_boxes(
                synth_centres, rtdetr_boxes, curr_frame.shape[:2],
            )

        boxes = rtdetr_boxes + self._last_apgcc_boxes

        # 4. IoU tracking → one track ID per box, in order. RT-DETRv2 boxes
        # come first in `boxes`, so when both sources compete for the same
        # track next frame, the greedy one-to-one match in IoUTracker favours
        # whichever pairs best — RT-DETRv2 identity is not displaced by an
        # APGCC box arriving after it in the same list.
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

        for box, tid in zip(boxes, track_ids, strict=False):
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
            # Flow and image coordinates both use +y downward.  _draw_marker()
            # rotates in those same image coordinates, so do not invert y here.
            heading_deg = math.degrees(math.atan2(hy, hx))

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
                "is_moving": (not personally_stationary) and speed >= noise_floor,
            })

        # Infer one or two direction streams from the current confirmed moving
        # tracks.  The helper uses deterministic initialization and orders the
        # resulting centres, so an equivalent frame produces the same labels.
        stream_centres = self._infer_direction_streams(track_records)
        stream_dir_a = stream_dir_b = None
        stream_ang_a = stream_ang_b = None
        if len(stream_centres) == 2:
            stream_dir_a, stream_ang_a = self._screen_direction_from_vector(*stream_centres[0])
            stream_dir_b, stream_ang_b = self._screen_direction_from_vector(*stream_centres[1])
            self._stream_centre_acc[0][0] += float(stream_centres[0][0])
            self._stream_centre_acc[0][1] += float(stream_centres[0][1])
            self._stream_centre_acc[1][0] += float(stream_centres[1][0])
            self._stream_centre_acc[1][1] += float(stream_centres[1][1])
            self._stream_centre_n += 1
        for tr in track_records:
            tr["crowd_direction"] = self._nearest_stream(tr["heading_vec"], stream_centres)
            if tr["crowd_direction"] == "stream_a":
                tr["stream_screen_direction"] = stream_dir_a
                tr["stream_angle_deg"] = stream_ang_a
            elif tr["crowd_direction"] == "stream_b":
                tr["stream_screen_direction"] = stream_dir_b
                tr["stream_angle_deg"] = stream_ang_b
            else:
                tr["stream_screen_direction"] = None
                tr["stream_angle_deg"] = None

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

            crowd_direction = tr["crowd_direction"]

            if not tr["confirmed"]:
                continue

            if p_stat:
                label = "person_stopped"
            elif local_crush_risk:
                label = "person_crush_zone"
            elif crowd_direction == "stream_a":
                label = "person_moving_stream_a"
            elif crowd_direction == "stream_b":
                label = "person_moving_stream_b"
            else:
                label = "person_moving"

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
                    "stream_screen_direction":   tr.get("stream_screen_direction"),
                    "stream_angle_deg":          tr.get("stream_angle_deg"),
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
            if tr["is_moving"]:
                self._track_directions[tid].append(crowd_direction)
            self._speed_records[label].append(tr["speed"])
            self._variance_records.append(local_var)
            self._entropy_records.append(local_entropy)
            if is_cf:
                self._counterflow_people_count += 1

            if tr["is_moving"]:
                self._stream_counts[crowd_direction] += 1
                bin_idx = int((hdeg + 180.0) / 20.0) % 18
                self._heading_hist_bins[bin_idx] += 1

        # Track per-frame crush & counterflow counts
        frame_crush_count = sum(1 for d in detections if d.label == "person_crush_zone")
        self._frame_crush_counts.append((frame_index, timestamp_sec, frame_crush_count))

        frame_cf_count = sum(1 for d in detections if d.extra.get("is_counterflow"))
        frame_cf_rate = frame_cf_count / max(1, len(detections))
        self._frame_counterflow_counts.append((frame_index, timestamp_sec, frame_cf_count, frame_cf_rate))

        self._record_frame_metrics(track_records, curr_frame.shape[:2], timestamp_sec)

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
                crowd_direction = tr["crowd_direction"]
                tid = tr["tid"]
                x1, y1 = tr["box"][0], tr["box"][1]

                if not confirmed:
                    colour = _COLOUR_PENDING
                elif p_stat:
                    colour = _COLOUR_STOPPED
                elif local_crush_risk:
                    colour = _COLOUR_CRUSH
                elif crowd_direction != "stream_b":
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

    # ------------------------------------------------------------------
    # Per-frame series for the run-level crowd-safety metrics
    # ------------------------------------------------------------------

    #: Counting line for specific flow, as a fraction of frame width. 0.5 is
    #: the vertical centre line. A single line is a deliberate simplification:
    #: without per-camera configuration there is no way to know where the
    #: meaningful gate, doorway or corridor cross-section is, and the centre
    #: line at least measures throughput across the whole scene.
    SPECIFIC_FLOW_LINE_X_FRAC = 0.5

    def _record_frame_metrics(self, track_records: list, frame_hw: tuple,
                              timestamp_sec: float) -> None:
        """
        Append this frame's sample to each run-level metric series.

        Called once per processed frame, after the per-person loop. Everything
        here is frame-level: density is a property of the frame, and stop-and-go
        and oscillation are autocorrelations over the frame sequence.
        """
        h, w = frame_hw
        self._frame_area_px = float(h * w)
        self._last_timestamp_sec = max(self._last_timestamp_sec, float(timestamp_sec))

        n = len(track_records)

        # Density, persons per MEGAPIXEL of image area.
        #
        # Not "per 1000 px²", which is the unit the dense-flow module uses for
        # its per-cell grids: a whole 1080p frame is 2074 of those units, so a
        # busy scene of 50 people reads 0.024 and rounds to nothing on a card.
        # Per megapixel puts the same measurement on a scale a person can
        # read (that scene is 24.1), and it is the identical quantity times
        # 1000 -- no information changes, only the exponent.
        #
        # Recorded even when the frame is empty: a zero is a measurement
        # ("nobody here"), and dropping empty frames would bias the run
        # average upwards towards whatever the busy moments looked like.
        megapixels = self._frame_area_px / 1_000_000.0
        density = (n / megapixels) if megapixels > 0 else 0.0
        self._frame_density.append(density)
        self._frame_person_count.append(n)

        if n == 0:
            self._frame_mean_speed.append(0.0)
            self._frame_mean_vec.append((0.0, 0.0))
            self._frame_pressure.append(0.0)
            return

        speeds = np.array([t["speed"] for t in track_records], dtype=np.float32)
        mean_speed = float(speeds.mean())
        self._frame_mean_speed.append(mean_speed)

        # Mean velocity VECTOR, not mean speed: oscillation is a reversal of
        # direction, and a crowd surging back and forth has a near-constant
        # mean speed the whole time. Only the vector shows the sign flip.
        vx = float(np.mean([t["heading_vec"][0] * t["speed"] for t in track_records]))
        vy = float(np.mean([t["heading_vec"][1] * t["speed"] for t in track_records]))
        self._frame_mean_vec.append((vx, vy))

        # Crowd pressure, Helbing's rho * Var(v). Variance ACROSS PEOPLE in
        # this frame, which is the spatial velocity variance the definition
        # calls for -- not the per-person local variance already accumulated
        # in _variance_records, which is a neighbourhood statistic.
        self._frame_pressure.append(density * float(speeds.var()))

        # Divergence: mean over the people present, so the series is one
        # sample per frame and comparable with the others above.
        divs = [t.get("local_divergence") for t in track_records
                if t.get("local_divergence") is not None]
        if divs:
            self._divergence_records.append(float(np.mean(divs)))

        # Specific flow: edge-triggered crossings of the counting line.
        line_x = self.SPECIFIC_FLOW_LINE_X_FRAC * w
        live: set = set()
        for t in track_records:
            tid = t["tid"]
            live.add(tid)
            side = 1 if t["cx"] >= line_x else -1
            prev = self._track_last_side.get(tid)
            if prev is not None and prev != side:
                if side > 0:
                    self._line_crossings_pos += 1
                else:
                    self._line_crossings_neg += 1
            self._track_last_side[tid] = side
        # Drop departed tracks so a recycled id cannot register a phantom
        # crossing from wherever its predecessor happened to be standing.
        for gone in [tid for tid in self._track_last_side if tid not in live]:
            del self._track_last_side[gone]

    @staticmethod
    def _negative_autocorrelation_score(series: np.ndarray,
                                        lags: tuple = (5, 10, 15)) -> float:
        """
        Strongest NEGATIVE autocorrelation across `lags`, clipped to [0, 1].

        Shared shape of stop-and-go and oscillation symmetry: both ask "does
        this series reverse itself at a characteristic period?". A queue that
        halts and restarts anti-correlates with itself one halt-period later.
        Positive autocorrelation is steady behaviour and scores 0.

        Mirrors CrowdMetricsEngine._compute_stop_go, but over the whole run at
        finalize() rather than a rolling window per zone -- this model has no
        zones, and a single pass over the completed series is both cheaper and
        less sensitive to where a window happens to land.
        """
        if series.ndim == 1:
            series = series[:, None]
        if len(series) < max(lags) + 5:
            return 0.0
        centred = series - series.mean(axis=0, keepdims=True)
        variance = float(np.sum(centred * centred))
        if variance < 1e-9:
            return 0.0
        scores = []
        for lag in lags:
            if lag < len(centred):
                ac = float(np.sum(centred[:-lag] * centred[lag:]) / variance)
                scores.append(max(0.0, -ac))
        return float(np.clip(max(scores) if scores else 0.0, 0.0, 1.0))

    def _crowd_metric_summary(self) -> dict:
        """
        The six crowd-safety metrics derived from the per-frame series.

        Every value here is in IMAGE-PLANE units because this model has no
        ground-plane homography. `metric_units` names them explicitly rather
        than leaving a bare number to be misread as m/s or persons/m², and
        `is_calibrated: False` says so in one place the UI can key on.
        """
        def _stat(series: list, fn, nd: int = 3):
            return round(float(fn(series)), nd) if series else None

        density_avg = _stat(self._frame_density, np.mean)
        density_peak = _stat(self._frame_density, np.max)
        pressure_avg = _stat(self._frame_pressure, np.mean)
        pressure_peak = _stat(self._frame_pressure, np.max)

        # Divergence is negative under compression, so the worst case is the
        # MINIMUM, not the maximum. Reporting max here would name the moment
        # the crowd was spreading out most as the peak crush signal.
        div_avg = _stat(self._divergence_records, np.mean, 5)
        div_worst = _stat(self._divergence_records, np.min, 5)

        stop_go = self._negative_autocorrelation_score(
            np.asarray(self._frame_mean_speed, dtype=np.float32))
        oscillation = self._negative_autocorrelation_score(
            np.asarray(self._frame_mean_vec, dtype=np.float32).reshape(-1, 2))

        # Specific flow: NET people per second across the counting line. Net,
        # because the metric describes throughput in a direction -- two people
        # passing each other in opposite directions is not two people through
        # the gate. Gross is reported alongside so a busy but balanced
        # crossing is not mistaken for a deserted one.
        duration = max(self._last_timestamp_sec, 1e-6)
        net = self._line_crossings_pos - self._line_crossings_neg
        gross = self._line_crossings_pos + self._line_crossings_neg

        return {
            # 1. Density
            "avg_density": density_avg,
            "peak_density": density_peak,
            "avg_person_count": _stat(self._frame_person_count, np.mean, 1),
            "peak_person_count": int(max(self._frame_person_count)) if self._frame_person_count else 0,
            # 2. Velocity field -- avg_speed_px_frame already exists above;
            #    peak is added here so the card can show both.
            "peak_speed_px_frame": _stat(self._frame_mean_speed, np.max, 2),
            # 3. Specific flow
            "specific_flow_net_per_sec": round(net / duration, 3),
            "specific_flow_gross_per_sec": round(gross / duration, 3),
            "specific_flow_crossings": gross,
            # 5. Crowd pressure
            "avg_crowd_pressure": pressure_avg,
            "peak_crowd_pressure": pressure_peak,
            # 6. Divergence
            "avg_divergence": div_avg,
            "strongest_compression": div_worst,
            # 9. Stop-and-go
            "stop_go_score": round(stop_go, 3),
            # 10. Oscillation symmetry
            "oscillation_symmetry": round(oscillation, 3),

            "frames_measured": len(self._frame_density),
            "is_calibrated": False,
            "metric_units": {
                "density": "persons per megapixel of image area",
                "speed": "px/frame",
                "specific_flow": "persons/sec across frame centre line",
                "crowd_pressure": "density x velocity variance (px² units)",
                "divergence": "px/frame per cell (negative = compression)",
                "stop_go": "0-1 (unitless)",
                "oscillation_symmetry": "0-1 (unitless)",
                "directional_entropy": "bits (0 = aligned, 3 = uniform)",
                "velocity_variance": "px²/frame²",
                "note": (
                    "Image-plane units: this model has no ground-plane "
                    "homography, so density is NOT persons/m² and pressure is "
                    "NOT in s⁻². Values are comparable across frames of the "
                    "same camera, not between cameras or against published "
                    "physical thresholds."
                ),
            },
        }

    def _compute_summary(self) -> dict:
        """Calculate comprehensive run-level summary metrics."""
        total = self._total_detections_count
        label_counts = dict(self._label_counter)
        n_stopped = label_counts.get("person_stopped", 0)
        n_crush = label_counts.get("person_crush_zone", 0)
        n_moving_single = label_counts.get("person_moving", 0)
        n_moving_stream_a = label_counts.get("person_moving_stream_a", 0)
        n_moving_stream_b = label_counts.get("person_moving_stream_b", 0)
        n_moving = n_moving_single + n_moving_stream_a + n_moving_stream_b + n_crush

        pct_moving = round((n_moving / total * 100), 1) if total > 0 else 0.0
        pct_stationary = round((n_stopped / total * 100), 1) if total > 0 else 0.0
        pct_crush_risk = round((n_crush / total * 100), 1) if total > 0 else 0.0
        pct_moving_single = round((n_moving_single / total * 100), 1) if total > 0 else 0.0
        pct_moving_stream_a = round((n_moving_stream_a / total * 100), 1) if total > 0 else 0.0
        pct_moving_stream_b = round((n_moving_stream_b / total * 100), 1) if total > 0 else 0.0

        # Crush events: distinct periods where 3+ people are simultaneously crush-flagged
        crush_events = 0
        in_crush_event = False
        peak_crush_count = 0
        peak_crush_timestamp_sec = 0.0

        for _f_idx, t_sec, c_cnt in self._frame_crush_counts:
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

        for _f_idx, t_sec, cf_cnt, cf_rate in self._frame_counterflow_counts:
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
            a_cnt = dirs.count("stream_a")
            b_cnt = dirs.count("stream_b")
            single_cnt = dirs.count("moving")
            flip_data.append((tid, flips, a_cnt, b_cnt, single_cnt))

        total_tracks = len(flip_data)
        stable_tracks = sum(1 for _, f, _, _, _ in flip_data if f == 0)
        unstable_tracks = [x for x in flip_data if x[1] > 5]
        avg_flips = (sum(f for _, f, _, _, _ in flip_data) / total_tracks) if total_tracks else 0.0
        pct_stable_tracks = round((stable_tracks / total_tracks * 100), 1) if total_tracks else 0.0

        worst_tracks = sorted(flip_data, key=lambda x: -x[1])[:10]
        suspicious_list = []
        for tid, f, a, b, single in worst_tracks:
            if f > 5:
                counts = {"moving": single, "stream_a": a, "stream_b": b}
                dom = max(counts, key=counts.get)
                pct_dom = round(counts[dom] / sum(counts.values()) * 100) if sum(counts.values()) else 0
                suspicious_list.append({
                    "track_id": tid,
                    "flips": f,
                    "stream_a_count": a,
                    "stream_b_count": b,
                    "moving_count": single,
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
            heading_bins.append({
                "range": [round(lo, 1), round(hi, 1)],
                "count": cnt,
            })

        self.summary = {
            "total_detections": total,
            "total_tracks": total_tracks,
            "pct_moving": pct_moving,
            "pct_stationary": pct_stationary,
            "pct_crush_risk": pct_crush_risk,
            "pct_moving_single_stream": pct_moving_single,
            "pct_moving_stream_a": pct_moving_stream_a,
            "pct_moving_stream_b": pct_moving_stream_b,
            "stream_counts": dict(self._stream_counts),
            "crush_event_count": crush_events,
            "peak_crush_timestamp_sec": round(peak_crush_timestamp_sec, 2),
            "peak_crush_people_count": peak_crush_count,
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

        self.summary.update(self._crowd_metric_summary())

        # Screen-relative stream directions from the run's accumulated centres.
        # Additive keys only — existing summary schema is unchanged.
        dir_a, ang_a, dir_b, ang_b = self._run_stream_directions()
        self.summary["stream_a_direction"] = dir_a
        self.summary["stream_b_direction"] = dir_b
        self.summary["stream_a_angle_deg"] = ang_a
        self.summary["stream_b_angle_deg"] = ang_b

        logger.info(
            "CrowdMotionMonitor summary: total=%d, moving=%.1f%% (single:%.1f%%, A:%.1f%%, B:%.1f%%), "
            "crush=%.1f%% (%d events), counterflow=%.1f%% (%d events), entropy=%.3f, var=%.3f",
            total, pct_moving, pct_moving_single, pct_moving_stream_a, pct_moving_stream_b, pct_crush_risk,
            crush_events, pct_cf_people, counterflow_events, avg_entropy, avg_var,
        )
        return self.summary

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apgcc_points_to_boxes(
        points: np.ndarray,
        rtdetr_boxes: list[list[float]],
        frame_shape: tuple[int, int],
    ) -> list[list[float]]:
        """
        Turn surviving APGCC head points into synthetic [x1,y1,x2,y2] boxes,
        dropping any point RT-DETRv2 already has a box on.

        A point is "claimed" if it falls inside an RT-DETRv2 box expanded by
        _APGCC_CLAIM_MARGIN on every side. Only unclaimed points become new
        boxes — this is the whole dedup rule: RT-DETRv2 gets first claim,
        APGCC only fills what it missed, so nobody gets two triangles.
        """
        if points is None or len(points) == 0:
            return []

        h, w = frame_shape
        expanded = []
        for x1, y1, x2, y2 in rtdetr_boxes:
            bw, bh = x2 - x1, y2 - y1
            mx, my = bw * _APGCC_CLAIM_MARGIN, bh * _APGCC_CLAIM_MARGIN
            expanded.append((x1 - mx, y1 - my, x2 + mx, y2 + my))

        out: list[list[float]] = []
        half = _APGCC_SYNTH_HALF_BOX_PX
        for px, py in points:
            claimed = any(ex1 <= px <= ex2 and ey1 <= py <= ey2
                          for ex1, ey1, ex2, ey2 in expanded)
            if claimed:
                continue
            x1 = max(0.0, float(px) - half)
            y1 = max(0.0, float(py) - half)
            x2 = min(float(w), float(px) + half)
            y2 = min(float(h), float(py) + half)
            if x2 > x1 and y2 > y1:
                out.append([x1, y1, x2, y2])
        return out

    @staticmethod
    def _infer_direction_streams(track_records: list[dict]) -> list[tuple[float, float]]:
        """Return two deterministic heading modes, or no modes for one-way flow."""
        vectors = [tr["heading_vec"] for tr in track_records
                   if tr["confirmed"] and tr["is_moving"]]
        if len(vectors) < _STREAM_MIN_TRACKS:
            return []

        points = np.asarray(vectors, dtype=np.float64)
        # Select the most separated pair as deterministic k-means seeds.
        dots = np.clip(points @ points.T, -1.0, 1.0)
        distances = 1.0 - dots
        seed_a, seed_b = np.unravel_index(np.argmax(distances), distances.shape)
        if seed_a == seed_b:
            return []
        centres = np.array([points[seed_a], points[seed_b]], dtype=np.float64)

        assignments = np.zeros(len(points), dtype=np.int8)
        for _ in range(8):
            next_assignments = np.argmax(points @ centres.T, axis=1)
            if np.array_equal(assignments, next_assignments):
                break
            assignments = next_assignments
            updated = []
            for cluster in (0, 1):
                members = points[assignments == cluster]
                if len(members) == 0:
                    return []
                centre = members.mean(axis=0)
                norm = np.linalg.norm(centre)
                if norm <= 1e-6:
                    return []
                updated.append(centre / norm)
            centres = np.asarray(updated)

        counts = np.bincount(assignments, minlength=2)
        separation = math.degrees(math.acos(float(np.clip(centres[0] @ centres[1], -1.0, 1.0))))
        if (min(counts) / len(points) < _STREAM_MIN_CLUSTER_SHARE
                or separation < _STREAM_MIN_SEPARATION_DEG):
            return []

        # Fixed angular ordering gives stream labels stable meanings per frame.
        ordered = sorted((tuple(c) for c in centres), key=lambda c: math.atan2(c[1], c[0]))
        return ordered

    @staticmethod
    def _nearest_stream(
        heading: tuple[float, float], centres: list[tuple[float, float]],
    ) -> str:
        if len(centres) != 2:
            return "moving"
        dots = [heading[0] * centre[0] + heading[1] * centre[1] for centre in centres]
        return "stream_a" if dots[0] >= dots[1] else "stream_b"

    @staticmethod
    def _screen_direction_from_vector(cx: float, cy: float) -> tuple[str, float]:
        """
        Map an image-space heading vector (cx, cy) to a 4-way screen-relative
        label and the raw heading angle in degrees.

        Axis convention matches heading_deg elsewhere in this module:
        ``angle = atan2(cy, cx)`` with +x right and +y downward (OpenCV /
        optical-flow image space).  The existing 2-way colour comments treat
        heading_deg in ±90° as rightward (TEAL-GREEN) and outside ±90° as
        leftward (ELECTRIC BLUE).  This helper refines that into four 90°
        quadrants centred on the axes — not 8-way diagonal buckets.

        These labels are relative to the camera framing, not geographic
        compass directions.  The pipeline has no camera calibration /
        homography by default (see the "Camera is uncalibrated" warning
        path in zones.py).

        Quadrants (boundaries at ±45° / ±135°):
          Rightward        : |angle| <= 45°          (+x)
          Toward camera    :  45° < angle <= 135°    (+y, down the frame)
          Leftward         : |angle| > 135°          (-x)
          Away from camera : -135° <= angle < -45°   (-y, up the frame)
        """
        angle_deg = math.degrees(math.atan2(cy, cx))
        if -45.0 <= angle_deg <= 45.0:
            label = "Rightward"
        elif 45.0 < angle_deg <= 135.0:
            label = "Toward camera"
        elif angle_deg < -135.0 or angle_deg > 135.0:
            label = "Leftward"
        else:
            label = "Away from camera"
        return label, round(angle_deg, 2)

    def _run_stream_directions(self) -> tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
        """One screen-relative label per stream from accumulated centres."""
        if self._stream_centre_n <= 0:
            return None, None, None, None
        out: list[tuple[Optional[str], Optional[float]]] = []
        for acc in self._stream_centre_acc:
            n = math.hypot(acc[0], acc[1])
            if n <= 1e-6:
                out.append((None, None))
                continue
            out.append(self._screen_direction_from_vector(acc[0] / n, acc[1] / n))
        (dir_a, ang_a), (dir_b, ang_b) = out[0], out[1]
        return dir_a, ang_a, dir_b, ang_b

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
