"""
Fall detection via dense optical flow + sudden-vertical-motion heuristic.

Doesn't rely on pose estimation at all — instead looks for the motion
signature of a fall directly in the flow field: a strong, sustained
downward flow vector concentrated in a compact region (a body dropping),
followed by that region's motion collapsing toward near-zero (impact +
stillness). This is deliberately a different failure mode than the
pose-based wrappers: it doesn't break when keypoint estimation fails
(heavy occlusion, extreme distance, motion blur), but it also can't
distinguish "a person fell" from "a person sat down abruptly" or "an
object was dropped" — it has no notion of *what* the moving region is.

Useful primarily as a robustness comparison point against the
pose-based models rather than a standalone production candidate:
where pose models silently degrade in dense/occluded crowds, this one
keeps producing signal (of lower precision) since it needs no person
detection step at all.

Three properties this needs to detect anything at all, none of which a
plain fixed-grid mean-flow threshold has:

  - **Global motion compensation.** The frame's median flow vector is
    subtracted before anything else. Without it, a downward camera tilt or
    a pan reads as every cell falling simultaneously.
  - **A falling body moves between cells.** Requiring N consecutive
    downward frames in one fixed cell almost never holds — a body crossing
    24px cells fragments its own history. Drop runs propagate through each
    cell's 8-neighbourhood, so the run follows the body.
  - **Percentile, not mean, per cell.** A person occupies part of a cell;
    averaging their motion with the static background around them dilutes
    the drop below any useful threshold.

Classical CV (Farneback flow), no GPU/training required — same
building block as optical_flow_crush.py but tuned for fall's
motion pattern rather than crowd-turbulence's.
"""

import cv2
import numpy as np

from models.base import BaseModelWrapper, Detection


class OpticalFlowFallDetector(BaseModelWrapper):
    gpu_accelerated = False  # CPU-only (classical OpenCV optical flow, no torch/GPU involved)
    consumption_type = "flow_pair"
    name = "fall_optical_flow"

    def __init__(self, grid_cell_px: int = 24, drop_velocity_threshold: float = 3.0,
                 stillness_threshold: float = 0.8, drop_frames_required: int = 3,
                 downward_percentile: float = 75.0,
                 compensate_global_motion: bool = True,
                 device=None):
        super().__init__(device=device)  # unused (classical CV), kept for interface consistency
        self.grid_cell_px = grid_cell_px
        self.drop_velocity_threshold = drop_velocity_threshold  # px/frame downward flow to flag as a drop
        self.stillness_threshold = stillness_threshold           # px/frame magnitude counted as "settled"
        self.drop_frames_required = drop_frames_required
        self.downward_percentile = downward_percentile
        self.compensate_global_motion = compensate_global_motion
        # cell (row, col) -> (consecutive drop frames, peak downward velocity)
        self._drop_runs: dict[tuple[int, int], tuple[int, float]] = {}

    def load(self):
        self._model = "farneback"  # marker that load() was called; no weights to load
        self._drop_runs = {}

    def _neighbourhood_run(self, row: int, col: int) -> tuple[int, float]:
        """Longest drop run among this cell and its 8 neighbours.

        This is what lets a run follow a body as it descends across cell
        boundaries instead of resetting every time it crosses one.
        """
        best_len, best_peak = 0, 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                run = self._drop_runs.get((row + dr, col + dc))
                if run is None:
                    continue
                run_len, peak = run
                if run_len > best_len:
                    best_len, best_peak = run_len, peak
                elif run_len == best_len:
                    best_peak = max(best_peak, peak)
        return best_len, best_peak

    def predict(self, frame_pair, frame_index: int, timestamp_sec: float) -> list[Detection]:
        prev_frame, curr_frame = frame_pair
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        global_motion = (0.0, 0.0)
        if self.compensate_global_motion:
            # Median over the whole frame approximates camera motion: in a
            # scene where most pixels are background, the median flow vector
            # is the background's, and subtracting it leaves body motion.
            gx = float(np.median(flow[..., 0]))
            gy = float(np.median(flow[..., 1]))
            global_motion = (gx, gy)
            flow = flow - np.array([gx, gy], dtype=flow.dtype)

        h, w = flow.shape[:2]
        g = self.grid_cell_px
        detections = []
        next_runs: dict[tuple[int, int], tuple[int, float]] = {}

        for y in range(0, h - g + 1, g):
            for x in range(0, w - g + 1, g):
                cell = flow[y:y + g, x:x + g]
                fy = cell[..., 1]  # vertical component; positive = downward in image coords

                # Upper-percentile downward flow, so a body occupying part of
                # the cell isn't averaged away by the static background.
                v_down = float(np.percentile(fy, self.downward_percentile))
                magnitude = float(np.median(np.sqrt(cell[..., 0] ** 2 + cell[..., 1] ** 2)))

                row, col = y // g, x // g
                prior_len, prior_peak = self._neighbourhood_run(row, col)

                if v_down > self.drop_velocity_threshold:
                    # Still dropping — extend the run through this cell.
                    next_runs[(row, col)] = (prior_len + 1, max(prior_peak, v_down))
                    continue

                if prior_len >= self.drop_frames_required and magnitude < self.stillness_threshold:
                    # Sustained descent that has just come to rest: a body
                    # hitting the ground and staying there.
                    conf = min(1.0, prior_peak / (self.drop_velocity_threshold * 2))
                    detections.append(Detection(
                        model_name=self.name,
                        label="fall",
                        confidence=conf,
                        timestamp_sec=timestamp_sec,
                        frame_index=frame_index,
                        bbox=[x, y, x + g, y + g],
                        extra={
                            "peak_downward_velocity": prior_peak,
                            "settle_magnitude": magnitude,
                            "drop_run_frames": prior_len,
                            "global_motion": list(global_motion),
                        },
                    ))

        # Runs not extended this frame are over; only surviving ones carry on.
        self._drop_runs = next_runs
        return detections
