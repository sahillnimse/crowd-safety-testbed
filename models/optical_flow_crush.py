"""
Crowd crush / turbulence detection via dense optical flow (Farneback).

Unlike per-person models (pose, YOLO), this computes a motion vector for
EVERY pixel between two consecutive frames — no detection step required,
so coverage isn't limited by how many people a detector successfully
identifies. This is what gives it full-frame (100%) coverage vs the
3-5% ceiling of person-detection-based turbulence estimates.

Turbulence/crush signal is derived from the flow field itself:
  - convergence -> people being compressed together
  - directional incoherence -> counterflow / turbulence
    (people moving in conflicting directions in the same area = crush risk)

This is classical CV (no GPU/training needed), so it's cheap to run
on every frame pair — good for real-time use once tuned.

**Turbulence must be measured with circular statistics.** Flow angles live
on a circle and wrap at +/-pi, so the plain standard deviation of
`arctan2(fy, fx)` is not a valid measure of directional spread. It has the
signal exactly backwards: coherent *leftward* motion has angles clustered
near +/-pi, which the wraparound splits across both ends of the range for a
std of ~3.1, while genuinely random directions cap at pi/sqrt(3) ~= 1.81.
Any threshold high enough to look selective was therefore unreachable by
real turbulence and fired only on leftward camera pans. Circular variance
(1 - |mean resultant vector|) is bounded in [0, 1], is invariant to which
way the crowd happens to be moving, and increases monotonically with
incoherence — which is what "turbulence" means here.
"""

import numpy as np
import cv2

from models.base import BaseModelWrapper, Detection


class OpticalFlowCrushDetector(BaseModelWrapper):
    consumption_type = "flow_pair"
    name = "optical_flow_crush"
    # Farnebäck dense flow through OpenCV: no torch, no CUDA, nothing that a
    # device setting reaches.  This inherited the base class's default of True
    # simply by not declaring anything, which reported it as GPU-accelerated
    # in every run summary and made the CPU-only set look smaller than it is.
    gpu_accelerated = False

    # Defaults are the ~p99 of each statistic measured over the crowd footage
    # in test_videos/, i.e. they flag roughly the top 1% of moving cells.
    # Both are footage-dependent — re-derive with
    # `python scripts/calibrate_optical_flow.py --video <file>` for new
    # camera angles, framerates, or crowd densities. Note the divergence
    # scale in particular is nowhere near a "small negative number": the
    # p50 of the compression statistic on real crowd footage is about -0.59,
    # so a threshold of -0.15 fires on ~91% of all moving cells.
    def __init__(self,
                 grid_cell_px: int = 32,                  # analyze flow in grid cells, not per-pixel
                 turbulence_threshold: float = 0.65,       # circular variance in [0,1]; higher = more incoherent
                 convergence_threshold: float = -2.0,      # negative divergence = compression
                 min_magnitude: float = 0.3,               # px/frame below which a cell is treated as static
                 device=None):
        super().__init__(device=device)  # unused (CPU-only classical CV), kept for interface consistency
        self.grid_cell_px = grid_cell_px
        self.turbulence_threshold = turbulence_threshold
        self.convergence_threshold = convergence_threshold
        self.min_magnitude = min_magnitude

    def load(self):
        # No model weights to load — Farneback is a classical algorithm,
        # already available via cv2.calcOpticalFlowFarneback.
        self._model = "farneback"  # marker that load() was called

    @staticmethod
    def circular_variance(fx: np.ndarray, fy: np.ndarray) -> float:
        """Magnitude-weighted directional incoherence of a flow patch, in [0, 1].

        0 = every vector points the same way, 1 = directions cancel out
        entirely. Weighting by magnitude keeps near-static pixels, whose
        angles are pure noise, from dominating the estimate.
        """
        magnitude = np.sqrt(fx ** 2 + fy ** 2)
        total = float(magnitude.sum())
        if total <= 1e-9:
            return 0.0
        # Mean resultant vector: sum the unit direction vectors weighted by
        # speed, then normalize. Its length is 1 for perfectly coherent flow.
        mean_x = float((fx).sum()) / total
        mean_y = float((fy).sum()) / total
        resultant = min(1.0, float(np.hypot(mean_x, mean_y)))
        return 1.0 - resultant

    @staticmethod
    def compression(fx: np.ndarray, fy: np.ndarray) -> float:
        """Strongest local inward compression in the patch (negative = converging).

        The mean of the divergence field telescopes to the patch boundary,
        so interior compression cancels against surrounding expansion and
        the very thing being looked for averages itself away. Taking a low
        percentile of the divergence field instead keeps localized
        compression, which is what a crush actually looks like.
        """
        div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
        return float(np.percentile(div, 10))

    def predict(self, frame_pair, frame_index: int, timestamp_sec: float) -> list[Detection]:
        prev_frame, curr_frame = frame_pair
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        # flow shape: (H, W, 2) -> per-pixel (dx, dy)

        h, w = flow.shape[:2]
        g = self.grid_cell_px
        detections = []

        for y in range(0, h - g + 1, g):
            for x in range(0, w - g + 1, g):
                cell = flow[y:y + g, x:x + g]
                fx, fy = cell[..., 0], cell[..., 1]

                magnitude = float(np.mean(np.sqrt(fx ** 2 + fy ** 2)))
                if magnitude < self.min_magnitude:
                    continue  # skip near-static regions (not enough motion to judge)

                incoherence = self.circular_variance(fx, fy)
                div = self.compression(fx, fy)

                is_turbulent = incoherence >= self.turbulence_threshold
                is_converging = div <= self.convergence_threshold
                if not (is_turbulent or is_converging):
                    continue

                # Both conditions can hold at once, and a converging *and*
                # turbulent cell is the highest-risk case there is — emit it
                # rather than letting the turbulence label mask it.
                if is_turbulent and is_converging:
                    label = "crush_risk"
                    conf = max(
                        incoherence,
                        min(1.0, abs(div) / (abs(self.convergence_threshold) * 2)),
                    )
                elif is_turbulent:
                    label = "turbulence"
                    conf = incoherence
                else:
                    label = "convergence"
                    conf = min(1.0, abs(div) / (abs(self.convergence_threshold) * 2))

                detections.append(Detection(
                    model_name=self.name,
                    label=label,
                    confidence=conf,
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    bbox=[x, y, x + g, y + g],
                    extra={"circular_variance": incoherence,
                           "divergence_p10": div,
                           "magnitude": magnitude},
                ))

        return detections
