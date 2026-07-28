"""
Calibration helper: analyzes the real distribution of optical flow statistics
(circular variance, divergence, magnitude) across a video, so you can set
thresholds based on actual percentiles instead of guessing.

Uses the exact same statistics as models/optical_flow_crush.py — importing
them rather than reimplementing, so calibration can't drift away from what
the detector actually computes. It previously did reimplement them, and
calibrated a metric (plain std of flow angles) that measured leftward
motion rather than turbulence; the percentiles it printed were real, they
just described the wrong thing.

Note the percentile-based suggestions assume the video is *mostly* benign:
they pick a threshold that fires on the top few percent of cells. If the
clip is wall-to-wall crush footage, that logic calibrates the danger away —
run it on ordinary crowd footage and then verify against the incident clip.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.optical_flow_crush import OpticalFlowCrushDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--grid_cell_px", type=int, default=32)
    parser.add_argument("--sample_every_n_frames", type=int, default=3)
    parser.add_argument("--min_magnitude", type=float, default=0.3)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read any frames from {args.video}")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    variances = []
    divergences = []
    magnitudes = []

    frame_idx = 1
    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        frame_idx += 1

        if frame_idx % args.sample_every_n_frames != 0:
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        prev_gray = curr_gray

        h, w = flow.shape[:2]
        g = args.grid_cell_px
        for y in range(0, h - g + 1, g):
            for x in range(0, w - g + 1, g):
                cell = flow[y:y + g, x:x + g]
                fx, fy = cell[..., 0], cell[..., 1]
                magnitude = float(np.mean(np.sqrt(fx ** 2 + fy ** 2)))
                if magnitude < args.min_magnitude:
                    continue
                variances.append(OpticalFlowCrushDetector.circular_variance(fx, fy))
                divergences.append(OpticalFlowCrushDetector.compression(fx, fy))
                magnitudes.append(magnitude)

    pbar.close()
    cap.release()

    if not variances:
        print(f"No cell exceeded min_magnitude={args.min_magnitude} — the video "
              "is essentially static at this grid size. Nothing to calibrate.")
        return

    variances = np.array(variances)
    divergences = np.array(divergences)
    magnitudes = np.array(magnitudes)

    print(f"\nAnalyzed {len(variances)} moving grid cells across the video.\n")

    print("=== CIRCULAR VARIANCE (turbulence signal, 0=coherent 1=incoherent) ===")
    for p in [50, 75, 90, 95, 97, 99, 99.5]:
        print(f"  p{p}: {np.percentile(variances, p):.3f}")

    print("\n=== DIVERGENCE p10 (negative = convergence/compression) ===")
    for p in [0.5, 1, 3, 5, 10, 25, 50]:
        print(f"  p{p}: {np.percentile(divergences, p):.3f}")

    print("\n=== MAGNITUDE (motion strength) ===")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(magnitudes, p):.3f}")

    print("\n--- Suggested starting thresholds ---")
    print(f"turbulence_threshold  ~ {np.percentile(variances, 97):.3f}")
    print(f"convergence_threshold ~ {np.percentile(divergences, 3):.3f}")
    print("\nStart here, then tighten further if it still fires too often visually.")


if __name__ == "__main__":
    main()
