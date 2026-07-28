"""
Skeleton sequence preparation for the temporal fall models (LSTM, ST-GCN,
PoseC3D).

The critical rule here is **normalize over the clip, never per frame**.

Every sequence model in this testbed was previously fed either raw pixel
coordinates (so the model learned "where in the frame" instead of "what
shape") or coordinates renormalized against each frame's own keypoint
extent (which rescales a descending body back to full size on every frame,
deleting the vertical motion that *is* the fall).

Normalizing once against clip-level statistics fixes both: the skeleton is
translation- and scale-invariant, but motion within the clip survives.
"""

from typing import Optional

import numpy as np

# Indices into a COCO-17 skeleton, used to define the body-centred frame.
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12


def _mid(points: np.ndarray, i: int, j: int) -> np.ndarray:
    return (points[i] + points[j]) / 2.0


def normalize_sequence(keypoint_seq, eps: float = 1e-6) -> np.ndarray:
    """(T, V, 2) pixel keypoints -> (T, V, 2) normalized skeleton sequence.

    Origin is the mid-hip of the *first* frame and the scale is the median
    torso length across the clip. Because both are computed once for the
    whole clip rather than per frame, a body that drops 200px still drops
    in the normalized sequence.
    """
    seq = np.asarray(keypoint_seq, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[0] == 0:
        raise ValueError(f"expected (T, V, 2) keypoint sequence, got {seq.shape}")
    seq = seq[..., :2]

    num_joints = seq.shape[1]
    if num_joints > max(KP_LEFT_HIP, KP_RIGHT_HIP):
        origin = _mid(seq[0], KP_LEFT_HIP, KP_RIGHT_HIP)
        torso_lengths = [
            np.linalg.norm(_mid(f, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
                           - _mid(f, KP_LEFT_HIP, KP_RIGHT_HIP))
            for f in seq
        ]
        scale = float(np.median(torso_lengths))
    else:
        # Unknown layout: fall back to the clip's own centroid and spread.
        origin = seq[0].mean(axis=0)
        scale = float(np.median(np.linalg.norm(seq - seq.mean(axis=1, keepdims=True), axis=-1)))

    if not np.isfinite(scale) or scale < eps:
        # Degenerate skeleton (all joints coincident, or all dropped) — use
        # the clip's overall extent so we at least stay numerically sane.
        scale = float(max(np.ptp(seq[..., 0]), np.ptp(seq[..., 1]), 1.0))

    return (seq - origin) / scale


def clip_bbox(keypoint_seq, margin: float = 0.1):
    """Bounding box (x_min, y_min, x_max, y_max) over the whole clip.

    Clip-level, deliberately: rendering each frame into its own per-frame
    box is what erased the fall motion from the PoseC3D heatmaps.
    """
    seq = np.asarray(keypoint_seq, dtype=np.float32)
    xs, ys = seq[..., 0], seq[..., 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    # Pad so joints at the extremes aren't rendered exactly on the border,
    # where half of their gaussian would fall outside the heatmap.
    return (x_min - margin * w, y_min - margin * h,
            x_max + margin * w, y_max + margin * h)


def render_heatmap_volume(keypoint_seq, heatmap_size: int = 56,
                          sigma: float = 2.0,
                          keypoint_scores: Optional[np.ndarray] = None,
                          min_kp_conf: float = 0.0) -> np.ndarray:
    """(T, V, 2) keypoints -> (T, V, H, W) gaussian heatmap volume.

    This is the input format PoseC3D consumes. Two properties matter:

    - **Actual gaussians**, not single lit pixels. A one-hot pixel in a
      56x56 map is very nearly zero signal once the 3D-CNN's stem has
      pooled it away; the spatial support of the gaussian is what the
      convolutions have to work with.
    - **Clip-level coordinate normalization** (see `clip_bbox`), so the
      joint trajectory across the clip is preserved.
    """
    seq = np.asarray(keypoint_seq, dtype=np.float32)[..., :2]
    t, num_joints = seq.shape[0], seq.shape[1]

    x_min, y_min, x_max, y_max = clip_bbox(seq)
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)

    volume = np.zeros((t, num_joints, heatmap_size, heatmap_size), dtype=np.float32)

    # Precompute the pixel grid once; each joint just re-centres the gaussian.
    grid = np.arange(heatmap_size, dtype=np.float32)
    two_sigma_sq = 2.0 * sigma * sigma
    # Beyond ~3 sigma the gaussian is numerically irrelevant — only paint a
    # window that size around each joint instead of the whole map.
    radius = int(np.ceil(3 * sigma))

    for frame_i in range(t):
        for j in range(num_joints):
            if keypoint_scores is not None and keypoint_scores[frame_i][j] < min_kp_conf:
                continue  # unobserved joint: leave its channel empty this frame

            cx = (seq[frame_i, j, 0] - x_min) / span_x * (heatmap_size - 1)
            cy = (seq[frame_i, j, 1] - y_min) / span_y * (heatmap_size - 1)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue

            x0, x1 = max(0, int(cx) - radius), min(heatmap_size, int(cx) + radius + 1)
            y0, y1 = max(0, int(cy) - radius), min(heatmap_size, int(cy) + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue

            gx = np.exp(-((grid[x0:x1] - cx) ** 2) / two_sigma_sq)
            gy = np.exp(-((grid[y0:y1] - cy) ** 2) / two_sigma_sq)
            volume[frame_i, j, y0:y1, x0:x1] = np.outer(gy, gx)

    return volume


def resample_sequence(seq: list, target_len: int) -> list:
    """Uniformly resample a list to exactly `target_len` entries.

    Used so a track shorter or longer than the model's expected window
    still produces a correctly-shaped clip.
    """
    if not seq:
        return seq
    idxs = np.linspace(0, len(seq) - 1, target_len).round().astype(int)
    return [seq[i] for i in idxs]
