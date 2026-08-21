"""
Shared pose geometry for the fall detectors.

Every pose-based wrapper (YOLO-pose, MediaPipe, MoveNet) scores posture the
same way here so the comparison between backbones measures the *backbone*,
not three subtly different heuristics.

Two things this module exists to get right, both of which were silently
wrong when each wrapper rolled its own:

1. **Keypoint confidence gating.** Pose estimators return missing joints as
   (0, 0) with score 0 rather than omitting them. Averaging those in drags
   the torso line toward the image origin and manufactures ~45deg "falls"
   out of occluded people — the dominant false-positive source in crowds.
   Joints below `min_kp_conf` are dropped, and if a whole endpoint (both
   shoulders, or both hips) is unusable the frame yields no posture at all
   rather than a fabricated one.

2. **Fall confidence must describe the fall.** The wrappers used to report
   the *person-detector's* confidence, so a "fall @ 0.92" meant "a person
   was detected at 0.92" — uncorrelated with whether they fell. Every
   threshold sweep and PR curve downstream was therefore sweeping over the
   wrong axis. `posture_score()` returns a real posture measure and
   `fall_confidence()` maps it to a calibrated 0-1 confidence.
"""

from typing import Optional, Sequence

import numpy as np

# COCO-17 keypoint indices — shared by YOLOv8-pose, MoveNet, and most
# top-down estimators. MediaPipe/BlazePose uses its own 33-point layout and
# passes its indices in explicitly.
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12

# Below this, a keypoint is treated as "not observed" rather than as (0, 0).
DEFAULT_MIN_KP_CONF = 0.3

# Aspect ratio (bbox width / height) of an upright person vs one lying down.
# Used as a second, pose-independent posture cue: it still works when
# keypoints are noisy, and it disagrees with the torso angle in exactly the
# case the angle is weakest (person falling toward/away from the camera).
UPRIGHT_ASPECT = 0.6
LYING_ASPECT = 1.4

# How much of the posture score comes from the skeleton vs the bbox shape.
ANGLE_WEIGHT = 0.65
ASPECT_WEIGHT = 0.35


def _mean_confident_point(points: Sequence, scores: Optional[Sequence],
                          min_conf: float) -> Optional[np.ndarray]:
    """Mean of the given (x, y) points, skipping any below `min_conf`.

    Returns None if no point survives — the caller must then treat the
    endpoint as unobserved rather than substituting a default.
    """
    usable = []
    for i, pt in enumerate(points):
        if scores is not None and scores[i] < min_conf:
            continue
        x, y = float(pt[0]), float(pt[1])
        # A joint at exactly the origin is the estimator's "missing" sentinel
        # even when it forgot to zero the score alongside it.
        if scores is None and x == 0.0 and y == 0.0:
            continue
        usable.append((x, y))
    if not usable:
        return None
    return np.mean(np.asarray(usable, dtype=np.float64), axis=0)


def torso_angle_deg(keypoints_xy, keypoint_scores=None,
                    min_kp_conf: float = DEFAULT_MIN_KP_CONF,
                    shoulder_idx=(KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
                    hip_idx=(KP_LEFT_HIP, KP_RIGHT_HIP)) -> Optional[float]:
    """Angle of the shoulder->hip line away from vertical, in degrees.

    0deg = perfectly upright, 90deg = perfectly horizontal (lying down).
    Returns None when either endpoint has no confident keypoint, which the
    caller should treat as "no posture reading" rather than "upright".
    """
    kps = np.asarray(keypoints_xy, dtype=np.float64)
    scores = np.asarray(keypoint_scores, dtype=np.float64) if keypoint_scores is not None else None

    max_idx = max(*shoulder_idx, *hip_idx)
    if kps.shape[0] <= max_idx:
        return None

    shoulder = _mean_confident_point(
        [kps[i] for i in shoulder_idx],
        [scores[i] for i in shoulder_idx] if scores is not None else None,
        min_kp_conf,
    )
    hip = _mean_confident_point(
        [kps[i] for i in hip_idx],
        [scores[i] for i in hip_idx] if scores is not None else None,
        min_kp_conf,
    )
    if shoulder is None or hip is None:
        return None

    dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
    if dx == 0.0 and dy == 0.0:
        return None
    # abs() on both components folds the angle into [0, 90]: we care how far
    # from vertical the torso is, not which way it leans or which end is up.
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def bbox_aspect_ratio(bbox) -> Optional[float]:
    """width / height of an [x1, y1, x2, y2] box; None if degenerate."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    if h <= 0:
        return None
    return float((x2 - x1) / h)


def posture_score(angle_deg: Optional[float], aspect: Optional[float] = None) -> Optional[float]:
    """Combined 0-1 "how horizontal is this person" score.

    0 = upright, 1 = flat. Blends the torso angle with the bbox aspect
    ratio; falls back to whichever cue is available. Returns None if
    neither is, so the caller can skip the person instead of guessing.
    """
    angle_score = _clamp01(angle_deg / 90.0) if angle_deg is not None else None
    aspect_score = (
        _clamp01((aspect - UPRIGHT_ASPECT) / (LYING_ASPECT - UPRIGHT_ASPECT))
        if aspect is not None else None
    )

    if angle_score is not None and aspect_score is not None:
        return ANGLE_WEIGHT * angle_score + ASPECT_WEIGHT * aspect_score
    if angle_score is not None:
        return angle_score
    return aspect_score


def fall_confidence(score: float, threshold: float) -> float:
    """Map a posture score onto a 0-1 confidence in the reported label.

    At the decision boundary confidence is 0.5 for both labels and it grows
    toward 1.0 as the score moves away from it — so confidence is
    comparable across models and monotonic in how sure the call is.
    """
    if score >= threshold:
        span = max(1e-6, 1.0 - threshold)
        return 0.5 + 0.5 * _clamp01((score - threshold) / span)
    span = max(1e-6, threshold)
    return 0.5 + 0.5 * _clamp01((threshold - score) / span)


def verdict_from_scores(scores: Sequence[Optional[float]], threshold: float,
                        confirm_frames: int) -> tuple[bool, float]:
    """Collapse a track's per-frame posture scores into (is_fall, confidence).

    Used by the temporal models when no trained checkpoint is available, so
    they still produce a meaningful fall verdict from geometry instead of
    scoring with randomly-initialized weights. Requires the last
    `confirm_frames` frames to all be past the threshold, and reports
    confidence from the mean score over that confirmed window.
    """
    usable = [s if s is not None else 0.0 for s in scores]
    if len(usable) < confirm_frames:
        return False, fall_confidence(usable[-1] if usable else 0.0, threshold)

    window = usable[-confirm_frames:]
    is_fall = all(s >= threshold for s in window)
    mean_score = float(np.mean(window))
    return is_fall, fall_confidence(mean_score, threshold)


def angle_threshold_to_score(horizontal_angle_threshold_deg: float) -> float:
    """Convert the wrappers' public `horizontal_angle_threshold_deg` knob
    into the equivalent posture-score threshold, so the existing default of
    45deg still means "halfway between upright and flat"."""
    return _clamp01(horizontal_angle_threshold_deg / 90.0)
