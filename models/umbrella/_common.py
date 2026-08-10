"""
Shared emission logic for the umbrella detectors.

The three umbrella models exist to be compared against each other, so the
part *after* the network — box filtering, area gating, per-frame counting,
Detection construction — is deliberately identical across all of them.
Anything that differs between the wrappers is then a difference in the
detector, not in the bookkeeping wrapped around it. Same reasoning as
models/fall/_geometry.py sharing one posture heuristic across three pose
backbones.

The `umbrellas_in_frame` count is computed *after* filtering so the number
reported always matches the boxes actually emitted, rather than the raw
detector output.
"""

from typing import Optional, Sequence

from models.base import Detection

# Drop boxes below this fraction of frame area. Filters the speck-sized
# false positives busy backgrounds produce, without discarding genuinely
# distant umbrellas.
DEFAULT_MIN_AREA_FRAC = 0.0002


def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    return [x1, y1, x2, y2]


def emit_umbrellas(boxes: Sequence, confidences: Sequence,
                   track_ids: Optional[Sequence], frame_shape,
                   model_name: str, frame_index: int, timestamp_sec: float,
                   min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
                   extra_fields: Optional[dict] = None) -> list[Detection]:
    """Filter raw detector output and build the common Detection list."""
    height, width = frame_shape[:2]
    frame_area = float(height * width)
    if frame_area <= 0:
        return []

    kept = []
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = clamp_box(box, width, height)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or area / frame_area < min_area_frac:
            continue
        tid = None
        if track_ids is not None and i < len(track_ids) and track_ids[i] is not None:
            tid = int(track_ids[i])
        kept.append(([x1, y1, x2, y2], float(conf), area, tid))

    in_frame = len(kept)
    base_extra = extra_fields or {}

    return [
        Detection(
            model_name=model_name,
            label="umbrella",
            confidence=conf,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            bbox=bbox,
            extra={
                "track_id": tid,
                "umbrellas_in_frame": in_frame,
                "area_frac": round(area / frame_area, 5),
                **base_extra,
            },
        )
        for bbox, conf, area, tid in kept
    ]


# COCO class index for "umbrella".  Shared by every backend that scores
# against COCO rather than a fine-tuned single-class head.
UMBRELLA_COCO_CLASS = 25
