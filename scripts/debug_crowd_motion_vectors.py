"""Compare sampled Farneback motion with tracked-box displacement.

This diagnostic consumes an existing CrowdMotionMonitor detections JSON and
the matching source video.  It is intentionally read-only: use --track-id for
people identified by inspecting the annotated video, then compare the raw flow
vector with the track-centroid displacement over the same frame pair.

Example:
    python scripts/debug_crowd_motion_vectors.py --track-id 1817 --track-id 1508
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO = Path("test_videos/Foregin Crowd.mp4")
DEFAULT_DETECTIONS = Path("outputs/runs/Foregin Crowd/crowd_motion_monitor/detections.json")


def _centre(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read source frame {index}")
    return frame


def _sample_flow(prev: np.ndarray, curr: np.ndarray, box: list[float]) -> tuple[float, float]:
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    sx1, sx2 = x1 + max(1, int(bw * 0.20)), x2 - max(1, int(bw * 0.20))
    sy1, sy2 = y1 + int(bh * 0.40), y1 + int(bh * 0.70)
    if sy2 <= sy1:
        inset = max(1, int(bh * 0.20))
        sy1, sy2 = y1 + inset, y2 - inset
    patch = flow[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return 0.0, 0.0
    return float(np.median(patch[..., 0])), float(np.median(patch[..., 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--track-id", type=int, action="append", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    by_track: dict[int, list[dict]] = defaultdict(list)
    for detection in detections:
        by_track[detection["extra"]["track_id"]].append(detection)
    cap = cv2.VideoCapture(str(args.video))
    try:
        print("track frame  flow(vx,vy)  centroid(dx,dy)  old_deg  corrected_deg")
        for track_id in args.track_id:
            records = sorted(by_track.get(track_id, []), key=lambda item: item["frame_index"])
            if len(records) < 2:
                print(f"{track_id}: fewer than two samples")
                continue
            printed = 0
            for previous, current in zip(records, records[1:]):
                frame_gap = current["frame_index"] - previous["frame_index"]
                if frame_gap <= 0:
                    continue
                if args.start_frame is not None and current["frame_index"] < args.start_frame:
                    continue
                if args.end_frame is not None and current["frame_index"] > args.end_frame:
                    continue
                prev_frame = _frame(cap, previous["frame_index"])
                curr_frame = _frame(cap, current["frame_index"])
                vx, vy = _sample_flow(prev_frame, curr_frame, current["bbox"])
                px, py = _centre(previous["bbox"])
                cx, cy = _centre(current["bbox"])
                dx, dy = cx - px, cy - py
                old_deg = math.degrees(math.atan2(-vy, vx))
                corrected_deg = math.degrees(math.atan2(vy, vx))
                print(
                    f"{track_id:5d} {current['frame_index']:5d}  "
                    f"({vx:7.2f},{vy:7.2f})  ({dx:7.2f},{dy:7.2f})  "
                    f"{old_deg:7.1f}  {corrected_deg:7.1f}"
                )
                printed += 1
                if args.max_samples is not None and printed >= args.max_samples:
                    break
    finally:
        cap.release()


if __name__ == "__main__":
    main()
