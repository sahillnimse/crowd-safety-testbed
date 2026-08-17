"""
analyse_crowd_directions.py
---------------------------
Accuracy / sanity check for the CrowdMotionMonitor 3-crowd-type colour scheme.

Usage:
    python scripts/analyse_crowd_directions.py
    python scripts/analyse_crowd_directions.py --json path/to/detections.json

What it reports
---------------
1.  Label distribution  -- how many detections per crowd type
2.  Heading angle histogram -- are right/left splits where we expect them?
3.  Per-track direction consistency -- each track should ideally stick to
    one direction; high flip rate = noisy heading or real direction change
4.  Collision-zone co-occurrence -- which % of crush-zone detections are also
    near heading-boundary (+-90 deg), i.e. genuinely opposing crowds meeting
5.  Speed vs. direction cross-tab -- stopped people should not be split R/L
6.  Suspicious tracks -- flips direction > FLIP_THRESHOLD times
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

# ---- Config ------------------------------------------------------------------
DEFAULT_JSON = (
    r"outputs\runs\Foregin Crowd\crowd_motion_monitor\detections.json"
)
FLIP_THRESHOLD    = 5     # flag a track if it changes direction > N times
BOUNDARY_MARGIN   = 15.0  # degrees: "near +-90 deg" = within this of the boundary
RIGHT_THRESH      = 90.0  # must match _HEADING_RIGHT_THRESH in the model
HIST_BINS         = 18    # 20-deg bins across 360 deg
BAR_WIDTH         = 40    # chars for histogram bar


def heading_to_direction(deg: float) -> str:
    return "right" if abs(deg) < RIGHT_THRESH else "left"


def is_near_boundary(deg: float) -> bool:
    return abs(abs(deg) - RIGHT_THRESH) < BOUNDARY_MARGIN


def load_detections(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "detections" in data:
        return data["detections"]
    raise ValueError(f"Unrecognised JSON structure in {path}")


def run_analysis(json_path: str) -> None:
    print(f"\n{'='*70}")
    print(f"  Crowd Direction Accuracy Analysis")
    print(f"  Source: {json_path}")
    print(f"{'='*70}\n")

    dets = load_detections(json_path)
    total = len(dets)
    if total == 0:
        print("No detections found.")
        return

    print(f"Total detections loaded : {total:,}\n")

    # -- 1. Label distribution -------------------------------------------------
    label_counts: Counter = Counter()
    for d in dets:
        label_counts[d.get("label", "unknown")] += 1

    print("-- 1. Label Distribution ------------------------------------------")
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<25}  {cnt:>7,}  ({pct:5.1f}%)  {bar}")
    print()

    # -- 2. Heading angle histogram --------------------------------------------
    print("-- 2. Heading Angle Histogram (20-deg bins, -180 to +180) ---------")
    bin_size = 360 / HIST_BINS
    bin_counts = [0] * HIST_BINS

    heading_vals: list[float] = []
    for d in dets:
        extra = d.get("extra", {})
        h = extra.get("heading_deg")
        if h is not None:
            heading_vals.append(h)
            idx = int((h + 180) / bin_size) % HIST_BINS
            bin_counts[idx] += 1

    max_bin = max(bin_counts) or 1
    for i, cnt in enumerate(bin_counts):
        lo = -180 + i * bin_size
        hi = lo + bin_size
        bar_len = int(cnt / max_bin * BAR_WIDTH)
        bar = "#" * bar_len
        direction = " <- LEFT" if (lo < -RIGHT_THRESH or hi > RIGHT_THRESH) else " -> RIGHT"
        print(f"  [{lo:+7.1f}, {hi:+7.1f})  {cnt:>6,}  {bar:<{BAR_WIDTH}}{direction}")
    print()

    right_count = left_count = 0
    if heading_vals:
        right_count = sum(1 for h in heading_vals if abs(h) < RIGHT_THRESH)
        left_count  = len(heading_vals) - right_count
        print(f"  Rightward detections : {right_count:,}  ({right_count/len(heading_vals)*100:.1f}%)")
        print(f"  Leftward  detections : {left_count:,}  ({left_count/len(heading_vals)*100:.1f}%)")
    print()

    # -- 3. Per-track direction consistency ------------------------------------
    print("-- 3. Per-Track Direction Consistency -----------------------------")
    track_dirs: dict[int, list[str]] = defaultdict(list)
    for d in dets:
        extra = d.get("extra", {})
        tid = extra.get("track_id")
        h   = extra.get("heading_deg")
        if tid is not None and h is not None:
            track_dirs[tid].append(heading_to_direction(h))

    flip_data: list[tuple[int, int, int, int]] = []
    for tid, dirs in track_dirs.items():
        flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
        right = dirs.count("right")
        left  = dirs.count("left")
        flip_data.append((tid, flips, right, left))

    total_tracks = len(flip_data)
    stable   = sum(1 for _, f, _, _ in flip_data if f == 0)
    unstable = sum(1 for _, f, _, _ in flip_data if f > FLIP_THRESHOLD)
    avg_flips = sum(f for _, f, _, _ in flip_data) / total_tracks if total_tracks else 0

    print(f"  Total confirmed tracks : {total_tracks:,}")
    print(f"  Zero-flip (stable)     : {stable:,}  ({stable/total_tracks*100:.1f}%)")
    print(f"  High-flip (>{FLIP_THRESHOLD})       : {unstable:,}  ({unstable/total_tracks*100:.1f}%)")
    print(f"  Average flips / track  : {avg_flips:.2f}")
    print()

    if unstable:
        print(f"  Suspicious tracks (>{FLIP_THRESHOLD} direction flips) [top 10]:")
        worst = sorted(flip_data, key=lambda x: -x[1])[:10]
        for tid, f, r, l in worst:
            if f > FLIP_THRESHOLD:
                dominant = "right" if r >= l else "left"
                pct_dom  = max(r, l) / (r + l) * 100 if (r + l) else 0
                print(f"    Track {tid:>5}: {f:>3} flips  "
                      f"(right={r}, left={l}, dominant={dominant} {pct_dom:.0f}%)")
        print()

    # -- 4. Collision-zone boundary co-occurrence ------------------------------
    print("-- 4. Collision-Zone Near-Boundary Analysis -----------------------")
    crush_dets = [d for d in dets if d.get("label") == "person_crush_zone"]
    if crush_dets:
        near_boundary = sum(
            1 for d in crush_dets
            if is_near_boundary(d.get("extra", {}).get("heading_deg", 0.0))
        )
        pct = near_boundary / len(crush_dets) * 100
        print(f"  Crush-zone detections          : {len(crush_dets):,}")
        print(f"  Near direction boundary (+-{BOUNDARY_MARGIN:.0f}deg): {near_boundary:,}  ({pct:.1f}%)")
        print(f"  => {pct:.0f}% of collision zones are where right/left crowds meet")
    else:
        print("  No crush-zone detections found  (expected after first re-run with new code).")
    print()

    # -- 5. Speed vs. direction cross-tab --------------------------------------
    print("-- 5. Speed vs. Direction Cross-Tab -------------------------------")
    speed_dir: dict[str, list[float]] = defaultdict(list)
    for d in dets:
        extra = d.get("extra", {})
        spd = extra.get("speed_px_frame")
        lbl = d.get("label", "")
        if spd is not None and lbl:
            speed_dir[lbl].append(spd)

    for lbl, speeds in sorted(speed_dir.items()):
        avg = sum(speeds) / len(speeds)
        mx  = max(speeds)
        print(f"  {lbl:<25}  avg={avg:5.2f} px/fr  max={mx:5.2f}  n={len(speeds):,}")
    print()

    # -- Summary verdict -------------------------------------------------------
    print("-- Summary / Accuracy Verdict -------------------------------------")
    issues: list[str] = []

    if heading_vals:
        ratio = right_count / len(heading_vals)
        if ratio > 0.85 or ratio < 0.15:
            issues.append(
                f"Very uneven direction split ({ratio*100:.0f}% right). "
                "Scene may be mostly unidirectional -- verify visually."
            )

    if avg_flips > 3:
        issues.append(
            f"High average flip rate ({avg_flips:.1f} flips/track). "
            "Consider raising EMA alpha or stationary_frames for smoother heading."
        )

    # Check that old labels no longer appear (migration check)
    old_labels = {"person_moving"}
    found_old = [l for l in label_counts if l in old_labels]
    if found_old:
        issues.append(
            f"Old label(s) found: {found_old}. "
            "This JSON was from a previous run -- re-process the video."
        )

    if issues:
        print("  ISSUES / NOTES:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  OK -- Direction detection looks healthy, no major issues found.")

    print(f"\n{'='*70}")
    print("  Tip: re-run this script after processing a new video to see")
    print("  updated direction metrics with the 3-crowd-type colour scheme.")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse crowd direction detection accuracy from detections.json"
    )
    parser.add_argument(
        "--json", default=DEFAULT_JSON,
        help=f"Path to detections.json (default: {DEFAULT_JSON})"
    )
    args = parser.parse_args()

    json_path = args.json
    if not os.path.isabs(json_path):
        project_root = Path(__file__).resolve().parent.parent
        json_path = str(project_root / json_path)

    if not os.path.exists(json_path):
        print(f"\nERROR: File not found:\n  {json_path}")
        print("Run the webapp on a video first, then re-run this script.\n")
        return

    run_analysis(json_path)


if __name__ == "__main__":
    main()
