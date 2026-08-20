"""
Summarizes a pose_fall detection JSON log into a readable timeline table,
similar to summarize_log.py but for standing/fall labels instead of
turbulence/convergence.

Usage:
    python scripts/summarize_pose_log.py outputs/logs/YzcawvDGe4Y_pose_fall.json
"""

import argparse
import json
import sys
import os
from collections import defaultdict


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Summarize pose fall detection JSON log.")
    parser.add_argument("input_path", help="Path to pose fall detection JSON log")
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)
    print(f"Loading {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        detections = json.load(f)

    if not detections:
        print("No detections found in this log.")
        return

    total = len(detections)
    standing = [d for d in detections if d["label"] == "standing"]
    fall = [d for d in detections if d["label"] == "fall"]

    max_ts = max(d["timestamp_sec"] for d in detections)

    print("=" * 60)
    print(" POSE FALL DETECTION SUMMARY (Model: pose_fall)")
    print(f" Total Person-Frame Detections: {total:,}")
    print(f" Video Duration (approx):       {format_time(max_ts)} ({max_ts:.1f}s)")
    print("=" * 60)
    print(f" Standing: {len(standing):,} ({100*len(standing)/total:.2f}%)")
    print(f" Fall candidates: {len(fall):,} ({100*len(fall)/total:.2f}%)")
    print("=" * 60)

    if not fall:
        print("\nNo fall candidates detected in this video.")
        return

    # Bucket fall candidates into 5-second windows, like the optical flow summary
    window_size = 5
    buckets = defaultdict(int)
    for d in fall:
        window_start = int(d["timestamp_sec"] // window_size) * window_size
        buckets[window_start] += 1

    print("\nFall Candidate Timeline (5-second windows with >=1 flag):")
    print("-" * 60)
    print(f"{'Time Window':<16}| {'Fall Flags':<12}| Bar")
    print("-" * 60)
    for window_start in sorted(buckets.keys()):
        window_end = window_start + window_size
        count = buckets[window_start]
        bar = "#" * count
        label = f"{format_time(window_start)}-{format_time(window_end)}"
        print(f"{label:<16}| {count:<12}| {bar}")

    print("=" * 60)
    print("\nNOTE: These are single-frame candidate flags with no temporal")
    print("smoothing. Manually review the annotated video at these exact")
    print("timestamps before treating any of these as confirmed falls —")
    print("bending, crouching, and sitting can trigger the same heuristic.")

    # Export CSV alongside, matching the style of the optical flow summary
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    csv_path = os.path.join(base_dir, f"{base_name}_fall_summary.csv")
    with open(csv_path, "w") as f:
        f.write("window_start_sec,window_end_sec,timestamp_formatted,fall_flags\n")
        for window_start in sorted(buckets.keys()):
            window_end = window_start + window_size
            label = f"{format_time(window_start)}-{format_time(window_end)}"
            f.write(f"{window_start},{window_end},{label},{buckets[window_start]}\n")

    print(f"\nCSV summary exported to: {csv_path}")


if __name__ == "__main__":
    main()