"""
Summarizes detection JSON logs into a human-readable text timeline and CSV summary.

Usage:
    python scripts/summarize_log.py outputs/logs/DJd5F3G9Qbg_optical_flow.json
"""

import sys
import os
import json
from collections import defaultdict

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def summarize(json_path: str, bucket_sec: int = 5):
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return

    print(f"Loading {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    if not detections:
        print("No detections found in log.")
        return

    model_name = detections[0].get("model_name", "unknown")
    total_detections = len(detections)
    
    # Group by time bucket
    buckets = defaultdict(lambda: defaultdict(int))
    max_time = 0.0

    for d in detections:
        t = d.get("timestamp_sec", 0.0)
        label = d.get("label", "event")
        bucket_idx = int(t // bucket_sec) * bucket_sec
        buckets[bucket_idx][label] += 1
        if t > max_time:
            max_time = t

    print("\n" + "="*60)
    print(f" DETECTED EVENTS TIMELINE SUMMARY (Model: {model_name})")
    print(f" Total Detections: {total_detections:,}")
    print(f" Video Duration:   {format_timestamp(max_time)} ({max_time:.1f}s)")
    print("="*60)
    print(f"{'Time Window':<15} | {'Turbulence':<12} | {'Convergence':<12} | Incident Bar")
    print("-" * 60)

    max_events_in_bucket = max(sum(b.values()) for b in buckets.values()) if buckets else 1

    sorted_buckets = sorted(buckets.keys())
    for b_start in sorted_buckets:
        b_end = b_start + bucket_sec
        turb = buckets[b_start].get("turbulence", 0)
        conv = buckets[b_start].get("convergence", 0)
        total_b = turb + conv
        
        # Simple ASCII intensity bar
        bar_len = int((total_b / max_events_in_bucket) * 20)
        bar = "#" * bar_len

        time_str = f"{format_timestamp(b_start)} - {format_timestamp(b_end)}"
        print(f"{time_str:<15} | {turb:<12} | {conv:<12} | {bar}")

    print("="*60 + "\n")

    # Export CSV summary
    csv_path = json_path.replace(".json", "_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("start_sec,end_sec,timestamp_formatted,turbulence,convergence,total\n")
        for b_start in sorted_buckets:
            b_end = b_start + bucket_sec
            turb = buckets[b_start].get("turbulence", 0)
            conv = buckets[b_start].get("convergence", 0)
            t_fmt = f"{format_timestamp(b_start)}-{format_timestamp(b_end)}"
            f.write(f"{b_start},{b_end},{t_fmt},{turb},{conv},{turb+conv}\n")

    print(f"Readable CSV summary exported to: {csv_path}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/logs/DJd5F3G9Qbg_optical_flow.json"
    summarize(path)
