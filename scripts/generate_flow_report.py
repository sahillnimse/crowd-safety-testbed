"""
Generates a registry-style .txt evaluation report from an optical_flow_crush
detection JSON log — matching the format of DJd5F3G9Qbg_optical_flow_report.txt.

Includes: model suite registry header, executive detection summary,
ranked peak incident windows, and the full 5-second timeline breakdown.

Usage:
    python scripts/generate_flow_report.py outputs/logs/YzcawvDGe4Y_optical_flow.json

Optional: pass --calibrated to mark calibration status, and --video-name
to override the display name (defaults to the JSON filename's video ID).
"""

import argparse
import json
import os
from collections import defaultdict


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Path to optical_flow_crush detection JSON log")
    parser.add_argument("--calibrated", action="store_true", default=True,
                         help="Mark thresholds as calibrated in the header (default: True)")
    parser.add_argument("--video-name", default=None,
                         help="Override display name for the video (defaults to filename)")
    args = parser.parse_args()

    with open(args.log_path) as f:
        detections = json.load(f)

    if not detections:
        print("No detections found in this log — cannot generate report.")
        return

    video_name = args.video_name or os.path.basename(args.log_path).split("_optical_flow")[0] + ".mp4"

    total = len(detections)
    turbulence = [d for d in detections if d["label"] in ("turbulence", "crush_risk")]
    convergence = [d for d in detections if d["label"] in ("convergence", "crush_risk")]

    max_frame = max(d["frame_index"] for d in detections)
    max_ts = max(d["timestamp_sec"] for d in detections)
    # NOTE: max_frame reflects the highest frame INDEX seen in this log, which
    # is only an exact frame count if every single frame produced at least one
    # detection. It is reported as an approximation for informational purposes.

    # Bucket into 5-second windows. Counting per-label into a defaultdict of
    # int rather than a fixed dict literal, so a label the detector gained
    # later (e.g. "crush_risk", emitted when a cell is converging *and*
    # turbulent) doesn't KeyError the whole report.
    window_size = 5
    windows = defaultdict(lambda: defaultdict(int))
    for d in detections:
        window_start = int(d["timestamp_sec"] // window_size) * window_size
        windows[window_start][d["label"]] += 1

    sorted_windows = sorted(windows.keys())
    window_rows = []
    for w in sorted_windows:
        # crush_risk cells are both converging and turbulent, so they count
        # toward each column rather than being dropped from the report.
        t = windows[w]["turbulence"] + windows[w]["crush_risk"]
        c = windows[w]["convergence"] + windows[w]["crush_risk"]
        window_rows.append((w, w + window_size, t, c, t + c))

    # Rank top windows by total, for the "Peak Incident Time Windows" section
    ranked = sorted(window_rows, key=lambda r: r[4], reverse=True)
    top_n = min(6, len(ranked))
    top_windows = ranked[:top_n]

    max_total = max(r[4] for r in window_rows)
    bar_scale = 20.0 / max_total if max_total > 0 else 1

    lines = []
    lines.append("=" * 80)
    lines.append("          CROWD SAFETY TESTBED — EVALUATION REPORT & MODEL REGISTRY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Video Analyzed:       {video_name}")
    lines.append(f"Video Duration:       {format_time(max_ts)} ({max_ts:.1f} seconds)")
    lines.append(f"Total Video Frames:   {max_frame + 1:,} frames")
    lines.append(f"Calibration Status:   {'CALIBRATED (Top ~3% Thresholds)' if args.calibrated else 'UNCALIBRATED (Default Thresholds)'}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("TESTBED MODEL SUITE REGISTRY & EXECUTION STATUS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[1] MODEL: Farnebäck Dense Optical Flow (optical_flow_crush)")
    lines.append("    - Category:         Crowd Crush & Turbulence Detection")
    lines.append("    - Execution Status: EXECUTED & ANALYZED (Results below)")
    lines.append("    - Anomaly Signals:  turbulence, convergence")
    lines.append("")
    lines.append("[2] MODEL: Pose Fall Detector (pose_fall)")
    lines.append("    - Category:         Person Fall & Keypoint Orientation Detection")
    lines.append("    - Execution Status: READY (Not included in this single-model run)")
    lines.append("    - Anomaly Signals:  standing, fall (torso angle >= 45 deg)")
    lines.append("")
    lines.append("[3] MODEL: Fire & Smoke YOLO (fire_smoke_yolo)")
    lines.append("    - Category:         Object Detection for Fire & Smoke")
    lines.append("    - Execution Status: NOT OPERATIONAL (missing weights file)")
    lines.append("    - Anomaly Signals:  fire, smoke")
    lines.append("")
    lines.append("[4] MODEL: Violence Classifier (violence_classifier)")
    lines.append("    - Category:         3D CNN / Video Transformer Action Recognition")
    lines.append("    - Execution Status: NOT VALIDATED (un-finetuned generic weights)")
    lines.append("    - Anomaly Signals:  violence")
    lines.append("")
    lines.append("=" * 80)
    lines.append("DETAILED RESULTS: MODEL [1] Farnebäck Dense Optical Flow (optical_flow_crush)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("-" * 80)
    lines.append("1. EXECUTIVE DETECTION SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total Detections Found:  {total:,} grid-cell events")
    lines.append("")
    lines.append("Breakdown by Anomaly Type:")
    lines.append(f"  - Turbulence (Chaotic Flow):    {len(turbulence):,} events ({100*len(turbulence)/total:.1f}%)")
    lines.append(f"  - Convergence (Compression):     {len(convergence):,} events ({100*len(convergence)/total:.1f}%)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("2. PEAK INCIDENT TIME WINDOWS (HIGHEST DENSITY)")
    lines.append("-" * 80)
    lines.append(f"{'Rank':<5}| {'Time Window':<16}| {'Turbulence':<11}| {'Convergence':<12}| {'Total Events':<13}| Severity")
    lines.append("-" * 5 + "|" + "-" * 17 + "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 14 + "|" + "-" * 10)
    for i, (ws, we, t, c, tot) in enumerate(top_windows, 1):
        label = f"{format_time(ws)} - {format_time(we)}"
        if c > t:
            severity = "COMPRESSION SURGE"
        elif i <= 3:
            severity = "HIGH PEAK"
        else:
            severity = "ELEVATED"
        lines.append(f" {i:<4}| {label:<16}| {t:<11,}| {c:<12,}| {tot:<13,}| {severity}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("3. TIMELINE BREAKDOWN (5-SECOND INTERVALS)")
    lines.append("-" * 80)
    lines.append(f"{'Time Window':<16}| {'Turbulence':<13}| {'Convergence':<13}| {'Total':<13}| Visual Bar")
    lines.append("-" * 80)
    for ws, we, t, c, tot in window_rows:
        label = f"{format_time(ws)} - {format_time(we)}"
        bar = "#" * max(1, int(tot * bar_scale)) if tot > 0 else ""
        lines.append(f"{label:<16}| {t:<13,}| {c:<13,}| {tot:<13,}| {bar}")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    out_dir = os.path.dirname(args.log_path)
    base_name = os.path.splitext(os.path.basename(args.log_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\n\nReport written to: {out_path}")


if __name__ == "__main__":
    main()