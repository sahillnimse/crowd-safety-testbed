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
7.  Auto-exports summary.json, summary.txt, and report.html into the run folder!
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Config ------------------------------------------------------------------
DEFAULT_JSON = (
    r"outputs\runs\Foregin Crowd\crowd_motion_monitor\detections.json"
)
FLIP_THRESHOLD    = 5     # flag a track if it changes direction > N times
BOUNDARY_MARGIN   = 15.0  # degrees: "near +-90 deg" = within this of the boundary
RIGHT_THRESH      = 90.0  # must match _HEADING_RIGHT_THRESH in the model
HIST_BINS         = 18    # 20-deg bins across 360 deg
BAR_WIDTH         = 36    # chars for histogram bar

# Reconfigure stdout to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---- ANSI Color codes ---------------------------------------------------------
USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"

if USE_COLOR:
    C_RESET   = "\033[0m"
    C_BOLD    = "\033[1m"
    C_CYAN    = "\033[36m"
    C_GREEN   = "\033[32m"
    C_YELLOW  = "\033[33m"
    C_RED     = "\033[31m"
    C_MAGENTA = "\033[35m"
    C_BLUE    = "\033[34m"
    C_DIM     = "\033[2m"
else:
    C_RESET = C_BOLD = C_CYAN = C_GREEN = C_YELLOW = C_RED = C_MAGENTA = C_BLUE = C_DIM = ""


def make_block_bar(pct: float, width: int = 30, color: str = "") -> str:
    """Create a high-resolution Unicode block bar."""
    filled_len = int(round(pct / 100 * width))
    filled_len = max(0, min(width, filled_len))
    empty_len = width - filled_len
    bar = "█" * filled_len + "░" * empty_len
    if color and USE_COLOR:
        return f"{color}{bar}{C_RESET}"
    return bar


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


def run_analysis(json_path: str) -> dict:
    print(f"\n{C_BOLD}{C_CYAN}{'═'*72}{C_RESET}")
    print(f"  {C_BOLD}CROWD DIRECTION & KINEMATIC ACCURACY ANALYSIS{C_RESET}")
    print(f"  {C_DIM}Source: {json_path}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'═'*72}{C_RESET}\n")

    dets = load_detections(json_path)
    total = len(dets)
    if total == 0:
        print(f"{C_RED}No detections found.{C_RESET}")
        return {}

    print(f"  {C_BOLD}Total detections loaded :{C_RESET} {C_GREEN}{total:,}{C_RESET}\n")

    # -- 1. Label distribution -------------------------------------------------
    label_counts: Counter = Counter()
    for d in dets:
        label_counts[d.get("label", "unknown")] += 1

    print(f"{C_BOLD}{C_BLUE}── 1. Label Distribution ──────────────────────────────────────────{C_RESET}")
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        color = C_GREEN if "right" in label else (C_CYAN if "left" in label else (C_YELLOW if "crush" in label else C_RED))
        bar = make_block_bar(pct, width=28, color=color)
        print(f"  {label:<25}  {cnt:>7,}  ({pct:5.1f}%)  {bar}")
    print()

    # -- 2. Heading angle histogram --------------------------------------------
    print(f"{C_BOLD}{C_BLUE}── 2. Heading Angle Histogram (20-deg bins, -180° to +180°) ──────{C_RESET}")
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
    heading_hist_data = []
    for i, cnt in enumerate(bin_counts):
        lo = -180 + i * bin_size
        hi = lo + bin_size
        pct_rel = (cnt / max_bin * 100)
        is_left = (lo < -RIGHT_THRESH or hi > RIGHT_THRESH)
        color = C_CYAN if is_left else C_GREEN
        direction_tag = f"{C_CYAN}← LEFT{C_RESET}" if is_left else f"{C_GREEN}→ RIGHT{C_RESET}"
        bar = make_block_bar(pct_rel, width=BAR_WIDTH, color=color)
        print(f"  [{lo:+6.0f}°, {hi:+6.0f}°)  {cnt:>6,}  {bar}  {direction_tag}")
        heading_hist_data.append({
            "range": [round(lo, 1), round(hi, 1)],
            "count": cnt,
            "direction": "left" if is_left else "right",
        })
    print()

    right_count = left_count = 0
    if heading_vals:
        right_count = sum(1 for h in heading_vals if abs(h) < RIGHT_THRESH)
        left_count  = len(heading_vals) - right_count
        r_pct = right_count / len(heading_vals) * 100
        l_pct = left_count / len(heading_vals) * 100
        print(f"  {C_BOLD}Rightward detections :{C_RESET} {right_count:>7,}  ({r_pct:5.1f}%)  {make_block_bar(r_pct, 20, C_GREEN)}")
        print(f"  {C_BOLD}Leftward  detections :{C_RESET} {left_count:>7,}  ({l_pct:5.1f}%)  {make_block_bar(l_pct, 20, C_CYAN)}")
    print()

    # -- 3. Per-track direction consistency ------------------------------------
    print(f"{C_BOLD}{C_BLUE}── 3. Per-Track Direction Consistency ─────────────────────────────{C_RESET}")
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

    print(f"  Total confirmed tracks : {C_BOLD}{total_tracks:,}{C_RESET}")
    print(f"  Zero-flip (stable)     : {C_GREEN}{stable:,}{C_RESET}  ({stable/total_tracks*100:.1f}%)")
    print(f"  High-flip (>{FLIP_THRESHOLD})       : {C_YELLOW if unstable else C_GREEN}{unstable:,}{C_RESET}  ({unstable/total_tracks*100:.1f}%)")
    print(f"  Average flips / track  : {avg_flips:.2f}")
    print()

    suspicious_list = []
    if unstable:
        print(f"  {C_YELLOW}Suspicious tracks (>{FLIP_THRESHOLD} direction flips) [top 10]:{C_RESET}")
        worst = sorted(flip_data, key=lambda x: -x[1])[:10]
        for tid, f, r, l in worst:
            if f > FLIP_THRESHOLD:
                dominant = "right" if r >= l else "left"
                pct_dom  = max(r, l) / (r + l) * 100 if (r + l) else 0
                dom_color = C_GREEN if dominant == "right" else C_CYAN
                print(f"    Track {tid:>5}: {C_RED}{f:>2} flips{C_RESET}  "
                      f"(right={r:>2}, left={l:>2}, dominant={dom_color}{dominant}{C_RESET} {pct_dom:.0f}%)")
                suspicious_list.append({
                    "track_id": tid,
                    "flips": f,
                    "right_count": r,
                    "left_count": l,
                    "dominant_direction": dominant,
                    "dominant_pct": round(pct_dom),
                })
        print()

    # -- 4. Collision-zone boundary co-occurrence ------------------------------
    print(f"{C_BOLD}{C_BLUE}── 4. Collision-Zone Near-Boundary Analysis ───────────────────────{C_RESET}")
    crush_dets = [d for d in dets if d.get("label") == "person_crush_zone"]
    boundary_pct = 0.0
    if crush_dets:
        near_boundary = sum(
            1 for d in crush_dets
            if is_near_boundary(d.get("extra", {}).get("heading_deg", 0.0))
        )
        boundary_pct = near_boundary / len(crush_dets) * 100
        print(f"  Crush-zone detections          : {len(crush_dets):,}")
        print(f"  Near direction boundary (±{BOUNDARY_MARGIN:.0f}°): {near_boundary:,}  ({boundary_pct:.1f}%)")
        print(f"  => {C_YELLOW}{boundary_pct:.0f}% of collision zones are where right/left crowds meet{C_RESET}")
    else:
        print("  No crush-zone detections found.")
    print()

    # -- 5. Speed vs. direction cross-tab --------------------------------------
    print(f"{C_BOLD}{C_BLUE}── 5. Speed vs. Direction Cross-Tab ──────────────────────────────{C_RESET}")
    speed_dir: dict[str, list[float]] = defaultdict(list)
    for d in dets:
        extra = d.get("extra", {})
        spd = extra.get("speed_px_frame")
        lbl = d.get("label", "")
        if spd is not None and lbl:
            speed_dir[lbl].append(spd)

    speed_stats = {}
    for lbl, speeds in sorted(speed_dir.items()):
        avg = sum(speeds) / len(speeds)
        mx  = max(speeds)
        speed_stats[lbl] = {
            "avg_px_frame": round(avg, 2),
            "max_px_frame": round(mx, 2),
            "count": len(speeds),
        }
        print(f"  {lbl:<25}  avg={avg:5.2f} px/fr  max={mx:5.2f}  n={len(speeds):,}")
    print()

    # -- Summary verdict -------------------------------------------------------
    print(f"{C_BOLD}{C_BLUE}── Summary / Accuracy Verdict ────────────────────────────────────{C_RESET}")
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

    old_labels = {"person_moving"}
    found_old = [l for l in label_counts if l in old_labels]
    if found_old:
        issues.append(
            f"Old label(s) found: {found_old}. "
            "This JSON was from a previous run -- re-process the video."
        )

    if issues:
        print(f"  {C_YELLOW}ISSUES / NOTES:{C_RESET}")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print(f"  {C_GREEN}OK -- Direction detection looks healthy, no major issues found.{C_RESET}")

    # Build summary dict for file export
    n_stopped = label_counts.get("person_stopped", 0)
    n_crush = label_counts.get("person_crush_zone", 0)
    n_right = label_counts.get("person_moving_right", 0)
    n_left = label_counts.get("person_moving_left", 0)
    n_moving = n_right + n_left + n_crush

    # Calculate crush events and peak from detections
    frame_crush = defaultdict(int)
    frame_t = {}
    for d in dets:
        f_idx = d.get("frame_index")
        t_sec = d.get("timestamp_sec", 0.0)
        lbl = d.get("label")
        if lbl == "person_crush_zone" and f_idx is not None:
            frame_crush[f_idx] += 1
            frame_t[f_idx] = t_sec

    crush_events = 0
    in_ev = False
    peak_cnt = 0
    peak_t = 0.0
    for f_idx in sorted(frame_crush.keys()):
        cnt = frame_crush[f_idx]
        if cnt > peak_cnt:
            peak_cnt = cnt
            peak_t = frame_t.get(f_idx, 0.0)
        if cnt >= 3:
            if not in_ev:
                crush_events += 1
                in_ev = True
        else:
            in_ev = False

    summary_data = {
        "total_detections": total,
        "total_tracks": total_tracks,
        "pct_moving": round(n_moving / total * 100, 1) if total else 0.0,
        "pct_stationary": round(n_stopped / total * 100, 1) if total else 0.0,
        "pct_crush_risk": round(n_crush / total * 100, 1) if total else 0.0,
        "pct_moving_right": round(n_right / total * 100, 1) if total else 0.0,
        "pct_moving_left": round(n_left / total * 100, 1) if total else 0.0,
        "pct_heading_right": round(right_count / len(heading_vals) * 100, 1) if heading_vals else 0.0,
        "pct_heading_left": round(left_count / len(heading_vals) * 100, 1) if heading_vals else 0.0,
        "crush_event_count": crush_events,
        "peak_crush_timestamp_sec": round(peak_t, 2),
        "peak_crush_people_count": peak_cnt,
        "boundary_crush_pct": round(boundary_pct, 1),
        "label_counts": dict(label_counts),
        "avg_speed_px_frame": round(sum([s.get("avg_px_frame", 0) for s in speed_stats.values()]) / max(1, len(speed_stats)), 2),
        "speed_by_label": speed_stats,
        "stable_tracks_count": stable,
        "stable_tracks_pct": round(stable / total_tracks * 100, 1) if total_tracks else 0.0,
        "unstable_tracks_count": unstable,
        "avg_flips_per_track": round(avg_flips, 2),
        "suspicious_tracks": suspicious_list,
        "heading_histogram": heading_hist_data,
    }

    # Save summary.json, summary.txt, and report.html in the same directory as detections.json
    out_dir = os.path.dirname(json_path)
    summary_json_path = os.path.join(out_dir, "summary.json")
    summary_txt_path = os.path.join(out_dir, "summary.txt")
    report_html_path = os.path.join(out_dir, "report.html")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Crowd Safety Analysis Summary\n")
        f.write(f"Source: {json_path}\n")
        f.write(f"Total Detections: {total:,}\n")
        f.write(f"Total Tracks: {total_tracks:,} ({summary_data['stable_tracks_pct']}% stable)\n")
        f.write(f"Moving Left: {summary_data['pct_moving_left']}%\n")
        f.write(f"Moving Right: {summary_data['pct_moving_right']}%\n")
        f.write(f"Crush Risk: {summary_data['pct_crush_risk']}% ({crush_events} events, peak @ {peak_t:.1f}s)\n")
        f.write(f"Stationary: {summary_data['pct_stationary']}%\n")
        if suspicious_list:
            f.write("\nSuspicious Tracks:\n")
            for st in suspicious_list:
                f.write(f"  Track {st['track_id']}: {st['flips']} flips (dominant {st['dominant_direction']} {st['dominant_pct']}%)\n")

    try:
        from pipeline.html_report import export_html_report
        video_name = os.path.basename(os.path.dirname(out_dir))
        model_name = os.path.basename(out_dir)
        export_html_report(report_html_path, video_name, model_name, summary_data, dets)
    except Exception as e:
        print(f"[WARN] Failed to export HTML report: {e}")

    print(f"\n{C_BOLD}{C_GREEN}✓ Exported summary artifacts:{C_RESET}")
    print(f"  📄 {summary_json_path}")
    print(f"  📝 {summary_txt_path}")
    print(f"  📊 {report_html_path}")

    print(f"\n{C_BOLD}{C_CYAN}{'═'*72}{C_RESET}\n")
    return summary_data


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
