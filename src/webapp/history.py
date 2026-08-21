"""
Reconstructs past runs from the files in outputs/.

The job registry in webapp/jobs.py is in-memory, so restarting the server
loses every run — and runs launched from scripts/run_single.py never
appeared in the UI at all. Both are surprising: the results are sitting
right there on disk.

This module treats `outputs/logs/` as the durable record. Every
`<video>_<model>.json` is a completed run, whoever produced it, so the UI
can show history that survives restarts and includes CLI work.

Summaries are cached on (mtime, size) because the detection logs run to
megabytes and the frontend polls; re-parsing them every poll would be
pointless work.
"""

import json
import os
from collections import Counter, defaultdict

from webapp.jobs import (POSITIVE_LABELS, RUNS_DIR, RUN_CSV,
                         RUN_JSON, RUN_REPORT, RUN_SUMMARY, RUN_VIDEO)
from webapp.registry import BY_KEY

# path -> (mtime, size, summary dict)
_CACHE: dict[str, tuple[float, int, dict]] = {}


def compute_detections_summary(rows: list) -> dict:
    """Compute comprehensive run-level aggregate stats from Detection dicts or objects."""
    total = len(rows)
    if total == 0:
        return {
            "total_detections": 0,
            "total_tracks": 0,
            "pct_moving": 0.0,
            "pct_stationary": 0.0,
            "pct_crush_risk": 0.0,
            "pct_moving_single_stream": 0.0,
            "pct_moving_stream_a": 0.0,
            "pct_moving_stream_b": 0.0,
            "pct_heading_right": 0.0,
            "pct_heading_left": 0.0,
            "crush_event_count": 0,
            "peak_crush_timestamp_sec": 0.0,
            "peak_crush_people_count": 0,
            "boundary_crush_pct": 0.0,
            "label_counts": {},
            "avg_speed_px_frame": 0.0,
            "speed_by_label": {},
            "stable_tracks_count": 0,
            "stable_tracks_pct": 0.0,
            "unstable_tracks_count": 0,
            "avg_flips_per_track": 0.0,
            "suspicious_tracks": [],
            "heading_histogram": [],
        }

    label_counts = Counter()
    track_dirs = defaultdict(list)
    frame_crush = defaultdict(int)
    frame_cf = defaultdict(int)
    frame_timestamps = {}
    speed_records = defaultdict(list)
    var_records = []
    entropy_records = []
    heading_bins = [0] * 18
    heading_r = 0
    heading_l = 0
    boundary_crush = 0
    counterflow_count = 0

    for r in rows:
        lbl = getattr(r, "label", None) or (r.get("label") if isinstance(r, dict) else "")
        f_idx = getattr(r, "frame_index", None) if not isinstance(r, dict) else r.get("frame_index")
        t_sec = getattr(r, "timestamp_sec", None) if not isinstance(r, dict) else r.get("timestamp_sec")
        extra = getattr(r, "extra", None) if not isinstance(r, dict) else r.get("extra")
        if not isinstance(extra, dict):
            extra = {}

        label_counts[lbl] += 1
        tid = extra.get("track_id")
        cdir = extra.get("crowd_direction")
        spd = extra.get("speed_px_frame")
        hdeg = extra.get("heading_deg")
        is_crush = (lbl == "person_crush_zone" or extra.get("local_crush_risk"))
        is_cf = extra.get("is_counterflow", False)
        l_var = extra.get("local_velocity_variance")
        l_ent = extra.get("local_directional_entropy")

        if tid is not None and cdir is not None:
            track_dirs[tid].append(cdir)
        if spd is not None:
            speed_records[lbl].append(spd)
        if l_var is not None:
            var_records.append(l_var)
        if l_ent is not None:
            entropy_records.append(l_ent)
        if is_cf:
            counterflow_count += 1
            if f_idx is not None:
                frame_cf[f_idx] += 1

        if hdeg is not None:
            if abs(hdeg) < 90.0:
                heading_r += 1
            else:
                heading_l += 1
            b_idx = int((hdeg + 180.0) / 20.0) % 18
            heading_bins[b_idx] += 1
            if is_crush and abs(abs(hdeg) - 90.0) < 15.0:
                boundary_crush += 1

        if is_crush and f_idx is not None:
            frame_crush[f_idx] += 1
        if f_idx is not None and t_sec is not None:
            frame_timestamps[f_idx] = t_sec

    n_stopped = label_counts.get("person_stopped", 0)
    n_crush = label_counts.get("person_crush_zone", 0)
    n_moving_single = label_counts.get("person_moving", 0)
    n_moving_stream_a = label_counts.get("person_moving_stream_a", 0)
    n_moving_stream_b = label_counts.get("person_moving_stream_b", 0)
    n_moving = (n_moving_single + n_moving_stream_a + n_moving_stream_b + n_crush
                if (n_moving_single or n_moving_stream_a or n_moving_stream_b or n_crush)
                else (total - n_stopped))

    pct_moving = round((n_moving / total * 100), 1) if total else 0.0
    pct_stationary = round((n_stopped / total * 100), 1) if total else 0.0
    pct_crush_risk = round((n_crush / total * 100), 1) if total else 0.0
    pct_moving_single = round((n_moving_single / total * 100), 1) if total else 0.0
    pct_moving_stream_a = round((n_moving_stream_a / total * 100), 1) if total else 0.0
    pct_moving_stream_b = round((n_moving_stream_b / total * 100), 1) if total else 0.0

    h_total = heading_r + heading_l
    pct_heading_right = round((heading_r / h_total * 100), 1) if h_total else 0.0
    pct_heading_left = round((heading_l / h_total * 100), 1) if h_total else 0.0

    # Crush events: frames where count >= 3
    crush_events = 0
    in_event = False
    peak_crush_count = 0
    peak_crush_t = 0.0

    for f_idx in sorted(frame_crush.keys()):
        cnt = frame_crush[f_idx]
        if cnt > peak_crush_count:
            peak_crush_count = cnt
            peak_crush_t = frame_timestamps.get(f_idx, 0.0)
        if cnt >= 3:
            if not in_event:
                crush_events += 1
                in_event = True
        else:
            in_event = False

    # Counterflow events
    cf_events = 0
    in_cf_event = False
    peak_cf_count = 0
    peak_cf_t = 0.0

    for f_idx in sorted(frame_cf.keys()):
        cnt = frame_cf[f_idx]
        if cnt > peak_cf_count:
            peak_cf_count = cnt
            peak_cf_t = frame_timestamps.get(f_idx, 0.0)
        if cnt >= 2:
            if not in_cf_event:
                cf_events += 1
                in_cf_event = True
        else:
            in_cf_event = False

    pct_cf = round((counterflow_count / total * 100), 1) if total else 0.0
    avg_var = round(sum(var_records) / len(var_records), 3) if var_records else 0.0
    peak_var = round(max(var_records), 3) if var_records else 0.0
    avg_entropy = round(sum(entropy_records) / len(entropy_records), 3) if entropy_records else 0.0

    flip_data = []
    for tid, dirs in track_dirs.items():
        flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
        r_cnt = dirs.count("right")
        l_cnt = dirs.count("left")
        flip_data.append((tid, flips, r_cnt, l_cnt))

    total_tracks = len(flip_data)
    stable_tracks = sum(1 for _, f, _, _ in flip_data if f == 0)
    unstable_tracks = [x for x in flip_data if x[1] > 5]
    avg_flips = (sum(f for _, f, _, _ in flip_data) / total_tracks) if total_tracks else 0.0
    pct_stable = round((stable_tracks / total_tracks * 100), 1) if total_tracks else 0.0

    worst_tracks = sorted(flip_data, key=lambda x: -x[1])[:10]
    suspicious_list = []
    for tid, f, r, l in worst_tracks:
        if f > 5:
            dom = "right" if r >= l else "left"
            pct_dom = round(max(r, l) / (r + l) * 100) if (r + l) else 0
            suspicious_list.append({
                "track_id": tid,
                "flips": f,
                "right_count": r,
                "left_count": l,
                "dominant_direction": dom,
                "dominant_pct": pct_dom,
            })

    speed_stats = {}
    all_speeds = []
    for lbl, spds in speed_records.items():
        if spds:
            speed_stats[lbl] = {
                "avg_px_frame": round(sum(spds) / len(spds), 2),
                "max_px_frame": round(max(spds), 2),
                "count": len(spds),
            }
            all_speeds.extend(spds)
    overall_avg_spd = round(sum(all_speeds) / len(all_speeds), 2) if all_speeds else 0.0

    heading_hist_bins = []
    for i, cnt in enumerate(heading_bins):
        lo = -180.0 + i * 20.0
        hi = lo + 20.0
        direction = "left" if (lo < -90.0 or hi > 90.0) else "right"
        heading_hist_bins.append({
            "range": [round(lo, 1), round(hi, 1)],
            "count": cnt,
            "direction": direction,
        })

    return {
        "total_detections": total,
        "total_tracks": total_tracks,
        "pct_moving": pct_moving,
        "pct_stationary": pct_stationary,
        "pct_crush_risk": pct_crush_risk,
        "pct_moving_single_stream": pct_moving_single,
        "pct_moving_stream_a": pct_moving_stream_a,
        "pct_moving_stream_b": pct_moving_stream_b,
        "pct_heading_right": pct_heading_right,
        "pct_heading_left": pct_heading_left,
        "crush_event_count": crush_events,
        "peak_crush_timestamp_sec": round(peak_crush_t, 2),
        "peak_crush_people_count": peak_crush_count,
        "boundary_crush_pct": round((boundary_crush / n_crush * 100), 1) if n_crush > 0 else 0.0,
        "counterflow_events_count": cf_events,
        "pct_counterflow_people": pct_cf,
        "peak_counterflow_timestamp_sec": round(peak_cf_t, 2),
        "peak_counterflow_people_count": peak_cf_count,
        "avg_velocity_variance": avg_var,
        "peak_velocity_variance": peak_var,
        "avg_directional_entropy": avg_entropy,
        "label_counts": dict(label_counts),
        "avg_speed_px_frame": overall_avg_spd,
        "speed_by_label": speed_stats,
        "stable_tracks_count": stable_tracks,
        "stable_tracks_pct": pct_stable,
        "unstable_tracks_count": len(unstable_tracks),
        "avg_flips_per_track": round(avg_flips, 2),
        "suspicious_tracks": suspicious_list,
        "heading_histogram": heading_hist_bins,
    }


def _summarize(path: str) -> dict:
    """Parse one detection log into the counts the UI table needs."""
    stat = os.stat(path)
    cached = _CACHE.get(path)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

    # Check if summary.json already exists in the same folder
    folder = os.path.dirname(path)
    summary_path = os.path.join(folder, RUN_SUMMARY)
    stored_summary = None
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as sf:
                stored_summary = json.load(sf)
        except Exception:
            pass

    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        summary = {"detections": 0, "positives": 0, "label_counts": {},
                   "scoring_modes": {}, "summary": {}, "error": f"unreadable log: {e}"}
        _CACHE[path] = (stat.st_mtime, stat.st_size, summary)
        return summary

    labels = Counter(r.get("label") for r in rows)
    scoring = Counter(
        r["extra"].get("scoring") for r in rows
        if isinstance(r.get("extra"), dict) and r["extra"].get("scoring")
    )

    if stored_summary is None:
        stored_summary = compute_detections_summary(rows)
        # Also persist summary.json for fast reload
        try:
            with open(summary_path, "w", encoding="utf-8") as sf:
                json.dump(stored_summary, sf, indent=2)
        except Exception:
            pass

    # If report.html doesn't exist, create it too!
    report_path = os.path.join(folder, RUN_REPORT)
    if not os.path.isfile(report_path):
        try:
            from pipeline.html_report import export_html_report
            video_name = os.path.basename(os.path.dirname(folder))
            model_key = os.path.basename(folder)
            export_html_report(report_path, video_name, model_key, stored_summary, rows)
        except Exception:
            pass

    summary = {
        "detections": len(rows),
        "positives": sum(n for lbl, n in labels.items() if lbl in POSITIVE_LABELS),
        "label_counts": dict(labels),
        "scoring_modes": dict(scoring),
        "summary": stored_summary,
        "error": None,
    }
    _CACHE[path] = (stat.st_mtime, stat.st_size, summary)
    return summary


def scan() -> list[dict]:
    """
    All completed runs on disk, grouped by video, newest video first.

    Walks outputs/runs/<video>/<model>/ rather than parsing filenames.
    """
    if not os.path.isdir(RUNS_DIR):
        return []

    by_video: dict[str, list] = defaultdict(list)
    newest: dict[str, float] = {}

    for video in os.listdir(RUNS_DIR):
        video_dir = os.path.join(RUNS_DIR, video)
        if not os.path.isdir(video_dir):
            continue

        for model_key in os.listdir(video_dir):
            model_dir = os.path.join(video_dir, model_key)
            json_path = os.path.join(model_dir, RUN_JSON)
            if not os.path.isfile(json_path):
                continue  # no detections written; not a completed run

            summary = _summarize(json_path)
            mtime = os.path.getmtime(json_path)
            spec = BY_KEY.get(model_key)
            rel = f"{video}/{model_key}"

            by_video[video].append({
                "model_key": model_key,
                "model_label": spec.label if spec else model_key,
                "status": "done",
                "progress": 1.0,
                "detections": summary["detections"],
                "positives": summary["positives"],
                "label_counts": summary["label_counts"],
                "scoring_modes": summary["scoring_modes"],
                "summary": summary.get("summary", {}),
                "error": summary["error"],
                "elapsed_sec": None,          # not recorded on disk
                "modified_at": mtime,
                "log_json": f"{rel}/{RUN_JSON}",
                "log_csv": (f"{rel}/{RUN_CSV}"
                            if os.path.exists(os.path.join(model_dir, RUN_CSV))
                            else None),
                "log_summary": (f"{rel}/{RUN_SUMMARY}"
                               if os.path.exists(os.path.join(model_dir, RUN_SUMMARY))
                               else None),
                "report_html": (f"{rel}/{RUN_REPORT}"
                               if os.path.exists(os.path.join(model_dir, RUN_REPORT))
                               else None),
                "annotated": (f"{rel}/{RUN_VIDEO}"
                              if os.path.exists(os.path.join(model_dir, RUN_VIDEO))
                              else None),
            })
            newest[video] = max(newest.get(video, 0), mtime)

    out = []
    for video, stages in by_video.items():
        stages.sort(key=lambda s: s["model_key"])
        out.append({
            "video": video,
            "modified_at": newest[video],
            "stages": stages,
            "total_positives": sum(s["positives"] for s in stages),
        })
    out.sort(key=lambda g: g["modified_at"], reverse=True)
    return out
