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

from webapp.jobs import ANNOTATED_DIR, LOG_DIR, POSITIVE_LABELS
from webapp.registry import BY_KEY

# path -> (mtime, size, summary dict)
_CACHE: dict[str, tuple[float, int, dict]] = {}


def _split_name(stem: str) -> tuple[str, str] | None:
    """'pG4HqTGkHPg_fall_yolo_pose' -> ('pG4HqTGkHPg', 'fall_yolo_pose').

    Split against the known model keys rather than on the last underscore:
    model keys contain underscores themselves, and video IDs can too.
    """
    for key in sorted(BY_KEY, key=len, reverse=True):
        suffix = "_" + key
        if stem.endswith(suffix):
            return stem[: -len(suffix)], key
    return None


def _summarize(path: str) -> dict:
    """Parse one detection log into the counts the UI table needs."""
    stat = os.stat(path)
    cached = _CACHE.get(path)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        summary = {"detections": 0, "positives": 0, "label_counts": {},
                   "scoring_modes": {}, "error": f"unreadable log: {e}"}
        _CACHE[path] = (stat.st_mtime, stat.st_size, summary)
        return summary

    labels = Counter(r.get("label") for r in rows)
    scoring = Counter(
        r["extra"].get("scoring") for r in rows
        if isinstance(r.get("extra"), dict) and r["extra"].get("scoring")
    )
    summary = {
        "detections": len(rows),
        "positives": sum(n for lbl, n in labels.items() if lbl in POSITIVE_LABELS),
        "label_counts": dict(labels),
        "scoring_modes": dict(scoring),
        "error": None,
    }
    _CACHE[path] = (stat.st_mtime, stat.st_size, summary)
    return summary


def scan() -> list[dict]:
    """All completed runs on disk, grouped by video, newest video first."""
    if not os.path.isdir(LOG_DIR):
        return []

    by_video: dict[str, list] = defaultdict(list)
    newest: dict[str, float] = {}

    for name in os.listdir(LOG_DIR):
        if not name.endswith(".json"):
            continue
        parsed = _split_name(name[:-len(".json")])
        if parsed is None:
            continue  # not one of ours; leave it alone
        video, model_key = parsed

        json_path = os.path.join(LOG_DIR, name)
        summary = _summarize(json_path)
        mtime = os.path.getmtime(json_path)

        csv_name = f"{video}_{model_key}.csv"
        mp4_name = f"{video}_{model_key}.mp4"
        spec = BY_KEY.get(model_key)

        by_video[video].append({
            "model_key": model_key,
            "model_label": spec.label if spec else model_key,
            "status": "done",
            "progress": 1.0,
            "detections": summary["detections"],
            "positives": summary["positives"],
            "label_counts": summary["label_counts"],
            "scoring_modes": summary["scoring_modes"],
            "error": summary["error"],
            "elapsed_sec": None,          # not recorded on disk
            "modified_at": mtime,
            "log_json": name,
            "log_csv": csv_name if os.path.exists(os.path.join(LOG_DIR, csv_name)) else None,
            "annotated": mp4_name if os.path.exists(os.path.join(ANNOTATED_DIR, mp4_name)) else None,
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
