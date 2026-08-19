"""Summarise CrowdMotionMonitor stream labels from a detection log.

The monitor emits direction-neutral ``moving``, ``stream_a``, and ``stream_b``
values. Stream A/B are per-run heading clusters, never screen-left/right.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_JSON = Path("outputs/runs/Foregin Crowd/crowd_motion_monitor/detections.json")
FLIP_THRESHOLD = 5


def load_detections(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data["detections"]


def run_analysis(json_path: Path) -> dict:
    detections = load_detections(json_path)
    total = len(detections)
    labels = Counter(item.get("label", "unknown") for item in detections)
    directions: dict[int, list[str]] = defaultdict(list)
    for item in detections:
        extra = item.get("extra", {})
        direction = extra.get("crowd_direction")
        track_id = extra.get("track_id")
        if track_id is not None and direction in {"moving", "stream_a", "stream_b"}:
            directions[track_id].append(direction)

    print(f"Crowd-motion stream analysis: {json_path}")
    print(f"Total detections: {total:,}")
    for label, count in labels.most_common():
        print(f"  {label:<24} {count:>7,} ({count / total * 100 if total else 0:5.1f}%)")

    stream_counts = {
        "moving": labels["person_moving"],
        "stream_a": labels["person_moving_stream_a"],
        "stream_b": labels["person_moving_stream_b"],
    }
    has_streams = bool(stream_counts["stream_a"] or stream_counts["stream_b"])
    if has_streams:
        print("Direction modes: Stream A / Stream B (per-run heading clusters)")
    else:
        print("Direction modes: one moving stream; no artificial split")

    flip_data = []
    for track_id, values in directions.items():
        flips = sum(a != b for a, b in zip(values, values[1:]))
        counts = Counter(values)
        flip_data.append((track_id, flips, counts))
    suspicious = []
    for track_id, flips, counts in sorted(flip_data, key=lambda row: -row[1])[:10]:
        if flips > FLIP_THRESHOLD:
            dominant, dominant_count = counts.most_common(1)[0]
            suspicious.append({
                "track_id": track_id,
                "flips": flips,
                "moving_count": counts["moving"],
                "stream_a_count": counts["stream_a"],
                "stream_b_count": counts["stream_b"],
                "dominant_direction": dominant,
                "dominant_pct": round(dominant_count / sum(counts.values()) * 100),
            })

    stopped = labels["person_stopped"]
    crush = labels["person_crush_zone"]
    moving = sum(stream_counts.values())
    summary = {
        "total_detections": total,
        "total_tracks": len(flip_data),
        "pct_moving": round((moving + crush) / total * 100, 1) if total else 0.0,
        "pct_stationary": round(stopped / total * 100, 1) if total else 0.0,
        "pct_crush_risk": round(crush / total * 100, 1) if total else 0.0,
        "pct_moving_single_stream": round(stream_counts["moving"] / total * 100, 1) if total else 0.0,
        "pct_moving_stream_a": round(stream_counts["stream_a"] / total * 100, 1) if total else 0.0,
        "pct_moving_stream_b": round(stream_counts["stream_b"] / total * 100, 1) if total else 0.0,
        "label_counts": dict(labels),
        "unstable_tracks_count": len(suspicious),
        "suspicious_tracks": suspicious,
    }
    out_dir = json_path.parent
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(
        "Crowd Motion Stream Analysis\n"
        f"Source: {json_path}\n"
        f"Total detections: {total:,}\n"
        f"Moving: {summary['pct_moving']:.1f}%\n"
        f"Single stream: {summary['pct_moving_single_stream']:.1f}%\n"
        f"Stream A: {summary['pct_moving_stream_a']:.1f}%\n"
        f"Stream B: {summary['pct_moving_stream_b']:.1f}%\n",
        encoding="utf-8",
    )
    from pipeline.html_report import generate_report_html
    report = generate_report_html(json_path.parent.parent.name, "crowd_motion_monitor", summary, detections)
    (out_dir / "report.html").write_text(report, encoding="utf-8")
    print(f"High-flip tracks: {len(suspicious)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    run_analysis(args.json)


if __name__ == "__main__":
    main()
