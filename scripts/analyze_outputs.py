"""
analyze_outputs.py
Summarizes all JSON benchmark outputs in outputs/logs/ and prints quantitative comparisons.
"""

import json
from pathlib import Path
from collections import defaultdict

LOGS_DIR = Path("outputs/logs")

def analyze_all():
    json_files = list(LOGS_DIR.glob("*.json"))
    if not json_files:
        print("No JSON log files found in outputs/logs/")
        return

    # Group by benchmark video / category
    by_category = defaultdict(list)
    for jf in json_files:
        name_parts = jf.stem.split("_", 1)
        video_prefix = name_parts[0]
        model_name = name_parts[1] if len(name_parts) > 1 else jf.stem
        by_category[video_prefix].append((model_name, jf))

    for cat, files in by_category.items():
        print(f"\n========================================================")
        print(f" CATEGORY / VIDEO: {cat}")
        print(f"========================================================")

        results = []
        for model_name, path in files:
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Could not load {path.name}: {e}")
                    continue

            total_dets = len(data)
            if total_dets == 0:
                results.append({
                    "model": model_name, "total": 0, "tracks": 0, "avg_conf": 0.0, "labels": {}
                })
                continue

            unique_tracks = set()
            confs = []
            labels = defaultdict(int)

            for d in data:
                confs.append(d.get("confidence", 0.0))
                labels[d.get("label", "unknown")] += 1
                extra = d.get("extra", {})
                if "track_id" in extra:
                    unique_tracks.add(extra["track_id"])

            avg_conf = sum(confs) / len(confs) if confs else 0.0
            results.append({
                "model": model_name,
                "total": total_dets,
                "tracks": len(unique_tracks),
                "avg_conf": avg_conf,
                "labels": dict(labels)
            })

        # Print table
        print(f"{'Model':<30} | {'Detections':<10} | {'Unique Tracks':<14} | {'Avg Conf':<10} | {'Top Labels / Breakdown'}")
        print("-" * 90)
        for r in sorted(results, key=lambda x: x["total"], reverse=True):
            lbl_str = ", ".join(f"{k}:{v}" for k, v in list(r["labels"].items())[:3])
            print(f"{r['model']:<30} | {r['total']:<10} | {r['tracks']:<14} | {r['avg_conf']:<10.3f} | {lbl_str}")

if __name__ == "__main__":
    analyze_all()
