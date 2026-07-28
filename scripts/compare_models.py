"""
Side-by-side comparison of all models within a category (fall or violence)
after they've been run against the same video via run_all.py / run_single.py.

Reads a combined detection log (or a directory of per-model logs produced
by split_log_by_model.py) and prints a comparison table: detection counts,
positive-label rate, confidence distribution, and — if ground truth is
supplied via configs/test_videos.yaml — precision/recall/F1 per model.

This is the "which model gives the best output" step: run_all.py produces
detections, this script turns them into a decision.

Usage:
    # Combined log from a single run_all.py category (all models, one video):
    python scripts/compare_models.py outputs/logs/<video>_<category>.json --category fall

    # Or point at a directory of split-by-model logs:
    python scripts/compare_models.py outputs/logs/split_dir/ --category violence

    # With ground truth (list of {start_sec, end_sec} true-positive windows
    # for the target label, taken from configs/test_videos.yaml):
    python scripts/compare_models.py outputs/logs/<video>_<category>.json \\
        --category fall --ground_truth configs/test_videos.yaml --video_url "<url>"
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import yaml

POSITIVE_LABELS = {
    "fall": {"fall"},
    "violence": {"violence"},
}

# Reminder when reading the output: `confidence` describes the reported
# label, so threshold sweeps over it are meaningful. Check each detection's
# `extra.scoring` before comparing models head-to-head — a wrapper running
# its geometric fallback ("geometric_fallback") or zero-shot Kinetics
# scoring ("kinetics_zeroshot") is not the same model as one running a
# fine-tuned checkpoint, even though both appear as a row in this table.


def load_detections(path: str) -> list[dict]:
    """Accepts either a single combined-log JSON file or a directory of
    per-model JSON files (as produced by split_log_by_model.py)."""
    if os.path.isdir(path):
        all_dets = []
        for fp in glob.glob(os.path.join(path, "*.json")):
            with open(fp) as f:
                all_dets.extend(json.load(f))
        return all_dets
    with open(path) as f:
        return json.load(f)


def load_ground_truth(config_path: str, video_url: str, category: str) -> list[dict]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for entry in config.get("videos", []):
        if entry["url"] == video_url:
            return entry.get("ground_truth", [])
    print(f"[WARN] No config entry found for {video_url} in {config_path} — skipping ground truth.")
    return []


def overlaps_ground_truth(timestamp_sec: float, gt_windows: list[dict], tolerance_sec: float = 2.0) -> bool:
    for w in gt_windows:
        if w["start_sec"] - tolerance_sec <= timestamp_sec <= w["end_sec"] + tolerance_sec:
            return True
    return False


def compute_metrics(dets: list[dict], positive_labels: set, gt_windows: list[dict] | None):
    positives = [d for d in dets if d["label"] in positive_labels]
    total = len(dets)
    pos_rate = len(positives) / total if total else 0.0
    avg_conf = sum(d["confidence"] for d in positives) / len(positives) if positives else 0.0

    metrics = {
        "total_detections": total,
        "positive_detections": len(positives),
        "positive_rate": pos_rate,
        "avg_positive_confidence": avg_conf,
    }

    if gt_windows is not None:
        tp = sum(1 for d in positives if overlaps_ground_truth(d["timestamp_sec"], gt_windows))
        fp = len(positives) - tp
        # naive recall proxy: fraction of GT windows with >=1 matching detection
        matched_windows = sum(
            1 for w in gt_windows
            if any(w["start_sec"] - 2.0 <= d["timestamp_sec"] <= w["end_sec"] + 2.0 for d in positives)
        )
        recall = matched_windows / len(gt_windows) if gt_windows else None
        precision = tp / (tp + fp) if (tp + fp) else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

        metrics.update({
            "true_positives_approx": tp,
            "false_positives_approx": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return metrics


def print_comparison_table(results: dict, has_ground_truth: bool):
    print("\n" + "=" * 100)
    print(" MODEL COMPARISON")
    print("=" * 100)

    if has_ground_truth:
        header = f"{'Model':<28} | {'Total':>7} | {'Pos.':>6} | {'Pos.Rate':>8} | {'AvgConf':>7} | {'Prec':>6} | {'Recall':>6} | {'F1':>6}"
    else:
        header = f"{'Model':<28} | {'Total':>7} | {'Pos.':>6} | {'Pos.Rate':>8} | {'AvgConf':>7}"
    print(header)
    print("-" * len(header))

    # Sort by F1 if available, else by positive detection count, best first
    def sort_key(item):
        _, m = item
        if has_ground_truth and m.get("f1") is not None:
            return -m["f1"]
        return -m["positive_detections"]

    for model_name, m in sorted(results.items(), key=sort_key):
        row = f"{model_name:<28} | {m['total_detections']:>7} | {m['positive_detections']:>6} | " \
              f"{m['positive_rate']:>7.1%} | {m['avg_positive_confidence']:>7.3f}"
        if has_ground_truth:
            prec = f"{m['precision']:.2f}" if m.get("precision") is not None else "  n/a"
            rec = f"{m['recall']:.2f}" if m.get("recall") is not None else "  n/a"
            f1 = f"{m['f1']:.2f}" if m.get("f1") is not None else "  n/a"
            row += f" | {prec:>6} | {rec:>6} | {f1:>6}"
        print(row)

    print("=" * 100)
    if has_ground_truth:
        print("Best by F1 shown first. Precision/recall are approximate (timestamp-window")
        print("overlap with a fixed 2s tolerance) — treat as a ranking signal, not a final metric.")
    else:
        print("No ground truth supplied — ranked by raw positive-detection count only.")
        print("Add ground_truth windows to configs/test_videos.yaml for precision/recall/F1.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Combined detection log JSON, or directory of per-model logs")
    parser.add_argument("--category", required=True, choices=list(POSITIVE_LABELS.keys()))
    parser.add_argument("--ground_truth", help="Path to configs/test_videos.yaml")
    parser.add_argument("--video_url", help="Video URL key to look up in the ground truth config")
    args = parser.parse_args()

    dets = load_detections(args.log_path)
    if not dets:
        print("No detections found in log(s).")
        sys.exit(1)

    by_model = defaultdict(list)
    for d in dets:
        by_model[d["model_name"]].append(d)

    gt_windows = None
    if args.ground_truth and args.video_url:
        gt_windows = load_ground_truth(args.ground_truth, args.video_url, args.category)
        if not gt_windows:
            gt_windows = None  # fall back to no-ground-truth mode if empty/not found

    results = {
        model_name: compute_metrics(model_dets, POSITIVE_LABELS[args.category], gt_windows)
        for model_name, model_dets in by_model.items()
    }

    print_comparison_table(results, has_ground_truth=gt_windows is not None)

    # Export CSV alongside for spreadsheet review
    out_dir = os.path.dirname(args.log_path) if os.path.isfile(args.log_path) else args.log_path
    csv_path = os.path.join(out_dir, f"comparison_{args.category}.csv")
    with open(csv_path, "w") as f:
        cols = ["model_name"] + list(next(iter(results.values())).keys())
        f.write(",".join(cols) + "\n")
        for model_name, m in results.items():
            f.write(",".join([model_name] + [str(m.get(c, "")) for c in cols[1:]]) + "\n")
    print(f"\nCSV exported to: {csv_path}")


if __name__ == "__main__":
    main()
