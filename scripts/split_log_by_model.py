"""
Splits a combined detection JSON (multiple models, one run) into
separate per-model JSON files, plus prints a summary count per model.

Usage:
    python scripts/split_log_by_model.py outputs/logs/DJd5F3G9Qbg_all_models_dense_crowd.json
"""

import json
import sys
import os
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/split_log_by_model.py <path_to_combined_log.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path) as f:
        detections = json.load(f)

    by_model = defaultdict(list)
    for d in detections:
        by_model[d["model_name"]].append(d)

    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    print(f"Total detections: {len(detections)}\n")
    for model_name, dets in by_model.items():
        out_path = os.path.join(base_dir, f"{base_name}__{model_name}.json")
        with open(out_path, "w") as f:
            json.dump(dets, f, indent=2)
        print(f"  {model_name}: {len(dets)} detections -> {out_path}")


if __name__ == "__main__":
    main()
