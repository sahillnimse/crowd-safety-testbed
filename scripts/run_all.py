"""
Runs the full test_videos.yaml set through the appropriate models for
each video's category, and exports annotated videos + logs for all of them.

Usage:
    python scripts/run_all.py --config configs/test_videos.yaml

Models come from webapp/registry.py, the same table the web UI builds from.
This script used to keep its own MODEL_REGISTRY dict, and the two drifted:
the local copy covered 12 of the project's models and never gained traffic,
ANPR or umbrella, so `--only-model rtdetrv2_traffic` failed with a KeyError
that read like a broken model rather than a missing table entry.  It also
kept listing models after they were deleted.  One table, one source of truth.
"""

import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ingestion.youtube_fetch import fetch_youtube_video
from pipeline.runner import PipelineRunner
from pipeline.annotate import export_annotated_video, export_detection_log, export_detection_csv
from pipeline.device import resolve_device, require_gpu, print_gpu_report
from webapp import registry
from webapp.jobs import RUN_CSV, RUN_JSON, RUN_VIDEO, run_dir


def build_models(names: list[str], device: str, video_name: str = "run"):
    return [registry.build_model(n, device, video_name=video_name)
            for n in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_videos.yaml")
    parser.add_argument(
        "--device", default=None,
        help="cuda / cuda:0 / cpu. Default: auto-detect. "
             "Passing 'cuda' explicitly makes the run FAIL LOUDLY instead of "
             "silently falling back to CPU if no GPU is visible.",
    )
    parser.add_argument(
        "--require-gpu", action="store_true",
        help="Abort immediately if no CUDA device is visible, before doing any work.",
    )
    parser.add_argument(
        "--only-model", default=None,
        help="Run just this one model name across all matching category "
             "entries, skipping the rest — useful for isolating a single "
             "model's real error without a full multi-model batch run.",
    )
    args = parser.parse_args()

    if args.require_gpu:
        require_gpu()
    print_gpu_report()
    device = resolve_device(args.device)
    print(f"[run_all] using device={device!r} for all models\n")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for entry in config["videos"]:
        url = entry["url"]
        category = entry["category"]
        model_names = entry["models"]

        print(f"\n=== {category}: {url} ===")
        if "REPLACE_ME" in url:
            print("  Skipping — placeholder URL, update configs/test_videos.yaml with a real video.")
            continue

        try:
            video_path = fetch_youtube_video(url)
        except Exception as e:
            print(f"  Failed to fetch video: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Run each model SEPARATELY so every model gets its own video/log/csv —
        # no shared/overwritten output when multiple models share a category.
        # A failure in one model (missing dependency, OOM, etc.) is caught
        # and logged so the REST of the models still run instead of the
        # whole batch dying.
        failed_models = []
        active_model_names = [args.only_model] if args.only_model else model_names
        for model_name in active_model_names:
            if model_name not in model_names:
                continue
            try:
                model = build_models([model_name], device=device,
                                     video_name=base_name)[0]
                tag = "GPU" if (model.gpu_accelerated and device.startswith("cuda")) else "CPU"
                print(f"\n  --- [{tag}] {model.name} (device={model.device}) ---")

                runner = PipelineRunner(models=[model])
                runner.load_models()

                detections = runner.run(video_path)
                print(f"  {len(detections)} detections for {model.name}.")

                # Same layout the web UI writes and reads:
                # outputs/runs/<video>/<model>/.  The CLI used to write flat
                # "<video>_<category>_<model>.<ext>" files into outputs/logs
                # and outputs/annotated, so a run started here was invisible
                # in the UI's history and vice versa.
                out_dir = run_dir(base_name, model_name, create=True)
                export_annotated_video(video_path, detections,
                                       os.path.join(out_dir, RUN_VIDEO))
                export_detection_log(detections,
                                     os.path.join(out_dir, RUN_JSON))
                export_detection_csv(detections,
                                     os.path.join(out_dir, RUN_CSV))
                print(f"  -> {out_dir}")
            except Exception as e:
                print(f"  FAILED: {model_name} — {type(e).__name__}: {e}")
                failed_models.append(model_name)
                continue

        if failed_models:
            print(f"\n  {len(failed_models)} model(s) failed and were skipped: {failed_models}")


if __name__ == "__main__":
    main()
