"""
Runs the full test_videos.yaml set through the appropriate models for
each video's category, and exports annotated videos + logs for all of them.

Usage:
    python scripts/run_all.py --config configs/test_videos.yaml
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.youtube_fetch import fetch_youtube_video
from pipeline.runner import PipelineRunner
from pipeline.annotate import export_annotated_video, export_detection_log, export_detection_csv
from pipeline.device import resolve_device, require_gpu, print_gpu_report

from models.fire_smoke_yolo import FireSmokeYOLO
from models.optical_flow_crush import OpticalFlowCrushDetector

from models.fall import (
    YOLOPoseFallDetector,
    MediaPipeFallDetector,
    AlphaPoseFallDetector,
    STGCNFallDetector,
    PoseC3DFallDetector,
    MoveNetFallDetector,
    OpticalFlowFallDetector,
)
from models.violence import (
    X3DViolenceClassifier,
    SlowFastViolenceClassifier,
    VideoMAEViolenceClassifier,
    I3DViolenceClassifier,
    C3DViolenceClassifier,
    TSMViolenceClassifier,
    MMActionSlowOnlyClassifier,
)


MODEL_REGISTRY = {
    "fire_smoke_yolo": FireSmokeYOLO,
    "optical_flow_crush": OpticalFlowCrushDetector,

    # Fall detection (7)
    "fall_yolo_pose": YOLOPoseFallDetector,
    "fall_mediapipe_pose": MediaPipeFallDetector,
    "fall_alphapose_lstm": AlphaPoseFallDetector,
    "fall_stgcn": STGCNFallDetector,
    "fall_posec3d": PoseC3DFallDetector,
    "fall_movenet": MoveNetFallDetector,
    "fall_optical_flow": OpticalFlowFallDetector,

    # Violence / altercation detection (7)
    "violence_x3d": X3DViolenceClassifier,
    "violence_slowfast": SlowFastViolenceClassifier,
    "violence_videomae": VideoMAEViolenceClassifier,
    "violence_i3d": I3DViolenceClassifier,
    "violence_c3d": C3DViolenceClassifier,
    "violence_tsm": TSMViolenceClassifier,
    "violence_mmaction_slowonly": MMActionSlowOnlyClassifier,
}

FALL_MODELS = [
    "fall_yolo_pose", "fall_mediapipe_pose", "fall_alphapose_lstm",
    "fall_stgcn", "fall_posec3d", "fall_movenet", "fall_optical_flow",
]
VIOLENCE_MODELS = [
    "violence_x3d", "violence_slowfast", "violence_videomae", "violence_i3d",
    "violence_c3d", "violence_tsm", "violence_mmaction_slowonly",
]


def build_models(names: list[str], device: str):
    return [MODEL_REGISTRY[n](device=device) for n in names]


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

    os.makedirs("outputs/annotated", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

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
                model = build_models([model_name], device=device)[0]
                tag = "GPU" if (model.gpu_accelerated and device.startswith("cuda")) else "CPU"
                print(f"\n  --- [{tag}] {model.name} (device={model.device}) ---")

                runner = PipelineRunner(models=[model])
                runner.load_models()

                detections = runner.run(video_path)
                print(f"  {len(detections)} detections for {model.name}.")

                tag_str = f"{base_name}_{category}_{model.name}"
                out_video = f"outputs/annotated/{tag_str}.mp4"
                out_log = f"outputs/logs/{tag_str}.json"
                out_csv = f"outputs/logs/{tag_str}.csv"

                export_annotated_video(video_path, detections, out_video)
                export_detection_log(detections, out_log)
                export_detection_csv(detections, out_csv)
            except Exception as e:
                print(f"  FAILED: {model_name} — {type(e).__name__}: {e}")
                failed_models.append(model_name)
                continue

        if failed_models:
            print(f"\n  {len(failed_models)} model(s) failed and were skipped: {failed_models}")


if __name__ == "__main__":
    main()
