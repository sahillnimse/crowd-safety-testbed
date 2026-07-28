"""
Test one model against one YouTube video. Useful for quick iteration
while tuning a single model, without running the whole stack.

Usage:
    # Local file already on disk:
    python scripts/run_single.py --video test_videos/clip.mp4 --model fall_yolo_pose

    # Or fetch from YouTube (cached in test_videos/ by video ID):
    python scripts/run_single.py --video_url "https://youtube.com/watch?v=..." --model fire_smoke_yolo
    python scripts/run_single.py --video_url "..." --model fall_yolo_pose --pose_size s
    python scripts/run_single.py --video_url "..." --model violence_x3d

Writes three artifacts per run, named <video>_<model>:
    outputs/annotated/<video>_<model>.mp4   detections burned onto the frames
    outputs/logs/<video>_<model>.json       structured log for compare_models.py
    outputs/logs/<video>_<model>.csv        same, flat
"""

import argparse
import os
import sys

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


MODEL_FACTORY = {
    "fire_smoke_yolo": lambda args, device: FireSmokeYOLO(device=device),
    "optical_flow_crush": lambda args, device: OpticalFlowCrushDetector(device=device),

    # Fall detection (7)
    "fall_yolo_pose": lambda args, device: YOLOPoseFallDetector(model_size=args.pose_size, device=device),
    "fall_mediapipe_pose": lambda args, device: MediaPipeFallDetector(device=device),
    "fall_alphapose_lstm": lambda args, device: AlphaPoseFallDetector(device=device),
    "fall_stgcn": lambda args, device: STGCNFallDetector(device=device),
    "fall_posec3d": lambda args, device: PoseC3DFallDetector(device=device),
    "fall_movenet": lambda args, device: MoveNetFallDetector(device=device),
    "fall_optical_flow": lambda args, device: OpticalFlowFallDetector(device=device),

    # Violence / altercation detection (7)
    "violence_x3d": lambda args, device: X3DViolenceClassifier(device=device),
    "violence_slowfast": lambda args, device: SlowFastViolenceClassifier(device=device),
    "violence_videomae": lambda args, device: VideoMAEViolenceClassifier(device=device),
    "violence_i3d": lambda args, device: I3DViolenceClassifier(device=device),
    "violence_c3d": lambda args, device: C3DViolenceClassifier(device=device),
    "violence_tsm": lambda args, device: TSMViolenceClassifier(device=device),
    "violence_mmaction_slowonly": lambda args, device: MMActionSlowOnlyClassifier(device=device),
}


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video_url", help="YouTube URL to fetch (cached in test_videos/)")
    source.add_argument("--video", help="Path to a local video file, skipping any download")
    parser.add_argument("--model", required=True, choices=list(MODEL_FACTORY.keys()))
    parser.add_argument("--pose_size", default="s", choices=["n", "s", "m", "l", "x"],
                         help="Only used by --model fall_yolo_pose")
    parser.add_argument("--sample_every_n_frames", type=int, default=1)
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
    args = parser.parse_args()

    if args.require_gpu:
        require_gpu()
    print_gpu_report()
    device = resolve_device(args.device)
    print(f"[run_single] using device={device!r}\n")

    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"No such video file: {video_path}")
        print(f"[run_single] using local video: {video_path}")
    else:
        video_path = fetch_youtube_video(args.video_url)

    model = MODEL_FACTORY[args.model](args, device)
    tag = "GPU" if (model.gpu_accelerated and device.startswith("cuda")) else "CPU"
    print(f"  [{tag}] {model.name} (device={model.device})")
    runner = PipelineRunner(models=[model], sample_every_n_frames=args.sample_every_n_frames)
    runner.load_models()

    detections = runner.run(video_path)
    print(f"\n{len(detections)} detections found.")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video = f"outputs/annotated/{base_name}_{args.model}.mp4"
    out_log = f"outputs/logs/{base_name}_{args.model}.json"
    out_csv = f"outputs/logs/{base_name}_{args.model}.csv"

    os.makedirs("outputs/annotated", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    export_annotated_video(video_path, detections, out_video)
    export_detection_log(detections, out_log)
    export_detection_csv(detections, out_csv)


if __name__ == "__main__":
    main()
