"""
Test one model against one YouTube video. Useful for quick iteration
while tuning a single model, without running the whole stack.

Usage:
    # Local file already on disk:
    python scripts/run_single.py --video test_videos/clip.mp4 --model dense_flow

    # Or fetch from YouTube (cached in test_videos/ by video ID):
    python scripts/run_single.py --video_url "https://youtube.com/watch?v=..." --model rtdetrv2_traffic
    python scripts/run_single.py --video_url "..." --model violence_x3d --threshold 0.5

`--model` accepts any key in webapp/registry.py; run with `--help` for the
current list.

Writes three artifacts per run into outputs/runs/<video>/<model>/ — the same
layout the web UI writes and reads, so a run started here appears in the UI's
history:
    annotated.mp4     detections burned onto the frames
    detections.json   structured log for compare_models.py
    detections.csv    same, flat
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.youtube_fetch import fetch_youtube_video
from pipeline.runner import PipelineRunner
from pipeline.annotate import export_annotated_video, export_detection_log, export_detection_csv
from pipeline.device import resolve_device, require_gpu, print_gpu_report

from webapp import registry
from webapp.jobs import RUN_CSV, RUN_JSON, RUN_VIDEO, run_dir

# Model names come from webapp/registry.py — the same table the web UI and
# run_all.py build from.  This script previously kept its own MODEL_FACTORY
# covering 12 of the project's models, so --model rtdetrv2_traffic was
# rejected by argparse as an invalid choice even though the model exists and
# works.  Deleted models also lingered in the choices list.
MODEL_KEYS = sorted(registry.BY_KEY)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video_url", help="YouTube URL to fetch (cached in test_videos/)")
    source.add_argument("--video", help="Path to a local video file, skipping any download")
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--sample_every_n_frames", type=int, default=1)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold override. Default: the model's own.",
    )
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

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    model = registry.build_model(args.model, device, video_name=base_name,
                                 threshold=args.threshold)
    tag = "GPU" if (model.gpu_accelerated and device.startswith("cuda")) else "CPU"
    print(f"  [{tag}] {model.name} (device={model.device})")
    runner = PipelineRunner(models=[model], sample_every_n_frames=args.sample_every_n_frames)
    runner.load_models()

    detections = runner.run(video_path)
    print(f"\n{len(detections)} detections found.")

    # outputs/runs/<video>/<model>/ — the layout the web UI reads, so a CLI
    # run shows up in the UI's history instead of being invisible to it.
    import json
    from pipeline.html_report import export_html_report
    from webapp.history import compute_detections_summary

    out_dir = run_dir(base_name, args.model, create=True)
    out_video = os.path.join(out_dir, RUN_VIDEO)
    out_log = os.path.join(out_dir, RUN_JSON)
    out_csv = os.path.join(out_dir, RUN_CSV)
    out_summary = os.path.join(out_dir, "summary.json")
    out_report = os.path.join(out_dir, "report.html")

    own_video = getattr(model, "annotated_video_path", None)
    if own_video and os.path.exists(own_video):
        if os.path.abspath(own_video) != os.path.abspath(out_video):
            import shutil
            shutil.move(own_video, out_video)
    else:
        export_annotated_video(video_path, detections, out_video)

    export_detection_log(detections, out_log)
    export_detection_csv(detections, out_csv)

    summary_data = getattr(model, "summary", None)
    if not summary_data:
        summary_data = compute_detections_summary(detections)

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    export_html_report(out_report, base_name, args.model, summary_data, detections)
    print(f"Summary JSON written to {out_summary}")
    print(f"HTML Report written to {out_report}")


if __name__ == "__main__":
    main()
