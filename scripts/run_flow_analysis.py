"""
CLI entry point for the dense optical flow crowd-safety analyser.

Supports:
  - Video file
  - Image sequence (glob pattern, e.g. "frames/*.jpg")
  - RTSP stream (rtsp://...)

Writes:
  - Per-frame metrics CSV (and optionally Parquet)
  - Annotated output video (H.264 via ffmpeg, falls back to mp4v)
  - Per-zone time-series PNG plots

Usage
-----
  # Video file:
  python scripts/run_flow_analysis.py \\
      --source test_videos/ram_kund_crowd.mp4 \\
      --config configs/crowd_flow.yaml \\
      --camera ram_kund_approach \\
      --output-dir outputs/flow_run_001

  # Live RTSP stream:
  python scripts/run_flow_analysis.py \\
      --source rtsp://192.168.1.100:554/stream1 \\
      --config configs/crowd_flow.yaml \\
      --camera ram_kund_bridge \\
      --output-dir outputs/live_run

  # Run synthetic validation suite and exit:
  python scripts/run_flow_analysis.py --validate

  # Show live OpenCV preview (disable for headless servers):
  python scripts/run_flow_analysis.py --source ... --show

Flags
-----
  --no-vis          Disable all visualisation (saves ~5-12 ms/frame)
  --stride N        Process every Nth frame (default 1 = every frame)
  --max-frames N    Stop after N processed frames
  --show            Display annotated frames in an OpenCV window (local only)
  --validate        Run synthetic validation suite and exit (no video needed)
  --fps FLOAT       Override source FPS (useful for image sequences)
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root and src are on the path so `models/` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.crowd_flow.dense_flow_analyser import DenseFlowAnalyser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_flow_analysis")


# ──────────────────────────────────────────────────────────────────────────────
# Frame source
# ──────────────────────────────────────────────────────────────────────────────

def _open_source(source: str):
    """
    Returns (cap, fps, total_frames, is_glob).

    For RTSP/camera streams, total_frames is -1.
    For image sequences (glob), returns a sorted list of paths.
    """
    if "*" in source or "?" in source:
        paths = sorted(glob.glob(source))
        if not paths:
            raise FileNotFoundError(f"No files matched: {source!r}")
        return paths, None, len(paths), True

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source!r}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, max(total, -1), False


# ──────────────────────────────────────────────────────────────────────────────
# Main run loop
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # Load config
    from config_io import load_yaml
    full_cfg = load_yaml(args.config)
    cfg = full_cfg.get("crowd_flow", {})

    os.makedirs(args.output_dir, exist_ok=True)

    # Open the source and read its real frame rate BEFORE the analyser is
    # built.  load() constructs the AlertEngine, which converts each zone's
    # min_duration_sec into a frame count using fps — so an analyser built at
    # the 30.0 default and corrected afterwards keeps an AlertEngine timing
    # alerts against a frame rate the source does not have.  On this project's
    # own footage that is 25 fps read as 30 (alerts need 20% longer than
    # configured to fire) and 10 fps read as 30 (3x longer).
    source, src_fps, total_frames, is_glob = _open_source(args.source)
    fps = args.fps or (src_fps if src_fps else 30.0)

    # Clamped once, here, rather than at each use: the sampling test below is
    # `fidx % stride`, so a --stride 0 reached it as a ZeroDivisionError on the
    # first frame while the analyser had already been built with max(1, ...).
    stride = max(1, args.stride)

    if args.no_people:
        # Run-scoped override: mutate the in-memory cfg only — the config
        # file is not written back, and webapp jobs build their analyser
        # through webapp/registry (never this CLI), so they keep
        # people_overlay: true.
        cfg["people_overlay"] = False
        logger.info("--no-people: person detector disabled for this run only.")

    analyser = DenseFlowAnalyser(
        config=cfg,
        camera_id=args.camera,
        output_dir=args.output_dir,
        visualise=not args.no_vis,
        fps=fps,
        frame_stride=stride,
    )
    analyser.load()

    # One annotated frame is produced per processed frame, so with --stride N
    # the output must play at fps/N to stay at real-world speed.
    analyser.output_fps = fps / stride

    logger.info(
        "Source: %s  FPS: %.2f  Total frames: %s  Camera: %s",
        args.source, fps,
        str(total_frames) if total_frames > 0 else "∞ (stream)",
        args.camera,
    )

    # The annotated video is written by DenseFlowAnalyser itself, streaming
    # frames to ffmpeg as they are produced.  This script used to run a second
    # encoder over the same frames, which doubled the encode cost and left two
    # near-identical files in the output directory.
    prev_frame:     np.ndarray | None = None
    frame_index:    int = 0
    processed:      int = 0
    all_detections: list = []
    t_start = time.perf_counter()

    def _iter_frames():
        nonlocal frame_index
        if is_glob:
            for path in source:
                # Decode only the frames that will actually be used.  The
                # consumer discards `fidx % stride != 0` anyway, so decoding
                # every JPEG first meant --stride 5 paid five times the
                # decode cost for one processed frame.
                #
                # The video branch below cannot do this: a container has to
                # be read sequentially to advance, so there the skip has to
                # stay on the consumer side.
                if frame_index % stride == 0:
                    img = cv2.imread(path)
                    if img is not None:
                        yield img, frame_index, frame_index / fps
                frame_index += 1
        else:
            cap = source
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame, frame_index, frame_index / fps
                frame_index += 1
            cap.release()

    try:
        for frame, fidx, ts in _iter_frames():
            # Stride skip.  prev_frame is deliberately NOT updated here: the
            # flow pair must span the sampling interval, so it holds the
            # previous PROCESSED frame.  Updating it on skipped frames pinned
            # the baseline to one source frame however coarse the sampling,
            # which is the smallest displacement the footage can offer and so
            # the worst signal-to-noise available.
            if fidx % stride != 0:
                continue

            if args.max_frames and processed >= args.max_frames:
                logger.info("--max-frames %d reached; stopping.", args.max_frames)
                break

            if prev_frame is not None:
                dets = analyser.predict((prev_frame, frame), fidx, ts)
                all_detections.extend(dets)

                # Live preview
                if args.show and analyser.latest_annotated_frame is not None:
                    cv2.imshow("Dense Flow — press Q to quit",
                               analyser.latest_annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User quit preview.")
                        break

                # Progress
                if processed % 30 == 0:
                    elapsed = time.perf_counter() - t_start
                    rate    = processed / elapsed if elapsed > 0 else 0
                    pct     = f"{100*processed/total_frames:.1f}%" if total_frames > 0 else "?"
                    logger.info(
                        "f=%d (%s)  detections=%d  %.1f fps",
                        fidx, pct, len(all_detections), rate,
                    )

                processed += 1

            prev_frame = frame

    finally:
        if args.show:
            cv2.destroyAllWindows()
        analyser.finalize()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Done.  %d frames processed in %.1f s (%.1f fps).  "
        "%d alert detections.",
        processed, elapsed, processed / elapsed if elapsed > 0 else 0,
        len(all_detections),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Dense optical flow crowd-safety analyser CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--source",       default="",
                    help="Video file, image-glob, or RTSP URL.")
    ap.add_argument("--config",       default="configs/crowd_flow.yaml",
                    help="Path to crowd_flow.yaml.")
    ap.add_argument("--camera",       default="ram_kund_approach",
                    help="Camera ID in the config file.")
    ap.add_argument("--output-dir",   default="outputs/flow",
                    help="Directory for CSV, Parquet, video, and plots.")
    ap.add_argument("--stride",       type=int, default=1,
                    help="Process every Nth frame.")
    ap.add_argument("--max-frames",   type=int, default=0,
                    help="Stop after this many processed frames (0 = no limit).")
    ap.add_argument("--fps",          type=float, default=0.0,
                    help="Override detected source FPS.")
    ap.add_argument("--no-vis",       action="store_true",
                    help="Disable visualisation (headless / multi-stream mode).")
    ap.add_argument("--no-people",    action="store_true",
                    help="Disable the people overlay (person detector) for "
                         "THIS RUN ONLY — calibration/stride workflows where "
                         "the detector is pure overhead. configs/"
                         "crowd_flow.yaml is not modified, and webapp runs "
                         "(which never pass through this CLI) are unaffected.")
    ap.add_argument("--show",         action="store_true",
                    help="Display live annotated preview.")
    ap.add_argument("--validate",     action="store_true",
                    help="Run synthetic validation suite and exit.")
    return ap


def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    if args.validate:
        # Delegate to the validation script
        val_script = _PROJECT_ROOT / "tests" / "validate_flow.py"
        logger.info("Running validation suite: %s", val_script)
        ret = subprocess.run(
            [sys.executable, str(val_script), "--all"],
            check=False,
        )
        sys.exit(ret.returncode)

    if not args.source:
        ap.error("--source is required unless --validate is given.")

    run(args)


if __name__ == "__main__":
    main()
