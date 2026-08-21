"""
Main pipeline runner: video file -> frames -> models -> results.

Routes each frame to the right models based on their `consumption_type`:
  - "frame"     : called every frame, immediately
  - "clip"      : fed from the rolling FrameBuffer, once it holds enough
                  frames for that model, every `clip_stride` frames
  - "flow_pair" : called with (previous_frame, current_frame)

Usage:
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(models=[...])
    detections = runner.run("test_videos/sample.mp4")
"""

import cv2
from collections import defaultdict
from tqdm import tqdm

from pipeline.frame_buffer import FrameBuffer
from models.base import Detection

# Per model, stop printing the same failure after this many occurrences. A
# model that is broken at load time fails on every single frame; unthrottled
# that buries the run in thousands of identical lines and leaves a
# zero-detection result that reads like "no events found".
MAX_REPEATED_ERRORS = 3


class PipelineRunner:
    def __init__(self, models: list, clip_buffer_len: int = None,
                 sample_every_n_frames: int = 1):
        """
        models: list of BaseModelWrapper instances (already constructed, not yet loaded)
        clip_buffer_len: how many frames to keep for clip-based models. Defaults
            to the largest clip_len among the models, so a model needing 32
            frames isn't silently fed a 16-frame buffer upsampled by frame
            duplication.
        sample_every_n_frames: process every Nth frame (1 = every frame; raise this
            to speed up testing on long videos at the cost of temporal resolution)
        """
        self.models = models
        required = max([getattr(m, "clip_len", 1) for m in models], default=1)
        if clip_buffer_len is None:
            clip_buffer_len = max(required, 2)
        elif clip_buffer_len < required:
            print(f"[WARN] clip_buffer_len={clip_buffer_len} is smaller than the "
                  f"largest model clip_len={required}; raising it, otherwise those "
                  f"models would score clips padded by duplicated frames.")
            clip_buffer_len = required
        self.clip_buffer_len = clip_buffer_len
        self.sample_every_n_frames = sample_every_n_frames
        self._frame_buffer = FrameBuffer(max_len=clip_buffer_len)

    def load_models(self):
        for m in self.models:
            print(f"Loading {m.name} on {m.device}...")
            m.load()

    def run(self, video_path: str, progress_callback=None,
            should_cancel=None) -> list[Detection]:
        """
        progress_callback: optional fn(frame_index, total_frames, n_detections)
            called as frames are consumed. Used by the web UI to report live
            progress; None keeps the plain tqdm-only behaviour.
        should_cancel: optional fn() -> bool, polled per frame. Returning True
            stops the run and returns the detections gathered so far, so a
            cancelled job still yields partial results instead of nothing.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_detections: list[Detection] = []
        error_counts: dict[str, int] = defaultdict(int)
        prev_sampled = None   # previous SAMPLED frame, for flow_pair models
        frame_index = 0
        sampled_index = 0  # counts only the frames we actually process

        pbar = tqdm(total=total_frames, desc=f"Processing {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % self.sample_every_n_frames == 0:
                timestamp_sec = frame_index / fps
                self._frame_buffer.push(frame)

                for model in self.models:
                    try:
                        if model.consumption_type == "frame":
                            dets = model.predict(frame, frame_index, timestamp_sec)

                        elif model.consumption_type == "clip":
                            dets = self._run_clip_model(
                                model, sampled_index, frame_index, timestamp_sec
                            )

                        elif model.consumption_type == "flow_pair":
                            # Pair with the previous SAMPLED frame, not the
                            # previous source frame.
                            #
                            # Using the source frame made the flow baseline
                            # always one frame, whatever the sampling setting
                            # said: at "every 5th frame" the model was handed
                            # frames 4 and 5, then 9 and 10.  So the setting
                            # controlled how OFTEN flow was computed and never
                            # how far apart the two frames were, and the
                            # measurement was permanently pinned to the
                            # smallest displacement — the worst signal-to-noise
                            # the footage can offer.  Every other consumption
                            # type already works on the sampled stream.
                            if prev_sampled is None:
                                dets = []
                            else:
                                dets = model.predict(
                                    (prev_sampled, frame), frame_index, timestamp_sec
                                )
                        else:
                            raise ValueError(f"Unknown consumption_type: {model.consumption_type}")

                        all_detections.extend(dets)

                    except Exception as e:
                        self._report_error(model, frame_index, e, error_counts)

                prev_sampled = frame
                sampled_index += 1

            frame_index += 1
            pbar.update(1)

            if progress_callback is not None:
                progress_callback(frame_index, total_frames, len(all_detections))
            if should_cancel is not None and should_cancel():
                print(f"[runner] cancelled at frame {frame_index}")
                break

        pbar.close()
        cap.release()

        # Models that accumulate state across the whole video (ANPR builds a
        # per-vehicle gallery) get a chance to write their output now that
        # every frame has been seen.
        for model in self.models:
            finalize = getattr(model, "finalize", None)
            if callable(finalize):
                try:
                    finalize()
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] {model.name}.finalize() failed: {e}")

        self._summarize_errors(error_counts)
        return all_detections

    def _run_clip_model(self, model, sampled_index: int, frame_index: int,
                        timestamp_sec: float) -> list[Detection]:
        """Invoke a clip model only when it has enough frames and is due.

        Both gates matter. Without the readiness gate the model scores a
        clip whose frames are duplicates of the two it actually has; without
        the stride gate a 32-frame 3D-CNN runs once per frame on inputs that
        overlap by ~97% — enormous cost for near-zero added information.
        """
        if len(self._frame_buffer) < model.min_clip_frames:
            return []
        if sampled_index % model.clip_stride != 0:
            return []
        return model.predict(self._frame_buffer.get_clip(), frame_index, timestamp_sec)

    def _report_error(self, model, frame_index: int, exc: Exception,
                      error_counts: dict) -> None:
        key = f"{model.name}: {exc.__class__.__name__}: {exc}"
        error_counts[key] += 1
        if error_counts[key] <= MAX_REPEATED_ERRORS:
            print(f"[WARN] {model.name} failed on frame {frame_index}: {exc}")
            if error_counts[key] == MAX_REPEATED_ERRORS:
                print(f"[WARN] ...suppressing further identical errors from {model.name}")

    def _summarize_errors(self, error_counts: dict) -> None:
        if not error_counts:
            return
        print("\n[WARN] Errors during this run (a model failing on every frame "
              "produces zero detections, which is not the same as finding nothing):")
        for key, count in sorted(error_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {count:6d}x  {key}")
