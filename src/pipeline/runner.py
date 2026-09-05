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

import time
from dataclasses import dataclass

import cv2
from collections import defaultdict
from tqdm import tqdm

from pipeline.frame_buffer import FrameBuffer
from models.base import Detection


@dataclass
class SourceStatus:
    """
    How the read loop ENDED, as distinct from whether models ran.

    This exists because ``cap.read()`` returning False was treated as
    end-of-input unconditionally. A dropped RTSP camera, a truncated file and
    a genuinely finished video are the same value of ``ret``, so a camera that
    failed three minutes into a six-hour shift terminated the run and the job
    reported "done - All models completed."

    In a control room that reads as "this location is monitored and clear",
    which is the most dangerous thing a safety system can say. Callers must
    check ``ok`` before treating a run as a clean pass over the footage.
    """
    is_stream: bool = False
    expected_frames: int = 0
    frames_read: int = 0
    reconnects: int = 0
    #: "completed" | "truncated" | "stream_lost" | "cancelled"
    outcome: str = "completed"

    @property
    def ok(self) -> bool:
        """True only for a clean, complete pass over the source."""
        return self.outcome == "completed"

    def describe(self) -> str:
        if self.outcome == "completed":
            return f"read {self.frames_read} frames to end of source"
        if self.outcome == "truncated":
            pct = (100.0 * self.frames_read / self.expected_frames
                   if self.expected_frames else 0.0)
            return (f"SOURCE TRUNCATED: read {self.frames_read} of "
                    f"{self.expected_frames} frames ({pct:.1f}%) before the "
                    f"stream ended early. Results cover only that portion.")
        if self.outcome == "stream_lost":
            return (f"STREAM LOST after {self.frames_read} frames and "
                    f"{self.reconnects} reconnect attempt(s). Monitoring "
                    f"STOPPED - this location is NOT covered.")
        return f"cancelled after {self.frames_read} frames"

# Per model, stop printing the same failure after this many occurrences. A
# model that is broken at load time fails on every single frame; unthrottled
# that buries the run in thousands of identical lines and leaves a
# zero-detection result that reads like "no events found".
MAX_REPEATED_ERRORS = 3

# A file that ends before this fraction of its declared frame count is treated
# as truncated rather than complete. Not 1.0: container metadata is routinely
# a frame or two out, and flagging every well-formed file would train an
# operator to ignore the warning that matters.
_TRUNCATION_TOLERANCE = 0.99

# Live-stream reconnection. A camera at a mass gathering drops for all sorts of
# transient reasons (PoE glitch, switch reboot, Wi-Fi backhaul); giving up on
# the first failed read would stop monitoring a location for the rest of the
# event. Bounded, because retrying forever hides a camera that is genuinely
# gone behind a process that looks busy.
_STREAM_RECONNECT_ATTEMPTS = 5
_STREAM_RECONNECT_DELAY_SEC = 2.0


class _AlertSinkFacade:
    """Named stand-in so a sink failure reports through the same throttled
    error path as a model failure, instead of printing once per frame."""
    name = "alert-sink"


_ALERT_SINK = _AlertSinkFacade()


class PipelineRunner:
    def __init__(self, models: list, clip_buffer_len: int = None,
                 sample_every_n_frames: int = 1, retain_detections: bool = True):
        """
        models: list of BaseModelWrapper instances (already constructed, not yet loaded)
        clip_buffer_len: how many frames to keep for clip-based models. Defaults
            to the largest clip_len among the models, so a model needing 32
            frames isn't silently fed a 16-frame buffer upsampled by frame
            duplication.
        sample_every_n_frames: process every Nth frame (1 = every frame; raise this
            to speed up testing on long videos at the cost of temporal resolution)
        retain_detections: keep every detection to return at the end of the run.
            True for batch runs, which report on the whole video afterwards.
            Set False for live streaming, where the consumer has already
            handled each frame through `on_detections` and the accumulated
            list is never read: a crowd camera produces ~650 detections per
            frame with no end to the stream, so retaining them is an
            unbounded leak in the one mode that is meant to run for a shift.
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
        self.retain_detections = retain_detections
        self._frame_buffer = FrameBuffer(max_len=clip_buffer_len)
        # Present before run() so a caller that inspects it after a crash (or
        # before starting) gets a defined object rather than AttributeError.
        self.source_status = SourceStatus()

    def load_models(self):
        for m in self.models:
            print(f"Loading {m.name} on {m.device}...")
            m.load()

    def run(self, video_path: str, progress_callback=None,
            should_cancel=None, on_detections=None) -> list[Detection]:
        """
        progress_callback: optional fn(frame_index, total_frames, n_detections)
            called as frames are consumed. Used by the web UI to report live
            progress; None keeps the plain tqdm-only behaviour.
        should_cancel: optional fn() -> bool, polled per frame. Returning True
            stops the run and returns the detections gathered so far, so a
            cancelled job still yields partial results instead of nothing.
        on_detections: optional fn(list[Detection]) called with THIS frame's
            detections as they are produced. Used to push alerts out live -
            waiting until the run ends would deliver a crush warning long
            after the crowd state that caused it. Never allowed to break the
            run: an exception here is reported and processing continues,
            because a broken notifier must not stop the monitoring.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_stream = self._is_stream(video_path, total_frames)

        all_detections: list[Detection] = []
        # Running total, so progress reporting stays correct when the
        # detections themselves are not retained.
        n_detections = 0
        error_counts: dict[str, int] = defaultdict(int)
        prev_sampled = None   # previous SAMPLED frame, for flow_pair models
        frame_index = 0
        sampled_index = 0  # counts only the frames we actually process

        # Outcome of the read loop, published on self.source_status so the
        # caller can tell a completed run from a truncated one. See the
        # SourceStatus docstring: this used to be indistinguishable, and a
        # camera that died mid-shift reported "All models completed."
        self.source_status = SourceStatus(is_stream=is_stream,
                                          expected_frames=total_frames)

        _accepts_frame = False
        if on_detections is not None:
            try:
                import inspect
                sig = inspect.signature(on_detections)
                _accepts_frame = "frame" in sig.parameters or len(sig.parameters) >= 2
            except Exception:
                _accepts_frame = False

        pbar = tqdm(total=total_frames, desc=f"Processing {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                # A failed read is NOT automatically end-of-input. For a live
                # stream it is almost always a dropped connection, and for a
                # file it can be a truncated or corrupt container. Both used
                # to `break` straight into a success report.
                if is_stream:
                    if self._reconnect(cap, video_path):
                        cap = self._reopen(video_path)
                        self.source_status.reconnects += 1
                        continue
                    self.source_status.outcome = "stream_lost"
                    break
                if total_frames > 0 and frame_index < total_frames * self.TRUNCATION_TOLERANCE:
                    self.source_status.outcome = "truncated"
                else:
                    self.source_status.outcome = "completed"
                break

            if frame_index % self.sample_every_n_frames == 0:
                timestamp_sec = frame_index / fps
                self._frame_buffer.push(frame)
                # This frame's detections across all models, so the live
                # alert hook sees one batch per frame rather than per model.
                frame_dets: list[Detection] = []

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

                        if self.retain_detections:
                            all_detections.extend(dets)
                        else:
                            n_detections += len(dets)
                        frame_dets.extend(dets)

                    except Exception as e:
                        self._report_error(model, frame_index, e, error_counts)

                if on_detections is not None:
                    try:
                        if _accepts_frame:
                            on_detections(
                                frame_dets,
                                frame=frame,
                                frame_index=frame_index,
                                timestamp_sec=timestamp_sec,
                            )
                        elif frame_dets:
                            on_detections(frame_dets)
                    except Exception as exc:  # noqa: BLE001
                        self._report_error(
                            _ALERT_SINK, frame_index, exc, error_counts)

                prev_sampled = frame
                sampled_index += 1

            frame_index += 1
            pbar.update(1)

            if progress_callback is not None:
                progress_callback(
                    frame_index, total_frames,
                    len(all_detections) if self.retain_detections else n_detections,
                )
            if should_cancel is not None and should_cancel():
                print(f"[runner] cancelled at frame {frame_index}")
                self.source_status.outcome = "cancelled"
                break

        pbar.close()
        self.source_status.frames_read = frame_index
        if not self.source_status.ok:
            # Loud, because the alternative is a truncated run that reads as a
            # clean one. This is the message an operator must not miss.
            print(f"\n[runner] !! {self.source_status.describe()}")
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

    TRUNCATION_TOLERANCE = _TRUNCATION_TOLERANCE

    @staticmethod
    def is_live_source(video_path: str) -> bool:
        """Whether this source produces frames on its own clock.

        A camera does: frames arrive whether or not anything is ready for
        them, so a consumer that cannot keep up must skip or fall behind
        reality.  A file does not — it waits, and a slow pass over it is
        simply a slow pass, not a growing lag against the world.

        The distinction decides whether the live preview is allowed to drop
        frames to keep pace.  Callers that only have a path (no open capture)
        use this; ``run()`` uses ``_is_stream`` with the capture's own frame
        count, which additionally catches a device index.
        """
        import cv2 as _cv2

        lowered = str(video_path).lower()
        if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
            return True
        cap = _cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return False
            return int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) <= 0
        finally:
            cap.release()

    @staticmethod
    def _is_stream(video_path: str, total_frames: int) -> bool:
        """A live source: a network URL, or a capture reporting no length."""
        lowered = str(video_path).lower()
        if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
            return True
        # A device index or a capture with no declared frame count is live too.
        return total_frames <= 0

    @staticmethod
    def _reopen(video_path: str):
        return cv2.VideoCapture(video_path)

    @staticmethod
    def _reconnect(cap, video_path: str) -> bool:
        """
        Try to bring a dropped live source back. True if reading resumed.

        Released before reopening: leaking a capture per attempt exhausts
        file descriptors on a long-running server, and the failure then looks
        like a camera problem rather than a resource leak.
        """
        for attempt in range(1, _STREAM_RECONNECT_ATTEMPTS + 1):
            print(f"[runner] source dropped; reconnect attempt "
                  f"{attempt}/{_STREAM_RECONNECT_ATTEMPTS} to {video_path}")
            try:
                cap.release()
            except Exception:  # noqa: BLE001 - already broken, nothing to save
                pass
            time.sleep(_STREAM_RECONNECT_DELAY_SEC)
            probe = cv2.VideoCapture(video_path)
            if probe.isOpened():
                ok, _ = probe.read()
                probe.release()
                if ok:
                    print(f"[runner] reconnected to {video_path}")
                    return True
            else:
                probe.release()
        print(f"[runner] GAVE UP reconnecting to {video_path} after "
              f"{_STREAM_RECONNECT_ATTEMPTS} attempts - monitoring STOPPED")
        return False

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
