"""
Job engine: one background worker thread per submitted job.

A job is (video, [models], settings). Each selected model runs as its own
*stage* so that one model refusing to load — which is an expected outcome
for the wrappers that need checkpoints — never aborts the others. Stage
failures are recorded and reported, not raised.

Progress is exposed as a snapshot dict polled by the frontend rather than
pushed over a websocket: the runner already reports per-frame, polling a
plain dict is far simpler to reason about than a socket that has to
survive page reloads, and the UI only needs ~2 Hz.

Models are loaded once per stage and reused across the whole video, which
is where nearly all of the wall-clock saving over the CLI comes from when
running several models on the same clip.
"""

import contextlib
import os
import threading
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# outputs/runs/<video>/<model>/{detections.json, detections.csv, annotated.mp4}
#
# One directory per model run, rather than flat folders of
# "<video>_<model>.<ext>" files.  The flat layout could not be read back
# unambiguously: video names contain underscores, so splitting a filename
# into video and model is guesswork, and three separate folders each held one
# file per run under the same stem — which is how an annotated output and a
# validation artifact end up looking like the same thing.  A directory per
# run needs no name parsing and puts everything a run produced in one place.
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")

# Retained for the ANPR gallery and for reading pre-restructure outputs.
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

# Filenames inside a run directory.  Fixed, because the directory already
# carries the video and model identity.
RUN_JSON = "detections.json"
RUN_CSV = "detections.csv"
RUN_VIDEO = "annotated.mp4"


def run_dir(video: str, model_key: str, create: bool = False) -> str:
    """Path to outputs/runs/<video>/<model_key>/."""
    path = os.path.join(RUNS_DIR, video, model_key)
    if create:
        os.makedirs(path, exist_ok=True)
    return path

# Labels that count as a positive event, per category. Everything else a
# model emits ("standing", "non_violence") is context, not a detection.
#
# Traffic is the odd one out: every vehicle row is a real detection, there
# is no "nothing happening" counterpart label, so both statuses count.
# Without them the Events column read 0 for every traffic run and the
# detections modal (positives-only by default) came back empty, making a
# working detector look like it had found nothing.
POSITIVE_LABELS = {"fall", "violence", "fire", "smoke",
                   "turbulence", "convergence", "crush_risk",
                   "vehicle_moving", "vehicle_parked",
                   # ANPR: every captured vehicle is a result, whether or not
                   # its plate turned out to be legible.
                   "vehicle_plate", "vehicle_unread",
                   "umbrella",
                   # Dense optical flow crowd-safety
                   "flow_analysis",
                   "mean_divergence_critical", "counterflow_warning",
                   "turbulence_index_critical", "stop_go_warning",
                   "mean_speed_warning", "vehicle_in_ped_zone"}


@dataclass
class Stage:
    model_key: str
    status: str = "pending"        # pending | queued | loading | running | done | failed | cancelled
    progress: float = 0.0          # 0..1
    frames_done: int = 0
    frames_total: int = 0
    detections: int = 0
    positives: int = 0
    label_counts: dict = field(default_factory=dict)
    scoring_modes: dict = field(default_factory=dict)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    log_json: Optional[str] = None
    log_csv: Optional[str] = None
    annotated: Optional[str] = None

    def to_dict(self) -> dict:
        elapsed = None
        if self.started_at and self.status not in ("pending", "queued"):
            elapsed = round((self.finished_at or time.time()) - self.started_at, 1)
        return {
            "model_key": self.model_key,
            "status": self.status,
            "progress": round(self.progress, 4),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "detections": self.detections,
            "positives": self.positives,
            "label_counts": self.label_counts,
            "scoring_modes": self.scoring_modes,
            "elapsed_sec": elapsed,
            "error": self.error,
            "log_json": self.log_json,
            "log_csv": self.log_csv,
            "annotated": self.annotated,
        }


@dataclass
class Job:
    id: str
    source: str                     # the URL or path the user submitted
    model_keys: list
    sample_every_n_frames: int
    device: Optional[str]
    export_video: bool
    thresholds: dict = field(default_factory=dict)
    status: str = "queued"          # queued | fetching | running | done | failed | cancelled
    message: str = ""
    video_path: Optional[str] = None
    video_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    stages: dict = field(default_factory=dict)
    error: Optional[str] = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "status": self.status,
            "message": self.message,
            "video_name": self.video_name,
            "video_path": self.video_path,
            "sample_every_n_frames": self.sample_every_n_frames,
            "device": self.device,
            "export_video": self.export_video,
            "created_at": self.created_at,
            "elapsed_sec": round((self.finished_at or time.time()) - self.created_at, 1),
            "error": self.error,
            "stages": [self.stages[k].to_dict() for k in self.model_keys
                       if k in self.stages],
        }


class JobManager:
    """In-memory job registry. One worker thread per job, jobs run serially
    against the GPU via a global lock — two 3D-CNNs sharing a 4 GB card is a
    reliable way to OOM both of them."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._gpu_lock = threading.Lock()

    @contextlib.contextmanager
    def gpu_guard(self, on_wait=None):
        """
        Hold the GPU lock for the duration of the block.

        The same lock ``_run_job`` takes, exposed so work outside the job
        pipeline — the validation runner, which lives in its own thread — can
        queue behind a running job instead of racing it onto the card.  Two
        networks on a 4 GB device is a reliable way to OOM both, and that is
        just as true when one of them belongs to validation.

        ``on_wait`` is called once, only if the lock is not immediately
        available, so the caller can tell the user it is queued rather than
        stalled.  Acquisition then blocks: a caller that gave up here would be
        back to racing for the card, which is the thing being prevented.
        """
        acquired = self._gpu_lock.acquire(blocking=False)
        if not acquired:
            if on_wait is not None:
                on_wait()
            self._gpu_lock.acquire(blocking=True)
        try:
            yield
        finally:
            self._gpu_lock.release()

    def create(self, source: str, model_keys: list, sample_every_n_frames: int,
               device: Optional[str], export_video: bool,
               local_path: Optional[str] = None, thresholds: Optional[dict] = None) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            source=source,
            model_keys=list(model_keys),
            sample_every_n_frames=max(1, int(sample_every_n_frames)),
            device=device,
            export_video=export_video,
            thresholds=thresholds or {},
        )
        job.video_path = local_path
        for key in job.model_keys:
            job.stages[key] = Stage(model_key=key)
        with self._lock:
            self._jobs[job.id] = job

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        job._cancel.set()
        job.message = "Cancelling..."
        return True

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _run_job(self, job: Job):
        acquired = self._gpu_lock.acquire(blocking=False)
        if not acquired:
            job.status = "queued"
            job.message = "Queued \u2014 waiting for previous job to finish..."
            self._gpu_lock.acquire(blocking=True)  # Block until previous job finishes completely

        try:
            self._prepare_video(job)
            if job._cancel.is_set():
                return self._finish(job, "cancelled", "Cancelled before processing.")

            job.status = "running"
            for key in job.model_keys:
                if job._cancel.is_set():
                    for k in job.model_keys:
                        if job.stages[k].status in ("pending", "queued"):
                            job.stages[k].status = "cancelled"
                    return self._finish(job, "cancelled", "Cancelled by user.")
                self._run_stage(job, key)

            failed = [s for s in job.stages.values() if s.status == "failed"]
            if failed and len(failed) == len(job.stages):
                self._finish(job, "failed", "Every model failed. See per-model errors.")
            elif failed:
                self._finish(job, "done",
                             f"{len(job.stages) - len(failed)} of {len(job.stages)} "
                             f"models completed; {len(failed)} could not run.")
            else:
                self._finish(job, "done", "All models completed.")

        except Exception as e:  # noqa: BLE001 - surfaced to the UI verbatim
            job.error = f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
            self._finish(job, "failed", job.error)
        finally:
            self._gpu_lock.release()

    def _prepare_video(self, job: Job):
        """Resolve the submitted source to a local file."""
        if job.video_path:
            if not os.path.exists(job.video_path):
                raise FileNotFoundError(f"No such video: {job.video_path}")
            job.video_name = os.path.basename(job.video_path)
            job.message = f"Using {job.video_name}"
            return

        job.status = "fetching"
        job.message = "Downloading video..."
        from ingestion.youtube_fetch import fetch_youtube_video
        job.video_path = fetch_youtube_video(job.source)
        job.video_name = os.path.basename(job.video_path)
        job.message = f"Downloaded {job.video_name}"

    def _run_stage(self, job: Job, model_key: str):
        from pipeline.runner import PipelineRunner
        from webapp.registry import build_model

        stage = job.stages[model_key]

        # Stage starts loading immediately as GPU lock is held by parent job
        stage.started_at = time.time()
        stage.status = "loading"
        job.message = f"Loading {model_key}..."

        try:
            model = build_model(model_key, job.device,
                                video_name=job.video_name or "run", threshold=job.thresholds.get(model_key))

            # Give flow-pair models the actual source FPS so speed conversion
            # and stop-go timing are correct.
            if getattr(model, "consumption_type", "") == "flow_pair" and job.video_path:
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(job.video_path)
                _src_fps = _cap.get(_cv2.CAP_PROP_FPS)
                _cap.release()
                if _src_fps and _src_fps > 0:
                    model._fps = float(_src_fps)
                    # The runner calls predict() once per SAMPLED frame, so the
                    # annotated video holds one frame per stride.  Writing it at
                    # the source rate would replay it `stride` times too fast.
                    stride = max(1, int(job.sample_every_n_frames or 1))
                    model.output_fps = float(_src_fps) / stride

            model.load()

            stage.status = "running"
            job.message = f"Running {model_key} on {job.video_name}"

            runner = PipelineRunner(models=[model],
                                    sample_every_n_frames=job.sample_every_n_frames)

            def on_progress(done, total, n_dets):
                stage.frames_done = done
                stage.frames_total = total
                stage.progress = (done / total) if total else 0.0
                stage.detections = n_dets

            detections = runner.run(job.video_path,
                                    progress_callback=on_progress,
                                    should_cancel=job._cancel.is_set)
        except Exception as e:  # noqa: BLE001
            # Expected for wrappers that refuse to load without a checkpoint.
            # Recorded on the stage so the other models still run.
            stage.status = "failed"
            stage.error = f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
            stage.finished_at = time.time()
            return

        if job._cancel.is_set():
            stage.status = "cancelled"
            stage.finished_at = time.time()
            return

        self._summarize(stage, detections)
        self._export(job, stage, detections, model_key, model)

        stage.status = "done"
        stage.progress = 1.0
        stage.finished_at = time.time()


    @staticmethod
    def _summarize(stage: Stage, detections: list):
        labels = Counter(d.label for d in detections)
        scoring = Counter(
            d.extra.get("scoring") for d in detections
            if isinstance(d.extra, dict) and d.extra.get("scoring")
        )
        stage.detections = len(detections)
        stage.label_counts = dict(labels)
        stage.scoring_modes = dict(scoring)
        stage.positives = sum(n for lbl, n in labels.items() if lbl in POSITIVE_LABELS)

    def _export(self, job: Job, stage: Stage, detections: list, model_key: str,
                model=None):
        from pipeline.annotate import (export_annotated_video, export_detection_csv,
                                       export_detection_log)

        video = os.path.splitext(os.path.basename(job.video_path))[0]
        out_dir = run_dir(video, model_key, create=True)

        export_detection_log(detections, os.path.join(out_dir, RUN_JSON))
        export_detection_csv(detections, os.path.join(out_dir, RUN_CSV))
        # Stage paths are relative to outputs/runs/ so the API can serve them
        # without the frontend needing to know the layout.
        stage.log_json = f"{video}/{model_key}/{RUN_JSON}"
        stage.log_csv = f"{video}/{model_key}/{RUN_CSV}"

        if job.export_video:
            job.message = f"Writing annotated video for {model_key}..."
            mp4_path = os.path.join(out_dir, RUN_VIDEO)

            # DenseFlowAnalyser (and any future flow model) writes its own
            # annotated video with the heatmap overlay during finalize().
            # Use it directly instead of re-rendering plain bboxes on top.
            own_video = getattr(model, "annotated_video_path", None)
            if own_video and os.path.exists(own_video):
                import shutil
                shutil.copy2(own_video, mp4_path)
            else:
                export_annotated_video(job.video_path, detections, mp4_path)
            stage.annotated = f"{video}/{model_key}/{RUN_VIDEO}"

    @staticmethod
    def _finish(job: Job, status: str, message: str):
        job.status = status
        job.message = message
        job.finished_at = time.time()


MANAGER = JobManager()
