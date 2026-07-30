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
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

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
                   "umbrella"}


@dataclass
class Stage:
    model_key: str
    status: str = "pending"        # pending | loading | running | done | failed | cancelled
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
        if self.started_at:
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
    pose_size: str = "s"
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

    def create(self, source: str, model_keys: list, sample_every_n_frames: int,
               device: Optional[str], export_video: bool, pose_size: str = "s",
               local_path: Optional[str] = None) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            source=source,
            model_keys=list(model_keys),
            sample_every_n_frames=max(1, int(sample_every_n_frames)),
            device=device,
            export_video=export_video,
            pose_size=pose_size,
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
    def _run_job(self, job: Job):
        try:
            self._prepare_video(job)
            if job._cancel.is_set():
                return self._finish(job, "cancelled", "Cancelled before processing.")

            job.status = "running"
            for key in job.model_keys:
                if job._cancel.is_set():
                    for k in job.model_keys:
                        if job.stages[k].status == "pending":
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
        from pipeline.annotate import (export_annotated_video, export_detection_csv,
                                       export_detection_log)
        from pipeline.runner import PipelineRunner
        from webapp.registry import build_model

        stage = job.stages[model_key]
        stage.started_at = time.time()
        stage.status = "loading"
        job.message = f"Loading {model_key}..."

        try:
            with self._gpu_lock:
                model = build_model(model_key, job.device, pose_size=job.pose_size,
                                    video_name=job.video_name or "run")
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

            if job._cancel.is_set():
                stage.status = "cancelled"
                stage.finished_at = time.time()
                return

            self._summarize(stage, detections)
            self._export(job, stage, detections, model_key)

            stage.status = "done"
            stage.progress = 1.0

        except Exception as e:  # noqa: BLE001
            # Expected for wrappers that refuse to load without a checkpoint.
            # Recorded on the stage so the other models still run.
            stage.status = "failed"
            stage.error = f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
        finally:
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

    def _export(self, job: Job, stage: Stage, detections: list, model_key: str):
        from pipeline.annotate import (export_annotated_video, export_detection_csv,
                                       export_detection_log)

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(ANNOTATED_DIR, exist_ok=True)

        base = os.path.splitext(os.path.basename(job.video_path))[0]
        stem = f"{base}_{model_key}"

        json_path = os.path.join(LOG_DIR, f"{stem}.json")
        csv_path = os.path.join(LOG_DIR, f"{stem}.csv")
        export_detection_log(detections, json_path)
        export_detection_csv(detections, csv_path)
        stage.log_json = f"{stem}.json"
        stage.log_csv = f"{stem}.csv"

        if job.export_video:
            job.message = f"Writing annotated video for {model_key}..."
            mp4_path = os.path.join(ANNOTATED_DIR, f"{stem}.mp4")
            export_annotated_video(job.video_path, detections, mp4_path)
            stage.annotated = f"{stem}.mp4"

    @staticmethod
    def _finish(job: Job, status: str, message: str):
        job.status = status
        job.message = message
        job.finished_at = time.time()


MANAGER = JobManager()
