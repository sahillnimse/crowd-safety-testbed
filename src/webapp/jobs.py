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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
SESSIONS_DIR = os.path.join(OUTPUT_DIR, "sessions")

# Retained for the ANPR gallery and for reading pre-restructure outputs.
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

# Filenames inside a run directory.  Fixed, because the directory already
# carries the video and model identity.
RUN_JSON = "detections.json"
RUN_CSV = "detections.csv"
RUN_VIDEO = "annotated.mp4"
RUN_SUMMARY = "summary.json"
RUN_REPORT = "report.html"


# Job state is mirrored here so it survives a process restart. The registry
# was in-memory only: restarting the server (deploy, crash, power event)
# erased every record of what had been running, including jobs that were
# mid-flight. At a multi-day event that is the difference between "camera 7
# was being monitored until 03:12" and no record at all.
STATE_DIR = os.path.join(OUTPUT_DIR, "state")
_STATE_VERSION = 1


def _available_devices() -> list:
    """
    Device strings this machine can run jobs on, one job per entry.

    Probed lazily and defensively: importing torch at module scope would undo
    the "torch is never imported at startup" property the webapp relies on for
    a fast page load, and a machine with no CUDA (or a broken driver) must
    still get a working single-slot CPU pool rather than an exception at
    import time.

    Concurrency is capped at the number of CARDS. One job per GPU, because two
    networks on one device is the OOM this pool exists to prevent -- the fix
    for a single global lock is per-device locks, not unlimited parallelism.
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001 - no torch / broken driver
        print(f"[jobs] CUDA probe failed ({exc}); running single-slot on CPU.")
    return ["cpu"]


def run_dir(video: str, model_key: str, create: bool = False) -> str:
    """Path to outputs/runs/<video>/<model_key>/."""
    path = os.path.join(RUNS_DIR, video, model_key)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


# The annotated video is usually H.264/mp4, but _AnnotatedVideoWriter falls
# back to MJPG/AVI on a machine with no ffmpeg.  Readers must therefore look
# for the run video by STEM rather than assume the extension, or a fallback
# encode is invisible to the history view even though it is on disk.
RUN_VIDEO_EXTS = (".mp4", ".avi")


def find_run_video(model_dir: str) -> Optional[str]:
    """Filename of the annotated video in ``model_dir``, or None."""
    stem = os.path.splitext(RUN_VIDEO)[0]
    for ext in RUN_VIDEO_EXTS:
        if os.path.exists(os.path.join(model_dir, stem + ext)):
            return stem + ext
    return None

# Labels that count as a positive event, per category. Everything else a
# model emits ("standing", "non_violence") is context, not a detection.
#
# Traffic is the odd one out: every vehicle row is a real detection, there
# is no "nothing happening" counterpart label, so both statuses count.
# Without them the Events column read 0 for every traffic run and the
# detections modal (positives-only by default) came back empty, making a
# working detector look like it had found nothing.
#
# The dense-flow entries are NOT written out here.  They are exactly
# "<metric>_<severity>" over the AlertEngine's threshold table, and the copy
# that used to live in this set had drifted: it listed "counterflow_warning",
# "stop_go_warning" and "vehicle_in_ped_zone", none of which the engine emits,
# while the real labels ("counterflow_score_warning",
# "vehicle_in_ped_zone_warning", "mean_curl_warning" and both crowd_pressure
# levels) were absent.  So genuine alerts were not counted in the Events
# column and were filtered out of the positives-only detections modal.
#
# Imported lazily rather than at module scope: webapp.app imports this module
# at startup and models.crowd_flow.zones pulls in cv2 and numpy, which is a
# second of page-load time for a set of strings.
POSITIVE_LABELS = {"fall", "violence", "fire", "smoke",
                   # OpticalFlowCrushDetector (src/models/crush/optical_flow_crush.py)
                   "turbulence", "convergence", "crush_risk",
                   "vehicle_moving", "vehicle_parked",
                   # ANPR: every captured vehicle is a result, whether or not
                   # its plate turned out to be legible.
                   "vehicle_plate", "vehicle_unread",
                   "umbrella",
                   # CrowdMotionMonitor (models/crowd_flow/crowd_motion_monitor.py)
                   "person_stopped", "person_crush_zone", "person_moving",
                   "person_moving_stream_a", "person_moving_stream_b",
                   # DMCountCrowdMonitor (models/dm_count/): per-head rows are
                   # real detections; its rule-based alert labels follow the
                   # same "<metric>_<severity>" convention as the dense-flow
                   # engine's and count as events. The "dm_frame_metrics"
                   # telemetry row is deliberately NOT here — it is emitted
                   # every frame by design and would drown the Events column.
                   "head_moving", "head_stopped",
                   "crowd_density_warning", "crowd_density_critical",
                   "crowd_compression_warning", "crowd_compression_critical",
                   "counter_flow_warning", "capacity_warning",
                   # DenseFlowAnalyser's per-frame summary row, which is not an
                   # alert and so is not in DENSE_FLOW_ALERT_LABELS.
                   "flow_analysis"}


def _add_dense_flow_labels() -> None:
    """Fold the dense-flow alert labels in, on first use rather than at import."""
    from models.crowd_flow.zones import DENSE_FLOW_ALERT_LABELS
    POSITIVE_LABELS.update(DENSE_FLOW_ALERT_LABELS)


def positive_labels() -> set:
    """POSITIVE_LABELS, with the dense-flow alert labels resolved."""
    if "mean_divergence_critical" not in POSITIVE_LABELS:
        _add_dense_flow_labels()
    return POSITIVE_LABELS


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
    summary: dict = field(default_factory=dict)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    log_json: Optional[str] = None
    log_csv: Optional[str] = None
    log_summary: Optional[str] = None
    report_html: Optional[str] = None
    annotated: Optional[str] = None
    # Source-integrity fields. `degraded` means the models ran fine but the
    # SOURCE did not survive the run, so the results cover less footage than
    # was asked for. Kept separate from `status`/`error` because nothing
    # failed in the pipeline -- the input did.
    degraded: bool = False
    source_outcome: str = "completed"
    source_detail: str = ""
    frames_read: int = 0

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
            "summary": self.summary,
            "elapsed_sec": elapsed,
            "error": self.error,
            "log_json": self.log_json,
            "log_csv": self.log_csv,
            "log_summary": self.log_summary,
            "report_html": self.report_html,
            "annotated": self.annotated,
            "degraded": self.degraded,
            "source_outcome": self.source_outcome,
            "source_detail": self.source_detail,
            "frames_read": self.frames_read,
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
    mode: str = "batch"             # batch | live
    status: str = "queued"          # queued | fetching | running | done | degraded | failed | cancelled
    message: str = ""
    video_path: Optional[str] = None
    video_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    stages: dict = field(default_factory=dict)
    error: Optional[str] = None
    #: Device the pool reserved for this job ("cuda:1", "cpu"). Reported so an
    #: operator can see the work spread across cards rather than guessing.
    assigned_device: Optional[str] = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "mode": self.mode,
            "status": self.status,
            "message": self.message,
            "video_name": self.video_name,
            "video_path": self.video_path,
            "sample_every_n_frames": self.sample_every_n_frames,
            "device": self.device,
            "assigned_device": self.assigned_device,
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
        # One lock PER DEVICE, not one globally.
        #
        # This was a single threading.Lock(), so the whole server ran exactly
        # one job at a time no matter how many GPUs the machine had -- adding
        # cards bought nothing, because nothing ever asked how many there
        # were. A multi-camera deployment needs concurrency proportional to
        # the hardware, so the pool is sized from torch.cuda.device_count()
        # and each worker holds only the card it is using.
        self._devices = _available_devices()
        self._device_locks = {d: threading.Lock() for d in self._devices}
        # Guards the scan for a free device so two threads cannot both decide
        # the same card is available.
        self._device_pool_lock = threading.Condition()
        self._restore_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _state_path(self, job_id: str) -> str:
        return os.path.join(STATE_DIR, f"{job_id}.json")

    def _persist(self, job: "Job") -> None:
        """
        Mirror one job to disk. Best-effort: a persistence failure must never
        take down a running job, so it warns and continues -- losing the audit
        record is bad, losing the monitoring is worse.

        Written to a temp file and replaced atomically, so a crash mid-write
        cannot leave a half-serialised job that fails to parse on restart.
        """
        import json
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            payload = job.to_dict()
            payload["_version"] = _STATE_VERSION
            tmp = self._state_path(job.id) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self._state_path(job.id))
        except Exception as exc:  # noqa: BLE001
            print(f"[jobs] WARN could not persist job {job.id}: {exc}")

    def _restore_state(self) -> None:
        """
        Reload jobs from disk at startup.

        A job recorded as running or queued cannot still be running -- this
        process just started, so its worker thread is gone. Those are marked
        `interrupted` rather than left claiming to be active: a status of
        "running" for a thread that does not exist is exactly the kind of
        false assurance this system must not give.
        """
        import json
        if not os.path.isdir(STATE_DIR):
            return
        restored = interrupted = 0
        for name in sorted(os.listdir(STATE_DIR)):
            if not name.endswith(".json") or name == "metric_store.json":
                continue
            try:
                with open(os.path.join(STATE_DIR, name), encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[jobs] WARN unreadable state file {name}: {exc}")
                continue
            if not isinstance(d, dict) or ("stages" not in d and "source" not in d):
                continue
            job = Job(
                id=d.get("id") or os.path.splitext(name)[0],
                source=d.get("source", ""),
                model_keys=list(d.get("model_keys") or
                                [st.get("model_key") for st in d.get("stages", [])]),
                sample_every_n_frames=d.get("sample_every_n_frames", 5),
                device=d.get("device"),
                export_video=d.get("export_video", True),
                mode=d.get("mode", "batch"),
            )
            job.status = d.get("status", "unknown")
            job.message = d.get("message", "")
            job.video_name = d.get("video_name")
            job.video_path = d.get("video_path")
            job.created_at = d.get("created_at", time.time())
            job.finished_at = d.get("finished_at")
            job.assigned_device = d.get("assigned_device")
            for st in d.get("stages", []):
                stage = Stage(model_key=st.get("model_key", "?"))
                for k, v in st.items():
                    if hasattr(stage, k):
                        setattr(stage, k, v)
                job.stages[stage.model_key] = stage
            if job.status in ("queued", "fetching", "running", "loading"):
                job.status = "interrupted"
                job.message = ("Server restarted while this job was running - "
                               "coverage for this period is INCOMPLETE.")
                job.finished_at = job.finished_at or time.time()
                for stage in job.stages.values():
                    if stage.status in ("pending", "queued", "loading", "running"):
                        stage.status = "interrupted"
                        stage.degraded = True
                        stage.source_detail = ("server restarted mid-run; "
                                               "footage after this point was not processed")
                interrupted += 1
                self._persist(job)
            self._jobs[job.id] = job
            restored += 1
        if restored:
            print(f"[jobs] restored {restored} job(s) from {STATE_DIR}"
                  + (f"; {interrupted} marked INTERRUPTED" if interrupted else ""))

    @property
    def capacity(self) -> int:
        """How many jobs can run concurrently on this machine."""
        return len(self._devices)

    @contextlib.contextmanager
    def gpu_guard(self, on_wait=None):
        """
        Hold the GPU lock for the duration of the block.

        The same lock ``_run_job`` takes, exposed so work outside the job
        pipeline — the validation runner, which lives in its own thread — can
        queue behind a running job instead of racing it onto the card.  Two
        networks on a 4 GB device is a reliable way to OOM both, and that is
        just as true when one of them belongs to validation.

        ``on_wait`` is called once, only if no device is immediately
        available, so the caller can tell the user it is queued rather than
        stalled.  Acquisition then blocks: a caller that gave up here would be
        back to racing for the card, which is the thing being prevented.

        Yields the DEVICE STRING it reserved ("cuda:0", "cuda:1", "cpu") so
        the caller can pin its model to that specific card. Callers that do
        not care may ignore it -- the previous no-argument `with` form still
        works.
        """
        device = self._acquire_device(on_wait=on_wait)
        try:
            yield device
        finally:
            self._release_device(device)

    def _acquire_device(self, on_wait=None) -> str:
        """Reserve one device, blocking until one frees up."""
        announced = False
        with self._device_pool_lock:
            while True:
                for dev in self._devices:
                    if self._device_locks[dev].acquire(blocking=False):
                        return dev
                if on_wait is not None and not announced:
                    on_wait()
                    announced = True
                # Condition.wait releases the pool lock while blocked, so a
                # finishing worker can take it to notify. A bare sleep-loop
                # here would hold it and deadlock every release.
                self._device_pool_lock.wait(timeout=1.0)

    def _release_device(self, device: str) -> None:
        with self._device_pool_lock:
            lock = self._device_locks.get(device)
            if lock is not None and lock.locked():
                lock.release()
            self._device_pool_lock.notify_all()

    def create(self, source: str, model_keys: list, sample_every_n_frames: int,
               device: Optional[str], export_video: bool,
               local_path: Optional[str] = None, thresholds: Optional[dict] = None,
               mode: str = "batch") -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            source=source,
            model_keys=list(model_keys),
            sample_every_n_frames=max(1, int(sample_every_n_frames)),
            device=device,
            export_video=export_video,
            thresholds=thresholds or {},
            mode=mode,
        )
        job.video_path = local_path
        for key in job.model_keys:
            job.stages[key] = Stage(model_key=key)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)

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
        self._persist(job)
        return True

    def delete(self, job_id: str) -> bool:
        """Remove a job from memory and disk if it is not currently running."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in ("running", "queued") and not job._cancel.is_set():
                return False
            self._jobs.pop(job_id, None)

        sp = self._state_path(job_id)
        if os.path.exists(sp):
            try:
                os.remove(sp)
            except OSError:
                pass
        return True

    def clear_finished(self) -> int:
        """Clear all completed/failed/cancelled/interrupted jobs from memory and disk."""
        to_remove = []
        with self._lock:
            for jid, job in self._jobs.items():
                if job.status in ("done", "failed", "cancelled", "interrupted"):
                    to_remove.append(jid)
            for jid in to_remove:
                self._jobs.pop(jid, None)

        for jid in to_remove:
            sp = self._state_path(jid)
            if os.path.exists(sp):
                try:
                    os.remove(sp)
                except OSError:
                    pass
        return len(to_remove)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _run_job(self, job: Job):
        # Fetch BEFORE taking the card. A YouTube download is network-bound and
        # touches no GPU, yet doing it under the lock meant a job that spent two
        # minutes downloading held every other job off the device for those two
        # minutes. Two jobs can now fetch concurrently and still serialise onto
        # the GPU, which is the only genuinely contended resource.
        try:
            self._prepare_video(job)
            if job._cancel.is_set():
                return self._finish(job, "cancelled", "Cancelled before processing.")
        except Exception as e:  # noqa: BLE001 - surfaced to the UI verbatim
            job.error = f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
            return self._finish(job, "failed", job.error)

        def _announce_wait() -> None:
            job.status = "queued"
            job.message = "Queued \u2014 waiting for previous job to finish..."

        # Through gpu_guard rather than acquiring the lock by hand: it is the
        # same lock the validation runner queues on, and two copies of the
        # acquire / announce / release logic are two places to drift apart.
        try:
            with self.gpu_guard(on_wait=_announce_wait) as device:
                # Pin this job to the card the pool reserved. Without this the
                # stage would resolve its own device and every concurrent job
                # would pick cuda:0, which is the OOM the pool exists to stop.
                # An explicit device from the user still wins -- they asked.
                if job.device is None:
                    job.device = device
                job.assigned_device = device
                self._run_stages(job)
        except Exception as e:  # noqa: BLE001 - surfaced to the UI verbatim
            job.error = f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
            self._finish(job, "failed", job.error)

    def _run_stages(self, job: Job) -> None:
        """Run each selected model in turn. Called with the GPU lock held."""
        try:
            job.status = "running"
            for key in job.model_keys:
                if job._cancel.is_set():
                    for k in job.model_keys:
                        if job.stages[k].status in ("pending", "queued"):
                            job.stages[k].status = "cancelled"
                    return self._finish(job, "cancelled", "Cancelled by user.")
                if job.mode == "live":
                    self._run_stage_live(job, key)
                else:
                    self._run_stage(job, key)

            failed = [s for s in job.stages.values() if s.status == "failed"]
            if failed and len(failed) == len(job.stages):
                self._finish(job, "failed", "Every model failed. See per-model errors.")
            elif failed:
                self._finish(job, "done",
                             f"{len(job.stages) - len(failed)} of {len(job.stages)} "
                             f"models completed; {len(failed)} could not run.")
            else:
                degraded = [s for s in job.stages.values() if s.degraded]
                if degraded:
                    detail = degraded[0].source_detail or "source did not complete"
                    self._finish(job, "degraded",
                                 f"INCOMPLETE COVERAGE - {detail}")
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
                # source_fps() falls back to ffprobe metadata (and then a
                # default) when the container reports CAP_PROP_FPS as 0/NaN —
                # skipping the correction in that case is what produced
                # stride-x sped-up annotated videos.
                from pipeline.video_meta import source_fps
                _src_fps = source_fps(job.video_path)
                stride = max(1, int(job.sample_every_n_frames or 1))
                model._fps = float(_src_fps)
                # The runner pairs consecutive SAMPLED frames, so the flow
                # baseline is `stride` source frames and predict() is
                # called at src_fps/stride.  The model needs both numbers:
                # one to keep px/frame units, one to time alert durations.
                model._frame_stride = stride
                # The runner calls predict() once per SAMPLED frame, so the
                # annotated video holds one frame per stride.  Writing it at
                # the source rate would replay it `stride` times too fast.
                model.output_fps = float(_src_fps) / stride

            model.load()

            stage.status = "running"
            job.message = f"Running {model_key} on {job.video_name}"

            runner = PipelineRunner(models=[model],
                                    sample_every_n_frames=job.sample_every_n_frames)

            # Alerts and metrics leave the process as they are produced, not at the end.
            from pipeline.alert_sink import dispatch_detections
            from topology.graph import TOPOLOGY
            from topology.metric_store import METRIC_STORE
            _dispatched = {"n": 0}
            # One-shot latch so a per-frame extraction failure is
            # reported once, not thousands of times.
            _metric_warned = {"done": False}

            def _resolve_camera_id() -> str:
                # Match against topology camera IDs or names
                v_name = (job.video_name or job.source or "").lower()
                for cid, node in TOPOLOGY.cameras.items():
                    if cid.lower() in v_name or node.name.lower() in v_name:
                        return cid
                if model_key in TOPOLOGY.cameras:
                    return model_key
                return job.video_name or model_key

            resolved_cam_id = _resolve_camera_id()
            cam_node = TOPOLOGY.get_camera(resolved_cam_id)
            cam_clock_offset = cam_node.clock_offset_sec if cam_node else 0.0

            def on_alerts(dets):
                _dispatched["n"] += dispatch_detections(
                    dets, camera_id=model_key, source=job.video_name or job.source)

                # Extract telemetry for multi-camera cross fusion
                if dets:
                    try:
                        raw_ts = float(getattr(dets[0], "timestamp_sec", 0.0))
                        person_dets = [d for d in dets if "person" in d.label or "head" in d.label]
                        moving_dets = [d for d in person_dets if "moving" in d.label or "crush" in d.label]
                        p_count = len(person_dets) if person_dets else len(dets)
                        
                        # Flow rate.
                        #
                        # This is NOT pax/min and is published as uncalibrated
                        # (see the METRIC_STORE.update call below). Real
                        # pedestrian flow is people crossing a line of known
                        # real-world width per unit time; deriving it needs a
                        # calibrated counting line, which no camera here has.
                        #
                        # What this actually computes is (moving detections x
                        # mean pixel speed x a constant). It is monotonic with
                        # real flow on ONE camera, so it is useful as a trend,
                        # and it is meaningless across cameras or against a
                        # physical capacity. The constant below is a display
                        # scale, not a unit conversion, and is named as such
                        # so nobody later mistakes it for one.
                        speed_samples = [
                            float(d.extra.get("speed_px_frame", 1.0))
                            for d in dets if isinstance(d.extra, dict) and "speed_px_frame" in d.extra
                        ]
                        avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 1.0
                        _DISPLAY_SCALE = 12.0
                        flow_rate = max(0.0, float(
                            len(moving_dets) * max(1.0, avg_speed) * _DISPLAY_SCALE))
                        
                        # Crush risk score (0.0 to 1.0)
                        crush_scores = [
                            float(d.extra.get("local_crush_risk", 0.0))
                            for d in dets if isinstance(d.extra, dict) and "local_crush_risk" in d.extra
                        ]
                        crush_risk = max(crush_scores) if crush_scores else 0.0
                        
                        # Dominant heading vector
                        headings = [
                            float(d.extra.get("heading_deg", 0.0))
                            for d in dets if isinstance(d.extra, dict) and "heading_deg" in d.extra
                        ]
                        if headings:
                            import math
                            rads = [math.radians(h) for h in headings]
                            avg_cos = sum(math.cos(r) for r in rads) / len(rads)
                            avg_sin = sum(math.sin(r) for r in rads) / len(rads)
                            dom_vec = (round(avg_cos, 3), round(avg_sin, 3))
                        else:
                            dom_vec = (1.0, 0.0)

                        # Density.
                        #
                        # CrowdMotionMonitor's own per-frame value is persons
                        # per MEGAPIXEL of image, not persons per m2 — it has
                        # no homography. The fallback below (count x 0.15) is
                        # not a physical quantity at all; it assumes every
                        # camera covers the same ground area, which is exactly
                        # what perspective makes false.
                        #
                        # Both are published as uncalibrated. That matters
                        # because `density_threshold: 2.5` in topology.yaml is
                        # a Helbing-scale pax/m2 figure: comparing either of
                        # these against it is off by orders of magnitude in an
                        # unknown direction.
                        density_val = float(p_count * 0.15)
                        density_units = "count x 0.15 (not a physical density)"
                        if hasattr(model, "_frame_density") and model._frame_density:
                            density_val = float(model._frame_density[-1])
                            density_units = "persons per megapixel of image"

                        # Calibration state is derived, never assumed. When a
                        # camera does get a homography or a fitted perspective
                        # map, this is the single place that has to change for
                        # the fusion rules to switch on.
                        is_calibrated = bool(
                            getattr(getattr(model, "_calib", None), "is_calibrated", False)
                            or getattr(model, "_calibrated", False)
                        )

                        METRIC_STORE.update(
                            camera_id=resolved_cam_id,
                            density=density_val,
                            flow_rate_pax_min=flow_rate,
                            dominant_direction_vector=dom_vec,
                            crush_risk_score=crush_risk,
                            person_count=p_count,
                            raw_timestamp_sec=raw_ts,
                            stream_start_epoch_ms=int(stage.started_at * 1000) if stage.started_at else None,
                            clock_offset_sec=cam_clock_offset,
                            flow_is_calibrated=is_calibrated,
                            density_is_calibrated=is_calibrated,
                            units=("pax/min, pax/m2" if is_calibrated
                                   else f"UNCALIBRATED: flow=relative, density={density_units}"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Never silent. This block indexes optional `extra`
                        # keys and reaches into model internals, so it is more
                        # likely to throw than most; swallowing that left the
                        # metric store permanently empty while the fusion
                        # engine reported "no data" and everything looked fine.
                        # Throttled to once per stage: it fires per frame.
                        if not _metric_warned["done"]:
                            _metric_warned["done"] = True
                            print(f"[jobs] WARN metric extraction failed for "
                                  f"{resolved_cam_id} ({exc.__class__.__name__}: {exc}); "
                                  f"cross-camera fusion will have no data from this "
                                  f"camera. Logged once per stage.")

            def on_progress(done, total, n_dets):
                stage.frames_done = done
                stage.frames_total = total
                stage.progress = (done / total) if total else 0.0
                stage.detections = n_dets

            detections = runner.run(job.video_path,
                                    progress_callback=on_progress,
                                    should_cancel=job._cancel.is_set,
                                    on_detections=on_alerts)

            # A run whose source died is NOT a completed run. Recorded on the
            # stage so the UI and history can show it, because "done" on a
            # truncated pass is the single most dangerous thing this system
            # can report: it means "monitored and clear" when the truth is
            # "stopped watching".
            status = getattr(runner, "source_status", None)
            if status is not None:
                stage.source_outcome = status.outcome
                stage.source_detail = status.describe()
                stage.frames_read = status.frames_read
                if not status.ok and status.outcome != "cancelled":
                    stage.degraded = True
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

        self._summarize(stage, detections, model)
        self._export(job, stage, detections, model_key, model)

        stage.status = "done"
        stage.progress = 1.0
        stage.finished_at = time.time()
        self._persist(job)

    def _run_stage_live(self, job: Job, model_key: str):
        """Live mode stage runner.

        Executes model with live streaming:
        1. Draws annotated frames with draw_frame (no lookahead smoothing).
        2. Encodes JPEG and broadcasts to LIVE_HUB for WebSocket clients.
        3. Paces execution to real-time wall-clock video rate.
        4. Dynamically skips frames if model inference falls behind real time.
        5. Does NOT write heavy video or CSV/JSON artifacts to disk.
        """
        import base64
        import math
        import cv2
        from pipeline.runner import PipelineRunner
        from webapp.registry import build_model
        from pipeline.annotate import draw_frame
        from webapp.live_hub import LIVE_HUB
        from pipeline.alert_sink import dispatch_detections
        from topology.graph import TOPOLOGY
        from topology.metric_store import METRIC_STORE

        stage = job.stages[model_key]
        stage.started_at = time.time()
        stage.status = "loading"
        job.message = f"Loading {model_key} (live)..."
        self._persist(job)

        try:
            model = build_model(
                model_key,
                job.device,
                video_name=job.video_name or "run",
                threshold=job.thresholds.get(model_key),
            )

            if getattr(model, "consumption_type", "") == "flow_pair" and job.video_path:
                from pipeline.video_meta import source_fps
                _src_fps = source_fps(job.video_path)
                stride = max(1, int(job.sample_every_n_frames or 1))
                model._fps = float(_src_fps)
                model._frame_stride = stride
                model.output_fps = float(_src_fps) / stride

            model.load()
            stage.status = "running"
            job.message = f"Live streaming {model_key} on {job.video_name}"
            self._persist(job)

            def _resolve_camera_id() -> str:
                v_name = (job.video_name or job.source or "").lower()
                for cid, node in TOPOLOGY.cameras.items():
                    if cid.lower() in v_name or node.name.lower() in v_name:
                        return cid
                if model_key in TOPOLOGY.cameras:
                    return model_key
                return job.video_name or model_key

            resolved_cam_id = _resolve_camera_id()
            cam_node = TOPOLOGY.get_camera(resolved_cam_id)
            cam_clock_offset = cam_node.clock_offset_sec if cam_node else 0.0

            from pipeline.video_meta import source_fps
            v_fps = float(source_fps(job.video_path) or 30.0)
            runner = PipelineRunner(models=[model], sample_every_n_frames=1)

            t_wall_start = time.time()
            last_frame_wall_time = time.time()
            fps_measured = 0.0
            positives_count = 0
            rolling_inf_times: list[float] = []

            def on_live_detections(dets, frame=None, frame_index=0, timestamp_sec=0.0):
                nonlocal last_frame_wall_time, fps_measured, positives_count
                call_start = time.time()

                dt = call_start - last_frame_wall_time
                last_frame_wall_time = call_start
                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps_measured = (
                        round(0.8 * fps_measured + 0.2 * instant_fps, 1)
                        if fps_measured > 0
                        else round(instant_fps, 1)
                    )

                person_dets = [d for d in dets if "person" in d.label or "head" in d.label]
                moving_dets = [d for d in person_dets if "moving" in d.label or "crush" in d.label]
                p_count = len(person_dets) if person_dets else (len(dets) if dets else 0)

                speed_samples = [
                    float(d.extra.get("speed_px_frame", 1.0))
                    for d in dets if isinstance(d.extra, dict) and "speed_px_frame" in d.extra
                ]
                avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 1.0
                _DISPLAY_SCALE = 12.0
                flow_rate = max(0.0, float(len(moving_dets) * max(1.0, avg_speed) * _DISPLAY_SCALE))

                crush_scores = [
                    float(d.extra.get("local_crush_risk", 0.0))
                    for d in dets if isinstance(d.extra, dict) and "local_crush_risk" in d.extra
                ]
                crush_risk = max(crush_scores) if crush_scores else 0.0

                headings = [
                    float(d.extra.get("heading_deg", 0.0))
                    for d in dets if isinstance(d.extra, dict) and "heading_deg" in d.extra
                ]
                if headings:
                    rads = [math.radians(h) for h in headings]
                    avg_cos = sum(math.cos(r) for r in rads) / len(rads)
                    avg_sin = sum(math.sin(r) for r in rads) / len(rads)
                    dom_vec = [round(avg_cos, 3), round(avg_sin, 3)]
                else:
                    dom_vec = [1.0, 0.0]

                density_val = float(p_count * 0.15)
                if hasattr(model, "_frame_density") and model._frame_density:
                    density_val = float(model._frame_density[-1])

                pos_count = sum(1 for d in dets if d.label in positive_labels())
                positives_count += pos_count
                stage.positives = positives_count
                stage.detections += len(dets)

                try:
                    is_calibrated = bool(
                        getattr(getattr(model, "_calib", None), "is_calibrated", False)
                        or getattr(model, "_calibrated", False)
                    )
                    METRIC_STORE.update(
                        camera_id=resolved_cam_id,
                        density=density_val,
                        flow_rate_pax_min=flow_rate,
                        dominant_direction_vector=tuple(dom_vec),
                        crush_risk_score=crush_risk,
                        person_count=p_count,
                        raw_timestamp_sec=timestamp_sec,
                        stream_start_epoch_ms=int(stage.started_at * 1000) if stage.started_at else None,
                        clock_offset_sec=cam_clock_offset,
                        flow_is_calibrated=is_calibrated,
                        density_is_calibrated=is_calibrated,
                    )
                except Exception:
                    pass

                if dets:
                    try:
                        dispatch_detections(
                            dets, camera_id=model_key, source=job.video_name or job.source
                        )
                    except Exception:
                        pass

                if frame is not None:
                    annotated = draw_frame(frame, dets)
                    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    jpeg_b64 = base64.b64encode(buf).decode("ascii") if ok else ""
                else:
                    jpeg_b64 = ""

                kpis = {
                    "person_count": p_count,
                    "flow_rate": round(flow_rate, 1),
                    "crush_risk": round(crush_risk, 3),
                    "density": round(density_val, 2),
                    "dominant_heading": dom_vec,
                    "positives": positives_count,
                    "total_detections": stage.detections,
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "fps": fps_measured,
                }

                LIVE_HUB.broadcast(job.id, {
                    "event": "frame",
                    "jpeg_b64": jpeg_b64,
                    "kpis": kpis,
                    "frame_index": frame_index,
                    "ts": round(timestamp_sec, 2),
                    "fps": fps_measured,
                    "model_key": model_key,
                })

                # Pacing: pace to actual wall-clock playback
                target_video_elapsed = timestamp_sec
                actual_wall_elapsed = time.time() - t_wall_start
                if actual_wall_elapsed < target_video_elapsed:
                    sleep_needed = target_video_elapsed - actual_wall_elapsed
                    time.sleep(sleep_needed)

                # Adaptive frame-skip governor
                step_cost = time.time() - call_start
                rolling_inf_times.append(step_cost)
                if len(rolling_inf_times) > 10:
                    rolling_inf_times.pop(0)
                avg_inf_cost = sum(rolling_inf_times) / len(rolling_inf_times)
                adaptive_stride = max(1, math.ceil(v_fps * avg_inf_cost))
                runner.sample_every_n_frames = adaptive_stride

            def on_progress(done, total, n_dets):
                stage.frames_done = done
                stage.frames_total = total
                stage.progress = (done / total) if total else 0.0

            detections = runner.run(
                job.video_path,
                progress_callback=on_progress,
                should_cancel=job._cancel.is_set,
                on_detections=on_live_detections,
            )

            status = getattr(runner, "source_status", None)
            if status is not None:
                stage.source_outcome = status.outcome
                stage.source_detail = status.describe()
                stage.frames_read = status.frames_read

        except Exception as e:
            stage.status = "failed"
            stage.error = f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
            LIVE_HUB.broadcast(job.id, {
                "event": "error",
                "job_id": job.id,
                "error": stage.error,
            })
            stage.finished_at = time.time()
            self._persist(job)
            return

        if job._cancel.is_set():
            stage.status = "cancelled"
            stage.finished_at = time.time()
            LIVE_HUB.broadcast(job.id, {
                "event": "done",
                "job_id": job.id,
                "model_key": model_key,
                "status": "cancelled",
            })
            self._persist(job)
            return

        stage.status = "done"
        stage.progress = 1.0
        stage.finished_at = time.time()
        self._summarize(stage, detections, model)
        LIVE_HUB.broadcast(job.id, {
            "event": "done",
            "job_id": job.id,
            "model_key": model_key,
            "status": "done",
        })
        self._persist(job)


    @staticmethod
    def _summarize(stage: Stage, detections: list, model=None):
        import json
        labels = Counter(d.label for d in detections)
        scoring = Counter(
            d.extra.get("scoring") for d in detections
            if isinstance(d.extra, dict) and d.extra.get("scoring")
        )
        stage.detections = len(detections)
        stage.label_counts = dict(labels)
        stage.scoring_modes = dict(scoring)
        positive = positive_labels()
        stage.positives = sum(n for lbl, n in labels.items() if lbl in positive)

        # Get summary from model instance if available, else compute from detections
        if model is not None and getattr(model, "summary", None):
            stage.summary = dict(model.summary)
        else:
            from webapp.history import compute_detections_summary
            stage.summary = compute_detections_summary(detections)

    def _export(self, job: Job, stage: Stage, detections: list, model_key: str,
                model=None):
        import json
        from pipeline.annotate import (export_annotated_video, export_detection_csv,
                                       export_detection_log)
        from pipeline.html_report import export_html_report

        video = os.path.splitext(os.path.basename(job.video_path))[0]
        out_dir = run_dir(video, model_key, create=True)

        export_detection_log(detections, os.path.join(out_dir, RUN_JSON))
        export_detection_csv(detections, os.path.join(out_dir, RUN_CSV))
        # Stage paths are relative to outputs/runs/ so the API can serve them
        # without the frontend needing to know the layout.
        stage.log_json = f"{video}/{model_key}/{RUN_JSON}"
        stage.log_csv = f"{video}/{model_key}/{RUN_CSV}"

        # Export summary.json
        summary_path = os.path.join(out_dir, RUN_SUMMARY)
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(stage.summary, f, indent=2)
            stage.log_summary = f"{video}/{model_key}/{RUN_SUMMARY}"
        except Exception as e:
            print(f"[WARN] Failed to write summary.json: {e}")

        # Export standalone report.html
        report_path = os.path.join(out_dir, RUN_REPORT)
        try:
            export_html_report(report_path, video, model_key, stage.summary, detections)
            stage.report_html = f"{video}/{model_key}/{RUN_REPORT}"
        except Exception as e:
            print(f"[WARN] Failed to write report.html: {e}")

        if job.export_video:
            job.message = f"Writing annotated video for {model_key}..."
            mp4_path = os.path.join(out_dir, RUN_VIDEO)

            # DenseFlowAnalyser (and any future flow model) writes its own
            # annotated video with the heatmap overlay during finalize().
            # Move it directly into the run directory instead of copying or
            # re-rendering plain bboxes on top.
            own_video = getattr(model, "annotated_video_path", None)
            if own_video and os.path.exists(own_video):
                # Keep the encoder's OWN extension.  _AnnotatedVideoWriter
                # falls back to MJPG/AVI when ffmpeg is missing, and forcing
                # that file to "annotated.mp4" produced an AVI wearing an mp4
                # name: the API then served it as video/mp4 and the browser
                # silently refused to play it, while the writer's "plays in
                # VLC, not a browser" warning was defeated by the rename.
                ext = os.path.splitext(own_video)[1].lower() or ".mp4"
                dest_name = os.path.splitext(RUN_VIDEO)[0] + ext
                dest = os.path.join(out_dir, dest_name)
                if os.path.abspath(own_video) != os.path.abspath(dest):
                    import shutil
                    shutil.move(own_video, dest)
            else:
                export_annotated_video(job.video_path, detections, mp4_path)
                dest_name = RUN_VIDEO
            stage.annotated = f"{video}/{model_key}/{dest_name}"

    def _finish(self, job: Job, status: str, message: str):
        job.status = status
        job.message = message
        job.finished_at = time.time()
        # Persist on every terminal transition: this is the record that has
        # to outlive the process.
        self._persist(job)


MANAGER = JobManager()
