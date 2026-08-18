"""
FastAPI backend for the crowd-safety testbed UI.

Run it:
    python -m webapp            (or: uvicorn webapp.app:app --reload)

Then open http://127.0.0.1:8000

Endpoints are deliberately thin — all the work lives in webapp/jobs.py and
webapp/registry.py. Torch is never imported at startup, so the page loads
instantly and the first model import happens inside a job thread.
"""

import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webapp import registry
from webapp.jobs import (ANNOTATED_DIR, LOG_DIR, MANAGER, PROJECT_ROOT,
                         RUNS_DIR, RUN_JSON, run_dir)

TEST_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "test_videos")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Silence the benign disconnect spam Windows' Proactor loop emits.

    Closing the video preview mid-download, reloading the page, or switching
    tabs aborts an in-flight response. On Windows asyncio then logs a full
    ConnectionResetError traceback from _call_connection_lost even though
    the request is already finished and nothing failed.

    That matters here beyond tidiness: per-frame model errors are reported
    only on this terminal, so burying it in tracebacks for events that are
    not errors actively hides the ones that are. Only ConnectionResetError
    and friends are dropped; every other loop exception still surfaces.
    """
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def handler(active_loop, context):
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        if previous is not None:
            previous(active_loop, context)
        else:
            active_loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    yield
    loop.set_exception_handler(previous)


app = FastAPI(title="Crowd Safety Testbed", version="1.0", lifespan=lifespan)


class JobRequest(BaseModel):
    source: str = Field(..., description="YouTube URL, or a filename in test_videos/")
    models: list[str] = Field(..., min_length=1)
    sample_every_n_frames: int = 5
    device: str | None = None          # None = auto-detect
    export_video: bool = True
    # One confidence threshold applied to every selected model.  Per-model
    # values were a false affordance: comparing detectors is only meaningful
    # when they are all judged at the same operating point, and a grid of
    # sliders invited tuning each one until it looked good, which is how a
    # benchmark stops measuring anything.  Models whose scores are not
    # comparable to the rest (SSD runs lower) keep their own default and are
    # documented as such rather than silently rescaled.
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Global confidence threshold for all selected models.",
    )


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/models")
def get_models():
    return {
        "categories": registry.CATEGORY_LABELS,
        "models": registry.list_models(),
    }


@app.get("/api/device")
def get_device():
    """Report GPU availability so the UI can warn before a job OOMs."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "cuda": True,
                "name": props.name,
                "total_gb": round(props.total_memory / (1024 ** 3), 1),
                "torch": torch.__version__,
            }
        return {"cuda": False, "name": None, "torch": torch.__version__}
    except Exception as e:  # noqa: BLE001
        return {"cuda": False, "name": None, "error": str(e)}


@app.get("/api/videos")
def list_videos():
    """Local clips already in test_videos/, so a re-run needs no download."""
    if not os.path.isdir(TEST_VIDEOS_DIR):
        return {"videos": []}
    out = []
    for name in sorted(os.listdir(TEST_VIDEOS_DIR)):
        if not name.lower().endswith(VIDEO_EXTS):
            continue
        path = os.path.join(TEST_VIDEOS_DIR, name)
        out.append({
            "name": name,
            "size_mb": round(os.path.getsize(path) / (1024 ** 2), 1),
        })
    return {"videos": out}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(VIDEO_EXTS):
        raise HTTPException(400, f"Unsupported file type: {file.filename}")
    os.makedirs(TEST_VIDEOS_DIR, exist_ok=True)
    dest = os.path.join(TEST_VIDEOS_DIR, os.path.basename(file.filename))
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"name": os.path.basename(dest),
            "size_mb": round(os.path.getsize(dest) / (1024 ** 2), 1)}


@app.post("/api/jobs")
def create_job(req: JobRequest):
    unknown = [m for m in req.models if m not in registry.BY_KEY]
    if unknown:
        raise HTTPException(400, f"Unknown model(s): {', '.join(unknown)}")

    # Fan the single global threshold out to every selected model that has a
    # threshold at all.  Models with default_threshold=None are classical-CV
    # detectors with no confidence score to compare against, so applying one
    # to them would be meaningless rather than merely unused.
    #
    # Two kinds of model are left out.  default_threshold=None means a
    # classical-CV detector with no confidence score at all, so a threshold
    # would be meaningless rather than merely unused.  comparable_threshold
    # =False means the scores exist but are not on the same scale as the rest
    # (SSDLite runs lower); handing it the same number is a handicap, not a
    # fair operating point, so it keeps its own documented default.
    thresholds: dict[str, float] = {}
    if req.threshold is not None:
        thresholds = {
            key: req.threshold for key in req.models
            if registry.BY_KEY[key].default_threshold is not None
            and registry.BY_KEY[key].comparable_threshold
        }

    # A local filename in test_videos/ skips the download path entirely.
    local_path = None
    candidate = os.path.join(TEST_VIDEOS_DIR, os.path.basename(req.source))
    if os.path.exists(candidate) and req.source.lower().endswith(VIDEO_EXTS):
        local_path = candidate
    elif os.path.exists(req.source):
        local_path = os.path.abspath(req.source)
    elif not req.source.lower().startswith(("http://", "https://")):
        raise HTTPException(
            400,
            f"'{req.source}' is neither a URL nor a file in test_videos/.",
        )

    job = MANAGER.create(
        source=req.source,
        model_keys=req.models,
        sample_every_n_frames=req.sample_every_n_frames,
        device=req.device or None,
        export_video=req.export_video,
        local_path=local_path,
        thresholds=thresholds,
    )
    return job.to_dict()


class ValidationRequest(BaseModel):
    source: str = Field(..., description="Filename in test_videos/, or a path")
    routes: str = "abc"
    # Budget, spread as a stride across the WHOLE video. 0 = every frame.
    max_frames: int = 240


@app.get("/api/validation/flow")
def get_flow_validation():
    """Latest dense-flow validation report, plus run state."""
    from webapp.validation import RUNNER
    return RUNNER.state()


@app.post("/api/validation/flow")
def run_flow_validation(req: ValidationRequest):
    """
    Start a validation run.

    The source is resolved against test_videos/ the same way job sources are,
    so the UI can pass the filename the user already picked.
    """
    from webapp.validation import RUNNER

    path = req.source
    if not os.path.isabs(path):
        candidate = os.path.join(TEST_VIDEOS_DIR, os.path.basename(path))
        if os.path.exists(candidate):
            path = candidate
    if not os.path.exists(path):
        raise HTTPException(400, f"Video not found: {req.source}")

    bad = set(req.routes.lower()) - set("abc")
    if bad:
        raise HTTPException(400, f"Unknown route(s): {''.join(sorted(bad))}")

    accepted, message = RUNNER.start(path, req.routes.lower(), req.max_frames)
    if not accepted:
        raise HTTPException(409, message)
    return {"ok": True, "message": message}


@app.delete("/api/validation/flow")
def delete_flow_validation():
    """Delete the stored validation report and its comparison video."""
    from webapp.validation import RUNNER
    removed, message = RUNNER.clear()
    if removed < 0:
        raise HTTPException(409, message)
    return {"removed": removed, "message": message}


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": MANAGER.list()}


@app.get("/api/history")
def list_history():
    """Completed runs reconstructed from outputs/ on disk.

    Survives server restarts and includes runs launched from the CLI —
    neither of which the in-memory job list can show.
    """
    from webapp import history
    return {"history": history.scan()}


@app.get("/api/history/{video}/{model_key}/detections")
def history_detections(video: str, model_key: str, limit: int = 500,
                       positives_only: bool = True):
    """Detection rows for a past run, read straight off disk."""
    import json

    from webapp.jobs import positive_labels

    path = os.path.join(
        run_dir(os.path.basename(video), os.path.basename(model_key)), RUN_JSON)
    if not os.path.exists(path):
        raise HTTPException(404, f"No run for {video} / {model_key}")

    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if positives_only:
        positive = positive_labels()
        rows = [r for r in rows if r.get("label") in positive]
    return {"total": len(rows), "rows": rows[:limit]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not MANAGER.cancel(job_id):
        raise HTTPException(409, "Job is not cancellable (already finished?)")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/detections/{model_key}")
def get_detections(job_id: str, model_key: str, limit: int = 500,
                   positives_only: bool = True):
    """Detection rows for the results table, newest-first is not useful here
    so they stay in timestamp order."""
    import json

    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    stage = job.stages.get(model_key)
    if stage is None or not stage.log_json:
        raise HTTPException(404, "No results for that model yet")

    with open(os.path.join(RUNS_DIR, stage.log_json), encoding="utf-8") as f:
        rows = json.load(f)

    from webapp.jobs import positive_labels
    if positives_only:
        positive = positive_labels()
        rows = [r for r in rows if r["label"] in positive]

    return {"total": len(rows), "rows": rows[:limit]}


ANPR_DIR = os.path.join(PROJECT_ROOT, "outputs", "anpr")


@app.get("/api/anpr")
def list_anpr_galleries():
    """Every ANPR capture on disk, newest first."""
    if not os.path.isdir(ANPR_DIR):
        return {"galleries": []}

    out = []
    for name in os.listdir(ANPR_DIR):
        manifest = os.path.join(ANPR_DIR, name, "manifest.json")
        if not os.path.exists(manifest):
            continue
        try:
            with open(manifest, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "video": name,
            "counts": data.get("counts", {}),
            "vehicles": data.get("vehicles", []),
            "modified_at": os.path.getmtime(manifest),
        })
    out.sort(key=lambda g: g["modified_at"], reverse=True)
    return {"galleries": out}


@app.get("/api/anpr/{video}/{kind}/{name}")
def anpr_image(video: str, kind: str, name: str):
    """Serve a captured vehicle or plate crop."""
    if kind not in ("vehicles", "plates"):
        raise HTTPException(400, "kind must be 'vehicles' or 'plates'")
    path = os.path.join(ANPR_DIR, os.path.basename(video), kind,
                        os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "No such image")
    return FileResponse(path, media_type="image/jpeg")


# `name:path` so the parameter matches "<video>/<file>".  A plain {name}
# stops at the first slash, so reports stored per video 404'd — the report
# named the right file and the route could not reach it.
@app.get("/api/files/validation/{name:path}")
def stream_validation_file(name: str):
    """
    Serve a file produced by a validation run (e.g. the comparison video).

    ``name`` is "<video>/<file>" now that reports are stored per source; a
    bare filename is still accepted for reports written before that split.
    Every component is reduced to a basename before use — these are URL
    segments, and joining them raw would let "../.." walk out of outputs/.
    """
    from webapp.validation import REPORT_DIR
    parts = [os.path.basename(p) for p in name.replace("\\", "/").split("/") if p]
    path = os.path.join(REPORT_DIR, *parts[-2:]) if parts else REPORT_DIR
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    media = "video/mp4" if path.lower().endswith(".mp4") else "application/json"
    return FileResponse(path, media_type=media)


@app.get("/api/files/run/{video}/{model_key}/{name}")
def stream_run_file(video: str, model_key: str, name: str):
    """
    Serve one artifact from outputs/runs/<video>/<model>/.

    Every path component is reduced to a basename before use: these are URL
    segments, and joining them raw would let "../.." walk out of outputs/.
    """
    path = os.path.join(
        run_dir(os.path.basename(video), os.path.basename(model_key)),
        os.path.basename(name),
    )
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    media = ("video/mp4" if path.lower().endswith(".mp4")
             else "text/html" if path.lower().endswith(".html")
             else "text/csv" if path.lower().endswith(".csv")
             else "application/json")
    return FileResponse(path, media_type=media)


@app.get("/api/files/logs/{name}")
def download_log(name: str):
    path = os.path.join(LOG_DIR, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "No such log")
    return FileResponse(path, filename=os.path.basename(path))


@app.get("/api/files/annotated/{name}")
def stream_annotated(name: str):
    path = os.path.join(ANNOTATED_DIR, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "No such video")
    return FileResponse(path, media_type="video/mp4")


@app.delete("/api/history/{video}")
def delete_history_video(video: str):
    """Delete output logs, annotated videos, and ANPR gallery for a specific video."""
    video_stem = os.path.basename(video)
    removed = 0

    # One directory holds everything this video produced, so the prefix
    # matching the old flat layout needed — which also deleted "clip_2" when
    # asked to delete "clip" — is gone.
    video_runs = os.path.join(RUNS_DIR, video_stem)
    if os.path.isdir(video_runs):
        shutil.rmtree(video_runs, ignore_errors=True)
        removed += 1

    anpr_folder = os.path.join(ANPR_DIR, video_stem)
    if os.path.isdir(anpr_folder):
        shutil.rmtree(anpr_folder, ignore_errors=True)
        removed += 1

    return {"ok": True, "removed": removed}


@app.delete("/api/history/{video}/{model_key}")
def delete_history_stage(video: str, model_key: str):
    """Delete a single model's saved outputs for a video."""
    target = run_dir(os.path.basename(video), os.path.basename(model_key))
    removed = 0
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
        removed += 1
        # Drop the video directory too once its last model run is gone.
        parent = os.path.dirname(target)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)

    return {"ok": True, "removed": removed}



@app.delete("/api/anpr/{video}")
def delete_anpr_gallery(video: str):
    """Delete ANPR gallery for a specific video."""
    video_stem = os.path.basename(video)
    anpr_folder = os.path.join(ANPR_DIR, video_stem)
    if os.path.isdir(anpr_folder):
        shutil.rmtree(anpr_folder, ignore_errors=True)
        return {"ok": True, "removed": 1}
    return {"ok": True, "removed": 0}


@app.delete("/api/outputs")
def clear_outputs():
    """Wipe generated artifacts. Only ever touches outputs/, never the
    source videos in test_videos/."""
    removed = 0
    for d in (LOG_DIR, ANNOTATED_DIR, RUNS_DIR):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            path = os.path.join(d, name)
            # runs/ holds a directory per video; logs/ and annotated/ hold
            # loose files from before the restructure. Handle both.
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            elif os.path.isfile(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass

    if os.path.isdir(ANPR_DIR):
        for name in os.listdir(ANPR_DIR):
            path = os.path.join(ANPR_DIR, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            elif os.path.isfile(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass

    # Validation output is a generated artifact under outputs/ too, so a
    # button promising to wipe generated artifacts has to include it.  Going
    # through the runner rather than deleting the files directly also clears
    # its in-memory report, which would otherwise keep serving a result whose
    # comparison video no longer exists.  Refused mid-run, and that refusal is
    # not an error for this endpoint — the rest of the wipe still happened.
    from webapp.validation import RUNNER
    n_val, _ = RUNNER.clear()
    if n_val > 0:
        removed += n_val

    return {"removed": removed}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav_path = os.path.join(FRONTEND_DIR, "favicon.ico")
    if os.path.exists(fav_path):
        return FileResponse(fav_path)
    from fastapi.responses import Response
    return Response(status_code=204)


# Static frontend last, so it doesn't shadow /api routes.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

