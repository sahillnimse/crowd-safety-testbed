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
from webapp.jobs import ANNOTATED_DIR, LOG_DIR, MANAGER, PROJECT_ROOT

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
    pose_size: str = "s"


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
        pose_size=req.pose_size,
        local_path=local_path,
    )
    return job.to_dict()


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

    from webapp.jobs import POSITIVE_LABELS

    name = f"{os.path.basename(video)}_{os.path.basename(model_key)}.json"
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"No log named {name}")

    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if positives_only:
        rows = [r for r in rows if r.get("label") in POSITIVE_LABELS]
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

    with open(os.path.join(LOG_DIR, stage.log_json)) as f:
        rows = json.load(f)

    from webapp.jobs import POSITIVE_LABELS
    if positives_only:
        rows = [r for r in rows if r["label"] in POSITIVE_LABELS]

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

    if os.path.isdir(LOG_DIR):
        for name in os.listdir(LOG_DIR):
            if name.startswith(f"{video_stem}_"):
                path = os.path.join(LOG_DIR, name)
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1

    if os.path.isdir(ANNOTATED_DIR):
        for name in os.listdir(ANNOTATED_DIR):
            if name.startswith(f"{video_stem}_"):
                path = os.path.join(ANNOTATED_DIR, name)
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1

    anpr_folder = os.path.join(ANPR_DIR, video_stem)
    if os.path.isdir(anpr_folder):
        shutil.rmtree(anpr_folder, ignore_errors=True)
        removed += 1

    return {"ok": True, "removed": removed}


@app.delete("/api/history/{video}/{model_key}")
def delete_history_stage(video: str, model_key: str):
    """Delete a single model's saved outputs for a video."""
    video_stem = os.path.basename(video)
    model_stem = os.path.basename(model_key)
    stem = f"{video_stem}_{model_stem}"
    removed = 0

    for ext in (".json", ".csv"):
        path = os.path.join(LOG_DIR, f"{stem}{ext}")
        if os.path.isfile(path):
            os.remove(path)
            removed += 1

    mp4_path = os.path.join(ANNOTATED_DIR, f"{stem}.mp4")
    if os.path.isfile(mp4_path):
        os.remove(mp4_path)
        removed += 1

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
    for d in (LOG_DIR, ANNOTATED_DIR):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            path = os.path.join(d, name)
            if os.path.isfile(path):
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

    return {"removed": removed}


# Static frontend last, so it doesn't shadow /api routes.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

