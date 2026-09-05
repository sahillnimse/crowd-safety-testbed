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
import contextlib
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from typing import Literal

from topology.graph import TOPOLOGY
from topology.metric_store import METRIC_STORE
from topology.fusion_engine import FUSION_ENGINE
from webapp import registry
from webapp.live_hub import LIVE_HUB
from webapp.jobs import (ANNOTATED_DIR, LOG_DIR, MANAGER, PROJECT_ROOT,
                         RUNS_DIR, RUN_JSON, SESSIONS_DIR, run_dir)
from webapp.session_jobs import (SESSION_MANAGER, CameraSlotConfig,
                                 RouteSessionRequest, _slugify)

logger = logging.getLogger(__name__)

TEST_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "test_videos")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")

# Live camera schemes accepted as a job source. rtsp is what fixed cameras
# speak; the others are here because OpenCV opens them the same way.
LIVE_STREAM_SCHEMES = ("rtsp://", "rtmp://", "udp://")

# A local path submitted as a job source must resolve inside one of these.
# Without the check any readable path on the host could be opened and
# re-encoded into outputs/, which on a networked deployment is arbitrary
# file disclosure through a feature meant for picking test clips.
_ALLOWED_MEDIA_ROOTS = (
    os.path.join(PROJECT_ROOT, "test_videos"),
    os.path.join(PROJECT_ROOT, "outputs"),
)


def _within_allowed_roots(path: str) -> bool:
    """True if ``path`` sits inside a permitted media directory."""
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    for root in _ALLOWED_MEDIA_ROOTS:
        try:
            if os.path.commonpath([real, os.path.realpath(root)]) == os.path.realpath(root):
                return True
        except ValueError:
            continue          # different drive on Windows
    return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan manager: silences Windows disconnect noise and runs fusion engine."""
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

    # Capture event loop for thread-safe LiveStreamHub push
    LIVE_HUB.set_loop(loop)

    # Start fusion engine reasoning loop
    fusion_task = asyncio.create_task(FUSION_ENGINE.run_loop())
    try:
        yield
    finally:
        fusion_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fusion_task
        loop.set_exception_handler(previous)


app = FastAPI(title="Crowd Safety Testbed", version="1.0", lifespan=lifespan)


# ----------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------
# Every endpoint was previously unauthenticated. On a laptop that is fine;
# on a deployment reachable by anything other than localhost it means any
# client can start jobs, read footage, and DELETE every stored run
# (`DELETE /api/outputs` wipes outputs/ outright).
#
# Enabled by setting CROWD_API_TOKEN. Deliberately opt-in so existing local
# workflows are unchanged, but the server REFUSES to bind to a non-loopback
# address without it (see __main__), which is where the risk actually is.
_ENV_API_TOKEN = "CROWD_API_TOKEN"

# Open endpoints: the health probe (load balancers cannot carry a token) and
# the static frontend, which must load in order to prompt for one.
_PUBLIC_PATHS = ("/api/health", "/favicon.ico")


def _auth_token() -> str | None:
    return os.environ.get(_ENV_API_TOKEN) or None


@app.middleware("http")
async def require_token(request, call_next):
    """Bearer-token gate on /api/*, active only when CROWD_API_TOKEN is set."""
    token = _auth_token()
    path = request.url.path
    if token and path.startswith("/api") and path not in _PUBLIC_PATHS:
        supplied = request.headers.get("authorization", "")
        if supplied.lower().startswith("bearer "):
            supplied = supplied[7:]
        else:
            supplied = request.headers.get("x-api-token", "")
        # Constant-time compare: a plain != leaks the token prefix through
        # response timing to anyone able to make repeated requests.
        import hmac
        if not hmac.compare_digest(supplied, token):
            return JSONResponse(
                {"detail": "Missing or invalid API token."}, status_code=401)
    return await call_next(request)


class JobRequest(BaseModel):
    source: str = Field(..., description="YouTube URL, or a filename in test_videos/")
    models: list[str] = Field(..., min_length=1)
    sample_every_n_frames: int = 5
    device: str | None = None          # None = auto-detect
    export_video: bool = True
    mode: Literal["batch", "live"] = "batch"
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


# Upload ceiling. Generous for a test clip, finite so a stray multi-gigabyte
# file cannot fill the disk from a single unauthenticated request.
MAX_UPLOAD_BYTES = 2 * 1024 ** 3          # 2 GiB
_UPLOAD_CHUNK = 1024 * 1024


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    # A multipart part can legitimately carry no filename, and `None.lower()`
    # made that a 500 with a traceback instead of the 400 it is.
    if not file.filename:
        raise HTTPException(400, "Upload is missing a filename.")
    name = os.path.basename(file.filename)
    if not name.lower().endswith(VIDEO_EXTS):
        raise HTTPException(400, f"Unsupported file type: {name}")

    os.makedirs(TEST_VIDEOS_DIR, exist_ok=True)
    dest = os.path.join(TEST_VIDEOS_DIR, name)

    # Copied in bounded chunks rather than shutil.copyfileobj so the size is
    # enforced DURING the write. Checking afterwards means the oversize file is
    # already on disk, which is the thing the limit exists to prevent. A partial
    # write is removed so a rejected upload cannot leave a truncated video that
    # later looks like a real, openable clip.
    written = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"Upload exceeds the {MAX_UPLOAD_BYTES // 1024 ** 3} GiB limit.",
                    )
                f.write(chunk)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(dest)
        raise

    return {"name": name,
            "size_mb": round(written / (1024 ** 2), 1)}


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
    src_lower = req.source.lower()
    candidate = os.path.join(TEST_VIDEOS_DIR, os.path.basename(req.source))

    if src_lower.startswith(LIVE_STREAM_SCHEMES):
        # Live camera. Passed through as the "local path" so _prepare_video
        # does not try to download it: OpenCV opens an RTSP URL directly, and
        # the runner now detects stream sources and reconnects on a drop
        # instead of reporting the dead camera as a finished run.
        local_path = req.source
    elif os.path.exists(candidate) and src_lower.endswith(VIDEO_EXTS):
        local_path = candidate
    elif os.path.exists(req.source):
        # Restricted to the media directories. This previously accepted ANY
        # path the server process could stat, so a request could name
        # /etc/passwd or a private video elsewhere on the box and have it
        # opened and re-encoded into outputs/. On a networked deployment that
        # is arbitrary local file disclosure.
        resolved = os.path.abspath(req.source)
        if not _within_allowed_roots(resolved):
            raise HTTPException(
                403,
                "Path is outside the permitted media directories "
                "(test_videos/, outputs/).",
            )
        local_path = resolved
    elif not src_lower.startswith(("http://", "https://")):
        raise HTTPException(
            400,
            f"'{req.source}' is neither a URL, a live stream, nor a file "
            f"in test_videos/.",
        )

    job = MANAGER.create(
        source=req.source,
        model_keys=req.models,
        sample_every_n_frames=req.sample_every_n_frames,
        device=req.device or None,
        export_video=req.export_video,
        local_path=local_path,
        thresholds=thresholds,
        mode=req.mode,
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


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a finished job from memory and disk."""
    if not MANAGER.delete(job_id):
        raise HTTPException(400, "Job cannot be deleted (active, running, or not found).")
    return {"ok": True}


@app.delete("/api/jobs")
def clear_all_jobs():
    """Clear all completed, failed, and cancelled jobs."""
    count = MANAGER.clear_finished()
    return {"ok": True, "removed": count}


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

    log_path = os.path.join(RUNS_DIR, stage.log_json)
    if not os.path.exists(log_path):
        raise HTTPException(404, f"Detection log '{stage.log_json}' not found on disk (run outputs may have been deleted).")

    try:
        with open(log_path, encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as exc:
        raise HTTPException(500, f"Error reading detection log: {exc}")

    from webapp.jobs import positive_labels
    if positives_only:
        positive = positive_labels()
        rows = [r for r in rows if r.get("label") in positive]

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
    # .avi is declared honestly rather than as video/mp4: the annotated
    # writer falls back to MJPG/AVI without ffmpeg, and mislabelling that as
    # mp4 makes the browser fail silently instead of offering a download.
    media = ("video/mp4" if path.lower().endswith(".mp4")
             else "video/x-msvideo" if path.lower().endswith(".avi")
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
    failed = 0
    for d in (LOG_DIR, ANNOTATED_DIR, RUNS_DIR):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            path = os.path.join(d, name)
            # runs/ holds a directory per video; logs/ and annotated/ hold
            # loose files from before the restructure. Handle both. A delete
            # that fails (locked by a player/AV scan, read-only flag) is
            # reported, not swallowed — the endpoint's whole job is deletion,
            # so "removed: N" with files left behind is a lie an operator
            # would then debug from the UI alone.
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
                else:
                    continue
                removed += 1
            except OSError as exc:
                failed += 1
                logger.warning("Could not delete %s: %s", path, exc)

    if os.path.isdir(ANPR_DIR):
        for name in os.listdir(ANPR_DIR):
            path = os.path.join(ANPR_DIR, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
                else:
                    continue
                removed += 1
            except OSError as exc:
                failed += 1
                logger.warning("Could not delete %s: %s", path, exc)

    if failed:
        logger.warning(
            "clear_outputs finished with %d of %d deletions FAILED "
            "(locked or undeletable paths remain on disk).",
            failed, removed + failed,
        )

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

    # Clear finished/inactive jobs from memory and outputs/state
    n_jobs = MANAGER.clear_finished()
    if n_jobs > 0:
        removed += n_jobs

    return {"removed": removed, "failed": failed}


# ----------------------------------------------------------------------
# Topology & Multi-Camera Fusion Endpoints
# ----------------------------------------------------------------------

@app.get("/api/topology")
def get_topology():
    """Return camera topology graph as JSON for frontend rendering."""
    return TOPOLOGY.to_dict()


@app.post("/api/topology")
def update_topology(data: dict):
    """
    Admin endpoint to update camera topology graph.
    Protected by CROWD_API_TOKEN middleware when deployed beyond localhost.
    """
    try:
        TOPOLOGY.update_from_dict(data)
        return {"ok": True, "topology": TOPOLOGY.to_dict()}
    except Exception as exc:
        raise HTTPException(400, f"Invalid topology format: {exc}")


@app.post("/api/topology/from-route")
def create_topology_from_route(data: dict):
    """
    Derive, validate, and persist a camera topology graph from an explicit route definition.
    Writes to configs/topology.generated.yaml, keeping configs/topology.yaml intact.
    Protected by CROWD_API_TOKEN middleware when deployed beyond localhost.
    """
    import yaml
    from topology.graph import build_topology_from_route, GENERATED_TOPOLOGY_PATH
    cameras = data.get("cameras", [])
    edges = data.get("edges", [])
    defaults = data.get("defaults")
    try:
        topo_dict = build_topology_from_route(cameras, edges, defaults=defaults)

        header = (
            "# topology.generated.yaml — Auto-generated by Route Builder\n"
            f"# Generated timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            "# Hand-authored deployment configuration remains intact at configs/topology.yaml.\n"
            "# To revert to the deployment baseline, click 'Reset to Baseline' in the UI or call POST /api/topology/reset.\n\n"
        )
        with open(GENERATED_TOPOLOGY_PATH, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(topo_dict, f, sort_keys=False)

        TOPOLOGY.config_path = GENERATED_TOPOLOGY_PATH
        TOPOLOGY.update_from_dict(topo_dict)
        return {"ok": True, "topology": TOPOLOGY.to_dict()}
    except ValueError as val_err:
        raise HTTPException(400, str(val_err))
    except Exception as exc:
        raise HTTPException(400, f"Failed to build topology: {exc}")


@app.post("/api/topology/reset")
def reset_topology_to_baseline():
    """
    Revert active topology back to the hand-authored configs/topology.yaml.
    Deletes configs/topology.generated.yaml if present.
    """
    try:
        TOPOLOGY.reset_to_default()
        return {"ok": True, "topology": TOPOLOGY.to_dict()}
    except Exception as exc:
        raise HTTPException(500, f"Failed to reset topology: {exc}")


@app.get("/api/fusion/alerts")
def get_fusion_alerts(active: bool = True):
    """Return current cross-camera fusion alerts."""
    alerts = FUSION_ENGINE.get_active_alerts() if active else FUSION_ENGINE.get_all_alerts()
    return {"alerts": [a.to_dict() for a in alerts]}


@app.get("/api/fusion/metrics")
def get_fusion_metrics():
    """Return current live metrics snapshot and predicted inflows for all cameras."""
    cams = TOPOLOGY.all_cameras()
    return {
        "cameras": {
            c.id: {
                "name": c.name,
                "capacity": c.corridor_capacity_pax_min,
                "snapshot": (METRIC_STORE.get_latest(c.id).to_dict()
                             if METRIC_STORE.get_latest(c.id) else None),
                "is_stale": METRIC_STORE.is_stale(c.id, TOPOLOGY.staleness_threshold_sec),
                # null, not 0, when no forecast could be made.
                "predicted_inflow": FUSION_ENGINE.get_predicted_inflow(c.id),
                # Whether that forecast used ALL upstream sources. A partial
                # forecast under-estimates inflow and so suppresses alerts;
                # the UI has to be able to distinguish "quiet" from "blind".
                "forecast_status": FUSION_ENGINE.get_forecast_status(c.id),
            }
            for c in cams
        }
    }


@app.get("/api/fusion/sparklines")
def get_fusion_sparklines(window_sec: float = 300.0):
    """Return historical time-series metric data per camera for sparkline rendering."""
    cams = TOPOLOGY.all_cameras()
    return {
        "sparklines": {
            c.id: [s.to_dict() for s in METRIC_STORE.get_history(c.id, window_sec=window_sec)]
            for c in cams
        }
    }


# ----------------------------------------------------------------------
# Multi-Camera Route Sessions
# ----------------------------------------------------------------------

class CameraSlotIn(BaseModel):
    camera_id: str
    video_source: str
    camera_name: str = ""
    include_in_session: bool = True


class RouteSessionIn(BaseModel):
    session_name: str
    slots: list[CameraSlotIn]
    models: list[str] = Field(default_factory=lambda: ["crowd_motion_monitor"])
    sample_every_n_frames: int = 5
    device: str | None = None
    export_video: bool = True
    threshold: float | None = None


@app.post("/api/sessions")
def create_route_session(req: RouteSessionIn):
    """Start an isolated multi-camera route session."""
    try:
        req_obj = RouteSessionRequest(
            session_name=req.session_name,
            slots=[
                CameraSlotConfig(
                    camera_id=s.camera_id,
                    video_source=s.video_source,
                    camera_name=s.camera_name,
                    include_in_session=s.include_in_session,
                )
                for s in req.slots
            ],
            models=req.models,
            sample_every_n_frames=req.sample_every_n_frames,
            device=req.device,
            export_video=req.export_video,
            threshold=req.threshold,
        )
        session = SESSION_MANAGER.start_session(req_obj)
        return session.to_manifest()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/sessions")
def list_route_sessions():
    """List all saved and active route sessions."""
    return {"sessions": SESSION_MANAGER.list_sessions()}


@app.get("/api/sessions/{name}")
def get_route_session(name: str):
    """Fetch details and camera runs for a specific session."""
    sess = SESSION_MANAGER.get_session(name)
    if not sess:
        raise HTTPException(404, f"Route session '{name}' not found.")
    return sess


@app.delete("/api/sessions/{name}")
def delete_route_session(name: str):
    """Delete a route session and all its outputs from disk."""
    ok = SESSION_MANAGER.delete_session(name)
    if not ok:
        raise HTTPException(404, f"Route session '{name}' not found.")
    return {"ok": True}


@app.get("/api/sessions/{name}/detections/{cam_id}")
def get_session_camera_detections(name: str, cam_id: str, limit: int = 500, positives_only: bool = False):
    """Detection rows for a specific camera in a route session."""
    safe_name = _slugify(name)
    path = os.path.join(SESSIONS_DIR, safe_name, cam_id, "detections.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"No detections found for camera '{cam_id}' in session '{name}'.")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if positives_only:
        from webapp.jobs import positive_labels
        pos = positive_labels()
        rows = [r for r in rows if r.get("label") in pos]
    return {"total": len(rows), "rows": rows[:limit]}


@app.get("/api/files/session/{name}/{path:path}")
def get_session_file(name: str, path: str):
    """Serve isolated session reports, annotated videos, or detection files."""
    safe_name = _slugify(name)
    target = os.path.realpath(os.path.join(SESSIONS_DIR, safe_name, path))
    if not target.startswith(os.path.realpath(SESSIONS_DIR)):
        raise HTTPException(403, "Access outside sessions directory forbidden.")
    if not os.path.exists(target):
        if path.endswith("report.html"):
            target_dir = os.path.dirname(target)
            sum_path = os.path.join(target_dir, "summary.json")
            if os.path.isfile(sum_path):
                try:
                    from pipeline.html_report import export_html_report
                    export_html_report(target, os.path.basename(target_dir), "crowd_motion_monitor", {})
                except Exception:
                    pass
        if not os.path.exists(target):
            raise HTTPException(404, f"File '{path}' not found in session '{name}'.")
    return FileResponse(target)


async def _ws_authorised(websocket: WebSocket) -> bool:
    """
    Token check for WebSocket routes.

    The HTTP middleware above only gates paths starting with "/api", so
    "/ws/fusion" bypassed authentication completely — it streamed live camera
    telemetry to any client that could reach the port, even with
    CROWD_API_TOKEN set. Websockets do not go through that middleware, so the
    check has to be repeated here.

    Browsers cannot set headers on a WebSocket handshake, so the token is
    accepted from the query string as well. That is a real trade-off: query
    strings turn up in proxy and server logs in a way headers do not. Prefer
    the header where the client allows it; the query parameter exists so the
    browser UI can connect at all.
    """
    token = _auth_token()
    if not token:
        return True                      # auth disabled, same as HTTP
    supplied = websocket.query_params.get("token", "")
    if not supplied:
        header = websocket.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else \
            websocket.headers.get("x-api-token", "")
    import hmac
    return hmac.compare_digest(supplied, token)


@app.websocket("/ws/fusion")
async def ws_fusion(websocket: WebSocket):
    """WebSocket stream pushing real-time fusion alerts and predicted inflows."""
    if not await _ws_authorised(websocket):
        # 1008 = policy violation. Closed BEFORE accept, so an unauthorised
        # client never receives a single telemetry frame.
        await websocket.close(code=1008)
        return
    await websocket.accept()
    q = asyncio.Queue()
    FUSION_ENGINE.register_subscriber(q)
    try:
        # Push initial snapshot on connect
        active_alerts = [a.to_dict() for a in FUSION_ENGINE.get_active_alerts()]
        inflows = {c.id: FUSION_ENGINE.get_predicted_inflow(c.id) for c in TOPOLOGY.all_cameras()}
        await websocket.send_json({
            "event": "init",
            "topology": TOPOLOGY.to_dict(),
            "alerts": active_alerts,
            "predicted_inflows": inflows,
        })
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket client disconnected: %s", exc)
    finally:
        FUSION_ENGINE.unregister_subscriber(q)


@app.websocket("/ws/live/{job_id}")
async def ws_live(websocket: WebSocket, job_id: str):
    """WebSocket stream pushing real-time annotated frames and KPIs for a live preview job."""
    if not await _ws_authorised(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()

    if LIVE_HUB._loop is None or LIVE_HUB._loop.is_closed():
        LIVE_HUB.set_loop(asyncio.get_running_loop())

    job = MANAGER.get(job_id)
    if job is None:
        await websocket.send_json({"event": "error", "error": f"Job {job_id} not found."})
        await websocket.close(code=1000)
        return

    q = asyncio.Queue(maxsize=5)
    LIVE_HUB.register(job_id, q)
    try:
        await websocket.send_json({
            "event": "init",
            "job_id": job_id,
            "status": job.status,
            "message": job.message,
            "video_name": job.video_name or "",
        })
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
            if msg.get("event") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("Live WebSocket client disconnected for job %s: %s", job_id, exc)
    finally:
        LIVE_HUB.unregister(job_id, q)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav_path = os.path.join(FRONTEND_DIR, "favicon.ico")
    if os.path.exists(fav_path):
        return FileResponse(fav_path)
    from fastapi.responses import Response
    return Response(status_code=204)


# Static frontend last, so it doesn't shadow /api routes.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

