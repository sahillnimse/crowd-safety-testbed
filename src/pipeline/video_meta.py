"""
Best-effort source-FPS probing for annotated-video pacing.

Flow-pair models write one output frame per SAMPLED frame and pace the
output at ``source_fps / stride``. Both callers (webapp/jobs.py and
scripts/run_single.py) need a source FPS they can trust, and
``cv2.VideoCapture.get(CAP_PROP_FPS)`` is not always one: some containers
(odd AVI variants, certain MKVs, raw streams) report 0 or NaN, which used
to silently skip the pacing correction and produce a stride-x sped-up
video.

Resolution order:
1. cv2's CAP_PROP_FPS, when positive and finite.
2. ffprobe ``nb_frames / duration`` — container-metadata arithmetic, no
   frame counting pass, so it is cheap.
3. ffprobe ``r_frame_rate`` / ``avg_frame_rate`` fraction.
4. ``default`` (25 fps, broadcast PAL and the most common surveillance
   rate) — pacing slightly off beats pacing absent.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess

logger = logging.getLogger(__name__)


def source_fps(video_path: str, default: float = 25.0) -> float:
    """Return a positive, finite FPS estimate for ``video_path``."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and math.isfinite(fps) and fps > 0:
        return float(fps)

    logger.warning(
        "cv2 reported FPS=%r for %s — falling back to ffprobe metadata.",
        fps, video_path,
    )
    probed = _ffprobe_fps(video_path)
    if probed:
        return probed

    logger.warning("ffprobe also failed for %s — using default %.1f fps.",
                   video_path, default)
    return float(default)


def _ffprobe_fps(video_path: str) -> float | None:
    """FPS from ffprobe metadata, or None when unavailable."""
    # find_binary("ffprobe") rather than deriving it from ffmpeg's directory:
    # that derivation missed every install where the two are not siblings --
    # notably the imageio_ffmpeg fallback, whose bundled directory holds
    # ffmpeg alone. The probe then returned None and the caller silently
    # dropped to the 25 fps default, reintroducing the mis-paced annotated
    # video this module exists to prevent. find_binary already searches the
    # env var, PATH, and the known package locations.
    from pipeline.ffmpeg import find_binary

    ffprobe = find_binary("ffprobe")
    if not ffprobe:
        return None

    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,duration,r_frame_rate,avg_frame_rate",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30,
        )
        stream = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
    except Exception as exc:  # noqa: BLE001 - any probe failure = no data
        logger.warning("ffprobe failed for %s: %s", video_path, exc)
        return None

    # Frame-count / duration is the most honest estimate: it reflects the
    # container's actual content rather than a declared rate.
    nb, dur = stream.get("nb_frames"), stream.get("duration")
    try:
        if nb and dur and float(dur) > 0:
            est = float(nb) / float(dur)
            if est > 0:
                return est
    except (TypeError, ValueError):
        pass

    for key in ("r_frame_rate", "avg_frame_rate"):
        rate = stream.get(key)
        try:
            if rate and "/" in rate:
                num, den = rate.split("/", 1)
                if float(den) > 0 and float(num) > 0:
                    return float(num) / float(den)
        except (TypeError, ValueError):
            continue
    return None
