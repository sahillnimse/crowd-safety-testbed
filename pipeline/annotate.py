"""
Burns detections onto video frames and exports an annotated video for
visual QA — usually the fastest way to sanity-check model behavior.

Also exports structured logs (JSON) for metrics/analysis.

**Encoding note.** `cv2.VideoWriter` with the `mp4v` fourcc produces
MPEG-4 Part 2, which no browser can play in an HTML5 <video> element —
the file opens, reports a duration, and renders nothing. Since the web UI
plays these back in the browser, output is encoded as H.264 (yuv420p,
+faststart) by piping frames to ffmpeg. OpenCV cannot do this itself on a
stock Windows install: its H.264 path needs the OpenH264 DLL, which isn't
bundled, so requesting `avc1` just fails and silently falls back to mp4v.

If ffmpeg genuinely isn't present, we still write the mp4v file rather
than losing the export entirely, but warn loudly that it won't play in a
browser.
"""

import csv
import json
import subprocess
from collections import Counter, defaultdict

import cv2

from models.base import Detection
from pipeline.ffmpeg import find_ffmpeg

COLOR_MAP = {
    "fire": (0, 69, 255),
    "smoke": (128, 128, 128),
    "fall": (0, 0, 255),
    "standing": (0, 200, 0),
    "turbulence": (0, 165, 255),
    "convergence": (0, 0, 200),
    "crush_risk": (0, 0, 139),   # converging *and* turbulent — the worst case
    "violence": (0, 0, 255),
    "non_violence": (0, 200, 0),
    # Traffic: distinct colours so moving vs parked is readable at a glance
    # in the annotated video rather than both falling through to yellow.
    "vehicle_moving": (255, 191, 0),   # cyan-blue
    "vehicle_parked": (0, 140, 255),   # orange
    "vehicle_plate": (0, 215, 255),    # amber — ANPR, plate read
    "vehicle_unread": (120, 120, 120), # grey — ANPR, plate not legible
    "umbrella": (203, 65, 200),        # magenta
    # Dense optical flow crowd-safety alerts (DenseFlowAnalyser).
    # Colour is a rendering choice only; Detection.extra carries the numbers.
    #
    # These keys are "<metric>_<severity>" exactly as
    # Alert.label_for_detection() builds them — see
    # models.crowd_flow.zones.DENSE_FLOW_ALERT_LABELS, which is derived from
    # the threshold table and is the list to check against when adding a
    # metric.  Three keys here were names the engine never emits
    # ("counterflow_warning", "stop_go_warning", "vehicle_in_ped_zone", plus a
    # "mean_speed_critical" tier that does not exist), so those alerts drew in
    # DEFAULT_COLOR and the crowd_pressure ones were missing entirely.
    "mean_speed_warning":         (0, 165, 255),   # orange — speed drop
    "mean_divergence_critical":   (0, 0, 200),     # red — compression/crush risk
    "mean_curl_warning":          (0, 200, 255),   # yellow — rotational flow
    "counterflow_score_warning":  (0, 215, 255),   # amber — entry/exit separation
    "turbulence_index_critical":  (0, 0, 160),     # dark red — Helbing turbulence
    "crowd_pressure_warning":     (140, 0, 220),   # purple — Helbing pressure
    "crowd_pressure_critical":    (90, 0, 150),    # dark purple — stampede range
    "vehicle_in_ped_zone_warning": (0, 180, 0),    # green — vehicle alert
    # Per-frame summary row, not an alert.
    "flow_analysis":              (200, 200, 200), # grey
}
DEFAULT_COLOR = (255, 255, 0)


class _FFmpegH264Writer:
    """Frame sink that pipes raw BGR into ffmpeg and gets browser-playable H.264.

    Encoding straight from the pipe (rather than writing mp4v and
    transcoding afterwards) avoids a second full pass over the video and a
    generation of quality loss.
    """

    def __init__(self, ffmpeg: str, output_path: str, fps: float, w: int, h: int):
        # yuv420p requires even dimensions; odd-sized sources would make
        # ffmpeg abort rather than round for us.
        self.w, self.h = w, h
        scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{fps:.6f}",
            "-i", "-", "-an",
            "-vf", scale,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            # faststart moves the moov atom to the front so the browser can
            # begin playing before the whole file has downloaded.
            "-movflags", "+faststart",
            output_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        _, err = self.proc.communicate()
        if self.proc.returncode != 0:
            msg = err.decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg failed (exit {self.proc.returncode}): {msg}")


def _open_writer(output_path: str, fps: float, w: int, h: int):
    """H.264 via ffmpeg when available, else mp4v with a warning."""
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        return _FFmpegH264Writer(ffmpeg, output_path, fps, w, h), "h264"

    print("[annotate] WARNING: ffmpeg not found — falling back to mp4v "
          "(MPEG-4 Part 2). The file will play in VLC but NOT in a browser, "
          "so the web UI's video preview will appear blank. Install ffmpeg "
          "or set FFMPEG_BINARY to fix.")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return writer, "mp4v"


def _detection_stride(detections: list[Detection]) -> int:
    """Infer the runner's sample_every_n_frames from the detections themselves.

    The renderer isn't told the stride, but it's recoverable: it's the most
    common gap between consecutive frames that produced any detection.
    Deriving it means the smoothing adapts to whatever stride the run used
    instead of needing a matching argument threaded through the UI.
    """
    frames = sorted({d.frame_index for d in detections})
    if len(frames) < 2:
        return 1
    gaps = Counter(b - a for a, b in zip(frames, frames[1:]) if b > a)
    return max(1, gaps.most_common(1)[0][0]) if gaps else 1


def _track_key(d: Detection):
    """Identity a box can be followed by across frames, or None if untracked."""
    if isinstance(d.extra, dict):
        tid = d.extra.get("track_id")
        if tid is not None:
            return (d.model_name, tid)
    return None


def _smooth_boxes(boxes: list, window: int) -> list:
    """Centered moving average over a track's box coordinates.

    Detector output jitters by a few pixels frame to frame even on a
    perfectly still object, which reads as the box vibrating. A *centered*
    window is used rather than an exponential average because this runs
    offline over the whole track — there's no reason to accept the lag that
    a causal filter would introduce, which would make boxes trail behind
    moving vehicles.
    """
    if window < 2 or len(boxes) < 2:
        return boxes
    half = window // 2
    n = len(boxes)
    out = []
    for i in range(n):
        # Shrink the window symmetrically near the ends rather than
        # truncating it on one side. A one-sided window averages a moving
        # object's future (or past) positions into its current one, which
        # drags the first and last boxes of every track toward the middle of
        # its path — on a vehicle moving 4 px/frame that was a 20 px error,
        # worse than the jitter being removed.
        k = min(half, i, n - 1 - i)
        if k == 0:
            out.append(list(boxes[i]))
            continue
        chunk = boxes[i - k:i + k + 1]
        out.append([sum(c[j] for c in chunk) / len(chunk) for j in range(4)])
    return out


def _lerp(a, b, t):
    return [a[k] + (b[k] - a[k]) * t for k in range(4)]


def build_render_plan(detections: list[Detection], fps: float,
                      smooth_window: int = 5,
                      hold_seconds: float = 0.25) -> dict:
    """frame_index -> list of things to draw on that frame.

    Three problems this solves, all of which made boxes strobe:

    1. **Sampling gaps.** The runner only processes every Nth frame, so
       detections exist only on those frames. Drawing them as-is means a box
       is visible 1 frame in N — at the UI's default stride of 5 that's a
       6 Hz flash. Boxes are interpolated between consecutive samples of the
       same track, so they move smoothly through the frames in between.
    2. **Detector jitter.** Coordinates wobble a few pixels per detection
       even on a stationary object. A centered moving average removes it.
    3. **Dropouts.** A detector missing an object for one sample punches a
       hole in an otherwise continuous track. Gaps up to a few samples wide
       are interpolated across rather than left blank.

    Anything without a track_id (optical-flow cells, clip-level banners)
    can't be interpolated — there's no identity to interpolate along — so it
    is simply held on screen for `hold_seconds`.
    """
    plan: dict[int, list] = defaultdict(list)
    if not detections:
        return plan

    stride = _detection_stride(detections)
    hold = max(stride, int(round(fps * hold_seconds)))
    # Bridge a couple of missed samples, but not so far that a departed
    # object leaves a box hanging over empty road.
    max_gap = stride * 3

    tracked: dict[tuple, list] = defaultdict(list)
    loose: list[Detection] = []

    for d in detections:
        key = _track_key(d)
        if key is not None and d.bbox:
            tracked[key].append(d)
        else:
            loose.append(d)

    # ---- tracked boxes: smooth, then interpolate between samples ----
    for key, dets in tracked.items():
        dets.sort(key=lambda x: x.frame_index)
        boxes = _smooth_boxes([list(map(float, x.bbox)) for x in dets], smooth_window)

        for i, det in enumerate(dets):
            f0, b0 = det.frame_index, boxes[i]

            if i + 1 < len(dets):
                f1, b1 = dets[i + 1].frame_index, boxes[i + 1]
                span = f1 - f0
                if 0 < span <= max_gap:
                    for f in range(f0, f1):
                        t = (f - f0) / span
                        plan[f].append({
                            "bbox": _lerp(b0, b1, t),
                            "label": det.label,
                            # Interpolating confidence too stops the printed
                            # number from jumping every stride frames.
                            "confidence": det.confidence
                            + (dets[i + 1].confidence - det.confidence) * t,
                            "model_name": det.model_name,
                        })
                    continue

            # last sample of the track, or a gap too wide to bridge: hold
            for f in range(f0, f0 + hold):
                plan[f].append({
                    "bbox": b0,
                    "label": det.label,
                    "confidence": det.confidence,
                    "model_name": det.model_name,
                })

    # ---- untracked boxes and clip-level banners: hold ----
    for d in loose:
        for f in range(d.frame_index, d.frame_index + hold):
            plan[f].append({
                "bbox": list(map(float, d.bbox)) if d.bbox else None,
                "label": d.label,
                "confidence": d.confidence,
                "model_name": d.model_name,
            })

    return plan


def export_annotated_video(video_path: str, detections: list[Detection], output_path: str,
                           smooth_window: int = 5, hold_seconds: float = 0.25):
    """Re-reads the source video and burns in bboxes/labels per frame, then writes output.

    Boxes are interpolated and smoothed across the frames the runner skipped
    (see build_render_plan) so they track objects continuously instead of
    flashing once per sampled frame.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    plan = build_render_plan(detections, fps, smooth_window=smooth_window,
                             hold_seconds=hold_seconds)

    writer, codec = _open_writer(output_path, fps, w, h)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        banner_slot = 0
        for item in plan.get(frame_index, []):
            color = COLOR_MAP.get(item["label"], DEFAULT_COLOR)
            if item["bbox"]:
                x1, y1, x2, y2 = (int(round(v)) for v in item["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{item['label']} {item['confidence']:.2f}",
                            (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # clip-level detection with no bbox (e.g. violence classifier)
                # -> banner text, stacked so two models don't overprint.
                y = 30 + banner_slot * 26
                banner_slot += 1
                cv2.putText(frame,
                            f"[{item['model_name']}] {item['label']} {item['confidence']:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    print(f"Annotated video written to {output_path} ({codec})")


def export_detection_log(detections: list[Detection], output_path: str):
    """Writes detections to a JSON file for downstream metrics/analysis."""
    serializable = [d.__dict__ for d in detections]
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Detection log written to {output_path}")


def export_detection_csv(detections: list[Detection], output_path: str):
    """Writes detections to a CSV file — one row per detection with flattened kinematic fields."""
    fieldnames = [
        "model_name", "label", "confidence", "timestamp_sec",
        "frame_index", "track_id", "crowd_direction", "heading_deg",
        "speed_px_frame", "personally_stationary", "local_crush_risk",
        "local_divergence", "plate", "plate_display", "vehicle_class",
        "bbox", "keypoints", "extra"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in detections:
            row = d.__dict__.copy()
            extra = row.get("extra") or {}
            row["track_id"] = extra.get("track_id", "")
            row["crowd_direction"] = extra.get("crowd_direction", "")
            row["heading_deg"] = extra.get("heading_deg", "")
            row["speed_px_frame"] = extra.get("speed_px_frame", "")
            row["personally_stationary"] = extra.get("personally_stationary", "")
            row["local_crush_risk"] = extra.get("local_crush_risk", "")
            row["local_divergence"] = extra.get("local_divergence", "")
            row["plate"] = extra.get("plate", "")
            row["plate_display"] = extra.get("plate_display", "")
            row["vehicle_class"] = extra.get("vehicle_class", "")
            row["bbox"] = json.dumps(row["bbox"]) if row["bbox"] else ""
            row["keypoints"] = json.dumps(row["keypoints"]) if row["keypoints"] else ""
            row["extra"] = json.dumps(extra) if extra else ""
            writer.writerow(row)
    print(f"Detection CSV written to {output_path}")

