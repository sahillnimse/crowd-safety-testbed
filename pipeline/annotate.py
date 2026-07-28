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
from collections import defaultdict

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


def export_annotated_video(video_path: str, detections: list[Detection], output_path: str):
    """Re-reads the source video and burns in bboxes/labels per frame, then writes output."""
    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d.frame_index].append(d)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, codec = _open_writer(output_path, fps, w, h)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for d in by_frame.get(frame_index, []):
            color = COLOR_MAP.get(d.label, DEFAULT_COLOR)
            if d.bbox:
                x1, y1, x2, y2 = map(int, d.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # clip-level detection with no bbox (e.g. violence classifier) -> banner text
                cv2.putText(frame, f"[{d.model_name}] {d.label} {d.confidence:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
    """Writes detections to a CSV file — one row per detection."""
    fieldnames = ["model_name", "label", "confidence", "timestamp_sec",
                  "frame_index", "bbox", "keypoints", "extra"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in detections:
            row = d.__dict__.copy()
            row["bbox"] = json.dumps(row["bbox"]) if row["bbox"] else ""
            row["keypoints"] = json.dumps(row["keypoints"]) if row["keypoints"] else ""
            row["extra"] = json.dumps(row["extra"]) if row["extra"] else ""
            writer.writerow(row)
    print(f"Detection CSV written to {output_path}")
