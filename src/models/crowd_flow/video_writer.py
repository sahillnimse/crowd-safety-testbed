"""Shared streaming video-output utility for crowd-flow wrappers."""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _AnnotatedVideoWriter:
    """Stream annotated frames to H.264, with an MJPG/AVI fallback."""

    def __init__(self, out_path: str, fps: float, w: int, h: int) -> None:
        from pipeline.ffmpeg import find_ffmpeg

        self._path = out_path
        self._proc = None
        self._writer = None
        self._failed = False

        fps = max(float(fps), 1.0)
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                "-r", f"{fps:.4f}", "-i", "-", "-an",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "23",
                "-movflags", "+faststart", out_path,
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        else:
            logger.warning(
                "ffmpeg not found; falling back to MJPG/AVI for the annotated "
                "crowd-flow video. The file will play in VLC but not in a browser."
            )
            self._path = out_path.replace(".mp4", ".avi")
            self._writer = cv2.VideoWriter(
                self._path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h),
            )

    def write(self, frame: np.ndarray) -> bool:
        if self._failed:
            return False
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
                return True
            except (BrokenPipeError, OSError) as exc:
                logger.error("Annotated video encoder stopped accepting data: %s", exc)
                self._failed = True
                return False
        if self._writer is not None:
            self._writer.write(frame)
            return True
        return False

    def close(self) -> Optional[str]:
        """Finish the file. Returns its path, or None if encoding failed."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            _, err = self._proc.communicate()
            if self._proc.returncode != 0:
                logger.error(
                    "ffmpeg failed (exit %d): %s",
                    self._proc.returncode, err.decode("utf-8", "replace").strip(),
                )
                return None
        if self._writer is not None:
            self._writer.release()
        return None if self._failed else self._path
