"""Shared streaming video-output utility for crowd-flow wrappers."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
from collections import deque
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
        self._stderr_tail: deque = deque(maxlen=40)
        self._stderr_thread: Optional[threading.Thread] = None

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
            # Drain stderr continuously on a side thread.
            #
            # The pipe holds ~64 KB. Nothing read it until close(), so an
            # ffmpeg that emitted more than that during a long encode blocked
            # writing to it; this process then blocked forever in
            # stdin.write(), and the run hung with no error anywhere. Unlikely
            # at -loglevel error, but the failure mode is a silent deadlock on
            # long jobs, which is exactly when it costs most.
            #
            # Only the tail is retained (see _stderr_tail above): it is
            # diagnostic context for a non-zero exit, not something to
            # accumulate for a whole run.
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True,
            )
            self._stderr_thread.start()
        else:
            logger.warning(
                "ffmpeg not found; falling back to MJPG/AVI for the annotated "
                "crowd-flow video. The file will play in VLC but not in a browser."
            )
            # splitext, not str.replace: replace() rewrites EVERY ".mp4" in
            # the path, so a run directory that happens to contain one (a
            # video named "clip.mp4" becomes the folder "clip.mp4/") had its
            # directory renamed too and the write landed somewhere else.
            self._path = os.path.splitext(out_path)[0] + ".avi"
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
            self._proc.wait()
            if self._stderr_thread is not None:
                # Bounded: the pipe closes when ffmpeg exits, so the reader
                # loop ends on its own. The timeout is belt-and-braces so a
                # wedged encoder cannot hang shutdown.
                self._stderr_thread.join(timeout=10)
            if self._proc.returncode != 0:
                logger.error(
                    "ffmpeg failed (exit %d): %s",
                    self._proc.returncode,
                    "\n".join(self._stderr_tail).strip(),
                )
                return None
        if self._writer is not None:
            self._writer.release()
        return None if self._failed else self._path

    def _drain_stderr(self) -> None:
        """Consume ffmpeg's stderr so its pipe buffer can never fill."""
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
        except (ValueError, OSError):
            # Pipe closed underneath us during shutdown; nothing to report.
            pass
        finally:
            with contextlib.suppress(Exception):
                stream.close()
