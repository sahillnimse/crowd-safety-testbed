"""
Downloads a YouTube video locally via yt-dlp for offline processing.

Caches downloads in test_videos/ keyed by video ID, so re-running tests
doesn't re-download. 720p is the default ceiling — plenty for these
models and much cheaper than 4K to decode/process.
"""

import glob
import os
import re
import shutil
import subprocess

from pipeline.ffmpeg import ffmpeg_dir

TEST_VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "test_videos")


def _known_deno_path() -> str | None:
    """Deno is used by yt-dlp to run YouTube's player JS. Optional — yt-dlp
    falls back to its own interpreter when it isn't present."""
    pattern = os.path.join(
        os.path.expanduser("~"), "AppData", "Local", "Microsoft", "WinGet",
        "Packages", "DenoLand.Deno_*", "deno.exe",
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return match.group(1)


def fetch_youtube_video(url: str, max_height: int = 720, force: bool = False) -> str:
    """
    Downloads `url` via yt-dlp if not already cached.
    Returns the local file path.
    """
    os.makedirs(TEST_VIDEOS_DIR, exist_ok=True)
    video_id = _extract_video_id(url)
    output_path = os.path.abspath(os.path.join(TEST_VIDEOS_DIR, f"{video_id}.mp4"))

    if os.path.exists(output_path) and not force:
        print(f"Using cached video: {output_path}")
        return output_path

    format_selector = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"

    # Optional locations, only needed when the binaries aren't on PATH. These
    # were previously hardcoded to one machine's WinGet paths, which made the
    # downloader fail anywhere else.
    optional: list[str] = []
    ffmpeg_bin_dir = ffmpeg_dir()
    if ffmpeg_bin_dir:
        optional += ["--ffmpeg-location", ffmpeg_bin_dir]
    deno = shutil.which("deno") or _known_deno_path()
    if deno:
        optional += ["--js-runtimes", f"deno:{deno}"]

    # Assembled in one piece. Splicing the optional flags into a prebuilt
    # list by negative index put them between "-o" and its value, so yt-dlp
    # read "--ffmpeg-location" as the output filename template and failed.
    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "--merge-output-format", "mp4",
        "--remote-components", "ejs:github",
        *optional,
        "-o", output_path,
        url,
    ]

    print(f"Downloading {url} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Surface yt-dlp's own message: "video unavailable", "sign in to
        # confirm you're not a bot", and a stale yt-dlp all look identical
        # from a bare CalledProcessError.
        detail = (result.stderr or result.stdout or "").strip()
        tail = "\n".join(detail.splitlines()[-6:])
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}) for {url}\n{tail}\n\n"
            "If this mentions bot checks or player extraction, run "
            "`python -m pip install -U yt-dlp` — YouTube changes break older "
            "versions regularly."
        )

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"yt-dlp reported success but {output_path} does not exist. "
            "The merge step may have written a different container; check "
            "the test_videos/ directory."
        )
    return output_path
