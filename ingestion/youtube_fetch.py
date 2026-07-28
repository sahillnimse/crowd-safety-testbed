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
    output_path = os.path.join(TEST_VIDEOS_DIR, f"{video_id}.mp4")

    if os.path.exists(output_path) and not force:
        print(f"Using cached video: {output_path}")
        return output_path

    format_selector = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "--merge-output-format", "mp4",
        "--remote-components", "ejs:github",
        "-o", output_path,
        url,
    ]

    # These were hardcoded to one machine's WinGet install paths, which made
    # the downloader fail anywhere else. Both are optional: yt-dlp only needs
    # the locations when the binaries aren't already on PATH.
    ffmpeg_bin_dir = ffmpeg_dir()
    if ffmpeg_bin_dir:
        cmd[-2:-2] = ["--ffmpeg-location", ffmpeg_bin_dir]

    deno = shutil.which("deno") or _known_deno_path()
    if deno:
        cmd[-2:-2] = ["--js-runtimes", f"deno:{deno}"]
    print(f"Downloading {url} -> {output_path}")
    subprocess.run(cmd, check=True)
    return output_path
