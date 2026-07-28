"""
Re-encodes already-exported annotated videos to browser-playable H.264.

Anything exported before the encoder fix is MPEG-4 Part 2 (`mp4v`), which
plays fine in VLC but renders nothing in an HTML5 <video> element — so the
web UI's preview opens to a black box. Re-running the models just to get a
playable file would be enormously wasteful when the frames are already
correct; this transcodes them in place instead.

Usage:
    python scripts/fix_annotated_codec.py            # fix outputs/annotated
    python scripts/fix_annotated_codec.py --dry-run  # just report codecs
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.ffmpeg import find_binary  # noqa: E402

ANNOTATED_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "annotated")

# Codecs a browser can actually decode in <video>.
PLAYABLE = {"h264", "vp8", "vp9", "av1"}


def probe_codec(ffprobe: str, path: str) -> str | None:
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None


def transcode(ffmpeg: str, path: str) -> None:
    """Re-encode to H.264 via a temp file, then swap it in."""
    tmp = path + ".h264.tmp.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", path,
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "23",
         "-movflags", "+faststart", tmp],
        check=True,
    )
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=ANNOTATED_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ffmpeg = find_binary("ffmpeg")
    ffprobe = find_binary("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg/ffprobe not found. Install ffmpeg or set FFMPEG_BINARY.")

    directory = os.path.abspath(args.dir)
    if not os.path.isdir(directory):
        raise SystemExit(f"No such directory: {directory}")

    files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".mp4"))
    if not files:
        print(f"No .mp4 files in {directory}")
        return

    needs_fix = []
    for name in files:
        path = os.path.join(directory, name)
        codec = probe_codec(ffprobe, path)
        ok = codec in PLAYABLE
        print(f"  {'OK  ' if ok else 'FIX '} {name}  ({codec or 'unknown'})")
        if not ok:
            needs_fix.append(path)

    if not needs_fix:
        print("\nAll videos are already browser-playable.")
        return
    if args.dry_run:
        print(f"\n{len(needs_fix)} file(s) would be re-encoded. Re-run without --dry-run.")
        return

    print(f"\nRe-encoding {len(needs_fix)} file(s) to H.264...")
    for i, path in enumerate(needs_fix, 1):
        print(f"  [{i}/{len(needs_fix)}] {os.path.basename(path)}")
        transcode(ffmpeg, path)
    print("Done - reload the web UI and the previews will play.")


if __name__ == "__main__":
    main()
