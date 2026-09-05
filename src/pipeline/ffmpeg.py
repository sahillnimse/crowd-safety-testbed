"""
Locating the ffmpeg/ffprobe binaries.

Centralized because two places need them and both used to hardcode one
developer's WinGet install path, which breaks on any other machine.

Search order: an explicit env var, then PATH, then the known Windows
package locations, then imageio-ffmpeg's bundled copy if that happens to
be installed.
"""

import os
import shutil

_ENV_VAR = "FFMPEG_BINARY"

# Where the Windows package managers put it. Globs are expanded because the
# version number is part of the path and changes on upgrade.
_WINDOWS_HINTS = [
    r"C:\Users\{user}\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-*-full_build\bin",
    r"C:\ProgramData\chocolatey\bin",
    r"C:\ffmpeg\bin",
]


def _candidate_dirs():
    import glob
    user = os.environ.get("USERNAME", "")
    for hint in _WINDOWS_HINTS:
        pattern = hint.format(user=user)
        yield from glob.glob(pattern)


def find_binary(name: str = "ffmpeg") -> str | None:
    """Absolute path to `ffmpeg`/`ffprobe`, or None if it can't be found."""
    explicit = os.environ.get(_ENV_VAR)
    if explicit and name == "ffmpeg" and os.path.exists(explicit):
        return explicit

    on_path = shutil.which(name)
    if on_path:
        return on_path

    exe = f"{name}.exe" if os.name == "nt" else name
    for d in _candidate_dirs():
        candidate = os.path.join(d, exe)
        if os.path.exists(candidate):
            return candidate

    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            pass
    return None


def find_ffmpeg() -> str | None:
    return find_binary("ffmpeg")


def ffmpeg_dir() -> str | None:
    """Directory holding ffmpeg, for tools that want --ffmpeg-location."""
    path = find_ffmpeg()
    return os.path.dirname(path) if path else None
