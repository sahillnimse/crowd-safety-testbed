"""
Resolve a weight filename to wherever it actually lives.

`YOLO("yolo11n.pt")` resolves the name against the *working directory* and,
failing that, downloads a fresh copy there. So once weights are collected
into `ML Models/`, every wrapper that names a checkpoint by bare filename
would quietly re-download its own duplicate into the project root — the
folder would fill up and be ignored at the same time.

`resolve()` puts the collected folder first in the search order, so moving
weights into `ML Models/ultralytics/` is safe and the folder is genuinely
the single source of truth. A bare filename is still returned unchanged
when nothing is found locally, which preserves ultralytics' own
download-on-demand for a clean checkout.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Searched in order. `ML Models/ultralytics` first so a curated download
# wins over a stray copy left in the project root by an earlier run.
SEARCH_DIRS = (
    os.path.join(PROJECT_ROOT, "ML Models", "ultralytics"),
    os.path.join(PROJECT_ROOT, "ML Models"),
    os.path.join(PROJECT_ROOT, "weights"),
    os.path.join(PROJECT_ROOT, "model_weights"),
    PROJECT_ROOT,
)


def resolve(name: str) -> str:
    """Absolute path to `name` if it's on disk, else `name` unchanged.

    Returning the bare name on a miss is deliberate: ultralytics treats a
    known checkpoint name as a download request, which is what should
    happen on a fresh clone.
    """
    if not name:
        return name
    if os.path.isabs(name) and os.path.exists(name):
        return name

    base = os.path.basename(name)
    for directory in SEARCH_DIRS:
        candidate = os.path.join(directory, base)
        if os.path.exists(candidate):
            return candidate
    return name


def found_in_collection(name: str) -> bool:
    """True if `name` is inside ML Models/ rather than loose in the repo."""
    resolved = resolve(name)
    ml_dir = os.path.join(PROJECT_ROOT, "ML Models")
    return os.path.isabs(resolved) and resolved.startswith(ml_dir)
