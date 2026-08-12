"""
Head-point label I/O, in the format APGCC and its dataset loader expect.

    <patches>/images/<stem>.jpg
    <patches>/labels/<stem>.txt      one "x y" per line
    <patches>/train.list             "images/<stem>.jpg labels/<stem>.txt"
    <patches>/val.list               held-out split

An empty label file is a hard negative — a real training sample asserting
"there are no heads here", which is how a counter learns not to fire on tin
roofs and garlands.  It is emphatically not a missing or skipped file, so
nothing here treats zero points as an error or refuses to write the file.

This module exists because the annotation tool was written against a
`apgcc_dataset` module that does not ship with it.  The format above is fully
specified by the tool's own documentation, so these are the real readers and
writers rather than a guess at someone else's API.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np


def require_gui() -> None:
    """
    Fail early and explain, if cv2 was built without a GUI.

    Raises SystemExit rather than letting cv2.namedWindow throw: OpenCV's own
    message ("Rebuild the library with Windows, GTK+ 2.x or Cocoa support")
    sends people off to compile OpenCV, when the real cause is almost always
    that opencv-python-headless is installed alongside opencv-python — easyocr
    depends on it — and won whichever unpacked last.
    """
    import cv2

    if "GUI:                           NONE" in cv2.getBuildInformation() or \
            not hasattr(cv2, "namedWindow"):
        raise SystemExit(
            "This tool needs a GUI-capable OpenCV, and the installed build "
            f"(cv2 {cv2.__version__}) has none.\n\n"
            "Almost certainly opencv-python-headless is installed alongside "
            "opencv-python — easyocr requires it — and they share the same "
            "cv2/ directory, so the headless one replaced the working build.\n\n"
            "Fix:\n"
            "  pip uninstall -y opencv-python opencv-python-headless "
            "opencv-contrib-python\n"
            "  pip install opencv-python\n"
        )


def read_points(path) -> np.ndarray:
    """
    Read a label file into an (N, 2) float32 array of (x, y) head points.

    A missing or empty file yields an empty (0, 2) array — a hard negative,
    not a failure.  Malformed lines are skipped rather than raising: a label
    set is hand-made, and one stray line should not make the whole run
    unopenable halfway through an annotation session.
    """
    path = str(path)
    if not os.path.exists(path):
        return np.zeros((0, 2), np.float32)

    pts: list[tuple[float, float]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not pts:
        return np.zeros((0, 2), np.float32)
    return np.asarray(pts, dtype=np.float32)


def write_points(path, points: Sequence | np.ndarray) -> None:
    """
    Write (N, 2) points as "x y" per line, creating the parent directory.

    Zero points writes an empty file on purpose — see the module docstring.
    Written to a temporary file and moved into place so that killing the
    annotator mid-write leaves the previous labels intact rather than a
    truncated file.
    """
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    body = "".join(f"{x:.2f} {y:.2f}\n" for x, y in arr)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
    os.replace(tmp, path)


def write_lists(root, stems: Iterable[str], val_frac: float = 0.15) -> tuple[int, int]:
    """
    Write train.list / val.list.  Returns (n_train, n_val).

    The split is by a stable hash of the stem, not by random shuffle and not
    by Python's built-in hash(): PYTHONHASHSEED is randomised per process, so
    `hash(stem) % 100` reshuffles the validation set on every run, and a
    moving val set makes epoch-to-epoch numbers incomparable — which is the
    one thing a val set exists to provide.
    """
    import zlib

    root = str(root)
    cut = int(max(0.0, min(1.0, val_frac)) * 100)
    train, val = [], []
    for stem in sorted(stems):
        bucket = zlib.crc32(stem.encode("utf-8")) % 100
        (val if bucket < cut else train).append(stem)

    for name, group in (("train.list", train), ("val.list", val)):
        lines = [f"images/{s}.jpg labels/{s}.txt" for s in group]
        with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
    return len(train), len(val)
