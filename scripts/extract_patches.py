"""
Cut training patches out of your own footage, ready for annotation.

    python scripts/extract_patches.py --source "test_videos/Nashik Crowd.mp4" \
        --out data/nashik/patches --every 25 --patch 512 --max 400

Then:
    python scripts/annotate_heads.py --patches data/nashik/patches
    python scripts/train_head_count.py --patches data/nashik/patches

Why patches rather than whole frames
------------------------------------
Annotating a full 1280x720 frame of a Kumbh crowd means several hundred
clicks in one sitting, and attention degrades long before the frame is done —
the bottom half of every frame ends up systematically under-labelled.  A
512 px patch is a few dozen clicks, finishable, and the resulting labels are
uniform in quality.  The model trains on crops anyway.

Sampling
--------
Patches are drawn from evenly spaced frames, and within each frame from
positions weighted towards texture (measured by local gradient energy).  Flat
sky and blank road are sampled at a low but non-zero rate deliberately: those
are the hard negatives, and a training set with none of them produces a
counter that fires on tarmac.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extract_patches")

# Fraction of patches drawn uniformly at random regardless of texture, so the
# set keeps genuine empty regions instead of only interesting ones.
FLAT_FRACTION = 0.25


def texture_map(gray: np.ndarray, patch: int) -> np.ndarray:
    """Per-position gradient energy, coarsely gridded to patch centres."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.boxFilter(mag, -1, (patch, patch), normalize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file")
    ap.add_argument("--out", required=True, help="patches dir to create")
    ap.add_argument("--every", type=int, default=25,
                    help="take patches from every Nth frame (default 25 ~ 1/sec)")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--per-frame", type=int, default=2)
    ap.add_argument("--max", type=int, default=400, help="stop after N patches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = Path(args.out) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("cannot open %s", args.source)
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stem = Path(args.source).stem.replace(" ", "_")

    written = 0
    frame_idx = 0
    while written < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every:
            frame_idx += 1
            continue
        h, w = frame.shape[:2]
        p = args.patch
        if h < p or w < p:
            log.error("frame %dx%d is smaller than patch %d", w, h, p)
            return 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tex = texture_map(gray, p)
        for _ in range(args.per_frame):
            if written >= args.max:
                break
            if rng.random() < FLAT_FRACTION:
                x = int(rng.integers(0, w - p + 1))
                y = int(rng.integers(0, h - p + 1))
            else:
                # Sample a centre with probability proportional to texture.
                sub = tex[p // 2:h - p // 2, p // 2:w - p // 2]
                flat = sub.ravel().astype(np.float64)
                s = flat.sum()
                if s <= 0:
                    x = int(rng.integers(0, w - p + 1))
                    y = int(rng.integers(0, h - p + 1))
                else:
                    k = int(rng.choice(flat.size, p=flat / s))
                    cy, cx = divmod(k, sub.shape[1])
                    x, y = cx, cy
            crop = frame[y:y + p, x:x + p]
            name = f"{stem}_f{frame_idx:06d}_x{x:04d}_y{y:04d}.jpg"
            cv2.imwrite(str(img_dir / name), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            written += 1
        frame_idx += 1

    cap.release()
    log.info("wrote %d patches of %dpx to %s", written, args.patch, img_dir)
    if total:
        log.info("(from %d frames, sampling every %d)", total, args.every)
    print(f"\nNext:\n  python scripts/annotate_heads.py --patches {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
