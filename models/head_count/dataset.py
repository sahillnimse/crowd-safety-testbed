"""
Dataset for the APGCC-format point labels that scripts/annotate_heads.py writes.

    <patches>/images/<stem>.jpg
    <patches>/labels/<stem>.txt      one "x y" per line; EMPTY = hard negative
    <patches>/train.list             "images/<stem>.jpg labels/<stem>.txt"
    <patches>/val.list

Point labels to density targets
-------------------------------
Each head point becomes a Gaussian on the stride-8 output grid, normalised so
that it contributes exactly 1.0 to the sum.  The sum of the target is then
the head count by construction, which is what makes the count loss and the
evaluation metric the same quantity.

The normalisation is done AFTER clipping to the grid, not before.  A head
near the edge of a patch has part of its Gaussian outside the image; scaling
a pre-normalised kernel would leave that person contributing 0.6 of a head
and make every crop-boundary person a systematic undercount.

Hard negatives are first-class
------------------------------
A label file with no points yields an all-zero target and is trained on like
any other sample.  These are what teach the model that tin roofs, garlands
and tarpaulins are not crowds — the failure mode that makes an unfiltered
counter useless on real festival footage.  Nothing here skips them, and
`Patches.negatives` reports how many there are so a training run can say out
loud whether it had any.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.crowd_flow.head_points import read_points
from models.head_count.model import OUTPUT_STRIDE

# Gaussian sigma for one head, in OUTPUT-GRID pixels.  At stride 8 a sigma of
# 1.5 spreads a head over roughly a 9x9 output patch, i.e. ~70 source px —
# about a head-and-shoulders at mid-distance.  Larger blurs the count into
# neighbouring metrics cells; smaller makes the target nearly a delta, which
# a convolutional model cannot fit and which destabilises the pixel loss.
DEFAULT_SIGMA = 1.5

# ImageNet statistics — the frontend is pretrained with them.
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalise(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> normalised RGB float32 CHW."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy()


def density_target(
    points: np.ndarray, h: int, w: int, sigma: float = DEFAULT_SIGMA
) -> np.ndarray:
    """
    (N, 2) source-pixel points -> (1, h/8, w/8) density map summing to N.
    """
    oh, ow = h // OUTPUT_STRIDE, w // OUTPUT_STRIDE
    target = np.zeros((oh, ow), np.float32)
    if len(points) == 0 or oh < 1 or ow < 1:
        return target[None]

    radius = max(1, int(round(3 * sigma)))
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))

    for px, py in points:
        px, py = float(px), float(py)
        # Drop only points genuinely outside the image.
        if not (0.0 <= px < w and 0.0 <= py < h):
            continue
        # Then clamp to the output grid.  Rounding alone drops anything in
        # the last half-stride: in a 256 px patch at stride 8, x=253 rounds
        # to cell 32, which does not exist, so a head plainly inside the
        # image contributed nothing.  A quiet undercount along the right and
        # bottom edges of every patch is exactly the kind of bias that
        # training cannot correct, because the labels themselves are wrong.
        cx = min(ow - 1, max(0, int(round(px / OUTPUT_STRIDE))))
        cy = min(oh - 1, max(0, int(round(py / OUTPUT_STRIDE))))
        x0, x1 = max(0, cx - radius), min(ow, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(oh, cy + radius + 1)
        kx0, ky0 = x0 - (cx - radius), y0 - (cy - radius)
        patch = kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
        s = float(patch.sum())
        if s > 0:
            # Normalise the CLIPPED kernel so this head contributes exactly
            # 1.0 even when most of it falls outside the image.
            target[y0:y1, x0:x1] += patch / s
    return target[None]


class Patches(Dataset):
    """
    Reads a .list file produced by scripts/annotate_heads.py.

    Parameters
    ----------
    root:
        The patches directory containing images/, labels/ and the .list files.
    split:
        "train" or "val".
    crop:
        Random crop size for training, or None to use whole patches.  Must be
        a multiple of OUTPUT_STRIDE so the target grid divides exactly.
    augment:
        Horizontal flip and mild brightness jitter.  Deliberately no vertical
        flip and no rotation: crowd imagery has a fixed gravity direction and
        a fixed perspective gradient, and training on upside-down crowds
        spends capacity on inputs that can never occur.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop: Optional[int] = 512,
        augment: bool = True,
        sigma: float = DEFAULT_SIGMA,
    ) -> None:
        self.root = root
        self.split = split
        self.crop = crop
        self.augment = augment and split == "train"
        self.sigma = sigma

        list_path = os.path.join(root, f"{split}.list")
        if not os.path.exists(list_path):
            raise FileNotFoundError(
                f"No {split}.list in {root}. Label some patches first with "
                f"scripts/annotate_heads.py --patches {root}"
            )
        self.items: list[tuple[str, str]] = []
        with open(list_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    self.items.append((os.path.join(root, parts[0]),
                                       os.path.join(root, parts[1])))
        if not self.items:
            raise ValueError(f"{list_path} is empty")

        if crop is not None and crop % OUTPUT_STRIDE:
            raise ValueError(
                f"crop={crop} must be a multiple of {OUTPUT_STRIDE} so the "
                f"density grid divides exactly"
            )

    def __len__(self) -> int:
        return len(self.items)

    @property
    def negatives(self) -> int:
        """Samples with zero heads — see the module docstring."""
        return sum(1 for _img, lbl in self.items if len(read_points(lbl)) == 0)

    def total_heads(self) -> int:
        return sum(len(read_points(lbl)) for _img, lbl in self.items)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.items[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"unreadable image: {img_path}")
        pts = read_points(lbl_path)

        if self.crop is not None:
            img, pts = self._crop(img, pts, self.crop)
        else:
            # Whole patch: trim to a multiple of the stride so the target
            # grid is exact rather than silently truncated.
            h = img.shape[0] - img.shape[0] % OUTPUT_STRIDE
            w = img.shape[1] - img.shape[1] % OUTPUT_STRIDE
            img = img[:h, :w]
            if len(pts):
                pts = pts[(pts[:, 0] < w) & (pts[:, 1] < h)]

        if self.augment:
            img, pts = self._augment(img, pts)

        h, w = img.shape[:2]
        target = density_target(pts, h, w, self.sigma)
        return (torch.from_numpy(normalise(img)),
                torch.from_numpy(target),
                torch.tensor(float(len(pts))))

    # ------------------------------------------------------------------

    def _crop(self, img: np.ndarray, pts: np.ndarray, size: int):
        h, w = img.shape[:2]
        if h < size or w < size:
            # Pad rather than resize: resizing changes head scale, and a
            # counter trained across inconsistent scales learns the wrong
            # relationship between apparent size and count.
            ph, pw = max(0, size - h), max(0, size - w)
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            h, w = img.shape[:2]
        x0 = np.random.randint(0, w - size + 1)
        y0 = np.random.randint(0, h - size + 1)
        img = img[y0:y0 + size, x0:x0 + size]
        if len(pts):
            pts = pts - np.array([x0, y0], np.float32)
            keep = ((pts[:, 0] >= 0) & (pts[:, 0] < size) &
                    (pts[:, 1] >= 0) & (pts[:, 1] < size))
            pts = pts[keep]
        return img, pts

    def _augment(self, img: np.ndarray, pts: np.ndarray):
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            if len(pts):
                pts = pts.copy()
                pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
        if np.random.rand() < 0.5:
            gain = np.float32(np.random.uniform(0.8, 1.2))
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return img, pts
