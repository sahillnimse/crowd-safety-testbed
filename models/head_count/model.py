"""
Point-based head counter — APGCC (VGG16-BN encoder, IFI decoder).

This module used to hold a custom ResNet-18 density-map net (HeadCountNet).
That model is gone. APGCC replaces it entirely, on your call: APGCC is
treated as ground truth for head detection, at a different point in the
pipeline than the old density net occupied.

    input   (B, 3, H, W), ImageNet-normalised, H and W multiples of 16
    output  points: (N, 2) head locations in source pixels
            scores: (N,)   float32 in [0, 1], softmax confidence per point

Why points now, not a density map
----------------------------------
APGCC is a detection-style point predictor (DETR-like point queries decoded
per anchor), not a density regressor — there is no per-pixel density map to
integrate; the count *is* len(points) after score-thresholding. The old
HeadCountNet's density_map()/count()-by-integration approach doesn't apply
here. HeadCounter (infer.py) keeps count() and points() as the public
surface so density.py and crowd_motion_monitor.py don't need to know the
model underneath changed shape, but internally count is now
len(points_above_threshold), not a spatial sum.

Why APGCC over the previous ResNet-18 density net
--------------------------------------------------
Per your call: APGCC's accuracy on this task is high enough to treat its
output as ground truth, and its detections are what crowd_motion_monitor.py
now fuses against RT-DETRv2 boxes (RT-DETRv2 claims a person first; APGCC
only supplies people RT-DETRv2 missed). A per-point confidence score is what
that fusion needs to decide which points are real detections — a density
map has no equivalent per-instance confidence to threshold on.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.head_count._apgcc_loader import build_apgcc, load_apgcc

# APGCC's own STRIDE: 8 config plus ENCODER_kwargs.last_pool=False means the
# deepest encoder feature is H/16, W/16.  Padding input to a multiple of 16
# (rather than trimming, as the old stride-8 density net did) keeps every
# source pixel in the count — APGCC has no tolerance for losing a partial
# cell the way a summed density map did.
INPUT_DIVISOR = 16

# ImageNet stats APGCC's own datasets/build.py normalises with.  Must match
# exactly: this is a pretrained checkpoint, not one trained here.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Upstream's own evaluate_crowd_counting[_and_loc] threshold (engine.py).
DEFAULT_SCORE_THRESHOLD = 0.5


def build_model(config: str = "shha", imagenet_init: bool = False):
    """Construct an untrained APGCC model. Rarely needed directly — see load_apgcc."""
    return build_apgcc(config=config, imagenet_init=imagenet_init)


def load_model(weights: str | None = None, device: str = "cpu",
                config: str = "shha", strict: bool = True):
    """Build APGCC and load a checkpoint. Returns (model, load_info)."""
    return load_apgcc(weights=weights, device=device, config=config, strict=strict)


def preprocess(frame_rgb_float01: torch.Tensor) -> torch.Tensor:
    """
    ImageNet-normalise a (B, 3, H, W) float tensor already in [0, 1].

    Kept as a standalone function (rather than inlined in infer.py) because
    it is the one piece of preprocessing that must byte-for-byte match
    datasets/build.py's transform — anything else here is free to change,
    this specific pair of constants is not.
    """
    mean = _MEAN.to(frame_rgb_float01.device, frame_rgb_float01.dtype)
    std = _STD.to(frame_rgb_float01.device, frame_rgb_float01.dtype)
    return (frame_rgb_float01 - mean) / std


@torch.no_grad()
def points_and_scores(model, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run APGCC and return (points, scores) BEFORE thresholding.

    points: (N, 2) in the *tensor's* pixel space (caller rescales to source).
    scores: (N,) softmax confidence for the person class (index 1 of 2,
    per engine.py's own evaluate_crowd_counting: index 0 is "no object").
    """
    out = model(tensor)
    scores = F.softmax(out["pred_logits"], dim=-1)[:, :, 1][0]
    points = out["pred_points"][0]
    return points, scores
