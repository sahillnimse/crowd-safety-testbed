"""
Scoring a counter or detector against point annotations.

This is the harness that was missing from both projects. Ujwal's __CMS__ loads
SAIVT annotations and can display them, but never compares a prediction to
them; this repo had compare_models.py, which computes precision/recall/F1 but
had no populated ground truth to eat. Data without a scorer, and a scorer
without data.

Two families of number, and they answer different questions
-----------------------------------------------------------
COUNT error (MAE / RMSE / MAPE)
    "Does the model get the headcount right?" This is what crowd-counting
    papers report, and it is what density and crowd pressure inherit. A model
    can score well here while putting every head in the wrong place, because
    over- and under-detections cancel within a frame.

LOCALISATION (precision / recall / F1)
    "Are the heads where the model says they are?" Matched one-to-one against
    annotations. This is what tracking, counter-flow and specific flow depend
    on, and it cannot be faked by cancellation.

Both are reported. A model that is good at the first and bad at the second is
usable for density and useless for flow, and only reporting both shows it.

Matching rule
-------------
Greedy nearest-neighbour, closest pair first, one-to-one, within a radius.

The radius is PERSPECTIVE-SCALED when a PerspectiveMap is available: a fixed
pixel radius is far too tight for a person near the camera (200 px tall) and
far too generous for one at the back (60 px), so a single number quietly
measures near-field accuracy at one end of the frame and near-random matching
at the other. Default is half a person's height at that image location, which
is the usual convention for head-point matching.

What this CANNOT tell you
-------------------------
Whether a crush would have been detected. These annotations are occupancy and
gate crossings; there is no labelled compression event in this dataset, and
counting accuracy is not evidence about crush-alert recall. The false-negative
rate of the crush path stays unmeasured until incident footage is annotated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrameResult:
    """Per-frame scoring outcome."""
    index: int
    gt_count: int
    pred_count: int
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    match_distances: list = field(default_factory=list)

    @property
    def count_error(self) -> int:
        return self.pred_count - self.gt_count


@dataclass
class EvalResult:
    """Aggregate scores over a set of frames."""
    camera_id: str
    model_name: str
    frames: list = field(default_factory=list)

    # -- count accuracy ------------------------------------------------
    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def mae(self) -> float:
        """Mean absolute count error, in people. The headline counting number."""
        if not self.frames:
            return float("nan")
        return float(np.mean([abs(f.count_error) for f in self.frames]))

    @property
    def rmse(self) -> float:
        """Penalises large misses more than MAE; the pair is conventional."""
        if not self.frames:
            return float("nan")
        return float(np.sqrt(np.mean([f.count_error ** 2 for f in self.frames])))

    @property
    def bias(self) -> float:
        """
        Signed mean error. Sign matters more than magnitude here: a negative
        bias means systematic UNDER-counting, which propagates into an
        underestimate of density and therefore of crowd pressure — the
        direction this project's own config calls "the failure mode that
        matters".
        """
        if not self.frames:
            return float("nan")
        return float(np.mean([f.count_error for f in self.frames]))

    @property
    def mape(self) -> float:
        """Mean absolute PERCENTAGE error over non-empty frames only.

        Empty frames are excluded rather than treated as 100% error: dividing
        by a ground truth of zero is undefined, and including them would let
        the score be dominated by how often the room was empty.
        """
        vals = [abs(f.count_error) / f.gt_count
                for f in self.frames if f.gt_count > 0]
        return float(100.0 * np.mean(vals)) if vals else float("nan")

    # -- localisation --------------------------------------------------
    @property
    def tp(self) -> int:
        return sum(f.true_positives for f in self.frames)

    @property
    def fp(self) -> int:
        return sum(f.false_positives for f in self.frames)

    @property
    def fn(self) -> int:
        return sum(f.false_negatives for f in self.frames)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p) or not (r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def mean_match_distance(self) -> float:
        d = [x for f in self.frames for x in f.match_distances]
        return float(np.mean(d)) if d else float("nan")

    @property
    def total_gt(self) -> int:
        return sum(f.gt_count for f in self.frames)

    @property
    def total_pred(self) -> int:
        return sum(f.pred_count for f in self.frames)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "model": self.model_name,
            "frames_scored": self.n_frames,
            "gt_people_total": self.total_gt,
            "predicted_total": self.total_pred,
            "count_mae": round(self.mae, 3),
            "count_rmse": round(self.rmse, 3),
            "count_bias": round(self.bias, 3),
            "count_mape_pct": round(self.mape, 2),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "mean_match_distance_px": round(self.mean_match_distance, 2),
        }

    def summary_line(self) -> str:
        return (f"{self.model_name:<18} {self.camera_id[:34]:<34} "
                f"MAE {self.mae:6.2f}  bias {self.bias:+6.2f}  "
                f"P {self.precision:.3f}  R {self.recall:.3f}  F1 {self.f1:.3f}")


# ----------------------------------------------------------------------

def match_points(pred: list, gt: list, radius_fn) -> tuple:
    """
    Greedy one-to-one matching, closest pair first.

    ``radius_fn(x, y) -> float`` gives the acceptance radius at a ground-truth
    point, so the tolerance can follow perspective.

    Greedy rather than Hungarian: for head points the two agree almost always,
    greedy is O(n log n) instead of O(n^3) on frames with hundreds of people,
    and the difference is far smaller than the annotation's own positional
    noise.
    """
    if not len(pred) or not len(gt):
        return 0, len(pred), len(gt), []

    P = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    G = np.asarray(gt, dtype=np.float64).reshape(-1, 2)

    d = np.sqrt(((P[:, None, :] - G[None, :, :]) ** 2).sum(axis=2))
    radii = np.array([radius_fn(float(g[0]), float(g[1])) for g in G])
    allowed = d <= radii[None, :]

    pi, gi = np.nonzero(allowed)
    if not len(pi):
        return 0, len(P), len(G), []

    order = np.argsort(d[pi, gi])
    used_p, used_g, dists = set(), set(), []
    for k in order:
        a, b = int(pi[k]), int(gi[k])
        if a in used_p or b in used_g:
            continue
        used_p.add(a)
        used_g.add(b)
        dists.append(float(d[a, b]))

    tp = len(dists)
    return tp, len(P) - tp, len(G) - tp, dists


def evaluate_camera(camera, predictor, model_name: str,
                    perspective=None, radius_px: Optional[float] = None,
                    radius_frac_of_height: float = 0.5,
                    roi_only: bool = True,
                    limit: Optional[int] = None,
                    progress=None) -> EvalResult:
    """
    Score ``predictor`` on one SAIVT camera's annotated stills.

    ``predictor(image_bgr) -> list[(x, y)]`` returns head points in pixels.
    ``roi_only`` restricts both predictions and ground truth to the camera's
    counting ROI — the annotations only cover that polygon, so a detection
    outside it is neither right nor wrong and must not be scored as a false
    positive.
    """
    import cv2

    result = EvalResult(camera_id=camera.camera_id, model_name=model_name)

    roi_poly = camera.roi if (roi_only and camera.roi is not None
                              and len(camera.roi) >= 3) else None

    def _inside(pt) -> bool:
        if roi_poly is None:
            return True
        return cv2.pointPolygonTest(
            roi_poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0

    if radius_px is not None:
        def radius_fn(x, y):
            return radius_px
    elif perspective is not None:
        def radius_fn(x, y):
            _, h = perspective.person_size_at_feet(x, y)
            return max(radius_frac_of_height * h, 8.0)
    else:
        # No perspective available: a flat radius, stated rather than hidden.
        def radius_fn(x, y):
            return 25.0

    indices = camera.annotated_indices
    if limit:
        indices = indices[:limit]

    for n, idx in enumerate(indices):
        path = camera.frame_path(idx)
        if path is None:
            continue
        img = cv2.imread(path)
        if img is None:
            continue

        gt = [p for p in camera.occupancy.get(idx, []) if _inside(p)]
        pred = [p for p in predictor(img) if _inside(p)]

        tp, fp, fn, dists = match_points(pred, gt, radius_fn)
        result.frames.append(FrameResult(
            index=idx, gt_count=len(gt), pred_count=len(pred),
            true_positives=tp, false_positives=fp, false_negatives=fn,
            match_distances=dists))
        if progress:
            progress(n + 1, len(indices))

    return result
