"""
Score this project's head counters against SAIVT ground truth.

This is the first time any model in this repo is measured against annotated
reality. Until now configs/test_videos.yaml carried four empty `ground_truth`
keys, so every accuracy claim was unverifiable.

Usage
-----
  # what can actually be scored, and where the data is
  python scripts/evaluate_on_saivt.py --coverage

  # score one model on one camera (fast smoke test)
  python scripts/evaluate_on_saivt.py --models apgcc --cameras P_Lev_4_Lift_6_ip_51 --limit 10

  # the real run: every counter, every annotated camera
  python scripts/evaluate_on_saivt.py --models apgcc,dmcount,rtdetr --all-cameras

  # write machine-readable results
  python scripts/evaluate_on_saivt.py --models apgcc --all-cameras --out outputs/eval/saivt.json

Reading the output
------------------
count_mae     mean absolute headcount error, in people
count_bias    SIGNED mean error. Negative = systematic UNDER-count, which
              propagates into an underestimate of density and therefore of
              crowd pressure. That is the dangerous direction, so read this
              before MAE.
precision     of the heads the model reported, how many were real
recall        of the people actually there, how many were found
f1            harmonic mean of the two

A model can post a good MAE with poor precision/recall when its false
positives and false negatives cancel within a frame. Density tolerates that;
tracking, counter-flow and specific flow do not.

Scope
-----
This scores COUNTING and LOCALISATION only. SAIVT annotates occupancy and gate
crossings; it contains no labelled crush or compression event, so nothing here
speaks to the false-negative rate of the crush-alert path. That remains
unmeasured until incident footage is annotated.

And note the domain: SAIVT is an indoor university building. Good numbers here
are a FLOOR, not a transfer — Nashik ghats are outdoor, denser, at night, in
monsoon, with umbrellas occluding heads.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT / "src"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import coverage_summary, list_cameras, load_camera  # noqa: E402
from evaluation import evaluate_camera  # noqa: E402
from models.crowd_flow.perspective import PerspectiveMap  # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)-7s %(name)s  %(message)s")
log = logging.getLogger("evaluate_on_saivt")


# ----------------------------------------------------------------------
# Predictors — each returns a callable(image_bgr) -> list[(x, y)]
# ----------------------------------------------------------------------

def _apgcc(device, threshold: float = 0.5):
    from models.head_count import get_head_counter
    hc = get_head_counter(device=device, score_threshold=threshold)
    hc.load()

    def predict(img):
        pts = hc.points(img)
        return [(float(p[0]), float(p[1])) for p in np.asarray(pts).reshape(-1, 2)]
    return predict


def _dmcount(device, threshold: float = 0.06):
    from models.dm_count.infer import DMCountCounter
    c = DMCountCounter(device=device)
    # `is_available` is a PROPERTY, not a method. Calling it invoked the
    # returned bool and raised "'bool' object is not callable", which the
    # runner then reported as a model-load failure -- so DM-Count looked
    # unavailable when its weights were present the whole time.
    if not c.is_available:
        raise RuntimeError(
            "DM-Count weights not found. Expected one of the paths in "
            "models/dm_count/infer.py (e.g. src/models/dm_count/weights/model_sh_A.pth)")
    c.load()

    def predict(img):
        out = c.predict(img)
        return [(float(p[0]), float(p[1]))
                for p in np.asarray(out.points).reshape(-1, 3)]
    return predict


def _rtdetr(device, threshold: float = 0.35, tile=(2, 2)):
    """RT-DETRv2 person boxes, reduced to a head point.

    Head point, not box centre: the annotations mark heads, so scoring a box
    centre against them would report a systematic vertical offset as
    localisation error. Top-centre plus ~15% of box height approximates where
    a head sits inside a full-body box.
    """
    from models._detectors import get_detector, COCO_PERSON
    det = get_detector(device=device)
    det.load()

    def predict(img):
        boxes = det.detect(img, classes=(COCO_PERSON,),
                           conf_threshold=threshold, tile_grid=tile)
        out = []
        for b in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
            x1, y1, x2, y2 = b
            out.append((float((x1 + x2) / 2.0), float(y1 + 0.15 * (y2 - y1))))
        return out
    return predict


PREDICTORS = {"apgcc": _apgcc, "dmcount": _dmcount, "rtdetr": _rtdetr}


# ----------------------------------------------------------------------

def print_coverage() -> None:
    s = coverage_summary()
    print("\nSAIVT coverage")
    print("=" * 78)
    print(f"annotations : {s['annotation_dir']}")
    print(f"imagery     : {s['imagery_root']}")
    t = s["totals"]
    print(f"\n{t['cameras']} cameras | occupancy GT on {t['with_occupancy_gt']} | "
          f"flow GT on {t['with_flow_gt']} | perspective on {t['with_perspective']}")
    print(f"{t['annotated_stills']} annotated stills, {t['annotated_people']} annotated people\n")
    print(f"{'camera':<38}{'stills':>7}{'people':>8}{'flow':>7}{'persp':>7}{'imgs':>6}")
    print("-" * 78)
    from datasets import load_all
    cams = load_all()
    for cid, c in s["cameras"].items():
        people = sum(len(v) for v in cams[cid].occupancy.values()) if cid in cams else 0
        print(f"{cid[:37]:<38}{c['occupancy_frames']:>7}{people:>8}"
              f"{c['flow_events']:>7}{'yes' if c['has_perspective'] else '-':>7}"
              f"{'yes' if c['frames_available'] else '-':>6}")
    print()
    if "not configured" in s["imagery_root"]:
        print("!! Frames not found. Scoring needs the imagery: set SAIVT_ROOT or\n"
              "   configs/saivt.yaml -> saivt.root to the AnnotatedData folder.\n")


def run(args) -> int:
    if args.coverage:
        print_coverage()
        return 0

    cams = (list_cameras() if args.all_cameras
            else [c.strip() for c in (args.cameras or "").split(",") if c.strip()])
    if not cams:
        print("Nothing to do: pass --cameras <id,...> or --all-cameras "
              "(or --coverage to see what exists).")
        return 2

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in PREDICTORS]
    if unknown:
        print(f"Unknown model(s): {unknown}. Available: {sorted(PREDICTORS)}")
        return 2

    results, skipped = [], []
    for name in wanted:
        print(f"\nLoading {name} ...", flush=True)
        try:
            predict = PREDICTORS[name](args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {exc}")
            skipped.append((name, str(exc)))
            continue

        for cid in cams:
            try:
                cam = load_camera(cid)
            except FileNotFoundError as exc:
                print(f"  SKIP {cid}: {exc}")
                continue
            if not cam.has_occupancy_gt:
                continue                      # flow-only camera; nothing to count against
            if not cam.frames_dir:
                skipped.append((cid, "frames not on disk - set SAIVT_ROOT"))
                continue

            pm = (PerspectiveMap.from_saivt(cam.perspective)
                  if cam.has_perspective else None)

            t0 = time.perf_counter()
            res = evaluate_camera(
                cam, predict, model_name=name, perspective=pm,
                radius_px=args.radius_px, limit=args.limit,
                progress=(lambda i, n: print(f"\r  {cid[:34]:<34} {i}/{n}",
                                             end="", flush=True))
                if not args.quiet else None)
            dt = time.perf_counter() - t0
            print(f"\r  {res.summary_line()}   [{dt:.1f}s]")
            results.append(res)

    if not results:
        print("\nNo camera was scored.")
        for what, why in skipped:
            print(f"  {what}: {why}")
        return 1

    print("\n" + "=" * 96)
    print("RESULTS".center(96))
    print("=" * 96)
    print(f"{'model':<10}{'camera':<36}{'stills':>7}{'MAE':>8}{'bias':>8}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    print("-" * 96)
    for r in results:
        print(f"{r.model_name:<10}{r.camera_id[:35]:<36}{r.n_frames:>7}"
              f"{r.mae:>8.2f}{r.bias:>+8.2f}{r.precision:>8.3f}"
              f"{r.recall:>8.3f}{r.f1:>8.3f}")

    # Per-model aggregate, weighted by annotated people rather than by camera:
    # a camera with 900 annotated people says more about a model than one with
    # 40, and a plain mean over cameras would weight them equally.
    print("-" * 96)
    for name in wanted:
        rs = [r for r in results if r.model_name == name]
        if not rs:
            continue
        gt = sum(r.total_gt for r in rs)
        tp, fp, fn = sum(r.tp for r in rs), sum(r.fp for r in rs), sum(r.fn for r in rs)
        p = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * rc / (p + rc) if (p == p and rc == rc and p + rc) else float("nan")
        mae = float(np.mean([abs(f.count_error) for r in rs for f in r.frames]))
        bias = float(np.mean([f.count_error for r in rs for f in r.frames]))
        print(f"{name:<10}{'ALL (' + str(len(rs)) + ' cameras, ' + str(gt) + ' people)':<36}"
              f"{sum(r.n_frames for r in rs):>7}{mae:>8.2f}{bias:>+8.2f}"
              f"{p:>8.3f}{rc:>8.3f}{f1:>8.3f}")

    print("\nbias < 0 means systematic UNDER-counting -> density and crowd "
          "pressure are under-read.\nSAIVT is an indoor building: treat these "
          "as a floor for Nashik, not a transfer.\n")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"results": [r.to_dict() for r in results],
                       "skipped": skipped}, f, indent=2)
        print(f"written: {args.out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score head counters against SAIVT ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--coverage", action="store_true",
                    help="Show what ground truth exists, then exit.")
    ap.add_argument("--models", default="apgcc",
                    help=f"Comma-separated: {sorted(PREDICTORS)}")
    ap.add_argument("--cameras", default="", help="Comma-separated camera ids.")
    ap.add_argument("--all-cameras", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Score only the first N annotated stills per camera.")
    ap.add_argument("--radius-px", type=float, default=None,
                    help="Fixed match radius. Default scales with perspective.")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--out", default="", help="Write results JSON here.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    args.limit = args.limit or None
    sys.exit(run(args))


if __name__ == "__main__":
    main()
