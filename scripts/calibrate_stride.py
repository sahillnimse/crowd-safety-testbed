"""
Find the right temporal baseline (frame stride) for a camera.

    python scripts/calibrate_stride.py --source "test_videos/Nashik Crowd.mp4"

The question this answers
------------------------
"Should we run at a higher frame rate to measure motion better?"

The intuition says yes and the physics says no.  Optical flow measures a
DISPLACEMENT between two frames, in pixels.  Velocity is that displacement
divided by the time between them:

    v = displacement_px * fps

Doubling the frame rate halves the displacement between consecutive frames.
The velocity is unchanged — it is a property of the crowd, not of the camera
— but the *measurement* is now half as large while the noise floor stays
exactly where it was.  A higher frame rate therefore makes each individual
measurement HARDER, not easier.

On this project's Nashik footage a pedestrian moves 0.49 px between adjacent
frames at 25 fps.  At 50 fps that would be 0.25 px: below what dense flow
resolves reliably, and mostly rejected by the validity gate as unmeasurable.

The useful lever is the opposite one: compare frames FURTHER APART.  A stride
of 3 gives three times the displacement for the same physical velocity, which
is three times the signal against an unchanged noise floor — for a third of
the compute, not more.

Where it stops working
----------------------
Not indefinitely.  Past some gap, people have moved far enough that the
algorithm matches the wrong parts of the scene: paths curve, bodies occlude
each other, and the correspondence breaks down.  That shows up here as the
recovered velocity falling away, because failed matches read as less motion
rather than more.

So there is a window, and this script finds it, by exploiting the one thing
that must be true: **the physical velocity does not depend on which pair of
frames you measure it from.**  Displacement must scale linearly with the time
gap.  Where it stops scaling linearly, the measurement — not the crowd — has
changed, and that is the edge of the usable window.

Reading the output
------------------
    stride 1..N     frames apart
    displacement    mean px over a fixed set of people
    velocity        displacement * (source_fps / stride), px/s
    deviation       velocity relative to the consistent window

A flat velocity column means the measurement is trustworthy across that
range.  Pick the largest stride still inside the flat region: it has the best
signal-to-noise and the lowest cost.
"""
from __future__ import annotations

import argparse
import logging
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
import yaml

from models.crowd_flow.flow_field import FlowField

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("calibrate_stride")

# Velocity may deviate this far from the agreed window before a stride is
# judged to have left the linear regime.  10% is well inside the spread of
# ordinary pedestrian speeds, so a stride flagged here is failing for
# measurement reasons rather than because the crowd changed.
DRIFT_TOLERANCE = 0.10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="configs/crowd_flow.yaml")
    ap.add_argument("--camera", default="default")
    ap.add_argument("--start", type=int, default=600, help="first frame to sample")
    ap.add_argument("--frames", type=int, default=81, help="frames to load")
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = {}
    if Path(args.config).exists():
        from config_io import load_yaml_dict
        cfg = load_yaml_dict(args.config).get("crowd_flow", {})
    cam = cfg.get("cameras", {}).get(args.camera, {})
    roi = tuple(cam.get("far_field_roi") or cfg.get("far_field_roi", (0.0, 0.0, 1.0, 0.45)))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("cannot open %s", args.source)
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frames = []
    for _ in range(args.frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < max(args.strides) + 2:
        log.error("only %d frames read; need more than max stride", len(frames))
        return 1
    H, W = frames[0].shape[:2]

    # A FIXED set of subjects.  Selecting pixels by "is it moving" instead
    # would compare different populations at each stride — at stride 1 only
    # the fastest clear the floor, at stride 12 nearly everyone does — and the
    # mean velocity would fall for that reason alone, which looks exactly like
    # the failure this script exists to detect.
    from models._detectors import get_detector, COCO_PERSON
    det = get_detector(device=args.device)
    det.load()
    boxes = det.detect(frames[0], classes=(COCO_PERSON,),
                       conf_threshold=0.30, tile_grid=(3, 3))
    subj = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        subj[max(0, int(y1 + bh * .2)):min(H, int(y2 - bh * .2)),
             max(0, int(x1 + bw * .2)):min(W, int(x2 - bw * .2))] = True
    if subj.sum() < 500:
        log.error("only %d people found; too few to calibrate on", len(boxes))
        return 1

    print(f"\nsource   : {Path(args.source).name}  {W}x{H}  {src_fps:.2f} fps")
    print(f"subjects : {len(boxes)} people ({subj.mean()*100:.1f}% of frame), "
          f"held fixed across every stride\n")
    print(f"{'stride':>6} {'eff.fps':>8} {'displacement':>13} "
          f"{'velocity px/s':>14} {'dev':>8}")
    print("-" * 56)

    rows = []
    for k in args.strides:
        ff = FlowField(
            target_px=cfg.get("downsample_target_px", 480),
            far_field=cfg.get("far_field_refinement", True),
            far_field_roi=roi,
            validity_gating=cfg.get("validity_gating", True),
            fb_consistency=cfg.get("fb_consistency", True),
            temporal_smooth_alpha=1.0,     # no EMA: each pair judged alone
            device=args.device,
        )
        ds = []
        for i in range(0, min(len(frames) - k, 40), k):
            res = ff.compute(frames[i], frames[i + k])
            mag = np.hypot(res.field_xy[..., 0], res.field_xy[..., 1])
            valid = (res.valid_mask.astype(bool) if res.valid_mask is not None
                     else np.ones(mag.shape, bool))
            sel = subj & valid
            if sel.sum() > 50:
                ds.append(float(mag[sel].mean()))
        if not ds:
            continue
        d = float(np.mean(ds))
        rows.append((k, d, d * (src_fps / k)))

    # The plateau, not a comparison against stride 1.
    #
    # Anchoring to stride 1 assumes stride 1 is the truth, which is exactly
    # what cannot be assumed: on fast scenes it is stride 1 that sits closest
    # to the noise floor and UNDER-reads.  Measured on this project's traffic
    # clip, velocity rose 32.5 -> 38.9 px/s from stride 1 to 8 and then kept
    # climbing; the trustworthy region was 3-8 (38.25 / 38.57 / 38.93, flat to
    # within 1%) and stride 1 was the outlier.  So the reference is the
    # longest run of consecutive strides that agree with each OTHER.
    best_run: list = []
    for i in range(len(rows)):
        for j in range(len(rows), i, -1):
            window = rows[i:j]
            if len(window) < 2:
                continue
            vels = [v for _k, _d, v in window]
            med = float(np.median(vels))
            if max(abs(v - med) / med for v in vels) <= DRIFT_TOLERANCE:
                if len(window) > len(best_run):
                    best_run = window
                break

    ref = float(np.median([v for _k, _d, v in best_run])) if best_run else None
    for k, d, vel in rows:
        mark = "  <-" if best_run and any(k == kk for kk, _, _ in best_run) else ""
        dev = f"{(vel - ref) / ref * 100:>+7.1f}%" if ref else "      -"
        print(f"{k:>6} {src_fps/k:>8.1f} {d:>10.2f} px {vel:>13.2f} {dev}{mark}")

    print()
    if best_run:
        best = max(k for k, _d, _v in best_run)
        lo = min(k for k, _d, _v in best_run)
        print(f"Consistent window: strides {lo}-{best}, velocity {ref:.1f} px/s "
              f"(flat to within {DRIFT_TOLERANCE*100:.0f}%).")
        print(f"RECOMMENDED: --stride {best}   (sample_every_n_frames: {best})")
        print(f"  {best}x the displacement per measurement against the same "
              f"noise floor,")
        print(f"  and {best}x less compute than stride 1.")
        if rows[0][0] == 1 and abs(rows[0][2] - ref) / ref > DRIFT_TOLERANCE:
            print(f"  NOTE: stride 1 reads {rows[0][2]:.1f} px/s, outside the "
                  f"window.")
            print("        At this frame rate consecutive frames are too close "
                  "together to")
            print("        measure this scene reliably — raising the camera's "
                  "fps would make")
            print("        that worse, not better.")
    else:
        print("No two strides agreed — the flow is not measuring this scene "
              "consistently at any baseline tested.")
    print("\nNote: raising the CAMERA frame rate moves in the opposite "
          "direction.\nIt shrinks the displacement per pair while the noise "
          "floor stays put.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
