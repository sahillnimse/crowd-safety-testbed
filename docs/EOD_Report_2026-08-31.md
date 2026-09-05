# EOD Report — 31 August 2026

**Subject:** Crowd Safety Testbed — ground-truth validation, model accuracy measured, production hardening

---

## Summary

Three things landed today:

1. **We can now measure model accuracy.** Integrated the SAIVT-BuildingMonitoring
   annotated dataset — 1,020 hand-annotated stills covering 5,095 individually
   marked people across 12 cameras. Until today no model in this project had ever
   been scored against human-verified data.
2. **First measured accuracy figures.** APGCC and RT-DETRv2 scored head-to-head on
   300 stills / 1,088 annotated people. Results below — they contradict an
   assumption we had been building on.
3. **Production hardening.** Nine deployment blockers closed, including one that
   caused a dead camera to be reported as a successful run.

---

## 1. Ground-truth validation capability (new)

Previously, `configs/test_videos.yaml` carried four `ground_truth` entries and all
were empty, so every accuracy claim in the project was unverifiable.

Integrated the SAIVT dataset (Denman et al., 2015, QUT), which provides three
separate kinds of annotation per camera:

| Annotation | What it gives us | Coverage |
|---|---|---|
| Per-frame occupancy | Every person visible, marked by hand | 10 of 12 cameras |
| Gate crossings + direction | 1,836 crossing events | 6 of 12 cameras |
| Perspective calibration | Real-world scale at each image position | 10 of 12 cameras |

Annotations (484 KB) are committed to the repo so scoring works on a fresh
checkout. The imagery (724 MB) and video (8 GB) are referenced by configurable
path rather than duplicated.

---

## 2. Measured model accuracy — first results

Ten cameras, 300 annotated stills, 1,088 annotated people.

| Model | Finds (recall) | Correct when it reports (precision) | F1 | Count error (MAE) |
|---|---|---|---|---|
| APGCC | **44%** | 39% | 0.415 | 2.67 people |
| RT-DETRv2 | **72%** | 47% | 0.568 | 2.40 people |

RT-DETRv2 scored better on 6 of 10 cameras.

### Key finding

**We had been treating APGCC as ~99% accurate and had proposed using its output as
a reference standard.** Measured against human annotation it locates 44% of the
people present. That assumption was feeding into density and crowd-pressure
calculations, and into a plan to use APGCC output in place of real ground truth.
That plan is now withdrawn.

### Direction of error matters

APGCC's error direction varies by camera, from 2.7 people under-counted to 7.3
over-counted, which cancels to a near-zero average and hides the per-camera
problem. RT-DETRv2 over-counts consistently.

For crowd safety these are not equivalent. Under-counting under-states density and
therefore crowd pressure, so a warning fails to fire. Over-counting produces false
alarms — bad for operator trust, but it errs toward warning rather than silence.

### Important caveat

This dataset is an indoor university building averaging **3.6 people per frame**.
Nashik will be two orders of magnitude denser. APGCC is designed for dense crowds
(trained on imagery averaging ~500 people), so it is being tested well outside its
intended operating range, and I would expect this ranking to reverse at Kumbh
densities — box detectors like RT-DETRv2 fail when bodies overlap, which is the
condition density-map counters exist to handle.

**These numbers should not be used to choose an architecture for Nashik.** What
they establish is that the measurement capability works, and that the 99%
assumption was wrong.

---

## 3. Camera calibration — now possible without a site visit

None of the four Nashik cameras has a homography, which disables speed in m/s,
density in persons/m², and all crowd-pressure alerting.

Implemented an alternative that needs no on-site measurement: person height across
the image is fitted from ~30–60 bounding boxes drawn on a single still, which
yields a real-world scale at every pixel. Works for both indoor and outdoor/ghat
cameras.

Validated by re-fitting SAIVT's own annotations and comparing against their
published coefficients: **0.0% disagreement**. During implementation I found the
convention must fit against the body centre rather than the feet — fitting against
feet produced a consistent 40% scale error, which would have propagated into every
speed and density figure.

---

## 4. Production hardening — 9 issues closed

| Issue | Status |
|---|---|
| Dead camera reported as a successful run | Fixed — truncated/lost sources now flagged |
| Single global GPU lock (extra GPUs unusable) | Fixed — per-device pool, verified 4-way concurrency |
| Job state lost on restart | Fixed — persisted; interrupted runs marked as such |
| No live camera (RTSP) support in the server | Added, with automatic reconnect |
| Alerts reached nobody | Added delivery (webhook + log), dispatched live mid-run |
| No API authentication | Added; server now refuses to expose itself unauthenticated |
| Placeholder zones ran silently | Preflight check reports blockers before any frame |
| Density disabled silently killed crush metric | Flagged as a blocker |
| No tests on critical paths | 65 tests added |

The first is the most significant: a camera failing three minutes into a shift
previously terminated the run and reported "All models completed", which in a
control room reads as "this location is monitored and clear".

---

## 5. Other work

- Completed all 10 crowd-safety metrics on the Crowd Motion Monitor (density,
  velocity, specific flow, velocity variance, crowd pressure, divergence,
  directional entropy, counter-flow, stop-and-go, oscillation symmetry) and
  surfaced them as dashboard KPI cards. Two cards were previously unreachable
  due to a conditional that could never be true.
- Removed dead code: two orphaned modules, two obsolete scripts, 17 unused imports.
- Repository size reduced from 8.0 GB to 146 MB.
- Test suite: 55 → 120 tests, all passing.

---

## Next steps

**Ready now**
- Score DM-Count on the same data (adapter bug fixed today; weights were present
  all along)
- Build the flow scorer to validate specific flow and counter-flow against the
  1,836 annotated gate crossings — this is the correct test for Crowd Motion
  Monitor, which cannot be validated on isolated stills

**Requires field work — not a software task**
- Calibrate the four Nashik cameras (now ~an afternoon per camera, no site
  measurement needed)
- Obtain and annotate footage at deployment density — every figure above is
  measured at 3.6 people/frame and does not transfer to Kumbh conditions

**Assessment:** the system is not yet ready for deployment, primarily because no
Nashik camera is calibrated and no crush event has ever been annotated, so the
miss rate of the crush-warning path remains unknown. Both are addressable, and
today's work makes both measurable for the first time.

---

*Sahil Nimse*
