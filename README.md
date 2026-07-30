# Crowd Safety Model Testbed

Test harness for evaluating fire/smoke detection, fall detection (pose),
violence/altercation classification, and crowd-crush (optical flow)
models against real YouTube footage.

## Structure

```
crowd-safety-testbed/
├── ingestion/
│   └── youtube_fetch.py       # yt-dlp wrapper: URL -> local video file
├── models/
│   ├── base.py                 # Common interface all model wrappers implement
│   ├── fire_smoke_yolo.py      # Fire/smoke YOLO wrapper
│   ├── optical_flow_crush.py   # Farnebäck dense optical flow (crowd crush/turbulence)
│   ├── fall/                   # Fall detection — 7 model wrappers
│   │   ├── yolo_pose.py          # YOLOv8-pose + geometric heuristic
│   │   ├── mediapipe_pose.py     # MediaPipe BlazePose (+ YOLO person detector for crowds)
│   │   ├── alphapose_lstm.py     # AlphaPose keypoints + temporal LSTM classifier
│   │   ├── stgcn.py              # ST-GCN skeleton-graph classifier
│   │   ├── posec3d.py            # PoseC3D (MMAction2) heatmap-volume 3D-CNN
│   │   ├── movenet.py            # Google MoveNet (single/multipose variants)
│   │   └── optical_flow_fall.py  # Pose-free: flow-based sudden-drop heuristic
│   └── violence/                # Violence/altercation detection — 7 model wrappers
│       ├── x3d.py                # Lightweight 3D-CNN (pytorchvideo)
│       ├── slowfast.py           # Dual-pathway 3D-CNN, strong on fast motion
│       ├── videomae.py           # Transformer, masked video autoencoding pretrain
│       ├── i3d.py                # Inflated 3D ConvNet, common literature baseline
│       ├── c3d.py                # Simple 3D-CNN baseline
│       ├── tsm.py                # Temporal Shift Module on ResNet-50 (efficient)
│       └── mmaction_slowonly.py  # MMAction2 SlowOnly config/checkpoint pipeline
├── pipeline/
│   ├── frame_buffer.py         # Sliding window buffer for clip-based models
│   ├── runner.py                # Orchestrates: video -> frames -> models -> results
│   └── annotate.py              # Burns detections onto frames, exports video
├── configs/
│   └── test_videos.yaml        # List of test videos + expected events (ground truth)
├── test_videos/                # Downloaded videos land here (gitignored)
├── outputs/
│   ├── annotated/               # Annotated output videos
│   └── logs/                    # Per-run JSON/CSV detection logs
├── scripts/
│   ├── setup.sh                  # Installs dependencies
│   ├── run_single.py            # Test one model on one video
│   ├── run_all.py               # Run full stack on full test_videos.yaml set
│   └── compare_models.py        # Head-to-head comparison table across models in a category
└── requirements.txt
```

## Model names (for --model / configs/test_videos.yaml)

**Fall detection:** `fall_yolo_pose`, `fall_mediapipe_pose`, `fall_alphapose_lstm`,
`fall_stgcn`, `fall_posec3d`, `fall_movenet`, `fall_optical_flow`

**Violence/altercation:** `violence_x3d`, `violence_slowfast`, `violence_videomae`,
`violence_i3d`, `violence_c3d`, `violence_tsm`, `violence_mmaction_slowonly`

**Other:** `fire_smoke_yolo`, `optical_flow_crush`

## Comparing models head-to-head

Once `run_all.py` has produced a combined log for a video/category:

```bash
python scripts/compare_models.py outputs/logs/<video>_<category>.json --category fall
```

Add `ground_truth: [{start_sec, end_sec}, ...]` windows per video in
`configs/test_videos.yaml` to get approximate precision/recall/F1 per model
instead of just raw detection counts.

## What each model needs to actually detect something

Not every wrapper can produce a real verdict from an off-the-shelf
checkpoint. Rather than letting the ones that can't emit confident-looking
noise, each either degrades to a clearly-tagged fallback or refuses to load.

| Model | With no weights supplied |
|---|---|
| `fall_yolo_pose`, `fall_mediapipe_pose`, `fall_movenet` | Fully working. Pose backbone + posture heuristic, no extra weights needed. |
| `fall_stgcn`, `fall_posec3d`, `fall_alphapose_lstm` | Falls back to the geometric posture verdict, tagged `extra.scoring="geometric_fallback"`. Supply a trained checkpoint to use the actual network. |
| `fall_optical_flow`, `optical_flow_crush` | Fully working. Classical CV, nothing to train. |
| `violence_x3d`, `violence_slowfast`, `violence_i3d`, `violence_videomae` | Working zero-shot: Kinetics-pretrained, scored by summing probability over Kinetics' fighting classes (`punching person`, `wrestling`, `slapping`, ...), tagged `extra.scoring="kinetics_zeroshot"`. |
| `violence_c3d`, `violence_tsm`, `violence_mmaction_slowonly` | **Raises at `load()`.** No pretrained checkpoint exists for these here, so an untrained binary head would label ~half of all clips "violence" at ~0.5 confidence. Fine-tune first or drop them from the run. |

Fine-tuning any violence model on **RWF-2000** (real CCTV/surveillance
footage — the closest domain match for this testbed's clips), Hockey Fight,
or RLVS and passing `weights_path` switches it to a binary head, which the
wrappers detect automatically from the checkpoint's final layer.

### Violence models crop to the people first (`use_person_roi`)

Feeding a wide surveillance frame straight into a Kinetics model does not
work, and it fails silently — scoring near zero on footage that plainly
contains a fight. Measured on this repo's own CCTV clip:

| Framing | Peak violence score | Top class |
|---|---|---|
| Centre-crop (standard Kinetics preprocessing) | 0.007 | — |
| Full-frame resize | 0.013 | — |
| **Person-ROI crop** | **0.85** | **punching person (boxing)** |

Two compounding reasons, both fixed by cropping to the detected people:

- **Position.** Centre-cropping a 16:9 frame keeps only the middle ~56% of
  its width. In that clip 3 of the 4 fighters were outside it entirely.
- **Scale.** People occupied ~4.5% of frame area — roughly 47x47 px once
  downscaled to the network input. Kinetics models are trained on clips
  where the action fills the frame.

A quiet window in the same video stayed at 0.001 with the crop enabled, so
this sharpens the signal rather than inflating every score. It costs one
YOLOv8-nano pass per clip inference. Disable with `use_person_roi=False`
if your footage is already tightly framed on the subjects.

## ANPR (number plates)

The `anpr` model captures every vehicle it tracks, reads its number plate,
and writes a gallery to `outputs/anpr/<video>/`:

```
outputs/anpr/<video>/
├── vehicles/vehicle_0007.jpg   # sharpest, largest view of that vehicle
├── plates/plate_0007.jpg       # the plate crop it read from
└── manifest.json               # plate, class, colour, timings per vehicle
```

The web UI renders this as a gallery: vehicle photo, with
`white car` and `MH 15 DS 7121` underneath.

**Plates must be big enough in frame — this is the binding constraint.**
Roughly **90 px of plate width** is the floor; below that the characters
are only a few pixels tall and no OCR recovers them. On this repo's own
traffic clip plates peak at 60×18 px and read as nothing, so vehicles are
reported with `plate_status: "too_small"` and the measured width rather
than a blank or a guess. If every plate comes back `too_small`, the camera
is too far from the traffic lane — that is a footage problem, not a model
one.

Two things make the readings reliable when plates *are* large enough:

- **Multi-frame voting.** A vehicle is read on many frames and the most
  agreed-on result wins, so one blurred frame can't corrupt the answer.
- **Format-aware correction.** OCR confuses `5`/`S`, `0`/`O`, `1`/`I`,
  `8`/`B` constantly — a rendered `MH15DS7121` came back as `MH1SDS7121`
  in testing. Indian plates have a fixed layout (`LL DD L(1-3) DDDD`), so a
  letter sitting in a digit slot is unambiguously an OCR error and gets
  corrected. Delhi's letter-bearing RTO codes (`DL 8C AF 5030`) are handled
  as a special case, and the parse that changes the fewest characters wins.

Vehicle **make/model** ("Maruti Suzuki Dzire") is *not* produced — that
needs either a registration-database lookup or a make/model classifier
trained on Indian vehicles. What you get is class + colour ("white car").
`_VehicleRecord` is where a resolver would slot in.

## Umbrella detection — 3 models

All three stay inside the 30 MB budget and share their filtering, counting
and output construction (`models/umbrella/_common.py`), so any difference
in results is a difference in the *detector*.

| Model | Weights | Approach |
|---|---|---|
| `umbrella_yolo` | 5.6 MB (`n`) / 19.3 MB (`s`) | YOLO11, COCO fixed class 25 |
| `umbrella_ssd` | 13.8 MB | SSDLite320 + MobileNetV3, older/lighter family, CPU-viable |
| `umbrella_world` | 25.9 MB | YOLO-World v2, open-vocabulary text prompts |

Measured on the same real crowd photos:

| Image | `yolo` (s) | `ssd` | `world` |
|---|---|---|---|
| Crowd holding umbrellas | 6 (top 0.86) | 4 (top 0.94) | **15** (top 0.82) |
| Crowd with women, angle 1 | 8 (top 0.74) | 2 (top 0.97) | **15** (top 0.95) |
| Crowd with women, angle 2 | 8 (top 0.73) | 1 (top 0.96) | **15** (top 0.96) |
| Thatched beach shelters | 0 | 0 | 0 |

The spread is the point: **SSDLite is high-precision/low-recall** (fewest
boxes, highest confidence), **YOLO sits in the middle**, and
**YOLO-World recalls roughly twice as many** at slightly lower confidence.
Pick by whether missed umbrellas or false ones cost you more.

`umbrella_yolo` rejects `model_size="m"` rather than silently exceeding the
budget (yolo11m is ~40 MB).

### Open-vocabulary finds what you *name*

The thatched-shelter row above is worth understanding. Those structures are
invisible to all three by default. With YOLO-World:

| Prompts | Found |
|---|---|
| `umbrella`, `parasol` | 0 |
| `umbrella`, `parasol`, `canopy` | 0 |
| `umbrella`, `parasol`, `beach hut`, `straw roof` | 0 |
| **`thatched roof shelter`** (+ others) | **35** |

Open-vocabulary does **not** generalise to "things that give shade" — near
synonyms fail outright. If your footage has shade structures you care
about, prompt for them literally:

```python
UmbrellaWorldDetector(prompts=("umbrella", "parasol", "thatched roof shelter"))
```

Each detection records which prompt matched in `extra["matched_class"]`, so
a run can be audited for whether extra recall came from real umbrellas or
from a broad term pulling in awnings and market stalls.

### Notes

Tracking is on by default in all three. Without persistent IDs one umbrella
held for 100 frames is indistinguishable from 100 umbrellas; `track_id`
identifies the individual and `umbrellas_in_frame` gives the live
concurrent count.

`umbrella_world` needs CLIP for its text encoder (in `requirements.txt`).
Without it ultralytics auto-installs mid-run and warns that a restart is
required, which silently degrades that first run.

In a crowd-safety context a rising umbrella count is a proxy for rain
(which bunches crowds and slows walking speed) and a caveat on the
pose-based fall detectors, which degrade when umbrellas occlude torsos from
overhead cameras.

## Traffic models: use a low frame stride

The traffic detectors classify each tracked vehicle as `vehicle_moving` or
`vehicle_parked` from how far its centroid drifts over a time window. Two
things follow:

**Set "Frame sampling" to every 1st or 2nd frame for traffic.** The
parked/moving window itself is measured in seconds and is stride-independent,
but ByteTrack is not — it associates boxes between the frames it is actually
given, so a large stride makes vehicles jump too far to match. On the test
clip, stride 5 more than halved the distinct track count (14 → 6). The
models print a warning when the effective rate drops below ~10 fps.

**`mog2_parked` cannot see a vehicle that was already parked when the clip
started.** It works by contrasting a fast- and a slow-adapting background
model, so it detects the *transition* into being parked. A car parked from
frame 0 is simply part of the background. Use it as a cross-check on the
detector-based models, not as a standalone parked-vehicle census.

## Design notes

- **Frame-based models** (fire/smoke YOLO, pose YOLO) run per-frame, stateless.
- **Clip-based models** pull from a rolling frame buffer
  (`pipeline/frame_buffer.py`). The runner sizes that buffer to the largest
  `clip_len` among the loaded models, waits until a model's `min_clip_frames`
  are actually available before calling it, and re-runs it every
  `clip_stride` frames rather than on every frame — consecutive clips
  otherwise overlap by ~97% at enormous cost.
- **Flow-based** (Farneback) only needs frame[t-1] and frame[t].
- All model wrappers implement a common `.predict(frame_or_clip)` interface
  (see `models/base.py`) so the runner doesn't care which model it's calling.
- Device (CPU/GPU) is auto-detected per model at runtime — no hardware-specific
  code paths needed. Everything runs on CPU by default; if CUDA is available
  it's used automatically.

### Interpreting `confidence`

For every model, `confidence` describes **the reported label**, not the
detector that found the person. A fall at 0.9 means the posture is strongly
horizontal; a violence detection at 0.9 means 0.9 probability mass on
violence. The pose wrappers keep the person-detector's own score in
`extra.detector_confidence` if you need it. This matters because
`compare_models.py` sweeps thresholds over `confidence` — sweeping over
detector confidence, as it previously did, measures nothing.

### Shared fall components

The pose wrappers deliberately share their posture scoring, tracking, and
temporal logic so a difference between them reflects the *backbone*:

- `models/fall/_geometry.py` — keypoint-confidence-gated torso angle, bbox
  aspect ratio, combined posture score, calibrated confidence.
- `models/fall/_tracker.py` — greedy **one-to-one** IoU tracker with age
  eviction, plus `sustained()` for K-consecutive-frame confirmation.
- `models/fall/_skeleton.py` — clip-level (never per-frame) keypoint
  normalization and gaussian heatmap rendering for the sequence models.

## Tuning the optical-flow detectors

Thresholds are footage-dependent. Derive them from the actual video rather
than guessing:

```bash
python scripts/calibrate_optical_flow.py --video test_videos/<file>.mp4
```

It reports percentiles of the exact statistics the detector computes
(circular variance for turbulence, divergence for compression) and suggests
starting thresholds.

## Web UI (recommended)

Replaces the download → edit → run-in-terminal loop. One command:

```bash
python -m webapp
```

Then open <http://127.0.0.1:8000>.

- **Left sidebar** lists every model with a live availability badge —
  `ready`, `fallback` (runs, but in the clearly-tagged degraded mode), or
  `blocked` (needs a checkpoint; the checkbox is disabled so a run can't
  half-fail). Only `ready` models are preselected.
- **Source** is a YouTube URL (downloaded and cached automatically), an
  already-downloaded clip from `test_videos/`, or a direct file upload.
- Each selected model runs as its own **stage** with a live progress bar,
  so one model refusing to load never aborts the rest of the run.
- Results link straight to the annotated video, the detection rows, and the
  JSON/CSV logs. "Delete all outputs" clears `outputs/` without touching
  your source videos.

Jobs run serially against the GPU on purpose — two 3D-CNNs sharing a small
card is a reliable way to OOM both.

## Quickstart (CLI)

```bash
bash scripts/setup.sh
python scripts/run_single.py --video test_videos/clip.mp4 --model fall_yolo_pose
python scripts/run_all.py --config configs/test_videos.yaml
```

## Status

Fall and violence wrappers produce real verdicts (see the table above for
which need checkpoints). `fire_smoke_yolo` still needs a fine-tuned
fire/smoke checkpoint — the wrapper is complete but there are no public
weights bundled here.
