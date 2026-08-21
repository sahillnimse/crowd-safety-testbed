# Crowd Safety Model Testbed

Comprehensive computer vision test harness, benchmarking framework, and interactive web evaluation platform for real-time video analytics: dense optical flow & crowd crush analysis (Kumbh Mela engine), fall detection (pose), violence/altercation classification, vehicle tracking & ANPR (number plates), and umbrella detection against real surveillance and crowd footage.

**24 models, no YOLO.** Every detector is Apache 2.0 or MIT — `ultralytics` (AGPL-3.0) is not a dependency, so nothing here carries a copyleft obligation onto surrounding code. Wrappers that need person or vehicle boxes share one RT-DETRv2 detector in [`src/models/_detectors.py`](src/models/_detectors.py) rather than each loading their own backbone.

The dense flow engine ships with **annotation-free accuracy measurement** — three routes that manufacture ground truth without anyone labelling a frame, so "is this measurement any good?" has a number behind it rather than an opinion. See [Validating the Flow Field](#-validating-the-flow-field-3-annotation-free-routes).

---

## 🏗️ Structure

```
crowd-safety-testbed/
├── src/                        # Application source root (add to sys.path / package root)
│   ├── ingestion/
│   │   └── youtube_fetch.py        # yt-dlp wrapper: URL -> local video file
│   ├── models/
│   │   ├── base.py                 # Common BaseModelWrapper interface & Detection schema
│   │   ├── _detectors.py           # SHARED RT-DETRv2 box detector (person / vehicle)
│   │   ├── _tracker.py             # Simple IoU tracker
│   │   ├── _weights.py             # Central model checkpoint search & resolution
│   │   ├── crush/                  # Crowd crush / turbulence — 1 model wrapper
│   │   │   └── optical_flow_crush.py # Farnebäck dense optical flow (circular variance & convergence)
│   │   ├── crowd_flow/             # Kumbh Mela Dense Optical Flow Crowd Safety Engine
│   │   │   ├── dense_flow_analyser.py # DIS flow pipeline wrapper (consumption_type="flow_pair")
│   │   │   ├── flow_field.py       # DIS/Farnebäck flow, GMC guards, smoothing, reliability
│   │   │   ├── ground_plane.py     # Camera perspective calibration & pixel <-> m/s conversion
│   │   │   ├── crowd_metrics.py    # Divergence, curl, coherence, stop-and-go waves, turbulence
│   │   │   ├── zones.py            # Polygonal zone thresholding, hysteresis & alert engine
│   │   │   ├── detector_masks.py   # Vehicle & umbrella exclusion mask layers
│   │   │   ├── visualise.py        # HSV flow overlays, divergence heatmaps, time-series plots
│   │   │   └── validation/         # Annotation-free accuracy measurement (3 routes)
│   │   │       ├── report.py         # Shared result types; every route states its own blind spot
│   │   │       ├── synthetic_warp.py # (a) known warp on a real frame -> endpoint error + sweep
│   │   │       ├── cross_camera.py   # (b) two calibrated views -> ground-plane disagreement
│   │   │       └── cross_family.py   # (c) tracker vs flow, incl. two-arrow comparison video
│   │   ├── fall/                   # Fall detection — 3 model wrappers
│   │   │   ├── mediapipe_pose.py     # MediaPipe BlazePose (+ RT-DETRv2 person detector)
│   │   │   ├── movenet.py            # Google MoveNet multipose
│   │   │   └── optical_flow_fall.py  # Pose-free flow drop heuristic
│   │   ├── violence/               # Violence/altercation detection — 8 model wrappers
│   │   │   ├── roboflow_combined.py  # Roboflow hosted violence & fall classifier
│   │   │   ├── x3d.py                # Lightweight 3D-CNN (pytorchvideo)
│   │   │   ├── slowfast.py           # Dual-pathway 3D-CNN (fast motion sensitive)
│   │   │   ├── videomae.py           # Transformer video classifier
│   │   │   ├── i3d.py                # Inflated 3D ConvNet literature baseline
│   │   │   ├── c3d.py                # Simple 3D-CNN baseline
│   │   │   ├── tsm.py                # Temporal Shift Module on ResNet-50
│   │   │   └── mmaction_slowonly.py  # MMAction2 SlowOnly config/checkpoint pipeline
│   │   ├── traffic/                # Traffic / vehicle tracking — 3 model wrappers
│   │   │   ├── rtdetrv2_traffic.py   # RT-DETRv2-S vehicle detector (Apache 2.0, 20M params)
│   │   │   ├── roboflow_traffic.py   # Roboflow hosted traffic model
│   │   │   └── mog2_parked.py        # MOG2 background subtraction parked car detector
│   │   ├── anpr/                   # Automatic Number Plate Recognition — 4 model wrappers
│   │   │   ├── anpr.py               # RT-DETRv2 vehicle detect + DETR plate crop + EasyOCR + Voting
│   │   │   ├── indian_anpr.py        # Roboflow Indian vehicle/plate detector + EasyOCR
│   │   │   ├── rapid_ocr_wrapper.py  # RapidOCR (PP-OCRv4 ONNX Runtime engine)
│   │   │   └── rtdetrv2_anpr.py      # RT-DETRv2 vehicle/color classification + RapidOCR/EasyOCR
│   │   └── umbrella/               # Umbrella detection & crowd density — 4 model wrappers
│   │       ├── umbrella_ssd.py       # SSDLite320 + MobileNetV3 (CPU-friendly)
│   │       ├── umbrella_rfdetr.py    # RF-DETR Nano (DINOv2 backbone for occluded umbrellas)
│   │       ├── umbrella_rtdetrv2.py  # RT-DETRv2-S COCO zero-shot detector
│   │       └── umbrella_trained.py  # RT-DETRv2 fine-tuned single-class umbrella model
│   ├── pipeline/
│   │   ├── frame_buffer.py         # Sliding window buffer for temporal clip models
│   │   ├── runner.py               # Main pipeline runner (video -> frames -> models -> results)
│   │   ├── annotate.py             # Box interpolation, centered smoothing & ffmpeg H.264 writer
│   │   └── device.py               # Central PyTorch CUDA / CPU resolution & hardware reports
│   └── webapp/                     # FastAPI Backend & Web Dashboard
│       ├── app.py                  # API endpoints, video streaming, upload, history & ANPR routes
│       ├── jobs.py                 # Background JobManager & thread worker pool with GPU locking
│       ├── registry.py             # Dynamic model catalog & live availability checker
│       ├── history.py              # Disk-backed run scanner (survives server restarts)
│       ├── validation.py           # Background runner for the dense-flow validation routes
│       └── frontend/               # Dashboard frontend (HTML5, Vanilla CSS, JS)
├── vendor/
│   └── apgcc/                  # Vendored APGCC crowd counter (upstream, unmodified)
│       ├── config.py, models/, util/, configs/
│       └── ...                 # Loaded dynamically by src/models/head_count/_apgcc_loader.py
├── tests/
│   ├── validate_flow.py        # 18 synthetic contract tests (sign conventions, GMC, ...)
│   └── validate_flow_routes.py # The 3 annotation-free accuracy routes + config sweep
├── docs/
│   └── Final_Models.md         # Model registry reference
├── configs/
│   ├── crowd_flow.yaml         # Camera calibration, polygons, and zone threshold configs
│   └── test_videos.yaml        # Test video catalog and ground-truth evaluation windows
├── scripts/                    # Operational CLI scripts (all add src/ to sys.path)
│   ├── setup.sh                # Environment setup
│   ├── run_single.py           # Single model CLI runner
│   ├── run_all.py              # Multi-model batch test runner
│   ├── compare_models.py       # Head-to-head metrics comparison table
│   ├── calibrate_optical_flow.py  # Optical flow threshold calibration CLI
│   ├── calibrate_ground_plane.py  # Homography builder (px -> m/s) per camera
│   └── run_flow_analysis.py       # Dense-flow CLI: video/glob/RTSP -> CSV, video, plots
├── outputs/                    # ALL generated artifacts (see outputs/README.md)
│   ├── runs/<video>/<model>/   #   detections.json, detections.csv, annotated.mp4
│   ├── anpr/<video>/           #   vehicle portraits, plate crops, manifest.json
│   └── validation/             #   TEST artifacts — not model output
├── main.py                     # Repo entry point (adds src/ to sys.path)
└── requirements.txt
```

---

## 🎯 Model Names (24 models)

**Fall detection (3):**
`fall_mediapipe_pose`, `fall_movenet`, `fall_optical_flow`

**Violence / altercation (8):**
`roboflow_combined`, `violence_x3d`, `violence_videomae`, `violence_slowfast`, `violence_i3d`, `violence_c3d`, `violence_tsm`, `violence_mmaction_slowonly`

**Traffic / vehicle counting (3):**
`rtdetrv2_traffic`, `roboflow_traffic`, `mog2_parked`

**ANPR / number plates (4):**
`anpr`, `indian_anpr`, `rapid_ocr`, `rtdetrv2_anpr`

**Umbrella detection (4):**
`umbrella_ssd`, `umbrella_rfdetr`, `umbrella_rtdetrv2`, `umbrella_trained`

**Crowd crush (2):**
`optical_flow_crush`, `dense_flow`

---

## 📊 Comparing Models Head-to-Head

Once `run_all.py` has produced a combined log for a video/category:

```bash
python scripts/compare_models.py outputs/runs/<video>/<model>/detections.json --category fall
```

Add `ground_truth: [{start_sec, end_sec}, ...]` windows per video in `configs/test_videos.yaml` to get approximate precision, recall, and F1 per model instead of just raw detection counts.

---

## ⚙️ Model Checkpoints & Execution Modes

Not every wrapper can produce a real verdict from an off-the-shelf checkpoint. Rather than letting models emit confident-looking noise, each wrapper either degrades to a clearly-tagged fallback or reports itself blocked until weights are supplied.

| Model | With no weights supplied |
|---|---|
| `fall_mediapipe_pose`, `fall_movenet` | **Fully working.** Pose backbone + posture heuristic; RT-DETRv2 supplies person boxes. |
| `fall_optical_flow`, `optical_flow_crush`, `dense_flow` | **Fully working.** Classical CV / DIS optical flow, nothing to train. |
| `violence_x3d`, `violence_slowfast`, `violence_i3d`, `violence_videomae` | **Working zero-shot.** Kinetics-pretrained, scored by summing probability over Kinetics' fighting classes (`punching person`, `wrestling`, `slapping`, ...), tagged `extra.scoring="kinetics_zeroshot"`. |
| `violence_c3d`, `violence_tsm`, `violence_mmaction_slowonly` | **Tagged fallback / blocked.** Random binary head without fine-tuning; tagged `violence_untrained` and excluded from positive event counts. |
| `anpr`, `rapid_ocr`, `rtdetrv2_anpr` | **Fully working.** Multi-stage vehicle capture + license plate detection + OCR engine (EasyOCR / RapidOCR PP-OCRv4 ONNX). |
| `umbrella_ssd`, `umbrella_rtdetrv2`, `umbrella_rfdetr` | **Fully working.** COCO / open-vocab umbrella detection. `umbrella_trained` requires `umbrella_v1_best.zip` unzipped in `ML Models/umbrella_trained/`. |

Fine-tuning any violence model on **RWF-2000** (real CCTV/surveillance footage — the closest domain match), Hockey Fight, or RLVS and passing `weights_path` switches it to a binary head, which wrappers detect automatically.

---

## 🥊 Violence Models Crop to People First (`use_person_roi`)

Feeding a wide surveillance frame straight into a Kinetics model fails silently — scoring near zero on footage that plainly contains a fight. Measured on this repo's CCTV clip:

| Framing | Peak violence score | Top class |
|---|---|---|
| Centre-crop (standard Kinetics preprocessing) | 0.007 | — |
| Full-frame resize | 0.013 | — |
| **Person-ROI crop** | **0.85** | **punching person (boxing)** |

Two compounding reasons, both fixed by cropping to detected people:
- **Position:** Centre-cropping a 16:9 frame keeps only the middle ~56% of width. In surveillance clips, fighters are often near edges.
- **Scale:** People occupy ~4.5% of frame area (~47×47 px downscaled). Kinetics models expect action filling the frame.

A quiet window in the same video stayed at 0.001 with cropping enabled, sharpening signal without inflating scores.

---

## 🌊 Dense Flow Analysis Engine (`dense_flow`)

The Kumbh Mela dense optical flow module ([dense_flow_analyser.py](file:///c:/Users/sahil/Downloads/Projects/crowd-safety-testbed/src/models/crowd_flow/dense_flow_analyser.py)) is designed for high-density crowd safety monitoring:

- **DIS Optical Flow Field (`flow_field.py`)**: Computes dense motion vectors at downsampled compute resolution (480px default, see below), featuring Global Motion Compensation (GMC) to strip out camera shake, temporal smoothing, and rain/low-light gradient reliability gating.
- **Ground Plane Calibration (`ground_plane.py`)**: Transforms pixel velocities into real physical speed (`m/s`) via camera homography or height/pitch perspective calibration.
- **Crowd Metrics Engine (`crowd_metrics.py`)**:
  - **Divergence ($\nabla \cdot \mathbf{v}$)**: Spatial compression signature ($\text{negative} = \text{crowd compression / crush risk}$).
  - **Curl ($\nabla \times \mathbf{v}$)**: Rotational flow / turbulence.
  - **Helbing Turbulence Index**: Velocity variance divided by squared mean speed ($Var(V) / \bar{V}^2$).
  - **Stop-and-Go Wave Detection**: Short-lag temporal autocorrelation of zone mean speeds.
  - **Counterflow Score**: Fraction of cells moving opposingly to the dominant corridor direction.
- **Zone Alert Engine (`zones.py`)**: Evaluates user-defined polygonal zones with hysteresis ($T_{\text{clear}} = T_{\text{fire}} \times (1 - \text{hysteresis})$) and minimum duration gating (`min_duration_sec`). Zone polygons may be given in **pixels or as fractions of the frame** (all values in `[0, 1]`); fractional polygons are resolved against the real frame size at run time. A pixel polygon authored for one resolution silently measures the wrong region on any other, so mismatches are logged as WARNING and zones fully outside the frame are dropped rather than reporting zeros.
- **Exclusion Masks (`detector_masks.py`)**: Masks out moving vehicles and held umbrellas so non-pedestrian motions do not corrupt crowd statistics. Leaving `vehicle_model_path` empty means vehicles are **not** excluded — a rigid vehicle then contributes strong divergence at its motion boundary, which is a boundary artifact, not crowd compression.

### Global Motion Compensation is guarded, not trusted

GMC removes apparent whole-frame motion from camera sway. Its failure mode is worse than not running it at all: if the estimator locks onto scene content instead of the background, subtracting that estimate injects a uniform velocity into every pixel, and a still scene reads as fully in motion. Three guards apply, in order:

| Guard | Rejects | Catches |
|---|---|---|
| **Inlier ratio** (`≥ 0.55`) | Transforms supported by a minority of features | The estimator locking onto moving vehicles or a frame-filling crowd |
| **Magnitude cap** (`gmc_max_correction_px`) | Corrections implausible as camera sway | Gross outliers (measured up to **179 px/frame** on real traffic footage) |
| **Does it help?** (`gmc_min_improvement`) | Corrections that do **not** reduce the field's median magnitude | A static camera, where ORB returns a self-consistent sub-pixel shift at 0.96 inlier ratio that is pure noise |

The third guard is decisive and cheap (one median per frame). Compensation exists to quiet the background; one that leaves the field noisier is wrong however confident the feature matcher was. Rejected frames are logged with `gmc_method = "rejected"` and no correction is applied — always safer than subtracting a wrong global vector.

### Source-quality flags

Optical flow is undefined on frames that are not temporally adjacent, and the module says so rather than emitting confident numbers:

- **Temporal discontinuity** — the previous frame is warped by the computed flow and compared to the current one. If the flow explains almost none of a *substantial* frame difference, the pair is flagged (`discontinuity_flag` in the CSV, `SCENE-CUT?` in the video banner). Gated on the baseline difference so a quiet camera is not mistaken for a broken one. **Flagged, never suppressed** — a source that is discontinuous throughout would otherwise freeze the field and hide the problem.
- **Frozen source** — a sustained run of identical frames (a stalled feed, or a container padded above its real frame rate) is reported once per run. A single duplicate is an ordinary dropped frame and is not warned about.
- **Brightness jump** — detected as a shift in the frame's **mean grey level**, which only a global illumination change produces. Mean *absolute* difference grows with scene motion, so using it suppresses a large fraction of perfectly good frames on a busy or low-frame-rate camera.

### Units and display

- **Divergence and curl are reported per grid cell** — the velocity difference accumulated across one cell width, i.e. `(∂vx/∂x + ∂vy/∂y) × grid_cell_px`. Reporting the raw per-pixel derivative makes the value depend on compute resolution, since the field is computed downsampled and upsampled back; at the default settings that is a factor of ~4, putting measured values an order of magnitude below the configured thresholds so no divergence alert could ever fire.
- **Overlays use a per-pixel alpha with an adaptive floor.** A uniform alpha tints everything that is perfectly still and washes the frame out. The display floor is `max(min_draw_magnitude_px, background_floor_factor × frame median magnitude)` — the median is a robust estimate of "what not moving looks like on this frame", so the overlay adapts to each source's own noise level instead of a constant tuned elsewhere. **Display only; metrics and alerts are unaffected.**
- **Sampled runs write the annotated video at `source_fps / stride`.** `predict()` is called once per *sampled* frame, so writing at the source rate replays the footage `stride` times too fast — and any speed judged by eye off that video is wrong by the same factor.

### Measured configuration

Endpoint error against known synthetic warps on a 1280×720 traffic camera, via `validate_flow_routes.py --routes a --sweep`:

| Configuration | Mean endpoint error | Flow cost |
|---|---|---|
| `target_px=640`, DIS medium | 0.193 px | 127 ms/frame |
| **`target_px=480`, DIS medium** (default) | **0.228 px** | **105 ms/frame** |
| `target_px=320`, DIS medium | 0.326 px | 92 ms/frame |
| `target_px=320`, Farnebäck | 0.406 px | — |
| `target_px=320`, DIS fast | 0.475 px | — |

480 px buys 30% lower error for 13% more time. The ranking is source-dependent — re-run the sweep when the footage changes.

**Throughput is roughly 2.4 fps at 1080p on CPU with visualisation on** — not real time. Metrics and overlays run at source resolution while the field only carries information at the compute resolution, so the cheapest lever is a smaller source; `visualise=False` removes the rendering term entirely for headless deployments.

---

## 🔬 Validating the Flow Field (3 annotation-free routes)

Nobody has hand-labelled the true velocity of every person in a crowd, and for thousands of people nobody ever will. Instead of finding ground truth, these routes **manufacture** it three ways, chosen so each covers the others' blind spots.

| | Route | Exact error? | Real motion? | High density? |
|---|---|---|---|---|
| **(a)** | Synthetic warp — displace a real frame by a field you define | **yes** | no | **yes** |
| **(b)** | Cross-camera — two calibrated views must agree on the ground plane | bound only | **yes** | **yes** |
| **(c)** | Cross-family — a detector+tracker gives an independent velocity | approximate | **yes** | no |

Only **(b)** is simultaneously real-motion and high-density, which makes it the load-bearing route for crush conditions and the one most dependent on calibration quality. It needs two cameras viewing overlapping ground; with none configured it reports **skipped**, never *pass* — an unrun route must not read as a satisfied one.

**Route (c) earns its keep.** It is the only route that sees independently moving objects, which is exactly what a synthetic warp cannot contain. Verified by injecting the GMC failure mode directly:

| Injected whole-field corruption | Route (c) | Speed error | Direction error | Agreement |
|---|---|---|---|---|
| none | **PASS** | 0.49 px/f | 22.7° | 60.0% |
| +4 px/frame | **FAIL** | 2.52 px/f | 131.7° | 0.45% |
| +15 px/frame | **FAIL** | 13.49 px/f | 136.3° | 0.00% |

All 11 synthetic contract tests pass clean on the same corruption.

### Reading the output

Each route reports its numbers **and an explicit `caveat` naming what it cannot tell you**, because green results are a necessary condition for trusting the estimator and never a sufficient one.

Two distinctions the routes are careful about, both of which otherwise train people to ignore the suite:

- **"Not enough evidence" is reported as SKIPPED, not FAIL.** A COCO person detector largely fails on overhead views and umbrella-covered crowds; that is a detector limitation and says nothing about the flow. Route (c) requires 50+ comparisons before judging anything.
- **A correlation is only judged when the data can support it.** On a near-stationary clip, tracked "velocity" is mostly detection-box jitter, so correlating it against a correctly-near-zero field yields `r ≈ 0` — which means the clip was quiet, not that the estimator is wrong. Below a minimum speed spread the correlation is reported for information only.

None of the three validates whether the **derived** metrics (divergence, counterflow, turbulence) predict crush risk. They validate that the velocity field is measured correctly. Whether a divergence of −1.5 per cell actually means danger is a separate question.

### The comparison video

Route (c) can write a video making the check visible instead of statistical. Every tracked person carries two arrows from a shared origin dot:

- **White** — the tracker's velocity (bounding-box motion)
- **Cyan** — dense optical flow's velocity (pixel motion)

Agreement reads as one thick arrow and a green box; disagreement as a visible V, a red box, and the angle between them. Both arrows use the same exaggeration (printed on frame), since pedestrian motion of 1–2 px/frame would otherwise be a few pixels long and unjudgeable. Medians tell you how well the two agree overall; only the video tells you *where* and *when* they diverge.

```bash
# All three routes, with the comparison video
python tests/validate_flow_routes.py --source "test_videos/Nashik Crowd.mp4" \
    --routes abc --comparison-video

# Route (a) plus a configuration sweep
python tests/validate_flow_routes.py --source test_videos/clip.mp4 --routes a --sweep

# Route (b) machinery self-test on a synthetic camera pair (no real pair needed)
python tests/validate_flow_routes.py --source test_videos/clip.mp4 \
    --routes b --selftest-cross-camera
```

---

## 💳 Automatic Number Plate Recognition (ANPR)

The ANPR subsystem ([anpr.py](file:///c:/Users/sahil/Downloads/Projects/crowd-safety-testbed/src/models/anpr/anpr.py)) captures each tracked vehicle and exports a photo gallery to `outputs/anpr/<video>/`:

```
outputs/anpr/<video>/
├── vehicles/vehicle_0007.jpg   # sharpest, largest portrait of that vehicle
├── plates/plate_0007.jpg       # cropped license plate image
└── manifest.json               # plate, vehicle class, colour, and timings
```

### Key Technical Mechanisms:
1. **Binding Resolution Constraint**: License plates must be physically **~90 px wide or larger** in frame. On distant camera shots (e.g. 60×18 px plates), characters are ~8 px tall and unresolvable by any OCR. Vehicles are still reported with `plate_status: "too_small"` and measured pixel widths.
2. **Multi-Frame Voting (`PlateVote`)**: Every frame a vehicle is visible feeds a temporal voting log (`_plate_text.py`) so a single blurred frame does not corrupt the final plate reading.
3. **Format-Aware Indian Plate Correction**: Automatically corrects standard OCR character confusions (`5`/`S`, `0`/`O`, `1`/`I`, `8`/`B`). Enforces Indian standard format (`LL DD L(1-3) DDDD`) and handles special cases like Delhi RTO codes (`DL 8C AF 5030`).
4. **Portrait Scoring**: Vehicle portraits are saved by maximizing $\text{area} \times \text{sharpness}$ ($\text{Laplacian variance}$), avoiding blurry or edge-of-frame crops.

---

## ☂️ Umbrella Detection & Crowd Density (4 Models)

Umbrella models estimate crowd density and serve as an overhead rain proxy (which impacts walking speeds and occludes torso keypoints for fall detectors).

| Model | Weights / Backbone | Approach & Characteristics |
|---|---|---|
| `umbrella_ssd` | 13.8 MB | SSDLite320 + MobileNetV3. High-precision / lower recall, CPU viable. |
| `umbrella_rfdetr` | RF-DETR Nano | DINOv2 backbone. Superior recall on small/occluded umbrellas in dense crowds. |
| `umbrella_rtdetrv2` | 20M params | RT-DETRv2-S zero-shot transformer detector (Apache 2.0). |
| `umbrella_trained` | 42.7M params | Fine-tuned single-class RT-DETRv2 umbrella checkpoint (F1 0.711). |


---

## 🚘 Traffic Models & Frame Sampling Rules

1. **Low Frame Stride Required (Stride 1 or 2)**: Centroid drift classification (`vehicle_moving` vs `vehicle_parked`) relies on IoU frame association. High sampling strides (e.g. stride 5) cause fast vehicles to jump across frames, fragmenting track IDs. The pipeline prints a warning when the sampling rate drops below ~10 fps.
2. **`mog2_parked` Limitation**: MOG2 background subtraction contrasts fast- and slow-adapting models to detect the *transition* of a vehicle coming to a stop. Cars already parked at frame 0 are absorbed into the background model and require an object detector model (`rtdetrv2_traffic`).

---

## 🏗️ Design Notes & Technical Framework

- **Consumption Types**:
  - `"frame"`: Single image input (traffic, ANPR, umbrella).
  - `"clip"`: Sliding window list of $N$ frames (`FrameBuffer`). Sized to model `clip_len` and re-run every `clip_stride` frames to eliminate 97% redundant computation.
  - `"flow_pair"`: Consecutive frame pair $(t-1, t)$ for optical flow metrics.
- **Interpreting Confidence**: `Detection.confidence` represents **the probability of the reported event label** (e.g., fall or violence), not the generic person-detector score. Upstream detector confidence is preserved in `extra.detector_confidence`.
- **Shared Fall Components**:
  - `src/models/fall/_geometry.py`: Torso angle calculation, aspect ratio, posture scoring.
  - `src/models/fall/_tracker.py`: Greedy 1-to-1 IoU tracking + $K$-consecutive frame confirmation (`sustained()`). Also supplies the independent velocity estimate for dense-flow validation route (c).
  - `src/models/fall/_skeleton.py`: Keypoint normalization and Gaussian heatmap rendering.

---

## 💻 Interactive Web Application

Start the web server:

```bash
python -m src.webapp
```

Then navigate to **`http://127.0.0.1:8000`**.

- **One Global Confidence Threshold**: a single slider in Execution Settings applies to every selected model. Per-model sliders were a false affordance — comparing detectors only means something when they are judged at the same operating point, and a grid of sliders invites tuning each until it looks good, which is how a benchmark stops measuring anything. Classical-CV models with no confidence score are skipped rather than given a meaningless value.
- **Sidebar Model Registry**: Displays model status badges (`ready`, `fallback`, `blocked`) based on local checkpoint availability without paying PyTorch startup import costs.
- **Video Ingestion**: YouTube URL downloading & caching (`src/ingestion/youtube_fetch.py`), local `test_videos/` selection, or drag-and-drop file upload.
- **GPU Thread Pool Manager**: `JobManager` handles asynchronous background execution with single-GPU thread locks to prevent out-of-memory crashes.
- **Interactive Analytical Modal**: Inspect key KPIs, detection timelines, browser-compatible H.264 annotated video streams (`_FFmpegH264Writer`), raw JSON payloads, and — for `dense_flow` results — a **Validation** tab carrying the three routes and the two-arrow comparison video. Opening that tab on any other model says so plainly rather than showing flow numbers beside an unrelated result.
- **Dense Flow Validation Panel**: *Run validation* / *Delete output* / *Refresh* against the selected video. Skipped routes are called out (*"the picture is incomplete, not clean"*) and measurements with no tolerance render visually neutral, so a number that was never judged cannot be misread as one that passed. Deletion is refused while a run is in progress, since the worker would rewrite the files moments later.
- **ANPR Gallery Viewer**: Browse captured vehicle portraits, plate crops, color classifications, and voting confidence.
- **Durable Disk History**: `webapp/history.py` walks `outputs/runs/<video>/<model>/` to preserve past runs across server restarts and CLI runs. It reads directory structure rather than parsing `<video>_<model>.json` filenames — that split was ambiguous for any video name containing an underscore, and prefix-matching meant deleting `clip` also matched `clip_2`.

> The validation runner is a separate thread that does **not** hold `JobManager`'s GPU lock, so its person detector is pinned to CPU. On a 4 GB card two networks will OOM each other; keeping it off the GPU removes the interaction entirely and means validation never has to queue behind a long model run. Verified at 0 KB peak GPU memory.

---

## ⚡ CLI Quickstart

```bash
# 1. Environment Setup
bash scripts/setup.sh

# 2. Run Single Model Test
python scripts/run_single.py --video test_videos/clip.mp4 --model fall_movenet

# 3. Batch Evaluation Suite
python scripts/run_all.py --config configs/test_videos.yaml

# 4. Calibrate Dense Flow Thresholds
python scripts/calibrate_optical_flow.py --video test_videos/crowd_sample.mp4

# 5. Head-to-Head Model Benchmarking
python scripts/compare_models.py outputs/runs/<video>/<model>/detections.json --category fall

# 6. Dense Flow Analysis (video file, image glob, or rtsp:// stream)
python scripts/run_flow_analysis.py --source test_videos/crowd_sample.mp4 \
    --camera default --output-dir outputs/flow_run_001

# 7. Dense Flow Correctness — synthetic contract tests (sign conventions, GMC, calibration)
python tests/validate_flow.py --all

# 8. Dense Flow Accuracy — the 3 annotation-free routes, with the comparison video
python tests/validate_flow_routes.py --source test_videos/crowd_sample.mp4 \
    --routes abc --comparison-video
```

**`tests/validate_flow.py` vs `tests/validate_flow_routes.py`** — the first asks *"are the contracts intact?"* (sign conventions, unit handling, internal self-tests) on synthetic data, and must pass before a run is trusted at all. The second asks *"how wrong is it on real imagery?"* and produces the error figures. Both are needed: the eighteen contract tests passed clean while GMC was corrupting real footage by up to 179 px/frame, because synthetic frames contain no independently moving objects for the estimator to lock onto.

---

## 📄 License & References
- **Core Testbed**: MIT License
- **RT-DETRv2 Models**: Apache 2.0 License (`PekingU/rtdetr_v2_r18vd`)
- **RF-DETR Models**: Roboflow RF-DETR DINOv2 Engine
