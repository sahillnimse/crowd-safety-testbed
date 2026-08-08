# Crowd Safety Model Testbed

Comprehensive computer vision test harness, benchmarking framework, and interactive web evaluation platform for real-time video analytics: dense optical flow & crowd crush analysis (Kumbh Mela engine), fall detection (pose & skeleton), violence/altercation classification, vehicle tracking & ANPR (number plates), umbrella detection, and fire/smoke detection against real surveillance and crowd footage.

---

## 🏗️ Structure

```
crowd-safety-testbed/
├── ingestion/
│   └── youtube_fetch.py        # yt-dlp wrapper: URL -> local video file
├── models/
│   ├── base.py                 # Common BaseModelWrapper interface & Detection schema
│   ├── _weights.py             # Central model checkpoint search & resolution
│   ├── fire_smoke_yolo.py      # Fire & smoke YOLO detector (with cloud fallback)
│   ├── optical_flow_crush.py   # Farnebäck dense optical flow (circular variance & convergence)
│   ├── roboflow_combined.py    # Roboflow hosted violence & fall classifier
│   ├── crowd_flow/             # Kumbh Mela Dense Optical Flow Crowd Safety Engine
│   │   ├── dense_flow_analyser.py # DIS flow pipeline wrapper (consumption_type="flow_pair")
│   │   ├── flow_field.py       # DIS/Farnebäck flow, GMC, temporal smoothing, reliability
│   │   ├── ground_plane.py     # Camera perspective calibration & pixel -> m/s conversion
│   │   ├── crowd_metrics.py    # Divergence, curl, coherence, stop-and-go waves, turbulence
│   │   ├── zones.py            # Polygonal zone thresholding, hysteresis & alert engine
│   │   ├── detector_masks.py   # Vehicle & umbrella exclusion mask layers
│   │   └── visualise.py        # HSV flow overlays, divergence heatmaps, time-series plots
│   ├── fall/                   # Fall detection — 7 model wrappers
│   │   ├── yolo_pose.py          # YOLOv8-pose + posture angle heuristic
│   │   ├── mediapipe_pose.py     # MediaPipe BlazePose (+ YOLO person detector for crowds)
│   │   ├── alphapose_lstm.py     # AlphaPose keypoints + temporal LSTM classifier
│   │   ├── stgcn.py              # ST-GCN skeleton-graph classifier
│   │   ├── posec3d.py            # PoseC3D heatmap-volume 3D-CNN (MMAction2)
│   │   ├── movenet.py            # Google MoveNet multipose
│   │   └── optical_flow_fall.py  # Pose-free flow drop heuristic
│   ├── violence/               # Violence/altercation detection — 8 model wrappers
│   │   ├── x3d.py                # Lightweight 3D-CNN (pytorchvideo)
│   │   ├── slowfast.py           # Dual-pathway 3D-CNN (fast motion sensitive)
│   │   ├── videomae.py           # Transformer video classifier
│   │   ├── i3d.py                # Inflated 3D ConvNet literature baseline
│   │   ├── c3d.py                # Simple 3D-CNN baseline
│   │   ├── tsm.py                # Temporal Shift Module on ResNet-50
│   │   └── mmaction_slowonly.py  # MMAction2 SlowOnly config/checkpoint pipeline
│   ├── traffic/                # Traffic / vehicle tracking — 5 model wrappers
│   │   ├── yolo_traffic.py       # YOLOv11 + ByteTrack vehicle drift classifier
│   │   ├── rtdetr_traffic.py     # RT-DETR transformer detector + DeepSORT
│   │   ├── rtdetrv2_traffic.py   # RT-DETRv2-S vehicle detector (Apache 2.0, 20M params)
│   │   ├── roboflow_traffic.py   # Roboflow hosted traffic model
│   │   └── mog2_parked.py        # MOG2 background subtraction parked car detector
│   ├── anpr/                   # Automatic Number Plate Recognition — 4 model wrappers
│   │   ├── anpr.py               # YOLO vehicle detect + DETR plate crop + EasyOCR + Voting
│   │   ├── indian_anpr.py        # Roboflow Indian vehicle/plate detector + EasyOCR
│   │   ├── rapid_ocr_wrapper.py  # RapidOCR (PP-OCRv4 ONNX Runtime engine)
│   │   └── rtdetrv2_anpr.py      # RT-DETRv2 vehicle/color classification + RapidOCR/EasyOCR
│   └── umbrella/               # Umbrella detection & crowd density — 7 model wrappers
│       ├── umbrella_yolo.py      # YOLO11 (fixed COCO class 25)
│       ├── umbrella_ssd.py       # SSDLite320 + MobileNetV3 (CPU-friendly)
│       ├── umbrella_world.py     # YOLO-World v2 open-vocabulary text prompts
│       ├── umbrella_yolo26n.py   # YOLO26-Nano NMS-free edge detector
│       ├── umbrella_rfdetr.py    # RF-DETR Nano (DINOv2 backbone for occluded umbrellas)
│       ├── umbrella_rtdetrv2.py  # RT-DETRv2-S COCO zero-shot detector
│       └── umbrella_trained.py  # RT-DETRv2 fine-tuned single-class umbrella model
├── pipeline/
│   ├── frame_buffer.py         # Sliding window buffer for temporal clip models
│   ├── runner.py               # Main pipeline runner (video -> frames -> models -> results)
│   ├── annotate.py             # Box interpolation, centered smoothing & ffmpeg H.264 writer
│   └── device.py               # Central PyTorch CUDA / CPU resolution & hardware reports
├── webapp/                     # FastAPI Backend & Web Dashboard
│   ├── app.py                  # API endpoints, video streaming, upload, history & ANPR routes
│   ├── jobs.py                 # Background JobManager & thread worker pool with GPU locking
│   ├── registry.py             # Dynamic model catalog & live availability checker
│   ├── history.py              # Disk-backed output log scanner (survives server restarts)
│   └── frontend/               # Dashboard frontend (HTML5, Vanilla CSS, JS)
├── configs/
│   ├── crowd_flow.yaml         # Camera calibration, polygons, and zone threshold configs
│   └── test_videos.yaml        # Test video catalog and ground-truth evaluation windows
├── scripts/
│   ├── setup.sh                # Environment setup
│   ├── run_single.py           # Single model CLI runner
│   ├── run_all.py              # Multi-model batch test runner
│   ├── compare_models.py       # Head-to-head metrics comparison table
│   └── calibrate_optical_flow.py # Optical flow threshold calibration CLI
└── requirements.txt
```

---

## 🎯 Model Names (for `--model` / `configs/test_videos.yaml`)

**Fall detection (7):**
`fall_yolo_pose`, `fall_mediapipe_pose`, `fall_movenet`, `fall_optical_flow`, `fall_stgcn`, `fall_posec3d`, `fall_alphapose_lstm`

**Violence / altercation (8):**
`roboflow_combined`, `violence_x3d`, `violence_videomae`, `violence_slowfast`, `violence_i3d`, `violence_c3d`, `violence_tsm`, `violence_mmaction_slowonly`

**Traffic / vehicle counting (5):**
`yolo_traffic`, `rtdetr_traffic`, `rtdetrv2_traffic`, `roboflow_traffic`, `mog2_parked`

**ANPR / number plates (4):**
`anpr`, `indian_anpr`, `rapid_ocr`, `rtdetrv2_anpr`

**Umbrella detection (7):**
`umbrella_yolo`, `umbrella_ssd`, `umbrella_world`, `umbrella_yolo26n`, `umbrella_rfdetr`, `umbrella_rtdetrv2`, `umbrella_trained`

**Fire & Smoke / Crowd Crush (3):**
`fire_smoke_yolo`, `optical_flow_crush`, `dense_flow`

---

## 📊 Comparing Models Head-to-Head

Once `run_all.py` has produced a combined log for a video/category:

```bash
python scripts/compare_models.py outputs/logs/<video>_<category>.json --category fall
```

Add `ground_truth: [{start_sec, end_sec}, ...]` windows per video in `configs/test_videos.yaml` to get approximate precision, recall, and F1 per model instead of just raw detection counts.

---

## ⚙️ Model Checkpoints & Execution Modes

Not every wrapper can produce a real verdict from an off-the-shelf checkpoint. Rather than letting models emit confident-looking noise, each wrapper either degrades to a clearly-tagged fallback or reports itself blocked until weights are supplied.

| Model | With no weights supplied |
|---|---|
| `fall_yolo_pose`, `fall_mediapipe_pose`, `fall_movenet` | **Fully working.** Pose backbone + posture heuristic, no extra weights needed. |
| `fall_stgcn`, `fall_posec3d`, `fall_alphapose_lstm` | Falls back to geometric posture verdict, tagged `extra.scoring="geometric_fallback"`. Supply a trained checkpoint to use the skeleton network. |
| `fall_optical_flow`, `optical_flow_crush`, `dense_flow` | **Fully working.** Classical CV / DIS optical flow, nothing to train. |
| `violence_x3d`, `violence_slowfast`, `violence_i3d`, `violence_videomae` | **Working zero-shot.** Kinetics-pretrained, scored by summing probability over Kinetics' fighting classes (`punching person`, `wrestling`, `slapping`, ...), tagged `extra.scoring="kinetics_zeroshot"`. |
| `violence_c3d`, `violence_tsm`, `violence_mmaction_slowonly` | **Tagged fallback / blocked.** Random binary head without fine-tuning; tagged `violence_untrained` and excluded from positive event counts. |
| `yolo_traffic`, `rtdetr_traffic`, `rtdetrv2_traffic` | **Fully working.** COCO pretrained vehicle class tracking (`vehicle_moving` vs `vehicle_parked`). |
| `anpr`, `rapid_ocr`, `rtdetrv2_anpr` | **Fully working.** Multi-stage vehicle capture + license plate detection + OCR engine (EasyOCR / RapidOCR PP-OCRv4 ONNX). |
| `umbrella_yolo`, `umbrella_ssd`, `umbrella_world`, `umbrella_rtdetrv2`, `umbrella_yolo26n`, `umbrella_rfdetr` | **Fully working.** COCO / open-vocab umbrella detection. `umbrella_trained` requires `umbrella_v1_best.zip` unzipped in `ML Models/umbrella_trained/`. |

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

The Kumbh Mela dense optical flow module ([dense_flow_analyser.py](file:///c:/Users/sahil/Downloads/Projects/crowd-safety-testbed/models/crowd_flow/dense_flow_analyser.py)) is designed for high-density crowd safety monitoring:

- **DIS Optical Flow Field (`flow_field.py`)**: Computes dense motion vectors at downsampled compute resolution (320px default), featuring Global Motion Compensation (GMC) to strip out camera shake, temporal smoothing, and rain/low-light gradient reliability gating.
- **Ground Plane Calibration (`ground_plane.py`)**: Transforms pixel velocities into real physical speed (`m/s`) via camera homography or height/pitch perspective calibration.
- **Crowd Metrics Engine (`crowd_metrics.py`)**:
  - **Divergence ($\nabla \cdot \mathbf{v}$)**: Spatial compression signature ($\text{negative} = \text{crowd compression / crush risk}$).
  - **Curl ($\nabla \times \mathbf{v}$)**: Rotational flow / turbulence.
  - **Helbing Turbulence Index**: Velocity variance divided by squared mean speed ($Var(V) / \bar{V}^2$).
  - **Stop-and-Go Wave Detection**: Short-lag temporal autocorrelation of zone mean speeds.
  - **Counterflow Score**: Fraction of cells moving opposingly to the dominant corridor direction.
- **Zone Alert Engine (`zones.py`)**: Evaluates user-defined polygonal zones with hysteresis ($T_{\text{clear}} = T_{\text{fire}} \times (1 - \text{hysteresis})$) and minimum duration gating (`min_duration_sec`).
- **Exclusion Masks (`detector_masks.py`)**: Masks out moving vehicles and held umbrellas so non-pedestrian motions do not corrupt crowd statistics.

---

## 💳 Automatic Number Plate Recognition (ANPR)

The ANPR subsystem ([anpr.py](file:///c:/Users/sahil/Downloads/Projects/crowd-safety-testbed/models/anpr/anpr.py)) captures each tracked vehicle and exports a photo gallery to `outputs/anpr/<video>/`:

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

## ☂️ Umbrella Detection & Crowd Density (7 Models)

Umbrella models estimate crowd density and serve as an overhead rain proxy (which impacts walking speeds and occludes torso keypoints for fall detectors).

| Model | Weights / Backbone | Approach & Characteristics |
|---|---|---|
| `umbrella_yolo` | 5.6 MB (`n`) / 19.3 MB (`s`) | YOLO11, COCO fixed class 25. Balanced baseline. |
| `umbrella_ssd` | 13.8 MB | SSDLite320 + MobileNetV3. High-precision / lower recall, CPU viable. |
| `umbrella_world` | 25.9 MB | YOLO-World v2 open-vocabulary text prompts. Highest recall. |
| `umbrella_yolo26n` | 5.5 MB | YOLO26-Nano NMS-free end-to-end head. |
| `umbrella_rfdetr` | RF-DETR Nano | DINOv2 backbone. Superior recall on small/occluded umbrellas in dense crowds. |
| `umbrella_rtdetrv2` | 20M params | RT-DETRv2-S zero-shot transformer detector (Apache 2.0). |
| `umbrella_trained` | 42.7M params | Fine-tuned single-class RT-DETRv2 umbrella checkpoint (F1 0.711). |

### Open-Vocabulary Prompting (`umbrella_world`)
YOLO-World matches exact text prompts. Near-synonyms like `umbrella` vs `thatched roof shelter` yield distinct results:

| Prompts | Detections Found on Thatched Beach Shelters |
|---|---|
| `umbrella`, `parasol` | 0 |
| `umbrella`, `parasol`, `canopy` | 0 |
| **`thatched roof shelter`** (+ others) | **35** |

Prompt for exact structures using `UmbrellaWorldDetector(prompts=("umbrella", "parasol", "thatched roof shelter"))`. Matched prompt strings are recorded in `extra["matched_class"]`.

---

## 🚘 Traffic Models & Frame Sampling Rules

1. **Low Frame Stride Required (Stride 1 or 2)**: Centroid drift classification (`vehicle_moving` vs `vehicle_parked`) relies on ByteTrack frame association. High sampling strides (e.g. stride 5) cause fast vehicles to jump across frames, fragmenting track IDs. The pipeline prints a warning when the sampling rate drops below ~10 fps.
2. **`mog2_parked` Limitation**: MOG2 background subtraction contrasts fast- and slow-adapting models to detect the *transition* of a vehicle coming to a stop. Cars already parked at frame 0 are absorbed into the background model and require object detector models (`yolo_traffic` / `rtdetrv2_traffic`).

---

## 🏗️ Design Notes & Technical Framework

- **Consumption Types**:
  - `"frame"`: Single image input (YOLO, ANPR, Umbrella).
  - `"clip"`: Sliding window list of $N$ frames (`FrameBuffer`). Sized to model `clip_len` and re-run every `clip_stride` frames to eliminate 97% redundant computation.
  - `"flow_pair"`: Consecutive frame pair $(t-1, t)$ for optical flow metrics.
- **Interpreting Confidence**: `Detection.confidence` represents **the probability of the reported event label** (e.g., fall or violence), not the generic person-detector score. Upstream detector confidence is preserved in `extra.detector_confidence`.
- **Shared Fall Components**:
  - `models/fall/_geometry.py`: Torso angle calculation, aspect ratio, posture scoring.
  - `models/fall/_tracker.py`: Greedy 1-to-1 IoU tracking + $K$-consecutive frame confirmation (`sustained()`).
  - `models/fall/_skeleton.py`: Keypoint normalization and Gaussian heatmap rendering.

---

## 💻 Interactive Web Application

Start the web server:

```bash
python -m webapp
```

Then navigate to **`http://127.0.0.1:8000`**.

- **Sidebar Model Registry**: Displays model status badges (`ready`, `fallback`, `blocked`) based on local checkpoint availability without paying PyTorch startup import costs.
- **Video Ingestion**: YouTube URL downloading & caching (`ingestion/youtube_fetch.py`), local `test_videos/` selection, or drag-and-drop file upload.
- **GPU Thread Pool Manager**: `JobManager` handles asynchronous background execution with single-GPU thread locks to prevent out-of-memory crashes.
- **Interactive Analytical Modal**: Inspect key KPIs, detection timelines, browser-compatible H.264 annotated video streams (`_FFmpegH264Writer`), and raw JSON payloads.
- **ANPR Gallery Viewer**: Browse captured vehicle portraits, plate crops, color classifications, and voting confidence.
- **Durable Disk History**: `webapp/history.py` scans `outputs/logs/` to preserve past runs across server restarts and CLI runs.

---

## ⚡ CLI Quickstart

```bash
# 1. Environment Setup
bash scripts/setup.sh

# 2. Run Single Model Test
python scripts/run_single.py --video test_videos/clip.mp4 --model fall_yolo_pose

# 3. Batch Evaluation Suite
python scripts/run_all.py --config configs/test_videos.yaml

# 4. Calibrate Dense Flow Thresholds
python scripts/calibrate_optical_flow.py --video test_videos/crowd_sample.mp4

# 5. Head-to-Head Model Benchmarking
python scripts/compare_models.py outputs/logs/sample_fall.json --category fall
```

---

## 📄 License & References
- **Core Testbed**: MIT License
- **RT-DETRv2 Models**: Apache 2.0 License (`PekingU/rtdetr_v2_r18vd`)
- **RF-DETR Models**: Roboflow RF-DETR DINOv2 Engine
