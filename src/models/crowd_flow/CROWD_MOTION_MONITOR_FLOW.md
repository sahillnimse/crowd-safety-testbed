# Crowd Motion Monitor — Processing Flow

> **File:** `models/crowd_flow/crowd_motion_monitor.py`  
> **Class:** `CrowdMotionMonitor`  
> **Type:** Per-person velocity, heading, and crush-risk detector  
> **Pipeline mode:** `consumption_type = "flow_pair"` — runner calls `predict((prev_frame, curr_frame), frame_index, timestamp_sec)` once per frame pair.

---

## Models & Algorithms Used

| Component | Model / Algorithm | Library | Notes |
|---|---|---|---|
| **Person Detection** | RT-DETRv2 (shared project detector) | HuggingFace Transformers | Runs every `detect_every=5` frames; boxes carried forward |
| **Optical Flow** | Farneback (`cv2.calcOpticalFlowFarneback`) | OpenCV (CPU) | Full-frame, every frame |
| **Person Tracking** | IoU Tracker (`models/_tracker.py`) | Custom | Links boxes across frames by bounding-box overlap |
| **Direction Clustering** | Mini k-means on unit heading vectors | NumPy | Infers 1 or 2 direction streams |

---

## High-Level Architecture

```
prev_frame + curr_frame
        │
        ├──► [1] Farneback Optical Flow  ─────────────────────► flow (H,W,2)
        │                                                            │
        │                                         ┌─────────────────┴──────────────────┐
        │                                         ▼                                    ▼
        │                               [2a] div_grid (32px cells)        [2b] var_grid (32px cells)
        │                               ∂fx/∂x + ∂fy/∂y  → p10            Circular variance [0,1]
        │
        ├──► [3] RT-DETRv2 Detection   ──────────────────────► person boxes []
        │         (every 5th frame, 2×2 tiling)
        │
        └──► [4] IoU Tracker           ──────────────────────► track_ids []
                  (iou_thr=0.3, max_age=30)
                        │
                        ▼
              [5] Per-Person Kinematics (Pass 1)
                  sample flow in torso patch → vx, vy, speed
                  speed history → stationary flag (hysteresis)
                  EMA heading smoothing (α=0.35)
                        │
                        ▼
              [6] Direction Stream Inference
                  k-means on unit heading vectors
                  → stream_a / stream_b / moving
                        │
                        ▼
              [7] Local Safety Metrics (Pass 2)
                  local_divergence  (div_grid lookup)
                  crush_risk flag
                  directional_entropy (Shannon, 8-bin, 96px radius)
                  counterflow flag   (angle vs neighbourhood mean)
                        │
                        ▼
              [8] Label Assignment → Detection[]
                  person_stopped / person_crush_zone /
                  person_moving_stream_a / person_moving_stream_b / person_moving
                        │
                        ▼
              [9] Visual Overlay → annotated MP4
```

---

## Step-by-Step Detail

### Step 1 — Farneback Optical Flow

```python
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)  # → shape (H, W, 2), float32, units: px/frame
```

- Runs on **grayscale**, **full source resolution**, **every frame**
- Produces a dense field: every pixel gets `(dx, dy)`
- Uses an image pyramid (`levels=3`, `pyr_scale=0.5`) for robustness to fast motion
- Polynomial expansion (`poly_n=5`) approximates local flow as a 2D polynomial

---

### Step 2 — Spatial Grid Computation (32 px cells)

The frame is divided into **32×32 px cells**. Two grids are computed:

#### 2a. Divergence Grid

```
div = ∂fx/∂x + ∂fy/∂y      (computed with np.gradient)
div_grid[r,c] = p10(div in cell)   ← 10th percentile, robust to outliers
```

| Value | Meaning |
|---|---|
| **Negative** | Vectors converging → crowd compressing → **crush risk** |
| **Zero** | Neutral, laminar flow |
| **Positive** | Vectors diverging → crowd spreading out |

#### 2b. Variance Grid (Circular / Directional)

```
R = mean resultant length of unit flow vectors in cell
var_grid[r,c] = 1 − R       ∈ [0, 1]
```

| Value | Meaning |
|---|---|
| `0.0` | All pixels moving in exactly the same direction |
| `1.0` | Maximum directional disorder / chaos |

> Only pixels with magnitude > 0.3 px/frame contribute (noise floor).

---

### Step 3 — Person Detection (RT-DETRv2)

```python
# Runs every detect_every=5 frames; boxes carried on the 4 frames in between
boxes = detector.detect(
    curr_frame,
    classes=(COCO_PERSON,),
    tile_grid=(2, 2),           # 4 tiles + full frame = 5 overlapping crops
    conf_threshold=0.28,        # lower than shared default (distant people score lower)
)
```

**Why 2×2 tiling?**  
A single full-frame pass on 1280×720 input resizes to 640×640, halving a 20 px distant person to ~10 px — below the model's detection floor. Tiling keeps each crop at higher effective resolution. Measured improvement: **35 → 95 people** detected on dense crowd footage.

---

### Step 4 — IoU Tracking

```python
track_ids = self._tracker.update(boxes, frame_index)
# Parameters: iou_threshold=0.3, max_age=30 frames, history_len=64
```

- Each person box gets a persistent integer `track_id`
- Boxes matched by IoU ≥ 0.3 inherit the same ID across frames
- Tracks not matched for 30 frames are dropped
- Speed / heading state is **evicted** when the tracker drops a track

---

### Step 5 — Per-Person Kinematics (Pass 1)

For every tracked box, flow is sampled in a **shrunk torso region** to avoid body-edge background contamination:

```
horizontal inset: 20% on each side
vertical range:   40% – 70% of box height  (torso, not head or feet)

vx = median(flow[torso_patch, 0])
vy = median(flow[torso_patch, 1])
speed = hypot(vx, vy)              # px / frame
```

Median is used (not mean) — robust to limb / occluder outliers.

#### Stationary Detection (with hysteresis)

```
speed_history[tid] ← rolling deque of last stationary_frames=10 speeds

ENTER stationary:  all 10 history values < stationary_speed_px (1.5 px/frame)
EXIT  stationary:  speed >= threshold for resume_moving_frames=3 consecutive frames
```

#### Heading Smoothing (EMA)

```
unit_vec = (vx/speed, vy/speed)    # only when speed ≥ noise_floor
heading_vec[tid] = α × unit_vec + (1−α) × previous_heading_vec
heading_vec[tid] = heading_vec[tid] / ‖heading_vec[tid]‖   # re-normalise

α = 0.35  (favours past history, prevents jitter)
heading_deg = atan2(hy, hx)   in degrees, image-space (+y down)
```

---

### Step 6 — Direction Stream Inference

From all **confirmed + moving** tracks, the model detects whether crowd flow is **unidirectional** or **bidirectional** (e.g. separate entry/exit streams).

**Algorithm: Mini k-means on unit heading vectors**

```
1. Collect heading unit vectors from all confirmed moving tracks
2. Seeds: find the most angularly separated pair (max of distance matrix)
3. Run up to 8 k-means iterations on the unit circle
4. Accept two streams only if:
     - min cluster share ≥ 15%       (both streams are meaningful in size)
     - angular separation ≥ 60°      (streams are genuinely different directions)
5. Order centres by angle → deterministic labels per frame
6. Label each person: stream_a | stream_b | moving (if one-way)
```

---

### Step 7 — Local Safety Metrics per Person (Pass 2)

Within a **96 px radius** neighbourhood around each person:

#### 7a. Crush Risk

```python
local_div = div_grid[cy // 32, cx // 32]

local_crush_risk = (
    local_div < crush_divergence_threshold   # default: -1.0
    and speed <= crush_max_speed_px          # still moving but crowd converging
    and not personally_stationary            # not already flagged stopped
)
```

#### 7b. Directional Entropy (Shannon, 8-bin)

```python
bins = [0] * 8
for nb in neighbours:          # confirmed moving tracks within 96 px
    b_idx = int((nb.heading_deg + 180.0) / 45.0) % 8
    bins[b_idx] += 1

entropy = −Σ (p × log₂(p))    # where p = count / total_neighbours
```

| Value | Meaning |
|---|---|
| `0.0` | All neighbours moving in the same direction (laminar) |
| `3.0` | Maximum disorder — all 8 directions equally populated |

#### 7c. Counterflow Flag

```python
# mean heading of neighbourhood (excluding self)
dom = mean_unit_vector(neighbours)

angle_diff = acos(dot(person_heading, dom))   # in degrees

is_counterflow = angle_diff >= counterflow_angle_threshold_deg   # default: 120°
```

A person is counterflow if they are moving **more than 120°** away from the neighbourhood's dominant direction.

---

### Step 8 — Label Assignment & Detection Output

Each track with age ≥ `confirm_frames=3` emits one `Detection` per frame:

```
Priority order (highest wins):
  1. person_stopped       ← personally_stationary == True
  2. person_crush_zone    ← local_crush_risk == True
  3. person_moving_stream_a
     person_moving_stream_b
  4. person_moving        ← single-stream / undifferentiated
```

```
Detection.label      = one of the 5 labels above
Detection.confidence = 0.0                               if stopped
                     = min(1.0, speed / (5 × 1.5))      if moving
Detection.bbox       = [x1, y1, x2, y2]  (source pixels)
Detection.extra      = {
    track_id, speed_px_frame, heading_deg,
    crowd_direction,          # stream_a | stream_b | moving
    personally_stationary,
    local_divergence,
    local_crush_risk,
    local_velocity_variance,
    local_directional_entropy,
    is_counterflow,
    counterflow_angle_deg,
}
```

---

### Step 9 — Visual Overlay

Three overlay modes are supported (configurable via `overlay_mode`):

#### Marker Layer (`markers` or `combined`)

Each tracked person is rendered as a **filled equilateral triangle** rotated to point in `heading_deg`:

| Colour | State |
|---|---|
| 🟢 `TEAL-GREEN` | Moving → stream A |
| 🔵 `ELECTRIC BLUE` | Moving → stream B |
| 🔴 `RED` | Personally stationary |
| 🟠 `ORANGE` | Crush zone (converging crowd) |
| ⬛ `DARK GREY` | Track pending (not yet confirmed) |

> **Stationary** person → solid circle with white inner ring (no rotation, avoids misleading direction from head/torso sway)

#### Heatmap Layer (`heatmap` or `combined`)

| `heatmap_metric` | Colour map | What it shows |
|---|---|---|
| `divergence` | Red ↔ White ↔ Blue | Red = compression, Blue = expansion |
| `variance` | JET | Flow directional disorder |
| `entropy` | MAGMA | Directional chaos in neighbourhood |
| `counterflow` | HOT | Cells with counterflow activity |

---

### `finalize()` — Run-Level Summary

After all frames are processed, `finalize()` computes and stores `self.summary`:

| Key | Description |
|---|---|
| `pct_moving` | % of detections that were moving |
| `pct_stationary` | % of detections that were stopped |
| `pct_crush_risk` | % of detections in crush zones |
| `crush_event_count` | Distinct periods with ≥ 3 simultaneous crush-flagged people |
| `counterflow_events_count` | Distinct periods with ≥ 2 counterflow people |
| `avg/peak_velocity_variance` | Flow disorder over entire run |
| `avg_directional_entropy` | Average local directional chaos |
| `heading_histogram` | 18-bin (20° each) heading distribution for moving tracks |
| `suspicious_tracks` | Tracks with > 5 direction flips (potential tracking errors) |
| `speed_by_label` | avg/max speed per label |

Output video is written to:
```
{output_dir}/{video_name}_crowd_motion_monitor.mp4
```

---

## Key Parameters Reference

| Parameter | Default | Effect |
|---|---|---|
| `stationary_speed_px` | `1.5` px/frame | Speed floor for stopped classification |
| `stationary_frames` | `10` | Consecutive sub-threshold frames to enter stopped state |
| `resume_moving_frames` | `3` | Consecutive above-threshold frames to exit stopped state |
| `crush_divergence_threshold` | `-1.0` | div_grid value that triggers crush risk |
| `crush_max_speed_px` | `6.0` | Speed cap for crush flag (very fast = not trapped) |
| `counterflow_angle_threshold_deg` | `120°` | Angle from neighbourhood mean to flag counterflow |
| `confirm_frames` | `3` | Minimum track age before emitting a Detection |
| `detect_every` | `5` | Run RT-DETRv2 every N frames |
| `detect_tile_grid` | `(2, 2)` | Tiling for small/distant person detection |
| `detect_conf_threshold` | `0.28` | Detection confidence (lower than shared default) |
| `overlay_mode` | `markers` | `markers` / `heatmap` / `combined` |
