# Route View — assessment and redesign plan

Written as a handoff. If someone else (or a fresh session) picks this up, this
file should be enough to continue without re-deriving the analysis.

Status: **analysis complete, implementation not started.**

---

## 1. What Route View is meant to do

Map N videos of the SAME crowd filmed at DIFFERENT points along a route, so the
system can reason across cameras: crowd leaves CCTV1, arrives at CCTV2 some
seconds later, merges with another stream, continues to CCTV3.

That enables three things no single camera can see:

| Capability | Why it needs multiple cameras |
|---|---|
| **Conservation / accumulation** | `inflow − outflow` reveals people piling up in a segment no camera covers. Both ends can look healthy while 200/min accumulate between them. This is the crush precursor. |
| **Early warning** | A surge at CCTV1 with a 4-minute travel time gives the control room 4 minutes of warning. Reactive → predictive. |
| **Merge forecasting** | 300/min + 400/min into a junction rated 500/min is an arithmetically certain build-up, predictable before it starts. |

---

## 2. Root problem — the mental model is inverted

**User's model:** "I have N videos along a route. Let me tell the system the order."

**System's model:** "Here are the cameras declared in `configs/topology.yaml`.
Assign a video to each."

Everything below follows from that inversion.

### 2.1 "Only 3 videos can be mapped"

Not a bug — a consequence. `loadSessionSubmitPanel()` in `app.js` builds one
slot per camera in `GET /api/topology`, and `configs/topology.yaml` declares
exactly three (CCTV1, CCTV2, CCTV3). Mapping a 4th video requires hand-editing
YAML: a camera block, an `x/y` position, and edge(s) with travel times.

### 2.2 "Still not correct"

The shipped topology is a **MERGE**:

```
CCTV1 ──25s──┐
             ├──► CCTV3
CCTV2 ──20s──┘
```

If the actual footage is a **CHAIN** (`cam1 → cam2 → cam3`), the graph is wrong
for the videos and assigning videos cannot fix it — the relationship lives in a
file the user never edited. The UI lets you choose *which video goes where* but
never *what the route is*, which is the part that matters.

### 2.3 The editing path exists but is unreachable

`POST /api/topology` is implemented and auth-gated. **Nothing in the frontend
calls it.** `app.js` only ever GETs. So the graph is editable exclusively by
hand-editing YAML and restarting.

---

## 3. Proposed redesign

**Derive the topology FROM the videos, instead of assigning videos TO a
topology.**

### 3.1 Target user flow

1. Pick videos (any number, from `test_videos/` or upload)
2. Arrange them in route order — chain by default, with a "merges into" control
   for branches
3. System generates cameras, edges, and layout positions automatically
4. Travel times default to a placeholder and are editable — this is the one
   value that genuinely must be surveyed, so it must be visibly a user input,
   not a silent default
5. Save → `POST /api/topology` → YAML becomes a saved artifact rather than the
   only input

### 3.2 Why this is the right shape

- **Any N works** with no YAML editing
- **The graph always matches the footage**, because it was derived from it
- **YAML stays authoritative for deployment.** Surveyed travel times and
  corridor capacities must not be invented in a UI; the UI seeds the file, an
  operator refines it. Keep the "manually configured, not inferred" disclaimer.

### 3.3 Layout

No force-directed graph library. Positions are deterministic:

- Chain: evenly spaced left→right
- Merge: upstream nodes stacked vertically on the left, target to the right
- Deterministic layout means the graph does not jump between renders, which is
  what makes a live view readable

---

## 4. Correctness gaps to fix in the same pass

### 4.1 Video↔camera matching is a substring test

`app.js loadSessionSubmitPanel()` and `jobs.py _resolve_camera_id()`:

```js
v.name.toLowerCase().includes(cid.toLowerCase())
```

`CCTV1` matches a file named `CCTV11_*.mp4`. Should be an explicit
user-confirmed mapping (which the redesign gives for free), not inference.

### 4.2 Nothing validates time alignment

Cross-camera fusion assumes the clips are **simultaneous**. If clip A starts ten
minutes before clip B, every travel-time correlation is meaningless. Nothing
asks for or checks a start time.

Minimum fix: a per-slot "recording start time" input, defaulting to equal, with
a visible warning that fusion assumes simultaneity. `clock_offset_sec` already
exists on `CameraNode` and is plumbed through `MetricStore.update()` — it is
just never populated from the UI.

### 4.3 The chain case never exercises conservation

`_measured_outflow()` returns None when a camera has no downstream neighbour, so
`ACCUMULATION_RISING` cannot fire on the terminal node. In the shipped merge
topology CCTV3 is terminal, so **the conservation rule never runs on the default
config**. A chain (`A→B→C`) exercises it at B. Worth shipping a chain example.

### 4.4 Fusion rules are currently OFF by design

Since the calibration review, `FusionEngine` refuses to compare uncalibrated
flow against `corridor_capacity_pax_min` (it is not in pax/min without a
calibrated counting line). **So Route View will show no bottleneck alerts until
a camera is calibrated.** This is correct behaviour, but it must be visible in
the UI or it reads as "no problems detected".

Needs a banner: *"Fusion rules disabled — cameras uncalibrated. Showing relative
flow only."* The data is already there: `forecast_status[cam]["blocked"] ==
"uncalibrated_flow"`, and `snapshot.flow_is_calibrated`.

---

## 5. Implementation plan

Ordered so each step is independently testable.

### Step 1 — Backend: topology generation from an ordered list
`src/topology/graph.py`

```python
def from_route(segments: list[dict], defaults: dict) -> dict
```

`segments` is `[{camera_id, name, video, merges_into?, travel_time_sec?}, ...]`.
Returns a topology dict suitable for `update_from_dict()` / `POST /api/topology`.

- chain: consecutive entries linked in order
- merge: `merges_into` overrides the chain successor
- positions: computed by the layout rule in §3.3
- validation: reject cycles, unknown `merges_into`, duplicate camera ids

Tests: `tests/test_topology.py` — chain of 2/3/5, a merge, a cycle (must raise),
duplicate ids, position determinism.

### Step 2 — API
`src/webapp/app.py`

- `POST /api/topology/from-route` → validate, build, apply, persist to
  `configs/topology.yaml`
- Persist, because an in-memory-only topology is lost on restart while sessions
  that referenced it survive — they would then point at cameras that no longer
  exist
- Keep it behind the existing auth middleware

Tests: `tests/test_api_fusion.py` — round-trip, rejection of an invalid route,
auth required.

### Step 3 — Frontend: Route Builder
`src/webapp/frontend/index.html`, `app.js`, `styles.css`

Replace the fixed slot list with:

- "Add camera" button — appends a row (no YAML limit)
- Per row: video select/upload, display name, "merges into" dropdown (default
  "next in chain"), travel-time-from-previous input, recording start time
- Reorder controls (up/down buttons are sufficient; drag-and-drop is a nice-to-
  have, not required)
- Live preview of the generated graph beside the list
- "Save route" → `POST /api/topology/from-route`

Keep it vanilla JS and consistent with the existing single-page pattern. Note
`app.js` is already ~80 KB; consider a separate `route_builder.js` included
alongside rather than growing it further.

### Step 4 — Honesty surfaces in the UI
- Uncalibrated banner (§4.4)
- Per-edge "forecast incomplete — missing: X" (data already in
  `forecast_status`); the frontend fix for null-inflow rendering is already in
- Legend entry for the `edge-unknown` state

### Step 5 — Ship a chain example
Add `configs/topology.chain.example.yaml` (`A→B→C`) so conservation is
exercisable out of the box, and document switching between them.

---

## 6. What NOT to do

- **Do not infer travel times from the footage.** It is derivable in principle
  (cross-correlate flow signals) but would be a measurement presented as a
  survey. Keep it a user input.
- **Do not add person re-identification across cameras.** At Kumbh density it
  will not work — similar clothing, heavy occlusion, low resolution. Aggregate
  flux is sufficient and far more robust. (`deep-person-reid` is vendored in the
  colleague's repo; resist it.)
- **Do not let the UI invent corridor capacities.** They are survey figures.
  Default them visibly and require confirmation.

---

## 7. Current state of the surrounding code

Context a fresh session will need:

- `src/topology/graph.py` — `CameraTopology`, thread-safe, loads YAML
- `src/topology/metric_store.py` — per-camera snapshots + ring buffer.
  Snapshots carry `flow_is_calibrated` / `density_is_calibrated` / `units`;
  `reference_epoch_ms()` gives the source-clock "now"
- `src/topology/fusion_engine.py` — rules: `BOTTLENECK_PREDICTED`,
  `CRUSH_RISK_RISING`, `ACCUMULATION_RISING` (conservation). Refuses capacity
  comparisons on uncalibrated data. `get_forecast_status(cam)` reports
  completeness
- `src/webapp/session_jobs.py` — multi-camera session runner
- Flow/density from the pipeline are **uncalibrated** — a scaled detection count
  and persons-per-megapixel. Fusion rules are consequently disabled until a
  camera is calibrated (`src/models/crowd_flow/perspective.py` can fit a
  calibration from ~30–60 annotated person boxes, no site visit)

Tests: 153 passing. `tests/test_topology.py`, `test_fusion_engine.py`,
`test_metric_store.py`, `test_api_fusion.py`.
