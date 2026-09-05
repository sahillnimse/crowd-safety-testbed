# Handoff — current state, pending work, known bugs

Written for whoever continues this. Everything below is verified against the
code as it stands, not remembered.

**State:** 193 tests passing. 100/100 modules import. 0 unused imports. 30/30 GET
routes return without a 5xx. Test suite no longer mutates project config.

---

## 1. Completed and verified this session

All in `src/webapp/session_report.py` unless noted. Each was a real defect that
produced a confidently wrong number in the session HTML report — the artifact
most likely to be shown to someone making a decision.

| # | Defect | Fix |
|---|---|---|
| 1 | **Report bypassed the calibration gate.** It computed `capacity_utilization_pct = flow / capacity × 100` and printed "Inflow represents X% of downstream capacity". `FusionEngine` correctly *refuses* that comparison on uncalibrated data — flow is not pax/min without a calibrated counting line. The two components disagreed about whether the question was answerable. | Gated on `summary["is_calibrated"]`. Returns `None` + `status="unmeasured"` when uncalibrated. |
| 2 | **`or 0.0` erased "not measured".** `float(summary.get("avg_density") or 0.0)` — a camera that never measured density contributed `0.0` to a *weighted mean*, dragging the route average down in proportion to how many cameras failed. | `_opt()` helper preserves `None`; weighted means divide by the weight that actually contributed. |
| 3 | **Route flow summed, triple-counting the crowd.** `sum(flows)` labelled "total throughput" — the same people passing cam1→cam2→cam3 were counted three times. | Renamed `bottleneck_specific_flow`, uses `min()`. Route throughput is limited by its narrowest section. |
| 4 | **`abs()` on a NET flow.** Two balanced 300/min streams net to ≈0, so the report claimed an empty corridor carrying 600 people/min. | Prefers `specific_flow_gross_per_sec`. |
| 5 | **Density reported as mean only.** One camera at 6 pax/m² with two at 1 averages to a comfortable 2.7 while one spot is critical. | Added `peak_density`. Pressure already used max for this reason. |
| 6 | **Propagation check collected and discarded.** `peak_source_time` / `peak_target_time` were put in the narrative but never compared. | Added `observed_lag_sec` and `lag_consistent_with_topology` — the only evidence in the report that the *topology itself* is right. |
| 7 | **HTML renderer crashed on `None`.** `.get(k, 0)` returns `None` when the key exists with a None value; `{v:.0f}` then raises. `cdata['crush_risk_pct'] > 15` likewise. | `_fmt()` renders "n/a"; class selection returns `neutral` (never `good`) for unmeasured. |
| 8 | **No calibration banner on the report.** | Orange banner stating density is not persons/m² and flow is not pax/min, naming the uncalibrated cameras. |

Also updated `tests/test_session_report.py::test_aggregate_session_metrics` for
the renamed key (it asserted `total_specific_flow == 1030.0`, a sum).

---

## 2. PENDING — immediate

### 2.1 Regression tests for §1 (COMPLETED)

All regression tests implemented in `tests/test_session_report.py`:
- `test_uncalibrated_route_suppresses_capacity_utilization` (uncalibrated route → `capacity_utilization_pct is None`, `status == "unmeasured"`)
- `test_calibrated_route_computes_capacity_utilization` (calibrated route → percentage IS computed)
- `test_unmeasured_camera_does_not_dilute_density_weighted_mean` (unmeasured camera density abstains from weighted mean, not diluted toward zero)
- `test_bottleneck_specific_flow_takes_minimum_not_sum` (`bottleneck_specific_flow == min(...)`, and `total_specific_flow` absent)
- `test_gross_flow_preferred_over_net` (gross preferred over net when both present)
- `test_peak_density_captured_and_greater_equal_avg_density` (`peak_density >= avg_density`)
- `test_propagation_lag_consistency` (lag consistent / inconsistent / missing cases)
- `test_html_report_handles_none_values_safely` (HTML generates without raising when optional values are None, contains no literal "None", renders "n/a", neutral CSS)
- `test_html_report_calibration_banner` (orange UNCALIBRATED ROUTE banner present on uncalibrated, omitted on calibrated)

### 2.2 Route View redesign — IMPLEMENTED (verify against the open questions below)

Full plan in **`docs/ROUTE_VIEW_REDESIGN.md`**. Summary of the problem:

- The route lives in `configs/topology.yaml`; the UI generates one slot per
  camera declared there. "Only 3 videos" is a consequence, not a bug.
- `POST /api/topology` exists and **nothing in the frontend calls it**.
- Shipped topology is a **merge**; if the footage is a **chain** the graph is
  simply wrong for the videos and assigning videos cannot fix it.

Fix is to invert the model: pick videos → order them → generate the topology.

**This has since been built:** `POST /api/topology/from-route` +
`build_topology_from_route()` generate the graph, `POST /api/topology/reset`
reverts to the baseline, and the frontend calls both (`app.js` ~3431/3460) with
an add-camera Route Builder. Output goes to `configs/topology.generated.yaml`,
leaving the hand-authored `topology.yaml` intact — which was the right call (see
3.2b for the one thing that went wrong with it).

The five design questions below were raised BEFORE it was built and have not
been re-checked against the implementation. Worth verifying each.

**The five design questions I raised have all been resolved in the
implementation** — verified by direct test, not by reading:

| Question | Outcome |
|---|---|
| Splits (one camera → two downstream) | **Supported** — takes an explicit edge list rather than a linear order |
| Layout beyond 3 nodes | 6 nodes get distinct, non-overlapping positions |
| Writing over the hand-authored baseline | Writes to `topology.generated.yaml`; `topology.yaml` untouched |
| Travel-time default | **No default.** Rejects with *"travel time must be strictly positive and surveyed"* — the right call |
| Cycle rejection | Rejected, including 3-cycles. Also rejects duplicate IDs and edges to unknown cameras |

**Time alignment of mapped clips — FIXED.** Two separate defects:

1. **Session cameras had no shared time base.** They are processed
   *sequentially*, and each camera's telemetry was stamped with
   `stream_start_epoch_ms = cam_state.started_at` — the moment *that camera's
   processing* began. On a 9-camera session that separated the camera timelines
   by tens of minutes, while cross-camera fusion correlates readings tens of
   *seconds* apart. Every upstream lookup fell outside the retained history, so
   the fusion engine could **never** correlate two cameras from a route session
   — the feature was structurally inert, not merely unvalidated.
   `RouteSessionState.session_epoch_ms` is now captured once per session and
   used for every camera. Covered by
   `test_session_has_one_shared_time_base_for_all_cameras`.

2. **`clock_offset_sec` was never sent by the UI.** It existed on `CameraNode`,
   was accepted by `build_topology_from_route()`, and was plumbed through
   `MetricStore.update()` — but the Route Builder collected only id, name and
   capacities, so per-clip start skew could not be expressed at all. The camera
   row now has a **Clip offset (s)** field (negative allowed: the clip started
   before the reference). Covered by
   `test_route_builder_clock_offset_survives_round_trip`.

Still not validated: nothing *checks* the operator's claim that the clips are
simultaneous. The offset is now expressible and surveyed; it is not derived.

---

## 3. Known bugs / gaps NOT fixed

### 3.1 Travel time is a constant (design limitation)
`travel_time_sec` is fixed in YAML, but travel time is distance ÷ crowd speed.
A 30 m corridor is ~23 s at 1.3 m/s free-flow and ~100 s at 0.3 m/s in a dense
crowd — so the offset is ~4× wrong **exactly when it matters most**, and the
engine correlates the wrong time windows. Needs deriving from measured speed,
which needs calibration first.

### 3.2 Fusion rules OFF until calibrated — FIXED (visible now)
`FusionEngine` still refuses capacity comparisons on uncalibrated flow (correct),
but the UI no longer presents that silence as safety.

`renderFusionAlerts()` now distinguishes four reasons for zero alerts, where it
previously claimed "Corridor flow is currently within safe limits" for all of
them:

| State | Badge |
|---|---|
| no telemetry at all | `0 active (offline)` |
| flow uncalibrated, rules disabled | `0 active (uncalibrated)` |
| upstream incomplete, alerts suppressed | `0 active (incomplete)` |
| genuinely evaluated and clear | `0 active` |

The sparkline strip also marks unit provenance (`✓` calibrated / `~` relative),
because the rho and Q labels otherwise imply persons/m2 and pax/min on values
that are neither.

**MetricStore persistence — FIXED.** Telemetry is now mirrored to
`outputs/state/metric_store.json` (atomic temp-and-rename, flushed at most every
5 s) and restored on startup, so a restart no longer erases the history the
fusion engine needs for its time-shifted upstream lookups.

Restored samples keep their ORIGINAL `received_at`, so a camera that stopped
feeding before the restart is still correctly reported **stale** — history comes
back for the lookups without resurrecting a dead camera as live. Persistence is
opt-in (`persist_path=None` by default) so tests and library users never touch
shared on-disk state.

### 3.2b Generated topology was shadowing the baseline — FIXED
`configs/topology.generated.yaml` (Route Builder output) **takes precedence over**
the hand-authored `configs/topology.yaml` in `CameraTopology.__init__` — correct
behaviour, but the file was **committed to git**, so every clone silently ran one
developer's ad-hoc 6-camera route instead of the surveyed baseline. Untracked and
gitignored.

Two tests were also mutating real project config: `test_api_fusion.py` wrote to
and deleted the real generated file (leaving the working tree dirty and
destroying any route built through the UI), and
`test_calibration_fusion_integration.py` used a bare `CameraTopology()` that read
whatever happened to be on disk. Both now isolated.

### 3.3 No camera is calibrated
All four Nashik cameras report `NO_HOMOGRAPHY`. `ram_kund_approach` also still
carries two **placeholder zone polygons** authored for a 640×480 frame.
`CCTV1`/`CCTV2` in `topology.yaml` *are* calibrated, so the gate is passable.

### 3.4 No ground-truth validation of the crush path
SAIVT gives occupancy + gate crossings; there is **no labelled crush event**
anywhere. The false-negative rate of the crush-alert path is unmeasured and
will stay so until incident footage is annotated.

### 3.5 Flow scorer not built
6 SAIVT cameras carry **1,836 gate crossings with direction** — unused ground
truth for specific flow and counter-flow. This is the correct validation for
CrowdMotionMonitor, which **cannot** be validated on the annotated stills (they
are 72 s apart, so tracking across them is meaningless). Needs the scorer plus
making CMM's counting line configurable to sit on SAIVT's gate polygon rather
than the frame centre where it is hardcoded
(`CrowdMotionMonitor.SPECIFIC_FLOW_LINE_X_FRAC`).

### 3.6 Measured model accuracy (context for anyone reading the numbers)
Scored against 1,088 human-annotated people, 10 cameras, 300 stills:

| Model | Recall | Precision | F1 |
|---|---|---|---|
| APGCC | 0.441 | 0.391 | 0.415 |
| RT-DETRv2 | 0.723 | 0.468 | 0.568 |

APGCC is **not** ~99% accurate and must not be used as ground truth. Caveat: the
test data is indoor at **3.6 people/frame** while APGCC trains on ~500/image, so
this is out-of-distribution for it and the ranking may invert at Kumbh density.
DM-Count still unscored (adapter bug fixed, never re-run):

```bash
python scripts/evaluate_on_saivt.py --models dmcount --all-cameras --limit 30 --out outputs/eval/dmcount.json
```

---

## 4. The recurring pattern

Most defects found across this project share one shape:

> **A measurement that could not be made is rendered as a safe-looking number.**

`0.0` for unmeasured density; `~0` for balanced net flow; a capacity percentage
computed from uncalibrated units; a mean that hides the worst camera; a dropped
camera reported as a completed run; `None` inflow rendered as `0` and coloured
green.

When reviewing anything here, the first question worth asking is: *if this
measurement failed, what does the operator see?* If the answer is a number
rather than an absence, that is the bug.

---

## 5. Verification commands

```bash
python -m pytest tests/ -q                 # 193 passing
python -m compileall -q src scripts tests main.py
node --check src/webapp/frontend/app.js
python scripts/evaluate_on_saivt.py --coverage
```

## 6. Regenerating per-camera reports after a template change

`scripts/update_session_reports.py` re-renders every `report.html` under
`outputs/sessions/` from the artifacts already on disk (`summary.json` +
`detections.json`). It never re-runs a model.

```bash
python scripts/update_session_reports.py --dry-run
python scripts/update_session_reports.py
```

The model key comes from each session's `session_manifest.json`, never a
constant: the first version hardcoded `crowd_motion_monitor` and relabelled the
nine reports of a **`dm_count_crowd`** session as CrowdMotionMonitor. Sessions
with no readable manifest are skipped rather than guessed at.
