# Crowd-flow module map

This package remains flat by design. The planned `core/`, `engine/`, and
`wrappers/` directory split is deferred until the heading-direction fix and
directional-classification rework in `crowd_motion_monitor.py` are complete
and merged.

```text
crowd_flow/
├── core:       flow_field.py, ground_plane.py, zones.py
├── engine:     detector_masks.py, crowd_metrics.py, density.py,
│               head_points.py, people_overlay.py, visualise.py
├── wrappers:   dense_flow_analyser.py, crowd_motion_monitor.py
├── shared I/O: video_writer.py
└── validation: validation/cross_family.py
```

The labels above describe responsibilities, not directories yet.

| File | Purpose | Primary importer(s) | Registry key |
| --- | --- | --- | --- |
| `flow_field.py` | Dense optical-flow calculation and reliability. | `dense_flow_analyser.py`, validation | — |
| `ground_plane.py` | Camera calibration and pixel-to-world conversion. | `dense_flow_analyser.py` | — |
| `zones.py` | Zones, thresholds, and alert lifecycle. | `dense_flow_analyser.py` | — |
| `detector_masks.py` | Detector-driven masks for the flow pipeline. | `dense_flow_analyser.py` | — |
| `crowd_metrics.py` | Flow-derived crowd-safety metrics. | `dense_flow_analyser.py`, validation | — |
| `density.py` | Crowd-density helpers. | `people_overlay.py` | — |
| `head_points.py` | Head-point estimation helpers. | `people_overlay.py` | — |
| `people_overlay.py` | Person/density visual overlay. | `dense_flow_analyser.py` | — |
| `visualise.py` | Flow, heatmap, and zone visualisation. | `dense_flow_analyser.py` | — |
| `video_writer.py` | Shared streaming H.264 writer with MJPG fallback. | Both wrappers; validation | — |
| `dense_flow_analyser.py` | Configurable dense-flow crowd-safety wrapper. | package export, runner, web registry | `dense_flow` |
| `crowd_motion_monitor.py` | Per-person motion and crowd-risk wrapper. | package export, web registry | `crowd_motion_monitor` |
| `validation/cross_family.py` | Cross-family flow/tracker validation. | validation scripts/callers | — |

`crowd_motion_monitor.py` is intentionally standalone from dense-flow model
logic, metrics, zones, and overlays. Its only shared `crowd_flow` dependency
is `video_writer.py`'s H.264 writer utility; detector and tracker dependencies
come from the project-wide shared modules.
