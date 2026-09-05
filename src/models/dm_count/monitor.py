"""
DMCountCrowdMonitor — density-map head counting + tracked crowd risk metrics.

The newest member of the crowd-crush family, ported from the standalone
``Ujwal/__CMS__`` dashboard and rebuilt on this project's conventions:

  1. DM-Count (VGG19 density map, ShanghaiTech-A checkpoint) counts heads per
     frame; local maxima of the density map become head points
     (models/dm_count/infer.py).
  2. Head points are tracked with a BYTE-style two-stage nearest-point
     tracker (models/dm_count/tracking.py), giving per-person velocity.
  3. Farneback flow between the frame pair supplies the field divergence.
  4. The ten risk metrics (density, speed, velocity variance, Helbing
     pressure, divergence, directional entropy, dominant direction,
     counter-flow %, stop-and-go, oscillation) are computed per frame
     (models/dm_count/metrics.py).
  5. Rule-based alerts fire as Detection rows with ``<metric>_<severity>``
     labels (models/dm_count/alerts.py), exactly like the dense-flow
     engine's zone alerts.

Integration contract — identical to CrowdMotionMonitor:
- consumption_type = "flow_pair"  (runner calls predict((prev, curr), ...))
- Emits one Detection per confirmed tracked head point per frame:
    label      : "head_moving" | "head_stopped"
    confidence : normalised speed (moving) or 0.0 (stopped)
    bbox       : small synthetic box around the head point
    extra      : {track_id, x, y, dir_deg, status, density_value, track_age,
                  head_count, mean_speed, pressure, divergence, stop_and_go}
- Also emits one "dm_frame_metrics" context row per frame carrying every
  FrameMetrics value (the full time series lands in detections.json/csv);
  it is deliberately NOT in POSITIVE_LABELS because it is telemetry, not an
  alert.
- finalize() closes the streaming H.264 writer, sets self.annotated_video_path
  and fills self.summary (picked up by webapp/jobs.py / scripts/run_single.py).
- Reads self._fps / self._frame_stride / self.output_fps if set externally by
  jobs.py after construction (flow_pair protocol).

Calibration is OPTIONAL and uses this project's own convention: pass the
camera's block from configs/crowd_flow.yaml (``camera_block=`` +
``camera_id=``) to get m/s speeds and heads/m^2 density via the shared
homography machinery. Without it everything is px/frame + heads/frame, the
same uncalibrated default as CrowdMotionMonitor.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import cv2
import numpy as np

from models.base import BaseModelWrapper, Detection
from models.dm_count.alerts import AlertThresholds, CrowdAlertEngine
from models.dm_count.infer import DMCountCounter
from models.dm_count.metrics import CrowdMetricsEngine, FrameMetrics
from models.dm_count.tracking import PointTracker

# Re-use the project's streaming H.264 writer (encoder utility, not model
# logic — same rationale as CrowdMotionMonitor importing it).
from models.crowd_flow.video_writer import _AnnotatedVideoWriter

logger = logging.getLogger(__name__)

# ── Colours (BGR) ──────────────────────────────────────────────────────────
_COLOUR_MOVING   = (140, 200,   0)   # teal-green
_COLOUR_STOPPED  = (  0,  40, 220)   # red
_COLOUR_PENDING  = ( 80,  80,  80)   # dark grey — track not yet confirmed
_COLOUR_TEXT     = (255, 255, 255)

# Synthetic box half-size around a bare head point, for IoU-style consumers
# of detections.json and readable annotation. Same value CrowdMotionMonitor
# uses for APGCC points (_APGCC_SYNTH_HALF_BOX_PX).
_HEAD_HALF_BOX_PX = 12

# Arrow length is speed-proportional, clamped to these multiples of the
# half-box so slow drift still draws something visible.
_ARROW_MIN_PX = 8
_ARROW_MAX_PX = 48

# Farneback parameters — same as OpticalFlowCrushDetector/CrowdMotionMonitor
# for cross-model comparability.
_FB_PYR_SCALE, _FB_LEVELS, _FB_WINSIZE = 0.5, 3, 15
_FB_ITERATIONS, _FB_POLY_N, _FB_POLY_SIGMA, _FB_FLAGS = 3, 5, 1.2, 0


class DMCountCrowdMonitor(BaseModelWrapper):
    consumption_type = "flow_pair"
    name             = "dm_count_crowd"
    gpu_accelerated  = False   # DM-Count runs comfortably on CPU at the cap

    def __init__(
        self,
        device: Optional[str] = None,
        output_dir: str = "outputs/annotated",
        video_name: str = "run",
        weights: Optional[str] = None,
        # DM-Count counter knobs (see infer.py for provenance)
        max_long_side: int = 960,
        peak_min_distance_px: int = 6,
        peak_value_thresh: float = 0.06,
        # Tracker knobs
        max_dist_px: float = 60.0,
        tracker_high_thresh: float = 0.25,
        tracker_low_thresh: float = 0.06,
        tracker_max_age: int = 15,
        smooth_frames: int = 5,
        min_track_age: int = 3,
        # Motion thresholds — px/frame uncalibrated, m/s when calibrated
        stopped_speed: float = 1.5,
        # Optional homography calibration (configs/crowd_flow.yaml block)
        camera_block: Optional[dict] = None,
        camera_id: str = "default",
        safe_capacity: int | None = None,
        # Alert rule thresholds
        alert_thresholds: Optional[AlertThresholds] = None,
        # Rendering
        show_density_heatmap: bool = True,
    ) -> None:
        super().__init__(device=device)

        self._output_dir = output_dir
        self._video_name = video_name

        self.counter_kwargs = dict(
            weights=weights,
            device=self.device,
            peak_min_distance_px=peak_min_distance_px,
            peak_value_thresh=peak_value_thresh,
            max_long_side=max_long_side,
        )
        self.tracker_kwargs = dict(
            max_dist_px=max_dist_px,
            high_thresh=tracker_high_thresh,
            low_thresh=tracker_low_thresh,
            max_age=tracker_max_age,
            smooth_frames=smooth_frames,
        )
        self.min_track_age = min_track_age
        self.stopped_speed = stopped_speed
        self.camera_block = camera_block
        self.camera_id = camera_id
        self.safe_capacity = safe_capacity
        self.alert_thresholds = alert_thresholds or AlertThresholds()
        self.show_density_heatmap = show_density_heatmap

        # jobs.py sets these after construction for any flow_pair model.
        self._fps: float = 25.0
        self._frame_stride: int = 1
        self.output_fps: Optional[float] = None

        # Runtime state — initialised in load().
        self._counter: Optional[DMCountCounter] = None
        self._tracker: Optional[PointTracker] = None
        self._metrics: Optional[CrowdMetricsEngine] = None
        self._alerts: Optional[CrowdAlertEngine] = None
        self._calibration = None          # CameraCalibration | None
        self._roi_area_m2: Optional[float] = None

        # Streaming video writer.
        self._writer: Optional[_AnnotatedVideoWriter] = None
        self._frames_written: int = 0
        self.annotated_video_path: Optional[str] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None

        # Run-level summary accumulators.
        self.summary: dict = {}
        self._frames_processed: int = 0
        self._head_counts: list[int] = []
        self._mean_speeds: list[float] = []
        self._pressures: list[float] = []
        self._entropies: list[float] = []
        self._stop_go: list[float] = []
        self._oscillations: list[float] = []
        self._counterflow_pcts: list[float] = []
        self._alert_history: list[dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        from models.dm_count import get_dm_count_counter

        self._counter = get_dm_count_counter(**self.counter_kwargs)
        self._counter.load()

        self._calibration = None
        self._roi_area_m2 = None
        if self.camera_block:
            from models.crowd_flow.ground_plane import CameraCalibration
            self._calibration = CameraCalibration.from_yaml_block(
                self.camera_id, self.camera_block)

        self._tracker = PointTracker(**self.tracker_kwargs)
        self._metrics = CrowdMetricsEngine(
            safe_capacity=self.safe_capacity,
            calibrated=self._calibration is not None,
        )
        self._alerts = CrowdAlertEngine(
            thresholds=self.alert_thresholds,
            safe_capacity=self.safe_capacity,
        )

        self._reset_run_state()
        os.makedirs(self._output_dir, exist_ok=True)
        # Guard flag for predict() — same convention as CrowdMotionMonitor.
        self._model = "ready"
        logger.info(
            "DMCountCrowdMonitor loaded. device=%s calibrated=%s "
            "stopped_speed=%.3f max_long_side=%d peak_floor=%.3f",
            self.device, self._calibration is not None, self.stopped_speed,
            self.counter_kwargs["max_long_side"],
            self.counter_kwargs["peak_value_thresh"],
        )

    def _reset_run_state(self) -> None:
        self._writer = None
        self._frames_written = 0
        self.annotated_video_path = None
        self.latest_annotated_frame = None
        self.summary = {}
        self._frames_processed = 0
        self._head_counts.clear()
        self._mean_speeds.clear()
        self._pressures.clear()
        self._entropies.clear()
        self._stop_go.clear()
        self._oscillations.clear()
        self._counterflow_pcts.clear()
        self._alert_history.clear()

    # ──────────────────────────────────────────────────────────────────────
    # predict — called once per frame pair by the pipeline runner
    # ──────────────────────────────────────────────────────────────────────

    def predict(
        self,
        frame_pair: tuple[np.ndarray, np.ndarray],
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("DMCountCrowdMonitor.load() must be called before predict().")

        prev_frame, curr_frame = frame_pair
        h, w = curr_frame.shape[:2]

        # ROI area once, when a calibration exists.
        if self._calibration is not None and self._roi_area_m2 is None:
            min_x, min_y, max_x, max_y = self._calibration.image_footprint_world(w, h)
            self._roi_area_m2 = max((max_x - min_x) * (max_y - min_y), 1.0)
            self._metrics.roi_area_m2 = self._roi_area_m2

        # 1. Dense optical flow → field divergence (compression signal).
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            _FB_PYR_SCALE, _FB_LEVELS, _FB_WINSIZE,
            _FB_ITERATIONS, _FB_POLY_N, _FB_POLY_SIGMA, _FB_FLAGS,
        )
        div = np.gradient(flow[..., 0], axis=1) + np.gradient(flow[..., 1], axis=0)
        divergence_mean = float(np.mean(div))

        # 2. DM-Count head points.
        dm = self._counter.predict(curr_frame)

        # 3. Track points across frames.
        tracks = self._tracker.update(dm.points.tolist())

        # 4. Per-track velocity in run units.
        speeds: list[float] = []
        vxs: list[float] = []
        vys: list[float] = []
        positions: list[tuple[float, float]] = []
        rate = self._fps / max(1, self._frame_stride)  # predicts per second
        for t in tracks:
            vx, vy = t.vx, t.vy
            if self._calibration is not None:
                try:
                    vx, vy = self._calibration.pixel_velocity_to_ms(
                        t.x, t.y, t.vx, t.vy, rate)
                except Exception:  # noqa: BLE001 - keep the row, mark invalid
                    vx, vy = 0.0, 0.0
            vxs.append(vx)
            vys.append(vy)
            speeds.append(math.hypot(vx, vy))
            positions.append((t.x, t.y))

        # 5. The ten metrics.
        metrics = self._metrics.update(
            speeds=speeds, vxs=vxs, vys=vys,
            divergence_mean=divergence_mean,
            positions=positions, frame_shape=(h, w),
        )

        # 6. Alerts first, so they lead the row order for this frame.
        detections: list[Detection] = []
        for alert in self._alerts.evaluate(metrics, frame_index, timestamp_sec):
            detections.append(alert)
            self._alert_history.append({
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp_sec, 3),
                **alert.extra,
            })

        # 7. Per-head rows (confirmed tracks only).
        for t, spd, (vx, vy) in zip(tracks, speeds, zip(vxs, vys)):
            confirmed = t.age >= self.min_track_age
            if not confirmed:
                continue
            stopped = spd < self.stopped_speed
            label = "head_stopped" if stopped else "head_moving"
            x1 = max(0.0, t.x - _HEAD_HALF_BOX_PX)
            y1 = max(0.0, t.y - _HEAD_HALF_BOX_PX)
            x2 = min(float(w), t.x + _HEAD_HALF_BOX_PX)
            y2 = min(float(h), t.y + _HEAD_HALF_BOX_PX)
            conf = 0.0 if stopped else min(1.0, spd / max(self.stopped_speed * 5, 1e-6))
            detections.append(Detection(
                model_name=self.name,
                label=label,
                confidence=round(conf, 4),
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=[x1, y1, x2, y2],
                extra={
                    "track_id": t.track_id,
                    "x": round(t.x, 1),
                    "y": round(t.y, 1),
                    "dir_deg": round(math.degrees(math.atan2(t.vy, t.vx)) % 360.0, 2)
                               if t.speed_px > 1e-6 else None,
                    "status": "STOPPED" if stopped else "MOVING",
                    "speed": round(spd, 4),
                    "speed_unit": metrics.speed_unit,
                    "density_value": round(float(t.value), 4),
                    "track_age": t.age,
                    "head_count": metrics.head_count,
                },
            ))

        # 8. One telemetry row with the whole FrameMetrics record.
        detections.append(Detection(
            model_name=self.name,
            label="dm_frame_metrics",
            confidence=0.0,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            bbox=None,
            extra={
                "alert": False,
                "head_count": metrics.head_count,
                "density": metrics.density,
                "density_unit": metrics.density_unit,
                "mean_speed": metrics.mean_speed,
                "speed_unit": metrics.speed_unit,
                "velocity_variance": metrics.velocity_variance,
                "pressure": metrics.pressure,
                "divergence": metrics.divergence,
                "directional_entropy": metrics.directional_entropy,
                "dominant_dir_deg": metrics.dominant_dir_deg,
                "counter_flow_pct": metrics.counter_flow_pct,
                "stop_and_go": metrics.stop_and_go,
                "oscillation": metrics.oscillation,
                "occupancy_ratio": metrics.occupancy_ratio,
                "n_moving_tracks": metrics.n_moving_tracks,
                "n_stopped_tracks": metrics.n_stopped_tracks,
                "cell_density_mean": round(float(metrics.cell_density.mean()), 5),
                "cell_pressure_max": round(float(metrics.cell_pressure.max()), 5),
                "inference_s": round(dm.elapsed_s, 4),
            },
        ))

        # 9. Overlay + streaming write.
        annotated = self._render_overlay(curr_frame, dm, tracks, speeds, metrics)
        self.latest_annotated_frame = annotated
        self._write_frame(annotated)

        # 10. Summary accumulation.
        self._frames_processed += 1
        self._head_counts.append(metrics.head_count)
        self._mean_speeds.append(metrics.mean_speed)
        self._pressures.append(metrics.pressure)
        self._entropies.append(metrics.directional_entropy)
        self._stop_go.append(metrics.stop_and_go)
        self._oscillations.append(metrics.oscillation)
        self._counterflow_pcts.append(metrics.counter_flow_pct)

        return detections

    # ──────────────────────────────────────────────────────────────────────
    # finalize — called by the runner after all frames
    # ──────────────────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """Compute run-level summary, close video, publish annotated_video_path."""
        n = len(self._head_counts)
        alerts_by_label: dict[str, int] = {}
        for a in self._alert_history:
            alerts_by_label[a.get("metric", a.get("alert_severity", "?"))] = (
                alerts_by_label.get(a.get("metric", "?"), 0) + 1)

        def peak(series):
            if not series:
                return 0.0, 0.0
            i = int(np.argmax(series))
            return float(max(series)), round(i * (self.output_fps or (self._fps / max(1, self._frame_stride))) ** -1, 2)

        peak_pressure, peak_pressure_ts = peak(self._pressures)
        peak_heads, peak_heads_ts = peak(self._head_counts)

        self.summary = {
            "total_frames": self._frames_processed,
            "head_count_mean": round(float(np.mean(self._head_counts)), 2) if n else 0,
            "head_count_peak": peak_heads,
            "peak_head_count_timestamp_sec": peak_heads_ts,
            "density_unit": "heads/m2" if self._calibration is not None else "heads/frame",
            "speed_unit": "m/s" if self._calibration is not None else "px/frame",
            "mean_speed": round(float(np.mean(self._mean_speeds)), 3) if n else 0.0,
            "pressure_mean": round(float(np.mean(self._pressures)), 5) if n else 0.0,
            "pressure_peak": peak_pressure,
            "peak_pressure_timestamp_sec": peak_pressure_ts,
            "directional_entropy_mean": round(float(np.mean(self._entropies)), 4) if n else 0.0,
            "counter_flow_pct_mean": round(float(np.mean(self._counterflow_pcts)), 2) if n else 0.0,
            "stop_and_go_mean": round(float(np.mean(self._stop_go)), 4) if n else 0.0,
            "oscillation_mean": round(float(np.mean(self._oscillations)), 4) if n else 0.0,
            "total_tracks": max(0, (self._tracker._next_id - 1)) if self._tracker else 0,
            "total_alerts": len(self._alert_history),
            "alert_counts_by_metric": alerts_by_label,
            "alert_history": self._alert_history,
            "calibrated": self._calibration is not None,
        }

        self._close_video()

    # ──────────────────────────────────────────────────────────────────────
    # Rendering helpers
    # ──────────────────────────────────────────────────────────────────────

    def _render_overlay(self, frame, dm, tracks, speeds, metrics: FrameMetrics):
        annotated = frame.copy()

        # Density heatmap underlay — the thing only this model can draw.
        if self.show_density_heatmap and dm.density.size:
            dens = dm.density
            if float(dens.max()) > 0:
                norm = (dens / dens.max()).astype(np.float32)
            else:
                norm = dens.astype(np.float32)
            heat_8u = cv2.resize(norm, (annotated.shape[1], annotated.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            heat = cv2.applyColorMap((heat_8u * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
            cv2.addWeighted(heat, 0.35, annotated, 0.65, 0, dst=annotated)

        # Arrows: direction + state per track.
        for t, spd in zip(tracks, speeds):
            cx, cy = int(round(t.x)), int(round(t.y))
            pending = t.age < self.min_track_age
            colour = _COLOUR_PENDING if pending else (
                _COLOUR_STOPPED if spd < self.stopped_speed else _COLOUR_MOVING)
            if not pending and t.speed_px > 1e-3:
                length = float(np.clip(_HEAD_HALF_BOX_PX * 2 * t.speed_px,
                                       _ARROW_MIN_PX, _ARROW_MAX_PX))
                tip = (int(cx + t.vx / t.speed_px * length),
                       int(cy + t.vy / t.speed_px * length))
                cv2.arrowedLine(annotated, (cx, cy), tip, colour, 2,
                                cv2.LINE_AA, tipLength=0.35)
            cv2.circle(annotated, (cx, cy), 4, colour, -1, cv2.LINE_AA)

        # HUD.
        unit_v = metrics.speed_unit.replace("px/frame", "px/f")
        hud = [
            f"Heads: {metrics.head_count}  ({metrics.density:.1f} {metrics.density_unit})",
            f"Speed: {metrics.mean_speed:.2f} {unit_v}   Pressure: {metrics.pressure:.3f}",
            f"Entropy: {metrics.directional_entropy:.2f}  Stop&Go: {metrics.stop_and_go:.2f}"
            f"  Counterflow: {metrics.counter_flow_pct:.0f}%",
        ]
        for i, line in enumerate(hud):
            cv2.putText(annotated, line, (10, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line, (10, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOUR_TEXT, 1, cv2.LINE_AA)

        return annotated

    def _write_frame(self, annotated: np.ndarray) -> None:
        """Stream one annotated frame to the encoder, opening it on first use."""
        if self._writer is None:
            h, w = annotated.shape[:2]
            stem = os.path.splitext(os.path.basename(self._video_name))[0]
            out_path = os.path.join(
                self._output_dir, f"{stem}_{self.name}.mp4")
            fps = self.output_fps or self._fps
            self._writer = _AnnotatedVideoWriter(out_path, fps, w, h)
            logger.info(
                "DMCountCrowdMonitor: opening annotated video at %s (%.2f fps %dx%d)",
                out_path, fps, w, h,
            )
        if self._writer.write(annotated):
            self._frames_written += 1

    def _close_video(self) -> None:
        """Close the encoder and set annotated_video_path."""
        if self._writer is None:
            return
        path = self._writer.close()
        self._writer = None
        if path is not None:
            self.annotated_video_path = path
            logger.info(
                "DMCountCrowdMonitor: annotated video written: %s (%d frames)",
                path, self._frames_written,
            )
