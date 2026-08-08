"""
DenseFlowAnalyser — main BaseModelWrapper for the crowd-safety optical flow module.

Wires together:
  - FlowField:           DIS (or Farneback) flow, GMC, smoothing
  - CameraCalibration:   pixel → m/s conversion
  - DetectorMaskLayer:   vehicle + umbrella masking, detector stride
  - CrowdMetricsEngine:  divergence, curl, counterflow, stop-go, turbulence
  - AlertEngine:         hysteresis + duration gating, per-zone thresholds
  - FlowVisualiser:      HSV overlay, heatmap, zone overlay, detector boxes

This class is consumption_type = "flow_pair" so the existing pipeline runner
calls predict((prev_frame, curr_frame), frame_index, timestamp_sec) unchanged.

Performance (measured, CPU, 1920×1080 source, no detectors configured)
----------------------------------------------------------------------
  Flow (DIS MEDIUM at 320×180 + GMC + reliability)   ~100 ms/frame
  Grid metrics (block-reduced over the source field) ~145 ms/frame
  Visualisation (HSV + heatmap + arrows + labels)     ~90 ms/frame
  Alert engine, logging                                <5 ms/frame
  ──────────────────────────────────────────────────────────────
  Total                                              ~415 ms/frame  (~2.4 fps)

Add ~30-60 ms per configured detector, amortised over ``detector_stride``
frames.

This is NOT real-time at 1080p, and the module should not be described as
such.  The dominant remaining cost is that the metrics and overlays run at
source resolution while the flow field itself only carries information at the
compute resolution (320 px by default) — the extra pixels are interpolation,
not measurement.  Two levers, in order of effect:

  - Run at a lower source resolution.  Halving each dimension quarters the
    per-frame cost of everything downstream of the flow computation.
  - Raise ``grid_cell_px`` and ``overlay_work_px``, or set visualise=False
    for headless deployments, which removes the visualisation term entirely.

Moving the metrics layer onto the compute-resolution field would remove most
of the remaining cost, but it changes the MetricsFrame coordinate contract
and every consumer of it; it has not been done here.

Output: predict() returns one Detection per active alert.
  - Detection.label   = "<metric>_<severity>"  e.g. "divergence_critical"
  - Detection.bbox    = zone bounding box [x1,y1,x2,y2]
  - Detection.confidence = min(1, |measured| / |threshold|)
  - Detection.extra   = full alert dict + per-zone metrics snapshot

finalize() writes the CSV/Parquet metrics log and time-series PNG plots.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import cv2

from models.base import BaseModelWrapper, Detection
from models.crowd_flow.flow_field    import FlowField, FlowResult
from models.crowd_flow.ground_plane  import CameraCalibration
from models.crowd_flow.detector_masks import DetectorMaskLayer
from models.crowd_flow.crowd_metrics import CrowdMetricsEngine, MetricsFrame
from models.crowd_flow.zones         import Zone, AlertEngine, AlertSeverity
from models.crowd_flow.visualise     import FlowVisualiser

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class _AnnotatedVideoWriter:
    """
    Streaming H.264 writer for annotated frames.

    Frames are piped to ffmpeg as they are produced.  Falls back to MJPG in an
    AVI container when ffmpeg is not on PATH — playable in VLC, but not in a
    browser, so the fallback is logged rather than silent.
    """

    def __init__(self, out_path: str, fps: float, w: int, h: int) -> None:
        from pipeline.ffmpeg import find_ffmpeg
        import subprocess

        self._path = out_path
        self._proc = None
        self._writer = None
        self._failed = False

        fps = max(float(fps), 1.0)
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                "-r", f"{fps:.4f}",
                "-i", "-", "-an",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "23",
                "-movflags", "+faststart",
                out_path,
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        else:
            logger.warning(
                "ffmpeg not found; falling back to MJPG/AVI for the annotated "
                "dense-flow video.  The file will play in VLC but not in a "
                "browser."
            )
            self._path = out_path.replace(".mp4", ".avi")
            self._writer = cv2.VideoWriter(
                self._path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h),
            )

    def write(self, frame: np.ndarray) -> bool:
        if self._failed:
            return False
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
                return True
            except (BrokenPipeError, OSError) as exc:
                logger.error("Annotated video encoder stopped accepting data: %s", exc)
                self._failed = True
                return False
        if self._writer is not None:
            self._writer.write(frame)
            return True
        return False

    def close(self) -> Optional[str]:
        """Finish the file.  Returns its path, or None if encoding failed."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            _, err = self._proc.communicate()
            if self._proc.returncode != 0:
                logger.error(
                    "ffmpeg failed (exit %d): %s",
                    self._proc.returncode, err.decode("utf-8", "replace").strip(),
                )
                return None
        if self._writer is not None:
            self._writer.release()
        return None if self._failed else self._path


class DenseFlowAnalyser(BaseModelWrapper):
    """
    Dense optical flow crowd-safety analyser.

    Parameters
    ----------
    config : dict
        The ``crowd_flow`` subtree from configs/crowd_flow.yaml (already
        parsed).  Loaded by the CLI or by callers who instantiate this class
        directly.  See configs/crowd_flow.yaml for all keys.
    camera_id : str
        Which camera block to load zones and calibration from.
    output_dir : str
        Where to write per-frame CSV, metrics log, and time-series plots.
    visualise : bool
        Whether to produce annotated frames (disable for headless servers).
    fps : float
        Source frame rate; used for speed conversion and stop-go timing.
    device : str | None
        Passed to the embedded detector models.
    """

    consumption_type = "flow_pair"
    name             = "dense_flow"
    gpu_accelerated  = False   # flow is CPU; GPU used only by embedded detectors

    def __init__(
        self,
        config: dict,
        camera_id: str = "default",
        output_dir: str = "outputs",
        visualise: bool = True,
        fps: float = 30.0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self._config    = config
        self._camera_id = camera_id
        self._output_dir = output_dir
        self._visualise  = visualise
        self._fps        = fps

        # Playback rate of the annotated video.  This is NOT self._fps when
        # the caller samples frames: predict() is invoked once per SAMPLED
        # frame, so with a stride of 5 the output holds every fifth frame.
        # Writing those at the source rate replays 5 seconds of footage in
        # one — the output looks time-lapsed, and any speed read off it by
        # eye is wrong by the stride factor.  Callers that sample must set
        # output_fps = source_fps / stride.
        self.output_fps: Optional[float] = None

        # Sub-modules (initialised in load())
        self._flow:    Optional[FlowField]          = None
        self._calib:   Optional[CameraCalibration]  = None
        self._masks:   Optional[DetectorMaskLayer]  = None
        self._metrics: Optional[CrowdMetricsEngine] = None
        self._alerts:  Optional[AlertEngine]        = None
        self._vis:     Optional[FlowVisualiser]     = None

        self._zones: list[Zone] = []
        self._zones_resolved = False

        # Per-run accumulators (reset in load(); written in finalize())
        self._metrics_log: list[dict] = []
        self._frame_count = 0

        # Annotated frames are streamed to the encoder as they are produced.
        # Buffering the whole run in memory first — even as encoded JPEG —
        # costs roughly 100 KB per 720p frame, so an hour of footage at 25 fps
        # needs about 9 GB before a single byte is written.
        self._writer: Optional["_AnnotatedVideoWriter"] = None
        self._frames_written = 0
        self._source_hw: Optional[tuple[int, int]] = None  # (H, W)

        # Path to the annotated video written by finalize() -- picked up by jobs.py
        self.annotated_video_path: Optional[str] = None

        # Latest rendered frame (exposed for the web UI preview)
        self.latest_annotated_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialise all sub-modules from config."""
        cfg = self._config
        cam_cfg = cfg.get("cameras", {}).get(self._camera_id, {})

        # Flow field
        self._flow = FlowField(
            backend=cfg.get("flow_backend", "dis"),
            dis_preset=cfg.get("dis_preset", "medium"),
            target_px=cfg.get("downsample_target_px", 320),
            temporal_smooth_alpha=cfg.get("temporal_smooth_alpha", 0.4),
            global_motion_compensation=cfg.get("global_motion_compensation", True),
            gmc_warn_threshold_px=cfg.get("gmc_warn_threshold_px", 3.0),
            gmc_max_correction_px=cfg.get("gmc_max_correction_px", 8.0),
            gmc_min_improvement=cfg.get("gmc_min_improvement", 0.9),
            rain_mag_threshold=cfg.get("rain_mag_threshold", 8.0),
            lowlight_gradient_threshold=cfg.get("lowlight_gradient_threshold", 12.0),
            brightness_jump_threshold=cfg.get("brightness_jump_threshold", 30.0),
            min_flow_reliability=cfg.get("min_flow_reliability", 0.35),
        )

        # Ground plane calibration
        self._calib = CameraCalibration.from_yaml_block(self._camera_id, cam_cfg)

        # Zones
        self._zones = [Zone.from_dict(z) for z in cam_cfg.get("zones", [])]
        if not self._zones:
            logger.warning(
                "Camera '%s' has no zones configured.  "
                "No per-zone metrics or alerts will be produced.  "
                "Add a zones: list under cameras.%s in configs/crowd_flow.yaml.",
                self._camera_id, self._camera_id,
            )

        # Detector mask layer
        self._masks = DetectorMaskLayer(
            vehicle_model_path  = cfg.get("vehicle_model_path", ""),
            umbrella_model_path = cfg.get("umbrella_model_path", ""),
            device    = self.device,
            stride    = cfg.get("detector_stride", 5),
            conf_threshold = cfg.get("detector_conf_threshold", 0.35),
        )
        self._masks.load()

        # Metrics engine (includes sign-convention self-test)
        self._metrics = CrowdMetricsEngine(
            grid_cell_px=cfg.get("grid_cell_px", 16),
            min_magnitude=cfg.get("min_magnitude", 0.3),
            stop_go_lag_frames=cfg.get("stop_go_lag_frames", [5, 10, 15]),
        )

        # Alert engine
        self._alerts = AlertEngine(
            zones=self._zones,
            fps=self._fps,
            is_calibrated=self._calib.is_calibrated,
            speed_units=self._calib.speed_units,
        )

        # Visualiser
        if self._visualise:
            self._vis = FlowVisualiser(
                flow_alpha=cfg.get("flow_overlay_alpha", 0.5),
                heatmap_alpha=cfg.get("heatmap_alpha", 0.55),
                arrow_step=cfg.get("arrow_step_px", 20),
                max_magnitude_px=cfg.get("max_magnitude_px", 20.0),
                div_scale=cfg.get("div_scale", 3.0),
                auto_scale_overlay=cfg.get("auto_scale_overlay", True),
                min_draw_magnitude_px=cfg.get("min_draw_magnitude_px", 0.4),
                background_floor_factor=cfg.get("background_floor_factor", 3.0),
                overlay_work_px=cfg.get("overlay_work_px", 480),
            )

        # Output directory
        os.makedirs(self._output_dir, exist_ok=True)
        self._metrics_log = []
        self._frame_count = 0
        self._close_video()
        self._frames_written = 0
        self._source_hw = None
        self.annotated_video_path = None
        self._zones_resolved = False
        self._flow.reset()

        logger.info(
            "DenseFlowAnalyser loaded for camera '%s'.  "
            "Calibrated: %s.  Zones: %s.  Vehicle detector: %s.  "
            "Umbrella detector: %s.",
            self._camera_id,
            self._calib.is_calibrated,
            [z.name for z in self._zones],
            self._masks._vehicle_loaded,
            self._masks._umbrella_loaded,
        )

    # ------------------------------------------------------------------
    # predict — called per frame pair by the pipeline runner
    # ------------------------------------------------------------------

    def predict(
        self,
        frame_pair: tuple[np.ndarray, np.ndarray],
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        if self._flow is None:
            raise RuntimeError(
                "DenseFlowAnalyser.load() must be called before predict()."
            )

        prev_frame, curr_frame = frame_pair
        self._frame_count += 1

        # Zone polygons are resolved against the real frame size the first
        # time a frame is seen: fractional polygons are scaled up, and pixel
        # polygons authored for a different resolution are reported.
        if not self._zones_resolved:
            self._resolve_zones(curr_frame.shape[1], curr_frame.shape[0])

        # 1. Update detector masks
        self._masks.update(curr_frame, frame_index)
        ped_mask = self._masks.pedestrian_flow_mask(curr_frame.shape)

        # 2. Compute flow
        t0 = time.perf_counter()
        flow_result = self._flow.compute(
            prev_frame, curr_frame,
            exclusion_mask=self._masks.vehicle_mask(curr_frame.shape),
            timestamp_sec=timestamp_sec,
        )
        t_flow = (time.perf_counter() - t0) * 1000

        # 3. Compute metrics
        t1 = time.perf_counter()
        mf = self._metrics.compute(
            flow_result=flow_result,
            zones=self._zones,
            calibration=self._calib,
            ped_mask=ped_mask,
            umbrella_layer=self._masks,
            fps=self._fps,
        )
        t_metrics = (time.perf_counter() - t1) * 1000

        # 4. Check alerts
        vehicle_in_ped: dict[str, bool] = {
            z.name: self._masks.vehicle_in_pedestrian_zone(z.polygon)
            for z in self._zones
        }
        alerts = self._alerts.update(
            zone_metrics={n: zm for n, zm in mf.zones.items()},
            vehicle_in_ped=vehicle_in_ped,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )

        # 5. Log metrics
        self._log_metrics(mf, flow_result, frame_index, timestamp_sec)

        # 6. Visualise
        t2 = time.perf_counter()
        if self._vis is not None:
            # Order matters: the two translucent field overlays go down first,
            # then the line-art (arrows, zone outlines, boxes) on top.  Drawing
            # arrows before the heatmap blends them away.
            annotated = self._vis.hsv_flow_overlay(curr_frame, flow_result.field_xy)
            annotated = self._vis.divergence_heatmap(
                annotated, mf.cell_divergence, mf.cell_size_px,
                cell_speed=mf.cell_speed,
            )
            annotated = self._vis.sparse_arrow_grid(annotated, flow_result.field_xy)
            annotated = self._vis.zone_overlay(
                annotated, self._zones, mf.zones, alerts
            )
            annotated = self._vis.detector_boxes(
                annotated,
                self._masks.vehicle_boxes,
                self._masks.umbrella_boxes,
            )
            annotated = self._vis.info_banner(
                annotated,
                flags={
                    "rain_flag":             flow_result.is_rain_flagged,
                    "lowlight_flag":         flow_result.is_lowlight_flagged,
                    "brightness_suppressed": flow_result.is_brightness_suppressed,
                    "gmc_applied":           flow_result.gmc_applied,
                    "gmc_method":            flow_result.gmc_method,
                    "calibrated":            self._calib.is_calibrated,
                    "discontinuity":         flow_result.is_discontinuity_flagged,
                    "flow_reliability":      flow_result.flow_reliability,
                },
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
            self.latest_annotated_frame = annotated
            self._source_hw = (curr_frame.shape[0], curr_frame.shape[1])
            self._write_frame(annotated)
        t_vis = (time.perf_counter() - t2) * 1000

        logger.debug(
            "f=%d  flow=%.1fms  metrics=%.1fms  vis=%.1fms  "
            "alerts=%d  veh=%d  umb=%d",
            frame_index, t_flow, t_metrics, t_vis,
            len(alerts), self._masks.vehicle_count, self._masks.umbrella_count,
        )

        # 7. Convert alerts -> Detection objects for the runner.
        # Always emit at least one per-frame summary Detection so the UI shows
        # the model is active, even when no threshold is crossed.
        alert_dets = [self._alert_to_detection(a, frame_index, timestamp_sec)
                      for a in alerts]

        # Per-frame summary detection (label = "flow_analysis"):
        # carries the first zone's key metrics so the event log has data.
        if self._zones and mf.zones:
            zm = next(iter(mf.zones.values()))
            zone_name = next(iter(mf.zones))
            zone_bb = [0, 0, curr_frame.shape[1], curr_frame.shape[0]]
            for z in self._zones:
                if z.name == zone_name:
                    x1, y1, x2, y2 = z.bbox()
                    zone_bb = [x1, y1, x2, y2]
                    break
            summary_det = Detection(
                model_name=self.name,
                label="flow_analysis",
                confidence=float(np.clip(abs(zm.mean_divergence) / 2.0, 0.0, 1.0)),
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=zone_bb,
                extra={
                    "zone": zone_name,
                    "mean_divergence":   round(zm.mean_divergence, 4),
                    "mean_speed":        round(zm.mean_speed, 4),
                    "counterflow_score": round(zm.counterflow_score, 4),
                    "stop_go_score":     round(zm.stop_go_score, 4),
                    "turbulence_index":  round(zm.turbulence_index, 4),
                    "speed_units":       mf.speed_units,
                    "rain_flag":         flow_result.is_rain_flagged,
                    "lowlight_flag":     flow_result.is_lowlight_flagged,
                    "flow_reliability":  round(flow_result.flow_reliability, 3),
                    "discontinuity":     flow_result.is_discontinuity_flagged,
                },
            )
            return alert_dets + [summary_det]

        return alert_dets

    # ------------------------------------------------------------------
    # finalize — called by runner after all frames are processed
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Close the annotated video, write the CSV log and time-series plots."""
        # Always close the encoder, even with no metrics: an open ffmpeg pipe
        # would leave a truncated, unplayable file behind.
        self._close_video()

        if not self._metrics_log:
            logger.info("DenseFlowAnalyser: no metrics to write.")
            return

        self._write_csv()

        if self._config.get("output_parquet", False):
            self._write_parquet()

        if self._config.get("output_timeseries_plots", True) and self._vis is not None:
            self._vis.save_timeseries_plots(
                self._metrics_log, self._output_dir, self._camera_id
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_frame(self, annotated: np.ndarray) -> None:
        """Stream one annotated frame to the encoder, opening it on first use."""
        if self._writer is None:
            h, w = annotated.shape[:2]
            out_path = os.path.join(
                self._output_dir, f"{self._camera_id}_dense_flow_annotated.mp4"
            )
            fps = self.output_fps or self._fps
            if self.output_fps and abs(self.output_fps - self._fps) > 0.01:
                logger.info(
                    "Annotated video will be written at %.2f fps (source is "
                    "%.2f fps) because frames are sampled; this keeps playback "
                    "at real-world speed instead of time-lapsing it.",
                    fps, self._fps,
                )
            self._writer = _AnnotatedVideoWriter(out_path, fps, w, h)

        if self._writer.write(annotated):
            self._frames_written += 1

    def _close_video(self) -> None:
        """Close the encoder and publish the output path."""
        if self._writer is None:
            return
        path = self._writer.close()
        self._writer = None
        if path is not None:
            self.annotated_video_path = path
            logger.info(
                "Annotated dense-flow video written: %s (%d frames)",
                path, self._frames_written,
            )

    def _resolve_zones(self, width: int, height: int) -> None:
        """
        Resolve zone polygons against the actual frame size, once per run.

        Fractional polygons (all vertices in [0, 1]) are scaled to pixels.
        Pixel polygons that fall substantially outside the frame are reported
        as WARNING: a polygon authored for a 640×480 preview applied to a
        1280×720 stream covers a quarter of the intended area, in the wrong
        place, and every downstream number is quietly computed over the wrong
        region.  The zone is still processed — the operator decides whether
        the numbers are usable — but the mismatch is not silent.

        Zones with no overlap at all are dropped: they would report zeros for
        the whole run and read as "no crowd activity".
        """
        resolved: list[Zone] = []
        for zone in self._zones:
            was_fractional = zone.is_fractional
            z = zone.resolve_to_frame(width, height)

            coverage = z.coverage_fraction(width, height)
            if coverage <= 0.0:
                logger.warning(
                    "Zone '%s' lies entirely outside the %dx%d frame "
                    "(bbox %s) and has been DROPPED.  Its polygon in "
                    "configs/crowd_flow.yaml was authored for a different "
                    "resolution; re-author it, or express it as fractions of "
                    "the frame (values between 0 and 1).",
                    z.name, width, height, z.bbox(),
                )
                continue
            if coverage < 0.95:
                logger.warning(
                    "Zone '%s' is only %.0f%% inside the %dx%d frame "
                    "(bbox %s).  Metrics for this zone cover the clipped "
                    "region only.  Re-author the polygon for this "
                    "resolution, or express it as fractions of the frame.",
                    z.name, 100 * coverage, width, height, z.bbox(),
                )
            elif was_fractional:
                logger.info(
                    "Zone '%s' resolved from frame fractions to %s at %dx%d.",
                    z.name, z.bbox(), width, height,
                )
            resolved.append(z)

        self._zones = resolved
        self._zones_resolved = True

        # AlertEngine holds references to the pre-resolution Zone objects and
        # keys its threshold state by zone name, so it must be rebuilt against
        # the resolved list (which may be shorter).
        self._alerts = AlertEngine(
            zones=self._zones,
            fps=self._fps,
            is_calibrated=self._calib.is_calibrated,
            speed_units=self._calib.speed_units,
        )

    def _alert_to_detection(
        self, alert, frame_index: int, timestamp_sec: float
    ) -> Detection:
        # Find zone bounding box for the bbox field
        zone_bb = [0, 0, 1, 1]
        for z in self._zones:
            if z.name == alert.zone_name:
                x1, y1, x2, y2 = z.bbox()
                zone_bb = [x1, y1, x2, y2]
                break

        # Confidence: how far above threshold (capped at 1)
        if abs(alert.threshold_value) > 1e-9:
            conf = min(1.0, abs(alert.measured_value) / abs(alert.threshold_value))
        else:
            conf = 1.0

        return Detection(
            model_name=self.name,
            label=alert.label_for_detection(),
            confidence=conf,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            bbox=zone_bb,
            extra=alert.as_dict(),
        )

    def _log_metrics(
        self,
        mf: MetricsFrame,
        flow_result: FlowResult,
        frame_index: int,
        timestamp_sec: float,
    ) -> None:
        """Append one row to the in-memory metrics log."""
        entry: dict[str, Any] = {
            "frame_index":         frame_index,
            "timestamp_sec":       round(timestamp_sec, 3),
            "camera_id":           self._camera_id,
            "is_calibrated":       mf.is_calibrated,
            "speed_units":         mf.speed_units,
            "rain_flag":           flow_result.is_rain_flagged,
            "lowlight_flag":       flow_result.is_lowlight_flagged,
            "brightness_suppressed": flow_result.is_brightness_suppressed,
            "flow_reliability":    round(flow_result.flow_reliability, 3),
            "discontinuity_flag":  flow_result.is_discontinuity_flagged,
            "gmc_applied":         flow_result.gmc_applied,
            "gmc_tx_px":           round(flow_result.gmc_translation_px[0], 3),
            "gmc_ty_px":           round(flow_result.gmc_translation_px[1], 3),
            "gmc_method":          flow_result.gmc_method,
            "frame_mean_gradient": round(flow_result.frame_mean_gradient, 3),
            "frame_mean_magnitude": round(flow_result.frame_mean_magnitude, 3),
            "vehicle_count":       self._masks.vehicle_count,
            "umbrella_count":      self._masks.umbrella_count,
            "zones": {
                name: {
                    "mean_speed":       round(zm.mean_speed, 4),
                    "mean_divergence":  round(zm.mean_divergence, 4),
                    "mean_curl":        round(zm.mean_curl, 4),
                    "mean_coherence":   round(zm.mean_coherence, 4),
                    "mean_variance":    round(zm.mean_variance, 4),
                    "stop_go_score":    round(zm.stop_go_score, 4),
                    "counterflow_score": round(zm.counterflow_score, 4),
                    "turbulence_index": round(zm.turbulence_index, 4),
                    "active_cells":     zm.active_cells,
                    "umbrella_occluded_frac": round(zm.umbrella_occluded_frac, 4),
                }
                for name, zm in mf.zones.items()
            },
        }
        self._metrics_log.append(entry)

    def _write_csv(self) -> None:
        """Flatten the metrics log and write to CSV."""
        if not self._metrics_log:
            return

        # Build flat fieldnames: global fields + per-zone columns
        zone_names = sorted(
            {z for entry in self._metrics_log for z in entry.get("zones", {})}
        )
        zone_metric_names = [
            "mean_speed", "mean_divergence", "mean_curl", "mean_coherence",
            "mean_variance", "stop_go_score", "counterflow_score",
            "turbulence_index", "active_cells", "umbrella_occluded_frac",
        ]

        global_fields = [
            "frame_index", "timestamp_sec", "camera_id", "is_calibrated",
            "speed_units", "rain_flag", "lowlight_flag", "brightness_suppressed",
            "gmc_applied", "gmc_tx_px", "gmc_ty_px", "gmc_method",
            "flow_reliability", "discontinuity_flag",
            "frame_mean_gradient", "frame_mean_magnitude",
            "vehicle_count", "umbrella_count",
        ]
        zone_fields = [
            f"{zn}__{mn}" for zn in zone_names for mn in zone_metric_names
        ]
        fieldnames = global_fields + zone_fields

        out_path = os.path.join(
            self._output_dir, f"{self._camera_id}_flow_metrics.csv"
        )
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in self._metrics_log:
                row = {k: entry[k] for k in global_fields if k in entry}
                for zn in zone_names:
                    zm = entry.get("zones", {}).get(zn, {})
                    for mn in zone_metric_names:
                        row[f"{zn}__{mn}"] = zm.get(mn, "")
                writer.writerow(row)

        logger.info("Flow metrics CSV written to %s (%d rows)", out_path,
                    len(self._metrics_log))

    def _write_parquet(self) -> None:
        """Write metrics log to Parquet.  Requires pandas + pyarrow."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed; Parquet output skipped.")
            return

        # Flatten same as CSV
        rows = []
        zone_names = sorted(
            {z for entry in self._metrics_log for z in entry.get("zones", {})}
        )
        zone_metric_names = [
            "mean_speed", "mean_divergence", "mean_curl", "mean_coherence",
            "mean_variance", "stop_go_score", "counterflow_score",
            "turbulence_index", "active_cells", "umbrella_occluded_frac",
        ]
        for entry in self._metrics_log:
            row = {k: v for k, v in entry.items() if k != "zones"}
            for zn in zone_names:
                zm = entry.get("zones", {}).get(zn, {})
                for mn in zone_metric_names:
                    row[f"{zn}__{mn}"] = zm.get(mn, None)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = os.path.join(
            self._output_dir, f"{self._camera_id}_flow_metrics.parquet"
        )
        df.to_parquet(out_path, index=False)
        logger.info("Flow metrics Parquet written to %s", out_path)
