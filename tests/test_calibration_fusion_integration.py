"""
Integration tests for perspective calibration via fit_from_boxes and FusionEngine rule activation.

Validates that:
1. CameraCalibration loads perspective_map from YAML blocks.
2. DenseFlowAnalyser loaded with camera_id="CCTV1" or "CCTV2" inherits is_calibrated=True.
3. MetricStore updates for calibrated cameras mark flow_is_calibrated=True.
4. FusionEngine evaluates capacity and accumulation rules when cameras are calibrated.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_io import load_yaml_dict
from models.crowd_flow.ground_plane import CameraCalibration
from webapp.registry import build_model
from topology.graph import CameraTopology, DEFAULT_TOPOLOGY_PATH
from topology.metric_store import MetricStore
from topology.fusion_engine import FusionEngine


def test_camera_calibration_loads_perspective_map():
    cfg_path = os.path.join(_PROJECT_ROOT, "configs", "crowd_flow.yaml")
    cfg = load_yaml_dict(cfg_path).get("crowd_flow", {})
    cams = cfg.get("cameras", {})

    assert "CCTV1" in cams
    c1_calib = CameraCalibration.from_yaml_block("CCTV1", cams["CCTV1"])
    assert c1_calib.is_calibrated is True
    assert c1_calib.speed_units == "m/s"
    assert c1_calib.perspective_map is not None
    assert c1_calib.perspective_map.source == "fit_from_boxes"

    # Conversion of pixel displacement to m/s
    vx, vy = c1_calib.pixel_velocity_to_ms(x=384.0, y=288.0, dx_px=2.0, dy_px=0.0, fps=25.0)
    assert vx > 0.0
    assert vy == 0.0


def test_dense_flow_model_loads_camera_calibration():
    # build_model with camera_id="CCTV1"
    model = build_model("dense_flow", device="cpu", camera_id="CCTV1")
    model.load()
    assert model._calib is not None
    assert model._calib.is_calibrated is True


def test_calibrated_cameras_activate_fusion_rules():
    """
    When CCTV1 and CCTV2 are calibrated, the fusion engine must NOT block
    with 'uncalibrated_flow'. It must evaluate and fire the bottleneck alert.
    """
    # Explicit baseline, not the ambient global. Bare CameraTopology() reads
    # whichever config exists on disk, and a Route Builder run leaves a
    # generated file that takes precedence -- which made this test depend on
    # whether someone had clicked "Save route" beforehand.
    topo = CameraTopology(DEFAULT_TOPOLOGY_PATH)
    store = MetricStore()
    engine = FusionEngine(topology=topo, metric_store=store)

    now_ms = 1700000050000

    # CCTV1: flow 320 pax/min, calibrated
    store.update(
        camera_id="CCTV1",
        flow_rate_pax_min=320.0,
        flow_is_calibrated=True,
        explicit_epoch_ms=now_ms - 25000,
    )
    # CCTV2: flow 260 pax/min, calibrated
    store.update(
        camera_id="CCTV2",
        flow_rate_pax_min=260.0,
        flow_is_calibrated=True,
        explicit_epoch_ms=now_ms - 20000,
    )
    # CCTV3: target
    store.update(
        camera_id="CCTV3",
        density=1.0,
        flow_rate_pax_min=100.0,
        flow_is_calibrated=True,
        explicit_epoch_ms=now_ms,
    )

    alerts = engine.evaluate_step(current_time_epoch_ms=now_ms)
    status = engine.get_forecast_status("CCTV3")

    # Must NOT be blocked
    assert status.get("blocked") is None
    assert len(alerts) >= 1
    levels = {a.level for a in alerts}
    assert "BOTTLENECK_PREDICTED" in levels
