from models.base import BaseModelWrapper, Detection
from models.fire_smoke_yolo import FireSmokeYOLO
from models.optical_flow_crush import OpticalFlowCrushDetector
from models.roboflow_combined import RoboflowCombinedDetector

from models.fall import (
    YOLOPoseFallDetector,
    MediaPipeFallDetector,
    AlphaPoseFallDetector,
    STGCNFallDetector,
    PoseC3DFallDetector,
    MoveNetFallDetector,
    OpticalFlowFallDetector,
)
from models.violence import (
    X3DViolenceClassifier,
    SlowFastViolenceClassifier,
    VideoMAEViolenceClassifier,
    I3DViolenceClassifier,
    C3DViolenceClassifier,
    TSMViolenceClassifier,
    MMActionSlowOnlyClassifier,
)
from models.traffic import (
    YoloTrafficDetector,
    RtdetrTrafficDetector,
    RoboflowTrafficDetector,
    Mog2ParkedDetector,
)

__all__ = [
    "BaseModelWrapper",
    "Detection",
    "FireSmokeYOLO",
    "OpticalFlowCrushDetector",
    "RoboflowCombinedDetector",
    # Fall detection (7)
    "YOLOPoseFallDetector",
    "MediaPipeFallDetector",
    "AlphaPoseFallDetector",
    "STGCNFallDetector",
    "PoseC3DFallDetector",
    "MoveNetFallDetector",
    "OpticalFlowFallDetector",
    # Violence / altercation detection (7)
    "X3DViolenceClassifier",
    "SlowFastViolenceClassifier",
    "VideoMAEViolenceClassifier",
    "I3DViolenceClassifier",
    "C3DViolenceClassifier",
    "TSMViolenceClassifier",
    "MMActionSlowOnlyClassifier",
    # Traffic detection/counting (4)
    "YoloTrafficDetector",
    "RtdetrTrafficDetector",
    "RoboflowTrafficDetector",
    "Mog2ParkedDetector",
]