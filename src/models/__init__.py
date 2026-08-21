from models.base import BaseModelWrapper, Detection
from models.crush.optical_flow_crush import OpticalFlowCrushDetector
from models.crowd_flow import DenseFlowAnalyser

from models.fall import (
    MediaPipeFallDetector,
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
    RoboflowCombinedDetector,
)
from models.traffic import (
    RTDetrV2TrafficDetector,
    RoboflowTrafficDetector,
    Mog2ParkedDetector,
)
from models.anpr import (
    ANPRDetector,
    IndianANPRDetector,
    RapidOCRDetector,
    RTDetrV2ANPRDetector,
)
from models.umbrella import (
    UmbrellaSSDDetector,
    RFDETRNanoUmbrellaDetector,
    RTDetrV2UmbrellaDetector,
    TrainedUmbrellaDetector,
)

__all__ = [
    "BaseModelWrapper",
    "Detection",
    "OpticalFlowCrushDetector",
    "RoboflowCombinedDetector",
    # Dense optical flow crowd-safety
    "DenseFlowAnalyser",
    # Fall detection (3)
    "MediaPipeFallDetector",
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
    # Traffic detection/counting (3)
    "RTDetrV2TrafficDetector",
    "RoboflowTrafficDetector",
    "Mog2ParkedDetector",
    # ANPR (4)
    "ANPRDetector",
    "IndianANPRDetector",
    "RapidOCRDetector",
    "RTDetrV2ANPRDetector",
    # Umbrella detection (4)
    "UmbrellaSSDDetector",
    "RFDETRNanoUmbrellaDetector",
    "RTDetrV2UmbrellaDetector",
    "TrainedUmbrellaDetector",
]
