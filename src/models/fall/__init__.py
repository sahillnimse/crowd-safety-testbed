"""
Fall detection model family.

Pose comes from MediaPipe BlazePose or MoveNet; person boxes come from the
shared RT-DETRv2 detector in models/_detectors.py.

The skeleton-graph models (ST-GCN, PoseC3D, AlphaPose+LSTM) were removed
along with YOLO: each depended on YOLO-pose for its keypoints, and RT-DETRv2
is a detection model that returns boxes, not skeletons.  There was no
like-for-like replacement for them, and keeping them on a YOLO backbone would
have defeated the point of the removal.
"""

from models.fall.mediapipe_pose import MediaPipeFallDetector
from models.fall.movenet import MoveNetFallDetector
from models.fall.optical_flow_fall import OpticalFlowFallDetector

__all__ = [
    "MediaPipeFallDetector",
    "MoveNetFallDetector",
    "OpticalFlowFallDetector",
]
