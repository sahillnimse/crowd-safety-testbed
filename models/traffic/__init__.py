"""
Traffic detection/counting model family.

Each model here detects vehicles (car, bus, truck, motorcycle) per frame,
tracks them across frames (ID persistence), and classifies each tracked
vehicle as MOVING or PARKED based on centroid displacement over time.

Detection.label will be one of: "vehicle_moving", "vehicle_parked"
Detection.extra will include: {"vehicle_class": "car"/"bus"/"truck"/"motorcycle",
                                "track_id": int}
"""

from models.traffic.yolo_traffic import YoloTrafficDetector
from models.traffic.rtdetr_traffic import RtdetrTrafficDetector
from models.traffic.roboflow_traffic import RoboflowTrafficDetector
from models.traffic.mog2_parked import Mog2ParkedDetector

__all__ = [
    "YoloTrafficDetector",
    "RtdetrTrafficDetector",
    "RoboflowTrafficDetector",
    "Mog2ParkedDetector",
]