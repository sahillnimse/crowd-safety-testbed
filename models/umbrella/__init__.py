"""Umbrella detection — swappable model backends.

    umbrella_ssd         SSDLite320 MobileNetV3, lightweight CPU baseline
    umbrella_rfdetr      RF-DETR Nano, DINOv2 backbone for small/occluded objects
    umbrella_rtdetrv2    RT-DETRv2-S, ResNet-18vd backbone, Apache 2.0, COCO zero-shot
    umbrella_trained     RT-DETRv2 fine-tuned on umbrellas - the only trained one

The YOLO backends (umbrella_yolo, umbrella_yolo26n, umbrella_world) were
removed with the rest of the YOLO family.  RT-DETRv2 covers the same
zero-shot COCO ground under Apache 2.0, and umbrella_trained remains the
accuracy option.
"""

from models.umbrella._common import UMBRELLA_COCO_CLASS, emit_umbrellas
from models.umbrella.umbrella_rfdetr import RFDETRNanoUmbrellaDetector
from models.umbrella.umbrella_rtdetrv2 import RTDetrV2UmbrellaDetector
from models.umbrella.umbrella_ssd import UmbrellaSSDDetector
from models.umbrella.umbrella_trained import TrainedUmbrellaDetector, find_trained_dir


__all__ = [
    "UmbrellaSSDDetector",
    "RFDETRNanoUmbrellaDetector",
    "RTDetrV2UmbrellaDetector",
    "TrainedUmbrellaDetector",
    "find_trained_dir",
    "UMBRELLA_COCO_CLASS",
    "emit_umbrellas",
]
