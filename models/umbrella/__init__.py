"""Umbrella detection — swappable model backends.

    umbrella_yolo        YOLO11 n/s, COCO fixed vocabulary
    umbrella_ssd         SSDLite320 MobileNetV3, lightweight CPU baseline
    umbrella_world       YOLO-World v2, open-vocabulary prompts
    umbrella_yolo26n     YOLO26-Nano, fine-tuned NMS-free edge model
    umbrella_rfdetr      RF-DETR Nano, DINOv2 backbone for small/occluded objects
    umbrella_rtdetrv2    RT-DETRv2-S, ResNet-18vd backbone, Apache 2.0, COCO zero-shot
    umbrella_trained     RT-DETRv2 fine-tuned on umbrellas - the only trained one
"""

from models.umbrella._common import emit_umbrellas
from models.umbrella.umbrella_rfdetr import RFDETRNanoUmbrellaDetector
from models.umbrella.umbrella_rtdetrv2 import RTDetrV2UmbrellaDetector
from models.umbrella.umbrella_ssd import UmbrellaSSDDetector
from models.umbrella.umbrella_trained import TrainedUmbrellaDetector, find_trained_dir
from models.umbrella.umbrella_world import UmbrellaWorldDetector
from models.umbrella.umbrella_yolo import UMBRELLA_COCO_CLASS, UmbrellaDetector
from models.umbrella.umbrella_yolo26n import YOLO26NanoUmbrellaDetector

__all__ = [
    "UmbrellaDetector",
    "UmbrellaSSDDetector",
    "UmbrellaWorldDetector",
    "YOLO26NanoUmbrellaDetector",
    "RFDETRNanoUmbrellaDetector",
    "RTDetrV2UmbrellaDetector",
    "TrainedUmbrellaDetector",
    "find_trained_dir",
    "UMBRELLA_COCO_CLASS",
    "emit_umbrellas",
]
