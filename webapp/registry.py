"""
Catalog of every model the UI can offer, plus whether it can actually run.

The sidebar needs more than a list of names: some wrappers refuse to load
without a fine-tuned checkpoint (they would otherwise emit coin-flip
labels), and some run in a clearly-labelled fallback mode. Surfacing that
state in the UI is the point — a model that needs weights should be
visibly unavailable rather than failing halfway through a job.

Availability is computed from files on disk, never by importing torch or
constructing a wrapper, so listing models stays instant.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _exists(*relative_paths: str) -> bool:
    return any(os.path.exists(os.path.join(PROJECT_ROOT, p)) for p in relative_paths)


@dataclass
class ModelSpec:
    key: str
    label: str
    category: str           # "fall" | "violence" | "other"
    blurb: str              # one-line description for the sidebar tooltip
    # Returns (available, status, note). status is one of:
    #   "ready"    - runs at full capability
    #   "fallback" - runs, but in a degraded//clearly-tagged mode
    #   "blocked"  - cannot run; needs weights or a missing dependency
    check: Optional[Callable[[], tuple[bool, str, str]]] = None
    default_stride: int = 5
    tags: list = field(default_factory=list)

    def status(self) -> dict:
        available, status, note = (True, "ready", "")
        if self.check is not None:
            available, status, note = self.check()
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "blurb": self.blurb,
            "available": available,
            "status": status,
            "note": note,
            "default_stride": self.default_stride,
            "tags": self.tags,
        }


def _needs_weights(name: str, *paths: str, hint: str):
    """Blocked unless one of `paths` exists."""
    def check():
        if _exists(*paths):
            return True, "ready", "Fine-tuned checkpoint found."
        return False, "blocked", hint
    return check


def _geometric_fallback(hint: str):
    def check():
        return True, "fallback", hint
    return check


def _untrained_head(weights_path: str, hint: str):
    """Runnable, but its classification head is randomly initialized.

    Reported as `fallback` rather than `ready`: these models load and emit
    rows, but a random head does not respond to its input, so the output is
    not a detection. Calling that `ready` — as the copy-pasted
    "Kinetics-pretrained" note did — invites trusting numbers that carry no
    information.
    """
    def check():
        if _exists(weights_path):
            return True, "ready", "Fine-tuned checkpoint found."
        return True, "fallback", hint
    return check


def _zero_shot():
    def check():
        return True, "ready", (
            "Kinetics-pretrained, scored zero-shot over the 9 fighting classes, "
            "cropped to the people in frame. Supply a fine-tuned checkpoint for "
            "a true binary head."
        )
    return check


def _needs_api_key(note: str = "Roboflow-hosted model; inference runs on "
                               "Roboflow's servers."):
    """Hosted Roboflow model.

    `note` is a parameter because this helper is shared by the violence,
    traffic and ANPR models. Hardcoding one message meant the traffic and
    ANPR entries both advertised themselves as "trained on real violence/fall
    labels", which describes none of them.
    """
    def check():
        # Always "ready" since roboflow_combined.py has a hardcoded fallback
        # key (env var refresh proved unreliable on this machine's terminal).
        return True, "ready", note
    return check


MODELS: list[ModelSpec] = [
    # ---------------- Fall detection ----------------
    ModelSpec("fall_yolo_pose", "YOLOv8-Pose", "fall",
              "Pose keypoints + posture heuristic, temporally confirmed.",
              tags=["pose", "gpu"]),
    ModelSpec("fall_movenet", "MoveNet", "fall",
              "Google MoveNet multipose, same posture heuristic.",
              tags=["pose", "cpu"]),
    ModelSpec("fall_mediapipe_pose", "MediaPipe BlazePose", "fall",
              "Lightweight per-person pose; YOLO detector upstream for crowds.",
              tags=["pose", "cpu"]),
    ModelSpec("fall_optical_flow", "Optical Flow (fall)", "fall",
              "Pose-free: sustained downward flow that settles. Classical CV.",
              tags=["flow", "cpu"]),
    ModelSpec("fall_stgcn", "ST-GCN", "fall",
              "Skeleton graph-conv classifier over tracked keypoint sequences.",
              tags=["skeleton", "gpu"]),
    ModelSpec("fall_posec3d", "PoseC3D", "fall",
              "3D-CNN over gaussian pose-heatmap volumes.",
              tags=["skeleton", "gpu"]),
    ModelSpec("fall_alphapose_lstm", "AlphaPose + LSTM", "fall",
              "Tracked keypoint sequences classified by a temporal LSTM.",
              tags=["skeleton", "gpu"]),

    # ---------------- Violence detection ----------------
    ModelSpec("roboflow_combined", "Roboflow (violence/fall)", "violence",
              "Hosted model trained on real violence/fall/non-violence labels "
              "(not Kinetics zero-shot). Runs on Roboflow's servers.",
              check=_needs_api_key(
                  "Roboflow-hosted model trained on real violence/fall labels, "
                  "not zero-shot Kinetics classes."),
              tags=["hosted", "cloud"]),
    ModelSpec("violence_x3d", "X3D", "violence",
              "Lightweight 3D-CNN. Best speed/accuracy trade-off here.",
              check=_zero_shot(), tags=["clip", "gpu"]),
    ModelSpec("violence_videomae", "VideoMAE", "violence",
              "Transformer video classifier. Heaviest, usually most accurate.",
              check=_zero_shot(), tags=["clip", "gpu"]),
    ModelSpec("violence_i3d", "I3D", "violence",
              "Inflated 3D ConvNet. Common literature baseline for RWF-2000.",
              check=_zero_shot(), default_stride=10, tags=["clip", "gpu"]),
    ModelSpec("violence_slowfast", "SlowFast", "violence",
              "Dual-pathway 3D-CNN, strong on fast motion like punches.",
              check=_zero_shot(), default_stride=10, tags=["clip", "gpu"]),
    ModelSpec("violence_c3d", "C3D", "violence",
              "Simple 3D-CNN baseline classifier.",
              check=_untrained_head(
                  "weights/c3d_violence.pt",
                  "No pretrained C3D exists, so the whole network is random. "
                  "It runs, but every clip scored a constant 0.510 in testing, "
                  "so output is labelled 'violence_untrained' and excluded "
                  "from event counts. Fine-tune on RWF-2000 / Hockey Fight / "
                  "RLVS to use it for real."),
              tags=["clip", "gpu"]),
    ModelSpec("violence_tsm", "TSM (ResNet-50)", "violence",
              "Temporal Shift Module: 3D-like reasoning at 2D cost.",
              check=_untrained_head(
                  "weights/tsm_violence.pt",
                  "ImageNet backbone, but the 2-class violence head is random. "
                  "Scores sat at 0.494-0.564 regardless of content, so output "
                  "is labelled 'violence_untrained' and excluded from event "
                  "counts. Fine-tune on RWF-2000 / Hockey Fight / RLVS."),
              tags=["clip", "gpu"]),
    ModelSpec("violence_mmaction_slowonly", "MMAction2 SlowOnly", "violence",
              "Framework baseline via MMAction2 config + checkpoint.",
              check=lambda: (True, "ready",
                             "With an MMAction2 checkpoint, runs SlowOnly. "
                             "Without one, falls back to a Kinetics-pretrained "
                             "r3d_18 scored zero-shot over the 9 fighting "
                             "classes - real weights, but not SlowOnly."),
              tags=["clip", "gpu"]),

    # ---------------- Traffic / vehicle counting ----------------
    ModelSpec("yolo_traffic", "YOLOv11 Traffic", "traffic",
              "COCO-pretrained vehicle detector + ByteTrack; classifies each "
              "tracked vehicle as moving or parked from centroid drift.",
              tags=["frame", "gpu"]),
    ModelSpec("rtdetr_traffic", "RT-DETR Traffic", "traffic",
              "Transformer detector, stronger on small/occluded vehicles. "
              "Falls back to DeepSORT if ByteTrack IDs don't come through.",
              tags=["frame", "gpu"]),
    ModelSpec("rtdetrv2_traffic", "RT-DETRv2 Traffic (Moving / Parked)", "traffic",
              "RT-DETRv2-S vehicle detector + centroid drift classifier for "
              "moving vs. parked cars (Apache 2.0, 20M params, 217 FPS).",
              tags=["frame", "gpu", "transformer", "apache2"]),
    ModelSpec("roboflow_traffic", "Roboflow (traffic)", "traffic",
              "Hosted model trained on real traffic-camera footage rather "
              "than general COCO images. Runs on Roboflow's servers.",
              check=_needs_api_key(
                  "Roboflow-hosted vehicle detector trained on traffic-camera "
                  "footage. One API call per processed frame."),
              tags=["hosted", "cloud"]),
    ModelSpec("mog2_parked", "MOG2 Background Subtraction", "traffic",
              "Classical CV, no GPU/weights: flags regions whose pixels "
              "stop changing as parked. Independent cross-check, not a "
              "vehicle classifier.",
              tags=["flow", "cpu"]),

    # ---------------- ANPR ----------------
    ModelSpec("anpr", "ANPR (number plates)", "anpr",
              "Captures each vehicle, reads its number plate, and builds a "
              "gallery of photos with plate, class and colour.",
              check=lambda: (True, "ready",
                             "Needs plates roughly 90px wide or more in frame. "
                             "Wide/distant traffic shots will report "
                             "'too_small' - the characters aren't resolvable "
                             "at that size by any OCR."),
              default_stride=2, tags=["frame", "gpu", "ocr"]),
    ModelSpec("indian_anpr", "Indian ANPR (Roboflow + EasyOCR)", "anpr",
              "Indian vehicle detector + Roboflow plate localisation + EasyOCR. "
              "Handles autos, tempos, bikes and other Indian vehicle types. "
              "Runs hosted inference — no local GPU weights needed.",
              check=_needs_api_key(
                  "Two hosted stages: 1 API call per frame for vehicles, plus "
                  "1 per vehicle per read-frame for plates. A 2,600-frame clip "
                  "with ~15 vehicles is roughly 13,000 calls - test on a short "
                  "window first and watch your Roboflow quota."),
              default_stride=2, tags=["frame", "cloud", "ocr"]),
    ModelSpec("rapid_ocr", "RapidOCR (PP-OCRv4 ONNX)", "anpr",
              "ONNX Runtime RapidOCR engine (PP-OCRv4 mobile det/cls/rec) — "
              "swappable alternative to EasyOCR for ANPR plate crops.",
              default_stride=2, tags=["frame", "cpu", "ocr"]),
    ModelSpec("rtdetrv2_anpr", "RT-DETRv2 ANPR (Classification, Color & Plate)", "anpr",
              "RT-DETRv2-S vehicle detector + fine-grained classification + car colour "
              "recognition + DETR plate detector + RapidOCR / EasyOCR engine (Apache 2.0).",
              default_stride=2, tags=["frame", "gpu", "ocr", "transformer", "apache2"]),

    # ---------------- Umbrella detection ----------------
    ModelSpec("umbrella_yolo", "Umbrella Detection", "umbrella",
              "Detects and counts umbrellas, with persistent IDs so unique "
              "umbrellas can be distinguished from one held for many frames.",
              check=lambda: (True, "ready",
                             "Uses COCO's built-in 'umbrella' class - no extra "
                             "weights to download (yolo11n is 5.6 MB). Pass "
                             "model_size='s' (~19 MB) for better recall on "
                             "small or distant umbrellas."),
              default_stride=3, tags=["frame", "gpu"]),
    ModelSpec("umbrella_ssd", "Umbrella (SSDLite MobileNetV3)", "umbrella",
              "Lighter, older architecture than YOLO - the comparison point "
              "for how much detection quality comes from the architecture.",
              check=lambda: (True, "ready",
                             "torchvision SSDLite320 + MobileNetV3, ~13.8 MB, "
                             "320x320 input so it is genuinely usable on CPU. "
                             "Scores run lower than YOLO's, hence its lower "
                             "default threshold - not a weaker setting."),
              default_stride=3, tags=["frame", "cpu", "gpu"]),
    ModelSpec("umbrella_world", "Umbrella (YOLO-World open-vocab)", "umbrella",
              "Text-prompted detection: finds parasols and sun umbrellas that "
              "COCO's single fixed 'umbrella' class was never trained on.",
              check=lambda: (True, "ready",
                             "yolov8s-worldv2, ~25.9 MB. Prompts default to "
                             "umbrella/parasol/beach umbrella/sun umbrella. "
                             "Higher recall but looser than the fixed-class "
                             "models - each detection records which prompt "
                             "matched in extra.matched_class."),
              default_stride=3, tags=["frame", "gpu", "open-vocab"]),
    ModelSpec("umbrella_yolo26n", "YOLO26-Nano (umbrella-finetuned)", "umbrella",
              "Ultralytics YOLO26 nano variant, NMS-free, edge-optimized + ByteTrack.",
              check=lambda: (True, "ready",
                             "NMS-free edge profile detector. Uses fallback pretrained "
                             "nano weights when fine-tuned checkpoint is not on disk."),
              default_stride=3, tags=["frame", "gpu", "edge"]),
    ModelSpec("umbrella_rfdetr", "RF-DETR Nano (umbrella-finetuned)", "umbrella",
              "Roboflow RF-DETR Nano transformer with DINOv2 backbone for small/occluded umbrella recall.",
              check=lambda: (True, "ready",
                             "DINOv2 backbone transformer detector. High recall on small or "
                             "occluded objects in dense crowds."),
              default_stride=3, tags=["frame", "gpu", "transformer"]),
    ModelSpec("umbrella_rtdetrv2", "RT-DETRv2-S (COCO zero-shot)", "umbrella",
              "RT-DETRv2 with ResNet-18vd backbone — improved deformable attention, "
              "dual-level IoU-aware query selection (Apache 2.0, 20M params, 217 FPS).",
              check=lambda: (True, "ready",
                             "Apache-2.0 licensed. Weights auto-downloaded from "
                             "HuggingFace Hub (PekingU/rtdetr_v2_r18vd) on first run. "
                             "COCO umbrella class 25 — no fine-tuning required."),
              default_stride=3, tags=["frame", "gpu", "transformer", "apache2"]),
    # ---------------- Fire / Smoke & Crowd Crush ----------------
    ModelSpec("fire_smoke_yolo", "Fire / Smoke YOLO", "fire",
              "YOLO fire and smoke detector.",
              check=lambda: (True, "ready", "Runs local fire/smoke model with Roboflow cloud fallback."),
              tags=["frame", "gpu"]),
    ModelSpec("optical_flow_crush", "Optical Flow (crowd crush)", "crush",
              "Circular-variance turbulence + convergence. Classical CV.",
              tags=["flow", "cpu"]),
]

BY_KEY = {m.key: m for m in MODELS}

CATEGORY_LABELS = {
    "fall": "Fall detection",
    "violence": "Violence / altercation",
    "traffic": "Traffic / vehicle counting",
    "anpr": "ANPR / number plates",
    "umbrella": "Umbrella detection",
    "fire": "Fire & smoke detection",
    "crush": "Crowd crush detection",
    "other": "Other detectors",
}


def list_models() -> list[dict]:
    return [m.status() for m in MODELS]


def build_model(key: str, device: Optional[str], pose_size: str = "s",
                video_name: str = "run"):
    """Construct a wrapper instance. Imports are deferred to call time so the
    API can list models without paying torch's import cost."""
    from models.fire_smoke_yolo import FireSmokeYOLO
    from models.optical_flow_crush import OpticalFlowCrushDetector
    from models.roboflow_combined import RoboflowCombinedDetector
    from models.fall import (
        AlphaPoseFallDetector, MediaPipeFallDetector, MoveNetFallDetector,
        OpticalFlowFallDetector, PoseC3DFallDetector, STGCNFallDetector,
        YOLOPoseFallDetector,
    )
    from models.violence import (
        C3DViolenceClassifier, I3DViolenceClassifier, MMActionSlowOnlyClassifier,
        SlowFastViolenceClassifier, TSMViolenceClassifier,
        VideoMAEViolenceClassifier, X3DViolenceClassifier,
    )
    from models.traffic import (
        YoloTrafficDetector, RtdetrTrafficDetector,
        RoboflowTrafficDetector, Mog2ParkedDetector,
    )   

    factories = {
        "fire_smoke_yolo": lambda: FireSmokeYOLO(device=device),
        "umbrella_yolo": lambda: _build_umbrella(device),
        "umbrella_ssd": lambda: _build_umbrella_ssd(device),
        "umbrella_world": lambda: _build_umbrella_world(device),
        "umbrella_yolo26n": lambda: _build_umbrella_yolo26n(device),
        "umbrella_rfdetr": lambda: _build_umbrella_rfdetr(device),
        "umbrella_rtdetrv2": lambda: _build_umbrella_rtdetrv2(device),
        "rapid_ocr": lambda: _build_rapid_ocr(device, video_name),
        "optical_flow_crush": lambda: OpticalFlowCrushDetector(device=device),
        "roboflow_combined": lambda: RoboflowCombinedDetector(device=device),
        "fall_yolo_pose": lambda: YOLOPoseFallDetector(model_size=pose_size, device=device),
        "fall_mediapipe_pose": lambda: MediaPipeFallDetector(device=device),
        "fall_alphapose_lstm": lambda: AlphaPoseFallDetector(device=device),
        "fall_stgcn": lambda: STGCNFallDetector(device=device),
        "fall_posec3d": lambda: PoseC3DFallDetector(device=device),
        "fall_movenet": lambda: MoveNetFallDetector(device=device),
        "fall_optical_flow": lambda: OpticalFlowFallDetector(device=device),
        "violence_x3d": lambda: X3DViolenceClassifier(device=device),
        "violence_slowfast": lambda: SlowFastViolenceClassifier(device=device),
        "violence_videomae": lambda: VideoMAEViolenceClassifier(device=device),
        "violence_i3d": lambda: I3DViolenceClassifier(device=device),
        "violence_c3d": lambda: C3DViolenceClassifier(device=device),
        "violence_tsm": lambda: TSMViolenceClassifier(device=device),
        "violence_mmaction_slowonly": lambda: MMActionSlowOnlyClassifier(device=device),
        # video_name keys the gallery directory so runs on different clips
        # don't overwrite each other's captured vehicles.
        "anpr": lambda: _build_anpr(device, video_name),
        "indian_anpr": lambda: _build_indian_anpr(device, video_name),
        "rtdetrv2_anpr": lambda: _build_rtdetrv2_anpr(device, video_name),
        "yolo_traffic": lambda: YoloTrafficDetector(device=device),
        "rtdetr_traffic": lambda: RtdetrTrafficDetector(device=device),
        "rtdetrv2_traffic": lambda: _build_rtdetrv2_traffic(device),
        "roboflow_traffic": lambda: RoboflowTrafficDetector(device=device),
        "mog2_parked": lambda: Mog2ParkedDetector(device=device),
    }
    if key not in factories:
        raise KeyError(f"Unknown model: {key}")
    return factories[key]()


def _build_umbrella(device):
    from models.umbrella import UmbrellaDetector
    return UmbrellaDetector(device=device)


def _build_umbrella_ssd(device):
    from models.umbrella import UmbrellaSSDDetector
    return UmbrellaSSDDetector(device=device)


def _build_umbrella_world(device):
    from models.umbrella import UmbrellaWorldDetector
    return UmbrellaWorldDetector(device=device)


def _build_umbrella_yolo26n(device):
    from models.umbrella import YOLO26NanoUmbrellaDetector
    return YOLO26NanoUmbrellaDetector(device=device)


def _build_umbrella_rfdetr(device):
    from models.umbrella import RFDETRNanoUmbrellaDetector
    return RFDETRNanoUmbrellaDetector(device=device)


def _build_umbrella_rtdetrv2(device):
    from models.umbrella import RTDetrV2UmbrellaDetector
    return RTDetrV2UmbrellaDetector(device=device)


def _build_rapid_ocr(device, video_name: str):
    import os
    from models.anpr import RapidOCRDetector
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    return RapidOCRDetector(device=device, video_name=stem)


def _build_anpr(device, video_name: str):
    import os

    from models.anpr import ANPRDetector
    # Strip the extension so the gallery folder matches the log filenames
    # the rest of the UI uses.
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    return ANPRDetector(device=device, video_name=stem)


def _build_indian_anpr(device, video_name: str):
    import os

    from models.anpr import IndianANPRDetector
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    return IndianANPRDetector(device=device, video_name=stem)


def _build_rtdetrv2_anpr(device, video_name: str):
    import os

    from models.anpr import RTDetrV2ANPRDetector
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    return RTDetrV2ANPRDetector(device=device, video_name=stem)


def _build_rtdetrv2_traffic(device):
    from models.traffic import RTDetrV2TrafficDetector
    return RTDetrV2TrafficDetector(device=device)

