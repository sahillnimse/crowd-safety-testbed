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


def _zero_shot():
    def check():
        return True, "ready", (
            "Kinetics-pretrained, scored zero-shot over the 9 fighting classes, "
            "cropped to the people in frame. Supply a fine-tuned checkpoint for "
            "a true binary head."
        )
    return check


def _needs_api_key():
    def check():
        # Always "ready" since roboflow_combined.py has a hardcoded fallback
        # key (env var refresh proved unreliable on this machine's terminal).
        return True, "ready", "Roboflow-hosted model, trained on real violence/fall labels."
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
              check=_needs_api_key(), tags=["hosted", "cloud"]),
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
              check=_zero_shot(), tags=["clip", "gpu"]),
    ModelSpec("violence_tsm", "TSM (ResNet-50)", "violence",
              "Temporal Shift Module: 3D-like reasoning at 2D cost.",
              check=_zero_shot(), tags=["clip", "gpu"]),
    ModelSpec("violence_mmaction_slowonly", "MMAction2 SlowOnly", "violence",
              "Framework baseline via MMAction2 config + checkpoint.",
              check=_zero_shot(), tags=["clip", "gpu"]),

    # ---------------- Traffic / vehicle counting ----------------
    ModelSpec("yolo_traffic", "YOLOv11 Traffic", "traffic",
              "COCO-pretrained vehicle detector + ByteTrack; classifies each "
              "tracked vehicle as moving or parked from centroid drift.",
              tags=["frame", "gpu"]),
    ModelSpec("rtdetr_traffic", "RT-DETR Traffic", "traffic",
              "Transformer detector, stronger on small/occluded vehicles. "
              "Falls back to DeepSORT if ByteTrack IDs don't come through.",
              tags=["frame", "gpu"]),
    ModelSpec("roboflow_traffic", "Roboflow (traffic)", "traffic",
              "Hosted model trained on real traffic-camera footage rather "
              "than general COCO images. Runs on Roboflow's servers.",
              check=_needs_api_key(), tags=["hosted", "cloud"]),
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
              check=_needs_api_key(),
              default_stride=2, tags=["frame", "cloud", "ocr"]),

    # ---------------- Other ----------------
    ModelSpec("optical_flow_crush", "Optical Flow (crowd crush)", "other",
              "Circular-variance turbulence + convergence. Classical CV.",
              tags=["flow", "cpu"]),
    ModelSpec("fire_smoke_yolo", "Fire / Smoke YOLO", "other",
              "YOLO fire and smoke detector.",
              check=lambda: (True, "ready", "Runs local fire/smoke model with Roboflow cloud fallback."),
              tags=["frame", "gpu"]),
]

BY_KEY = {m.key: m for m in MODELS}

CATEGORY_LABELS = {
    "fall": "Fall detection",
    "violence": "Violence / altercation",
    "traffic": "Traffic / vehicle counting",
    "anpr": "ANPR / number plates",
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
        "roboflow_combined": lambda: RoboflowCombinedDetector(device=device),
        # video_name keys the gallery directory so runs on different clips
        # don't overwrite each other's captured vehicles.
        "anpr": lambda: _build_anpr(device, video_name),
        "indian_anpr": lambda: _build_indian_anpr(device, video_name),
        "yolo_traffic": lambda: YoloTrafficDetector(device=device),
        "rtdetr_traffic": lambda: RtdetrTrafficDetector(device=device),
        "roboflow_traffic": lambda: RoboflowTrafficDetector(device=device),
        "mog2_parked": lambda: Mog2ParkedDetector(device=device),
    }
    if key not in factories:
        raise KeyError(f"Unknown model: {key}")
    return factories[key]()


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