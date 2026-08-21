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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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
    default_threshold: float | None = None
    # False for a model whose confidence scores are not on the same scale as
    # the rest, so the run-wide threshold must not be applied to it.  SSDLite
    # scores run materially lower than the transformer detectors', and giving
    # it the same numeric threshold is not a fair comparison but a handicap.
    # Such a model keeps default_threshold and is documented as doing so.
    comparable_threshold: bool = True
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
            "default_threshold": self.default_threshold,
            "comparable_threshold": self.comparable_threshold,
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


def _real_arch_check(finetuned_path: str, finetuned_note: str, stock_note: str):
    """Ready either way: fine-tuned if present, else the real stock weights.

    Distinct from `_untrained_head`, which marks a model `fallback` because
    it is not the architecture it claims. These two now load their actual
    networks in both cases, so `ready` is accurate — the note just says
    which weights are in play.
    """
    def check():
        if _exists(finetuned_path):
            return True, "ready", finetuned_note
        return True, "ready", stock_note
    return check


def _trained_umbrella_check():
    """Available only when the fine-tuned checkpoint is actually on disk."""
    def check():
        from models.umbrella.umbrella_trained import find_trained_dir
        d = find_trained_dir()
        if d:
            return True, "ready", (
                "Fine-tuned RT-DETRv2, val F1 0.711 (precision 0.769 / recall "
                "0.661). NMS is applied - the raw checkpoint emits ~2.7x "
                "duplicate boxes despite RT-DETR being nominally NMS-free.")
        return False, "blocked", (
            "No fine-tuned checkpoint. Unzip umbrella_v1_best.zip into "
            "'ML Models/umbrella_trained/' (needs model.safetensors + "
            "config.json + preprocessor_config.json).")
    return check


MODELS: list[ModelSpec] = [
    # ---------------- Fall detection ----------------
    ModelSpec("fall_movenet", "MoveNet", "fall",
              "Google MoveNet multipose, posture heuristic.",
              default_threshold=0.4, tags=["pose", "cpu"]),
    ModelSpec("fall_mediapipe_pose", "MediaPipe BlazePose", "fall",
              "Per-person pose; RT-DETRv2 person detector upstream for crowds.",
              default_threshold=0.4, tags=["pose", "cpu"]),
    ModelSpec("fall_optical_flow", "Optical Flow (fall)", "fall",
              "Pose-free: sustained downward flow that settles. Classical CV.",
              tags=["flow", "cpu"]),

    # ---------------- Violence detection ----------------
    ModelSpec("roboflow_combined", "Roboflow (violence/fall)", "violence",
              "Hosted model trained on real violence/fall/non-violence labels "
              "(not Kinetics zero-shot). Runs on Roboflow's servers.",
              check=_needs_api_key(
                  "Roboflow-hosted model trained on real violence/fall labels, "
                  "not zero-shot Kinetics classes."),
              default_threshold=0.5, tags=["hosted", "cloud"]),
    ModelSpec("violence_x3d", "X3D", "violence",
              "Lightweight 3D-CNN. Best speed/accuracy trade-off here.",
              check=_zero_shot(), default_threshold=0.5, tags=["clip", "gpu"]),
    ModelSpec("violence_videomae", "VideoMAE", "violence",
              "Transformer video classifier. Heaviest, usually most accurate.",
              check=_zero_shot(), default_threshold=0.5, tags=["clip", "gpu"]),
    ModelSpec("violence_i3d", "I3D", "violence",
              "Inflated 3D ConvNet. Common literature baseline for RWF-2000.",
              check=_zero_shot(), default_stride=10, default_threshold=0.5, tags=["clip", "gpu"]),
    ModelSpec("violence_slowfast", "SlowFast", "violence",
              "Dual-pathway 3D-CNN, strong on fast motion like punches.",
              check=_zero_shot(), default_stride=10, default_threshold=0.5, tags=["clip", "gpu"]),
    ModelSpec("violence_c3d", "C3D", "violence",
              "Simple 3D-CNN baseline classifier.",
              check=_untrained_head(
                  "weights/c3d_violence.pt",
                  "No pretrained C3D exists, so the whole network is random. "
                  "It runs, but every clip scored a constant 0.510 in testing, "
                  "so output is labelled 'violence_untrained' and excluded "
                  "from event counts. Fine-tune on RWF-2000 / Hockey Fight / "
                  "RLVS to use it for real."),
              default_threshold=0.5, tags=["clip", "gpu"]),
    ModelSpec("violence_tsm", "TSM (ResNet-50)", "violence",
              "Temporal Shift Module: 3D-like reasoning at 2D cost.",
              check=_untrained_head(
                  "weights/tsm_violence.pt",
                  "ImageNet backbone, but the 2-class violence head is random. "
                  "Scores sat at 0.494-0.564 regardless of content, so output "
                  "is labelled 'violence_untrained' and excluded from event "
                  "counts. Fine-tune on RWF-2000 / Hockey Fight / RLVS."),
              default_threshold=0.5, tags=["clip", "gpu"]),
    ModelSpec("violence_mmaction_slowonly", "MMAction2 SlowOnly", "violence",
              "Framework baseline via MMAction2 config + checkpoint.",
              check=lambda: (True, "ready",
                             "With an MMAction2 checkpoint, runs SlowOnly. "
                             "Without one, falls back to a Kinetics-pretrained "
                             "r3d_18 scored zero-shot over the 9 fighting "
                             "classes - real weights, but not SlowOnly."),
              default_threshold=0.5, tags=["clip", "gpu"]),

    # ---------------- Traffic / vehicle counting ----------------
    ModelSpec("rtdetrv2_traffic", "RT-DETRv2 Traffic (Moving / Parked)", "traffic",
              "RT-DETRv2-S vehicle detector + centroid drift classifier for "
              "moving vs. parked cars (Apache 2.0, 20M params, 217 FPS).",
              default_threshold=0.35, tags=["frame", "gpu", "transformer", "apache2"]),
    ModelSpec("roboflow_traffic", "Roboflow (traffic)", "traffic",
              "Hosted model trained on real traffic-camera footage rather "
              "than general COCO images. Runs on Roboflow's servers.",
              check=_needs_api_key(
                  "Roboflow-hosted vehicle detector trained on traffic-camera "
                  "footage. One API call per processed frame."),
              default_threshold=0.35, tags=["hosted", "cloud"]),
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
              default_stride=2, default_threshold=0.35, tags=["frame", "gpu", "ocr"]),
    ModelSpec("indian_anpr", "Indian ANPR (Roboflow + EasyOCR)", "anpr",
              "Indian vehicle detector + Roboflow plate localisation + EasyOCR. "
              "Handles autos, tempos, bikes and other Indian vehicle types. "
              "Runs hosted inference — no local GPU weights needed.",
              check=_needs_api_key(
                  "Two hosted stages: 1 API call per frame for vehicles, plus "
                  "1 per vehicle per read-frame for plates. A 2,600-frame clip "
                  "with ~15 vehicles is roughly 13,000 calls - test on a short "
                  "window first and watch your Roboflow quota."),
              default_stride=2, default_threshold=0.35, tags=["frame", "cloud", "ocr"]),
    ModelSpec("rapid_ocr", "RapidOCR (PP-OCRv4 ONNX)", "anpr",
              "ONNX Runtime RapidOCR engine (PP-OCRv4 mobile det/cls/rec) — "
              "swappable alternative to EasyOCR for ANPR plate crops.",
              default_stride=2, default_threshold=0.35, tags=["frame", "cpu", "ocr"]),
    ModelSpec("rtdetrv2_anpr", "RT-DETRv2 ANPR (Classification, Color & Plate)", "anpr",
              "RT-DETRv2-S vehicle detector + fine-grained classification + car colour "
              "recognition + DETR plate detector + RapidOCR / EasyOCR engine (Apache 2.0).",
              default_stride=2, default_threshold=0.35, tags=["frame", "gpu", "ocr", "transformer", "apache2"]),

    # ---------------- Umbrella detection ----------------
    ModelSpec("umbrella_ssd", "Umbrella (SSDLite MobileNetV3)", "umbrella",
              "Lighter, older architecture than the transformers - the "
              "comparison point for how much detection quality comes from the "
              "architecture.",
              check=lambda: (True, "ready",
                             "torchvision SSDLite320 + MobileNetV3, ~13.8 MB, "
                             "320x320 input so it is genuinely usable on CPU. "
                             "Scores run lower than the transformer models', "
                             "hence its lower default threshold - not a weaker "
                             "setting.  The run-wide threshold is NOT applied "
                             "to this model for that reason; it always runs at "
                             "0.25."),
              default_stride=3, default_threshold=0.25,
              comparable_threshold=False, tags=["frame", "cpu", "gpu"]),
    ModelSpec("umbrella_rfdetr", "RF-DETR Nano (umbrella-finetuned)", "umbrella",
              "Roboflow RF-DETR Nano transformer with DINOv2 backbone for small/occluded umbrella recall.",
              check=_real_arch_check(
                  "weights/rfdetr_nano_umbrella.pt",
                  "Fine-tuned RF-DETR umbrella checkpoint found.",
                  "Real RF-DETR Nano COCO weights via the rfdetr package "
                  "(DINOv2 backbone), detecting COCO's umbrella class. Drop "
                  "weights/rfdetr_nano_umbrella.pt in to use a fine-tuned "
                  "version instead."),
              default_stride=3, default_threshold=0.35, tags=["frame", "gpu", "transformer"]),
    ModelSpec("umbrella_trained", "RT-DETRv2 (trained)", "umbrella",
              "Fine-tuned on umbrella data - the only umbrella model here not "
              "relying on COCO's generic class. 42.7M params, single class.",
              check=_trained_umbrella_check(),
              default_stride=3, default_threshold=0.35, tags=["frame", "gpu", "finetuned"]),
    ModelSpec("umbrella_rtdetrv2", "RT-DETRv2-S (COCO zero-shot)", "umbrella",
              "RT-DETRv2 with ResNet-18vd backbone — improved deformable attention, "
              "dual-level IoU-aware query selection (Apache 2.0, 20M params, 217 FPS).",
              check=lambda: (True, "ready",
                             "Apache-2.0 licensed. Weights auto-downloaded from "
                             "HuggingFace Hub (PekingU/rtdetr_v2_r18vd) on first run. "
                             "COCO umbrella class 25 — no fine-tuning required."),
              default_stride=3, default_threshold=0.35, tags=["frame", "gpu", "transformer", "apache2"]),
    # ---------------- Fire / Smoke & Crowd Crush ----------------
    ModelSpec("optical_flow_crush", "Optical Flow (crowd crush)", "crush",
              "Circular-variance turbulence + convergence. Classical CV.",
              tags=["flow", "cpu"]),
    ModelSpec("dense_flow", "Dense Flow Analysis (Kumbh Mela)", "crush",
              "DIS dense flow field analysis: divergence compression, counterflow, "
              "stop-and-go waves, and perspective m/s speed.",
              check=lambda: (True, "ready", "DIS optical flow + zone metrics + perspective correction."),
              tags=["flow", "cpu", "kumbh-mela"]),
    ModelSpec("crowd_motion_monitor", "Crowd Motion Monitor", "crush",
              "Per-person velocity + heading triangle overlay. Flags stationary "
              "individuals and local crowd compression (crush risk) in red. "
              "RT-DETRv2 person detector + IoU tracker + Farneback flow.",
              check=lambda: (True, "ready",
                             "RT-DETRv2 person detector + IoU tracker + Farneback "
                             "dense flow.  No fine-tuned weights required; "
                             "weights auto-downloaded from HuggingFace Hub on first run."),
              comparable_threshold=False,
              tags=["flow", "cpu", "tracker"]),
]

BY_KEY = {m.key: m for m in MODELS}

CATEGORY_LABELS = {
    "fall": "Fall detection",
    "violence": "Violence / altercation",
    "traffic": "Traffic / vehicle counting",
    "anpr": "ANPR / number plates",
    "umbrella": "Umbrella detection",
    "crush": "Crowd crush detection",
    "other": "Other detectors",
}


def list_models() -> list[dict]:
    return [m.status() for m in MODELS]


def build_model(key: str, device: Optional[str],
                video_name: str = "run", threshold: Optional[float] = None):
    """Construct a wrapper instance. Imports are deferred to call time so the
    API can list models without paying torch's import cost."""
    from models.optical_flow_crush import OpticalFlowCrushDetector
    from models.roboflow_combined import RoboflowCombinedDetector
    from models.fall import (
        MediaPipeFallDetector, MoveNetFallDetector, OpticalFlowFallDetector,
    )
    from models.violence import (
        C3DViolenceClassifier, I3DViolenceClassifier, MMActionSlowOnlyClassifier,
        SlowFastViolenceClassifier, TSMViolenceClassifier,
        VideoMAEViolenceClassifier, X3DViolenceClassifier,
    )
    from models.traffic import (
        RoboflowTrafficDetector, Mog2ParkedDetector,
    )

    def _kw(extra: dict = None):
        kw = dict(extra) if extra else {}
        if threshold is not None:
            kw["conf_threshold"] = threshold
        return kw

    factories = {
        "umbrella_ssd": lambda: _build_umbrella_ssd(device, threshold=threshold),
        "umbrella_rfdetr": lambda: _build_umbrella_rfdetr(device, threshold=threshold),
        "umbrella_rtdetrv2": lambda: _build_umbrella_rtdetrv2(device, threshold=threshold),
        "umbrella_trained": lambda: _build_umbrella_trained(device, threshold=threshold),
        "rapid_ocr": lambda: _build_rapid_ocr(device, video_name, threshold=threshold),
        "optical_flow_crush": lambda: OpticalFlowCrushDetector(device=device),
        "dense_flow": lambda: _build_dense_flow(device),
        "crowd_motion_monitor": lambda: _build_crowd_motion_monitor(device, video_name=video_name, threshold=threshold),
        "roboflow_combined": lambda: RoboflowCombinedDetector(device=device, **_kw()),
        "fall_mediapipe_pose": lambda: MediaPipeFallDetector(device=device, **_kw()),
        "fall_movenet": lambda: MoveNetFallDetector(device=device, **_kw()),
        "fall_optical_flow": lambda: OpticalFlowFallDetector(device=device),
        "violence_x3d": lambda: X3DViolenceClassifier(device=device, **_kw()),
        "violence_slowfast": lambda: SlowFastViolenceClassifier(device=device, **_kw()),
        "violence_videomae": lambda: VideoMAEViolenceClassifier(device=device, **_kw()),
        "violence_i3d": lambda: I3DViolenceClassifier(device=device, **_kw()),
        "violence_c3d": lambda: C3DViolenceClassifier(device=device, **_kw()),
        "violence_tsm": lambda: TSMViolenceClassifier(device=device, **_kw()),
        "violence_mmaction_slowonly": lambda: MMActionSlowOnlyClassifier(device=device, **_kw()),
        # video_name keys the gallery directory so runs on different clips
        # don't overwrite each other's captured vehicles.
        "anpr": lambda: _build_anpr(device, video_name, threshold=threshold),
        "indian_anpr": lambda: _build_indian_anpr(device, video_name, threshold=threshold),
        "rtdetrv2_anpr": lambda: _build_rtdetrv2_anpr(device, video_name, threshold=threshold),
        "rtdetrv2_traffic": lambda: _build_rtdetrv2_traffic(device, threshold=threshold),
        "roboflow_traffic": lambda: RoboflowTrafficDetector(device=device, **_kw()),
        "mog2_parked": lambda: Mog2ParkedDetector(device=device),
    }
    if key not in factories:
        raise KeyError(f"Unknown model: {key}")
    return factories[key]()


def _build_umbrella_ssd(device, threshold=None):
    from models.umbrella import UmbrellaSSDDetector
    kw = {"device": device}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return UmbrellaSSDDetector(**kw)


def _build_umbrella_trained(device, threshold=None):
    from models.umbrella import TrainedUmbrellaDetector
    kw = {"device": device}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return TrainedUmbrellaDetector(**kw)


def _build_umbrella_rfdetr(device, threshold=None):
    from models.umbrella import RFDETRNanoUmbrellaDetector
    kw = {"device": device}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return RFDETRNanoUmbrellaDetector(**kw)


def _build_umbrella_rtdetrv2(device, threshold=None):
    from models.umbrella import RTDetrV2UmbrellaDetector
    kw = {"device": device}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return RTDetrV2UmbrellaDetector(**kw)


def _build_rapid_ocr(device, video_name: str, threshold=None):
    import os
    from models.anpr import RapidOCRDetector
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    kw = {"device": device, "video_name": stem}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return RapidOCRDetector(**kw)


def _build_anpr(device, video_name: str, threshold=None):
    import os

    from models.anpr import ANPRDetector
    # Strip the extension so the gallery folder matches the log filenames
    # the rest of the UI uses.
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    kw = {"device": device, "video_name": stem}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return ANPRDetector(**kw)


def _build_indian_anpr(device, video_name: str, threshold=None):
    import os

    from models.anpr import IndianANPRDetector
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    kw = {"device": device, "video_name": stem}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return IndianANPRDetector(**kw)


def _build_rtdetrv2_anpr(device, video_name: str, threshold=None):
    import os

    from models.anpr import RTDetrV2ANPRDetector
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    kw = {"device": device, "video_name": stem}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return RTDetrV2ANPRDetector(**kw)


def _build_rtdetrv2_traffic(device, threshold=None):
    from models.traffic import RTDetrV2TrafficDetector
    kw = {"device": device}
    if threshold is not None:
        kw["conf_threshold"] = threshold
    return RTDetrV2TrafficDetector(**kw)


def _build_dense_flow(device):
    import yaml
    from models.crowd_flow import DenseFlowAnalyser
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "crowd_flow.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f).get("crowd_flow", {})
    else:
        cfg = {}
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "annotated")
    os.makedirs(output_dir, exist_ok=True)
    return DenseFlowAnalyser(
        config=cfg,
        # The web UI runs arbitrary uploaded footage, so it uses the
        # resolution-independent "default" camera.  The site-specific camera
        # blocks carry pixel polygons surveyed for one particular view and
        # would measure the wrong region of any other video.
        camera_id="default",
        output_dir=output_dir,
        device=device,
    )


def _build_crowd_motion_monitor(device, video_name: str = "run", threshold=None):
    from models.crowd_flow import CrowdMotionMonitor
    stem = os.path.splitext(os.path.basename(video_name or "run"))[0]
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "annotated")
    os.makedirs(output_dir, exist_ok=True)
    kw = {"device": device, "video_name": stem, "output_dir": output_dir}
    # threshold maps to stationary_speed_px: the px/frame floor below which a
    # person is considered stopped.  comparable_threshold=False in the ModelSpec
    # means the run-wide confidence threshold is NOT applied, but an explicit
    # per-model threshold from the UI is still honoured here.
    if threshold is not None:
        kw["stationary_speed_px"] = float(threshold)
    return CrowdMotionMonitor(**kw)
