"""
collect_ml_models.py
====================
Collects every model weight used by the crowd-safety-testbed into the
'ML Models/' folder so it can be shared as a single self-contained
directory for senior team review.

Run from the project root:
    python scripts/collect_ml_models.py

What it does
------------
1. Copies all local .pt / .task / .onnx weights already on disk.
2. Downloads the three missing MediaPipe .task variants.
3. Downloads PyTorchVideo pretrained weights (X3D-S, SlowFast-R50, I3D-R50)
   via torch.hub and saves them as standalone .pt files.
4. Downloads VideoMAE (HuggingFace) config + pytorch_model.bin.
5. Downloads the HuggingFace DETR plate-detection model.
6. Downloads EasyOCR recognition weights (used by indian_anpr).
7. Writes a MODEL_MANIFEST.md inside 'ML Models/' cataloguing every registered model,
   their weights, download sources, and status.

Models that live entirely on hosted APIs (Roboflow, MoveNet via TF-Hub,
RapidOCR ONNX auto-download) are documented in the manifest but have no
local weight file to copy — the manifest entry explains this clearly.
"""

import os
import sys
import shutil
import urllib.request
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_MODELS   = PROJECT_ROOT / "ML Models"

# Sub-directories inside ML Models
DIRS = {
    "mediapipe":     ML_MODELS / "mediapipe",
    "pytorchvideo":  ML_MODELS / "pytorchvideo",
    "videomae":      ML_MODELS / "videomae",
    "detr_plate":    ML_MODELS / "detr_plate",
    "easyocr":       ML_MODELS / "easyocr",
    "fire_smoke":    ML_MODELS / "fire_smoke",
    "optical_flow":  ML_MODELS / "optical_flow_classical",
    "roboflow_api":  ML_MODELS / "roboflow_api",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
OK    = "[OK]"
SKIP  = "[SKIP]"
ERR   = "[ERR]"
WARN  = "[WARN]"
INFO  = "[INFO]"


def log(sym, msg):
    print(f"  {sym}  {msg}")


def copy_if_exists(src: Path, dst_dir: Path, label: str) -> str:
    """Copy src -> dst_dir/src.name. Returns status string."""
    if not src.exists():
        log(SKIP, f"{label}: not found at {src.relative_to(PROJECT_ROOT)}, skipping copy")
        return "missing_local"
    dst = dst_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        log(OK, f"{label}: already in ML Models ({src.name})")
        return "present"
    shutil.copy2(src, dst)
    mb = dst.stat().st_size / 1e6
    log(OK, f"{label}: copied {src.name}  ({mb:.1f} MB)")
    return "copied"


def download_if_missing(url: str, dst: Path, label: str) -> str:
    """Download url -> dst. Returns status string."""
    if dst.exists() and dst.stat().st_size > 0:
        log(OK, f"{label}: already downloaded ({dst.name})")
        return "present"
    log(INFO, f"{label}: downloading {dst.name} ...")
    try:
        urllib.request.urlretrieve(url, dst)
        mb = dst.stat().st_size / 1e6
        log(OK, f"{label}: downloaded {dst.name}  ({mb:.1f} MB)")
        return "downloaded"
    except Exception as e:
        log(ERR, f"{label}: FAILED - {e}")
        return f"error: {e}"


# ---------------------------------------------------------------------------
# STATUS REGISTRY — filled in as we go, used for the manifest
# ---------------------------------------------------------------------------
STATUS = {}   # model_key -> {"file": str, "status": str, "size_mb": float, "note": str}


def record(key, file_path, status, note=""):
    mb = 0.0
    if file_path and Path(file_path).exists():
        mb = Path(file_path).stat().st_size / 1e6
    STATUS[key] = {"file": str(file_path) if file_path else "N/A",
                   "status": status, "size_mb": round(mb, 1), "note": note}


# ===========================================================================
# SECTION 1 — ULTRALYTICS / YOLO / RT-DETR  (already on disk)
# ===========================================================================
print("\n" + "="*60)
print("SECTION 1 — Shared RT-DETRv2 detector")
print("="*60)

# The six ultralytics .pt files this section used to copy are gone with the
# YOLO removal; every wrapper that needs person or vehicle boxes now goes
# through the one Apache-2.0 checkpoint below.  It is fetched by transformers
# into the HuggingFace cache rather than copied from the project root, so it
# is recorded here and downloaded by scripts/download_models.py.
record("PekingU/rtdetr_v2_r18vd",
       None,
       "hosted",
       "RT-DETRv2-R18 (shared person/vehicle detector, Apache 2.0)")


# ===========================================================================
# SECTION 2 — MEDIAPIPE BlazePose .task models
# ===========================================================================
print("\n" + "="*60)
print("SECTION 2 — MediaPipe BlazePose .task files")
print("="*60)

MEDIAPIPE_URLS = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_landmarker_heavy.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

# Also check if one already downloaded in model_weights/mediapipe
existing_mp = PROJECT_ROOT / "model_weights" / "mediapipe"
for fname, url in MEDIAPIPE_URLS.items():
    dst  = DIRS["mediapipe"] / fname
    src  = existing_mp / fname
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        log(OK, f"MediaPipe {fname}: copied from model_weights/mediapipe")
        st = "copied"
    else:
        st = download_if_missing(url, dst, f"MediaPipe {fname}")
    record(f"mediapipe/{fname}", dst, st, "Used by fall_mediapipe_pose (BlazePose backbone)")


# ===========================================================================
# SECTION 3 — PyTorchVideo pretrained weights (X3D, SlowFast, I3D)
# ===========================================================================
print("\n" + "="*60)
print("SECTION 3 — PyTorchVideo pretrained weights (X3D-S, SlowFast-R50, I3D-R50)")
print("="*60)

pv_models = [
    ("x3d_s",       "violence_x3d",     "x3d_s_kinetics400.pt"),
    ("slowfast_r50","violence_slowfast", "slowfast_r50_kinetics400.pt"),
]

for hub_name, model_key, out_name in pv_models:
    dst = DIRS["pytorchvideo"] / out_name
    if dst.exists() and dst.stat().st_size > 1_000_000:
        log(OK, f"{model_key} ({hub_name}): already saved ({out_name})")
        record(model_key, dst, "present", f"Kinetics-400 pretrained {hub_name} via torch.hub")
        continue
    log(INFO, f"{model_key}: loading {hub_name} via torch.hub (downloads on first run) ...")
    try:
        import torch
        model = torch.hub.load("facebookresearch/pytorchvideo", model=hub_name, pretrained=True)
        torch.save(model.state_dict(), dst)
        mb = dst.stat().st_size / 1e6
        log(OK, f"{model_key}: saved {out_name}  ({mb:.1f} MB)")
        record(model_key, dst, "downloaded", f"Kinetics-400 pretrained {hub_name} via torch.hub")
    except Exception as e:
        log(ERR, f"{model_key}: FAILED - {e}")
        record(model_key, None, f"error: {e}", f"Needs pytorchvideo installed (`pip install pytorchvideo`)")

# I3D via pytorchvideo
i3d_dst = DIRS["pytorchvideo"] / "i3d_r50_kinetics400.pt"
if i3d_dst.exists() and i3d_dst.stat().st_size > 1_000_000:
    log(OK, f"violence_i3d: already saved ({i3d_dst.name})")
    record("violence_i3d", i3d_dst, "present", "Kinetics-400 pretrained i3d_r50 via pytorchvideo")
else:
    log(INFO, "violence_i3d: loading i3d_r50 via pytorchvideo ...")
    try:
        from pytorchvideo.models.hub import i3d_r50
        import torch
        model = i3d_r50(pretrained=True)
        torch.save(model.state_dict(), i3d_dst)
        mb = i3d_dst.stat().st_size / 1e6
        log(OK, f"violence_i3d: saved {i3d_dst.name}  ({mb:.1f} MB)")
        record("violence_i3d", i3d_dst, "downloaded", "Kinetics-400 pretrained i3d_r50 via pytorchvideo")
    except Exception as e:
        log(ERR, f"violence_i3d: FAILED - {e}")
        record("violence_i3d", None, f"error: {e}", "Needs pytorchvideo (`pip install pytorchvideo`)")


# ===========================================================================
# SECTION 4 — VideoMAE (HuggingFace)
# ===========================================================================
print("\n" + "="*60)
print("SECTION 4 — VideoMAE HuggingFace model")
print("="*60)

videomae_dst = DIRS["videomae"]
videomae_marker = videomae_dst / "config.json"

if videomae_marker.exists():
    log(OK, "violence_videomae: already downloaded to ML Models/videomae/")
    record("violence_videomae", videomae_dst, "present",
           "MCG-NJU/videomae-base-finetuned-kinetics (HuggingFace)")
else:
    log(INFO, "violence_videomae: downloading from HuggingFace (MCG-NJU/videomae-base-finetuned-kinetics) ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="MCG-NJU/videomae-base-finetuned-kinetics",
            local_dir=str(videomae_dst),
            ignore_patterns=["*.msgpack", "flax*", "tf_*"],
        )
        log(OK, "violence_videomae: downloaded to ML Models/videomae/")
        record("violence_videomae", videomae_dst, "downloaded",
               "MCG-NJU/videomae-base-finetuned-kinetics (HuggingFace)")
    except Exception as e:
        log(ERR, f"violence_videomae: FAILED - {e}")
        record("violence_videomae", None, f"error: {e}",
               "pip install huggingface_hub, then re-run")


# ===========================================================================
# SECTION 5 — DETR Plate Detection model (HuggingFace)
# ===========================================================================
print("\n" + "="*60)
print("SECTION 5 — DETR Plate Detection model (HuggingFace)")
print("="*60)

detr_dst    = DIRS["detr_plate"]
detr_marker = detr_dst / "config.json"

if detr_marker.exists():
    log(OK, "anpr plate_detector: already downloaded to ML Models/detr_plate/")
    record("anpr_detr_plate", detr_dst, "present",
           "nickmuchi/detr-resnet50-license-plate-detection (HuggingFace)")
else:
    log(INFO, "anpr plate_detector: downloading nickmuchi/detr-resnet50-license-plate-detection ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="nickmuchi/detr-resnet50-license-plate-detection",
            local_dir=str(detr_dst),
            ignore_patterns=["*.msgpack", "flax*", "tf_*", "rust_*"],
        )
        log(OK, "anpr plate_detector: downloaded to ML Models/detr_plate/")
        record("anpr_detr_plate", detr_dst, "downloaded",
               "nickmuchi/detr-resnet50-license-plate-detection (HuggingFace)")
    except Exception as e:
        log(ERR, f"anpr plate_detector: FAILED - {e}")
        record("anpr_detr_plate", None, f"error: {e}",
               "pip install huggingface_hub, then re-run")


# ===========================================================================
# SECTION 6 — EasyOCR recognition model (~50 MB)
# ===========================================================================
print("\n" + "="*60)
print("SECTION 6 — EasyOCR recognition weights")
print("="*60)

easyocr_dst = DIRS["easyocr"]
easyocr_model_url = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
easyocr_zip  = easyocr_dst / "english_g2.zip"
easyocr_det_url = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
easyocr_det_zip = easyocr_dst / "craft_mlt_25k.zip"

for url, zipfile, label in [
    (easyocr_det_url, easyocr_det_zip, "EasyOCR detector (CRAFT)"),
    (easyocr_model_url, easyocr_zip,   "EasyOCR recognizer (English G2)"),
]:
    extracted_name = zipfile.stem
    extracted_pt   = easyocr_dst / f"{extracted_name}.pth"
    if extracted_pt.exists():
        log(OK, f"{label}: already extracted ({extracted_pt.name})")
        record(label.replace(" ", "_"), extracted_pt, "present",
               f"EasyOCR {label} — used by indian_anpr OCR stage")
        continue
    st = download_if_missing(url, zipfile, label)
    if st in ("downloaded", "present") and zipfile.exists():
        import zipfile as zf
        try:
            with zf.ZipFile(zipfile, "r") as z:
                z.extractall(easyocr_dst)
            log(OK, f"{label}: extracted to ML Models/easyocr/")
            record(label.replace(" ", "_"), easyocr_dst, "downloaded",
                   f"EasyOCR {label} — used by indian_anpr OCR stage")
        except Exception as e:
            log(ERR, f"{label}: extraction failed - {e}")
            record(label.replace(" ", "_"), None, f"error: {e}", "")


# ===========================================================================
# SECTION 7 — Models with NO local weights (classical CV / API-only)
# ===========================================================================
print("\n" + "="*60)
print("SECTION 7 — Classical CV & API-only models (no local weights)")
print("="*60)

no_weights = {
    "fall_optical_flow":        "Classical CV — Farnebäck dense optical flow. No weights needed.",
    "optical_flow_crush":       "Classical CV — Farnebäck dense optical flow. No weights needed.",
    "mog2_parked":              "Classical CV — Dual-rate MOG2 background subtraction. No weights needed.",
    "roboflow_combined":        "API-only — Roboflow hosted model 'violence-ftjyp/1'. Requires ROBOFLOW_API_KEY.",
    "roboflow_traffic":         "API-only — Roboflow hosted 'vehicle-detection-3mmwj/1'. Requires ROBOFLOW_API_KEY.",
    "indian_anpr_vehicle_stage":"API-only — Roboflow 'traffic-indian-vehicles/4'. Stage 1 of indian_anpr.",
    "indian_anpr_plate_stage":  "API-only — Roboflow 'license-plate-recognition-rxg4e/4'. Stage 2 of indian_anpr.",
    "violence_c3d":             "No pretrained weights — C3D architecture built from scratch. Requires fine-tuning on RWF-2000 / Hockey Fight / RLVS.",
    "violence_tsm":             "Partial — ImageNet ResNet-50 backbone (torchvision auto-download); 2-class violence head is random without fine-tuning.",
    "violence_mmaction_slowonly":"Fallback — Uses torchvision SlowFast_R50 (Kinetics) as fallback when MMAction2 checkpoint absent.",
    "umbrella_trained":         "Needs YOUR fine-tuned RT-DETRv2 checkpoint in 'ML Models/umbrella_trained/'.",
    "umbrella_rfdetr":          "Needs YOUR fine-tuned RF-DETR checkpoint; the rfdetr package fetches its own base weights.",
    "fall_movenet":             "TF-Hub — Loaded from tfhub.dev at runtime (cached by tensorflow_hub). No standalone file to copy.",
    "rapid_ocr":                "Auto-download — PP-OCRv4 ONNX models (~12 MB) downloaded by rapidocr_onnxruntime at first use into its own cache.",
}

# Write placeholder README files into relevant subdirs
api_readme = DIRS["roboflow_api"] / "README.txt"
api_readme.write_text(
    "These models run on Roboflow's hosted inference servers.\n"
    "No local weight files exist — inference requires a ROBOFLOW_API_KEY.\n\n"
    "Models:\n"
    "  roboflow_combined     — violence-ftjyp/1\n"
    "  roboflow_traffic      — vehicle-detection-3mmwj/1\n"
    "  indian_anpr (stage 1) — traffic-indian-vehicles/4\n"
    "  indian_anpr (stage 2) — license-plate-recognition-rxg4e/4\n",
    encoding="utf-8"
)
log(OK, "Roboflow API: wrote README.txt to ML Models/roboflow_api/")

optical_readme = DIRS["optical_flow"] / "README.txt"
optical_readme.write_text(
    "These models use classical computer vision — no neural network weights.\n\n"
    "  fall_optical_flow  — Farnebäck dense optical flow (sudden-drop heuristic)\n"
    "  optical_flow_crush — Farnebäck dense optical flow (crowd turbulence/compression)\n"
    "  mog2_parked        — Dual-rate MOG2 background subtraction (parked vehicle detection)\n",
    encoding="utf-8"
)
log(OK, "Classical CV: wrote README.txt to ML Models/optical_flow_classical/")

for key, note in no_weights.items():
    log(INFO, f"{key}: {note[:80]}...")
    record(key, None, "no_local_weights", note)


# ===========================================================================
# SECTION 8 — Write MODEL_MANIFEST.md
# ===========================================================================
print("\n" + "="*60)
print("SECTION 8 — Writing MODEL_MANIFEST.md")
print("="*60)

manifest_path = ML_MODELS / "MODEL_MANIFEST.md"

# The canonical model table.  Cross-checked against webapp/registry.py below:
# this list is hand-maintained (it carries architecture and provenance the
# registry does not model), so it drifts silently unless something compares
# them.  It previously listed nine models that had been deleted and described
# five others as running on YOLO backbones they no longer use.
ALL_MODELS = [
    # key,                          category,               architecture,                      local_weight_file,                 hf_or_hub_source
    ("fall_mediapipe_pose",         "Fall Detection",        "MediaPipe BlazePose (full) + RT-DETRv2 person stage", "mediapipe/pose_landmarker_full.task + HF rtdetr_v2_r18vd", "storage.googleapis.com/mediapipe-models"),
    ("fall_movenet",                "Fall Detection",        "Google MoveNet multipose/Lightning",  "TF-Hub (runtime cache)",         "tfhub.dev/google/movenet/multipose/lightning/1"),
    ("fall_optical_flow",           "Fall Detection",        "Classical CV — Farnebäck",            "None (no weights)",              "N/A"),
    ("optical_flow_crush",          "Crowd Crush",           "Classical CV — Farnebäck",            "None (no weights)",              "N/A"),
    ("dense_flow",                  "Crowd Crush",           "Classical CV — DIS optical flow + ORB global-motion compensation", "None (no weights)", "N/A"),
    ("crowd_motion_monitor",        "Crowd Crush",           "RT-DETRv2-R18 + APGCC Point Detector + Farneback Flow", "models/head_count/weights/APGCC_SHHA_best.pth + HF rtdetr_v2_r18vd", "ShanghaiTech-A APGCC + PekingU/rtdetr_v2_r18vd (Apache 2.0)"),
    ("roboflow_combined",           "Violence Detection",    "Roboflow hosted model",               "API only",                       "universe.roboflow.com — violence-ftjyp/1"),
    ("violence_x3d",                "Violence Detection",    "X3D-S + Kinetics-400 zero-shot",      "pytorchvideo/x3d_s_kinetics400.pt", "torch.hub facebookresearch/pytorchvideo"),
    ("violence_slowfast",           "Violence Detection",    "SlowFast-R50 + Kinetics-400 zero-shot","pytorchvideo/slowfast_r50_kinetics400.pt", "torch.hub facebookresearch/pytorchvideo"),
    ("violence_videomae",           "Violence Detection",    "VideoMAE-Base Kinetics-400",          "videomae/ (HuggingFace snapshot)", "MCG-NJU/videomae-base-finetuned-kinetics"),
    ("violence_i3d",                "Violence Detection",    "I3D-R50 + Kinetics-400 zero-shot",    "pytorchvideo/i3d_r50_kinetics400.pt", "pytorchvideo.models.hub.i3d_r50"),
    ("violence_c3d",                "Violence Detection",    "C3D (scratch) — no pretrained weights","None (fine-tune required)",      "RWF-2000 / Hockey Fight / RLVS fine-tune needed"),
    ("violence_tsm",                "Violence Detection",    "TSM-ResNet50 (ImageNet backbone)",    "torchvision auto-download",      "torchvision ResNet50_Weights.DEFAULT (ImageNet)"),
    ("violence_mmaction_slowonly",  "Violence Detection",    "MMAction2 SlowOnly (falls back to torchvision SlowFast)", "torchvision auto-download", "MMAction2 checkpoint required; fallback: torchvision"),
    ("rtdetrv2_traffic",            "Traffic Monitoring",    "RT-DETRv2-R18 + centroid-drift moving/parked classifier", "HF rtdetr_v2_r18vd", "PekingU/rtdetr_v2_r18vd (Apache 2.0, COCO)"),
    ("roboflow_traffic",            "Traffic Monitoring",    "Roboflow hosted model + DeepSORT",    "API only",                       "universe.roboflow.com — vehicle-detection-3mmwj/1"),
    ("mog2_parked",                 "Traffic Monitoring",    "Classical CV — Dual-rate MOG2",       "None (no weights)",              "N/A"),
    ("anpr",                        "ANPR",                  "RT-DETRv2 (vehicle) + DETR (plate) + EasyOCR/RapidOCR", "HF rtdetr_v2_r18vd + detr_plate/ + easyocr/", "nickmuchi/detr-resnet50-license-plate-detection"),
    ("indian_anpr",                 "ANPR",                  "Roboflow Indian Vehicles + Roboflow Plate + EasyOCR", "easyocr/ (stage 3 OCR)", "Roboflow API stages 1 & 2; EasyOCR stage 3"),
    ("rtdetrv2_anpr",               "ANPR",                  "RT-DETRv2 (vehicle) + DETR (plate) + OCR", "HF rtdetr_v2_r18vd + detr_plate/", "PekingU/rtdetr_v2_r18vd (Apache 2.0)"),
    ("rapid_ocr",                   "ANPR",                  "PP-OCRv4 ONNX (RapidOCR)",           "Auto-download by rapidocr_onnxruntime", "PP-OCRv4 mobile det/cls/rec ONNX (~12 MB total)"),
    ("umbrella_ssd",                "Umbrella Detection",    "SSDLite320-MobileNetV3 (COCO class 28)", "torchvision auto-download",   "torchvision SSDLite320_MobileNet_V3_Large_Weights.COCO_V1"),
    ("umbrella_rtdetrv2",           "Umbrella Detection",    "RT-DETRv2-R18 (COCO umbrella class 25)", "HF rtdetr_v2_r18vd",           "PekingU/rtdetr_v2_r18vd (Apache 2.0)"),
    ("umbrella_rfdetr",             "Umbrella Detection",    "RF-DETR Nano (DINOv2 backbone)",      "weights/rfdetr_nano_umbrella.pt if present", "rfdetr package fetches its own base weights"),
    ("umbrella_trained",            "Umbrella Detection",    "RT-DETRv2 fine-tuned on umbrellas",   "ML Models/umbrella_trained/ (yours)", "your fine-tuned checkpoint"),
]

# Fail loudly on drift rather than emitting a confident, wrong manifest.
try:
    _src = PROJECT_ROOT / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from webapp import registry as _registry
    _known = set(_registry.BY_KEY)
    _listed = {row[0] for row in ALL_MODELS}
    _missing, _extra = sorted(_known - _listed), sorted(_listed - _known)
    if _missing:
        log(WARN, f"in the registry but missing from this manifest: {', '.join(_missing)}")
    if _extra:
        log(WARN, f"in this manifest but NOT in the registry (deleted?): {', '.join(_extra)}")
    if not _missing and not _extra:
        log(OK, f"manifest matches the registry ({len(_known)} models)")
except Exception as _exc:  # noqa: BLE001
    log(WARN, f"could not cross-check the manifest against the registry: {_exc}")

lines = [
    "# ML Models — Complete Model Manifest",
    "",
    "**Project**: Crowd Safety & ANPR Testbed  ",
    "**Prepared for**: Senior AI/ML Engineering & Architecture Review  ",
    f"**Total Models**: {len(ALL_MODELS)}  ",
    "",
    "---",
    "",
    "## Directory Structure",
    "",
    "```",
    "ML Models/",
    "├── mediapipe/            # BlazePose .task files (lite / full / heavy)",
    "├── pytorchvideo/         # X3D-S, SlowFast-R50, I3D-R50 state dicts",
    "├── videomae/             # VideoMAE-Base HuggingFace snapshot",
    "├── detr_plate/           # DETR license-plate detector (HuggingFace snapshot)",
    "├── easyocr/              # EasyOCR CRAFT detector + English recognizer",
    "├── roboflow_api/         # README only — models run on Roboflow servers",
    "└── optical_flow_classical/ # README only — classical CV, no weights",
    "```",
    "",
    "---",
    "",
    "## Complete Model Table",
    "",
    "| # | Model Key | Category | Architecture | Local Weight File | Source |",
    "|---|-----------|----------|--------------|-------------------|--------|",
]

for i, (key, cat, arch, wfile, src) in enumerate(ALL_MODELS, 1):
    lines.append(f"| {i} | `{key}` | {cat} | {arch} | `{wfile}` | {src} |")

lines += [
    "",
    "---",
    "",
    "## Status Key",
    "",
    "| Status | Meaning |",
    "|--------|---------|",
    "| **present** | Weight file already existed locally |",
    "| **copied** | Copied from project root / model_weights/ into ML Models/ |",
    "| **downloaded** | Freshly downloaded during collection |",
    "| **no_local_weights** | Classical CV or API-only — no file to collect |",
    "| **error** | Download failed — see script output |",
    "",
    "---",
    "",
    "## Notes for Reviewers",
    "",
    "1. **Violence models (C3D, TSM, SlowOnly)** — These three require domain-specific",
    "   fine-tuning on RWF-2000, Hockey Fight, or RLVS datasets to produce real violence",
    "   detections. Without fine-tuned weights they run in a clearly-tagged degraded mode",
    "   (`extra.untrained=True`). The weights are not included because no fine-tuned",
    "   checkpoint has been produced for this project yet.",
    "",
    "2. **Fall models (ST-GCN, PoseC3D, AlphaPose-LSTM)** — These fall back to a shared",
    "   geometric posture heuristic (YOLOv8s-pose.pt backbone) when no trained classifier",
    "   checkpoint is supplied. This fallback is tagged `extra.scoring='geometric_fallback'`.",
    "",
    "3. **MoveNet** — Loaded from TF-Hub at runtime; tensorflow_hub caches it in",
    "   `~/.cache/tfhub_modules/`. Not a standalone .pt file.",
    "",
    "4. **RapidOCR** — PP-OCRv4 ONNX models (~12 MB) are auto-downloaded by the",
    "   `rapidocr_onnxruntime` package on first use into its own package cache.",
    "",
    "5. **Roboflow API models** — Require a `ROBOFLOW_API_KEY` environment variable.",
    "",
    "6. **RT-DETRv2** — `PekingU/rtdetr_v2_r18vd` (81 MB, Apache 2.0) is the shared",
    "   person/vehicle detector for every wrapper that needs boxes. transformers",
    "   caches it; `scripts/download_models.py` fetches it ahead of a timed run.",
    "",
    "7. **umbrella_rfdetr / umbrella_trained** — Need fine-tuned umbrella-specific",
    "   checkpoints you supply. Without them the registry marks them accordingly",
    "   rather than silently substituting a different model.",
]

manifest_path.write_text("\n".join(lines), encoding="utf-8")
log(OK, f"MODEL_MANIFEST.md written to {manifest_path.relative_to(PROJECT_ROOT)}")


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "="*60)
print("COLLECTION COMPLETE")
print("="*60)

# Count files in ML Models
total_files = sum(1 for _ in ML_MODELS.rglob("*") if _.is_file())
total_mb    = sum(f.stat().st_size for f in ML_MODELS.rglob("*") if f.is_file()) / 1e6

print(f"\n  ML Models/  ->  {total_files} files,  {total_mb:.1f} MB total")
print(f"\n  Models collected: {len(ALL_MODELS)} / 29")
print()

# Print any errors
errors = [(k, v) for k, v in STATUS.items() if "error" in v.get("status", "")]
if errors:
    print(f"  {ERR} {len(errors)} model(s) failed to download:")
    for k, v in errors:
        print(f"       {k}: {v['status']}")
else:
    print(f"  {OK} No download errors.")

print(f"\n  Full manifest: ML Models/MODEL_MANIFEST.md")
print()
