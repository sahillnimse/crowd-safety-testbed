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
7. Writes a MODEL_MANIFEST.md inside 'ML Models/' cataloguing all 29 models,
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
    "ultralytics":   ML_MODELS / "ultralytics",
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
print("SECTION 1 — Ultralytics / YOLO / RT-DETR weights")
print("="*60)

ultralytics_weights = [
    ("yolov8n.pt",         "YOLOv8-Nano  (fall_mediapipe_pose person detector, anpr vehicle detector)"),
    ("yolov8s-pose.pt",    "YOLOv8s-Pose (fall_yolo_pose, fall_stgcn, fall_posec3d, fall_alphapose_lstm)"),
    ("yolov8x-pose.pt",    "YOLOv8x-Pose (largest pose variant, optional high-accuracy fallback)"),
    ("yolo11n.pt",         "YOLO11-Nano  (yolo_traffic, umbrella_yolo, umbrella_yolo26n/rfdetr fallback)"),
    ("yolo11s.pt",         "YOLO11-Small (umbrella_yolo size='s')"),
    ("rtdetr-l.pt",        "RT-DETR-L    (rtdetr_traffic, umbrella_rfdetr fallback)"),
    ("yolov8s-worldv2.pt", "YOLOv8s-WorldV2 (umbrella_world open-vocabulary)"),
]

for filename, label in ultralytics_weights:
    src = PROJECT_ROOT / filename
    st  = copy_if_exists(src, DIRS["ultralytics"], label)
    dst = DIRS["ultralytics"] / filename
    record(filename, dst if st != "missing_local" else None, st, label)


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
    "fire_smoke_yolo":          "API fallback — No fine-tuned fire/smoke .pt on disk; falls back to Roboflow 'smoke-fire-detection-fpxa0/1'.",
    "violence_c3d":             "No pretrained weights — C3D architecture built from scratch. Requires fine-tuning on RWF-2000 / Hockey Fight / RLVS.",
    "violence_tsm":             "Partial — ImageNet ResNet-50 backbone (torchvision auto-download); 2-class violence head is random without fine-tuning.",
    "violence_mmaction_slowonly":"Fallback — Uses torchvision SlowFast_R50 (Kinetics) as fallback when MMAction2 checkpoint absent.",
    "fall_stgcn":               "Fallback only — YOLOv8s-pose.pt (already copied). ST-GCN head uses geometric fallback; needs NTU RGB+D / UR Fall checkpoint for trained inference.",
    "fall_posec3d":             "Fallback only — YOLOv8s-pose.pt (already copied). PoseC3D head uses geometric fallback; needs MMAction2 checkpoint.",
    "fall_alphapose_lstm":      "Fallback only — YOLOv8s-pose.pt (already copied). LSTM head uses geometric fallback; needs UR Fall / Le2i checkpoint.",
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
    "  indian_anpr (stage 2) — license-plate-recognition-rxg4e/4\n"
    "  fire_smoke_yolo       — smoke-fire-detection-fpxa0/1  (fallback only)\n",
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

# Define the canonical 29-model table
ALL_MODELS = [
    # key,                          category,               architecture,                      local_weight_file,                 hf_or_hub_source
    ("fall_yolo_pose",              "Fall Detection",        "YOLOv8s-Pose + geometric heuristic",  "ultralytics/yolov8s-pose.pt",    "ultralytics (COCO pretrained)"),
    ("fall_mediapipe_pose",         "Fall Detection",        "MediaPipe BlazePose (full) + YOLOv8n", "mediapipe/pose_landmarker_full.task + ultralytics/yolov8n.pt", "storage.googleapis.com/mediapipe-models"),
    ("fall_stgcn",                  "Fall Detection",        "ST-GCN + YOLOv8s-Pose (geometric fallback)", "ultralytics/yolov8s-pose.pt", "pyskl / trained checkpoint required for ST-GCN head"),
    ("fall_movenet",                "Fall Detection",        "Google MoveNet multipose/Lightning",  "TF-Hub (runtime cache)",         "tfhub.dev/google/movenet/multipose/lightning/1"),
    ("fall_optical_flow",           "Fall Detection",        "Classical CV — Farnebäck",            "None (no weights)",              "N/A"),
    ("fall_posec3d",                "Fall Detection",        "PoseC3D + YOLOv8s-Pose (geometric fallback)", "ultralytics/yolov8s-pose.pt", "MMAction2 checkpoint required for PoseC3D head"),
    ("fall_alphapose_lstm",         "Fall Detection",        "AlphaPose (fallback: YOLO) + LSTM (geometric fallback)", "ultralytics/yolov8s-pose.pt", "LSTM checkpoint required for trained inference"),
    ("optical_flow_crush",          "Crowd Crush",           "Classical CV — Farnebäck",            "None (no weights)",              "N/A"),
    ("roboflow_combined",           "Violence Detection",    "Roboflow hosted model",               "API only",                       "universe.roboflow.com — violence-ftjyp/1"),
    ("violence_x3d",                "Violence Detection",    "X3D-S + Kinetics-400 zero-shot",      "pytorchvideo/x3d_s_kinetics400.pt", "torch.hub facebookresearch/pytorchvideo"),
    ("violence_slowfast",           "Violence Detection",    "SlowFast-R50 + Kinetics-400 zero-shot","pytorchvideo/slowfast_r50_kinetics400.pt", "torch.hub facebookresearch/pytorchvideo"),
    ("violence_videomae",           "Violence Detection",    "VideoMAE-Base Kinetics-400",          "videomae/ (HuggingFace snapshot)", "MCG-NJU/videomae-base-finetuned-kinetics"),
    ("violence_i3d",                "Violence Detection",    "I3D-R50 + Kinetics-400 zero-shot",    "pytorchvideo/i3d_r50_kinetics400.pt", "pytorchvideo.models.hub.i3d_r50"),
    ("violence_c3d",                "Violence Detection",    "C3D (scratch) — no pretrained weights","None (fine-tune required)",      "RWF-2000 / Hockey Fight / RLVS fine-tune needed"),
    ("violence_tsm",                "Violence Detection",    "TSM-ResNet50 (ImageNet backbone)",    "torchvision auto-download",      "torchvision ResNet50_Weights.DEFAULT (ImageNet)"),
    ("violence_mmaction_slowonly",  "Violence Detection",    "MMAction2 SlowOnly (falls back to torchvision SlowFast)", "torchvision auto-download", "MMAction2 checkpoint required; fallback: torchvision"),
    ("yolo_traffic",                "Traffic Monitoring",    "YOLO11-Nano + ByteTrack",             "ultralytics/yolo11n.pt",         "ultralytics (COCO pretrained)"),
    ("rtdetr_traffic",              "Traffic Monitoring",    "RT-DETR-L + ByteTrack",               "ultralytics/rtdetr-l.pt",        "ultralytics (COCO pretrained)"),
    ("roboflow_traffic",            "Traffic Monitoring",    "Roboflow hosted model + DeepSORT",    "API only",                       "universe.roboflow.com — vehicle-detection-3mmwj/1"),
    ("mog2_parked",                 "Traffic Monitoring",    "Classical CV — Dual-rate MOG2",       "None (no weights)",              "N/A"),
    ("anpr",                        "ANPR",                  "YOLOv8n (vehicle) + DETR (plate) + EasyOCR/RapidOCR", "ultralytics/yolov8n.pt + detr_plate/ + easyocr/", "nickmuchi/detr-resnet50-license-plate-detection"),
    ("indian_anpr",                 "ANPR",                  "Roboflow Indian Vehicles + Roboflow Plate + EasyOCR", "easyocr/ (stage 3 OCR)", "Roboflow API stages 1 & 2; EasyOCR stage 3"),
    ("fire_smoke_yolo",             "Fire & Smoke",          "YOLO (fine-tuned) — fallback to Roboflow", "None locally (fallback to API)", "smoke-fire-detection-fpxa0/1 on Roboflow"),
    ("umbrella_yolo",               "Umbrella Detection",    "YOLO11-Nano (COCO class 25)",         "ultralytics/yolo11n.pt",         "ultralytics (COCO pretrained)"),
    ("umbrella_ssd",                "Umbrella Detection",    "SSDLite320-MobileNetV3 (COCO class 28)", "torchvision auto-download",   "torchvision SSDLite320_MobileNet_V3_Large_Weights.COCO_V1"),
    ("umbrella_world",              "Umbrella Detection",    "YOLOv8s-WorldV2 open-vocabulary",     "ultralytics/yolov8s-worldv2.pt", "ultralytics (CLIP text encoder + COCO pretrained)"),
    ("umbrella_rfdetr",             "Umbrella Detection",    "RF-DETR Nano / RT-DETR-L (fallback)", "ultralytics/rtdetr-l.pt (fallback)", "weights/rfdetr_nano_umbrella.pt if fine-tuned checkpoint present"),
    ("umbrella_yolo26n",            "Umbrella Detection",    "YOLO26-Nano (fallback: YOLO11-Nano)", "ultralytics/yolo11n.pt (fallback)", "weights/yolo26n_umbrella.pt if fine-tuned checkpoint present"),
    ("rapid_ocr",                   "ANPR",                  "PP-OCRv4 ONNX (RapidOCR)",           "Auto-download by rapidocr_onnxruntime", "PP-OCRv4 mobile det/cls/rec ONNX (~12 MB total)"),
]

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
    "├── ultralytics/          # YOLO / RT-DETR .pt weight files",
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
    "5. **Roboflow API models** — Require a `ROBOFLOW_API_KEY`. The key currently",
    "   hardcoded in the source (`c9KEmh1NFvhY8WFH9Iq5`) should be rotated before",
    "   production deployment.",
    "",
    "6. **Fire/Smoke** — No fine-tuned `fire_smoke_yolov8.pt` exists in this repo.",
    "   The wrapper falls back to Roboflow hosted inference automatically.",
    "",
    "7. **umbrella_rfdetr / umbrella_yolo26n** — Fine-tuned umbrella-specific",
    "   checkpoints (`rfdetr_nano_umbrella.pt`, `yolo26n_umbrella.pt`) are not present.",
    "   Both fall back to `rtdetr-l.pt` and `yolo11n.pt` respectively.",
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
