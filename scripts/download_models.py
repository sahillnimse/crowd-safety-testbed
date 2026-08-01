"""
Download every model weight the testbed can use into one folder.

    python scripts/download_models.py --dry-run     # show what/how big
    python scripts/download_models.py               # download everything
    python scripts/download_models.py --only umbrella anpr

Why a script rather than a list of curl commands: the weights come from six
different ecosystems (ultralytics, torch.hub, HuggingFace, EasyOCR, TF-Hub,
MediaPipe), and each caches to its own hidden directory by default —
`~/.cache/torch`, `~/.cache/huggingface`, `~/.EasyOCR`, and so on. Pointing
them all at one folder means setting the right environment variable for
each *before* the library is imported, which is exactly the kind of thing
that is easy to get subtly wrong by hand.

**Not all 29 registered models have weights.** Three are classical CV with
nothing to download, three run entirely on Roboflow's servers, and two
umbrella variants have no distinct checkpoint of their own — they fall back
to weights another model already uses. The script reports those explicitly
rather than pretending it fetched something.
"""

import argparse
import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_DIR = os.path.join(PROJECT_ROOT, "ML Models")

# Sub-folder per ecosystem, so it stays obvious which library owns what and
# a partial re-download can be done by deleting one folder.
DIRS = {
    "ultralytics": os.path.join(ML_DIR, "ultralytics"),
    "torch_hub": os.path.join(ML_DIR, "torch_hub"),
    "huggingface": os.path.join(ML_DIR, "huggingface"),
    "easyocr": os.path.join(ML_DIR, "easyocr"),
    "tfhub": os.path.join(ML_DIR, "tfhub"),
    "mediapipe": os.path.join(ML_DIR, "mediapipe"),
    "rapidocr": os.path.join(ML_DIR, "rapidocr"),
}


def prepare_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


def set_cache_env():
    """Point every library's cache at ML Models.

    Must run before torch/transformers/easyocr are imported — they read
    these at import time, so setting them afterwards silently has no effect
    and the weights land in the default cache instead.
    """
    os.environ["TORCH_HOME"] = DIRS["torch_hub"]
    os.environ["HF_HOME"] = DIRS["huggingface"]
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(DIRS["huggingface"], "hub")
    os.environ["EASYOCR_MODULE_PATH"] = DIRS["easyocr"]
    os.environ["TFHUB_CACHE_DIR"] = DIRS["tfhub"]


# (key, label, approx MB, category, fn) — fn is called to fetch it.
# Sizes are approximate download sizes, used for the dry-run estimate.
def _ultra(name):
    def go():
        from ultralytics import YOLO, RTDETR
        cls = RTDETR if name.startswith("rtdetr") else YOLO
        cls(name)                      # triggers download into CWD
        _collect_to(name, DIRS["ultralytics"])
    return go


def _collect_to(filename, dest):
    """Ultralytics downloads to the working directory; move it into place."""
    src = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(src):
        target = os.path.join(dest, filename)
        if os.path.abspath(src) != os.path.abspath(target):
            shutil.move(src, target)


def _hf_detection(model_id):
    def go():
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        AutoImageProcessor.from_pretrained(model_id)
        AutoModelForObjectDetection.from_pretrained(model_id)
    return go


def _hf_video(model_id):
    def go():
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        VideoMAEImageProcessor.from_pretrained(model_id)
        VideoMAEForVideoClassification.from_pretrained(model_id)
    return go


def _torchhub_pytorchvideo(name):
    def go():
        import torch
        torch.hub.load("facebookresearch/pytorchvideo", model=name, pretrained=True)
    return go


def _torchvision(fn_name, weights_attr):
    def go():
        import torchvision.models as tvm
        import torchvision.models.detection as tvd
        import torchvision.models.video as tvv
        for mod in (tvd, tvv, tvm):
            if hasattr(mod, fn_name):
                weights = getattr(mod, weights_attr).DEFAULT
                getattr(mod, fn_name)(weights=weights)
                return
        raise AttributeError(f"torchvision has no {fn_name}")
    return go


def _easyocr():
    def go():
        import easyocr
        easyocr.Reader(["en"], gpu=False, verbose=False,
                       model_storage_directory=DIRS["easyocr"],
                       download_enabled=True)
    return go


def _mediapipe():
    def go():
        from models.fall.mediapipe_pose import _get_model_path
        _get_model_path(1)             # "full" variant, the wrapper's default
    return go


def _movenet():
    def go():
        import tensorflow_hub as hub
        hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return go


def _rapidocr():
    def go():
        from rapidocr_onnxruntime import RapidOCR
        RapidOCR()                     # pulls PP-OCRv4 det/cls/rec ONNX
    return go


def _clip():
    def go():
        # YOLO-World's text encoder. Downloaded by ultralytics on first
        # set_classes(); fetching it here keeps it out of a timed run.
        from ultralytics import YOLO
        m = YOLO(os.path.join(DIRS["ultralytics"], "yolov8s-worldv2.pt"))
        m.set_classes(["umbrella"])
    return go


TASKS = [
    # key,                     label,                             MB,   used by
    ("yolo11n",       "yolo11n.pt",                                6,  "traffic, umbrella, ANPR", _ultra("yolo11n.pt")),
    ("yolo11s",       "yolo11s.pt",                               19,  "umbrella (size 's')",     _ultra("yolo11s.pt")),
    ("yolov8n",       "yolov8n.pt",                                7,  "person detector",         _ultra("yolov8n.pt")),
    ("yolov8s_pose",  "yolov8s-pose.pt",                          23,  "all fall models",         _ultra("yolov8s-pose.pt")),
    ("rtdetr_l",      "rtdetr-l.pt",                              64,  "rtdetr_traffic, rfdetr",  _ultra("rtdetr-l.pt")),
    ("world",         "yolov8s-worldv2.pt",                       25,  "umbrella_world",          _ultra("yolov8s-worldv2.pt")),
    ("clip",          "CLIP ViT-B-32 (YOLO-World text encoder)", 338,  "umbrella_world",          _clip()),

    ("x3d",           "X3D-S (Kinetics)",                         15,  "violence_x3d",            _torchhub_pytorchvideo("x3d_s")),
    ("slowfast",      "SlowFast R50 (Kinetics)",                 264,  "violence_slowfast",       _torchhub_pytorchvideo("slowfast_r50")),
    ("i3d",           "I3D R50 (Kinetics)",                      200,  "violence_i3d",            _torchhub_pytorchvideo("i3d_r50")),
    ("r3d18",         "r3d_18 (Kinetics)",                       127,  "mmaction fallback",       _torchvision("r3d_18", "R3D_18_Weights")),
    ("resnet50",      "ResNet-50 (ImageNet)",                     98,  "violence_tsm backbone",   _torchvision("resnet50", "ResNet50_Weights")),
    ("ssdlite",       "SSDLite320 MobileNetV3 (COCO)",            14,  "umbrella_ssd",            _torchvision("ssdlite320_mobilenet_v3_large", "SSDLite320_MobileNet_V3_Large_Weights")),

    ("videomae",      "VideoMAE-base (Kinetics)",                345,  "violence_videomae",       _hf_video("MCG-NJU/videomae-base-finetuned-kinetics")),
    ("plate_detr",    "DETR-R50 licence plate",                  167,  "anpr plate localisation", _hf_detection("nickmuchi/detr-resnet50-license-plate-detection")),

    ("easyocr",       "EasyOCR EN detector + recogniser",         100,  "anpr, indian_anpr",       _easyocr()),
    ("rapidocr",      "RapidOCR PP-OCRv4 ONNX",                   15,  "rapid_ocr",               _rapidocr()),
    ("mediapipe",     "MediaPipe PoseLandmarker (full)",           9,  "fall_mediapipe_pose",     _mediapipe()),
    ("movenet",       "MoveNet MultiPose Lightning",               25,  "fall_movenet",           _movenet()),
]

# Registered models with nothing to fetch — reported so the count adds up.
NO_DOWNLOAD = {
    "fall_optical_flow":   "classical CV (Farneback optical flow)",
    "optical_flow_crush":  "classical CV (Farneback optical flow)",
    "mog2_parked":         "classical CV (MOG2 background subtraction)",
    "roboflow_combined":   "hosted - runs on Roboflow's servers",
    "roboflow_traffic":    "hosted - runs on Roboflow's servers",
    "indian_anpr":         "hosted vehicle+plate stages; OCR weights covered by easyocr",
    "violence_c3d":        "architecture built in code; needs YOUR fine-tuned checkpoint",
    "fire_smoke_yolo":     "needs YOUR fine-tuned fire/smoke checkpoint",
    "umbrella_yolo26n":    "no distinct weights - falls back to yolo11n.pt",
    "umbrella_rfdetr":     "no distinct weights - falls back to rtdetr-l.pt",
}

CATEGORIES = {
    "core":     ["yolo11n", "yolov8n", "yolov8s_pose"],
    "umbrella": ["yolo11n", "yolo11s", "world", "clip", "ssdlite", "rtdetr_l"],
    "violence": ["x3d", "slowfast", "i3d", "r3d18", "resnet50", "videomae"],
    "anpr":     ["plate_detr", "easyocr", "rapidocr", "yolo11n"],
    "traffic":  ["yolo11n", "rtdetr_l"],
    "fall":     ["yolov8s_pose", "yolov8n", "mediapipe", "movenet"],
}


def human(mb):
    return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be downloaded, with sizes")
    ap.add_argument("--only", nargs="+", metavar="CATEGORY",
                    choices=sorted(CATEGORIES),
                    help=f"limit to categories: {', '.join(sorted(CATEGORIES))}")
    args = ap.parse_args()

    if args.only:
        wanted = set()
        for c in args.only:
            wanted.update(CATEGORIES[c])
        tasks = [t for t in TASKS if t[0] in wanted]
    else:
        tasks = TASKS

    total = sum(t[2] for t in tasks)

    print(f"\nTarget folder: {ML_DIR}")
    print(f"{len(tasks)} weight set(s), roughly {human(total)} to download.\n")
    print(f"  {'model':<42} {'size':>8}  used by")
    print("  " + "-" * 86)
    for _, label, mb, used, _fn in tasks:
        print(f"  {label:<42} {human(mb):>8}  {used}")

    if args.dry_run:
        print(f"\n{len(NO_DOWNLOAD)} registered model(s) need no download:")
        for k, why in NO_DOWNLOAD.items():
            print(f"  {k:<24} {why}")
        print("\nRe-run without --dry-run to fetch.")
        return

    prepare_dirs()
    print()

    ok, failed = [], []
    for i, (key, label, mb, _used, fn) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {label} ...", flush=True)
        try:
            fn()
            ok.append(label)
        except Exception as e:  # noqa: BLE001 - one failure must not stop the rest
            failed.append((label, f"{e.__class__.__name__}: {e}"))
            print(f"        FAILED: {e.__class__.__name__}: {e}")

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print("\nFailures (re-run to retry; downloads already fetched are skipped):")
        for label, err in failed:
            print(f"  {label}: {err[:120]}")

    print(f"\nOn disk in {ML_DIR}:")
    grand = 0
    for name, path in sorted(DIRS.items()):
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _dn, fn_ in os.walk(path) for f in fn_)
        grand += size
        if size:
            print(f"  {name:<14} {size/1e6:8.1f} MB")
    print(f"  {'TOTAL':<14} {grand/1e6:8.1f} MB")

    print("\nTo make the app use this folder, set these before launching:")
    print(f'  set TORCH_HOME={DIRS["torch_hub"]}')
    print(f'  set HF_HOME={DIRS["huggingface"]}')
    print(f'  set EASYOCR_MODULE_PATH={DIRS["easyocr"]}')
    print(f'  set TFHUB_CACHE_DIR={DIRS["tfhub"]}')


if __name__ == "__main__":
    set_cache_env()                    # before any heavy import
    sys.path.insert(0, PROJECT_ROOT)
    main()
