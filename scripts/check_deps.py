"""
Checks every model's dependencies WITHOUT running inference — catches
missing/broken imports up front in one pass, instead of discovering them
one at a time mid-batch-run.

Usage:
    python scripts/check_deps.py
"""

import sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Every key in webapp/registry.py, so the check covers what the project can
# actually run.  This listed 11 of 24 models and included three that had been
# deleted — which made a clean report mean considerably less than it looked.
# A mismatch against the registry is itself reported below.
MODELS_TO_CHECK = {
    "optical_flow_crush": "models.optical_flow_crush",
    "roboflow_combined": "models.roboflow_combined",
    "dense_flow": "models.crowd_flow.dense_flow_analyser",
    # Fall
    "fall_mediapipe_pose": "models.fall.mediapipe_pose",
    "fall_movenet": "models.fall.movenet",
    "fall_optical_flow": "models.fall.optical_flow_fall",
    # Violence
    "violence_x3d": "models.violence.x3d",
    "violence_slowfast": "models.violence.slowfast",
    "violence_videomae": "models.violence.videomae",
    "violence_i3d": "models.violence.i3d",
    "violence_c3d": "models.violence.c3d",
    "violence_tsm": "models.violence.tsm",
    "violence_mmaction_slowonly": "models.violence.mmaction_slowonly",
    # Traffic
    "rtdetrv2_traffic": "models.traffic.rtdetrv2_traffic",
    "roboflow_traffic": "models.traffic.roboflow_traffic",
    "mog2_parked": "models.traffic.mog2_parked",
    # ANPR
    "anpr": "models.anpr.anpr",
    "indian_anpr": "models.anpr.indian_anpr",
    "rapid_ocr": "models.anpr.rapid_ocr_wrapper",
    "rtdetrv2_anpr": "models.anpr.rtdetrv2_anpr",
    # Umbrella
    "umbrella_ssd": "models.umbrella.umbrella_ssd",
    "umbrella_rfdetr": "models.umbrella.umbrella_rfdetr",
    "umbrella_rtdetrv2": "models.umbrella.umbrella_rtdetrv2",
    "umbrella_trained": "models.umbrella.umbrella_trained",
    # Crowd motion
    "crowd_motion_monitor": "models.crowd_flow.crowd_motion_monitor",
}

# Each model's actual third-party import happens lazily inside load(), not
# at module import time — so we check the SPECIFIC libraries each model
# needs, not just whether the wrapper module imports cleanly.
EXTRA_LIB_CHECKS = {
    # transformers: the shared RT-DETRv2 person detector that crops each
    # person before BlazePose runs.  This said "ultralytics" until the YOLO
    # removal, and kept saying it afterwards — so the check passed on a
    # library the model no longer uses and would have missed the one it does.
    "fall_mediapipe_pose": ["mediapipe", "transformers"],
    "fall_movenet": ["tensorflow_hub", "tensorflow"],
    "violence_videomae": ["transformers"],
    "violence_i3d": ["pytorchvideo"],
    "violence_mmaction_slowonly": ["mmaction"],
    # The RT-DETRv2 family — shared detector and the wrappers built on it.
    "dense_flow": ["cv2"],
    "anpr": ["transformers", "easyocr"],
    "indian_anpr": ["transformers", "easyocr"],
    "rapid_ocr": ["rapidocr_onnxruntime"],
    "rtdetrv2_anpr": ["transformers"],
    "rtdetrv2_traffic": ["transformers"],
    "umbrella_rtdetrv2": ["transformers"],
    "umbrella_trained": ["transformers"],
    "umbrella_rfdetr": ["rfdetr"],
    "umbrella_ssd": ["torchvision"],
    "roboflow_combined": ["inference_sdk"],
    "roboflow_traffic": ["inference_sdk"],
    "crowd_motion_monitor": ["cv2", "transformers"],
}


def check_module_import(module_path: str):
    import importlib
    try:
        importlib.import_module(module_path)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_lib(lib_name: str):
    import importlib
    try:
        importlib.import_module(lib_name)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_registry_coverage() -> None:
    """
    Report any drift between this script's table and webapp/registry.py.

    Silent drift is how the previous version ended up checking three deleted
    models and skipping half the live ones while still printing a clean bill
    of health.  A missing model here is not a failure of the project, but it
    is a failure of this check to mean what it says.
    """
    try:
        from webapp import registry
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] could not import the registry to cross-check: {exc}")
        return

    known = set(registry.BY_KEY)
    checked = set(MODELS_TO_CHECK)
    missing = sorted(known - checked)
    extra = sorted(checked - known)
    if missing:
        print(f"  [WARN] in the registry but NOT checked here: {', '.join(missing)}")
    if extra:
        print(f"  [WARN] checked here but NOT in the registry (deleted?): "
              f"{', '.join(extra)}")
    if not missing and not extra:
        print(f"  All {len(known)} registry models are covered by this check.")


def main():
    print("Cross-checking against webapp/registry.py...\n")
    check_registry_coverage()

    print("\nChecking wrapper modules import cleanly (structural check)...\n")
    results = {}
    for model_name, module_path in MODELS_TO_CHECK.items():
        ok, err = check_module_import(module_path)
        results[model_name] = {"wrapper_ok": ok, "wrapper_err": err, "lib_issues": []}
        if not ok:
            print(f"  [WRAPPER BROKEN] {model_name}: {err}")

    print("\nChecking each model's specific runtime dependencies...\n")
    for model_name, libs in EXTRA_LIB_CHECKS.items():
        for lib in libs:
            ok, err = check_lib(lib)
            if not ok:
                results[model_name]["lib_issues"].append((lib, err))
                print(f"  [MISSING DEP] {model_name} needs '{lib}': {err}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ready, broken = [], []
    for model_name, r in results.items():
        if r["wrapper_ok"] and not r["lib_issues"]:
            ready.append(model_name)
        else:
            broken.append(model_name)

    print(f"\nReady to run ({len(ready)}):")
    for m in ready:
        print(f"  OK  {m}")

    print(f"\nWill likely fail ({len(broken)}):")
    for m in broken:
        print(f"  XX  {m}")

    if broken:
        print("\nFix these BEFORE running run_all.py to avoid mid-batch crashes.")


if __name__ == "__main__":
    main()