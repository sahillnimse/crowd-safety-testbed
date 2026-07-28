"""
Checks every model's dependencies WITHOUT running inference — catches
missing/broken imports up front in one pass, instead of discovering them
one at a time mid-batch-run.

Usage:
    python scripts/check_deps.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_TO_CHECK = {
    "fire_smoke_yolo": "models.fire_smoke_yolo",
    "optical_flow_crush": "models.optical_flow_crush",
    "fall_yolo_pose": "models.fall.yolo_pose",
    "fall_mediapipe_pose": "models.fall.mediapipe_pose",
    "fall_alphapose_lstm": "models.fall.alphapose_lstm",
    "fall_stgcn": "models.fall.stgcn",
    "fall_posec3d": "models.fall.posec3d",
    "fall_movenet": "models.fall.movenet",
    "fall_optical_flow": "models.fall.optical_flow_fall",
    "violence_x3d": "models.violence.x3d",
    "violence_slowfast": "models.violence.slowfast",
    "violence_videomae": "models.violence.videomae",
    "violence_i3d": "models.violence.i3d",
    "violence_c3d": "models.violence.c3d",
    "violence_tsm": "models.violence.tsm",
    "violence_mmaction_slowonly": "models.violence.mmaction_slowonly",
}

# Each model's actual third-party import happens lazily inside load(), not
# at module import time — so we check the SPECIFIC libraries each model
# needs, not just whether the wrapper module imports cleanly.
EXTRA_LIB_CHECKS = {
    "fall_mediapipe_pose": ["mediapipe", "ultralytics"],
    "fall_alphapose_lstm": ["alphapose"],
    "fall_stgcn": ["ultralytics"],  # pyskl is optional (falls back to stand-in)
    "fall_posec3d": ["mmaction"],
    "fall_movenet": ["tensorflow_hub", "tensorflow"],
    "violence_videomae": ["transformers"],
    "violence_i3d": ["pytorchvideo"],
    "violence_mmaction_slowonly": ["mmaction"],
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


def main():
    print("Checking wrapper modules import cleanly (structural check)...\n")
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