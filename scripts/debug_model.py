"""
Loads ONE model and runs it on a single dummy frame (or frame pair) —
useful for seeing a model's real error message without waiting through
an entire video run or digging through scrollback.

Usage:
    python scripts/debug_model.py --model fall_movenet
    python scripts/debug_model.py --model fall_optical_flow
"""

import argparse
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from webapp import registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(registry.BY_KEY.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video_frame", default=None,
                         help="Path to a video file — reads its first frame instead of using a blank dummy.")
    args = parser.parse_args()

    model = registry.build_model(args.model, device=args.device)
    print(f"Loading {model.name} ...")
    model.load()
    print(f"Loaded. consumption_type={model.consumption_type}")

    parser_video = args
    if getattr(args, "video_frame", None):
        import cv2
        cap = cv2.VideoCapture(args.video_frame)
        ret, dummy = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read a frame from {args.video_frame}")
        print(f"Using a real frame from {args.video_frame} instead of a blank dummy.")
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    print(f"Calling predict() once with a dummy {model.consumption_type} input...")
    if args.model == "fall_movenet" and model.consumption_type == "frame":
        # Bypass predict()'s confidence filtering to see RAW scores
        import cv2
        import tensorflow as tf
        h, w = dummy.shape[:2]
        input_size = 256
        img = cv2.resize(dummy, (input_size, input_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = tf.cast(tf.expand_dims(img_rgb, axis=0), dtype=tf.int32)
        outputs = model._infer(input_tensor)
        people = outputs["output_0"].numpy()[0]
        print(f"Raw per-person confidence scores (threshold is {model.conf_threshold}):")
        for i, person in enumerate(people):
            print(f"  person {i}: conf={person[55]:.4f}")
        result = model.predict(dummy, frame_index=0, timestamp_sec=0.0)
    elif model.consumption_type == "frame":
        result = model.predict(dummy, frame_index=0, timestamp_sec=0.0)
    elif model.consumption_type == "flow_pair":
        result = model.predict((dummy, dummy), frame_index=0, timestamp_sec=0.0)
    elif model.consumption_type == "clip":
        result = model.predict([dummy] * 8, frame_index=0, timestamp_sec=0.0)
    else:
        raise ValueError(f"Unknown consumption_type: {model.consumption_type}")

    print(f"predict() succeeded. Returned {len(result)} detection(s).")


if __name__ == "__main__":
    main()