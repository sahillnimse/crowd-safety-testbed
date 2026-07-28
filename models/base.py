"""
Common interface for all model wrappers in this testbed.

Every model (fire/smoke YOLO, pose/fall, violence classifier, optical flow)
implements this interface so the pipeline runner can call any of them
identically without knowing internals.

Two consumption patterns exist:
  - FRAME-based models consume a single frame (np.ndarray, BGR, HWC).
  - CLIP-based models consume a list/array of consecutive frames.
  - FLOW-based models consume a pair of consecutive frames.

`predict()` normalizes all three into one call signature: it always
receives the full available context (single frame OR clip OR frame pair)
and returns a common Detection result list, so downstream code
(logging, annotation) doesn't need per-model branching.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

# NB: `resolve_device` is imported lazily inside __init__ rather than at
# module scope. `pipeline/__init__.py` imports the runner, which imports
# this module — so a top-level `from pipeline.device import ...` here makes
# `import models` fail outright with a circular-import error (importing
# `pipeline` first happened to work, which is why this went unnoticed).


@dataclass
class Detection:
    """Normalized output from any model in the stack."""
    model_name: str
    label: str                     # e.g. "fire", "fall", "violence", "turbulence"
    confidence: float               # 0.0 - 1.0
    timestamp_sec: float
    frame_index: int
    bbox: Optional[list] = None     # [x1, y1, x2, y2] if applicable
    keypoints: Optional[list] = None  # pose keypoints if applicable
    extra: dict = field(default_factory=dict)  # model-specific extras (e.g. flow field stats)


class BaseModelWrapper:
    """
    Subclass this for every model. Keeps the runner model-agnostic.

    consumption_type tells the pipeline runner what input this model needs:
      - "frame": single current frame
      - "clip":  a sliding window of N frames (see pipeline/frame_buffer.py)
      - "flow_pair": exactly two consecutive frames
    """

    consumption_type: str = "frame"  # override in subclass
    name: str = "base"
    gpu_accelerated: bool = True

    #: Frames a clip model needs before its first prediction is meaningful.
    #: The runner used to invoke every clip model as soon as 2 frames were
    #: buffered, and the model's own uniform sampling then duplicated those
    #: 2 images into a 16- or 64-frame "clip" — a degenerate input scored as
    #: if it were real. Clip models override this with their `clip_len`.
    clip_len: int = 1

    #: How often a clip model is re-run, in sampled frames. Running a
    #: 32-frame 3D-CNN on *every* frame means ~30 inferences per second of
    #: video whose inputs overlap by ~97%: enormous cost, near-zero added
    #: information. Default is half a clip, i.e. 50% overlap between
    #: consecutive inferences.
    @property
    def clip_stride(self) -> int:
        return max(1, self.clip_len // 2)

    @property
    def min_clip_frames(self) -> int:
        return self.clip_len

    def __init__(self, device: Optional[str] = None):
        from pipeline.device import resolve_device
        self.device = resolve_device(device)
        self._model = None  # loaded lazily in load()

    def load(self):
        """Load weights / initialize model. Called once before predict loop."""
        raise NotImplementedError

    def predict(self, data: Any, frame_index: int, timestamp_sec: float) -> list[Detection]:
        """
        data:
          - np.ndarray (H,W,3) if consumption_type == "frame"
          - list[np.ndarray] if consumption_type == "clip"
          - tuple(np.ndarray, np.ndarray) if consumption_type == "flow_pair"

        Returns list of Detection (can be empty if nothing detected).
        """
        raise NotImplementedError
