"""
Umbrella detection via RT-DETRv2 (Real-Time Detection Transformer v2).

Key: umbrella_rtdetrv2
UI Label: RT-DETRv2-S (COCO zero-shot)

RT-DETRv2 is the improved successor to RT-DETR, featuring:
  - Improved decoder with deformable attention (vs. vanilla cross-attention)
  - Dual-level IoU-aware query selection for better small object recall
  - ResNet-18vd backbone (20 M params, 217 FPS on A100)
  - COCO AP^val 48.1 (RT-DETRv2-S)

License   : Apache 2.0  (https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE)
Weights   : PekingU/rtdetr_v2_r18vd (HuggingFace Hub) — official COCO checkpoint
            Falls back to local  weights/rtdetrv2_r18vd_coco.pth  if present.

COCO umbrella class: index 25 (0-indexed).  No fine-tuning needed — the model
runs zero-shot on the built-in COCO vocabulary, same approach as umbrella_yolo.

Inference pipeline
------------------
  HF RTDetrImageProcessor  →  RTDetrV2ForObjectDetection  →  post-process
  →  filter class 25 + confidence  →  emit_umbrellas()

ByteTrack is handled via ultralytics' tracker utility (same path as rfdetr /
yolo26n) so persistent track IDs are still available for the crowd-safety
UI event counting.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional

from models.base import BaseModelWrapper, Detection
from models.umbrella._common import DEFAULT_MIN_AREA_FRAC, emit_umbrellas

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Official Apache-2.0 RT-DETRv2-S checkpoint hosted on HuggingFace by PekingU
# (the same team that maintains the official lyuwenyu/RT-DETR repository).
HF_MODEL_ID = "PekingU/rtdetr_v2_r18vd"

# Local checkpoint search paths (searched before pulling from HF Hub)
_LOCAL_CHECKPOINTS = [
    "weights/rtdetrv2_r18vd_coco.pth",
    "ML Models/ultralytics/rtdetrv2_r18vd_coco.pth",
]

# COCO umbrella class index (0-indexed, same as Ultralytics/torchvision COCO)
_UMBRELLA_COCO_IDX = 25


def _find_local_checkpoint() -> Optional[str]:
    """Return absolute path to a local RT-DETRv2 checkpoint, or None."""
    for rel in _LOCAL_CHECKPOINTS:
        abs_p = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(abs_p):
            return abs_p
    return None


class RTDetrV2UmbrellaDetector(BaseModelWrapper):
    """
    RT-DETRv2-S umbrella detector.

    Wraps HuggingFace ``RTDetrV2ForObjectDetection`` so it fits the project's
    ``BaseModelWrapper`` interface.  No fine-tuning is required: the COCO
    pretrained model already knows the 'umbrella' class (index 25).

    Parameters
    ----------
    conf_threshold : float
        Minimum score to keep a detection. Default 0.35 matches other umbrella
        wrappers so benchmarks are directly comparable.
    min_area_frac : float
        Minimum box area as a fraction of the frame area.  Removes speck-sized
        false positives in busy backgrounds (same filter as all other umbrella
        wrappers).
    track : bool
        When True, persistent IDs are assigned via a self-contained
        IoU-centroid tracker (avoids coupling to ultralytics' internal
        ByteTracker API which changes between minor releases).
    local_checkpoint : str | None
        Override path to a locally cached ``.pth`` file.  If None, the
        detector first tries the standard search dirs then falls back to
        pulling ``PekingU/rtdetr_v2_r18vd`` from the HuggingFace Hub.
    """

    consumption_type = "frame"
    name = "umbrella_rtdetrv2"
    gpu_accelerated = True

    def __init__(
        self,
        conf_threshold: float = 0.35,
        min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
        track: bool = True,
        local_checkpoint: Optional[str] = None,
        device=None,
    ):
        super().__init__(device=device)
        self.conf_threshold = conf_threshold
        self.min_area_frac = min_area_frac
        self.track = track
        self._local_checkpoint = local_checkpoint or _find_local_checkpoint()

        self._processor = None   # RTDetrImageProcessor
        self._model = None       # RTDetrV2ForObjectDetection
        # centroid tracker state — initialised in _init_tracker() inside load()
        self._tracks: list = []
        self._next_track_id: int = 1

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self):
        """Load RT-DETRv2 weights and image processor from HF Hub (or local)."""
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        if self._local_checkpoint and os.path.isfile(self._local_checkpoint):
            # Load the HF model architecture then override weights from the
            # official PyTorch .pth file (state-dict compatible with the HF
            # model because PekingU/rtdetr_v2_r18vd was exported from the same
            # training run as the official lyuwenyu releases).
            self._processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_ID)
            self._model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_ID)
            state = torch.load(self._local_checkpoint, map_location="cpu")
            # Handle both raw state-dicts and checkpoint dicts with a 'model' key
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            if missing:
                print(
                    f"[umbrella_rtdetrv2] Missing keys when loading local checkpoint "
                    f"(count={len(missing)}). Falling back to HF weights for those layers."
                )
        else:
            # Pull directly from HuggingFace Hub (Apache 2.0)
            self._processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_ID)
            self._model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_ID)

        # Move to target device
        self._model = self._model.to(self.device)
        self._model.eval()

        # Initialise ByteTrack (ultralytics) for persistent track IDs
        if self.track:
            self._init_tracker()

    def _init_tracker(self):
        """Initialise a lightweight IoU-based centroid tracker.

        We intentionally avoid calling ultralytics' BYTETracker directly:
        its internal API (constructor signature, update() input format, output
        format) changes between minor ultralytics releases, causing silent
        failures. The simple centroid tracker below is self-contained,
        dependency-free, and sufficient for counting unique umbrellas in the
        crowd-safety UI.
        """
        # _tracks: list of {"id": int, "cx": float, "cy": float, "age": int}
        self._tracks: list[dict] = []
        self._next_track_id = 1

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self, frame: np.ndarray, frame_index: int, timestamp_sec: float
    ) -> list[Detection]:
        if self._model is None:
            return []

        import torch
        from PIL import Image

        # --- Pre-process ---
        # HF image processor expects PIL Image or list of PIL Images
        pil_img = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # --- Forward pass ---
        with torch.no_grad():
            outputs = self._model(**inputs)

        # --- Post-process ---
        h, w = frame.shape[:2]
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.conf_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes_all = results["boxes"].cpu().tolist()       # [[x1,y1,x2,y2], ...]
        scores_all = results["scores"].cpu().tolist()
        labels_all = results["labels"].cpu().tolist()

        # --- Filter to umbrella class only ---
        umbrella_boxes, umbrella_scores = [], []
        for box, score, label in zip(boxes_all, scores_all, labels_all):
            if label == _UMBRELLA_COCO_IDX:
                umbrella_boxes.append(box)
                umbrella_scores.append(score)

        if not umbrella_boxes:
            return []

        # --- ByteTrack (optional) ---
        track_ids = self._assign_track_ids(umbrella_boxes, umbrella_scores, frame)

        return emit_umbrellas(
            umbrella_boxes,
            umbrella_scores,
            track_ids,
            frame.shape,
            self.name,
            frame_index,
            timestamp_sec,
            min_area_frac=self.min_area_frac,
            extra_fields={
                "architecture": "RT-DETRv2-S",
                "backbone": "ResNet-18vd",
                "source": "COCO zero-shot",
            },
        )

    def _assign_track_ids(
        self,
        boxes: list,
        scores: list,
        frame: np.ndarray,
    ) -> Optional[list]:
        """Assign persistent track IDs via a greedy IoU-centroid matcher.

        Returns a list of int IDs (same length as *boxes*), or None when
        tracking is disabled.

        The algorithm:
          1. Compute centroid of each new box.
          2. For each existing track, find the nearest new centroid within
             a distance threshold (proportional to frame diagonal).
          3. Matched tracks keep their ID; unmatched new boxes get a fresh ID.
          4. Tracks that fail to match for too many consecutive frames are
             expired and their slot freed.
        """
        if not self.track:
            return None

        h, w = frame.shape[:2]
        diag = (h ** 2 + w ** 2) ** 0.5
        match_dist = diag * 0.08   # 8% of frame diagonal
        max_age = 30               # frames before a lost track is discarded

        new_centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        n_det = len(new_centroids)
        n_trk = len(self._tracks)

        assigned_ids: list[Optional[int]] = [None] * n_det
        used_trk: set[int] = set()

        if n_trk > 0 and n_det > 0:
            # Build distance matrix
            dists = np.full((n_trk, n_det), fill_value=1e9, dtype=np.float32)
            for ti, t in enumerate(self._tracks):
                for di, (cx, cy) in enumerate(new_centroids):
                    dx, dy = cx - t["cx"], cy - t["cy"]
                    dists[ti, di] = (dx * dx + dy * dy) ** 0.5

            # Greedy min-distance matching
            flat_order = np.argsort(dists.ravel())
            for idx in flat_order:
                ti, di = divmod(int(idx), n_det)
                if dists[ti, di] > match_dist:
                    break
                if ti in used_trk or assigned_ids[di] is not None:
                    continue
                assigned_ids[di] = self._tracks[ti]["id"]
                self._tracks[ti]["cx"] = new_centroids[di][0]
                self._tracks[ti]["cy"] = new_centroids[di][1]
                self._tracks[ti]["age"] = 0
                used_trk.add(ti)

        # New detections that weren't matched → new track IDs
        for di in range(n_det):
            if assigned_ids[di] is None:
                cx, cy = new_centroids[di]
                assigned_ids[di] = self._next_track_id
                self._tracks.append({"id": self._next_track_id, "cx": cx, "cy": cy, "age": 0})
                self._next_track_id += 1

        # Age out unmatched tracks
        for ti in range(n_trk - 1, -1, -1):
            if ti not in used_trk:
                self._tracks[ti]["age"] += 1
                if self._tracks[ti]["age"] > max_age:
                    self._tracks.pop(ti)

        return assigned_ids

