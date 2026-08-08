"""
Detector integration layer: vehicle and umbrella masks.

Detection is much slower than flow (~30-60 ms per detector vs ~5-10 ms for
DIS).  Running both detectors every frame would dominate the pipeline budget.
This module runs each detector every ``stride`` flow frames and carries the
last-seen bounding boxes forward between detections.

Vehicle masking
---------------
A moving vehicle produces large coherent flow vectors unrelated to crowd
pedestrian motion.  Masking detected vehicle boxes out of the pedestrian flow
aggregation prevents vehicles from inflating speed, divergence, and turbulence
statistics.  Separately, the presence of a vehicle inside a pedestrian zone at
high crowd density is flagged as its own alert (vehicle_in_ped_zone).

Umbrella masking
----------------
Umbrella canopies:
  - Occlude heads from overhead / oblique cameras, breaking density estimation.
  - Move independently of the person underneath (tilt, bob, wind rotation),
    injecting spurious motion vectors into the flow field.
  - Will be widespread during Shahi Snans (monsoon tail, Aug-Sep 2027).

This module tracks umbrella boxes and reports:
  - umbrella_mask(): binary mask of canopy regions
  - umbrella_occluded_fraction(): fraction of frame area under umbrella canopies
    (used by crowd_metrics.py to annotate density estimates as potentially low)

Detector loading
----------------
Both detectors use RTDetrV2ForObjectDetection.from_pretrained(<folder>), the
same interface as the user's existing trained models.  Vehicle class labels are
read from model.config.id2label rather than hard-coded, so the 14 UVH-26
classes are respected without any changes here.

If either model path is empty or missing, that detector is skipped and the
corresponding mask is zeroed (logged as WARNING at startup, not raised).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Confidence threshold for both detectors.
_DEFAULT_CONF = 0.35

# How many pixels to expand vehicle bounding boxes when building the mask.
# Gives a small safety margin around partial detections.
_DEFAULT_MARGIN_PX = 12


class DetectorMaskLayer:
    """
    Runs the two crowd-safety RT-DETRv2 detectors (vehicle + umbrella) every
    ``stride`` flow frames and exposes the resulting binary masks.

    Thread-safety: not thread-safe.  One instance per camera stream.
    """

    def __init__(
        self,
        vehicle_model_path: str = "",
        umbrella_model_path: str = "",
        device: str = "cpu",
        stride: int = 5,
        conf_threshold: float = _DEFAULT_CONF,
        vehicle_mask_margin_px: int = _DEFAULT_MARGIN_PX,
        umbrella_mask_margin_px: int = _DEFAULT_MARGIN_PX,
    ) -> None:
        self.vehicle_model_path  = vehicle_model_path
        self.umbrella_model_path = umbrella_model_path
        self.device              = device
        self.stride              = stride
        self.conf_threshold      = conf_threshold
        self.vehicle_margin      = vehicle_mask_margin_px
        self.umbrella_margin     = umbrella_mask_margin_px

        # Detector state (loaded in load())
        self._v_proc  = None
        self._v_model = None
        self._u_proc  = None
        self._u_model = None

        # Label maps
        self._vehicle_labels: dict[int, str] = {}  # populated from id2label
        self._umbrella_class_idx: int = 0           # single-class model → 0

        # Carry-forward state
        self._vehicle_boxes:  list[list[float]] = []
        self._umbrella_boxes: list[list[float]] = []

        # Track whether detectors actually loaded
        self._vehicle_loaded  = False
        self._umbrella_loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load both detectors.  Logs WARNING (not raises) on missing paths."""
        self._vehicle_loaded  = self._load_vehicle()
        self._umbrella_loaded = self._load_umbrella()

    def _load_vehicle(self) -> bool:
        if not self.vehicle_model_path:
            logger.warning(
                "vehicle_model_path is empty; vehicle masking is disabled.  "
                "Set crowd_flow.vehicle_model_path in configs/crowd_flow.yaml."
            )
            return False
        try:
            import torch
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

            logger.info("Loading vehicle detector from %s ...", self.vehicle_model_path)
            self._v_proc  = RTDetrImageProcessor.from_pretrained(self.vehicle_model_path)
            self._v_model = RTDetrV2ForObjectDetection.from_pretrained(self.vehicle_model_path)
            self._v_model = self._v_model.to(self.device)
            self._v_model.eval()

            # Read class labels from the model's own config so the 14 UVH-26
            # classes are respected without hard-coding them here.
            self._vehicle_labels = {
                int(k): v
                for k, v in self._v_model.config.id2label.items()
            }
            logger.info(
                "Vehicle detector loaded (%d classes: %s).",
                len(self._vehicle_labels),
                ", ".join(self._vehicle_labels.values()),
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to load vehicle detector from '%s': %s.  "
                "Vehicle masking is disabled for this run.",
                self.vehicle_model_path, exc,
            )
            return False

    def _load_umbrella(self) -> bool:
        if not self.umbrella_model_path:
            logger.warning(
                "umbrella_model_path is empty; umbrella masking is disabled.  "
                "Set crowd_flow.umbrella_model_path in configs/crowd_flow.yaml."
            )
            return False
        try:
            import torch
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

            logger.info("Loading umbrella detector from %s ...", self.umbrella_model_path)
            self._u_proc  = RTDetrImageProcessor.from_pretrained(self.umbrella_model_path)
            self._u_model = RTDetrV2ForObjectDetection.from_pretrained(self.umbrella_model_path)
            self._u_model = self._u_model.to(self.device)
            self._u_model.eval()
            # Single-class trained model: class 0 = umbrella
            self._umbrella_class_idx = 0
            logger.info("Umbrella detector loaded.")
            return True
        except Exception as exc:
            logger.warning(
                "Failed to load umbrella detector from '%s': %s.  "
                "Umbrella masking is disabled for this run.",
                self.umbrella_model_path, exc,
            )
            return False

    # ------------------------------------------------------------------
    # Update (called every frame by DenseFlowAnalyser)
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, frame_index: int) -> None:
        """
        Run detectors and refresh box state.

        Only actually runs inference every ``stride`` frames; carry-forward
        happens automatically on the intervening frames.
        """
        if frame_index % self.stride != 0:
            return

        if self._vehicle_loaded:
            self._vehicle_boxes = self._run_detector(
                frame, self._v_proc, self._v_model,
                class_filter=set(self._vehicle_labels.keys()),
            )

        if self._umbrella_loaded:
            self._umbrella_boxes = self._run_detector(
                frame, self._u_proc, self._u_model,
                class_filter={self._umbrella_class_idx},
            )

    def _run_detector(
        self,
        frame: np.ndarray,
        proc,
        model,
        class_filter: Optional[set] = None,
    ) -> list[list[float]]:
        """Run one RTDetrV2 detector and return [[x1,y1,x2,y2], ...] boxes."""
        import torch
        from PIL import Image

        h, w = frame.shape[:2]
        pil_img = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        inputs  = proc(images=pil_img, return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = proc.post_process_object_detection(
            outputs,
            threshold=self.conf_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes_out: list[list[float]] = []
        for box, label in zip(
            results["boxes"].cpu().tolist(),
            results["labels"].cpu().tolist(),
        ):
            if class_filter is None or int(label) in class_filter:
                boxes_out.append([float(v) for v in box])

        return boxes_out

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def vehicle_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Binary uint8 mask (H, W).  1 where vehicle bounding boxes (+ margin)
        are present, 0 elsewhere.  Zero mask if no vehicle detector is loaded.
        """
        return self._boxes_to_mask(self._vehicle_boxes, shape, self.vehicle_margin)

    def umbrella_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Binary uint8 mask (H, W).  1 where umbrella canopies are detected.
        Zero mask if no umbrella detector is loaded.
        """
        return self._boxes_to_mask(self._umbrella_boxes, shape, self.umbrella_margin)

    def pedestrian_flow_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Binary uint8 mask (H, W).  1 where pedestrian flow analysis should
        be performed (inverse of vehicle_mask).
        """
        v_mask = self.vehicle_mask(shape)
        result = np.ones(shape[:2], dtype=np.uint8)
        result[v_mask == 1] = 0
        return result

    def umbrella_occluded_fraction(self, shape: tuple[int, int]) -> float:
        """
        Fraction of the frame area covered by umbrella canopies (0.0 – 1.0).
        Used by crowd_metrics.py to annotate density estimates as potentially
        lower-bounded (occlusion → undercounting).
        """
        h, w = shape[:2]
        total = h * w
        if total == 0:
            return 0.0
        return float(self.umbrella_mask(shape).sum()) / total

    def vehicle_in_pedestrian_zone(
        self, zone_polygon: list[tuple[float, float]]
    ) -> bool:
        """
        True if any vehicle box centroid falls inside zone_polygon.

        Used by AlertEngine to flag "vehicle present in pedestrian zone".
        """
        if not self._vehicle_boxes or not zone_polygon:
            return False
        poly = np.array(zone_polygon, dtype=np.float32)
        for (x1, y1, x2, y2) in self._vehicle_boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vehicle_boxes(self) -> list[list[float]]:
        return list(self._vehicle_boxes)

    @property
    def umbrella_boxes(self) -> list[list[float]]:
        return list(self._umbrella_boxes)

    @property
    def vehicle_count(self) -> int:
        return len(self._vehicle_boxes)

    @property
    def umbrella_count(self) -> int:
        return len(self._umbrella_boxes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _boxes_to_mask(
        boxes: list[list[float]],
        shape: tuple[int, int],
        margin: int,
    ) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x1, y1, x2, y2) in boxes:
            xi1 = max(0, int(x1) - margin)
            yi1 = max(0, int(y1) - margin)
            xi2 = min(w, int(x2) + margin)
            yi2 = min(h, int(y2) + margin)
            mask[yi1:yi2, xi1:xi2] = 1
        return mask
