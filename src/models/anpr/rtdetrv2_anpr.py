"""
RT-DETRv2 ANPR: vehicle detection, vehicle class profiling, car colour recognition,
and license plate reading using RT-DETRv2 (Apache 2.0).

Key: rtdetrv2_anpr
UI Label: RT-DETRv2 ANPR (Classification, Color & Plate)

Pipeline per frame:
    1. RT-DETRv2 vehicle detection (COCO classes: car, motorcycle, bus, truck)
    2. Centroid tracking across frames for stable vehicle IDs
    3. Crop each vehicle → extract dominant vehicle colour (HSV core sampling)
    4. DETR plate locator on vehicle crop → crop plate
    5. Plate OCR (RapidOCR ONNX or EasyOCR) → format correction & voting
    6. Sharpness x size scoring to maintain best vehicle portrait

On `finalize()`, exports gallery manifest with vehicle classification, colour,
and plate text.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from models.base import BaseModelWrapper, Detection
from models.anpr._ocr import (
    DEFAULT_MIN_PLATE_WIDTH,
    PlateDetector,
    dominant_colour,
    get_ocr_engine,
)
from models.anpr._plate_text import PlateVote, format_display

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_GALLERY_DIR = os.path.join(PROJECT_ROOT, "outputs", "anpr")

# HF HuggingFace Hub checkpoint (Apache 2.0)
HF_MODEL_ID = "PekingU/rtdetr_v2_r18vd"

# Local checkpoint search paths
_LOCAL_CHECKPOINTS = [
    "weights/rtdetrv2_r18vd_coco.pth",
    "ML Models/ultralytics/rtdetrv2_r18vd_coco.pth",
]

# COCO vehicle classes mapping
_VEHICLE_COCO_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _sharpness(img: np.ndarray) -> float:
    """Variance of Laplacian focus measure."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _find_local_checkpoint() -> Optional[str]:
    """Return absolute path to a local RT-DETRv2 checkpoint, if present."""
    for rel in _LOCAL_CHECKPOINTS:
        abs_p = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(abs_p):
            return abs_p
    return None


@dataclass
class _VehicleRecord:
    track_id: int
    vehicle_class: str = "vehicle"
    colour: str = "unknown"
    first_sec: float = 0.0
    last_sec: float = 0.0
    votes: PlateVote = field(default_factory=PlateVote)
    best_score: float = -1.0
    best_image: Optional[np.ndarray] = None
    best_plate_image: Optional[np.ndarray] = None
    best_plate_width: int = 0
    plate_status: str = "no_plate_found"
    n_frames: int = 0
    ocr_attempts: int = 0
    plate_detections: int = 0


class RTDetrV2ANPRDetector(BaseModelWrapper):
    """
    ANPR & Vehicle Profiling pipeline powered by RT-DETRv2-S.

    Combines:
      - RT-DETRv2 object detector for high-accuracy vehicle detection & classification
      - HSV core sampling for car colour recognition (white, black, red, blue, etc.)
      - DETR license plate locator on vehicle crops
      - Swappable OCR engine (RapidOCR / EasyOCR) with multi-frame voting
      - Sharpness-based gallery portrait extraction
    """

    consumption_type = "frame"
    name = "rtdetrv2_anpr"
    gpu_accelerated = True

    def __init__(
        self,
        conf_threshold: float = 0.35,
        plate_conf: float = 0.5,
        min_plate_width: int = DEFAULT_MIN_PLATE_WIDTH,
        read_every_n_frames: int = 3,
        gallery_dir: Optional[str] = None,
        video_name: str = "run",
        save_gallery: bool = True,
        ocr_backend: str = "rapidocr",
        local_checkpoint: Optional[str] = None,
        device=None,
    ):
        super().__init__(device=device)
        self.conf_threshold = conf_threshold
        self.plate_conf = plate_conf
        self.min_plate_width = min_plate_width
        self.read_every_n_frames = max(1, read_every_n_frames)
        self.gallery_dir = gallery_dir or DEFAULT_GALLERY_DIR
        self.video_name = video_name
        self.save_gallery = save_gallery
        self.ocr_backend = ocr_backend
        self._local_checkpoint = local_checkpoint or _find_local_checkpoint()

        self._processor = None
        self._model = None
        self._records: dict[int, _VehicleRecord] = {}
        self._plate_detector = PlateDetector(conf_threshold=plate_conf)
        self._ocr = get_ocr_engine(ocr_backend, min_plate_width=min_plate_width)

        # Centroid tracker state
        self._tracks: list[dict] = []
        self._next_track_id: int = 1

    def load(self):
        """Load RT-DETRv2 vehicle detector, DETR plate detector, and OCR engine."""
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        if self._local_checkpoint and os.path.isfile(self._local_checkpoint):
            self._processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_ID)
            self._model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_ID)
            state = torch.load(self._local_checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self._model.load_state_dict(state, strict=False)
        else:
            self._processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_ID)
            self._model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_ID)

        self._model = self._model.to(self.device)
        self._model.eval()

        self._plate_detector.device = str(self.device)
        self._plate_detector.load()

        self._ocr.use_gpu = str(self.device).startswith("cuda")
        self._ocr.load()

        self._records = {}
        self._tracks = []
        self._next_track_id = 1

    def predict(
        self, frame: np.ndarray, frame_index: int, timestamp_sec: float
    ) -> list[Detection]:
        if self._model is None:
            return []

        import torch
        from PIL import Image

        h, w = frame.shape[:2]

        # --- Forward pass via RT-DETRv2 ---
        pil_img = Image.fromarray(frame[:, :, ::-1])
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=self.conf_threshold, target_sizes=target_sizes
        )[0]

        boxes_all = results["boxes"].cpu().tolist()
        scores_all = results["scores"].cpu().tolist()
        labels_all = results["labels"].cpu().tolist()

        # Filter vehicle classes (car, motorcycle, bus, truck)
        v_boxes, v_scores, v_classes = [], [], []
        for box, score, label in zip(boxes_all, scores_all, labels_all, strict=False):
            if label in _VEHICLE_COCO_CLASSES:
                v_boxes.append(box)
                v_scores.append(score)
                v_classes.append(_VEHICLE_COCO_CLASSES[label])

        if not v_boxes:
            return []

        # --- Track vehicles ---
        track_ids = self._assign_track_ids(v_boxes, frame.shape)
        read_this_frame = (frame_index % self.read_every_n_frames == 0)
        detections = []

        for box, conf, v_cls, tid in zip(v_boxes, v_scores, v_classes, track_ids, strict=False):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue

            crop = frame[y1:y2, x1:x2]
            rec = self._records.get(tid)
            if rec is None:
                rec = _VehicleRecord(track_id=tid, first_sec=timestamp_sec)
                self._records[tid] = rec

            rec.vehicle_class = v_cls
            rec.last_sec = timestamp_sec
            rec.n_frames += 1

            self._update_portrait(rec, crop)

            if read_this_frame:
                self._read_plate(rec, crop)

            plate, agreement = rec.votes.result()
            detections.append(
                Detection(
                    model_name=self.name,
                    label="vehicle_plate" if plate else "vehicle_unread",
                    confidence=float(conf),
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    bbox=[x1, y1, x2, y2],
                    extra={
                        "track_id": tid,
                        "vehicle_class": rec.vehicle_class,
                        "colour": rec.colour,
                        "plate": plate,
                        "plate_display": format_display(plate) if plate else None,
                        "plate_agreement": round(agreement, 3),
                        "plate_status": rec.plate_status,
                        "plate_width_px": rec.best_plate_width,
                        "architecture": "RT-DETRv2-S",
                    },
                )
            )

        return detections

    def _assign_track_ids(self, boxes: list, frame_shape: tuple) -> list[int]:
        """Greedy centroid distance tracking across frames."""
        h, w = frame_shape[:2]
        diag = (h ** 2 + w ** 2) ** 0.5
        match_dist = diag * 0.10
        max_age = 30

        centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        n_det = len(centroids)
        n_trk = len(self._tracks)

        assigned: list[Optional[int]] = [None] * n_det
        used_trk: set[int] = set()

        if n_trk > 0 and n_det > 0:
            dists = np.full((n_trk, n_det), fill_value=1e9, dtype=np.float32)
            for ti, t in enumerate(self._tracks):
                for di, (cx, cy) in enumerate(centroids):
                    dx, dy = cx - t["cx"], cy - t["cy"]
                    dists[ti, di] = (dx * dx + dy * dy) ** 0.5

            flat_order = np.argsort(dists.ravel())
            for idx in flat_order:
                ti, di = divmod(int(idx), n_det)
                if dists[ti, di] > match_dist:
                    break
                if ti in used_trk or assigned[di] is not None:
                    continue
                assigned[di] = self._tracks[ti]["id"]
                self._tracks[ti]["cx"] = centroids[di][0]
                self._tracks[ti]["cy"] = centroids[di][1]
                self._tracks[ti]["age"] = 0
                used_trk.add(ti)

        for di in range(n_det):
            if assigned[di] is None:
                cx, cy = centroids[di]
                tid = self._next_track_id
                assigned[di] = tid
                self._tracks.append({"id": tid, "cx": cx, "cy": cy, "age": 0})
                self._next_track_id += 1

        for ti in range(n_trk - 1, -1, -1):
            if ti not in used_trk:
                self._tracks[ti]["age"] += 1
                if self._tracks[ti]["age"] > max_age:
                    self._tracks.pop(ti)

        return assigned

    def _update_portrait(self, rec: _VehicleRecord, crop: np.ndarray) -> None:
        """Update vehicle portrait and extract dominant vehicle colour."""
        area = crop.shape[0] * crop.shape[1]
        score = area * (1.0 + _sharpness(crop))
        if score > rec.best_score:
            rec.best_score = score
            rec.best_image = crop.copy()
            rec.colour = dominant_colour(crop)

    def _read_plate(self, rec: _VehicleRecord, crop: np.ndarray) -> None:
        """Detect plate in crop and read text."""
        plates = self._plate_detector.detect(crop)
        if not plates:
            if rec.plate_status == "no_plate_found":
                rec.plate_status = "no_plate_found"
            return

        px1, py1, px2, py2, _score = plates[0]
        px1, py1 = max(0, px1), max(0, py1)
        px2 = min(crop.shape[1], px2)
        py2 = min(crop.shape[0], py2)
        if px2 - px1 < 4 or py2 - py1 < 3:
            return

        rec.plate_detections += 1
        plate_img = crop[py1:py2, px1:px2]
        width = plate_img.shape[1]
        if width > rec.best_plate_width:
            rec.best_plate_width = width
            rec.best_plate_image = plate_img.copy()

        rec.ocr_attempts += 1
        text, conf, status = self._ocr.read(plate_img)
        if rec.plate_status != "ok":
            rec.plate_status = status
        if status == "ok" and text:
            rec.votes.add(text, conf)
            if rec.votes.readings:
                rec.plate_status = "ok"

    def finalize(self) -> dict:
        """Export ANPR vehicle gallery and JSON manifest."""
        summary = {
            "video": self.video_name,
            "vehicles": [],
            "counts": {
                "total": 0,
                "with_plate": 0,
                "too_small": 0,
                "unreadable": 0,
                "no_plate_found": 0,
            },
        }
        if not self._records:
            return summary

        out_dir = os.path.join(self.gallery_dir, self.video_name)
        if self.save_gallery:
            os.makedirs(os.path.join(out_dir, "vehicles"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "plates"), exist_ok=True)

        for tid, rec in sorted(self._records.items()):
            plate, agreement = rec.votes.result()
            status = "ok" if plate else rec.plate_status

            entry = {
                "track_id": tid,
                "vehicle_class": rec.vehicle_class,
                "colour": rec.colour,
                "caption": f"{rec.colour} {rec.vehicle_class}",
                "plate": plate,
                "plate_display": format_display(plate) if plate else None,
                "plate_agreement": round(agreement, 3),
                "plate_status": status,
                "plate_width_px": rec.best_plate_width,
                "first_seen_sec": round(rec.first_sec, 2),
                "last_seen_sec": round(rec.last_sec, 2),
                "frames_seen": rec.n_frames,
                "plate_detections": rec.plate_detections,
                "ocr_attempts": rec.ocr_attempts,
                "reads_with_text": rec.votes.n_reads,
                "image": None,
                "plate_image": None,
            }

            if self.save_gallery and rec.best_image is not None:
                fn = f"vehicle_{tid:04d}.jpg"
                cv2.imwrite(os.path.join(out_dir, "vehicles", fn), rec.best_image)
                entry["image"] = fn
            if self.save_gallery and rec.best_plate_image is not None:
                fn = f"plate_{tid:04d}.jpg"
                cv2.imwrite(os.path.join(out_dir, "plates", fn), rec.best_plate_image)
                entry["plate_image"] = fn

            summary["vehicles"].append(entry)
            summary["counts"]["total"] += 1
            key = "with_plate" if plate else status
            summary["counts"][key] = summary["counts"].get(key, 0) + 1

        if self.save_gallery:
            with open(
                os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(summary, f, indent=2)

        return summary
