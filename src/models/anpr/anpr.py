"""
ANPR: vehicle capture with number-plate reading.

Pipeline per frame:

    RT-DETRv2 vehicle detect + shared IoU tracker
        -> crop each tracked vehicle
        -> DETR plate detection inside that crop
        -> OCR the plate, correct against the Indian plate format
        -> vote the reading across every frame the vehicle appears in
        -> keep the sharpest, largest view of the vehicle as its portrait

and on `finalize()` a gallery is written: one image per vehicle plus a
manifest with plate, class, colour and timing.

**Why per-vehicle rather than per-frame.** A vehicle is visible for dozens
of frames at wildly varying quality — small and blurred on approach, large
and sharp alongside, then gone. Reading the plate once and reporting it is
a coin flip on which frame you happened to catch. Instead every frame's
reading feeds a vote (models/anpr/_plate_text.py), and the saved portrait
is chosen by a sharpness x size score rather than being whichever frame
came last. Both are what makes the difference between an ANPR system that
works and one that emits a stream of nearly-right plates.

**Known limitation, and it is the binding one.** Plates must be physically
large in the frame. On this repo's own traffic clip they peak at 60x18 px
and read as nothing; the detector finds them fine, but the characters are
about 8 px tall and no OCR recovers that. Vehicles are reported regardless,
with `plate_status` explaining why the plate is absent, so a run on
unsuitable footage is legible as such instead of looking like a bug. For
readable plates you need the camera close to the traffic lane and pointed
along it — roughly 100 px of plate width or more.
"""

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
from models.traffic._tracker import ParkedMovingClassifier

_VEHICLE_COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_GALLERY_DIR = os.path.join(PROJECT_ROOT, "outputs", "anpr")


def _sharpness(img) -> float:
    """Variance of Laplacian — the standard cheap focus measure."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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
    # Every OCR call made on this vehicle, whatever came back. Distinct from
    # PlateVote.n_reads, which only counts attempts that produced text — so
    # reporting only the latter made a run that gated 47 plates as too small
    # look like it had never tried to read any of them.
    ocr_attempts: int = 0
    plate_detections: int = 0


class ANPRDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "anpr"
    gpu_accelerated = True

    def __init__(self, conf_threshold: float = 0.35,
                 plate_conf: float = 0.5,
                 min_plate_width: int = DEFAULT_MIN_PLATE_WIDTH,
                 read_every_n_frames: int = 3,
                 gallery_dir: str = None, video_name: str = "run",
                 save_gallery: bool = True, ocr_backend: str = "easyocr", device=None):
        # No `weights` parameter: the vehicle detector is the shared
        # RT-DETRv2 in models/_detectors.py, which owns its own checkpoint.
        # The old parameter survived the YOLO removal as an empty string that
        # nothing read.
        super().__init__(device=device)
        self.conf_threshold = conf_threshold
        self.plate_conf = plate_conf
        self.min_plate_width = min_plate_width
        # Plate detection is the expensive stage. A vehicle stays in frame
        # for many frames, so reading every Nth is plenty for voting and
        # keeps the run tractable.
        self.read_every_n_frames = max(1, read_every_n_frames)
        self.gallery_dir = gallery_dir or DEFAULT_GALLERY_DIR
        self.video_name = video_name
        self.save_gallery = save_gallery
        self.ocr_backend = ocr_backend

        self._records: dict[int, _VehicleRecord] = {}
        self._classifier = ParkedMovingClassifier(model_name=self.name)
        self._plate_detector = PlateDetector(conf_threshold=plate_conf)
        self._ocr = get_ocr_engine(ocr_backend, min_plate_width=min_plate_width)
        # RT-DETRv2 is a detector, not a tracker.  ANPR needs stable per-vehicle
        # identity to accumulate plate votes and pick a best portrait, so
        # association is done with the project's shared IoU tracker.
        from models._tracker import IoUTracker
        self._tracker = IoUTracker(iou_threshold=0.3, max_age=30)

    def load(self):
        from models._detectors import get_detector
        self._model = get_detector(device=self.device)
        self._model.load()
        self._tracker.reset()

        self._plate_detector.device = self.device
        self._plate_detector.load()
        self._ocr.use_gpu = str(self.device).startswith("cuda")
        self._ocr.load()

        self._records = {}
        self._classifier.reset()

    # ------------------------------------------------------------------
    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        raw = self._model.detect_with_labels(
            frame, classes=tuple(_VEHICLE_COCO_CLASSES),
            conf_threshold=self.conf_threshold,
        )
        if not raw:
            return []

        det_boxes = [b for b, _c, _s in raw]
        track_ids = self._tracker.update(det_boxes, frame_index)

        h, w = frame.shape[:2]
        read_this_frame = (frame_index % self.read_every_n_frames == 0)
        detections = []

        for (box, cls_id, conf), tid in zip(raw, track_ids, strict=False):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue

            track_id = int(tid)
            crop = frame[y1:y2, x1:x2]
            rec = self._records.get(track_id)
            if rec is None:
                rec = _VehicleRecord(track_id=track_id, first_sec=timestamp_sec)
                self._records[track_id] = rec

            rec.vehicle_class = _VEHICLE_COCO_CLASSES.get(int(cls_id), "vehicle")
            rec.last_sec = timestamp_sec
            rec.n_frames += 1

            self._update_portrait(rec, crop)

            if read_this_frame:
                self._read_plate(rec, crop)

            plate, agreement = rec.votes.result()
            detections.append(Detection(
                model_name=self.name,
                label="vehicle_plate" if plate else "vehicle_unread",
                confidence=float(conf),
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                bbox=[x1, y1, x2, y2],
                extra={
                    "track_id": track_id,
                    "vehicle_class": rec.vehicle_class,
                    "colour": rec.colour,
                    "plate": plate,
                    "plate_display": format_display(plate) if plate else None,
                    "plate_agreement": round(agreement, 3),
                    "plate_status": rec.plate_status,
                    "plate_width_px": rec.best_plate_width,
                },
            ))

        return detections

    # ------------------------------------------------------------------
    def _update_portrait(self, rec: _VehicleRecord, crop) -> None:
        """Keep the best view of this vehicle for the gallery.

        Score is area x sharpness: a large blurred crop and a small sharp
        one are both poor portraits, and either alone picks bad frames.
        """
        area = crop.shape[0] * crop.shape[1]
        score = area * (1.0 + _sharpness(crop))
        if score > rec.best_score:
            rec.best_score = score
            rec.best_image = crop.copy()
            rec.colour = dominant_colour(crop)

    def _read_plate(self, rec: _VehicleRecord, crop) -> None:
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
        # Don't let a later, worse frame overwrite a successful read.
        if rec.plate_status != "ok":
            rec.plate_status = status
        if status == "ok" and text:
            rec.votes.add(text, conf)
            if rec.votes.readings:
                rec.plate_status = "ok"

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        """Write the gallery. Called by the runner after the last frame."""
        summary = {
            "video": self.video_name,
            "vehicles": [],
            "counts": {"total": 0, "with_plate": 0, "too_small": 0,
                       "unreadable": 0, "no_plate_found": 0},
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
                # The caption the gallery renders under the photo, e.g.
                # "white car - MH 15 DS 7121".
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
            with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        c = summary["counts"]
        print(f"[{self.name}] {c['total']} vehicle(s) captured; "
              f"{c['with_plate']} with a readable plate. "
              f"too_small={c.get('too_small', 0)} "
              f"unreadable={c.get('unreadable', 0)} "
              f"no_plate_found={c.get('no_plate_found', 0)}")
        if c["total"] and not c["with_plate"]:
            widths = [v["plate_width_px"] for v in summary["vehicles"]
                      if v["plate_width_px"]]
            if widths:
                print(f"[{self.name}] No plate was readable. Widest plate seen was "
                      f"{max(widths)} px; roughly {self.min_plate_width}+ px is "
                      f"needed. This footage is too wide/distant for ANPR - the "
                      f"characters are not resolvable, regardless of model.")
        return summary
