"""
Indian ANPR: three-stage pipeline optimised for Indian traffic footage.

Stage 1 – Vehicle detection:   Roboflow traffic-indian-vehicles/4
           Trained on Indian vehicle types (auto-rickshaw, truck, bus, car,
           motorcycle, tempo, etc.) — more reliable than COCO-YOLO on Indian
           roads where 3-wheelers and tempos are common but absent from COCO.
           Runs on Roboflow's servers, so no local weight download.

Stage 2 – Plate localisation:  Roboflow license-plate-recognition-rxg4e/4
           Workspace model fine-tuned on licence plates. Run inside each
           vehicle crop (not the whole frame) for precision and to get a
           per-vehicle plate owner without a separate association step.
           Also runs on Roboflow's servers — zero local download.

Stage 3 – OCR:                 EasyOCR (already a project dependency, ~50 MB)
           Restricted to the plate alphabet. Followed by the same
           Indian-plate format correction and multi-frame voting that the
           existing ANPRDetector uses (_plate_text.py).

Tracking: DeepSORT (already installed via rtdetr_traffic.py) gives persistent
          vehicle IDs so plate reads from multiple frames can be voted and
          the best portrait frame saved.

Total new local download: ~50 MB (EasyOCR recognition model), well under
the 60 MB budget. Both Roboflow models are hosted inference.

Output per frame:
    Detection(
        label   = "vehicle_plate" | "vehicle_unread",
        bbox    = [x1, y1, x2, y2],   # vehicle bounding box
        extra   = {
            "track_id":       int,
            "vehicle_class":  str,     # Roboflow class, e.g. "auto", "car"
            "colour":         str,     # dominant colour heuristic
            "plate":          str | None,         # best voted reading
            "plate_display":  str | None,         # formatted "MH 15 DS 7121"
            "plate_agreement": float,             # 0-1 vote agreement
            "plate_status":   str,                # ok | too_small | unreadable | no_plate_found
            "plate_width_px": int,
        }
    )

Gallery: same manifest.json + vehicle/plate JPEG layout as ANPRDetector.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from models.base import BaseModelWrapper, Detection
from models.anpr._ocr import enhance_plate, dominant_colour, PLATE_ALPHABET, get_ocr_engine, PlateOCR
from models.anpr._plate_text import PlateVote, format_display
from models.traffic._tracker import _stable_track_id

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_GALLERY_DIR = os.path.join(PROJECT_ROOT, "outputs", "anpr")

# Indian plates are generally smaller in frame than European ones (narrower
# lanes, more distant cameras). Drop the minimum to 60 px — still above
# the absolute floor where characters are recoverable at all.
_DEFAULT_MIN_PLATE_WIDTH = 60


def _sharpness(img) -> float:
    """Variance of Laplacian — standard cheap focus measure."""
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
    ocr_attempts: int = 0
    plate_detections: int = 0


class IndianANPRDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "indian_anpr"
    # Roboflow inference runs on Roboflow's servers; EasyOCR runs locally.
    gpu_accelerated = False

    def __init__(self,
                 vehicle_model_id: str = "traffic-indian-vehicles/4",
                 plate_model_id: str = "license-plate-recognition-rxg4e/4",
                 api_key: str = None,
                 conf_threshold: float = 0.40,
                 plate_conf: float = 0.35,
                 min_plate_width: int = _DEFAULT_MIN_PLATE_WIDTH,
                 read_every_n_frames: int = 3,
                 gallery_dir: str = None,
                 video_name: str = "run",
                 save_gallery: bool = True,
                 ocr_backend: str = "easyocr",
                 device=None):
        super().__init__(device=device)
        self.vehicle_model_id = vehicle_model_id
        self.plate_model_id = plate_model_id
        self.api_key = (
            api_key
            or os.environ.get("ROBOFLOW_API_KEY")
        )
        self.conf_threshold = conf_threshold
        self.plate_conf = plate_conf
        self.min_plate_width = min_plate_width
        self.read_every_n_frames = max(1, read_every_n_frames)
        self.gallery_dir = gallery_dir or DEFAULT_GALLERY_DIR
        self.video_name = video_name
        self.save_gallery = save_gallery
        self.ocr_backend = ocr_backend

        self._records: dict[int, _VehicleRecord] = {}
        self._rf_client = None
        self._tracker = None
        self._ocr = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    def load(self):
        from inference_sdk import InferenceHTTPClient
        self._rf_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.api_key,
        )

        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(max_age=30)

        use_gpu = str(self.device).startswith("cuda")
        self._ocr = get_ocr_engine(self.ocr_backend, use_gpu=use_gpu, min_plate_width=self.min_plate_width)
        self._ocr.load()

        self._records = {}
        self._frame_count = 0

    # ------------------------------------------------------------------
    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        h, w = frame.shape[:2]
        # Roboflow expects RGB; OpenCV gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Stage 1: vehicle detection --------------------------------
        try:
            veh_result = self._rf_client.infer(rgb, model_id=self.vehicle_model_id)
        except Exception as e:
            print(f"[{self.name}] vehicle API failed frame {frame_index}: {e}")
            return []

        ds_input = []
        for pred in veh_result.get("predictions", []):
            conf = float(pred.get("confidence", 0.0))
            if conf < self.conf_threshold:
                continue
            label = pred.get("class", "vehicle")
            cx, cy = pred.get("x", 0), pred.get("y", 0)
            pw, ph = pred.get("width", 0), pred.get("height", 0)
            ds_input.append(([cx - pw / 2, cy - ph / 2, pw, ph], conf, label))

        if not ds_input:
            return []

        # ---- Tracking --------------------------------------------------
        # DeepSORT needs the original BGR frame to build appearance embeddings.
        tracks = self._tracker.update_tracks(ds_input, frame=frame)

        read_this_frame = (self._frame_count % self.read_every_n_frames == 0)
        self._frame_count += 1
        detections = []

        for t in tracks:
            if not t.is_confirmed():
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in t.to_ltrb())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue

            track_id = _stable_track_id(t.track_id)
            vehicle_class = t.get_det_class() or "vehicle"
            conf = t.get_det_conf() or 0.5
            crop = frame[y1:y2, x1:x2]

            rec = self._records.get(track_id)
            if rec is None:
                rec = _VehicleRecord(
                    track_id=track_id, first_sec=timestamp_sec,
                    vehicle_class=vehicle_class,
                )
                self._records[track_id] = rec

            rec.vehicle_class = vehicle_class
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
        """Keep the sharpest, largest view of this vehicle for the gallery."""
        area = crop.shape[0] * crop.shape[1]
        score = area * (1.0 + _sharpness(crop))
        if score > rec.best_score:
            rec.best_score = score
            rec.best_image = crop.copy()
            rec.colour = dominant_colour(crop)

    def _read_plate(self, rec: _VehicleRecord, crop) -> None:
        """Stage 2: plate bbox via Roboflow. Stage 3: EasyOCR on the crop."""
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        try:
            plate_result = self._rf_client.infer(rgb_crop, model_id=self.plate_model_id)
        except Exception as e:
            print(f"[{self.name}] plate API failed for track {rec.track_id}: {e}")
            return

        preds = plate_result.get("predictions", [])
        if not preds:
            return

        # Pick the highest-confidence plate prediction inside this crop.
        best = max(preds, key=lambda p: p.get("confidence", 0))
        if best.get("confidence", 0) < self.plate_conf:
            return

        cx, cy = best.get("x", 0), best.get("y", 0)
        pw, ph = best.get("width", 0), best.get("height", 0)
        px1 = max(0, int(cx - pw / 2))
        py1 = max(0, int(cy - ph / 2))
        px2 = min(crop.shape[1], int(cx + pw / 2))
        py2 = min(crop.shape[0], int(cy + ph / 2))

        if px2 - px1 < 4 or py2 - py1 < 3:
            return

        rec.plate_detections += 1
        plate_img = crop[py1:py2, px1:px2]
        width = plate_img.shape[1]
        if width > rec.best_plate_width:
            rec.best_plate_width = width
            rec.best_plate_image = plate_img.copy()

        if width < self.min_plate_width:
            if rec.plate_status == "no_plate_found":
                rec.plate_status = "too_small"
            return

        # Stage 3: OCR
        rec.ocr_attempts += 1
        text, conf, status = self._ocr.read(plate_img)
        if status == "ok" and text:
            rec.votes.add(text, conf)
            rec.plate_status = "ok"
        elif status != "ok" and rec.plate_status not in ("ok",):
            rec.plate_status = status

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        """Write the per-vehicle gallery. Called by the runner after the last frame."""
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
        print(
            f"[{self.name}] {c['total']} vehicle(s) captured; "
            f"{c['with_plate']} with a readable plate. "
            f"too_small={c.get('too_small', 0)} "
            f"unreadable={c.get('unreadable', 0)} "
            f"no_plate_found={c.get('no_plate_found', 0)}"
        )
        if c["total"] and not c["with_plate"]:
            widths = [v["plate_width_px"] for v in summary["vehicles"]
                      if v["plate_width_px"]]
            if widths:
                print(
                    f"[{self.name}] Widest plate seen: {max(widths)} px. "
                    f"Need ~{self.min_plate_width}+ px for OCR to work. "
                    f"Camera may be too distant."
                )
        return summary
