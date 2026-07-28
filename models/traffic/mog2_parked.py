"""
Classical background-subtraction based parked-vehicle detector (MOG2).

Unlike the other 4 traffic models, this one does NOT use a neural network
at all — no pretrained weights, no GPU. It works purely by modeling the
"background" (static scene) over time and flagging any region that stays
foreground (i.e. different from the learned background) for a long time
as a parked object.

Why this exists alongside YOLO/RT-DETR/Roboflow: those all depend on
tracker ID continuity to decide "parked" (see _tracker.py) — if a track ID
breaks (object briefly occluded, detector missed a frame, a car leaves and
a different car parks in the exact same spot moments later), the parked
classification can silently reset or misfire. MOG2 has no such dependency:
it only cares whether *pixels in a region* have stopped changing, so it
acts as an independent cross-check for parked-vehicle claims, same role
optical_flow_crush.py plays as a non-DNN cross-check elsewhere in this repo.

Trade-off: MOG2 cannot classify vehicle type (car/bus/truck) and is more
sensitive to lighting changes, shadows, and camera shake than a real
detector — it's a supplementary signal, not a replacement for #1-3.
"""

import cv2
import numpy as np

from models.base import BaseModelWrapper, Detection


class Mog2ParkedDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "mog2_parked"
    gpu_accelerated = False

    def __init__(self, min_area_px: int = 800, parked_frames_threshold: int = 90,
                 var_threshold: float = 16.0, device=None):
        """
        min_area_px: ignore contours smaller than this (filters out noise,
                     pedestrians, small debris — vehicles are large blobs)
        parked_frames_threshold: how many consecutive frames a foreground
                     blob must persist in roughly the same location before
                     being reported as "parked" (default 90 @ ~30fps = 3s)
        var_threshold: MOG2 sensitivity — higher = less sensitive to noise
                     but slower to detect genuine changes
        """
        super().__init__(device=device)
        self.min_area_px = min_area_px
        self.parked_frames_threshold = parked_frames_threshold
        self.var_threshold = var_threshold
        # region_id -> {"centroid": (x,y), "first_frame": int, "last_frame": int}
        self._stable_regions: dict[int, dict] = {}
        self._next_region_id = 0

    def load(self):
        self._model = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=self.var_threshold,
        )

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        fg_mask = self._model.apply(frame)

        # Shadows are marked as 127 by MOG2 (detectShadows=True) — drop them,
        # keep only confident foreground (255), to avoid shadow blobs being
        # mistaken for parked vehicles.
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            current_boxes.append((x, y, x + w, y + h))

        detections = self._match_and_classify(current_boxes, frame_index, timestamp_sec)
        return detections

    def _match_and_classify(self, boxes, frame_index, timestamp_sec) -> list[Detection]:
        detections = []
        matched_region_ids = set()

        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # find nearest existing stable region within a reasonable radius
            best_id, best_dist = None, 40.0  # px — matching radius
            for rid, info in self._stable_regions.items():
                if rid in matched_region_ids:
                    continue
                rx, ry = info["centroid"]
                dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, rid

            if best_id is not None:
                info = self._stable_regions[best_id]
                info["centroid"] = (cx, cy)  # smooth-ish update, not averaged
                info["last_frame"] = frame_index
                matched_region_ids.add(best_id)
                region_id = best_id
            else:
                region_id = self._next_region_id
                self._next_region_id += 1
                self._stable_regions[region_id] = {
                    "centroid": (cx, cy),
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                }

            info = self._stable_regions[region_id]
            duration_frames = info["last_frame"] - info["first_frame"]

            if duration_frames >= self.parked_frames_threshold:
                detections.append(Detection(
                    model_name=self.name,
                    label="vehicle_parked",
                    confidence=min(1.0, duration_frames / (self.parked_frames_threshold * 2)),
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    bbox=[x1, y1, x2, y2],
                    extra={"vehicle_class": "unknown", "track_id": region_id,
                           "source": "background_subtraction"},
                ))

        # prune regions not matched this frame for a long time (object left)
        stale = [rid for rid, info in self._stable_regions.items()
                 if frame_index - info["last_frame"] > self.parked_frames_threshold]
        for rid in stale:
            del self._stable_regions[rid]

        return detections