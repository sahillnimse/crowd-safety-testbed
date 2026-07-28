"""
Classical background-subtraction based parked-vehicle detector
(dual-rate MOG2).

Unlike the other 3 traffic models, this one does NOT use a neural network
at all — no pretrained weights, no GPU. It flags vehicles that have stopped
and stayed stopped, purely from pixel statistics.

Why this exists alongside YOLO/RT-DETR/Roboflow: those all depend on
tracker ID continuity to decide "parked" (see _tracker.py) — if a track ID
breaks (object briefly occluded, detector missed a frame, a car leaves and
a different car parks in the exact same spot moments later), the parked
classification can silently reset or misfire. This has no such dependency:
it only cares whether *pixels in a region* have stopped changing, so it
acts as an independent cross-check for parked-vehicle claims, same role
optical_flow_crush.py plays as a non-DNN cross-check elsewhere in this repo.

**Why two background models.** A single MOG2 cannot detect parked objects,
because absorbing static content into the background is precisely what it
is designed to do. A car that parks stops being foreground within a second
or two, so "has been foreground for a long time" — the obvious formulation,
and what this file used to do — can never fire. Measured on a synthetic
scene where a car arrives and parks permanently, the single-model version
produced zero detections across 400 frames.

The standard fix is two background models learning at different rates:

  - FAST model (high learning rate): absorbs a stopped object quickly, so
    it stops reporting the parked car as foreground almost immediately.
  - SLOW model (low learning rate): holds on to it far longer, so the
    parked car stays foreground there.

    static foreground = foreground in SLOW  AND  background in FAST

A moving vehicle is foreground in both, so it cancels out. A permanent part
of the scene is background in both. Only something that recently stopped
and stayed put satisfies the pair — which is exactly a parked vehicle.

Trade-off: this cannot classify vehicle type (car/bus/truck) and is more
sensitive to lighting changes, shadows, and camera shake than a real
detector — it's a supplementary signal, not a replacement for #1-3. It also
cannot see a vehicle that was already parked before the clip started: with
nothing to contrast against, that car is simply part of the background.
"""

import cv2
import numpy as np

from models.base import BaseModelWrapper, Detection


class Mog2ParkedDetector(BaseModelWrapper):
    consumption_type = "frame"
    name = "mog2_parked"
    gpu_accelerated = False

    def __init__(self, min_area_px: int = 800, parked_seconds: float = 3.0,
                 var_threshold: float = 16.0,
                 fast_learning_rate: float = 0.02,
                 # 0.0001 keeps a parked vehicle reported for ~28s after it
                 # stops; at 0.0008 the slow model absorbs it after ~1.4s and
                 # the detection silently stops even though the car is still
                 # there. Lower is stickier, at the cost of adapting more
                 # slowly to genuine scene changes like lighting.
                 slow_learning_rate: float = 0.0001,
                 match_radius_px: float = 40.0,
                 max_drift_px: float = 25.0,
                 warmup_seconds: float = 2.0,
                 device=None):
        """
        min_area_px: ignore contours smaller than this (filters out noise,
                     pedestrians, small debris — vehicles are large blobs)
        parked_seconds: how long a static-foreground region must persist
                     before being reported as "parked". In seconds, not
                     frames, so the sampling stride doesn't change the answer.
        var_threshold: MOG2 sensitivity — higher = less sensitive to noise
                     but slower to detect genuine changes
        fast/slow_learning_rate: the two adaptation speeds whose difference
                     is what makes stopped objects visible at all
        match_radius_px: how far a region may move between observations and
                     still be considered the same region
        max_drift_px: total displacement from where the region was first
                     seen, beyond which it is treated as moving rather than
                     parked. Without this a slowly-crawling vehicle
                     accumulates duration and is eventually mislabelled.
        warmup_seconds: ignore detections before the background models have
                     seen enough of the scene to be meaningful
        """
        super().__init__(device=device)
        self.min_area_px = min_area_px
        self.parked_seconds = parked_seconds
        self.var_threshold = var_threshold
        self.fast_learning_rate = fast_learning_rate
        self.slow_learning_rate = slow_learning_rate
        self.match_radius_px = match_radius_px
        self.max_drift_px = max_drift_px
        self.warmup_seconds = warmup_seconds
        # region_id -> {"centroid", "origin", "first_sec", "last_sec"}
        self._stable_regions: dict[int, dict] = {}
        self._next_region_id = 0

    def load(self):
        self._fast = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=self.var_threshold,
        )
        self._slow = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=self.var_threshold,
        )
        self._model = (self._fast, self._slow)  # marker that load() ran
        self._stable_regions = {}
        self._next_region_id = 0

    def _static_foreground(self, frame) -> np.ndarray:
        """Mask of pixels the slow model calls foreground but the fast model
        has already absorbed — i.e. things that recently stopped moving."""
        fg_fast = self._fast.apply(frame, learningRate=self.fast_learning_rate)
        fg_slow = self._slow.apply(frame, learningRate=self.slow_learning_rate)

        # Shadows are marked 127 by MOG2 (detectShadows=True) — threshold at
        # 200 so only confident foreground (255) survives, otherwise shadow
        # blobs get mistaken for parked vehicles.
        _, fast_bin = cv2.threshold(fg_fast, 200, 255, cv2.THRESH_BINARY)
        _, slow_bin = cv2.threshold(fg_slow, 200, 255, cv2.THRESH_BINARY)

        static = cv2.bitwise_and(slow_bin, cv2.bitwise_not(fast_bin))
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # Close then dilate so a vehicle broken into fragments by texture
        # reassembles into one blob rather than several small ones.
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        return cv2.dilate(static, np.ones((3, 3), np.uint8), iterations=2)

    def _is_ghost(self, frame, box) -> bool:
        """True if this static region is where an object *used to be*.

        The slow model keeps a departed vehicle in its background for a
        while after it leaves. The vacated road then reads as
        static-foreground exactly like a parked car does, and it sits
        perfectly still, so neither persistence nor drift can tell them
        apart — in testing this reported a car as "parked" at its starting
        position while it was three-quarters of the way across the frame.

        Edges separate the two cases. A real parked vehicle has structure in
        the *current* frame; a ghost has structure only in the stale
        background model, with empty road in the frame now.
        """
        bg = self._slow.getBackgroundImage()
        if bg is None:
            return False

        x1, y1, x2, y2 = (int(v) for v in box)
        cur_patch = frame[y1:y2, x1:x2]
        bg_patch = bg[y1:y2, x1:x2]
        if cur_patch.size == 0 or bg_patch.shape != cur_patch.shape:
            return False

        def edge_density(patch):
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            return float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / max(gray.size, 1)

        # A margin, not a bare >: noise alone can nudge either side.
        return edge_density(cur_patch) <= edge_density(bg_patch) * 1.1

    def predict(self, frame, frame_index: int, timestamp_sec: float) -> list[Detection]:
        static = self._static_foreground(frame)

        contours, _ = cv2.findContours(static, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            box = (x, y, x + w, y + h)
            if self._is_ghost(frame, box):
                continue
            current_boxes.append(box)

        return self._match_and_classify(current_boxes, frame_index, timestamp_sec)

    def _match_and_classify(self, boxes, frame_index, timestamp_sec) -> list[Detection]:
        detections = []
        matched_region_ids = set()

        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # find nearest existing stable region within a reasonable radius
            best_id, best_dist = None, self.match_radius_px
            for rid, info in self._stable_regions.items():
                if rid in matched_region_ids:
                    continue
                rx, ry = info["centroid"]
                dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, rid

            if best_id is not None:
                info = self._stable_regions[best_id]
                info["centroid"] = (cx, cy)
                info["last_sec"] = timestamp_sec
                matched_region_ids.add(best_id)
                region_id = best_id
            else:
                region_id = self._next_region_id
                self._next_region_id += 1
                self._stable_regions[region_id] = {
                    "centroid": (cx, cy),
                    "origin": (cx, cy),   # where it was first seen
                    "first_sec": timestamp_sec,
                    "last_sec": timestamp_sec,
                }

            info = self._stable_regions[region_id]
            duration = info["last_sec"] - info["first_sec"]

            # Total displacement from the origin, not just per-frame motion.
            # Matching alone tolerates match_radius_px of movement *every*
            # observation, which lets a crawling vehicle rack up duration and
            # be reported as parked.
            ox, oy = info["origin"]
            drift = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if drift > self.max_drift_px:
                # It's moving — restart its clock from here.
                info["origin"] = (cx, cy)
                info["first_sec"] = timestamp_sec
                continue

            if timestamp_sec < self.warmup_seconds:
                continue  # background models haven't stabilised yet

            if duration >= self.parked_seconds:
                detections.append(Detection(
                    model_name=self.name,
                    label="vehicle_parked",
                    confidence=min(1.0, duration / (self.parked_seconds * 2)),
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    bbox=[x1, y1, x2, y2],
                    extra={"vehicle_class": "unknown", "track_id": region_id,
                           "parked_seconds": round(duration, 2),
                           "drift_px": round(drift, 1),
                           "source": "background_subtraction"},
                ))

        # prune regions not matched for a while (object left the scene)
        stale = [rid for rid, info in self._stable_regions.items()
                 if timestamp_sec - info["last_sec"] > self.parked_seconds]
        for rid in stale:
            del self._stable_regions[rid]

        return detections
