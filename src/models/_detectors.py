"""
Shared RT-DETRv2 box detector.

One detector, used by every wrapper that needs person or vehicle boxes as an
input to its own work — the violence models' person-ROI crop, MediaPipe's
person stage, ANPR's vehicle capture, and the dense-flow validation route.

Why one shared module
---------------------
Each of those wrappers used to construct its own YOLO instance from its own
hard-coded checkpoint name.  That put the same responsibility in five places
with five slightly different confidence defaults and class filters, and it
meant swapping the backbone touched every one of them.  The detector is a
dependency of those models, not part of what they are.

Why RT-DETRv2 rather than YOLO
------------------------------
Apache 2.0 licensed (``PekingU/rtdetr_v2_r18vd``), and unlike YOLO's
AGPL-3.0 it carries no copyleft obligation on the surrounding code — which
matters for anything deployed rather than demonstrated.  It is also already
the backbone of the traffic, ANPR and umbrella detectors in this project, so
the number of distinct architectures to reason about goes down.

What it cannot do
-----------------
Keypoints.  RT-DETRv2 is a detection model: it returns boxes, not skeletons.
Any wrapper needing pose has to use a pose model (MediaPipe, MoveNet) — it
cannot be served from here.  This is why the skeleton-based fall models,
which depended on YOLO-pose, were removed rather than ported.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Apache 2.0, COCO-pretrained.  Downloaded and cached by transformers on
# first use.
DEFAULT_CHECKPOINT = "PekingU/rtdetr_v2_r18vd"

# COCO class indices used across the project.
COCO_PERSON = 0
COCO_VEHICLES = (2, 3, 5, 7)          # car, motorcycle, bus, truck
COCO_UMBRELLA = 25

# One instance per (checkpoint, device).  These wrappers are constructed per
# job and several may run in the same process; loading a separate copy of the
# same weights for each would multiply GPU memory for no benefit.
_CACHE: dict[tuple[str, str], "BoxDetector"] = {}
_CACHE_LOCK = threading.Lock()

# Tiling defaults.  20% overlap so a person standing on a tile boundary is
# whole in at least one crop; NMS at 0.55 to reconcile the duplicates that
# overlap necessarily creates.
_DEFAULT_TILE_OVERLAP: float = 0.2
_DEFAULT_MERGE_IOU: float = 0.55

# A crop smaller than this has less detail than the model's own input stride
# can use, so tiling past it costs passes and returns nothing.
_MIN_TILE_PX: int = 96


class BoxDetector:
    """
    RT-DETRv2 object detector returning plain [x1, y1, x2, y2] boxes.

    Thread-safety: inference is guarded by an internal lock, so one instance
    may be shared between the job worker and the validation runner.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device or "cpu"
        self.conf_threshold = conf_threshold
        self._proc = None
        self._model = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def load(self) -> None:
        # Double-checked under the same lock inference takes.  get_detector
        # hands one instance to several threads, and an unguarded load lets
        # two of them both see _model is None and both pull the weights: two
        # copies on the card, and a window where one thread reads self._proc
        # while the other is still assigning it.  The cheap read outside the
        # lock keeps the common case (already loaded) uncontended.
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

            logger.info("Loading RT-DETRv2 detector (%s) on %s ...",
                        self.checkpoint, self.device)
            proc = RTDetrImageProcessor.from_pretrained(self.checkpoint)
            model = RTDetrV2ForObjectDetection.from_pretrained(self.checkpoint)
            model = model.to(self.device).eval()
            # Publish only once everything is built: a partially-assigned
            # detector is what the early-return above would otherwise hand to
            # the next thread through.
            self._proc, self._torch, self._model = proc, torch, model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(
        self, image: np.ndarray, conf: float
    ) -> list[tuple[list[float], int, float]]:
        """One forward pass over one BGR image, in that image's coordinates."""
        from PIL import Image

        h, w = image.shape[:2]
        with self._lock:
            inputs = self._proc(images=Image.fromarray(image[:, :, ::-1]),
                                return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            target = self._torch.tensor([[h, w]], device=self.device)
            results = self._proc.post_process_object_detection(
                outputs, threshold=conf, target_sizes=target)[0]

        return [
            ([float(v) for v in box], int(label), float(score))
            for box, label, score in zip(results["boxes"].cpu().tolist(),
                                         results["labels"].cpu().tolist(),
                                         results["scores"].cpu().tolist())
        ]

    def _is_oom(self, exc: BaseException) -> bool:
        """Whether this exception means "not enough memory" specifically.

        torch.cuda.OutOfMemoryError only exists on a CUDA build, and both CPU
        allocation failures and some CUDA paths surface as a plain
        RuntimeError, so the message is checked too.  Deliberately narrow: a
        broken model must keep raising, not be quietly retried one image at a
        time and reported as a memory problem.
        """
        oom = getattr(getattr(self._torch, "cuda", None), "OutOfMemoryError", None)
        if oom is not None and isinstance(exc, oom):
            return True
        if isinstance(exc, (RuntimeError, MemoryError)):
            text = str(exc).lower()
            return "out of memory" in text or "cuda oom" in text
        return False

    def _infer_many(
        self, images: list[np.ndarray], conf: float
    ) -> list[list[tuple[list[float], int, float]]]:
        """One forward pass over several BGR images, each in its own coordinates.

        Tiling asks the detector the same question five times per frame (four
        crops plus the full frame).  Run one at a time that is five separate
        preprocessing passes, five host-to-device copies and five forward
        passes over a GPU that was never close to full on any of them.

        The processor resizes every input to the checkpoint's fixed size, so
        the five tensors agree in shape and stack into one batch.  RT-DETR
        has no cross-image interaction and runs in eval mode, so a batched
        pass returns per-image results identical to the serial ones.  If a
        processor ever does hand back mismatched shapes, fall back to the
        serial path rather than guessing.
        """
        from PIL import Image

        if not images:
            return []
        if len(images) == 1:
            return [self._infer(images[0], conf)]

        results = None
        with self._lock:
            inputs = self._proc(images=[Image.fromarray(im[:, :, ::-1]) for im in images],
                                return_tensors="pt")
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None and pixel_values.shape[0] == len(images):
                try:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with self._torch.no_grad():
                        outputs = self._model(**inputs)
                    target = self._torch.tensor(
                        [[im.shape[0], im.shape[1]] for im in images], device=self.device)
                    results = self._proc.post_process_object_detection(
                        outputs, threshold=conf, target_sizes=target)
                except Exception as exc:  # noqa: BLE001 - re-raised unless OOM
                    # A batch of five costs several times the peak memory of
                    # one pass, which is the whole point of it -- and on a
                    # small card, with other wrappers resident, that headroom
                    # is not guaranteed.  Speed is the optional part here, so
                    # give the memory back and let the serial path below
                    # produce the same detections more slowly.
                    if not self._is_oom(exc):
                        raise
                    print(f"[detector] batched inference out of memory "
                          f"({exc.__class__.__name__}); falling back to one "
                          f"image at a time")
                    results = None
                    try:
                        self._torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001 - CPU build, or no CUDA
                        pass

        # Fallback runs OUTSIDE the lock: _infer takes the same non-reentrant
        # lock, so calling it from in here would deadlock the detector for
        # every model sharing it.
        if results is None:
            return [self._infer(im, conf) for im in images]

        out = []
        for res in results:
            out.append([
                ([float(v) for v in box], int(label), float(score))
                for box, label, score in zip(res["boxes"].cpu().tolist(),
                                             res["labels"].cpu().tolist(),
                                             res["scores"].cpu().tolist())
            ])
        return out

    def _tile_rects(
        self, h: int, w: int, grid: tuple[int, int], overlap: float
    ) -> list[tuple[int, int, int, int]]:
        """Overlapping tile rectangles, plus the full frame."""
        nx, ny = grid
        tw, th = w / nx, h / ny
        ox, oy = tw * overlap, th * overlap
        rects = []
        for iy in range(ny):
            for ix in range(nx):
                x1 = int(max(0, ix * tw - ox))
                y1 = int(max(0, iy * th - oy))
                x2 = int(min(w, (ix + 1) * tw + ox))
                y2 = int(min(h, (iy + 1) * th + oy))
                if x2 - x1 >= _MIN_TILE_PX and y2 - y1 >= _MIN_TILE_PX:
                    rects.append((x1, y1, x2, y2))
        # The full frame as well.  A tile is a crop, so anything larger than
        # one tile — a person close to the camera — is cut in half by every
        # tile it touches and detected as two partial bodies or not at all.
        # The full-frame pass is what still sees those; the tiles are what
        # see the small ones.  NMS below reconciles the two.
        rects.append((0, 0, w, h))
        return rects

    def _merge(
        self,
        dets: list[tuple[list[float], int, float]],
        iou_threshold: float,
    ) -> list[tuple[list[float], int, float]]:
        """
        Class-aware NMS over detections pooled from overlapping tiles.

        Necessary rather than cosmetic: with 20% overlap a person standing in
        a seam is detected by two tiles and again by the full-frame pass, so
        without this the tiled count is inflated by every object near a
        boundary — which would look like a recall win and be a counting bug.
        """
        if not dets:
            return []
        import torch
        from torchvision.ops import batched_nms

        boxes = torch.tensor([d[0] for d in dets], dtype=torch.float32)
        labels = torch.tensor([d[1] for d in dets], dtype=torch.int64)
        scores = torch.tensor([d[2] for d in dets], dtype=torch.float32)
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        return [dets[i] for i in keep.tolist()]

    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[tuple[int, ...]] = None,
        conf_threshold: Optional[float] = None,
        tile_grid: Optional[tuple[int, int]] = None,
        tile_overlap: float = _DEFAULT_TILE_OVERLAP,
        merge_iou: float = _DEFAULT_MERGE_IOU,
    ) -> list[list[float]]:
        """
        Detect objects in a BGR frame.

        classes: COCO indices to keep, or None for everything.
        Returns [[x1, y1, x2, y2], ...] in source pixel coordinates.
        """
        return [box for box, _label, _score in self.detect_with_labels(
            frame, classes=classes, conf_threshold=conf_threshold,
            tile_grid=tile_grid, tile_overlap=tile_overlap,
            merge_iou=merge_iou)]

    def detect_with_labels(
        self,
        frame: np.ndarray,
        classes: Optional[tuple[int, ...]] = None,
        conf_threshold: Optional[float] = None,
        tile_grid: Optional[tuple[int, int]] = None,
        tile_overlap: float = _DEFAULT_TILE_OVERLAP,
        merge_iou: float = _DEFAULT_MERGE_IOU,
    ) -> list[tuple[list[float], int, float]]:
        """
        As detect(), but returns (box, class_index, score) per detection.

        Tiling
        ------
        ``tile_grid=(nx, ny)`` splits the frame into overlapping crops, runs
        the detector on each plus the whole frame, and merges the results
        with class-aware NMS.

        Why it helps so much: the processor resizes whatever it is given to
        640x640.  A 1280x720 frame is therefore halved before the model sees
        anything, so a person 20 px tall in the source arrives at 10 px —
        below what the model resolves.  Handing it a 427x240 crop instead
        means that same person arrives near full size.  Measured on a dense
        Nashik crowd frame: 35 people full-frame at conf 0.35, 95 with a 3x3
        grid.  Same weights, no fine-tuning — the recall was lost in the
        resize, not in the model.

        Measured on that frame, GPU, persons at conf 0.35:

            grid     people   median ms   vs full-frame
            none         35         135           1.0x
            (2, 2)       96         473           3.5x
            (3, 3)      109         876           6.5x
            (4, 4)      145        1437          10.7x

        (2, 2) is the best recall per unit cost; past it you pay roughly
        linearly in passes for diminishing returns.  Nothing here is free —
        pick a grid against the latency you actually have.

        One caveat worth knowing before enabling this anywhere: more
        detections is not automatically better downstream.  Feeding the extra
        (small, distant, jitter-prone) boxes into the dense-flow validation
        route made its agreement statistic worse and flipped its verdict, not
        because the flow degraded but because the route cannot track that
        many people.  See CrossFamilyValidator.tile_grid.
        """
        self.load()
        conf = self.conf_threshold if conf_threshold is None else conf_threshold

        if not tile_grid or tile_grid == (1, 1):
            dets = self._infer(frame, conf)
        else:
            h, w = frame.shape[:2]
            rects = self._tile_rects(h, w, tile_grid, tile_overlap)
            crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in rects]
            dets = []
            for (x1, y1, _, _), tile_dets in zip(rects, self._infer_many(crops, conf)):
                for box, label, score in tile_dets:
                    dets.append(([box[0] + x1, box[1] + y1,
                                  box[2] + x1, box[3] + y1], label, score))
            dets = self._merge(dets, merge_iou)

        keep = set(classes) if classes else None
        if keep is None:
            return dets
        return [d for d in dets if d[1] in keep]

    @property
    def id2label(self) -> dict:
        self.load()
        return {int(k): v for k, v in self._model.config.id2label.items()}


def get_detector(
    device: Optional[str] = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> BoxDetector:
    """
    Shared detector for (checkpoint, device).

    Prefer this over constructing BoxDetector directly: several wrappers can
    be active at once, and each holding its own copy of the same weights is
    the quickest way to exhaust a small card.

    There is deliberately no ``conf_threshold`` here.  It used to take one and
    pass it to the constructor, which only had an effect for whichever caller
    happened to build the instance first — every later caller silently got
    that one's value.  Threshold is a per-call property of a *query*, not of
    the shared model, so it belongs on ``detect`` / ``detect_with_labels``,
    where each caller's value actually applies.
    """
    key = (checkpoint, device or "cpu")
    with _CACHE_LOCK:
        det = _CACHE.get(key)
        if det is None:
            det = BoxDetector(checkpoint=checkpoint, device=device)
            _CACHE[key] = det
    return det
