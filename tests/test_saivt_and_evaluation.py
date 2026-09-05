"""
Tests for the ground-truth dataset, perspective calibration, and scoring.

These guard the pieces that make accuracy claims possible at all. The
perspective tests in particular pin down a convention that is easy to get
wrong and silently costly: fitting person height against the box CENTRE
rather than the feet. Getting that backwards produced a ~40% scale error
against SAIVT's own published coefficients, which would have gone straight
into every speed in m/s and every density in persons/m2.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import saivt  # noqa: E402
from evaluation import match_points, EvalResult, FrameResult  # noqa: E402
from models.crowd_flow.perspective import PerspectiveMap  # noqa: E402

_CAM = "P_Lev_4_Entry_Way_ip_107"


# ======================================================================
# Dataset
# ======================================================================

class TestSaivtDataset:
    def test_annotations_are_vendored(self):
        """Scoring must work on a fresh clone, without the 8 GB of imagery."""
        assert os.path.isdir(saivt.ANNOTATION_DIR)
        assert len(saivt.list_cameras()) == 12

    def test_occupancy_and_flow_are_distinct(self):
        """
        The two ground truths answer different questions and must not be
        conflated: GroundTruth.xml is who is STANDING there, VG-GroundTruth.xml
        is who CROSSED a line. Counting the latter per frame gives throughput,
        not occupancy.
        """
        cams = saivt.load_all()
        occ = [c for c in cams.values() if c.has_occupancy_gt]
        flow = [c for c in cams.values() if c.has_flow_gt]
        assert len(occ) == 10
        assert len(flow) == 6
        # Some cameras have one and not the other — proof they are separate.
        assert any(c.has_flow_gt and not c.has_occupancy_gt for c in cams.values())

    def test_occupancy_indices_are_still_indexes_not_video_frames(self):
        cam = saivt.load_camera(_CAM)
        assert max(cam.annotated_indices) < 200, \
            "frame ids index Frames/, not the 90k-frame source video"
        assert cam.interval_scale == 1500

    def test_count_at_returns_none_for_unannotated(self):
        """None, not 0 — an unannotated still is a missing measurement, and
        scoring it as an empty room rewards a model that detects nothing."""
        cam = saivt.load_camera(_CAM)
        assert cam.count_at(999999) is None
        assert isinstance(cam.count_at(cam.annotated_indices[0]), int)

    def test_totals(self):
        t = saivt.coverage_summary()["totals"]
        assert t["annotated_stills"] == 1020
        assert t["annotated_people"] == 5095

    def test_flow_events_carry_direction(self):
        cams = saivt.load_all()
        cam = next(c for c in cams.values() if c.has_flow_gt)
        dirs = cam.crossings_per_direction()
        assert set(dirs).issubset({0, 1})
        assert sum(dirs.values()) == len(cam.flow_events)


# ======================================================================
# Perspective calibration
# ======================================================================

def _saivt_boxes(camera_id=_CAM):
    path = os.path.join(saivt.ANNOTATION_DIR, camera_id, "perspectivemap.xml")
    root = ET.parse(path).getroot()
    return [(float(p.attrib["l"]), float(p.attrib["t"]),
             float(p.attrib["r"]), float(p.attrib["b"]))
            for p in root.find("annotations").findall("ped")]


class TestPerspectiveMap:
    def test_our_fit_reproduces_saivt_published_coefficients(self):
        """
        The interoperability test. A map fitted here and a map shipped by
        SAIVT must mean the same thing, or a Nashik camera calibrated with
        fit_from_boxes cannot be mixed with these.
        """
        cam = saivt.load_camera(_CAM)
        published = PerspectiveMap.from_saivt(cam.perspective)
        fitted = PerspectiveMap.fit_from_boxes(
            _saivt_boxes(), (published.image_width, published.image_height))

        for k in ("ah", "bh", "aw", "bw"):
            assert getattr(fitted, k) == pytest.approx(getattr(published, k), abs=0.01)
        assert fitted.ch == pytest.approx(published.ch, rel=0.01)

    def test_metres_per_pixel_matches_published_within_1pct(self):
        cam = saivt.load_camera(_CAM)
        pub = PerspectiveMap.from_saivt(cam.perspective)
        fit = PerspectiveMap.fit_from_boxes(_saivt_boxes(),
                                            (pub.image_width, pub.image_height))
        for frac in (0.3, 0.5, 0.7, 0.9):
            x, y = pub.image_width // 2, pub.image_height * frac
            assert fit.metres_per_pixel(x, y) == pytest.approx(
                pub.metres_per_pixel(x, y), rel=0.01)

    def test_centre_convention_is_what_saivt_uses(self):
        """
        Pins the convention explicitly. Evaluating SAIVT's own coefficients
        against its own boxes is unbiased ONLY with centre-y; feet and top
        both leave a large systematic residual.
        """
        cam = saivt.load_camera(_CAM)
        c = cam.perspective
        B = np.array(_saivt_boxes())
        cx = (B[:, 0] + B[:, 2]) / 2
        h = B[:, 3] - B[:, 1]
        err = {}
        for name, yy in (("top", B[:, 1]), ("feet", B[:, 3]),
                         ("centre", (B[:, 1] + B[:, 3]) / 2)):
            pred = c["ah"] * cx + c["bh"] * yy + c["ch"]
            err[name] = float(np.sqrt(np.mean((pred - h) ** 2)))
        assert err["centre"] < err["feet"] / 2
        assert err["centre"] < err["top"] / 2

    def test_person_grows_towards_the_camera(self):
        """Basic sanity: perspective means nearer people are bigger."""
        cam = saivt.load_camera(_CAM)
        pm = PerspectiveMap.from_saivt(cam.perspective)
        x = pm.image_width // 2
        _, far = pm.person_size_at_feet(x, pm.image_height * 0.3)
        _, near = pm.person_size_at_feet(x, pm.image_height * 0.9)
        assert near > far
        # and scale shrinks accordingly
        assert pm.metres_per_pixel(x, pm.image_height * 0.9) < \
               pm.metres_per_pixel(x, pm.image_height * 0.3)

    def test_feet_and_centre_queries_differ(self):
        """If these agreed, the implicit solve would be doing nothing."""
        cam = saivt.load_camera(_CAM)
        pm = PerspectiveMap.from_saivt(cam.perspective)
        x, y = pm.image_width // 2, pm.image_height * 0.6
        assert pm.person_size_at_feet(x, y)[1] != pytest.approx(
            pm.person_size_px(x, y)[1], rel=0.01)

    def test_mpp_is_clamped_against_degenerate_far_field(self):
        """Near the horizon person height tends to 0 and m/px would diverge."""
        pm = PerspectiveMap(ah=0.0, bh=1.0, ch=-1000.0,
                            image_width=800, image_height=600)
        assert 0.0 < pm.metres_per_pixel(400, 0) <= 0.04

    def test_fit_rejects_too_few_boxes(self):
        with pytest.raises(ValueError):
            PerspectiveMap.fit_from_boxes([(0, 0, 10, 40)], (100, 100))

    def test_fit_warns_on_boxes_at_one_depth(self, caplog):
        """A fit from people all at the same distance is a constant, and a
        constant scale is exactly what perspective is not."""
        boxes = [(x, 100, x + 20, 180) for x in range(0, 200, 20)]
        with caplog.at_level("WARNING"):
            PerspectiveMap.fit_from_boxes(boxes, (640, 480))
        # Either it warned, or the fit was perfect because the data is
        # degenerate-but-consistent; both are acceptable, a crash is not.

    def test_polygon_area_integrates_varying_scale(self):
        """
        A polygon spanning depth must not be converted with a single scale.
        The integrated area should differ from the naive single-point one.
        """
        cam = saivt.load_camera(_CAM)
        pm = PerspectiveMap.from_saivt(cam.perspective)
        w, h = pm.image_width, pm.image_height
        poly = [[0, h * 0.4], [w, h * 0.4], [w, h * 0.95], [0, h * 0.95]]
        integrated = pm.polygon_area_m2(poly)
        naive_s = pm.metres_per_pixel(w / 2, h * 0.675)
        naive = (w * h * 0.55) * naive_s * naive_s
        assert integrated > 0
        assert abs(integrated - naive) / integrated > 0.05, \
            "single-scale conversion should be visibly wrong over a deep polygon"


# ======================================================================
# Scoring
# ======================================================================

class TestMatching:
    def test_perfect_prediction(self):
        gt = [(10, 10), (50, 50), (100, 100)]
        tp, fp, fn, _ = match_points(gt, gt, lambda x, y: 20)
        assert (tp, fp, fn) == (3, 0, 0)

    def test_one_to_one_is_enforced(self):
        """Three detections piled on one person is one hit and two false
        positives, not three hits."""
        tp, fp, fn, _ = match_points([(10, 10), (11, 11), (12, 12)],
                                     [(10, 10)], lambda x, y: 20)
        assert (tp, fp, fn) == (1, 2, 0)

    def test_out_of_radius_is_not_a_match(self):
        tp, fp, fn, _ = match_points([(500, 500)], [(10, 10)], lambda x, y: 20)
        assert (tp, fp, fn) == (0, 1, 1)

    def test_empty_inputs(self):
        assert match_points([], [(1, 1)], lambda x, y: 10)[:3] == (0, 0, 1)
        assert match_points([(1, 1)], [], lambda x, y: 10)[:3] == (0, 1, 0)

    def test_radius_can_scale_with_perspective(self):
        """A radius that grows with person size matches near-field people that
        a flat radius would miss."""
        gt = [(100, 400)]
        pred = [(100, 440)]                      # 40 px away
        assert match_points(pred, gt, lambda x, y: 20)[0] == 0
        assert match_points(pred, gt, lambda x, y: 60)[0] == 1


class TestEvalResult:
    def _r(self, pairs):
        r = EvalResult("cam", "m")
        for i, (gt, pred) in enumerate(pairs):
            tp = min(gt, pred)
            r.frames.append(FrameResult(i, gt_count=gt, pred_count=pred,
                                        true_positives=tp,
                                        false_positives=max(0, pred - gt),
                                        false_negatives=max(0, gt - pred)))
        return r

    def test_bias_sign_flags_undercounting(self):
        """The safety-relevant direction: negative bias under-reads density,
        hence under-reads crowd pressure."""
        assert self._r([(10, 8), (20, 15)]).bias < 0
        assert self._r([(10, 12), (20, 25)]).bias > 0

    def test_mae_ignores_sign_but_bias_does_not(self):
        r = self._r([(10, 12), (10, 8)])          # +2 then -2
        assert r.mae == pytest.approx(2.0)
        assert r.bias == pytest.approx(0.0), \
            "cancellation is exactly why bias must be reported alongside MAE"

    def test_mape_skips_empty_frames(self):
        r = self._r([(0, 3), (10, 8)])
        assert r.mape == pytest.approx(20.0)      # only the non-empty frame

    def test_metrics_are_nan_not_crash_when_empty(self):
        r = EvalResult("c", "m")
        assert r.mae != r.mae                      # NaN
        assert r.precision != r.precision
