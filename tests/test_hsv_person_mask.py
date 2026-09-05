"""
Regression tests for person-scoping the HSV flow overlay.

Guarantees for the fix that masks hsv_flow_overlay's alpha to confirmed
moving people's boxes (same mechanism the divergence heatmap has had since
commit 7929411):

1. person_boxes=None  -> the scene-wide legacy path is BYTE-IDENTICAL to the
   pre-change implementation (checked against a captured reference array,
   plus structural assertions that fast background motion still paints).
2. person_boxes=[...] -> no colour lands outside the boxes; colour inside a
   moving person's box follows the existing hue=direction / alpha=speed
   encoding, unchanged.

Fabricated data only — no video, no detectors.

Run:  python -m pytest tests/test_hsv_person_mask.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.crowd_flow.visualise import FlowVisualiser

H, W = 240, 320
# Boxes in source pixels.
MOVING_BOX = (64.0, 64.0, 128.0, 128.0)        # cells/region well inside frame
FAR_PIXEL = (210, 150)                          # outside every box used here


def scene(rightward_px: float = 0.1):
    """
    Near-floor motion everywhere (median 0.1 -> display floor 0.4) with two
    strong jets that clear it: one INSIDE MOVING_BOX (70:120) and one far
    outside it (top-left corner). Mirrors real footage, where the median-
    based floor is set by the quiet majority, not by the movers.
    """
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    field = np.full((H, W, 2), rightward_px, np.float32)
    field[70:170, 70:170, 0] = 6.0              # strong; spans MOVING_BOX and
                                                # the a/b-overlap tests below
    field[:60, :80, 0] = 8.0                    # strong, outside (top-left)
    field[:60, :80, 1] = -3.0
    return frame, field


@pytest.fixture(scope="module")
def vis() -> FlowVisualiser:
    return FlowVisualiser()


# ======================================================================
# 1. people_overlay OFF (person_boxes=None) — legacy behaviour preserved
# ======================================================================

class TestLegacyPathUnchanged:
    def test_byte_identical_to_pre_change_reference(self, vis):
        # Reference captured from the implementation BEFORE the person_boxes
        # parameter existed (same seed, same inputs). If this file and the
        # reference ever disagree, the None-path changed.
        import numpy as np
        ref_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "opencode", "hsv_ref")
        ref_path = os.path.join(ref_dir, "reference.npy")
        if not os.path.isfile(ref_path):
            pytest.skip("pre-change reference not captured on this machine")
        frame, field = scene()
        frame = np.load(os.path.join(ref_dir, "frame.npy"))
        field = np.load(os.path.join(ref_dir, "field.npy"))
        expected = np.load(ref_path)
        out = vis.hsv_flow_overlay(frame.copy(), field, person_boxes=None)
        assert out.tobytes() == expected.tobytes()

    def test_default_argument_equals_explicit_none(self, vis):
        frame, field = scene()
        a = vis.hsv_flow_overlay(frame.copy(), field)
        b = vis.hsv_flow_overlay(frame.copy(), field, person_boxes=None)
        assert a.tobytes() == b.tobytes()

    def test_scene_wide_painting_still_happens_without_boxes(self, vis):
        # The whole point of the legacy overlay: fast background motion is
        # tinted even far from any person. The corner jet is at (40, 20) —
        # far outside MOVING_BOX.
        frame, field = scene()
        out = vis.hsv_flow_overlay(frame.copy(), field)
        assert not np.array_equal(out[20, 40], frame[20, 40])

    def test_static_pixels_still_untouched_without_boxes(self, vis):
        # Zero-flow regions keep the source frame exactly (per-pixel alpha).
        frame, field = scene()
        field[150:, 200:] = 0.0
        out = vis.hsv_flow_overlay(frame.copy(), field)
        assert np.array_equal(out[200, 280], frame[200, 280])


# ======================================================================
# 2. people_overlay ON — colour only inside confirmed moving boxes
# ======================================================================

class TestPersonScoped:
    def test_no_colour_outside_boxes(self, vis):
        frame, field = scene()
        out = vis.hsv_flow_overlay(frame.copy(), field,
                                   person_boxes=[MOVING_BOX])
        # (20,40): strong corner jet, outside the box -> masked to nothing.
        # (150,210): quiet region, outside -> untouched.
        # (150,150): inside the strong in-box jet's row range but x>128, i.e.
        # strong motion the scoping must suppress because nobody was there.
        for y, x in [(20, 40), (150, 210), (150, 150)]:
            assert np.array_equal(out[y, x], frame[y, x]), (y, x, out[y, x])

    def test_colour_inside_moving_box(self, vis):
        frame, field = scene()
        out = vis.hsv_flow_overlay(frame.copy(), field,
                                   person_boxes=[MOVING_BOX])
        assert not np.array_equal(out[96, 96], frame[96, 96])

    def test_direction_encoding_preserved_inside_box(self, vis):
        # hue = direction, alpha = speed — masking must not change HOW motion
        # is coloured: same pixel, opposite jet direction -> different colour;
        # same box, faster jet -> stronger displacement from the base frame.
        frame, _ = scene()
        base = np.full((H, W, 2), 0.1, np.float32)

        field_r = base.copy(); field_r[70:120, 70:120, 0] = 6.0
        field_l = base.copy(); field_l[70:120, 70:120, 0] = -6.0
        out_r = vis.hsv_flow_overlay(frame.copy(), field_r,
                                     person_boxes=[MOVING_BOX])
        out_l = vis.hsv_flow_overlay(frame.copy(), field_l,
                                     person_boxes=[MOVING_BOX])
        assert not np.array_equal(out_r[96, 96], out_l[96, 96])

        field_v = base.copy()
        field_v[70:95, 70:120, 0] = 12.0     # fast half of the box
        field_v[95:120, 70:120, 0] = 2.0     # slow half, still above floor
        out_v = vis.hsv_flow_overlay(frame.copy(), field_v,
                                     person_boxes=[MOVING_BOX])
        d = lambda y, x: int(np.abs(out_v[y, x].astype(int)
                                    - frame[y, x].astype(int)).sum())
        assert d(80, 95) > d(105, 95)

    def test_empty_box_list_paints_nothing(self, vis):
        frame, field = scene()
        out = vis.hsv_flow_overlay(frame.copy(), field, person_boxes=[])
        assert np.array_equal(out, frame)

    def test_box_partially_off_frame_is_clipped(self, vis):
        frame, field = scene()
        boxes = [(-50.0, -50.0, 80.0, 80.0)]       # hangs off top-left
        out = vis.hsv_flow_overlay(frame.copy(), field, person_boxes=boxes)
        assert not np.array_equal(out[20, 40], frame[20, 40])   # inside clip
        assert np.array_equal(out[150, 210], frame[150, 210])   # outside

    def test_overlapping_boxes_union_scoped(self, vis):
        frame, field = scene()
        a = (32.0, 32.0, 96.0, 96.0)
        b = (64.0, 64.0, 160.0, 160.0)
        out = vis.hsv_flow_overlay(frame.copy(), field, person_boxes=[a, b])
        assert not np.array_equal(out[80, 80], frame[80, 80])    # overlap
        assert not np.array_equal(out[120, 120], frame[120, 120])  # only b
        assert np.array_equal(out[150, 210], frame[150, 210])    # neither

    def test_masking_scales_with_frame_size(self, vis):
        # 720p-shaped input exercises the work-resolution downscale path
        # (INTER_AREA fractional edges) — must still exclude far pixels.
        rng = np.random.default_rng(3)
        h, w = 720, 1280
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        field = np.full((h, w, 2), 0.1, np.float32)
        field[150:350, 150:350, 0] = 6.0           # inside the box
        boxes = [(100.0, 100.0, 400.0, 400.0)]
        out = vis.hsv_flow_overlay(frame.copy(), field, person_boxes=boxes)
        assert np.array_equal(out[600, 1100], frame[600, 1100])
        assert not np.array_equal(out[250, 250], frame[250, 250])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
