"""
Unit tests for the person-masked divergence heatmap (commit 7929411,
"feat: implemented new detection function").

Covers, with fabricated data only — no video files, no detector or torch
inference:

1. PeopleOverlay.confirmed_boxes()      — snapshot of confirmed tracks.
2. DenseFlowAnalyser._split_moving_stationary()
                                        — velocity-threshold partitioning
                                          wired into the heatmap draw call.
3. FlowVisualiser.divergence_heatmap()  — person_boxes / stationary_boxes
                                          masking: flow colour may only land
                                          inside detected moving people;
                                          stationary people get a flat grey
                                          marker that must not be readable
                                          as crush/compression red.

The bug being regression-guarded: textured static surfaces (floor tiles,
background) generate small flow-field noise that clears the motion floor,
so the heatmap painted the whole scene instead of just people.

Run:  python -m pytest tests/test_heatmap_person_mask.py -v
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

from models.crowd_flow.dense_flow_analyser import DenseFlowAnalyser
from models.crowd_flow.people_overlay import PeopleOverlay, _Track
from models.crowd_flow.visualise import (
    _DIV_COLOR_NEG,
    _PERSON_STATIONARY_ALPHA,
    _PERSON_STATIONARY_FILL,
    FlowVisualiser,
)

# Synthetic scene geometry. Grid matches crowd_metrics.py's tiling:
# n_y = h // g rows x n_x = w // g cols; cell (r, c) spans
# [r*g,(r+1)*g) x [c*g,(c+1)*g).
G = 16
N_Y, N_X = 15, 20                       # -> 240 x 320 frame
H, W = N_Y * G, N_X * G

MOVING_BOX = (64.0, 64.0, 128.0, 128.0)        # cells r4:r8, c4:c8
STATIONARY_BOX = (192.0, 64.0, 224.0, 96.0)    # cells r4:r6, c12:c14


def make_track(track_id, box, vx=0.0, vy=0.0, confirmed=True, missing=0):
    return _Track(
        box=np.array(box, dtype=np.float32), track_id=track_id,
        vx=vx, vy=vy, confirmed=confirmed, missing=missing,
    )


def base_scene():
    """(frame, divergence grid, cell-speed grid) used by heatmap tests.

    Strong compression everywhere (t = -1 -> full red) so any ungated cell
    is maximally colourful; speeds below the adaptive motion floor except
    where a test opts a region in (median-based floor: most cells at 0.1 ->
    floor = max(0.4, 3*0.1) = 0.4 px/frame).
    """
    frame = np.full((H, W, 3), 40, np.uint8)
    div = np.full((N_Y, N_X), -3.0, np.float32)
    speed = np.full((N_Y, N_X), 0.1, np.float32)
    return frame, div, speed


def render(frame, div, speed, person_boxes=None, stationary_boxes=None):
    vis = FlowVisualiser()
    return vis.divergence_heatmap(
        frame.copy(), div, G, cell_speed=speed,
        person_boxes=person_boxes, stationary_boxes=stationary_boxes,
    )


# ======================================================================
# 1. PeopleOverlay.confirmed_boxes()
# ======================================================================

class TestConfirmedBoxes:
    def test_returns_only_confirmed_tracks(self):
        ov = PeopleOverlay()            # no load(): no detector involved
        ov._tracks = {
            1: make_track(1, [10, 20, 50, 60], vx=2.0, vy=-0.5),
            2: make_track(2, [1, 1, 9, 9], vx=99.0, vy=99.0, confirmed=False),
            3: make_track(3, [100, 100, 130, 140], confirmed=True),
        }
        boxes = ov.confirmed_boxes()
        assert len(boxes) == 2
        assert all(b[:4] != (1.0, 1.0, 9.0, 9.0) for b in boxes)

    def test_tuple_contents_and_order(self):
        ov = PeopleOverlay()
        ov._tracks = {7: make_track(7, [10, 20, 50, 60], vx=2.0, vy=-0.5)}
        assert ov.confirmed_boxes() == [(10.0, 20.0, 50.0, 60.0, 2.0, -0.5)]

    def test_all_values_are_plain_floats(self):
        ov = PeopleOverlay()
        ov._tracks = {1: make_track(1, [0, 0, 30, 40], vx=1, vy=2)}
        assert all(isinstance(v, float) for b in ov.confirmed_boxes() for v in b)

    def test_no_tracks_returns_empty_list(self):
        assert PeopleOverlay().confirmed_boxes() == []

    def test_coasting_confirmed_track_still_included(self):
        # missing > 0 means the detector has not seen it recently, but a
        # confirmed track is being coasted on purpose — it is still drawn,
        # so the heatmap must still mask to it.
        ov = PeopleOverlay()
        ov._tracks = {5: make_track(5, [8, 8, 40, 44], vx=1.0, missing=3)}
        assert len(ov.confirmed_boxes()) == 1


# ======================================================================
# 2. DenseFlowAnalyser._split_moving_stationary()
# ======================================================================

class TestMovingStationarySplit:
    SPLIT = staticmethod(DenseFlowAnalyser._split_moving_stationary)

    def test_nonzero_velocity_is_moving(self):
        moving, stationary = self.SPLIT([(10, 20, 50, 60, 2.0, 0.5)], 1.5)
        assert moving == [(10, 20, 50, 60)] and stationary == []

    def test_zero_velocity_is_stationary(self):
        moving, stationary = self.SPLIT([(100, 100, 130, 140, 0.0, 0.0)], 1.5)
        assert moving == [] and stationary == [(100, 100, 130, 140)]

    def test_exactly_at_threshold_is_moving(self):
        # hypot(vx, vy) == stopped_speed_px is NOT "< threshold": a person at
        # the stopped floor is not reliably standing still, so it keeps its
        # flow colour rather than being greyed out.
        moving, stationary = self.SPLIT([(0, 0, 10, 10, 1.5, 0.0)], 1.5)
        assert moving == [(0, 0, 10, 10)] and stationary == []

    def test_just_below_threshold_is_stationary(self):
        moving, stationary = self.SPLIT([(0, 0, 10, 10, 1.49, 0.0)], 1.5)
        assert moving == [] and stationary == [(0, 0, 10, 10)]

    def test_diagonal_velocity_uses_magnitude_not_components(self):
        # Both components below the threshold but hypot above it -> moving.
        moving, _ = self.SPLIT([(0, 0, 10, 10, 1.2, 1.2)], 1.5)
        assert moving == [(0, 0, 10, 10)]           # hypot(1.2, 1.2) ~= 1.70

    def test_custom_threshold_respected(self):
        boxes = [(0, 0, 10, 10, 2.0, 0.0)]
        assert self.SPLIT(boxes, 1.5)[0] == [(0, 0, 10, 10)]     # fast @ 1.5
        assert self.SPLIT(boxes, 3.0)[1] == [(0, 0, 10, 10)]     # slow @ 3.0

    def test_batch_split_preserves_every_box_exactly_once(self):
        batch = [
            (0, 0, 10, 10, 2.0, 0.0),     # moving
            (20, 0, 30, 10, 0.1, 0.0),    # stationary
            (40, 0, 50, 10, 0.0, 1.4),    # stationary (just under)
            (60, 0, 70, 10, 9.0, 9.0),    # moving
        ]
        moving, stationary = self.SPLIT(batch, 1.5)
        assert len(moving) + len(stationary) == len(batch)
        assert sorted(moving + stationary) == sorted(b[:4] for b in batch)

    def test_empty_input_yields_two_empty_lists(self):
        assert self.SPLIT([], 1.5) == ([], [])

    def test_output_entries_are_xyxy_quadruples(self):
        moving, stationary = self.SPLIT(
            [(0, 0, 10, 10, 5.0, 0.0), (90, 90, 99, 99, 0.0, 0.1)], 1.5)
        assert all(len(b) == 4 for b in moving + stationary)


# ======================================================================
# 3. FlowVisualiser.divergence_heatmap() masking
# ======================================================================

class TestHeatmapPersonMasking:
    def test_pixels_outside_person_boxes_untouched(self):
        # THE regression: a fast-moving textured background region away from
        # any box must be pixel-identical once the person mask is on.
        frame, div, speed = base_scene()
        speed[9:14, 16:20] = 5.0                      # fast background block
        out = render(frame, div, speed,
                     person_boxes=[MOVING_BOX], stationary_boxes=[STATIONARY_BOX])
        assert np.array_equal(out[150, 260], frame[150, 260])

    def test_inside_moving_box_shows_heatmap_color(self):
        frame, div, speed = base_scene()
        speed[4:8, 4:8] = 5.0
        out = render(frame, div, speed, person_boxes=[MOVING_BOX],
                     stationary_boxes=[STATIONARY_BOX])
        px = out[96, 96]                              # centre of MOVING_BOX
        assert px[2] > frame[96, 96][2] + 40                     # red up
        assert px[0] < frame[96, 96][0]                          # blue down

    def test_no_tracker_configured_legacy_whole_grid_paints(self):
        # person_boxes=None (people overlay off) keeps previous behaviour.
        frame, div, speed = base_scene()
        speed[9:14, 16:20] = 5.0
        out = render(frame, div, speed)
        assert out[150, 260, 2] > frame[150, 260, 2]

    def test_empty_person_lists_paint_nothing(self):
        frame, div, speed = base_scene()
        speed[:, :] = 5.0                             # whole field "fast"
        out = render(frame, div, speed, person_boxes=[], stationary_boxes=[])
        assert np.array_equal(out, frame)

    def test_stationary_box_shows_flat_grey_fill(self):
        frame, div, speed = base_scene()
        speed[4:6, 12:14] = 5.0                       # noise would paint here
        out = render(frame, div, speed, person_boxes=[MOVING_BOX],
                     stationary_boxes=[STATIONARY_BOX])
        expected = round(40 * (1 - _PERSON_STATIONARY_ALPHA)
                         + _PERSON_STATIONARY_FILL[0] * _PERSON_STATIONARY_ALPHA)
        px = out[80, 200]                             # inside STATIONARY_BOX
        assert all(abs(int(c) - expected) <= 1 for c in px)

    def test_grey_fill_is_not_crush_red_or_flow_hue(self):
        # The marker must be neutral (B==G==R). Compression red blended on
        # this same base would give R >> B; any HSV hue has B != R too.
        frame, div, speed = base_scene()
        speed[4:6, 12:14] = 5.0
        out = render(frame, div, speed, person_boxes=[],       # nothing else
                     stationary_boxes=[STATIONARY_BOX])         # masks all red
        px = [int(c) for c in out[80, 200]]
        spread = max(px) - min(px)
        assert spread <= 2, f"not neutral grey: {px}"
        # ...and it is genuinely different from what full compression red
        # at heatmap alpha would have produced on the same base pixel.
        red_blend = [int(round(40 + (c - 40) * 0.55)) for c in _DIV_COLOR_NEG]
        assert abs(px[2] - red_blend[2]) > 40

    def test_overlapping_boxes_union_is_painted(self):
        frame, div, speed = base_scene()
        speed[2:11, 2:11] = 5.0
        a = (32.0, 32.0, 96.0, 96.0)                  # overlaps b
        b = (64.0, 64.0, 160.0, 160.0)
        out = render(frame, div, speed, person_boxes=[a, b])
        assert out[80, 80, 2] > frame[80, 80, 2]      # overlap region
        assert out[120, 120, 2] > frame[120, 120, 2]  # only inside b
        assert np.array_equal(out[180, 280], frame[180, 280])   # outside both

    def test_sub_cell_box_still_paints_its_cell(self):
        # A distant person smaller than one cell contains no cell centre;
        # overlap-based marking must still light their cell.
        frame, div, speed = base_scene()
        speed[6:8, 6:8] = 5.0
        tiny = (104.0, 104.0, 112.0, 112.0)           # fully inside cell r6,c6
        out = render(frame, div, speed, person_boxes=[tiny])
        assert out[110, 110, 2] > frame[110, 110, 2]

    def test_motion_floor_still_gates_inside_person_box(self):
        # Person present but the flow there is below the display floor:
        # no colour — the two gates compose instead of either overriding.
        frame, div, speed = base_scene()
        speed[4:8, 4:8] = 0.05                        # under floor of ~0.4
        out = render(frame, div, speed, person_boxes=[MOVING_BOX])
        assert np.array_equal(out[96, 96], frame[96, 96])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
