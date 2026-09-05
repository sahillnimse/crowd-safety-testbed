"""Per-camera HTML report rendering.

The report is the artifact most likely to be put in front of someone making a
decision, so the tests here are about one thing: a metric that could not be
measured must render as an ABSENCE, and must never take the report down.
"""

from pipeline.html_report import generate_report_html

# Every optional key present but null -- the shape a camera's summary takes
# when the metric pipeline ran but produced nothing.
NULL_SUMMARY = {
    "camera_id": "CCTV9",
    "camera_name": "Ram Kund Approach",
    "total_detections": 120,
    "pct_moving": None,
    "pct_stationary": None,
    "pct_crush_risk": None,
    "pct_moving_single_stream": None,
    "pct_moving_stream_a": None,
    "pct_moving_stream_b": None,
    "peak_crush_timestamp_sec": None,
    "peak_crush_people_count": None,
    "peak_counterflow_timestamp_sec": None,
    "peak_counterflow_people_count": None,
    "pct_counterflow_people": None,
    "avg_density": None,
    "peak_density": None,
    "avg_person_count": None,
    "peak_person_count": None,
    "avg_speed_px_frame": None,
    "specific_flow_gross_per_sec": None,
    "peak_crowd_pressure": None,
    "strongest_compression": None,
    "stop_go_score": None,
    "oscillation_symmetry": None,
    "avg_directional_entropy": None,
    "avg_velocity_variance": None,
}


def test_report_renders_when_every_optional_metric_is_null():
    """`dict.get(k, 0.0)` returns None for a key that exists with a null value,
    and `f"{None:.1f}"` raises -- so a camera that measured nothing crashed the
    renderer instead of producing a report saying it measured nothing."""
    html = generate_report_html("CCTV9.mp4", "crowd_motion_monitor",
                                dict(NULL_SUMMARY), None)
    assert html
    assert ">None<" not in html
    assert ": None" not in html


def test_report_renders_from_an_empty_summary():
    html = generate_report_html("x.mp4", "crowd_motion_monitor", {}, None)
    assert html
    assert ">None<" not in html


def test_unmeasured_counterflow_is_not_rendered_as_zero_percent():
    """0.0% reads as "no counter-flow detected". That is the opposite of
    "counter-flow was never computed", and on a corridor it is the reassuring
    reading of the two."""
    html = generate_report_html("CCTV9.mp4", "crowd_motion_monitor",
                                dict(NULL_SUMMARY), None)
    i = html.index("Counterflow Friction")
    card = html[i:i + 400]
    assert "0.0%" not in card
    assert "—" in card


def test_measured_zero_counterflow_still_renders_as_zero():
    """The absence must be distinguishable from a genuine measured zero."""
    summary = dict(NULL_SUMMARY, pct_counterflow_people=0.0)
    html = generate_report_html("CCTV9.mp4", "crowd_motion_monitor", summary, None)
    i = html.index("Counterflow Friction")
    assert "0.0%" in html[i:i + 400]


def test_detections_accepted_as_dicts_and_as_objects():
    """`jobs.py` passes Detection objects; the batch report regeneration script
    passes the raw dicts out of detections.json. Both must render."""
    class _Det:
        timestamp_sec = 1.5
        confidence = 0.9
        label = "person_moving"
        extra = {"track_id": 7}

    as_dicts = [{"timestamp_sec": 1.5, "confidence": 0.9,
                 "label": "person_moving", "extra": {"track_id": 7}}]
    from_dicts = generate_report_html("a.mp4", "m", dict(NULL_SUMMARY), as_dicts)
    from_objs = generate_report_html("a.mp4", "m", dict(NULL_SUMMARY), [_Det()])

    # Same values in, same report out -- the two callers must not disagree.
    assert from_dicts == from_objs
    assert '"moving"' in from_dicts   # the record reached the detections table
