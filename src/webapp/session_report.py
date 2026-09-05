"""
Route Session Fused Report Generator.

Aggregates the 10 crowd-safety metrics across all cameras in a route session,
analyzes physical crowd propagation along configured topology corridors,
and generates a unified, standalone HTML report and JSON summary.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from topology.graph import TOPOLOGY, CameraTopology

logger = logging.getLogger(__name__)


def aggregate_session_metrics(
    camera_summaries: dict[str, dict[str, Any]],
    topology: Optional[CameraTopology] = None,
) -> dict[str, Any]:
    """
    Compute route-level aggregated metrics across all camera summaries in a session.

    Aggregation rationale:
      - Density: Weighted average by corridor capacity (pax/m² or normalised)
      - Velocity Field: Arithmetic mean of avg walking speed (px/frame or m/s)
      - Specific Flow: Sum across cameras (total throughput)
      - Crowd Pressure: Maximum observed (worst-case safety signal)
      - Divergence: Minimum observed (most negative compression)
      - Stop & Go: Weighted average
      - Oscillation Symmetry: Maximum observed
      - Crush Risk %: Maximum observed across cameras + sum of crush events
      - Counterflow Friction: Weighted average + sum of friction events
      - Directional Entropy: Maximum observed (most chaotic camera)
      - Velocity Variance: Mean of spatial velocity variance
    """
    topo = topology or TOPOLOGY

    # ------------------------------------------------------------------
    # Helpers: a missing measurement is not a zero.
    # ------------------------------------------------------------------
    def _opt(summary: dict, *keys) -> Optional[float]:
        """
        First present, non-None value among ``keys``, else None.

        NOT ``summary.get(k) or 0.0``. That idiom was used throughout this
        function and it collapses three different states into one number:
        "not measured", "measured as zero", and "measured as a falsy value".
        A camera that never produced a density then contributed 0.0 to a
        WEIGHTED MEAN, dragging the route average down in proportion to how
        many cameras failed — so a route where two of three cameras measured
        nothing reported a third of the real density. Under-reporting, which
        is the direction that matters.
        """
        for k in keys:
            v = summary.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    def _mean(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _max(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def _min(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    def _rnd(v: Optional[float], nd: int) -> Optional[float]:
        return round(v, nd) if v is not None else None

    if not camera_summaries:
        return {
            "camera_count": 0,
            "avg_density": 0.0,
            "avg_speed": 0.0,
            "peak_speed": 0.0,
            "total_specific_flow": 0.0,
            "max_crowd_pressure": 0.0,
            "worst_divergence": 0.0,
            "avg_stop_go": 0.0,
            "max_oscillation_symmetry": 0.0,
            "max_crush_risk_pct": 0.0,
            "total_crush_events": 0,
            "avg_counterflow_pct": 0.0,
            "total_counterflow_events": 0,
            "max_directional_entropy": 0.0,
            "avg_velocity_variance": 0.0,
            "total_detections": 0,
            "total_tracks": 0,
            "cameras": {},
        }

    total_weight = 0.0
    density_weight = 0.0          # only cameras that ACTUALLY measured density
    weighted_density_sum = 0.0
    stop_go_weight = 0.0
    weighted_stop_go_sum = 0.0
    cf_weight = 0.0
    weighted_cf_sum = 0.0
    densities = []                # for the max, which is what safety needs
    uncalibrated_cams = []
    
    speeds = []
    peak_speeds = []
    flows = []
    pressures = []
    divergences = []
    oscillations = []
    crush_pcts = []
    crush_events = 0
    cf_events = 0
    entropies = []
    variances = []
    total_dets = 0
    total_trks = 0

    per_cam_data = {}

    for cam_id, summary in camera_summaries.items():
        node = topo.get_camera(cam_id)
        capacity = float(node.corridor_capacity_pax_min) if node else 400.0
        weight = max(1.0, capacity)
        total_weight += weight

        # Whether this camera's physical quantities mean what their names say.
        # Both model families write this key; absent is treated as False.
        cam_calibrated = bool(summary.get("is_calibrated", False))
        if not cam_calibrated:
            uncalibrated_cams.append(cam_id)

        # 1. Density. Weighted mean over the cameras that MEASURED it, so an
        #    unmeasured camera abstains rather than voting zero.
        d_val = _opt(summary, "avg_density")
        if d_val is not None:
            weighted_density_sum += d_val * weight
            density_weight += weight
            densities.append(d_val)

        # 2. Velocity
        spd = _opt(summary, "avg_speed_px_frame", "mean_speed_avg")
        p_spd = _opt(summary, "peak_speed_px_frame", "mean_speed_peak")
        if p_spd is None:
            p_spd = spd
        speeds.append(spd)
        peak_speeds.append(p_spd)

        # 3. Specific flow. GROSS, not net: net is (one way - other way), so
        #    two balanced 300/min streams net to ~0 and would report an empty
        #    corridor that is actually carrying 600 people a minute.
        flw = _opt(summary, "specific_flow_gross_per_sec",
                   "specific_flow_current", "specific_flow_net_per_sec")
        flows.append(flw)

        # 4. Crowd pressure
        cp = _opt(summary, "avg_crowd_pressure")
        p_cp = _opt(summary, "peak_crowd_pressure")
        if p_cp is None:
            p_cp = cp
        pressures.append(p_cp)

        # 5. Divergence (compression = negative)
        div = _opt(summary, "strongest_compression", "divergence_strongest_compression")
        divergences.append(div)

        # 6. Stop & Go
        sg = _opt(summary, "stop_go_score", "stop_go_avg")
        if sg is not None:
            weighted_stop_go_sum += sg * weight
            stop_go_weight += weight

        # 7. Oscillation symmetry
        osc = _opt(summary, "oscillation_symmetry", "oscillation_symmetry_avg")
        oscillations.append(osc)

        # 8. Crush risk
        cr_pct = _opt(summary, "pct_crush_risk")
        cr_evts = int(summary.get("crush_event_count") or 0)
        crush_pcts.append(cr_pct)
        crush_events += cr_evts

        # 9. Counterflow
        cf_pct = _opt(summary, "pct_counterflow_people")
        cf_evts = int(summary.get("counterflow_events_count") or 0)
        if cf_pct is not None:
            weighted_cf_sum += cf_pct * weight
            cf_weight += weight
        cf_events += cf_evts

        # 10. Entropy & Variance
        ent = _opt(summary, "avg_directional_entropy")
        var = _opt(summary, "avg_velocity_variance")
        entropies.append(ent)
        variances.append(var)

        total_dets += int(summary.get("total_detections") or 0)
        total_trks += int(summary.get("total_tracks") or 0)

        per_cam_data[cam_id] = {
            "name": node.name if node else cam_id,
            "capacity": capacity,
            # None is preserved through to the report: "not measured" must
            # stay distinguishable from "measured as zero" all the way to what
            # the reader sees.
            "density": _rnd(d_val, 2),
            "speed": _rnd(spd, 2),
            "peak_speed": _rnd(p_spd, 2),
            "specific_flow": _rnd(flw, 2),
            "pressure": _rnd(p_cp, 3),
            "divergence": _rnd(div, 3),
            "stop_go": _rnd(sg, 2),
            "oscillation": _rnd(osc, 2),
            "crush_risk_pct": _rnd(cr_pct, 1),
            "crush_events": cr_evts,
            "counterflow_pct": _rnd(cf_pct, 1),
            "counterflow_events": cf_evts,
            "entropy": _rnd(ent, 2),
            "variance": _rnd(var, 2),
            "is_calibrated": cam_calibrated,
            "peak_crush_timestamp_sec": float(summary.get("peak_crush_timestamp_sec") or 0.0),
            "total_detections": int(summary.get("total_detections") or 0),
        }

    # Weighted means divide by the weight that ACTUALLY CONTRIBUTED, not by
    # the total. Dividing by total_weight while only some cameras supplied a
    # value silently scales the result down by the fraction that abstained.
    agg_density = (weighted_density_sum / density_weight) if density_weight > 0 else None
    # Max as well as mean. A mean hides the worst location, and the worst
    # location is the whole question: one camera at 6 pax/m2 with two at 1
    # averages to a comfortable-looking 2.7 while one spot is critical.
    # Crowd pressure already reports max for exactly this reason.
    agg_density_max = _max(densities)
    agg_speed = _mean(speeds)
    agg_peak_speed = _max(peak_speeds)
    # Route throughput is NOT the sum of per-camera flows. On a route the same
    # crowd passes cam1, then cam2, then cam3 -- summing counts those people
    # three times. What limits a route is its narrowest measured section, so
    # the bottleneck (min) is the meaningful figure; the per-camera values are
    # in `cameras` for anyone who wants them individually.
    agg_flow = _min(flows)
    agg_pressure = _max(pressures)
    agg_divergence = _min(divergences)
    agg_stop_go = (weighted_stop_go_sum / stop_go_weight) if stop_go_weight > 0 else None
    agg_oscillation = _max(oscillations)
    agg_crush_pct = _max(crush_pcts)
    agg_cf_pct = (weighted_cf_sum / cf_weight) if cf_weight > 0 else None
    agg_entropy = _max(entropies)
    agg_variance = _mean(variances)

    # Physical crowd transit propagation analysis
    transit_narratives = []
    for edge in topo.edges:
        src = edge.from_cam
        dst = edge.to_cam
        if src in per_cam_data and dst in per_cam_data:
            src_node = topo.get_camera(src)
            dst_node = topo.get_camera(dst)
            src_name = src_node.name if src_node else src
            dst_name = dst_node.name if dst_node else dst
            t_tau = edge.travel_time_sec

            t_src_peak = per_cam_data[src]["peak_crush_timestamp_sec"]
            t_dst_peak = per_cam_data[dst]["peak_crush_timestamp_sec"]

            src_flow = per_cam_data[src]["specific_flow"]
            dst_cap = per_cam_data[dst]["capacity"]

            # Is this corridor's flow a physical rate?
            #
            # `corridor_capacity_pax_min` is a surveyed figure in pax/min. An
            # uncalibrated camera cannot produce pax/min -- without the real
            # width of the counting line its "flow" is a scaled detection
            # count. Dividing one by the other yields a confident, wrongly
            # scaled percentage.
            #
            # The FusionEngine already refuses that comparison. This report was
            # performing it anyway and printing "Inflow represents X% of
            # downstream capacity" into the standalone HTML -- the artifact
            # most likely to be shown to someone making a decision. The two
            # components must not disagree about whether the question is
            # answerable.
            src_summary = camera_summaries.get(src, {})
            flow_is_physical = bool(src_summary.get("is_calibrated", False))

            throughput_pax_sec = abs(src_flow) if src_flow is not None else None
            throughput_pax_min = (round(throughput_pax_sec * 60.0, 1)
                                  if throughput_pax_sec is not None else None)

            if flow_is_physical and throughput_pax_min is not None and dst_cap > 0:
                ratio = min(999.0, throughput_pax_min / dst_cap * 100.0)
                status_level = ("critical" if ratio >= 100.0
                                else "warning" if ratio >= 70.0 else "nominal")
            else:
                ratio = None
                status_level = "unmeasured"

            # Propagation check: does the downstream peak actually lag the
            # upstream one by roughly the configured travel time? This is the
            # only evidence in the report that the TOPOLOGY itself is right --
            # both timestamps were already being collected and then discarded.
            lag_sec = None
            lag_consistent = None
            if t_src_peak and t_dst_peak:
                lag_sec = round(t_dst_peak - t_src_peak, 1)
                # Generous tolerance: travel time is a free-flow constant while
                # real transit slows with density, so the measured lag is
                # expected to run longer, never shorter.
                lag_consistent = (-5.0 <= lag_sec - t_tau <= max(30.0, t_tau))
            
            narrative = {
                "source_cam": src,
                "source_name": src_name,
                "target_cam": dst,
                "target_name": dst_name,
                "travel_time_sec": t_tau,
                "source_flow": (round(throughput_pax_sec, 2)
                                if throughput_pax_sec is not None else None),
                "source_flow_pax_min": throughput_pax_min,
                "target_capacity": dst_cap,
                "capacity_utilization_pct": (round(ratio, 1) if ratio is not None else None),
                "flow_is_calibrated": flow_is_physical,
                "peak_source_time": t_src_peak,
                "peak_target_time": t_dst_peak,
                "observed_lag_sec": lag_sec,
                "lag_consistent_with_topology": lag_consistent,
                "status": status_level,
                "summary_text": (
                    f"Crowd moving from {src_name} ({src}) -> {dst_name} ({dst}): "
                    + (f"Measured corridor flow is {throughput_pax_min:.0f} pax/min. "
                       if (flow_is_physical and throughput_pax_min is not None)
                       else "Corridor flow is UNCALIBRATED (relative units only, not pax/min). ")
                    + f"Configured transit time is {t_tau:.1f}s. "
                    + (f"Inflow represents {ratio:.1f}% of downstream capacity "
                       f"({dst_cap:.0f} pax/min)."
                       if ratio is not None else
                       "Capacity utilisation NOT computed: it requires a calibrated "
                       "counting line, without which the comparison is meaningless.")
                    + (f" Observed peak lag {lag_sec:.0f}s vs configured {t_tau:.0f}s"
                       f" ({'consistent' if lag_consistent else 'INCONSISTENT - check the topology'})."
                       if lag_sec is not None else "")
                ),
            }
            transit_narratives.append(narrative)

    all_calibrated = not uncalibrated_cams
    return {
        "camera_count": len(camera_summaries),
        "avg_density": _rnd(agg_density, 2),
        "peak_density": _rnd(agg_density_max, 2),
        "avg_speed": _rnd(agg_speed, 2),
        "peak_speed": _rnd(agg_peak_speed, 2),
        # Renamed from "total_specific_flow": it is the route BOTTLENECK, not a
        # total. Summing per-camera flow along a route triple-counts one crowd.
        "bottleneck_specific_flow": _rnd(agg_flow, 2),
        "max_crowd_pressure": _rnd(agg_pressure, 3),
        "worst_divergence": _rnd(agg_divergence, 3),
        "avg_stop_go": _rnd(agg_stop_go, 2),
        "max_oscillation_symmetry": _rnd(agg_oscillation, 2),
        "max_crush_risk_pct": _rnd(agg_crush_pct, 1),
        "total_crush_events": crush_events,
        "avg_counterflow_pct": _rnd(agg_cf_pct, 1),
        "total_counterflow_events": cf_events,
        "max_directional_entropy": _rnd(agg_entropy, 2),
        "avg_velocity_variance": _rnd(agg_variance, 2),
        "total_detections": total_dets,
        "total_tracks": total_trks,
        # Provenance travels with the numbers. Without it a reader cannot tell
        # whether "density 2.4" is persons/m2 or an image-plane proxy, and the
        # capacity percentages below are only meaningful when this is True.
        "is_calibrated": all_calibrated,
        "uncalibrated_cameras": uncalibrated_cams,
        "units_note": (
            "All physical quantities are calibrated." if all_calibrated else
            "UNCALIBRATED: flow is not pax/min and density is not pax/m2 for "
            + ", ".join(uncalibrated_cams)
            + ". Capacity percentages are NOT computed for those corridors."
        ),
        "cameras": per_cam_data,
        "transit_narratives": transit_narratives,
    }



def _fmt(value: Optional[float], nd: int = 1, suffix: str = "",
         missing: str = "n/a") -> str:
    """
    Render a number for the HTML report, or an honest placeholder for None.

    None reaches here whenever a quantity was not measured — an unmeasured
    density, or a capacity percentage deliberately withheld because the
    corridor is uncalibrated. Formatting it with an f-string spec raises
    TypeError, and defaulting it to 0 would print a reassuring zero for
    something nobody measured. "n/a" says the true thing.
    """
    if value is None:
        return missing
    try:
        return f"{float(value):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return missing


def generate_session_report_html(
    session_name: str,
    session_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Generate a self-contained HTML report for the route session."""
    created_at = manifest.get("created_at") or datetime.now(timezone.utc).isoformat()
    status = manifest.get("status", "done").upper()

    cams = session_summary.get("cameras", {})
    narratives = session_summary.get("transit_narratives", [])

    # Table rows for cameras
    cam_rows = ""
    for cid, cdata in cams.items():
        # A missing measurement gets a NEUTRAL class, never "good".
        # Colouring an unmeasured cell green tells the reader the camera is
        # fine when the truth is that nobody looked.
        _cr = cdata['crush_risk_pct']
        cr_class = ("neutral" if _cr is None
                    else "bad" if _cr > 15 else "warn" if _cr > 5 else "good")
        _dv = cdata['divergence']
        div_class = ("neutral" if _dv is None
                     else "bad" if _dv < -1.5 else "warn" if _dv < -0.5 else "good")
        
        # Link to sub-report if present
        rel_report = f"{cid}/report.html"

        cam_rows += f"""
        <tr>
          <td><strong>{cdata['name']}</strong> <span class="badge-code">{cid}</span></td>
          <td>{cdata['capacity']} pax/min</td>
          <td><span class="num-highlight">{_fmt(cdata['density'], 2)}</span></td>
          <td>{_fmt(cdata['speed'], 2)} (pk {_fmt(cdata['peak_speed'], 2)})</td>
          <td>{_fmt(cdata['specific_flow'], 2)}</td>
          <td>{_fmt(cdata['pressure'], 3)}</td>
          <td class="{div_class}">{_fmt(cdata['divergence'], 3)}</td>
          <td>{_fmt(cdata['stop_go'], 2)}</td>
          <td>{_fmt(cdata['oscillation'], 2)}</td>
          <td class="{cr_class}">{_fmt(cdata['crush_risk_pct'], 1, '%')} ({cdata['crush_events']} evts)</td>
          <td>{_fmt(cdata['counterflow_pct'], 1, '%')}</td>
          <td>{_fmt(cdata['entropy'], 2)} bits</td>
          <td><a href="{rel_report}" target="_blank" class="sub-link">📄 View Report</a></td>
        </tr>
        """

    # Calibration banner.
    #
    # Without this the report presents density and flow figures whose units
    # depend on a calibration the reader cannot see. On an uncalibrated route
    # the numbers are still useful as RELATIVE trends for one camera, and
    # actively misleading if read as persons/m2 or pax/min — so the page has
    # to say which it is, at the top, before any figure.
    if session_summary.get("is_calibrated"):
        calib_banner = ""
    else:
        _uncal = ", ".join(session_summary.get("uncalibrated_cameras") or []) or "all cameras"
        calib_banner = (
            '<div class="calib-banner">'
            '<strong>UNCALIBRATED ROUTE</strong> &mdash; '
            f'{_uncal} have no ground-plane calibration. Density is NOT persons/m&sup2; '
            'and flow is NOT pax/min: both are image-plane proxies, comparable '
            'across frames of the same camera only. Corridor capacity percentages '
            'are therefore not computed. Calibrate with '
            '<code>scripts/calibrate_ground_plane.py</code> or a fitted perspective '
            'map to enable them.'
            '</div>'
        )

    # Transit narratives
    narrative_html = ""
    for n in narratives:
        badge_cls = "alert-badge-warn" if n["status"] in ("warning", "critical") else "alert-badge-ok"
        # `.get(k, 0)` is not enough: the key EXISTS with a None value when the
        # corridor is uncalibrated, so the default never fires and the format
        # spec raises TypeError.
        flow_val = n.get("source_flow_pax_min")
        flow_str = (f"Flow: {_fmt(flow_val, 0)} pax/min · " if flow_val is not None
                    else "Flow: uncalibrated · ")
        demand_str = _fmt(n.get("capacity_utilization_pct"), 1, "%", missing="not computed")
        narrative_html += f"""
        <div class="narrative-card {n['status']}">
          <div class="narrative-header">
            <span class="route-title">📍 {n['source_name']} ➔ {n['target_name']}</span>
            <span class="{badge_cls}">{flow_str}Transit Time: {n['travel_time_sec']}s · Demand: {demand_str}</span>
          </div>
          <p class="narrative-text">{n['summary_text']}</p>
        </div>
        """
    if not narrative_html:
        narrative_html = "<p class='hint-text'>No transit corridors configured between these active cameras.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Route Session Report — {session_name}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #38bdf8;
      --orange: #f97316;
      --red: #ef4444;
      --green: #10b981;
      --amber: #f59e0b;
      --purple: #a855f7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      padding: 32px 24px;
      line-height: 1.5;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    header {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{ font-size: 24px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }}
    .meta-tag {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text-muted); }}
    .status-badge {{
      display: inline-block;
      padding: 6px 14px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--green);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 700;
    }}
    .section-title {{
      font-size: 17px;
      font-weight: 700;
      color: #fff;
      margin: 28px 0 14px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 20px;
      position: relative;
    }}
    .kpi-lbl {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 6px; }}
    .kpi-val {{ font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }}
    .kpi-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 6px; }}
    
    .table-box {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      overflow-x: auto;
      margin-bottom: 28px;
    }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
    th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--card-border); white-space: nowrap; }}
    th {{ background: #162032; font-weight: 700; color: #cbd5e1; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
    tr:last-child td {{ border-bottom: none; }}
    .badge-code {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; background: #1f293d; padding: 2px 6px; border-radius: 4px; color: #94a3b8; }}
    .num-highlight {{ font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }}
    .good {{ color: var(--green); }}
    /* Unmeasured: muted, so it reads as absent rather than healthy. */
    .neutral {{ color: var(--text-muted); font-style: italic; }}
    .calib-banner {{
      background: rgba(249, 115, 22, 0.12);
      border: 1px solid var(--orange);
      border-left: 4px solid var(--orange);
      border-radius: 8px;
      padding: 12px 16px;
      margin: 16px 0 4px;
      color: var(--text);
      font-size: 14px;
      line-height: 1.5;
    }}
    .calib-banner code {{ color: var(--accent); }}
    .warn {{ color: var(--amber); font-weight: 700; }}
    .bad {{ color: var(--red); font-weight: 700; }}
    .sub-link {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.1);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid rgba(56, 189, 248, 0.2);
    }}
    .sub-link:hover {{ background: rgba(56, 189, 248, 0.2); }}

    .narrative-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-left: 4px solid var(--accent);
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 12px;
    }}
    .narrative-card.warning {{ border-left-color: var(--amber); background: rgba(245, 158, 11, 0.04); }}
    .narrative-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
    .route-title {{ font-weight: 700; font-size: 15px; color: #fff; }}
    .alert-badge-ok {{ font-size: 12px; font-weight: 600; color: var(--green); background: rgba(16, 185, 129, 0.1); padding: 3px 8px; border-radius: 6px; }}
    .alert-badge-warn {{ font-size: 12px; font-weight: 600; color: var(--amber); background: rgba(245, 158, 11, 0.1); padding: 3px 8px; border-radius: 6px; }}
    .narrative-text {{ font-size: 13px; color: #cbd5e1; }}
    .hint-text {{ font-size: 13px; color: var(--text-muted); font-style: italic; }}

    footer {{
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid var(--card-border);
      text-align: center;
      font-size: 12px;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

<div class="container">
  <header>
    <div>
      <h1>🗺️ Route Session: {session_name}</h1>
      <div class="meta-tag">Simhastha Kumbh Mela 2027 — Multi-Camera Fused Report · Generated {created_at[:19].replace('T', ' ')} UTC</div>
    </div>
    <div class="status-badge">STATUS: {status} ({session_summary.get('camera_count', 0)} CAMERAS)</div>
  </header>

  {calib_banner}

  <!-- 10 Aggregated Metrics -->
  <div class="section-title">📊 Route-Level Aggregated Metrics (10 Crowd-Safety Indicators)</div>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-lbl">Route Density</div>
      <div class="kpi-val" style="color: var(--orange);">{_fmt(session_summary.get('avg_density'), 2)}</div>
      <div class="kpi-sub">Capacity-weighted (pax/m²)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Velocity Field</div>
      <div class="kpi-val" style="color: var(--accent);">{_fmt(session_summary.get('avg_speed'), 2)}</div>
      <div class="kpi-sub">Peak {_fmt(session_summary.get('peak_speed'), 2)} px/frame</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Specific Flow</div>
      <div class="kpi-val" style="color: #22d3ee;">{_fmt(session_summary.get('bottleneck_specific_flow'), 2)}</div>
      <div class="kpi-sub">Total route throughput (ppl/s)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Worst Crowd Pressure</div>
      <div class="kpi-val" style="color: var(--red);">{_fmt(session_summary.get('max_crowd_pressure'), 3)}</div>
      <div class="kpi-sub">Peak across all cameras (s⁻²)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Worst Divergence</div>
      <div class="kpi-val" style="color: #fb7185;">{_fmt(session_summary.get('worst_divergence'), 3)}</div>
      <div class="kpi-sub">Negative = compression / crush</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Stop & Go Score</div>
      <div class="kpi-val" style="color: var(--amber);">{_fmt(session_summary.get('avg_stop_go'), 2)}</div>
      <div class="kpi-sub">Periodic halting index (0..1)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Max Oscillation</div>
      <div class="kpi-val" style="color: #f472b6;">{_fmt(session_summary.get('max_oscillation_symmetry'), 2)}</div>
      <div class="kpi-sub">Crowd surging / rocking symmetry</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Max Crush Risk</div>
      <div class="kpi-val" style="color: #fb923c;">{_fmt(session_summary.get('max_crush_risk_pct'), 1, '%')}</div>
      <div class="kpi-sub">{session_summary['total_crush_events']} total crush risk events</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Counter-Flow Friction</div>
      <div class="kpi-val" style="color: #eab308;">{_fmt(session_summary.get('avg_counterflow_pct'), 1, '%')}</div>
      <div class="kpi-sub">{session_summary['total_counterflow_events']} total friction events</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">Directional Entropy</div>
      <div class="kpi-val" style="color: var(--purple);">{_fmt(session_summary.get('max_directional_entropy'), 2)}</div>
      <div class="kpi-sub">Max disorder across route (bits)</div>
    </div>
  </div>

  <!-- Physical Crowd Transit Narrative -->
  <div class="section-title">🚶 Physical Crowd Transit & Corridor Propagation</div>
  <div style="margin-bottom: 24px;">
    {narrative_html}
  </div>

  <!-- Cross-Camera Comparison Table -->
  <div class="section-title">📹 Per-Camera Detailed Comparison</div>
  <div class="table-box">
    <table>
      <thead>
        <tr>
          <th>Camera Location</th>
          <th>Capacity</th>
          <th>Density</th>
          <th>Speed</th>
          <th>Flow</th>
          <th>Pressure</th>
          <th>Divergence</th>
          <th>Stop&Go</th>
          <th>Oscill.</th>
          <th>Crush Risk</th>
          <th>Counterflow</th>
          <th>Entropy</th>
          <th>Detailed Report</th>
        </tr>
      </thead>
      <tbody>
        {cam_rows}
      </tbody>
    </table>
  </div>

  <footer>
    Simhastha Kumbh Mela 2027 Multi-Camera Route Safety Platform · Outputs isolated under <code>outputs/sessions/{session_name}/</code>
  </footer>
</div>

</body>
</html>
"""
    return html


def build_session_report(
    session_dir: str,
    session_name: str,
    topology: Optional[CameraTopology] = None,
    manifest: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """
    Read per-camera outputs in ``session_dir``, build session summary and HTML report.
    Returns (summary_json_path, report_html_path).
    """
    topo = topology or TOPOLOGY
    manifest = manifest or {}

    camera_summaries: dict[str, dict[str, Any]] = {}

    # Scan session_dir subdirectories for camera runs
    if os.path.isdir(session_dir):
        for entry in os.listdir(session_dir):
            cam_dir = os.path.join(session_dir, entry)
            if not os.path.isdir(cam_dir):
                continue
            # Look for summary.json in cam_dir or model subdirectories
            sum_path = os.path.join(cam_dir, "summary.json")
            if not os.path.exists(sum_path):
                # Search subdirectories (e.g. cam_dir/crowd_motion_monitor/summary.json)
                for sub in os.listdir(cam_dir):
                    sub_sum = os.path.join(cam_dir, sub, "summary.json")
                    if os.path.exists(sub_sum):
                        sum_path = sub_sum
                        break
            
            if os.path.exists(sum_path):
                try:
                    with open(sum_path, "r", encoding="utf-8") as f:
                        camera_summaries[entry] = json.load(f)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", sum_path, e)

    session_summary = aggregate_session_metrics(camera_summaries, topology=topo)

    # Write session_summary.json
    sum_out = os.path.join(session_dir, "session_summary.json")
    with open(sum_out, "w", encoding="utf-8") as f:
        json.dump(session_summary, f, indent=2)

    # Write session_report.html
    html_content = generate_session_report_html(session_name, session_summary, manifest)
    report_out = os.path.join(session_dir, "session_report.html")
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(html_content)

    return sum_out, report_out
