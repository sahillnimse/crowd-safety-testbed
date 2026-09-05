"""
HTML Summary Report Generator for Crowd Safety Testbed runs.

Generates a standalone, self-contained HTML report (report.html) in the run
directory with zero external CDN dependencies. Works fully offline in any browser.
Presents the complete 10 crowd-safety metrics KPI cards and full frame-by-frame
detections with interactive search, filtering, and pagination.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any, Optional


def _esc(val: Any) -> str:
    return html.escape(str(val if val is not None else "—"))


def _fmt(value: Optional[float], nd: int = 1, suffix: str = "", missing: str = "—") -> str:
    if value is None:
        return missing
    try:
        return f"{float(value):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return missing


def _opt(summary: dict, *keys: str) -> Optional[float]:
    for k in keys:
        v = summary.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _num(summary: dict, *keys: str, default: float = 0.0) -> float:
    """Like `_opt`, but always returns a float.

    Use ONLY where the value is interpolated straight into a format spec or a
    CSS width -- a bar segment for a category that was never counted is
    legitimately zero-width. `dict.get(k, 0.0)` is NOT equivalent: it returns
    None when the key is present but null, and `f"{None:.1f}"` raises, which
    took the whole report down.

    For anything an operator reads as a measurement, use `_opt` + `_fmt` so an
    absent value renders as an absence rather than as a reassuring zero.
    """
    v = _opt(summary, *keys)
    return default if v is None else v


def generate_report_html(
    video_name: str,
    model_key: str,
    summary: dict,
    detections: Optional[list] = None,
) -> str:
    """Generate standalone HTML string for a run or camera sub-report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cam_name = summary.get("camera_name") or summary.get("camera_id") or video_name
    cam_id = summary.get("camera_id") or ""
    display_title = f"{cam_name} ({cam_id})" if cam_id and cam_id != cam_name else cam_name

    total_dets = summary.get("total_detections", len(detections) if detections else 0)
    total_tracks = summary.get("total_tracks", "—")
    pct_moving = _num(summary, "pct_moving")
    pct_stationary = _num(summary, "pct_stationary")
    pct_single = _num(summary, "pct_moving_single_stream")
    pct_stream_a = _num(summary, "pct_moving_stream_a")
    pct_stream_b = _num(summary, "pct_moving_stream_b")
    # Peak crush risk is a headline safety reading: keep it Optional so an
    # unmeasured camera shows an absence in the KPI, and use the coerced
    # value only for the distribution bar geometry.
    pct_crush = _opt(summary, "pct_crush_risk", "crush_risk_pct")
    pct_crush_bar = pct_crush if pct_crush is not None else 0.0
    has_streams = bool(pct_stream_a or pct_stream_b)
    primary_stream_pct = pct_stream_a if has_streams else pct_single
    secondary_stream_pct = pct_stream_b if has_streams else 0.0

    dir_a = summary.get("stream_a_direction")
    dir_b = summary.get("stream_b_direction")
    label_a = _esc(f"Stream A ({dir_a})" if dir_a else "Stream A")
    label_b = _esc(f"Stream B ({dir_b})" if dir_b else "Stream B")
    moving_or_a = label_a if has_streams else "Moving"

    crush_events = summary.get("crush_event_count", summary.get("crush_events", 0))
    peak_crush_t = _num(summary, "peak_crush_timestamp_sec")
    peak_crush_count = summary.get("peak_crush_people_count", 0) or 0

    # 10 Aggregated / Individual Metrics
    # 1. Density
    avg_density = _opt(summary, "avg_density", "density")
    peak_density = _opt(summary, "peak_density")
    avg_pax = _opt(summary, "avg_person_count")
    peak_pax = _opt(summary, "peak_person_count")

    # 2. Velocity
    avg_spd = _opt(summary, "avg_speed_px_frame", "avg_speed", "speed", "mean_speed_avg")
    peak_spd = _opt(summary, "peak_speed_px_frame", "peak_speed", "mean_speed_peak")

    # 3. Flow
    flw = _opt(summary, "specific_flow_gross_per_sec", "specific_flow_current", "specific_flow", "specific_flow_net_per_sec")
    crossings = summary.get("specific_flow_crossings", summary.get("total_crossings", 0))

    # 4. Pressure
    p_cp = _opt(summary, "peak_crowd_pressure", "pressure", "avg_crowd_pressure")
    avg_cp = _opt(summary, "avg_crowd_pressure")

    # 5. Divergence
    worst_div = _opt(summary, "strongest_compression", "divergence", "worst_divergence", "avg_divergence")
    avg_div = _opt(summary, "avg_divergence")

    # 6. Stop & Go
    stop_go = _opt(summary, "stop_go_score", "stop_go", "stop_go_avg")

    # 7. Oscillation
    osc = _opt(summary, "oscillation_symmetry", "oscillation", "oscillation_symmetry_avg")
    osc_peak = _opt(summary, "oscillation_symmetry_peak")

    # 8. Crush Risk
    # (pct_crush and crush_events above)

    # 9. Counterflow
    # No `or 0.0`: a camera on which counter-flow was never computed must not
    # render 0.0% -- that reads as "no counter-flow detected", which is the
    # opposite of "not measured". `_fmt` renders None as an em dash.
    pct_cf = _opt(summary, "pct_counterflow_people", "counterflow_pct")
    cf_events = summary.get("counterflow_events_count", summary.get("counterflow_events", 0))
    peak_cf_t = _num(summary, "peak_counterflow_timestamp_sec")
    peak_cf_count = summary.get("peak_counterflow_people_count", 0) or 0

    # 10. Directional Entropy
    avg_entropy = _opt(summary, "avg_directional_entropy", "entropy", "max_directional_entropy")
    avg_var = _opt(summary, "avg_velocity_variance", "variance")
    peak_var = _opt(summary, "peak_velocity_variance")

    stable_pct = summary.get("stable_tracks_pct", "—")
    unstable_count = summary.get("unstable_tracks_count", 0)
    avg_flips = summary.get("avg_flips_per_track", "—")

    label_counts = summary.get("label_counts", {})
    speed_by_label = summary.get("speed_by_label", {})
    suspicious = summary.get("suspicious_tracks", [])
    heading_hist = summary.get("heading_histogram", [])

    # Table rows for label distribution
    label_rows = []
    for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_dets * 100) if total_dets else 0
        spd_info = speed_by_label.get(lbl, {})
        avg_s = spd_info.get("avg_px_frame", "—")
        max_s = spd_info.get("max_px_frame", "—")
        badge_class = "badge-other"
        if "right" in lbl:
            badge_class = "badge-right"
        elif "left" in lbl:
            badge_class = "badge-left"
        elif "crush" in lbl or "convergence" in lbl:
            badge_class = "badge-crush"
        elif "stopped" in lbl or "fall" in lbl or "violence" in lbl:
            badge_class = "badge-stopped"

        label_rows.append(f"""
        <tr>
          <td><span class="badge {badge_class}">{_esc(lbl)}</span></td>
          <td class="num">{count:,}</td>
          <td class="num">{pct:.1f}%</td>
          <td class="num">{_esc(avg_s)} px/fr</td>
          <td class="num">{_esc(max_s)} px/fr</td>
        </tr>
        """)
    label_table_html = "".join(label_rows) if label_rows else "<tr><td colspan='5'>No label data</td></tr>"

    # Suspicious tracks table
    susp_rows = []
    for st in suspicious:
        dom_dir = st.get("dominant_direction", "—")
        dom_pct = st.get("dominant_pct", 0)
        susp_rows.append(f"""
        <tr>
          <td class="num">#{st.get('track_id')}</td>
          <td class="num alert-text">{st.get('flips')} flips</td>
          <td class="num">{st.get('stream_a_count', st.get('right_count', '—'))}</td>
          <td class="num">{st.get('stream_b_count', st.get('left_count', '—'))}</td>
          <td><span class="badge {'badge-right' if dom_dir in ('stream_a', 'right') else 'badge-left'}">{_esc(str(dom_dir).upper())} ({dom_pct}%)</span></td>
        </tr>
        """)
    susp_table_html = "".join(susp_rows) if susp_rows else "<tr><td colspan='5' class='muted-text'>No high-flip tracks detected (excellent stability).</td></tr>"

    # Heading angle histogram chart
    max_h_bin = max([b.get("count", 0) for b in heading_hist], default=1) or 1
    hist_rows = []
    for b in heading_hist:
        lo, hi = b.get("range", [0, 0])
        cnt = b.get("count", 0)
        direction = b.get("direction", "right" if (lo + hi) / 2 >= 0 else "left")
        pct_bar = (cnt / max_h_bin * 100)
        is_left = direction == "left"
        bar_color = "var(--left-color)" if is_left else "var(--right-color)"
        arrow = "← LEFT" if is_left else "→ RIGHT"
        hist_rows.append(f"""
        <div class="hist-row">
          <div class="hist-range">[{lo:+6.0f}°, {hi:+6.0f}°)</div>
          <div class="hist-bar-wrap">
            <div class="hist-bar" style="width: {pct_bar:.1f}%; background: {bar_color};"></div>
          </div>
          <div class="hist-count">{cnt:,}</div>
          <div class="hist-dir {'hist-left' if is_left else 'hist-right'}">{arrow}</div>
        </div>
        """)
    hist_html = "".join(hist_rows)

    # -------------------------------------------------------------------------
    # Format All Detections for Fast Offline Interactive Viewing
    # -------------------------------------------------------------------------
    compact_detections = []
    first_table_rows = []
    total_det_count = 0

    if detections:
        total_det_count = len(detections)
        for idx, d in enumerate(detections):
            if isinstance(d, dict):
                t_sec = float(d.get("timestamp_sec") or 0.0)
                conf = float(d.get("confidence") or 1.0)
                lbl = str(d.get("label") or "")
                extra = d.get("extra") or {}
            else:
                t_sec = float(getattr(d, "timestamp_sec", 0.0) or 0.0)
                conf = float(getattr(d, "confidence", 1.0) or 1.0)
                lbl = str(getattr(d, "label", "") or "")
                extra = getattr(d, "extra", {}) or {}

            tid = extra.get("track_id", "—")
            spd = extra.get("speed_px_frame")
            hdeg = extra.get("heading_deg")
            cdir = extra.get("crowd_direction") or ("stream_a" if "stream_a" in lbl else ("stream_b" if "stream_b" in lbl else "moving"))
            screen_dir = extra.get("stream_screen_direction")
            is_crush = bool(extra.get("local_crush_risk") or ("crush" in lbl))
            is_stopped = bool(extra.get("personally_stationary") or ("stopped" in lbl))
            is_cf = bool(extra.get("is_counterflow"))
            cf_angle = extra.get("counterflow_angle_deg")
            loc_div = extra.get("local_divergence")
            loc_ent = extra.get("local_directional_entropy")

            status_code = "stopped" if is_stopped else ("crush" if is_crush else (cdir if cdir in ("stream_a", "stream_b") else "moving"))

            dir_text = ("Stream A" if cdir == "stream_a" else ("Stream B" if cdir == "stream_b" else "Moving"))
            if screen_dir:
                dir_text = f"{dir_text} ({screen_dir})"
            if hdeg is not None:
                dir_text = f"{dir_text} · {hdeg:.0f}°"

            spd_text = f"{spd:.2f}" if spd is not None else "—"
            div_text = f"{loc_div:.2f}" if loc_div is not None else "—"

            if is_cf:
                dyn_text = f"Opposing ({cf_angle:.0f}°)" if cf_angle is not None else "Opposing"
            elif loc_ent is not None and loc_ent > 1.5:
                dyn_text = f"Entropy {loc_ent:.1f}"
            else:
                dyn_text = "Aligned"

            conf_pct = round(conf * 100)

            # Compact record: [t_sec, tid, status_code, dir_text, spd_text, div_text, dyn_text, conf_pct, is_crush, is_stopped, is_cf]
            compact_detections.append([
                round(t_sec, 2),
                tid,
                status_code,
                dir_text,
                spd_text,
                div_text,
                dyn_text,
                conf_pct,
                1 if is_crush else 0,
                1 if is_stopped else 0,
                1 if is_cf else 0,
            ])

            # Pre-render top 35 rows for instant static display / no-JS fallback
            if idx < 35:
                badge_html = (
                    '<span class="badge badge-stopped">⏹ Stopped</span>' if is_stopped else
                    ('<span class="badge badge-crush">⚠️ Crush Zone</span>' if is_crush else
                     ('<span class="badge badge-right">Stream A</span>' if cdir == 'stream_a' else
                      ('<span class="badge badge-left">Stream B</span>' if cdir == 'stream_b' else
                       '<span class="badge badge-right">Moving</span>')))
                )
                first_table_rows.append(f"""
                <tr>
                  <td class="num">{t_sec:.2f}s</td>
                  <td class="num"><strong>#{tid}</strong></td>
                  <td>{badge_html}</td>
                  <td>{_esc(dir_text)}</td>
                  <td class="num">{spd_text} px/fr</td>
                  <td class="num {'alert-text' if is_crush else ''}">{div_text}</td>
                  <td><span class="badge {'badge-crush' if is_cf else 'badge-right'}">{_esc(dyn_text)}</span></td>
                  <td class="num">{conf_pct}%</td>
                </tr>
                """)

    det_table_html = "".join(first_table_rows) if first_table_rows else "<tr><td colspan='8' class='muted-text'>No detection rows available.</td></tr>"
    compact_json = json.dumps(compact_detections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Camera Sub-Report — {_esc(display_title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f293d;
      --card-hover: #17223b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --text-dim: #64748b;
      --accent: #38bdf8;
      --orange: #f97316;
      --red: #ef4444;
      --green: #10b981;
      --amber: #f59e0b;
      --purple: #a855f7;
      --right-color: #10b981;
      --left-color: #06b6d4;
      --crush-color: #f97316;
      --stopped-color: #ef4444;
      --font: 'Plus Jakarta Sans', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      line-height: 1.55;
      padding: 32px 24px;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
    }}
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
    .header-title h1 {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .header-title p {{
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 4px;
      font-family: var(--font-mono);
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: var(--green);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    .section-title {{
      font-size: 17px;
      font-weight: 700;
      color: #fff;
      margin: 28px 0 14px 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    
    /* KPI Grid - Full 10 Metrics */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: rgba(56, 189, 248, 0.4);
    }}
    .kpi-card.tier-one {{
      grid-column: span 2;
      border: 1px solid rgba(249, 115, 22, 0.5);
      background: linear-gradient(180deg, rgba(249, 115, 22, 0.08) 0%, var(--card-bg) 100%);
    }}
    .kpi-card.tier-one .kpi-val {{ font-size: 32px; }}
    .kpi-lbl {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}
    .kpi-val {{
      font-size: 26px;
      font-weight: 800;
      font-family: var(--font-mono);
      line-height: 1.1;
      margin: 4px 0 6px;
    }}
    .kpi-sub {{
      font-size: 12px;
      color: var(--text-dim);
    }}

    /* Distribution & Charts */
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .dist-bar-wrap {{ margin: 16px 0 12px; }}
    .dist-bar {{
      display: flex;
      height: 24px;
      border-radius: 8px;
      overflow: hidden;
      background: #090d16;
      border: 1px solid var(--card-border);
    }}
    .dist-seg {{ height: 100%; transition: width 0.3s ease; }}
    .seg-left {{ background: var(--left-color); }}
    .seg-right {{ background: var(--right-color); }}
    .seg-crush {{ background: var(--crush-color); }}
    .seg-stopped {{ background: var(--stopped-color); }}

    .legend-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
      margin-top: 14px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-muted);
    }}
    .legend-dot {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}

    /* Tables */
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
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .num {{ text-align: right; font-family: var(--font-mono); }}
    th.num {{ text-align: right; }}

    /* Badges */
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      font-family: var(--font-mono);
    }}
    .badge-right {{ background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-left {{ background: rgba(6, 182, 212, 0.18); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }}
    .badge-crush {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }}
    .badge-stopped {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .badge-other {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }}
    .alert-text {{ color: #fb923c; font-weight: 700; }}
    .muted-text {{ color: var(--text-dim); }}

    /* Histogram rows */
    .hist-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 6px;
      font-size: 12px;
      font-family: var(--font-mono);
    }}
    .hist-range {{ width: 140px; color: var(--text-muted); }}
    .hist-bar-wrap {{
      flex: 1;
      height: 14px;
      background: #090d16;
      border-radius: 4px;
      overflow: hidden;
    }}
    .hist-bar {{ height: 100%; border-radius: 4px; }}
    .hist-count {{ width: 70px; text-align: right; font-weight: 600; color: #fff; }}
    .hist-dir {{ width: 80px; font-weight: 700; font-size: 11px; text-align: right; }}
    .hist-left {{ color: var(--left-color); }}
    .hist-right {{ color: var(--right-color); }}

    /* Two-col layout */
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 900px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
      .kpi-card.tier-one {{ grid-column: span 1; }}
    }}

    /* Detections Section Toolbar */
    .det-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      background: #162032;
      padding: 14px 18px;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }}
    .filter-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .filter-btn {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filter-btn:hover, .filter-btn.active {{
      background: var(--accent);
      color: #0b0f19;
      border-color: var(--accent);
      font-weight: 700;
    }}
    .search-input {{
      background: #090d16;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      color: #fff;
      padding: 6px 12px;
      font-size: 13px;
      font-family: var(--font);
      outline: none;
      min-width: 220px;
    }}
    .search-input:focus {{
      border-color: var(--accent);
    }}
    .det-pagination {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 14px;
      justify-content: space-between;
      font-size: 13px;
      color: var(--text-muted);
    }}
    .page-btn {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
    }}
    .page-btn:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}
    .page-btn:not(:disabled):hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .page-select {{
      background: #090d16;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 12px;
    }}

    footer {{
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid var(--card-border);
      color: var(--text-dim);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .actions a {{
      color: var(--accent);
      text-decoration: none;
      margin-left: 16px;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.1);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid rgba(56, 189, 248, 0.2);
    }}
    .actions a:hover {{ background: rgba(56, 189, 248, 0.2); text-decoration: none; }}
  </style>
</head>
<body>

<div class="container">
  <header>
    <div class="header-title">
      <h1>🎥 {_esc(display_title)}</h1>
      <p>Source Video: <strong>{_esc(video_name)}</strong> · Model: <strong>{_esc(model_key)}</strong> · Generated: {now_str}</p>
    </div>
    <div>
      <span class="status-pill">✓ ANALYSIS COMPLETE</span>
    </div>
  </header>

  <!-- 10 CROWD-SAFETY METRICS KPI CARDS -->
  <div class="section-title">
    <span>📊 Camera Corridor Analytics (10 Crowd-Safety Indicators)</span>
    <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);">{total_dets:,} Total Detections</span>
  </div>

  <div class="kpi-grid">
    <!-- 1. Density -->
    <div class="kpi-card">
      <div class="kpi-lbl">Corridor Density</div>
      <div class="kpi-val" style="color: #f97316;">{_fmt(avg_density, 1)}</div>
      <div class="kpi-sub">Peak {_fmt(peak_density, 1)} · {f'{avg_pax:.1f} avg people' if avg_pax is not None else 'pax/area'}</div>
    </div>

    <!-- 2. Velocity -->
    <div class="kpi-card">
      <div class="kpi-lbl">Mean Velocity</div>
      <div class="kpi-val" style="color: #60a5fa;">{_fmt(avg_spd, 2)}</div>
      <div class="kpi-sub">Peak {_fmt(peak_spd, 2)} px/frame</div>
    </div>

    <!-- 3. Specific Flow Rate -->
    <div class="kpi-card">
      <div class="kpi-lbl">Specific Flow Rate</div>
      <div class="kpi-val" style="color: #22d3ee;">{_fmt(flw, 2)}</div>
      <div class="kpi-sub">Throughput ({crossings} crossings/s)</div>
    </div>

    <!-- 4. Crowd Pressure -->
    <div class="kpi-card">
      <div class="kpi-lbl">Max Crowd Pressure</div>
      <div class="kpi-val" style="color: #ef4444;">{_fmt(p_cp, 3)}</div>
      <div class="kpi-sub">Avg {_fmt(avg_cp, 3)} corridor pressure</div>
    </div>

    <!-- 5. Worst Divergence -->
    <div class="kpi-card">
      <div class="kpi-lbl">Worst Divergence</div>
      <div class="kpi-val" style="color: #fb7185;">{_fmt(worst_div, 3)}</div>
      <div class="kpi-sub">Negative = compression / pinch-point</div>
    </div>

    <!-- 6. Stop & Go Waves -->
    <div class="kpi-card">
      <div class="kpi-lbl">Stop &amp; Go Waves</div>
      <div class="kpi-val" style="color: #fbbf24;">{_fmt(stop_go, 2)}</div>
      <div class="kpi-sub">Shockwave propagation index (0..1)</div>
    </div>

    <!-- 7. Oscillation Symmetry -->
    <div class="kpi-card">
      <div class="kpi-lbl">Oscillation Symmetry</div>
      <div class="kpi-val" style="color: #f472b6;">{_fmt(osc, 2)}</div>
      <div class="kpi-sub">Transverse surge &amp; rocking (0..1)</div>
    </div>

    <!-- 8. Peak Crush Risk -->
    <div class="kpi-card tier-one">
      <div class="kpi-lbl">Peak Crush Risk</div>
      <div class="kpi-val" style="color: #fb923c;">{_fmt(pct_crush, 1, '%')}</div>
      <div class="kpi-sub">{crush_events} crush events · peak {peak_crush_count} people @ {peak_crush_t:.1f}s</div>
    </div>

    <!-- 9. Counterflow Friction -->
    <div class="kpi-card">
      <div class="kpi-lbl">Counterflow Friction</div>
      <div class="kpi-val" style="color: #f59e0b;">{_fmt(pct_cf, 1, '%')}</div>
      <div class="kpi-sub">{cf_events} friction events · peak {peak_cf_count} people @ {peak_cf_t:.1f}s</div>
    </div>

    <!-- 10. Directional Entropy -->
    <div class="kpi-card">
      <div class="kpi-lbl">Directional Entropy</div>
      <div class="kpi-val" style="color: #a78bfa;">{_fmt(avg_entropy, 2)}</div>
      <div class="kpi-sub">Disorder &amp; panic (0-3 bits)</div>
    </div>
  </div>

  <!-- MOVEMENT & DENSITY DISTRIBUTION -->
  <div class="section-card">
    <div class="section-title">
      <span>Crowd Flow Distribution</span>
      <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);">{total_dets:,} Total Detections</span>
    </div>
    <div class="dist-bar-wrap">
      <div class="dist-bar">
        <div class="dist-seg seg-left" style="width: {primary_stream_pct}%;" title="{moving_or_a}: {primary_stream_pct:.1f}%"></div>
        {f'<div class="dist-seg seg-right" style="width: {secondary_stream_pct}%;" title="{label_b}: {secondary_stream_pct:.1f}%"></div>' if has_streams else ''}
        <div class="dist-seg seg-crush" style="width: {pct_crush_bar}%;" title="Crush Risk: {pct_crush_bar:.1f}%"></div>
        <div class="dist-seg seg-stopped" style="width: {pct_stationary}%;" title="Stationary: {pct_stationary:.1f}%"></div>
      </div>
    </div>
    <div class="legend-grid">
      <div class="legend-item"><div class="legend-dot seg-left"></div> <strong>{moving_or_a}</strong>: {primary_stream_pct:.1f}% ({label_counts.get('person_moving_stream_a' if has_streams else 'person_moving', 0):,})</div>
      {f'<div class="legend-item"><div class="legend-dot seg-right"></div> <strong>{label_b}</strong>: {secondary_stream_pct:.1f}% ({label_counts.get("person_moving_stream_b", 0):,})</div>' if has_streams else ''}
      <div class="legend-item"><div class="legend-dot seg-crush"></div> <strong>Collision / Crush Zone</strong>: {_fmt(pct_crush, 1, "%")} ({label_counts.get('person_crush_zone', 0):,})</div>
      <div class="legend-item"><div class="legend-dot seg-stopped"></div> <strong>Stationary / Stopped</strong>: {pct_stationary:.1f}% ({label_counts.get('person_stopped', 0):,})</div>
    </div>
  </div>

  <div class="grid-2">
    <!-- LABEL & SPEED TABLE -->
    <div class="section-card">
      <div class="section-title">Class Distribution &amp; Velocities</div>
      <table>
        <thead>
          <tr>
            <th>Class Label</th>
            <th class="num">Count</th>
            <th class="num">Share</th>
            <th class="num">Avg Speed</th>
            <th class="num">Max Speed</th>
          </tr>
        </thead>
        <tbody>
          {label_table_html}
        </tbody>
      </table>
    </div>

    <!-- TRACK STABILITY / SUSPICIOUS TRACKS -->
    <div class="section-card">
      <div class="section-title">
        <span>Per-Track Direction Integrity</span>
        <span style="font-size: 12px; color: var(--text-dim);">{unstable_count} high-flip tracks</span>
      </div>
      <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">
        Tracks flipping direction &gt;5 times (noisy heading or genuine turnarounds):
      </p>
      <table>
        <thead>
          <tr>
            <th class="num">Track ID</th>
            <th class="num">Flips</th>
            <th class="num">Stream A</th>
            <th class="num">Stream B</th>
            <th>Dominant Dir</th>
          </tr>
        </thead>
        <tbody>
          {susp_table_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- HEADING ANGLE HISTOGRAM -->
  <div class="section-card">
    <div class="section-title">
      <span>Heading Angle Histogram (20° Bins)</span>
      <span style="font-size: 12px; color: var(--text-dim);">Direction Boundary at ±90°</span>
    </div>
    <div style="margin-top: 14px;">
      {hist_html}
    </div>
  </div>

  <!-- ALL DETECTIONS SECTION -->
  <div class="section-card" id="detections-section">
    <div class="section-title">
      <span>📋 Frame-by-Frame Detections &amp; Kinematic Telemetry</span>
      <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);" id="det-total-badge">{total_det_count:,} Total Logged Detections</span>
    </div>

    <div class="det-toolbar">
      <div class="filter-pills">
        <button class="filter-btn active" data-filter="all">All ({total_det_count:,})</button>
        <button class="filter-btn" data-filter="stream_a">Stream A</button>
        <button class="filter-btn" data-filter="stream_b">Stream B</button>
        <button class="filter-btn" data-filter="crush">Crush Risk</button>
        <button class="filter-btn" data-filter="stopped">Stationary</button>
        <button class="filter-btn" data-filter="counterflow">Counterflow</button>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <input type="text" class="search-input" id="det-search" placeholder="Search track, time, direction…" />
        <select class="page-select" id="page-size-select">
          <option value="25">25 / page</option>
          <option value="50" selected>50 / page</option>
          <option value="100">100 / page</option>
          <option value="250">250 / page</option>
          <option value="1000">1000 / page</option>
        </select>
      </div>
    </div>

    <div class="table-box" style="margin-bottom: 0;">
      <table>
        <thead>
          <tr>
            <th class="num">Time</th>
            <th class="num">Track ID</th>
            <th>Flow Status</th>
            <th>Direction (Heading)</th>
            <th class="num">Velocity</th>
            <th class="num">Compression</th>
            <th>Dynamics</th>
            <th class="num">Conf</th>
          </tr>
        </thead>
        <tbody id="det-table-body">
          {det_table_html}
        </tbody>
      </table>
    </div>

    <div class="det-pagination">
      <span id="det-page-info">Showing top records</span>
      <div style="display: flex; gap: 6px;">
        <button class="page-btn" id="btn-first">&laquo; First</button>
        <button class="page-btn" id="btn-prev">&lsaquo; Prev</button>
        <span id="page-num-display" style="padding: 4px 8px; font-weight: 600; font-family: var(--font-mono);">Page 1</span>
        <button class="page-btn" id="btn-next">Next &rsaquo;</button>
        <button class="page-btn" id="btn-last">Last &raquo;</button>
      </div>
    </div>
  </div>

  <footer>
    <div>Crowd Safety Testbed — Automated Optical Flow &amp; Safety Analytics</div>
    <div class="actions">
      <a href="detections.json" target="_blank">📄 detections.json</a>
      <a href="detections.csv" target="_blank">📊 detections.csv</a>
      <a href="annotated.mp4" target="_blank">🎥 annotated.mp4</a>
    </div>
  </footer>
</div>

<script>
(function() {{
  const RAW_DATA = {compact_json};
  if (!RAW_DATA || !RAW_DATA.length) return;

  let currentFilter = 'all';
  let searchTerm = '';
  let currentPage = 1;
  let pageSize = 50;

  const tbody = document.getElementById('det-table-body');
  const pageInfo = document.getElementById('det-page-info');
  const pageNumDisplay = document.getElementById('page-num-display');
  const btnFirst = document.getElementById('btn-first');
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const btnLast = document.getElementById('btn-last');
  const searchInput = document.getElementById('det-search');
  const pageSizeSelect = document.getElementById('page-size-select');
  const filterBtns = document.querySelectorAll('.filter-btn');

  function getFilteredRows() {{
    const q = searchTerm.toLowerCase().trim();
    return RAW_DATA.filter(r => {{
      // Filter tab check: [t_sec, tid, status_code, dir_text, spd_text, div_text, dyn_text, conf_pct, is_crush, is_stopped, is_cf]
      if (currentFilter === 'stream_a' && r[2] !== 'stream_a') return false;
      if (currentFilter === 'stream_b' && r[2] !== 'stream_b') return false;
      if (currentFilter === 'crush' && !r[8]) return false;
      if (currentFilter === 'stopped' && !r[9]) return false;
      if (currentFilter === 'counterflow' && !r[10]) return false;

      // Text search check
      if (q) {{
        const text = (r[0] + ' ' + r[1] + ' ' + r[2] + ' ' + r[3] + ' ' + r[4] + ' ' + r[5] + ' ' + r[6]).toLowerCase();
        if (!text.includes(q)) return false;
      }}
      return true;
    }});
  }}

  function renderTable() {{
    const filtered = getFilteredRows();
    const totalCount = filtered.length;
    const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, totalCount);
    const pageRows = filtered.slice(startIdx, endIdx);

    let html = '';
    for (let i = 0; i < pageRows.length; i++) {{
      const r = pageRows[i];
      const isStopped = r[9];
      const isCrush = r[8];
      const isCf = r[10];
      const statusCode = r[2];

      let badgeHtml = '<span class="badge badge-right">Moving</span>';
      if (isStopped) badgeHtml = '<span class="badge badge-stopped">⏹ Stopped</span>';
      else if (isCrush) badgeHtml = '<span class="badge badge-crush">⚠️ Crush Zone</span>';
      else if (statusCode === 'stream_a') badgeHtml = '<span class="badge badge-right">Stream A</span>';
      else if (statusCode === 'stream_b') badgeHtml = '<span class="badge badge-left">Stream B</span>';

      const dynBadgeClass = isCf ? 'badge-crush' : (r[6].includes('Entropy') ? 'badge-other' : 'badge-right');

      html += `<tr>
        <td class="num">${{r[0].toFixed(2)}}s</td>
        <td class="num"><strong>#${{r[1]}}</strong></td>
        <td>${{badgeHtml}}</td>
        <td>${{r[3]}}</td>
        <td class="num">${{r[4]}} px/fr</td>
        <td class="num ${{isCrush ? 'alert-text' : ''}}">${{r[5]}}</td>
        <td><span class="badge ${{dynBadgeClass}}">${{r[6]}}</span></td>
        <td class="num">${{r[7]}}%</td>
      </tr>`;
    }}

    if (!html) {{
      html = '<tr><td colspan="8" class="muted-text" style="text-align: center; padding: 24px;">No matching detections found.</td></tr>';
    }}

    tbody.innerHTML = html;
    pageInfo.textContent = `Showing ${{startIdx + 1}}–${{endIdx}} of ${{totalCount.toLocaleString()}} matching detections`;
    pageNumDisplay.textContent = `Page ${{currentPage}} of ${{totalPages}}`;

    btnFirst.disabled = currentPage === 1;
    btnPrev.disabled = currentPage === 1;
    btnNext.disabled = currentPage === totalPages;
    btnLast.disabled = currentPage === totalPages;
  }}

  // Filter clicks
  filterBtns.forEach(b => {{
    b.addEventListener('click', () => {{
      filterBtns.forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      currentFilter = b.dataset.filter;
      currentPage = 1;
      renderTable();
    }});
  }});

  // Search input
  let debounceTimer;
  searchInput.addEventListener('input', (e) => {{
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {{
      searchTerm = e.target.value;
      currentPage = 1;
      renderTable();
    }}, 150);
  }});

  // Page size select
  pageSizeSelect.addEventListener('change', (e) => {{
    pageSize = parseInt(e.target.value, 10) || 50;
    currentPage = 1;
    renderTable();
  }});

  // Pagination navigation
  btnFirst.addEventListener('click', () => {{ currentPage = 1; renderTable(); }});
  btnPrev.addEventListener('click', () => {{ if (currentPage > 1) {{ currentPage--; renderTable(); }} }});
  btnNext.addEventListener('click', () => {{ currentPage++; renderTable(); }});
  btnLast.addEventListener('click', () => {{
    const filtered = getFilteredRows();
    currentPage = Math.ceil(filtered.length / pageSize);
    renderTable();
  }});

  // Initial render
  renderTable();
}})();
</script>

</body>
</html>
"""


def export_html_report(
    output_path: str,
    video_name: str,
    model_key: str,
    summary: dict,
    detections: Optional[list] = None,
) -> str:
    """Write report.html to disk and return path."""
    out_dir = os.path.dirname(output_path) or "."
    
    # If detections not supplied, check if detections.json exists alongside
    if detections is None:
        cand_json = os.path.join(out_dir, "detections.json")
        if os.path.isfile(cand_json):
            try:
                with open(cand_json, "r", encoding="utf-8") as f:
                    detections = json.load(f)
            except Exception:
                detections = None

    # If summary is missing or lacking keys, try loading summary.json
    if not summary:
        cand_sum = os.path.join(out_dir, "summary.json")
        if os.path.isfile(cand_sum):
            try:
                with open(cand_sum, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

    html_content = generate_report_html(video_name, model_key, summary or {}, detections)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path
