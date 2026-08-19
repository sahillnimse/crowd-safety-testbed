"""
HTML Summary Report Generator for Crowd Safety Testbed runs.

Generates a standalone, self-contained HTML report (report.html) in the run
directory with zero external CDN dependencies. Works fully offline in any browser.
"""

import html
import json
import os
from datetime import datetime


def _esc(val) -> str:
    return html.escape(str(val if val is not None else "—"))


def generate_report_html(
    video_name: str,
    model_key: str,
    summary: dict,
    detections: list = None,
) -> str:
    """Generate standalone HTML string for a run summary."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_dets = summary.get("total_detections", len(detections) if detections else 0)
    total_tracks = summary.get("total_tracks", "—")
    pct_moving = summary.get("pct_moving", 0.0)
    pct_stationary = summary.get("pct_stationary", 0.0)
    pct_crush = summary.get("pct_crush_risk", 0.0)
    pct_single = summary.get("pct_moving_single_stream", 0.0)
    pct_stream_a = summary.get("pct_moving_stream_a", 0.0)
    pct_stream_b = summary.get("pct_moving_stream_b", 0.0)
    has_streams = bool(pct_stream_a or pct_stream_b)
    primary_stream_pct = pct_stream_a if has_streams else pct_single
    secondary_stream_pct = pct_stream_b if has_streams else 0.0
    crush_events = summary.get("crush_event_count", 0)
    peak_crush_t = summary.get("peak_crush_timestamp_sec", 0.0)
    peak_crush_count = summary.get("peak_crush_people_count", 0)
    avg_spd = summary.get("avg_speed_px_frame", "—")
    stable_pct = summary.get("stable_tracks_pct", "—")
    unstable_count = summary.get("unstable_tracks_count", 0)
    avg_flips = summary.get("avg_flips_per_track", "—")
    boundary_crush_pct = summary.get("boundary_crush_pct", "—")

    # New crowd dynamics metrics
    pct_cf = summary.get("pct_counterflow_people", 0.0)
    cf_events = summary.get("counterflow_events_count", 0)
    peak_cf_t = summary.get("peak_counterflow_timestamp_sec", 0.0)
    peak_cf_count = summary.get("peak_counterflow_people_count", 0)
    avg_var = summary.get("avg_velocity_variance", "—")
    peak_var = summary.get("peak_velocity_variance", "—")
    avg_entropy = summary.get("avg_directional_entropy", "—")

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
          <td class="num">{st.get('right_count')}</td>
          <td class="num">{st.get('left_count')}</td>
          <td><span class="badge { 'badge-right' if dom_dir=='right' else 'badge-left' }">{_esc(dom_dir.upper())} ({dom_pct}%)</span></td>
        </tr>
        """)
    susp_table_html = "".join(susp_rows) if susp_rows else "<tr><td colspan='5' class='muted-text'>No high-flip tracks detected (excellent stability).</td></tr>"

    # Heading angle histogram chart
    max_h_bin = max([b.get("count", 0) for b in heading_hist], default=1) or 1
    hist_rows = []
    for b in heading_hist:
        lo, hi = b.get("range", [0, 0])
        cnt = b.get("count", 0)
        direction = b.get("direction", "right")
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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Crowd Motion & Safety Report — {_esc(video_name)}</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #131b2e;
      --card-hover: #18223a;
      --border: #232f48;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent: #6366f1;
      --right-color: #10b981;
      --left-color: #06b6d4;
      --crush-color: #f97316;
      --stopped-color: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      line-height: 1.6;
      padding: 32px 20px;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 28px;
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
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #10b981;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    
    /* KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
    }}
    .kpi-card::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: var(--accent);
    }}
    .kpi-card.right::before {{ background: var(--right-color); }}
    .kpi-card.left::before {{ background: var(--left-color); }}
    .kpi-card.crush::before {{ background: var(--crush-color); }}
    .kpi-card.stopped::before {{ background: var(--stopped-color); }}
    .kpi-label {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .kpi-val {{
      font-size: 28px;
      font-weight: 800;
      color: #fff;
      margin: 6px 0 2px;
      letter-spacing: -0.5px;
    }}
    .kpi-sub {{
      font-size: 12px;
      color: var(--text-dim);
    }}

    /* Distribution Bar */
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .section-title {{
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .dist-bar-wrap {{
      margin: 16px 0 12px;
    }}
    .dist-bar {{
      display: flex;
      height: 24px;
      border-radius: 8px;
      overflow: hidden;
      background: #090d16;
      border: 1px solid var(--border);
    }}
    .dist-seg {{
      height: 100%;
      transition: width 0.3s ease;
    }}
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
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th {{
      text-align: left;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    tr:last-child td {{ border-bottom: none; }}
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
    .badge-right {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .badge-left {{ background: rgba(6, 182, 212, 0.2); color: #38bdf8; }}
    .badge-crush {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; }}
    .badge-stopped {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
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
    @media (max-width: 800px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}

    footer {{
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid var(--border);
      color: var(--text-dim);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .actions a {{
      color: var(--accent);
      text-decoration: none;
      margin-left: 16px;
      font-weight: 600;
    }}
    .actions a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>

<div class="container">
  <header>
    <div class="header-title">
      <h1>📊 Crowd Motion & Safety Analysis</h1>
      <p>Source Video: <strong>{_esc(video_name)}</strong> · Model: <strong>{_esc(model_key)}</strong> · Generated: {now_str}</p>
    </div>
    <div>
      <span class="status-pill">✓ Analysis Complete</span>
    </div>
  </header>

  <!-- KPI CARDS -->
  <div class="kpi-grid">
    <div class="kpi-card left">
      <div class="kpi-label">Direction Streams</div>
      <div class="kpi-val">{primary_stream_pct:.1f}% <span style="font-size: 16px; color: var(--left-color);">{'Stream A' if has_streams else 'Moving'}</span></div>
      <div class="kpi-sub">{'vs ' + format(secondary_stream_pct, '.1f') + '% Stream B' if has_streams else 'single detected movement stream'}</div>
    </div>
    <div class="kpi-card crush">
      <div class="kpi-label">Crush Risk Detections</div>
      <div class="kpi-val">{pct_crush:.1f}%</div>
      <div class="kpi-sub">{crush_events} peak events (peak: {peak_crush_t:.1f}s)</div>
    </div>
    <div class="kpi-card stopped">
      <div class="kpi-label">Personally Stationary</div>
      <div class="kpi-val">{pct_stationary:.1f}%</div>
      <div class="kpi-sub">{pct_moving:.1f}% actively moving</div>
    </div>
    <div class="kpi-card" style="border-top: 3px solid #f59e0b;">
      <div class="kpi-label">Counter-flow Opposition</div>
      <div class="kpi-val">{pct_cf:.1f}%</div>
      <div class="kpi-sub">{cf_events} friction events (peak: {peak_cf_t:.1f}s)</div>
    </div>
    <div class="kpi-card" style="border-top: 3px solid #8b5cf6;">
      <div class="kpi-label">Directional Entropy</div>
      <div class="kpi-val">{_esc(avg_entropy)} <span style="font-size: 13px; color: var(--text-dim);">bits</span></div>
      <div class="kpi-sub">Flow disorder (0: aligned, 3: chaotic)</div>
    </div>
    <div class="kpi-card" style="border-top: 3px solid #06b6d4;">
      <div class="kpi-label">Velocity Variance</div>
      <div class="kpi-val">{_esc(avg_var)}</div>
      <div class="kpi-sub">Peak circular variance: {_esc(peak_var)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Unique Tracks & Stability</div>
      <div class="kpi-val">{_esc(total_tracks)}</div>
      <div class="kpi-sub">{_esc(stable_pct)}% stable tracks (avg {avg_flips} flips)</div>
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
        <div class="dist-seg seg-left" style="width: {primary_stream_pct}%;" title="{'Stream A' if has_streams else 'Moving'}: {primary_stream_pct:.1f}%"></div>
        {f'<div class="dist-seg seg-right" style="width: {secondary_stream_pct}%;" title="Stream B: {secondary_stream_pct:.1f}%"></div>' if has_streams else ''}
        <div class="dist-seg seg-crush" style="width: {pct_crush}%;" title="Crush Risk: {pct_crush:.1f}%"></div>
        <div class="dist-seg seg-stopped" style="width: {pct_stationary}%;" title="Stationary: {pct_stationary:.1f}%"></div>
      </div>
    </div>
    <div class="legend-grid">
      <div class="legend-item"><div class="legend-dot seg-left"></div> <strong>{'Stream A' if has_streams else 'Moving'}</strong>: {primary_stream_pct:.1f}% ({label_counts.get('person_moving_stream_a' if has_streams else 'person_moving', 0):,})</div>
      {f'<div class="legend-item"><div class="legend-dot seg-right"></div> <strong>Stream B</strong>: {secondary_stream_pct:.1f}% ({label_counts.get("person_moving_stream_b", 0):,})</div>' if has_streams else ''}
      <div class="legend-item"><div class="legend-dot seg-crush"></div> <strong>Collision / Crush Zone</strong>: {pct_crush:.1f}% ({label_counts.get('person_crush_zone', 0):,})</div>
      <div class="legend-item"><div class="legend-dot seg-stopped"></div> <strong>Stationary / Stopped</strong>: {pct_stationary:.1f}% ({label_counts.get('person_stopped', 0):,})</div>
    </div>
  </div>

  <div class="grid-2">
    <!-- LABEL & SPEED TABLE -->
    <div class="section-card">
      <div class="section-title">Class Distribution & Velocities</div>
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
            <th class="num">Right</th>
            <th class="num">Left</th>
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

  <footer>
    <div>Crowd Safety Testbed — Automated Optical Flow & Direction Analytics</div>
    <div class="actions">
      <a href="detections.json" target="_blank">📄 detections.json</a>
      <a href="detections.csv" target="_blank">📊 detections.csv</a>
      <a href="annotated.mp4" target="_blank">🎥 annotated.mp4</a>
    </div>
  </footer>
</div>

</body>
</html>
"""


def export_html_report(
    output_path: str,
    video_name: str,
    model_key: str,
    summary: dict,
    detections: list = None,
) -> str:
    """Write report.html to disk and return path."""
    html_content = generate_report_html(video_name, model_key, summary, detections)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path
