/* Crowd Safety Testbed - Frontend Controller & Detail Inspector */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  models: [],
  categories: {},
  selected: new Set(),
  threshold: 0.35,         // one confidence threshold for every selected model
  sourceTab: 'url',
  pollTimer: null,
  openJobs: new Set(),
  openHistory: new Set(),
  openAnpr: new Set(),
  openAnprAll: new Set(),
  lastActive: false,
  historyData: [],
  anprData: [],
  searchQuery: '',
  historySearchQuery: '',
  currentDetail: null,
  activeModalTab: 'overview',
  validation: null,
  validationTimer: null,
  modalOpener: null,
};

/* ------------------------------------------------------------------ utils */
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.status === 204 ? null : res.json();
}

const postJSON = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

function fmtDuration(sec) {
  if (sec == null) return '—';
  if (sec < 60) return `${sec.toFixed(0)}s`;
  const m = Math.floor(sec / 60);
  return `${m}m ${Math.round(sec - m * 60)}s`;
}

/** Screen-relative stream slot label (relative to camera view, not compass). */
function streamSlotLabel(slot, summary) {
  const name = slot === 'b' ? 'Stream B' : 'Stream A';
  const dir = summary && (slot === 'b' ? summary.stream_b_direction : summary.stream_a_direction);
  return dir ? `${name} (${dir})` : name;
}

const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ------------------------------------------------------------------ KPI Stats */
function updateKPIs(jobs = []) {
  const deviceEl = $('#kpi-device');
  if (deviceEl && state.deviceInfo) {
    deviceEl.textContent = state.deviceInfo.cuda ? 'GPU (CUDA)' : 'CPU';
  }

  const runsEl = $('#kpi-runs');
  if (runsEl) runsEl.textContent = state.historyData.length;

  const totalEvents = state.historyData.reduce((acc, g) => acc + (g.total_positives || 0), 0);
  const eventsEl = $('#kpi-events');
  if (eventsEl) eventsEl.textContent = totalEvents;

  let totalAnprVehicles = 0;
  state.anprData.forEach((g) => {
    totalAnprVehicles += (g.vehicles || []).length;
  });
  const anprEl = $('#kpi-anpr');
  if (anprEl) anprEl.textContent = totalAnprVehicles;
}

/* ------------------------------------------------------------------ device */
async function loadDevice() {
  const badge = $('#device-badge');
  try {
    const d = await api('/api/device');
    state.deviceInfo = d;
    if (d.cuda) {
      badge.textContent = `GPU · ${d.name} (${d.total_gb} GB)`;
      badge.className = 'device-badge cuda';
      if (d.total_gb < 6) {
        badge.title = 'Under 6 GB: SlowFast and I3D may run out of memory. Force CPU if a run fails.';
        badge.textContent += ' ⚠';
      }
    } else {
      badge.textContent = 'CPU only — runs will be slow';
      badge.className = 'device-badge cpu';
    }
  } catch {
    badge.textContent = 'device unknown';
  }
  updateKPIs();
}

/* ------------------------------------------------------------------ models */
async function loadModels() {
  const data = await api('/api/models');
  state.models = data.models;
  state.categories = data.categories;
  state.selected.clear();
  renderModels();

  const searchInput = $('#model-search');
  if (searchInput && !searchInput.dataset.bound) {
    searchInput.dataset.bound = 'true';
    searchInput.addEventListener('input', (e) => {
      state.searchQuery = e.target.value.toLowerCase().trim();
      renderModels();
    });
  }
}

function renderModels() {
  const host = $('#model-list');
  const order = ['fall', 'violence', 'traffic', 'anpr', 'umbrella', 'crush', 'other'];
  const query = state.searchQuery || '';

  host.innerHTML = '';

  let visibleCount = 0;

  order.forEach((cat) => {
    let items = state.models.filter((m) => m.category === cat);
    if (query) {
      items = items.filter((m) =>
        m.label.toLowerCase().includes(query) ||
        m.key.toLowerCase().includes(query) ||
        m.blurb.toLowerCase().includes(query)
      );
    }
    if (!items.length) return;

    visibleCount += items.length;
    const activeInCat = items.filter((m) => state.selected.has(m.key)).length;

    const title = document.createElement('div');
    title.className = 'cat-title';
    title.innerHTML = `<span>${esc(state.categories[cat] || cat)}</span> <span class="subtle">(${activeInCat}/${items.length})</span>`;
    host.appendChild(title);

    items.forEach((m) => {
      const checked = state.selected.has(m.key);
      const el = document.createElement('label');
      el.className = 'model' + (checked ? ' checked' : '') +
        (m.status === 'blocked' ? ' blocked' : '');
      el.innerHTML = `
        <input type="checkbox" ${checked ? 'checked' : ''}
               ${m.status === 'blocked' ? 'disabled' : ''} data-key="${m.key}"
               aria-label="${esc(m.label)} — ${esc(m.status)}" />
        <div class="model-body">
          <div class="name">${esc(m.label)}
            <span class="pill ${m.status}">${m.status}</span></div>
          <div class="blurb">${esc(m.blurb)}</div>
          ${m.comparable_threshold === false ? `<div class="note fallback">Runs at its own threshold (${m.default_threshold != null ? m.default_threshold.toFixed(2) : 'default'
          }) — its scores are not on the same scale as the other detectors, so the run-wide threshold does not apply.</div>` : ''}
          ${m.note ? `<div class="note ${m.status}">${esc(m.note)}</div>` : ''}
        </div>`;

      const cb = el.querySelector('input[type=checkbox]');
      cb.addEventListener('change', (e) => {
        e.stopPropagation();
        if (cb.checked) {
          state.selected.add(m.key);
          el.classList.add('checked');
        } else {
          state.selected.delete(m.key);
          el.classList.remove('checked');
        }
        $('#selected-count').textContent = `${state.selected.size} selected of ${state.models.length}`;
        const catHeader = el.previousElementSibling;
        const catItems = state.models.filter((x) => x.category === cat);
        const catActive = catItems.filter((x) => state.selected.has(x.key)).length;
        if (catHeader && catHeader.classList.contains('cat-title')) {
          catHeader.innerHTML = `<span>${esc(state.categories[cat] || cat)}</span> <span class="subtle">(${catActive}/${catItems.length})</span>`;
        }
      });

      host.appendChild(el);
    });
  });

  if (query && visibleCount === 0) {
    host.innerHTML = `<div class="empty">No models match "${esc(query)}"</div>`;
  }

  $('#selected-count').textContent = `${state.selected.size} selected of ${state.models.length}`;
  host.setAttribute('aria-busy', 'false');
}

/* ------------------------------------------------------------------ videos */
async function loadVideos() {
  const sel = $('#video-select');
  try {
    const { videos } = await api('/api/videos');
    if (!videos.length) {
      sel.innerHTML = '<option value="">No videos in test_videos/ yet</option>';
      return;
    }
    sel.innerHTML = videos
      .map((v) => `<option value="${esc(v.name)}">${esc(v.name)} — ${v.size_mb} MB</option>`)
      .join('');
  } catch {
    sel.innerHTML = '<option value="">Could not list videos</option>';
  }
}

/* ------------------------------------------------------------------- jobs */
function stageRow(job, s) {
  const model = state.models.find((m) => m.key === s.model_key);
  const name = model ? model.label : s.model_key;
  const pct = Math.round(s.progress * 100);

  let progressCell;
  if (s.status === 'queued') {
    progressCell = `<span class="status queued">⏳ queued — waiting for another model to finish</span>`;
  } else if (s.status === 'running' || s.status === 'loading') {
    progressCell = `<div class="bar"><i style="width:${pct}%"></i></div>
                    <div class="meta">${s.status === 'loading' ? 'loading model…' : `${pct}%`}</div>`;
  } else {
    progressCell = `<span class="status ${s.status}">${s.status}</span>`;
  }

  if (s.status === 'failed') {
    return `<tr>
      <td>${esc(name)}</td>
      <td>${progressCell}</td>
      <td colspan="4" class="err">${esc(s.error || 'failed')}</td>
    </tr>`;
  }

  const fallbackTag = s.scoring_modes && s.scoring_modes.geometric_fallback
    ? '<div class="scoring-tag fallback">geometric fallback</div>'
    : (s.scoring_modes && s.scoring_modes.kinetics_zeroshot
      ? '<div class="scoring-tag zeroshot">kinetics zero-shot</div>' : '');

  const posClass = s.positives > 0 ? 'pos' : 'zero';
  return `<tr>
    <td class="col-model"><span class="model-name">${esc(name)}</span>${fallbackTag}</td>
    <td class="col-progress">${progressCell}</td>
    <td class="num col-events ${posClass}">${s.positives}</td>
    <td class="num col-rows">${s.detections}</td>
    <td class="num col-time">${fmtDuration(s.elapsed_sec)}</td>
    <td class="col-actions">
      <button class="btn-card-detail" data-inspect-job="${esc(job.id)}" data-inspect-model="${esc(s.model_key)}">Inspect Details</button>
    </td>
  </tr>`;
}

function jobCard(job) {
  const open = state.openJobs.has(job.id);
  const done = job.stages.filter((s) => ['done', 'failed', 'cancelled'].includes(s.status)).length;
  const cancellable = !['done', 'failed', 'cancelled'].includes(job.status);

  return `<div class="job" data-job="${job.id}">
    <div class="job-head" data-toggle="${job.id}">
      <div class="job-head-left">
        <span class="pill ${job.status}"><span class="status-dot"></span>${job.status.toUpperCase()}</span>
        <span class="title">${esc(job.video_name || job.source)}</span>
      </div>
      <div class="job-head-right">
        <div class="job-chips">
          <span class="meta-chip">📦 ${done}/${job.stages.length} models</span>
          <span class="meta-chip">⚡ Stride ${job.sample_every_n_frames}</span>
          <span class="meta-chip">⏱️ ${fmtDuration(job.elapsed_sec)}</span>
        </div>
        ${cancellable ? `<button class="btn-cancel" data-cancel="${job.id}">✕ Cancel</button>` : ''}
        <span class="toggle-icon">${open ? '▾' : '▸'}</span>
      </div>
    </div>
    ${open ? `<div class="job-body">
      ${job.message || job.error ? `<div class="job-msg">${esc(job.message || '')}${job.error ? ` — ${esc(job.error)}` : ''}</div>` : ''}
      <table class="stages">
        <thead><tr>
          <th class="col-model">Model</th>
          <th class="col-progress">Progress</th>
          <th class="u-num col-events">Events</th>
          <th class="u-num col-rows">Rows</th>
          <th class="u-num col-time">Time</th>
          <th class="col-actions">Actions</th>
        </tr></thead>
        <tbody>${job.stages.map((s) => stageRow(job, s)).join('')}</tbody>
      </table>
    </div>` : ''}
  </div>`;
}

async function refreshJobs() {
  const btn = $('#refresh-jobs');
  if (btn) btn.classList.add('spinning');
  let jobs = [];
  try { ({ jobs } = await api('/api/jobs')); } catch { if (btn) btn.classList.remove('spinning'); return; }
  if (btn) btn.classList.remove('spinning');

  const host = $('#jobs');
  if (!jobs.length) {
    host.innerHTML = '<div class="empty">No runs yet. Select models above and click Run.</div>';
    return;
  }
  if (jobs.length && !state.openJobs.size) state.openJobs.add(jobs[0].id);

  host.innerHTML = jobs.map(jobCard).join('');

  host.querySelectorAll('[data-toggle]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      if (ev.target.dataset.cancel || ev.target.dataset.inspectJob) return;
      const id = el.dataset.toggle;
      state.openJobs.has(id) ? state.openJobs.delete(id) : state.openJobs.add(id);
      refreshJobs();
    });
  });

  host.querySelectorAll('[data-cancel]').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      try { await postJSON(`/api/jobs/${el.dataset.cancel}/cancel`, {}); } catch { }
      refreshJobs();
    });
  });

  host.querySelectorAll('[data-inspect-job]').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const jobId = el.dataset.inspectJob;
      const modelKey = el.dataset.inspectModel;
      openJobDetailModal(jobId, modelKey);
    });
  });

  const active = jobs.some((j) => !['done', 'failed', 'cancelled'].includes(j.status));
  if (state.lastActive && !active) { refreshHistory(); refreshAnpr(); }
  state.lastActive = active;

  clearTimeout(state.pollTimer);
  if (active) state.pollTimer = setTimeout(refreshJobs, 1200);
}

/* ---------------------------------------------------------------- history cards */
function historyOutputCard(group) {
  const when = new Date(group.modified_at * 1000).toLocaleString();
  const stagesCount = group.stages.length;
  const hasEvents = group.total_positives > 0;

  // Primary Category determination
  let primaryCat = 'other';
  if (group.stages.some(s => s.model_key.includes('fall'))) primaryCat = 'fall';
  else if (group.stages.some(s => s.model_key.includes('anpr'))) primaryCat = 'anpr';
  else if (group.stages.some(s => s.model_key.includes('violence') || s.model_key.includes('altercation'))) primaryCat = 'violence';
  else if (group.stages.some(s => s.model_key.includes('crush') || s.model_key.includes('motion') || s.model_key.includes('flow') || s.model_key.includes('traffic'))) primaryCat = 'traffic';

  const catLabel = primaryCat === 'anpr' ? 'ANPR Read' : (primaryCat === 'traffic' ? 'Crowd / Motion' : (primaryCat.charAt(0).toUpperCase() + primaryCat.slice(1)));

  // Build model summaries with inline visual breakdown
  const modelPills = group.stages.map((s) => {
    const sum = s.summary || {};
    const tag = s.scoring_modes && s.scoring_modes.geometric_fallback
      ? '<span class="pill fallback">Geometric Fallback</span>'
      : (s.scoring_modes && s.scoring_modes.kinetics_zeroshot
        ? '<span class="pill ready">Kinetics Zero-Shot</span>' : '');

    let motionBar = '';
    if (sum && (sum.pct_moving_single_stream != null || sum.pct_moving_stream_a != null || sum.pct_crush_risk != null)) {
      const pSingle = sum.pct_moving_single_stream || 0;
      const pStreamA = sum.pct_moving_stream_a || 0;
      const pStreamB = sum.pct_moving_stream_b || 0;
      const hasStreams = pStreamA > 0 || pStreamB > 0;
      const pCrush = sum.pct_crush_risk || 0;
      const pStop = sum.pct_stationary || 0;
      const labelA = streamSlotLabel('a', sum);
      const labelB = streamSlotLabel('b', sum);
      motionBar = `
        <div class="card-dist-bar-wrap">
          <div class="card-dist-bar">
            ${hasStreams ? `<div class="card-dist-seg seg-left" style="width: ${pStreamA}%;" title="${esc(labelA)}: ${pStreamA}% (direction relative to camera view)"></div><div class="card-dist-seg seg-right" style="width: ${pStreamB}%;" title="${esc(labelB)}: ${pStreamB}% (direction relative to camera view)"></div>` : `<div class="card-dist-seg seg-left" style="width: ${pSingle}%;" title="Moving: ${pSingle}%"></div>`}
            <div class="card-dist-seg seg-crush" style="width: ${pCrush}%;" title="Crush: ${pCrush}%"></div>
            <div class="card-dist-seg seg-stopped" style="width: ${pStop}%;" title="Stopped: ${pStop}%"></div>
          </div>
          <div class="card-dist-labels">
            ${hasStreams ? `<span class="c-lbl left">${pStreamA.toFixed(0)}% ${esc(labelA)}</span><span class="c-lbl right">${pStreamB.toFixed(0)}% ${esc(labelB)}</span>` : `<span class="c-lbl left">${pSingle.toFixed(0)}% Moving</span>`}
            <span class="c-lbl crush">⚠️ ${pCrush.toFixed(0)}% Crush</span>
            <span class="c-lbl stop">⏹ ${pStop.toFixed(0)}% Stop</span>
          </div>
        </div>
      `;
    }

    const reportBtn = s.report_html
      ? `<a href="/api/files/run/${esc(s.report_html)}" target="_blank" class="report-btn" onclick="event.stopPropagation();">📄 HTML Report</a>`
      : '';

    return `<div class="model-summary-box">
      <div class="u-row">
        <span class="u-strong">${esc(s.model_label)}</span>
        <span class="${s.positives > 0 ? 'pos' : 'zero'} u-strong">(${s.positives} alerts)</span>
        ${tag}
        ${reportBtn}
      </div>
      ${motionBar}
    </div>`;
  }).join('');

  return `<div class="output-card ${hasEvents ? 'has-alert' : ''}" data-hist-card="${esc(group.video)}">
    <div class="card-head">
      <div>
        <div class="card-title">${esc(group.video)}</div>
        <div class="card-subtitle">Saved Run · ${when}</div>
      </div>
      <span class="category-tag ${primaryCat}">${esc(catLabel)}</span>
    </div>

    <div class="card-stats-row">
      <div class="stat-item">
        <span class="val ${hasEvents ? 'pos' : 'zero'}">${group.total_positives}</span>
        <span class="lbl">Positive Alerts</span>
      </div>
      <div class="stat-item">
        <span class="val">${stagesCount}</span>
        <span class="lbl">Models Scored</span>
      </div>
    </div>

    <div>
      <div class="u-label">Models & Analytics Breakdown</div>
      ${modelPills}
    </div>

    <div class="card-foot">
      <div class="card-actions">
        <button class="btn-card-detail" data-inspect-hist="${esc(group.video)}">🔍 Inspect Full Details</button>
      </div>
      <button class="link-btn danger-link" data-delete-history="${esc(group.video)}">Delete</button>
    </div>
  </div>`;
}

async function refreshHistory() {
  const btn = $('#refresh-history');
  if (btn) btn.classList.add('spinning');
  const host = $('#history');
  let history;
  try {
    ({ history } = await api('/api/history'));
    state.historyData = history || [];
    updateKPIs();
  } catch (e) {
    host.innerHTML = `<div class="err">${esc(e.message)}</div>`;
    if (btn) btn.classList.remove('spinning');
    return;
  }
  if (btn) btn.classList.remove('spinning');

  if (!state.historyData.length) {
    host.innerHTML = '<div class="empty">Nothing in outputs/ yet. Run a model above to see generated cards.</div>';
    return;
  }

  let filtered = state.historyData;
  if (state.historySearchQuery) {
    const q = state.historySearchQuery.toLowerCase();
    filtered = filtered.filter(g =>
      g.video.toLowerCase().includes(q) ||
      g.stages.some(s => s.model_label.toLowerCase().includes(q) || s.model_key.toLowerCase().includes(q))
    );
  }

  if (!filtered.length) {
    host.innerHTML = `<div class="empty">No output cards match "${esc(state.historySearchQuery)}"</div>`;
    return;
  }

  host.innerHTML = filtered.map(historyOutputCard).join('');

  host.querySelectorAll('[data-inspect-hist]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.stopPropagation();
      openHistoryDetailModal(el.dataset.inspectHist);
    });
  });

  host.querySelectorAll('[data-hist-card]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      if (ev.target.dataset.deleteHistory) return;
      openHistoryDetailModal(el.dataset.histCard);
    });
  });

  host.querySelectorAll('[data-delete-history]').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      const v = el.dataset.deleteHistory;
      if (!confirm(`Permanently delete all saved outputs for "${v}"?`)) return;
      try {
        await api(`/api/history/${encodeURIComponent(v)}`, { method: 'DELETE' });
        refreshHistory();
        refreshAnpr();
      } catch (err) {
        alert(`Could not delete "${v}": ${err.message}`);
      }
    });
  });
}

/* ------------------------------------------------------------------- ANPR gallery */
const GALLERY_PREVIEW = 10;

function makeTile(v, video) {
  const img = v.image
    ? `<img src="/api/anpr/${encodeURIComponent(video)}/vehicles/${encodeURIComponent(v.image)}" alt="" loading="lazy" onerror="this.onerror=null; this.outerHTML='<div class=\\'noimg\\'>no image</div>';" />`
    : '<div class="noimg">no image</div>';
  const plate = v.plate
    ? `<div class="plate">${esc(v.plate_display || v.plate)}</div>`
    : `<div class="plate none">${esc(statusText(v.plate_status, v.plate_width_px))}</div>`;
  return `<figure class="vcard${v.plate ? '' : ' unread'}" data-anpr-vcard="${esc(video)}" data-anpr-idx="${v.first_seen_sec}">
    ${img}
    <figcaption>
      <div class="vname">${esc(v.caption || v.vehicle_class)}</div>
      ${plate}
      <div class="vmeta">${v.first_seen_sec}s–${v.last_seen_sec}s · ${v.frames_seen} frames${v.plate ? ` · ${Math.round(v.plate_agreement * 100)}% agree` : ''}</div>
    </figcaption>
  </figure>`;
}

function anprCard(g) {
  const c = g.counts || {};
  const withPlate = g.vehicles.filter((v) => v.plate);
  const without = g.vehicles.filter((v) => !v.plate);
  const ordered = [...withPlate, ...without];
  const total = ordered.length;
  const isExpanded = state.openAnprAll.has(g.video);
  const visible = isExpanded ? ordered : ordered.slice(0, GALLERY_PREVIEW);
  const hiddenCount = total - GALLERY_PREVIEW;

  const tiles = visible.map((v) => makeTile(v, g.video)).join('');
  const showAllBtn = !isExpanded && hiddenCount > 0
    ? `<button class="gallery-more" data-show-all-anpr="${esc(g.video)}">
        ▼ Show all ${total} vehicles (${hiddenCount} more)
       </button>`
    : (isExpanded && total > GALLERY_PREVIEW
      ? `<button class="gallery-more" data-show-all-anpr="${esc(g.video)}">
            ▲ Show fewer
           </button>`
      : '');

  return `<div class="job">
    <div class="job-head" data-anpr-toggle="${esc(g.video)}">
      <span class="status ${withPlate.length ? 'done' : 'cancelled'}">${withPlate.length} read</span>
      <span class="title">${esc(g.video)}</span>
      <span class="meta">${total} vehicle(s) captured · ${withPlate.length} with a plate</span>
      <button class="link-btn danger-link" data-delete-anpr="${esc(g.video)}">delete</button>
      <span class="meta">${state.openAnpr.has(g.video) ? '▾' : '▸'}</span>
    </div>
    ${state.openAnpr.has(g.video) ? `<div class="job-body">
      ${withPlate.length === 0 ? `<div class="warnbox">No plate was legible in this video.
        ${c.too_small ? `${c.too_small} plate(s) were detected but too small to read` : ''}
        — ANPR needs roughly 90px of plate width for character resolution.</div>` : ''}
      <div class="gallery">${tiles}</div>
      ${showAllBtn}
    </div>` : ''}
  </div>`;
}

function statusText(status, width) {
  if (status === 'too_small') return `plate too small${width ? ` (${width}px)` : ''}`;
  if (status === 'unreadable') return 'plate unreadable';
  if (status === 'no_plate_found') return 'no plate visible';
  return 'no plate';
}

async function refreshAnpr() {
  const btn = $('#refresh-anpr');
  if (btn) btn.classList.add('spinning');
  const host = $('#anpr');
  let galleries;
  try {
    ({ galleries } = await api('/api/anpr'));
    state.anprData = galleries || [];
    updateKPIs();
  }
  catch (e) { host.innerHTML = `<div class="err">${esc(e.message)}</div>`; if (btn) btn.classList.remove('spinning'); return; }
  if (btn) btn.classList.remove('spinning');

  if (!state.anprData.length) {
    host.innerHTML = '<div class="empty">No ANPR runs yet. Select the ANPR model and run a video.</div>';
    return;
  }
  if (!state.openAnpr.size) state.openAnpr.add(state.anprData[0].video);

  host.innerHTML = state.anprData.map(anprCard).join('');

  host.querySelectorAll('[data-anpr-toggle]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      if (ev.target.dataset.deleteAnpr || ev.target.dataset.showAllAnpr) return;
      const v = el.dataset.anprToggle;
      state.openAnpr.has(v) ? state.openAnpr.delete(v) : state.openAnpr.add(v);
      refreshAnpr();
    });
  });

  host.querySelectorAll('[data-show-all-anpr]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      const v = el.dataset.showAllAnpr;
      state.openAnprAll.has(v) ? state.openAnprAll.delete(v) : state.openAnprAll.add(v);
      refreshAnpr();
    });
  });

  host.querySelectorAll('[data-delete-anpr]').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      const v = el.dataset.deleteAnpr;
      if (!confirm(`Permanently delete ANPR gallery for "${v}"?`)) return;
      try {
        await api(`/api/anpr/${encodeURIComponent(v)}`, { method: 'DELETE' });
        state.openAnpr.delete(v);
        state.openAnprAll.delete(v);
        refreshAnpr();
      } catch (err) {
        alert(`Could not delete ANPR gallery for "${v}": ${err.message}`);
      }
    });
  });
}

/* ------------------------------------------------------------------ DETAILED MODAL */
async function openHistoryDetailModal(videoName) {
  state.modalOpener = document.activeElement;
  const group = state.historyData.find(g => g.video === videoName);
  if (!group) return;

  const stage = group.stages[0]; // Primary stage for overview preview
  const title = `Classification Details — ${videoName}`;

  $('#modal-title').textContent = title;
  $('#modal-body').innerHTML = '<div class="loading">Loading detailed detection payload…</div>';
  $('#modal').classList.remove('hidden');

  // Collect ALL annotated videos from every stage that has one
  const allAnnotatedVideos = group.stages
    .filter(s => s.annotated)
    .map(s => ({ label: s.model_label, key: s.model_key, file: s.annotated }));

  try {
    let detections = { rows: [], total: 0 };
    if (stage) {
      detections = await api(`/api/history/${encodeURIComponent(videoName)}/${encodeURIComponent(stage.model_key)}/detections?limit=500`);
    }

    state.currentDetail = {
      videoName,
      group,
      primaryStage: stage,
      allStages: group.stages,           // all stages for timeline model selector
      activeTimelineModelKey: stage ? stage.model_key : null,
      detections,
      allAnnotatedVideos,
      activeAnnotatedVideo: allAnnotatedVideos.length > 0 ? allAnnotatedVideos[0].file : null,
    };

    renderModalTab('overview');
  } catch (err) {
    $('#modal-body').innerHTML = `<div class="err">${esc(err.message)}</div>`;
  }
}

async function openJobDetailModal(jobId, modelKey) {
  state.modalOpener = document.activeElement;
  $('#modal-title').textContent = `Job Details — ${jobId}`;
  $('#modal-body').innerHTML = '<div class="loading">Loading detections…</div>';
  $('#modal').classList.remove('hidden');

  try {
    const job = await api(`/api/jobs/${jobId}`);
    const stage = job && job.stages ? job.stages.find(s => s.model_key === modelKey) : null;
    const detections = await api(`/api/jobs/${jobId}/detections/${modelKey}?limit=500`);
    state.currentDetail = {
      videoName: (job && job.video_name) ? job.video_name : jobId,
      group: null,
      primaryStage: stage || { model_key: modelKey, model_label: modelKey },
      allStages: (job && job.stages) ? job.stages : [],
      activeTimelineModelKey: modelKey,
      detections,
      allAnnotatedVideos: stage && stage.annotated ? [{ label: stage.model_key, key: stage.model_key, file: stage.annotated }] : [],
      activeAnnotatedVideo: stage ? stage.annotated : null,
    };
    renderModalTab('overview');
  } catch (err) {
    $('#modal-body').innerHTML = `<div class="err">${esc(err.message)}</div>`;
  }
}

function renderModalTab(tabName) {
  state.activeModalTab = tabName;
  $$('#modal-nav .modal-tab-btn').forEach(btn => {
    const isActive = btn.dataset.mtab === tabName;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  const detail = state.currentDetail;
  if (!detail) return;

  const host = $('#modal-body');

  if (tabName === 'overview') {
    const g = detail.group;
    const s = detail.primaryStage || {};
    const d = detail.detections || {};
    const sum = s.summary || {};
    const totalPositives = g ? g.total_positives : (d.rows ? d.rows.filter(r => r.confidence > 0.5).length : 0);
    const maxConf = d.rows && d.rows.length ? Math.max(...d.rows.map(r => r.confidence || 0)) : 0;

    let analyticsHtml = '';
    if (sum && (sum.pct_moving != null || sum.pct_crush_risk != null || sum.pct_moving_single_stream != null)) {
      const pSingle = sum.pct_moving_single_stream || 0;
      const pStreamA = sum.pct_moving_stream_a || 0;
      const pStreamB = sum.pct_moving_stream_b || 0;
      const hasStreams = pStreamA > 0 || pStreamB > 0;
      const pCrush = sum.pct_crush_risk || 0;
      const pStop = sum.pct_stationary || 0;
      const pMoving = sum.pct_moving || (100 - pStop);
      const labelA = streamSlotLabel('a', sum);
      const labelB = streamSlotLabel('b', sum);
      const totalTracks = sum.total_tracks != null ? sum.total_tracks : '—';
      const stablePct = sum.stable_tracks_pct != null ? sum.stable_tracks_pct : '—';
      const crushEvents = sum.crush_event_count || 0;
      const peakCrushT = sum.peak_crush_timestamp_sec || 0;
      const peakCrushPeople = sum.peak_crush_people_count || 0;
      const pCounterflow = sum.pct_counterflow_people || 0;
      const cfEvents = sum.counterflow_events_count || 0;
      const peakCfT = sum.peak_counterflow_timestamp_sec || 0;
      const avgEntropy = sum.avg_directional_entropy != null ? `${sum.avg_directional_entropy} <span style="font-size: 12px; color: var(--muted); font-weight: normal;">bits</span>` : '<span class="no-data-text">No data</span>';
      const avgVar = sum.avg_velocity_variance != null ? sum.avg_velocity_variance : '<span class="no-data-text">No data</span>';

      const flowCurrent = sum.specific_flow_current != null ? sum.specific_flow_current.toFixed(2) : '<span class="no-data-text">No data</span>';
      const flowSub = sum.specific_flow_peak != null ? `peak ${sum.specific_flow_peak.toFixed(2)} ${sum.specific_flow_units || ''}` : '<span class="no-data-badge">Unconfigured</span>';

      const oscAvg = sum.oscillation_symmetry_avg != null ? sum.oscillation_symmetry_avg.toFixed(2) : '<span class="no-data-text">No data</span>';
      const oscSub = sum.oscillation_symmetry_peak != null ? `peak ${sum.oscillation_symmetry_peak.toFixed(2)}; ${oscZones} threshold samples` : '<span class="no-data-badge">Unconfigured</span>';

      analyticsHtml = `
        <div class="overview-analytics-box">
          <div class="section-title">
            <span>📈 Run Analytics & Crowd Dynamics</span>
            ${s.report_html ? `<a href="/api/files/run/${esc(s.report_html)}" target="_blank" class="btn-card-detail">📄 View Standalone HTML Report</a>` : ''}
          </div>

          <div class="overview-kpis">
            <div class="overview-kpi-item tier-one">
              <div class="overview-kpi-lbl">Crush Risk Level</div>
              <div class="overview-kpi-val" style="color: #fb923c;">${pCrush.toFixed(1)}%</div>
              <div class="overview-kpi-sub">${crushEvents} peak events (max ${peakCrushPeople} people @ ${peakCrushT.toFixed(1)}s)</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Direction Streams</div>
              <div class="overview-kpi-val" style="color: #38bdf8;">${hasStreams ? `A ${pStreamA.toFixed(1)}% <span style="font-size: 13px; color: var(--muted); font-weight: normal;">/ B ${pStreamB.toFixed(1)}%</span>` : `${pSingle.toFixed(1)}%`}</div>
              <div class="overview-kpi-sub">${hasStreams ? `${esc(labelA)}: ${(sum.label_counts?.person_moving_stream_a || 0).toLocaleString()} · ${esc(labelB)}: ${(sum.label_counts?.person_moving_stream_b || 0).toLocaleString()}` : `Single moving stream: ${(sum.label_counts?.person_moving || 0).toLocaleString()}`}</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Counter-Flow Friction</div>
              <div class="overview-kpi-val" style="color: #f59e0b;">${pCounterflow.toFixed(1)}%</div>
              <div class="overview-kpi-sub">${cfEvents} friction events (peak @ ${peakCfT.toFixed(1)}s)</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Specific Flow</div>
              <div class="overview-kpi-val" style="color: #22d3ee;">${flowCurrent}</div>
              <div class="overview-kpi-sub">${flowSub}</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Oscillation Symmetry</div>
              <div class="overview-kpi-val" style="color: #f472b6;">${oscAvg}</div>
              <div class="overview-kpi-sub">${oscSub}</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Movement Rate</div>
              <div class="overview-kpi-val" style="color: #34d399;">${pMoving.toFixed(1)}%</div>
              <div class="overview-kpi-sub">Stationary / Stopped: ${pStop.toFixed(1)}% (${sum.label_counts ? (sum.label_counts.person_stopped || 0).toLocaleString() : 0})</div>
            </div>
            <div class="overview-kpi-item tier-three">
              <div class="overview-kpi-lbl">Directional Entropy</div>
              <div class="overview-kpi-val" style="color: #a78bfa;">${avgEntropy}</div>
              <div class="overview-kpi-sub">Local vector disorder (0: aligned, 3: chaotic)</div>
            </div>
            <div class="overview-kpi-item tier-three">
              <div class="overview-kpi-lbl">Velocity Variance</div>
              <div class="overview-kpi-val" style="color: #06b6d4;">${avgVar}</div>
              <div class="overview-kpi-sub">Track Integrity: ${totalTracks} tracks (${stablePct}% stable)</div>
            </div>
          </div>

          <div class="u-label">Crowd Velocity & Flow Share</div>
          <div class="card-dist-bar-wrap" style="margin-top: 6px;">
            <div class="card-dist-bar" style="height: 16px; border-radius: 6px;">
              ${hasStreams ? `<div class="card-dist-seg seg-left" style="width: ${pStreamA}%;" title="${esc(labelA)}: ${pStreamA}% (direction relative to camera view)"></div><div class="card-dist-seg seg-right" style="width: ${pStreamB}%;" title="${esc(labelB)}: ${pStreamB}% (direction relative to camera view)"></div>` : `<div class="card-dist-seg seg-left" style="width: ${pSingle}%;" title="Moving: ${pSingle}%"></div>`}
              <div class="card-dist-seg seg-crush" style="width: ${pCrush}%;" title="Crush Risk: ${pCrush}%"></div>
              <div class="card-dist-seg seg-stopped" style="width: ${pStop}%;" title="Stopped: ${pStop}%"></div>
            </div>
          </div>
          <div class="legend-grid" style="margin-top: 10px;">
            ${hasStreams ? `<div class="legend-item"><div class="legend-dot seg-left"></div> <strong>${esc(labelA)}</strong>: ${pStreamA.toFixed(1)}%</div><div class="legend-item"><div class="legend-dot seg-right"></div> <strong>${esc(labelB)}</strong>: ${pStreamB.toFixed(1)}%</div>` : `<div class="legend-item"><div class="legend-dot seg-left"></div> <strong>Moving</strong>: ${pSingle.toFixed(1)}%</div>`}
            <div class="legend-item"><div class="legend-dot seg-crush"></div> <strong>Crush Zone</strong>: ${pCrush.toFixed(1)}%</div>
            <div class="legend-item"><div class="legend-dot seg-stopped"></div> <strong>Stationary</strong>: ${pStop.toFixed(1)}%</div>
          </div>
          ${hasStreams ? '<div class="hint" style="margin-top: 8px;">Stream directions are relative to camera view, not geographic compass headings.</div>' : ''}
        </div>
      `;
    }

    const labelsList = s.label_counts
      ? Object.entries(s.label_counts).map(([k, v]) => `<span class="plate-badge u-mr-1">${esc(k)}: ${v}</span>`).join(' ')
      : 'None';

    host.innerHTML = `
      <div class="detail-overview-grid">
        <div class="detail-metric-card">
          <div class="val ${totalPositives > 0 ? 'highlight' : ''}">${totalPositives}</div>
          <div class="lbl">Positive Alert Events</div>
        </div>
        <div class="detail-metric-card">
          <div class="val">${d.total || (s.detections || 0)}</div>
          <div class="lbl">Total Output Rows</div>
        </div>
        <div class="detail-metric-card">
          <div class="val">${maxConf > 0 ? `${(maxConf * 100).toFixed(1)}%` : '—'}</div>
          <div class="lbl">Peak Confidence Score</div>
        </div>
      </div>

      ${analyticsHtml}

      <div class="detail-section-title">Classifications & Metadata</div>
      <table class="detail-info-table">
        <tbody>
          <tr><td class="key">Source Video</td><td class="val">${esc(detail.videoName)}</td></tr>
          <tr><td class="key">Primary Model</td><td class="val">${esc(s.model_label || s.model_key || '—')}</td></tr>
          <tr><td class="key">Category</td><td class="val">${esc(s.model_key ? s.model_key.split('_')[0] : 'general')}</td></tr>
          <tr><td class="key">Detected Labels</td><td class="val">${labelsList}</td></tr>
          ${s.modified_at ? `<tr><td class="key">Run Date</td><td class="val">${new Date(s.modified_at * 1000).toLocaleString()}</td></tr>` : ''}
          <tr>
            <td class="key">Artifacts</td>
            <td class="val" style="display: flex; flex-wrap: wrap; gap: 8px;">
              ${s.report_html ? `<a href="/api/files/run/${esc(s.report_html)}" target="_blank" class="link-btn">📄 Standalone Report (HTML)</a>` : ''}
              ${s.log_json ? `<a href="/api/files/run/${esc(s.log_json)}" target="_blank" class="link-btn">📄 Detections (JSON)</a>` : ''}
              ${s.log_csv ? `<a href="/api/files/run/${esc(s.log_csv)}" target="_blank" class="link-btn">📊 Detections (CSV)</a>` : ''}
              ${s.log_summary ? `<a href="/api/files/run/${esc(s.log_summary)}" target="_blank" class="link-btn">📈 Summary Stats (JSON)</a>` : ''}
            </td>
          </tr>
        </tbody>
      </table>
    `;
  } else if (tabName === 'timeline') {
    const allStages = detail.allStages;

    // Single model or job context — show detections directly
    if (!allStages || allStages.length <= 1) {
      const modelLabel = detail.primaryStage ? (detail.primaryStage.model_label || detail.primaryStage.model_key) : '';
      renderDetectionsTable(detail.detections, modelLabel, host);
      return;
    }

    // Multiple models — show selector table + detections below
    const activeKey = detail.activeTimelineModelKey || allStages[0].model_key;

    const selectorRows = allStages.map(s => {
      const isActive = s.model_key === activeKey;
      return `<tr class="video-select-row u-clickable${isActive ? ' active-video-row is-active' : ''}" data-load-model="${esc(s.model_key)}" data-model-label="${esc(s.model_label)}">
        <td class="u-strong">${esc(s.model_label)}</td>
        <td class="${s.positives > 0 ? 'pos' : 'zero'} u-strong">${s.positives} alerts</td>
        <td>${s.detections} rows</td>
        <td><span class="u-accent">${isActive ? '⬤ Viewing' : '▶ Load'}</span></td>
      </tr>`;
    }).join('');

    host.innerHTML = `
      <div class="detail-section-title u-mt-0">⏱️ Select a Model to View Detections</div>
      <p class="hint u-mb-3">${allStages.length} models scored this video — click any row to load its detection rows below.</p>
      <table class="stages">
        <thead><tr><th>Model</th><th class="u-num">Alerts</th><th class="u-num">Total Rows</th><th></th></tr></thead>
        <tbody>${selectorRows}</tbody>
      </table>
      <div id="timeline-detections-host"><div class="loading">Loading detections…</div></div>
    `;

    // Wire row clicks to load that model's detections
    async function loadModelDetections(modelKey, modelLabel) {
      detail.activeTimelineModelKey = modelKey;

      // Update row highlights
      host.querySelectorAll('.video-select-row').forEach(r => {
        const isNow = r.dataset.loadModel === modelKey;
        r.style.background = isNow ? 'rgba(99,102,241,0.15)' : '';
        const action = r.querySelector('span');
        if (action) action.textContent = isNow ? '⬤ Viewing' : '▶ Load';
      });

      const detectHost = $('#timeline-detections-host');
      if (!detectHost) return;
      detectHost.innerHTML = '<div class="loading">Loading detections for ' + esc(modelLabel) + '…</div>';

      try {
        const d = await api(`/api/history/${encodeURIComponent(detail.videoName)}/${encodeURIComponent(modelKey)}/detections?limit=500`);
        renderDetectionsTable(d, modelLabel, detectHost);
      } catch (err) {
        if ($('#timeline-detections-host')) {
          $('#timeline-detections-host').innerHTML = `<div class="err">${esc(err.message)}</div>`;
        }
      }
    }

    host.querySelectorAll('[data-load-model]').forEach(row => {
      row.addEventListener('click', () => {
        loadModelDetections(row.dataset.loadModel, row.dataset.modelLabel);
      });
    });

    // Auto-load the active model immediately
    loadModelDetections(activeKey, allStages.find(s => s.model_key === activeKey)?.model_label || activeKey);
  } else if (tabName === 'video') {
    const allVids = detail.allAnnotatedVideos || [];
    if (allVids.length === 0) {
      host.innerHTML = '<div class="empty">No annotated video available for this run (export video was disabled or no output written).</div>';
    } else if (allVids.length === 1) {
      // Single video — just play it directly
      host.innerHTML = `
        <video controls autoplay src="/api/files/run/${esc(allVids[0].file)}"></video>
        <p class="hint u-mt-3"><strong>${esc(allVids[0].label)}</strong> — annotated video output with bounding box overlays.</p>
      `;
    } else {
      // Multiple videos — show selection table then player
      const currentFile = detail.activeAnnotatedVideo || allVids[0].file;
      const currentEntry = allVids.find(v => v.file === currentFile) || allVids[0];

      const tableRows = allVids.map((v) => {
        const isActive = v.file === currentFile;
        return `<tr class="video-select-row u-clickable${isActive ? ' active-video-row is-active' : ''}" data-play-video="${esc(v.file)}" data-play-label="${esc(v.label)}">
          <td class="u-strong">${esc(detail.videoName)}</td>
          <td class="u-muted">${esc(v.label)}</td>
          <td class="u-mono-sm">${esc(v.file)}</td>
          <td><span class="u-accent">${isActive ? '▶ Playing' : '▶ Play'}</span></td>
        </tr>`;
      }).join('');

      host.innerHTML = `
        <div class="detail-section-title u-mt-0">🎬 Select a Model Output Video</div>
        <p class="hint u-mb-3">${allVids.length} annotated outputs available — click any row to switch playback.</p>
        <table class="stages">
          <thead><tr>
            <th>Video File</th>
            <th>Model</th>
            <th>Filename</th>
            <th></th>
          </tr></thead>
          <tbody>${tableRows}</tbody>
        </table>
        <div class="u-caption">Now playing: <strong class="u-strong" id="video-now-playing">${esc(currentEntry.label)}</strong></div>
        <video id="modal-video-player" controls autoplay src="/api/files/run/${esc(currentFile)}"></video>
      `;

      // Wire up row clicks to switch video
      host.querySelectorAll('.video-select-row').forEach(row => {
        row.addEventListener('click', () => {
          const file = row.dataset.playVideo;
          const label = row.dataset.playLabel;
          detail.activeAnnotatedVideo = file;

          const player = $('#modal-video-player');
          if (player) { player.src = `/api/files/run/${file}`; player.play(); }

          const nowPlaying = $('#video-now-playing');
          if (nowPlaying) nowPlaying.textContent = label;

          // Update row highlighting
          host.querySelectorAll('.video-select-row').forEach(r => {
            const isNowActive = r.dataset.playVideo === file;
            r.classList.toggle('is-active', isNowActive);
            const actionCell = r.querySelector('span');
            if (actionCell) actionCell.textContent = isNowActive ? '▶ Playing' : '▶ Play';
          });
        });
      });
    }
  } else if (tabName === 'validation') {
    renderValidationTab(host, detail);
  } else if (tabName === 'raw') {
    const stageSummary = (detail.primaryStage && detail.primaryStage.summary) || {};
    const rawJsonStr = JSON.stringify({
      summary: stageSummary,
      detections: detail.detections,
    }, null, 2);
    host.innerHTML = `
      <div class="json-actions">
        <button class="btn-card-detail" id="copy-json-btn">📋 Copy JSON to Clipboard</button>
      </div>
      <pre class="json-code-box"><code>${esc(rawJsonStr)}</code></pre>
    `;
    const copyBtn = $('#copy-json-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(rawJsonStr);
        copyBtn.textContent = '✓ Copied!';
        setTimeout(() => { copyBtn.textContent = '📋 Copy JSON to Clipboard'; }, 2000);
      });
    }
  }
}

function renderDetectionsTable(d, modelLabel, containerEl) {
  if (!containerEl) return;
  if (!d || !d.rows || !d.rows.length) {
    containerEl.innerHTML = '<div class="empty">No positive detections recorded for this model.</div>';
    return;
  }

  const isMotionMonitor = d.rows.some(r =>
    r.model_name === 'crowd_motion_monitor' ||
    (r.extra && (r.extra.crowd_direction != null || r.extra.heading_deg != null || r.extra.local_crush_risk != null))
  );

  const hasPlates = d.rows.some((r) => r.extra && (r.extra.plate || r.extra.plate_display || r.extra.plate_status));

  if (isMotionMonitor) {
    const total = d.rows.length;
    const movingCount = d.rows.filter(r => r.label === 'person_moving' || (r.extra && r.extra.crowd_direction === 'moving')).length;
    const streamACount = d.rows.filter(r => r.label === 'person_moving_stream_a' || (r.extra && r.extra.crowd_direction === 'stream_a')).length;
    const streamBCount = d.rows.filter(r => r.label === 'person_moving_stream_b' || (r.extra && r.extra.crowd_direction === 'stream_b')).length;
    const hasStreams = streamACount > 0 || streamBCount > 0;
    const crushCount = d.rows.filter(r => r.label === 'person_crush_zone' || (r.extra && r.extra.local_crush_risk)).length;
    const stoppedCount = d.rows.filter(r => r.label === 'person_stopped' || (r.extra && r.extra.personally_stationary)).length;
    const cfCount = d.rows.filter(r => r.extra && r.extra.is_counterflow).length;
    const sum = (state.currentDetail && state.currentDetail.primaryStage && state.currentDetail.primaryStage.summary) || {};
    const labelA = streamSlotLabel('a', sum);
    const labelB = streamSlotLabel('b', sum);

    const pMoving = total ? (movingCount / total * 100).toFixed(1) : '0.0';
    const pStreamA = total ? (streamACount / total * 100).toFixed(1) : '0.0';
    const pStreamB = total ? (streamBCount / total * 100).toFixed(1) : '0.0';
    const pCrush = total ? (crushCount / total * 100).toFixed(1) : '0.0';
    const pStopped = total ? (stoppedCount / total * 100).toFixed(1) : '0.0';
    const pCf = total ? (cfCount / total * 100).toFixed(1) : '0.0';

    const rows = d.rows.map(r => {
      const extra = r.extra || {};
      const cdir = extra.crowd_direction || (r.label === 'person_moving_stream_a' ? 'stream_a' : (r.label === 'person_moving_stream_b' ? 'stream_b' : 'moving'));
      const hdeg = extra.heading_deg != null ? extra.heading_deg.toFixed(1) : null;
      const spd = extra.speed_px_frame != null ? extra.speed_px_frame.toFixed(2) : null;
      const isCrush = extra.local_crush_risk || r.label === 'person_crush_zone';
      const isStopped = extra.personally_stationary || r.label === 'person_stopped';
      const isCf = extra.is_counterflow;
      const cfAngle = extra.counterflow_angle_deg != null ? extra.counterflow_angle_deg : 0;
      const entropyVal = extra.local_directional_entropy != null ? extra.local_directional_entropy : null;

      let statusBadge = '';
      if (isStopped) {
        statusBadge = '<span class="badge badge-stopped">⏹ Stopped</span>';
      } else if (isCrush) {
        statusBadge = '<span class="badge badge-crush">⚠️ Crush Zone</span>';
      } else if (cdir === 'stream_a') {
        statusBadge = `<span class="badge badge-right">${esc(labelA)}</span>`;
      } else if (cdir === 'stream_b') {
        statusBadge = `<span class="badge badge-left">${esc(labelB)}</span>`;
      } else {
        statusBadge = '<span class="badge badge-right">Moving</span>';
      }

      const extraDir = extra.stream_screen_direction;
      const dirLabel = cdir === 'stream_a'
        ? (extraDir ? `Stream A (${extraDir})` : labelA)
        : (cdir === 'stream_b' ? (extraDir ? `Stream B (${extraDir})` : labelB) : 'Moving');
      const dirCell = `<span class="badge ${cdir === 'stream_b' ? 'badge-left' : 'badge-right'}">${esc(dirLabel)}${hdeg ? ` (${hdeg}°)` : ''}</span>`;
      const spdCell = spd ? `${spd} px/fr` : '—';
      const crushCell = isCrush
        ? `<span class="badge badge-crush">⚠️ Risk${extra.local_divergence != null ? ` (${extra.local_divergence.toFixed(2)})` : ''}</span>`
        : '<span class="badge badge-ok">✓ Safe</span>';

      const dynamicsCell = isCf
        ? `<span class="badge" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">⚡ Opposing (${cfAngle}°)</span>`
        : (entropyVal != null && entropyVal > 1.5
          ? `<span class="badge" style="background: rgba(139, 92, 246, 0.2); color: #c4b5fd;">🌀 Ent: ${entropyVal}</span>`
          : '<span class="badge badge-ok">✓ Aligned</span>');

      return `<tr>
        <td class="u-mono-sm">${r.timestamp_sec.toFixed(2)}s</td>
        <td><strong>#${extra.track_id != null ? extra.track_id : '—'}</strong></td>
        <td>${statusBadge}</td>
        <td>${dirCell}</td>
        <td class="u-num">${spdCell}</td>
        <td>${crushCell}</td>
        <td>${dynamicsCell}</td>
        <td class="u-num">${(r.confidence * 100).toFixed(0)}%</td>
      </tr>`;
    }).join('');

    containerEl.innerHTML = `
      <div class="motion-kpi-grid">
        ${hasStreams
          ? `<div class="motion-kpi-card" style="border-color: rgba(99,102,241,0.4);">
               <div class="motion-kpi-lbl">${esc(labelA)}</div>
               <div class="motion-kpi-val" style="color: #818cf8;">${pStreamA}%</div>
               <div class="motion-kpi-count">${streamACount.toLocaleString()} detections</div>
             </div>
             <div class="motion-kpi-card" style="border-color: rgba(139,92,246,0.4);">
               <div class="motion-kpi-lbl">${esc(labelB)}</div>
               <div class="motion-kpi-val" style="color: #c084fc;">${pStreamB}%</div>
               <div class="motion-kpi-count">${streamBCount.toLocaleString()} detections</div>
             </div>`
          : `<div class="motion-kpi-card" style="border-color: rgba(99,102,241,0.4);">
               <div class="motion-kpi-lbl">Moving</div>
               <div class="motion-kpi-val" style="color: #818cf8;">${pMoving}%</div>
               <div class="motion-kpi-count">${movingCount.toLocaleString()} detections</div>
             </div>`
        }
        <div class="motion-kpi-card" style="border-color: rgba(244,63,94,0.4);">
          <div class="motion-kpi-lbl">Crush Risk</div>
          <div class="motion-kpi-val" style="color: #fb7185;">${pCrush}%</div>
          <div class="motion-kpi-count">${crushCount.toLocaleString()} detections</div>
        </div>
        <div class="motion-kpi-card">
          <div class="motion-kpi-lbl">Stationary</div>
          <div class="motion-kpi-val" style="color: var(--muted);">${pStopped}%</div>
          <div class="motion-kpi-count">${stoppedCount.toLocaleString()} detections</div>
        </div>
        <div class="motion-kpi-card" style="border-color: rgba(245,158,11,0.4);">
          <div class="motion-kpi-lbl">Counter-Flow</div>
          <div class="motion-kpi-val" style="color: #fbbf24;">${pCf}%</div>
          <div class="motion-kpi-count">${cfCount.toLocaleString()} detections</div>
        </div>
      </div>
      <div class="hint u-mb-3" style="margin-bottom:12px;">Showing top ${d.rows.length} of ${d.total} detections for <strong>${esc(modelLabel || 'Crowd Motion Monitor')}</strong> with enriched kinematic telemetry.</div>
      <table class="stages">
        <thead>
          <tr>
            <th>Time</th>
            <th>Track ID</th>
            <th>Flow Status</th>
            <th>Direction (Heading)</th>
            <th class="u-num">Velocity</th>
            <th>Compression</th>
            <th>Dynamics</th>
            <th class="u-num">Conf</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
    return;
  }

  // Standard or ANPR table
  const rows = d.rows.map((r) => {
    const extra = r.extra || {};
    const plateStr = extra.plate_display || extra.plate || '';
    const plateStatus = extra.plate_status || '';
    let plateCell = '—';
    if (plateStr) {
      plateCell = `<span class="plate-badge">${esc(plateStr)}</span>`;
    } else if (plateStatus) {
      plateCell = `<span class="subtle">${esc(statusText(plateStatus, extra.plate_width_px))}</span>`;
    }

    const details = extra.scoring
      ? esc(extra.scoring)
      : (extra.vehicle_class ? `${esc(extra.vehicle_class)}${extra.colour ? ` (${esc(extra.colour)})` : ''}` : '—');

    return `<tr>
      <td>${r.timestamp_sec.toFixed(2)}s</td>
      <td><strong>${esc(r.label)}</strong></td>
      <td>${(r.confidence * 100).toFixed(1)}%</td>
      <td>${extra.track_id != null ? extra.track_id : '—'}</td>
      ${hasPlates ? `<td>${plateCell}</td>` : ''}
      <td>${details}</td>
    </tr>`;
  }).join('');

  containerEl.innerHTML = `
    <div class="hint u-mb-3" style="margin-bottom:12px;">Total positive detections: ${d.total}; showing top ${d.rows.length} rows for <strong>${esc(modelLabel || '')}</strong>.</div>
    <table class="stages">
      <thead><tr><th>Timestamp</th><th>Class Label</th><th class="u-num">Conf</th>
        <th>Track ID</th>${hasPlates ? '<th>Plate Read</th>' : ''}<th>Details</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderDetections(d) {
  renderDetectionsTable(d, '', $('#modal-body'));
}

function closeModal() {
  // Return focus to the trigger; without this, dismissing the dialog drops
  // the caret back to the top of the document.
  if (state.modalOpener && document.contains(state.modalOpener)) {
    state.modalOpener.focus({ preventScroll: true });
  }
  state.modalOpener = null;
  $('#modal').classList.add('hidden');
  $('#modal-body').innerHTML = '';
  state.currentDetail = null;
}

/* ------------------------------------------------------------------- run */
function currentSource() {
  if (state.sourceTab === 'url') return $('#video-url').value.trim();
  if (state.sourceTab === 'local') return $('#video-select').value;
  return $('#video-file').dataset.uploaded || '';
}

async function runJob() {
  const err = $('#run-error');
  err.textContent = '';

  const source = currentSource();
  if (!source) { err.textContent = 'Pick a video first.'; return; }
  if (!state.selected.size) { err.textContent = 'Select at least one model.'; return; }

  const btn = $('#run-btn');
  btn.disabled = true;
  btn.textContent = 'Starting job…';

  try {
    // One threshold for every selected model. The backend fans it out and
    // skips models that have no confidence score to threshold.
    await postJSON('/api/jobs', {
      source,
      models: [...state.selected],
      sample_every_n_frames: parseInt($('#stride').value, 10),
      device: $('#device').value || null,
      export_video: $('#export-video').checked,
      threshold: state.threshold,
    });
    state.openJobs.clear();
    refreshJobs();
  } catch (e) {
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run selected models';
  }
}

/* ------------------------------------------------------------------ wiring */
/* ------------------------------------------------- dense-flow validation */

const VALIDATION_STATUS_LABEL = {
  pass: 'PASS', fail: 'FAIL', skipped: 'SKIPPED', error: 'ERROR',
};

function measurementRow(m) {
  // A measurement with no tolerance is informational. Rendering it with a
  // pass/fail tint would imply a judgement that was never made -- which is
  // exactly how an unjudged correlation gets read as a satisfied one.
  const judged = m.passed !== null && m.passed !== undefined;
  const cls = judged ? (m.passed ? 'ok' : 'over') : 'info';
  const limit = m.tolerance == null
    ? ''
    : `<span class="meas-limit">limit ${m.higher_is_better ? '≥' : '≤'} ${m.tolerance}</span>`;
  const value = Number.isFinite(m.value) ? m.value.toFixed(3) : '—';
  return `
    <div class="meas ${cls}">
      <span class="meas-label">${esc(m.label)}</span>
      <span class="meas-value">${value} <span class="meas-units">${esc(m.units)}</span></span>
      ${limit}
      ${m.note ? `<div class="meas-note">${esc(m.note)}</div>` : ''}
    </div>`;
}

function routeCard(r) {
  const status = r.status || 'skipped';
  return `
    <div class="route-card ${status}">
      <div class="route-head">
        <span class="route-title">${esc(r.title)}</span>
        <span class="pill ${status}">${VALIDATION_STATUS_LABEL[status] || status}</span>
      </div>
      <div class="route-summary">${esc(r.summary)}</div>
      ${r.measurements && r.measurements.length
      ? `<div class="meas-grid">${r.measurements.map(measurementRow).join('')}</div>`
      : ''}
      ${r.caveat
      ? `<div class="route-caveat"><strong>Cannot tell you:</strong> ${esc(r.caveat)}</div>`
      : ''}
    </div>`;
}

function renderValidation() {
  const host = $('#flow-validation');
  const st = state.validation;
  if (!host) return;

  if (!st || (st.status === 'idle' && !st.report)) {
    host.innerHTML = `<div class="empty-state-card">
      <div class="empty-icon">🌊</div>
      <div class="empty-title">No Validation Run Yet</div>
      <div class="empty-desc">Pick a source video in the panel above and click Run validation to measure dense flow accuracy across 3 annotation-free routes.</div>
      <button class="btn-card-detail" onclick="document.getElementById('run-validation').click()">▶ Run Validation Now</button>
    </div>`;
    return;
  }

  if (st.status === 'running') {
    host.innerHTML = `<div class="validation-running">
      <span class="spinner"></span> ${esc(st.message || 'Running…')}
      <div class="hint">Route (c) runs a person detector per frame, so this
        takes a couple of minutes.</div>
    </div>`;
    return;
  }

  if (st.status === 'error') {
    host.innerHTML = `<div class="route-card error">
      <div class="route-head"><span class="route-title">Validation failed to run</span>
      <span class="pill error">ERROR</span></div>
      <div class="route-summary">${esc(st.message)}</div></div>`;
    return;
  }

  const rep = st.report;
  if (!rep) {
    host.innerHTML = `<div class="empty">${esc(st.message || 'No report.')}</div>`;
    return;
  }

  const skipped = (rep.routes || []).filter((r) => r.status === 'skipped').length;
  const when = rep.created_at
    ? new Date(rep.created_at * 1000).toLocaleString() : '—';

  host.innerHTML = `
    <div class="validation-head">
      <span class="pill ${rep.status}">${VALIDATION_STATUS_LABEL[rep.status] || rep.status}</span>
      <span class="subtle">${esc(rep.source || '')} · ${esc(when)}</span>
    </div>
    ${skipped ? `<div class="validation-incomplete">
      ${skipped} route${skipped > 1 ? 's' : ''} skipped — the picture is
      incomplete, not clean. A route that did not run is not a route that
      passed.</div>` : ''}
    <div class="route-grid">${(rep.routes || []).map(routeCard).join('')}</div>
    <div class="validation-footer">
      These routes measure the velocity field. They do not test whether
      divergence, counterflow, or turbulence actually predict crush risk —
      that is a separate question none of them answers.
    </div>`;
}

/*
 * Validation tab inside the result modal.
 *
 * Separate from the standalone panel because it answers a different question:
 * the panel is "is the flow estimator sound right now", this tab is "how much
 * should I trust THIS result". Same report, read for a different purpose.
 */
function renderValidationTab(host, detail) {
  const st = state.validation;

  const stages = detail.allStages ||
    (detail.group && detail.group.stages) ||
    (detail.primaryStage ? [detail.primaryStage] : []);
  const isFlow = stages.some((s) => s && s.model_key === 'dense_flow');

  if (!isFlow) {
    host.innerHTML = `<div class="empty">These validation routes measure dense
      optical flow. This result is from a different model, so they do not
      apply to it.</div>`;
    return;
  }

  if (!st) {
    // Not fetched yet (modal opened before the initial load finished).
    host.innerHTML = '<div class="loading">Loading validation report…</div>';
    refreshValidation().then(() => {
      if (state.activeModalTab === 'validation' && state.currentDetail) {
        renderValidationTab(host, state.currentDetail);
      }
    });
    return;
  }

  if (!st.report) {
    host.innerHTML = `<div class="empty">
      No validation has been run yet.<br><br>
      Use <strong>Run validation</strong> in the “Dense Flow Validation” panel
      on the main page, then reopen this tab.</div>`;
    return;
  }

  const rep = st.report;
  const cRoute = (rep.routes || []).find((r) => r.route === 'cross_family');
  const video = cRoute && cRoute.detail && cRoute.detail.comparison_video;
  const skipped = (rep.routes || []).filter((r) => r.status === 'skipped').length;

  host.innerHTML = `
    <div class="validation-head">
      <span class="pill ${rep.status}">${VALIDATION_STATUS_LABEL[rep.status] || rep.status}</span>
      <span class="subtle">${esc(rep.source || '')} ·
        ${rep.created_at ? new Date(rep.created_at * 1000).toLocaleString() : '—'}</span>
    </div>

    ${video ? `
    <h4 class="vt-heading">See it: tracker vs optical flow</h4>
    <p class="hint u-mb-3">
      Every person carries two arrows. <strong class="swatch-tracker">White</strong>
      is what the person-tracker measured; <strong class="swatch-flow">cyan</strong>
      is what dense optical flow measured. Arrows on top of each other means the
      two independent methods agree. Arrows splitting apart means they disagree —
      the box turns red and shows the angle between them.
    </p>
    <video class="vt-video" controls src="/api/files/validation/${esc(video)}"></video>
    <p class="hint u-mb-6">
      Arrows are drawn longer than life so they are visible; both use the same
      exaggeration, so the comparison between them stays fair.
    </p>` : ''}

    <h4 class="vt-heading">The three routes</h4>
    ${skipped ? `<div class="validation-incomplete">
      ${skipped} route${skipped > 1 ? 's' : ''} skipped — the picture is
      incomplete, not clean. A route that did not run is not a route that
      passed.</div>` : ''}
    <div class="route-grid">${(rep.routes || []).map(routeCard).join('')}</div>
    <div class="validation-footer">
      These routes check that the velocity field is measured correctly. They do
      not test whether divergence, counterflow, or turbulence actually predict
      crush risk — no amount of flow validation answers that.
    </div>`;
}

async function refreshValidation() {
  try {
    state.validation = await api('/api/validation/flow');
  } catch (e) {
    state.validation = { status: 'error', message: e.message, report: null };
  }
  renderValidation();

  // Poll only while a run is in flight.
  if (state.validation && state.validation.status === 'running') {
    if (!state.validationTimer) {
      state.validationTimer = setInterval(refreshValidation, 3000);
    }
  } else if (state.validationTimer) {
    clearInterval(state.validationTimer);
    state.validationTimer = null;
  }
}

async function deleteValidation() {
  if (!confirm('Delete the saved validation report and its comparison video?\n'
    + 'Source videos and model outputs are not touched.')) return;
  try {
    const r = await api('/api/validation/flow', { method: 'DELETE' });
    // Clear locally too: the modal's Validation tab reads the same state, so
    // leaving it populated would show a report whose video no longer exists.
    state.validation = null;
    await refreshValidation();
    if (state.activeModalTab === 'validation' && state.currentDetail) {
      renderValidationTab($('#modal-body'), state.currentDetail);
    }
    $('#run-error').textContent = r.message || 'Validation output deleted.';
  } catch (e) {
    alert(e.message);
  }
}

async function startValidation() {
  const src = currentSource();
  if (!src) {
    alert('Pick a video in the source panel first — validation runs against a video.');
    return;
  }
  try {
    await postJSON('/api/validation/flow', { source: src, routes: 'abc' });
    await refreshValidation();
  } catch (e) {
    alert(`Could not start validation: ${e.message}`);
  }
}

function wire() {
  $$('#source-tabs .tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('#source-tabs .tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      state.sourceTab = tab.dataset.tab;
      $$('.tab-body').forEach((b) => b.classList.toggle('hidden', b.dataset.body !== state.sourceTab));
      if (state.sourceTab === 'local') loadVideos();
    });
  });

  $$('[data-select]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.select;
      state.selected.clear();
      if (mode === 'all') {
        state.models.forEach((m) => { if (m.status !== 'blocked') state.selected.add(m.key); });
      } else if (mode === 'ready') {
        state.models.forEach((m) => { if (m.status === 'ready') state.selected.add(m.key); });
      }
      renderModels();
    });
  });

  const histSearch = $('#history-search');
  if (histSearch) {
    histSearch.addEventListener('input', (e) => {
      state.historySearchQuery = e.target.value.trim();
      refreshHistory();
    });
  }

  $$('#modal-nav .modal-tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      renderModalTab(btn.dataset.mtab);
    });
  });

  const thresh = $('#threshold');
  if (thresh) {
    const out = $('#threshold-value');
    const sync = () => {
      state.threshold = parseFloat(thresh.value);
      if (out) out.textContent = state.threshold.toFixed(2);
    };
    thresh.addEventListener('input', sync);
    sync();
  }

  $('#run-btn').addEventListener('click', runJob);
  $('#refresh-jobs').addEventListener('click', refreshJobs);
  const runVal = $('#run-validation');
  if (runVal) runVal.addEventListener('click', startValidation);
  const refVal = $('#refresh-validation');
  if (refVal) refVal.addEventListener('click', refreshValidation);
  const delVal = $('#delete-validation');
  if (delVal) delVal.addEventListener('click', deleteValidation);
  $('#refresh-history').addEventListener('click', refreshHistory);
  $('#refresh-anpr').addEventListener('click', refreshAnpr);
  $('#modal-close').addEventListener('click', closeModal);
  $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  $('#clear-outputs').addEventListener('click', async () => {
    if (!confirm('Delete every generated log and annotated video in outputs/?\nSource videos in test_videos/ are not touched.')) return;
    const r = await api('/api/outputs', { method: 'DELETE' });
    $('#run-error').textContent = `Deleted ${r.removed} item(s).`;
    state.openHistory.clear();
    state.openAnpr.clear();
    refreshJobs();
    refreshHistory();
    refreshAnpr();
  });

  const delHistAll = $('#delete-history-all');
  if (delHistAll) {
    delHistAll.addEventListener('click', async () => {
      if (!confirm('Delete all saved output logs, annotated videos, and galleries in outputs/?\nSource videos in test_videos/ will NOT be touched.')) return;
      await api('/api/outputs', { method: 'DELETE' });
      state.openHistory.clear();
      state.openAnpr.clear();
      refreshJobs();
      refreshHistory();
      refreshAnpr();
    });
  }

  const delAnprAll = $('#delete-anpr-all');
  if (delAnprAll) {
    delAnprAll.addEventListener('click', async () => {
      if (!confirm('Delete all captured ANPR vehicle galleries in outputs/anpr/?')) return;
      let galleries;
      try { ({ galleries } = await api('/api/anpr')); } catch { return; }
      for (const g of galleries) {
        try { await api(`/api/anpr/${encodeURIComponent(g.video)}`, { method: 'DELETE' }); } catch { }
      }
      state.openAnpr.clear();
      refreshAnpr();
    });
  }

  $('#video-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const status = $('#upload-status');
    status.textContent = `Uploading ${file.name}…`;
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await api('/api/videos/upload', { method: 'POST', body: fd });
      e.target.dataset.uploaded = r.name;
      status.textContent = `Ready: ${r.name} (${r.size_mb} MB)`;
      loadVideos();
    } catch (err) {
      status.textContent = err.message;
    }
  });
}

/* -------------------------------------------------------------------- go */
(async function init() {
  wire();
  loadDevice();
  await loadModels();
  loadVideos();
  refreshJobs();
  refreshHistory();
  refreshAnpr();
  refreshValidation();
})();
