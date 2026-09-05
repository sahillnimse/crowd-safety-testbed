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
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers);
  if (opts.body && typeof opts.body === 'string' && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const fetchOpts = Object.assign({}, opts, { headers });
  const res = await fetch(path, fetchOpts);
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
  const done = job.stages.filter((s) => ['done', 'failed', 'cancelled', 'interrupted'].includes(s.status)).length;
  const cancellable = !['done', 'failed', 'cancelled', 'interrupted'].includes(job.status);
  const canDelete = ['done', 'failed', 'cancelled', 'interrupted'].includes(job.status);
  const isLive = job.mode === 'live';

  return `<div class="job ${isLive ? 'job-live' : ''}" data-job="${job.id}">
    <div class="job-head" data-toggle="${job.id}">
      <div class="job-head-left">
        ${isLive ? `<span class="pill" style="background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.4); font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px; margin-right: 6px;">LIVE RUN</span>` : ''}
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
        ${canDelete ? `<button class="link-btn danger-link btn-del-job" data-del-job="${job.id}" title="Remove this job from list" style="padding: 2px 8px; font-size: 11px; margin-left: 4px;">✕</button>` : ''}
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
      if (ev.target.dataset.cancel || ev.target.dataset.inspectJob || ev.target.dataset.delJob) return;
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

  host.querySelectorAll('.btn-del-job').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const id = el.dataset.delJob;
      try {
        await api(`/api/jobs/${id}`, { method: 'DELETE' });
        state.openJobs.delete(id);
        refreshJobs();
      } catch (err) {
        alert(`Failed deleting job: ${err.message}`);
      }
    });
  });

  const clearJobsBtn = $('#clear-finished-jobs');
  if (clearJobsBtn && !clearJobsBtn.dataset.bound) {
    clearJobsBtn.dataset.bound = 'true';
    clearJobsBtn.addEventListener('click', async () => {
      try {
        await api('/api/jobs', { method: 'DELETE' });
        state.openJobs.clear();
        refreshJobs();
      } catch (err) {
        alert(`Failed clearing finished jobs: ${err.message}`);
      }
    });
  }

  host.querySelectorAll('[data-inspect-job]').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const jobId = el.dataset.inspectJob;
      const modelKey = el.dataset.inspectModel;
      openJobDetailModal(jobId, modelKey);
    });
  });

  const active = jobs.some((j) => !['done', 'failed', 'cancelled', 'interrupted'].includes(j.status));
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
  setLiveChrome({ tabVisible: false, streaming: false });

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
  setLiveChrome({ tabVisible: false, streaming: false });

  try {
    const job = await api(`/api/jobs/${jobId}`);
    const stage = job && job.stages ? job.stages.find(s => s.model_key === modelKey) : null;
    let detections = { rows: [], total: 0 };
    try {
      detections = await api(`/api/jobs/${jobId}/detections/${modelKey}?limit=500`);
    } catch (detErr) {
      $('#modal-body').innerHTML = `
        <div class="empty" style="padding: 40px 20px; text-align: center;">
          <div style="font-size: 28px; margin-bottom: 12px;">⚠️ Output Files Deleted</div>
          <p style="color: var(--muted); margin-bottom: 18px;">The detection files for this run no longer exist on disk (they were removed with Delete All Outputs).</p>
          <button class="link-btn danger-link" onclick="deleteJobAndCloseModal('${esc(jobId)}')" style="padding: 6px 14px; font-weight: 600;">🗑️ Remove this Run from List</button>
        </div>
      `;
      return;
    }

    state.currentDetail = {
      videoName: (job && job.video_name) ? job.video_name : jobId,
      jobId,
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

window.deleteJobAndCloseModal = async function(jobId) {
  try {
    await api(`/api/jobs/${jobId}`, { method: 'DELETE' });
    closeModal();
    refreshJobs();
  } catch (err) {
    alert(`Failed removing job: ${err.message}`);
  }
};

async function openRouteSessionDetailModal(sessionName) {
  state.modalOpener = document.activeElement;
  $('#modal-title').textContent = `Route Session Details — ${sessionName}`;
  $('#modal-body').innerHTML = '<div class="loading">Loading route metrics and camera telemetry…</div>';
  $('#modal').classList.remove('hidden');
  setLiveChrome({ tabVisible: false, streaming: false });

  try {
    const session = await api(`/api/sessions/${encodeURIComponent(sessionName)}`);
    if (!session) throw new Error('Route session not found');

    const cams = session.cameras || {};
    const camList = Object.values(cams);

    const allAnnotatedVideos = camList
      .filter(c => c.status === 'done')
      .map(c => ({
        label: `${c.camera_name || c.camera_id} (${c.camera_id})`,
        key: c.camera_id,
        file: `/api/files/session/${encodeURIComponent(sessionName)}/${encodeURIComponent(c.camera_id)}/annotated.mp4`,
      }));

    const primaryCam = camList.find(c => c.status === 'done') || camList[0];

    // Fetch primary camera detections
    let detections = { rows: [], total: 0 };
    if (primaryCam && primaryCam.status === 'done') {
      try {
        detections = await api(`/api/sessions/${encodeURIComponent(sessionName)}/detections/${encodeURIComponent(primaryCam.camera_id)}?limit=500`);
      } catch {}
    }

    state.currentDetail = {
      videoName: sessionName,
      isRouteSession: true,
      sessionData: session,
      summary: session.summary || {},
      primaryStage: primaryCam ? { model_key: primaryCam.camera_id, model_label: primaryCam.camera_name || primaryCam.camera_id, ...primaryCam } : null,
      allStages: camList.map(c => ({
        camera_id: c.camera_id,
        camera_name: c.camera_name,
        model_key: c.camera_id,
        model_label: `${c.camera_name || c.camera_id} (${c.camera_id})`,
        detections: c.detections || 0,
        positives: c.positives || 0,
        status: c.status,
        ...c
      })),
      activeTimelineModelKey: primaryCam ? primaryCam.camera_id : null,
      detections,
      allAnnotatedVideos,
      activeAnnotatedVideo: allAnnotatedVideos[0]?.file || null,
    };

    renderModalTab('overview');
  } catch (err) {
    $('#modal-body').innerHTML = `<div class="err">${esc(err.message)}</div>`;
  }
}

/* ---- Per-model selector for the modal's stage-scoped tabs ----
   The Detections Timeline tab already had its own per-model table
   (activeTimelineModelKey). Every other tab silently pinned to whichever
   stage was primary when the modal opened. These helpers give each tab its
   own dropdown, remembered per tab in detail.activeTabModelKeys, and make
   detail.primaryStage follow the selection so every existing stage-scoped
   read (overview KPIs, analytics, raw JSON) renders the chosen model. */

function getActiveStageForTab(detail, tabName) {
  if (!detail.activeTabModelKeys) detail.activeTabModelKeys = {};
  const wanted = detail.activeTabModelKeys[tabName]
    || (detail.primaryStage && detail.primaryStage.model_key);
  return (detail.allStages || []).find(s => s.model_key === wanted)
    || detail.primaryStage
    || (detail.allStages || [])[0]
    || null;
}

function stageModelSelectHtml(detail, tabName, activeStage) {
  const stages = detail.allStages || [];
  // Nothing to switch between with a single stage — the timeline tab hides
  // its selector in exactly that case.
  if (stages.length <= 1 || !activeStage) return '';
  const longest = stages.reduce((m, s) =>
    Math.max(m, (s.model_label || s.model_key || '').length), 0);
  const options = stages.map(s =>
    `<option value="${esc(s.model_key)}"${s.model_key === activeStage.model_key ? ' selected' : ''}>${esc(s.model_label || s.model_key)}</option>`
  ).join('');
  // Sized to the longest full label so no option is ever truncated
  // (capped so one verbose model name can't eat the whole header bar).
  const ch = Math.max(14, Math.min(36, longest + 2));
  return `
    <div class="detail-section-title modal-model-bar">
      <span>Model</span>
      <select class="modal-model-select" data-tab="${esc(tabName)}"
              style="width: auto; min-width: ${ch}ch;" aria-label="Model for this tab">${options}</select>
    </div>`;
}

function wireModelSelect(host, detail) {
  host.querySelectorAll('.modal-model-select').forEach(sel => {
    sel.addEventListener('change', async () => {
      const tabName = sel.dataset.tab;
      const modelKey = sel.value;
      detail.activeTabModelKeys[tabName] = modelKey;

      const stage = (detail.allStages || []).find(s => s.model_key === modelKey);
      if (stage) detail.primaryStage = stage;

      // Re-fetch detections scoped to the new model through the same
      // endpoints the modal openers use, then re-render the tab so every
      // stage-scoped read (KPIs, analytics, raw payload) follows.
      try {
        if (detail.group) {
          detail.detections = await api(`/api/history/${encodeURIComponent(detail.videoName)}/${encodeURIComponent(modelKey)}/detections?limit=500`);
        } else if (detail.jobId) {
          detail.detections = await api(`/api/jobs/${encodeURIComponent(detail.jobId)}/detections/${encodeURIComponent(modelKey)}?limit=500`);
        }
      } catch (err) {
        detail.detections = { rows: [], total: 0 };
      }

      // Annotated video follows the model when that model produced one.
      const vid = (detail.allAnnotatedVideos || []).find(v => v.key === modelKey);
      detail.activeAnnotatedVideo = vid ? vid.file : null;

      renderModalTab(tabName);
    });
  });
}

/* Move the live stage back to its holder before anything overwrites the
   modal body.  The canvas is fed by an open WebSocket, so it has to survive
   every tab switch — destroying it would drop the stream mid-run. */
function parkLiveStage() {
  const stage = document.getElementById('live-stage');
  const holder = document.getElementById('live-stage-holder');
  if (stage && holder && stage.parentElement !== holder) holder.appendChild(stage);
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
  const modalCard = $('#modal .modal-card');
  if (modalCard && tabName !== 'video') {
    modalCard.classList.remove('wide-modal');
  }

  parkLiveStage();

  if (tabName === 'live') {
    host.innerHTML = '';
    const stage = document.getElementById('live-stage');
    if (stage) {
      host.appendChild(stage);
      drawMetricSpark();
    } else {
      host.innerHTML = '<div class="empty">Live stream is not available for this run.</div>';
    }
    return;
  }

  if (detail.isRouteSession) {
    renderRouteSessionModalTab(tabName, detail, host);
    return;
  }

  if (tabName === 'overview') {
    const g = detail.group;
    const s = getActiveStageForTab(detail, tabName) || {};
    const d = detail.detections || {};
    const sum = s.summary || {};
    // Stage-scoped: this card sits next to "Total Output Rows", which counts
    // only the selected model's rows, so the alert count must come from the
    // same stage — not the video-wide sum across all models it used to read.
    const totalPositives = s.positives != null
      ? s.positives
      : (d.rows ? d.rows.filter(r => r.confidence > 0.5).length : 0);
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

      /* ---- The 10 crowd-safety metrics -------------------------------
         CrowdMotionMonitor's metric set. This whole analytics block is
         already gated on pct_moving / pct_crush_risk, which only that model
         writes, so these cards are scoped to it by construction.
         A metric with no value shows an explicit state ("No data") rather
         than a zero: a zero here is a real measurement, and "the crowd was
         not compressing" is a different claim from "never measured".  */
      const NO_DATA = '<span class="no-data-text">No data</span>';
      const unit = (t) => `<span style="font-size:12px;color:var(--muted);font-weight:normal;">${t}</span>`;
      const num = (v, nd = 2, u = '') =>
        v == null ? NO_DATA : `${Number(v).toFixed(nd)}${u ? ' ' + unit(u) : ''}`;

      // Specific flow: dense_flow writes specific_flow_current (needs
      // configured lines); crowd_motion_monitor writes a net rate across the
      // frame centre line. Prefer whichever exists.
      const sfVal = sum.specific_flow_current != null ? sum.specific_flow_current
                  : sum.specific_flow_net_per_sec;
      const sfUnits = sum.specific_flow_units || (sum.specific_flow_net_per_sec != null ? 'ppl/s' : '');
      const sfSub = sum.specific_flow_gross_per_sec != null
        ? `${sum.specific_flow_crossings || 0} crossings · ${sum.specific_flow_gross_per_sec.toFixed(2)} gross ppl/s`
        : (sum.specific_flow_peak != null
            ? `peak ${sum.specific_flow_peak.toFixed(2)} ${sum.specific_flow_units || ''}`
            : '<span class="no-data-badge">Unconfigured</span>');

      // Oscillation symmetry: dense_flow uses *_avg, CMM a single run score.
      const oscVal = sum.oscillation_symmetry_avg != null ? sum.oscillation_symmetry_avg
                   : sum.oscillation_symmetry;
      const oscSubTxt = sum.oscillation_symmetry_peak != null
        ? `peak ${sum.oscillation_symmetry_peak.toFixed(2)}`
        : 'Back-and-forth surging (0: steady, 1: rocking)';

      // Divergence: negative = compression. The headline is the WORST
      // (most negative) value, since that is the crush signal.
      const divWorst = sum.strongest_compression != null ? sum.strongest_compression
                     : sum.divergence_strongest_compression;
      const divAvg = sum.avg_divergence != null ? sum.avg_divergence : sum.divergence_avg;

      const speedAvg = sum.avg_speed_px_frame != null ? sum.avg_speed_px_frame : sum.mean_speed_avg;
      const speedPeak = sum.peak_speed_px_frame != null ? sum.peak_speed_px_frame : sum.mean_speed_peak;
      const speedUnits = sum.speed_units || 'px/frame';

      const stopGo = sum.stop_go_score != null ? sum.stop_go_score : sum.stop_go_avg;

      // Density / pressure need a ground-plane homography to be physical.
      // metric_units.note (CMM) or calibration.is_calibrated (dense_flow)
      // tells us; say so on the card instead of implying persons/m².
      const uncal = sum.is_calibrated === false;
      const densityUnits = (sum.metric_units && sum.metric_units.density) || 'persons/m²';
      const pressureNote = uncal
        ? 'Uncalibrated: density x velocity variance, image-plane units'
        : 'Helbing crowd pressure (s⁻²)';

      const metricCards = `
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Density</div>
              <div class="overview-kpi-val" style="color: #f97316;">${num(sum.avg_density, 1)}</div>
              <div class="overview-kpi-sub">${sum.avg_person_count != null ? `avg ${sum.avg_person_count} people (peak ${sum.peak_person_count})` : esc(densityUnits)}</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Velocity Field</div>
              <div class="overview-kpi-val" style="color: #60a5fa;">${num(speedAvg, 2)}</div>
              <div class="overview-kpi-sub">${speedPeak != null ? `peak ${Number(speedPeak).toFixed(2)} ` : ''}${esc(speedUnits)}</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Specific Flow</div>
              <div class="overview-kpi-val" style="color: #22d3ee;">${num(sfVal, 2)}</div>
              <div class="overview-kpi-sub">${sfUnits ? esc(sfUnits) + ' · ' : ''}${sfSub}</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Crowd Pressure</div>
              <div class="overview-kpi-val" style="color: #ef4444;">${num(sum.avg_crowd_pressure, 3)}</div>
              <div class="overview-kpi-sub">${sum.peak_crowd_pressure != null ? `peak ${sum.peak_crowd_pressure} · ` : ''}${pressureNote}</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Divergence</div>
              <div class="overview-kpi-val" style="color: #fb7185;">${num(divWorst, 3)}</div>
              <div class="overview-kpi-sub">${divAvg != null ? `mean ${Number(divAvg).toFixed(3)} · ` : ''}negative = compression</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Stop &amp; Go</div>
              <div class="overview-kpi-val" style="color: #fbbf24;">${num(stopGo, 2)}</div>
              <div class="overview-kpi-sub">Periodic halting (0: smooth, 1: strong waves)</div>
            </div>
            <div class="overview-kpi-item">
              <div class="overview-kpi-lbl">Oscillation Symmetry</div>
              <div class="overview-kpi-val" style="color: #f472b6;">${num(oscVal, 2)}</div>
              <div class="overview-kpi-sub">${oscSubTxt}</div>
            </div>`;

      // Specific Flow and Oscillation Symmetry used to live in a
      // `denseFlowOnlyCards` block gated on model_key === 'dense_flow'. That
      // block was unreachable: it sat inside an outer condition requiring
      // pct_moving / pct_crush_risk, which ONLY crowd_motion_monitor writes,
      // so the dense-flow-only cards could never render. Both metrics are now
      // in metricCards above, keyed on the summary fields rather than on the
      // model name, so each model shows whichever it actually produced.

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
${metricCards}
          </div>
          ${uncal ? `<div class="hint" style="margin-top:8px;">Image-plane units: this run has no ground-plane homography, so Density is per megapixel (not persons/m²) and Crowd Pressure is not in s⁻². Values compare across frames of this camera only.</div>` : ''}

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
      ${stageModelSelectHtml(detail, tabName, s)}
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

      ${g && (detail.allStages || []).length > 1
        ? `<div class="hint" style="margin-top: 10px;">All ${(detail.allStages || []).length} models on this video combined: ${g.total_positives} alert events (aggregate across models — the cards above show the selected model only).</div>`
        : ''}

      <div class="detail-section-title">Classifications & Metadata</div>
      <table class="detail-info-table">
        <tbody>
          <tr><td class="key">Source Video</td><td class="val">${esc(detail.videoName)}</td></tr>
          <tr><td class="key">Model</td><td class="val">${esc(s.model_label || s.model_key || '—')}</td></tr>
          ${s.algorithm ? `<tr><td class="key">Algorithm / Model</td><td class="val">${esc(s.algorithm)}</td></tr>` : ''}
          ${sum.calibration ? `<tr><td class="key">Calibration</td><td class="val"><strong>${sum.calibration.is_calibrated ? 'Calibrated' : 'Uncalibrated'}</strong> (${esc(sum.calibration.speed_units || 'px/frame')})${sum.calibration.note ? ` — ${esc(sum.calibration.note)}` : ''}</td></tr>` : ''}
          ${sum.detectors ? `<tr><td class="key">Detector Masks</td><td class="val">vehicle: ${sum.detectors.vehicle_loaded ? 'loaded' : 'not configured'} &middot; umbrella: ${sum.detectors.umbrella_loaded ? 'loaded' : 'not configured'}</td></tr>` : ''}
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
    wireModelSelect(host, detail);
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
    const activeStage = getActiveStageForTab(detail, tabName);
    const allVids = detail.allAnnotatedVideos || [];
    const header = stageModelSelectHtml(detail, tabName, activeStage);
    renderAnnotatedVideoGrid(host, allVids, {
      headerHtml: header,
      isRouteSession: false,
    });
    wireModelSelect(host, detail);
  } else if (tabName === 'validation') {
    // Validation routes measure the dense-flow engine, so the content is
    // video-scoped rather than per-stage — the selector is shown for
    // consistency and the body re-renders into its own container so the
    // header survives renderValidationTab's innerHTML writes.
    const activeStage = getActiveStageForTab(detail, tabName);
    host.innerHTML = `${stageModelSelectHtml(detail, tabName, activeStage)}<div id="validation-tab-body"></div>`;
    renderValidationTab($('#validation-tab-body'), detail);
    wireModelSelect(host, detail);
  } else if (tabName === 'raw') {
    const rawStage = getActiveStageForTab(detail, tabName) || {};
    const stageSummary = rawStage.summary || {};
    const rawJsonStr = JSON.stringify({
      summary: stageSummary,
      detections: detail.detections,
    }, null, 2);
    host.innerHTML = `
      ${stageModelSelectHtml(detail, tabName, rawStage)}
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
    wireModelSelect(host, detail);
  }
}

/* ------------------------------------------------------------------ ANNOTATED VIDEO GRID & MAXIMIZE */
function openVideoMaximize(initialIdx, allVids, startTime = 0, onExitCallback = null) {
  const existing = document.querySelector('.video-maximize-overlay');
  if (existing) existing.remove();

  let activeIdx = initialIdx;
  const overlay = document.createElement('div');
  overlay.className = 'video-maximize-overlay';

  function renderMaxContent() {
    const v = allVids[activeIdx] || allVids[0];
    const url = v.file.startsWith('/') ? v.file : `/api/files/run/${v.file}`;

    const pillsHtml = allVids.map((item, idx) => `
      <button class="cam-switch-pill ${idx === activeIdx ? 'active' : ''}" data-switch-idx="${idx}">
        ${esc(item.label)}
      </button>
    `).join('');

    overlay.innerHTML = `
      <div class="video-maximize-head">
        <div class="video-maximize-title">
          <span>🎥 <strong>${esc(v.label)}</strong></span>
          <span class="badge badge-right" style="font-size: 11px; margin-left: 8px;">FULLSCREEN VIEW</span>
        </div>
        <div class="video-maximize-actions">
          <button class="btn-sync-action" id="btn-native-fs" title="Enter OS Native Fullscreen">⛶ OS Fullscreen</button>
          <button class="btn-exit-maximize" id="btn-close-maximize" title="Exit Fullscreen (ESC)">✕ Exit Maximize</button>
        </div>
      </div>
      <div class="video-maximize-body">
        <video class="video-maximize-player" id="maximized-player" src="${esc(url)}" controls autoplay playsinline></video>
      </div>
      <div class="video-maximize-footer">
        <span style="font-size: 11.5px; color: var(--muted); margin-right: 6px;">Switch Camera Stream:</span>
        ${pillsHtml}
      </div>
    `;

    const player = overlay.querySelector('#maximized-player');
    if (player) {
      player.currentTime = startTime;
      player.play().catch(() => {});
    }

    const fsBtn = overlay.querySelector('#btn-native-fs');
    if (fsBtn && player) {
      fsBtn.addEventListener('click', () => {
        if (player.requestFullscreen) {
          player.requestFullscreen().catch(() => {});
        } else if (player.webkitRequestFullscreen) {
          player.webkitRequestFullscreen();
        }
      });
    }

    const closeBtn = overlay.querySelector('#btn-close-maximize');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        exitMaximize();
      });
    }

    overlay.querySelectorAll('[data-switch-idx]').forEach(pill => {
      pill.addEventListener('click', () => {
        const nextIdx = parseInt(pill.dataset.switchIdx, 10);
        if (nextIdx !== activeIdx) {
          const currT = player ? player.currentTime : 0;
          activeIdx = nextIdx;
          startTime = currT;
          renderMaxContent();
        }
      });
    });
  }

  function exitMaximize() {
    const player = overlay.querySelector('#maximized-player');
    const exitTime = player ? player.currentTime : 0;
    document.removeEventListener('keydown', onKeyDown);
    overlay.remove();
    if (onExitCallback) {
      onExitCallback(activeIdx, exitTime);
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Escape') {
      e.stopPropagation();
      exitMaximize();
    }
  }

  document.addEventListener('keydown', onKeyDown);
  renderMaxContent();
  document.body.appendChild(overlay);
}

function renderAnnotatedVideoGrid(containerEl, allVids, options = {}) {
  const count = allVids.length;
  const modalCard = document.querySelector('#modal .modal-card');

  if (count === 0) {
    if (modalCard) modalCard.classList.remove('wide-modal');
    containerEl.innerHTML = `${options.headerHtml || ''}<div class="empty">No annotated video available for this run or session.</div>`;
    return;
  }

  if (count === 1) {
    if (modalCard) modalCard.classList.remove('wide-modal');
    const v = allVids[0];
    const url = v.file.startsWith('/') ? v.file : `/api/files/run/${v.file}`;
    containerEl.innerHTML = `
      ${options.headerHtml || ''}
      <div class="video-grid-card" style="margin-top: 10px;">
        <div class="video-grid-card-head">
          <div class="video-card-title">
            <span style="color: var(--accent);">🎥</span> <strong>${esc(v.label)}</strong>
          </div>
          <div class="video-card-actions">
            <button class="btn-video-maximize" data-maximize-idx="0" title="Maximize this video">
              ⛶ Maximize
            </button>
          </div>
        </div>
        <video class="video-grid-player" id="grid-player-0" src="${esc(url)}" controls autoplay loop playsinline style="max-height: 480px;"></video>
        <div class="video-grid-card-footer">
          <span>Annotated Video Feed</span>
          <a href="${esc(url)}" download class="u-accent" style="text-decoration:none; font-size:11px; font-weight:600;">⬇ Download Video</a>
        </div>
      </div>
    `;

    const maxBtn = containerEl.querySelector('[data-maximize-idx="0"]');
    if (maxBtn) {
      maxBtn.addEventListener('click', () => {
        const p = containerEl.querySelector('#grid-player-0');
        openVideoMaximize(0, allVids, p ? p.currentTime : 0, (exitIdx, exitTime) => {
          if (p) {
            p.currentTime = exitTime;
            p.play().catch(() => {});
          }
        });
      });
    }
    return;
  }

  // Count > 1: Synchronized Grid
  if (modalCard) modalCard.classList.add('wide-modal');

  // Layout selection per user request:
  // - 5 videos: 2 top, 2 middle, 1 bottom (2 columns grid with 5th card single-span)
  // - 6 to 9+ videos: 3 columns (3 at top, 3 below, 3 below)
  // - 2 or 4 videos: 2 columns
  // - 3 videos: 3 columns
  let gridClass = 'video-grid-3';
  if (count === 2 || count === 4 || count === 5) {
    gridClass = 'video-grid-2';
  } else {
    gridClass = 'video-grid-3';
  }

  const cardsHtml = allVids.map((v, idx) => {
    const url = v.file.startsWith('/') ? v.file : `/api/files/run/${v.file}`;
    const isSingleSpan = (count === 5 && idx === 4);
    return `
      <div class="video-grid-card ${isSingleSpan ? 'single-span' : ''}" data-video-index="${idx}">
        <div class="video-grid-card-head">
          <div class="video-card-title" title="${esc(v.label)}">
            <span style="color: var(--accent);">🎥</span> <strong>${esc(v.label)}</strong>
          </div>
          <div class="video-card-actions">
            <button class="btn-video-maximize" data-maximize-idx="${idx}" title="Maximize ${esc(v.label)} to full screen">
              ⛶ Maximize
            </button>
          </div>
        </div>
        <video class="video-grid-player" id="grid-player-${idx}" src="${esc(url)}" autoplay loop muted playsinline controls></video>
        <div class="video-grid-card-footer">
          <span>Camera Feed ${idx + 1} of ${count}</span>
          <a href="${esc(url)}" download class="u-accent" style="text-decoration:none; font-size:11px; font-weight:600;">⬇ Download</a>
        </div>
      </div>
    `;
  }).join('');

  containerEl.innerHTML = `
    ${options.headerHtml || ''}
    <div class="video-sync-bar">
      <div class="video-sync-title">
        <span>🎥 Synchronized Multi-Camera Playback</span>
        <span class="badge badge-right" id="sync-active-count">${count} Concurrent Streams</span>
      </div>
      <div class="video-sync-actions">
        <button class="btn-sync-action" id="btn-sync-play">▶ Play All</button>
        <button class="btn-sync-action" id="btn-sync-pause">⏸ Pause All</button>
        <button class="btn-sync-action" id="btn-sync-restart">🔄 Sync &amp; Restart</button>
        <button class="btn-sync-action" id="btn-sync-mute">🔇 Mute All</button>
      </div>
    </div>
    <div class="video-multi-grid ${gridClass}" id="video-grid-container">
      ${cardsHtml}
    </div>
  `;

  // Autoplay all videos
  const players = Array.from(containerEl.querySelectorAll('.video-grid-player'));
  players.forEach(p => {
    p.play().catch(() => {});
  });

  // Sync toolbar handlers
  const btnPlay = containerEl.querySelector('#btn-sync-play');
  const btnPause = containerEl.querySelector('#btn-sync-pause');
  const btnRestart = containerEl.querySelector('#btn-sync-restart');
  const btnMute = containerEl.querySelector('#btn-sync-mute');

  if (btnPlay) {
    btnPlay.addEventListener('click', () => {
      players.forEach(p => p.play().catch(() => {}));
    });
  }
  if (btnPause) {
    btnPause.addEventListener('click', () => {
      players.forEach(p => p.pause());
    });
  }
  if (btnRestart) {
    btnRestart.addEventListener('click', () => {
      players.forEach(p => {
        p.currentTime = 0;
        p.play().catch(() => {});
      });
    });
  }
  let allMuted = true;
  if (btnMute) {
    btnMute.addEventListener('click', () => {
      allMuted = !allMuted;
      players.forEach(p => { p.muted = allMuted; });
      btnMute.textContent = allMuted ? '🔇 Mute All' : '🔊 Unmute All';
    });
  }

  // Maximize handlers
  containerEl.querySelectorAll('[data-maximize-idx]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.maximizeIdx, 10);
      const targetPlayer = containerEl.querySelector(`#grid-player-${idx}`);
      const currTime = targetPlayer ? targetPlayer.currentTime : 0;
      openVideoMaximize(idx, allVids, currTime, (exitIdx, exitTime) => {
        const exitPlayer = containerEl.querySelector(`#grid-player-${exitIdx}`);
        if (exitPlayer) {
          exitPlayer.currentTime = exitTime;
          exitPlayer.play().catch(() => {});
        }
      });
    });
  });
}

function renderRouteSessionModalTab(tabName, detail, host) {
  const modalCard = $('#modal .modal-card');
  if (modalCard && tabName !== 'video') {
    modalCard.classList.remove('wide-modal');
  }
  const sname = detail.videoName;
  const sess = detail.sessionData || {};
  const sum = sess.summary || {};
  const cams = sess.cameras || {};
  const camList = Object.values(cams);

  const num = (v, nd = 2, u = '') => v == null ? '—' : `${Number(v).toFixed(nd)}${u ? ' ' + u : ''}`;

  if (tabName === 'overview') {
    const totalDets = sum.total_detections || camList.reduce((acc, c) => acc + (c.detections || 0), 0);
    const cameraCount = camList.length;
    const maxCrush = sum.max_crush_risk_pct || 0;

    // Corridor transit narratives
    let narrativeHtml = '';
    if (sum.transit_narratives && sum.transit_narratives.length > 0) {
      narrativeHtml = `
        <div class="detail-section-title" style="margin-top: 18px;">📍 Corridor Transit &amp; Propagation Analysis</div>
        <div style="margin-top: 8px;">
          ${sum.transit_narratives.map(n => {
            const util = Math.abs(n.capacity_utilization_pct || 0);
            const flowMin = n.source_flow_pax_min != null ? Math.round(n.source_flow_pax_min) : Math.round(Math.abs(n.source_flow || 0) * 60);
            const badgeClass = (n.status === 'critical' || util >= 100) ? 'alert-badge-bad' : ((n.status === 'warning' || util >= 70) ? 'alert-badge-warn' : 'alert-badge-ok');
            return `
            <div class="narrative-card ${n.status || 'nominal'}" style="margin-bottom: 8px;">
              <div class="narrative-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span class="route-title" style="font-size: 14px; font-weight: 700;">📍 ${esc(n.source_name)} ➔ ${esc(n.target_name)}</span>
                <span class="${badgeClass}">🌊 Flow: ${flowMin} pax/min · ⏱️ ${Math.round(n.travel_time_sec)}s lead time · Demand: ${util.toFixed(1)}%</span>
              </div>
              <p class="narrative-text" style="font-size: 12.5px; color: var(--text-dim);">${esc(n.summary_text)}</p>
            </div>
          `;}).join('')}
        </div>
      `;
    }

    // Camera breakdown table
    const camRows = camList.map(c => {
      const camSum = (sum.cameras && sum.cameras[c.camera_id]) || {};
      const cDensity = camSum.density != null ? num(camSum.density, 1) : '—';
      const cSpeed = camSum.speed != null ? num(camSum.speed, 2) : '—';
      const cPressure = camSum.pressure != null ? num(camSum.pressure, 3) : '—';
      const cCrush = camSum.crush_risk_pct != null ? `${camSum.crush_risk_pct.toFixed(1)}%` : '—';
      const cCf = camSum.counterflow_pct != null ? `${camSum.counterflow_pct.toFixed(1)}%` : '—';
      const reportUrl = `/api/files/session/${esc(sname)}/${esc(c.camera_id)}/report.html`;
      const videoUrl = `/api/files/session/${esc(sname)}/${esc(c.camera_id)}/annotated.mp4`;
      const csvUrl = `/api/files/session/${esc(sname)}/${esc(c.camera_id)}/detections.csv`;

      return `
        <tr>
          <td><strong>${esc(c.camera_name || c.camera_id)}</strong> <small style="color:var(--muted);">(${esc(c.camera_id)})</small></td>
          <td><span class="session-cam-tag ${esc(c.status)}">${esc(c.status)}</span></td>
          <td style="font-family: var(--font-code);">${c.detections || 0}</td>
          <td style="font-family: var(--font-code); color: #f97316;">${cDensity}</td>
          <td style="font-family: var(--font-code); color: #60a5fa;">${cSpeed}</td>
          <td style="font-family: var(--font-code); color: #ef4444;">${cPressure}</td>
          <td style="font-family: var(--font-code); color: #fb923c;">${cCrush}</td>
          <td style="font-family: var(--font-code); color: #f59e0b;">${cCf}</td>
          <td style="display: flex; gap: 6px;">
            ${c.status === 'done' ? `
              <a href="${reportUrl}" target="_blank" class="link-btn" style="padding: 2px 8px; font-size: 11px;">📋 Report</a>
              <a href="${videoUrl}" download class="link-btn" style="padding: 2px 8px; font-size: 11px;">🎥 Video</a>
              <a href="${csvUrl}" download class="link-btn" style="padding: 2px 8px; font-size: 11px;">📊 CSV</a>
            ` : '—'}
          </td>
        </tr>
      `;
    }).join('');

    host.innerHTML = `
      <div class="detail-overview-grid">
        <div class="detail-metric-card">
          <div class="val highlight">${cameraCount} Cameras</div>
          <div class="lbl">Route Nodes in Session</div>
        </div>
        <div class="detail-metric-card">
          <div class="val">${totalDets.toLocaleString()}</div>
          <div class="lbl">Total Corridor Detections</div>
        </div>
        <div class="detail-metric-card">
          <div class="val" style="color: ${maxCrush > 20 ? 'var(--bad)' : 'var(--ok)'};">${maxCrush.toFixed(1)}%</div>
          <div class="lbl">Corridor Peak Crush Risk</div>
        </div>
      </div>

      <div class="overview-analytics-box">
        <div class="section-title">
          <span>📊 Corridor Aggregated Metrics (10 Crowd-Safety Indicators)</span>
          ${sess.report_html ? `<a href="/api/files/session/${esc(sname)}/session_report.html" target="_blank" class="btn-card-detail">📄 View Fused HTML Report</a>` : ''}
        </div>

        <div class="overview-kpis">
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Corridor Density</div>
            <div class="overview-kpi-val" style="color: #f97316;">${num(sum.avg_density, 1)}</div>
            <div class="overview-kpi-sub">Capacity-weighted (pax/m²)</div>
          </div>
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Fleet Mean Velocity</div>
            <div class="overview-kpi-val" style="color: #60a5fa;">${num(sum.avg_speed, 2)}</div>
            <div class="overview-kpi-sub">peak ${num(sum.peak_speed, 2)} px/frame</div>
          </div>
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Combined Flow Rate</div>
            <div class="overview-kpi-val" style="color: #22d3ee;">${num(sum.total_specific_flow, 2)}</div>
            <div class="overview-kpi-sub">Total throughput (pax/s)</div>
          </div>
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Max Crowd Pressure</div>
            <div class="overview-kpi-val" style="color: #ef4444;">${num(sum.max_crowd_pressure, 3)}</div>
            <div class="overview-kpi-sub">Peak corridor pressure</div>
          </div>
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Worst Divergence</div>
            <div class="overview-kpi-val" style="color: #fb7185;">${num(sum.worst_divergence, 3)}</div>
            <div class="overview-kpi-sub">negative = compression</div>
          </div>
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Stop &amp; Go Waves</div>
            <div class="overview-kpi-val" style="color: #fbbf24;">${num(sum.avg_stop_go, 2)}</div>
            <div class="overview-kpi-sub">Shockwave propagation index</div>
          </div>
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Oscillation Symmetry</div>
            <div class="overview-kpi-val" style="color: #f472b6;">${num(sum.max_oscillation_symmetry, 2)}</div>
            <div class="overview-kpi-sub">Transverse surge &amp; rocking</div>
          </div>
          <div class="overview-kpi-item tier-one">
            <div class="overview-kpi-lbl">Peak Crush Risk</div>
            <div class="overview-kpi-val" style="color: #fb923c;">${num(sum.max_crush_risk_pct, 1, '%')}</div>
            <div class="overview-kpi-sub">${sum.total_crush_events || 0} crush events</div>
          </div>
          <div class="overview-kpi-item">
            <div class="overview-kpi-lbl">Counterflow Friction</div>
            <div class="overview-kpi-val" style="color: #f59e0b;">${num(sum.avg_counterflow_pct, 1, '%')}</div>
            <div class="overview-kpi-sub">${sum.total_counterflow_events || 0} friction events</div>
          </div>
          <div class="overview-kpi-item tier-three">
            <div class="overview-kpi-lbl">Directional Entropy</div>
            <div class="overview-kpi-val" style="color: #a78bfa;">${num(sum.max_directional_entropy, 2)}</div>
            <div class="overview-kpi-sub">Disorder &amp; panic (0-3 bits)</div>
          </div>
        </div>
      </div>

      ${narrativeHtml}

      <div class="detail-section-title" style="margin-top: 20px;">🎥 Camera Pipeline Breakdown &amp; Sub-Reports</div>
      <table class="stages" style="width: 100%; margin-top: 8px;">
        <thead>
          <tr>
            <th>Camera Slot</th>
            <th>Status</th>
            <th>Detections</th>
            <th>Density</th>
            <th>Speed</th>
            <th>Pressure</th>
            <th>Crush %</th>
            <th>Counterflow %</th>
            <th>Artifacts</th>
          </tr>
        </thead>
        <tbody>
          ${camRows}
        </tbody>
      </table>

      <div class="detail-section-title" style="margin-top: 20px;">Session Metadata &amp; Links</div>
      <table class="detail-info-table">
        <tbody>
          <tr><td class="key">Session Name</td><td class="val">${esc(sname)}</td></tr>
          <tr><td class="key">Topology Model</td><td class="val">Simhastha Kumbh Mela 2027 Corridor Graph</td></tr>
          <tr><td class="key">Created At</td><td class="val">${esc(sess.created_at || '—')}</td></tr>
          <tr><td class="key">Status</td><td class="val"><strong>${esc(sess.status)}</strong></td></tr>
          <tr>
            <td class="key">Corridor Artifacts</td>
            <td class="val" style="display: flex; flex-wrap: wrap; gap: 8px;">
              ${sess.report_html ? `<a href="/api/files/session/${esc(sname)}/session_report.html" target="_blank" class="link-btn">📄 Fused HTML Report</a>` : ''}
              ${sess.summary_json ? `<a href="/api/files/session/${esc(sname)}/session_summary.json" target="_blank" class="link-btn">📈 Summary Stats (JSON)</a>` : ''}
              <a href="/api/files/session/${esc(sname)}/session_manifest.json" target="_blank" class="link-btn">📦 Manifest (JSON)</a>
            </td>
          </tr>
        </tbody>
      </table>
    `;
  } else if (tabName === 'timeline') {
    const allCams = detail.allStages || [];
    const activeKey = detail.activeTimelineModelKey || (allCams[0] ? allCams[0].camera_id : '');

    const selectorRows = allCams.map(c => {
      const isActive = c.camera_id === activeKey;
      return `<tr class="video-select-row u-clickable${isActive ? ' active-video-row is-active' : ''}" data-load-cam="${esc(c.camera_id)}" data-cam-label="${esc(c.camera_name || c.camera_id)}">
        <td class="u-strong">${esc(c.camera_name || c.camera_id)}</td>
        <td class="u-muted">${esc(c.camera_id)}</td>
        <td class="${c.positives > 0 ? 'pos' : 'zero'} u-strong">${c.positives || 0} alerts</td>
        <td>${c.detections || 0} rows</td>
        <td><span class="u-accent">${isActive ? '⬤ Viewing' : '▶ Load'}</span></td>
      </tr>`;
    }).join('');

    host.innerHTML = `
      <div class="detail-section-title u-mt-0">⏱️ Select Camera to View Frame-by-Frame Detections</div>
      <p class="hint u-mb-3">Click any camera row to load its detection log and kinematic vectors below.</p>
      <table class="stages">
        <thead><tr><th>Camera</th><th>ID</th><th class="u-num">Alerts</th><th class="u-num">Total Rows</th><th></th></tr></thead>
        <tbody>${selectorRows}</tbody>
      </table>
      <div id="timeline-detections-host"><div class="loading">Loading detections…</div></div>
    `;

    async function loadCameraDetections(camId, camLabel) {
      detail.activeTimelineModelKey = camId;
      host.querySelectorAll('.video-select-row').forEach(r => {
        const isNow = r.dataset.loadCam === camId;
        r.classList.toggle('is-active', isNow);
        const action = r.querySelector('span');
        if (action) action.textContent = isNow ? '⬤ Viewing' : '▶ Load';
      });

      const detectHost = $('#timeline-detections-host');
      if (!detectHost) return;
      detectHost.innerHTML = `<div class="loading">Loading detections for ${esc(camLabel)}…</div>`;

      try {
        const d = await api(`/api/sessions/${encodeURIComponent(sname)}/detections/${encodeURIComponent(camId)}?limit=500`);
        renderDetectionsTable(d, camLabel, detectHost);
      } catch (err) {
        if ($('#timeline-detections-host')) {
          $('#timeline-detections-host').innerHTML = `<div class="err">${esc(err.message)}</div>`;
        }
      }
    }

    host.querySelectorAll('[data-load-cam]').forEach(row => {
      row.addEventListener('click', () => {
        loadCameraDetections(row.dataset.loadCam, row.dataset.camLabel);
      });
    });

    if (activeKey) {
      const activeCam = allCams.find(c => c.camera_id === activeKey);
      loadCameraDetections(activeKey, activeCam ? (activeCam.camera_name || activeCam.camera_id) : activeKey);
    }
  } else if (tabName === 'video') {
    const allVids = detail.allAnnotatedVideos || [];
    renderAnnotatedVideoGrid(host, allVids, {
      isRouteSession: true,
      sessionName: sname,
    });
  } else if (tabName === 'raw') {
    host.innerHTML = `
      <div class="u-row u-mb-3" style="display:flex; justify-content:space-between; align-items:center;">
        <span class="u-caption">Full Route Session Manifest &amp; Summary JSON</span>
        <button id="copy-json-btn" class="link-btn">📋 Copy JSON to Clipboard</button>
      </div>
      <pre class="raw-json-block">${esc(JSON.stringify(sess, null, 2))}</pre>
    `;
    const copyBtn = $('#copy-json-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(JSON.stringify(sess, null, 2));
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
  const maxOverlay = document.querySelector('.video-maximize-overlay');
  if (maxOverlay) maxOverlay.remove();
  const modalCard = document.querySelector('#modal .modal-card');
  if (modalCard) modalCard.classList.remove('wide-modal');
  // The live stage lives in this modal's body while its tab is open; park
  // it before the body is cleared or the canvas would be destroyed with it.
  parkLiveStage();
  if (liveState.ws || liveState.active) closeLivePlayer();
  setLiveChrome({ tabVisible: false, streaming: false });
  $('#modal').classList.add('hidden');
  $('#modal-body').innerHTML = '';
  state.currentDetail = null;
}

/* ------------------------------------------------------------------- live preview */
let liveState = {
  ws: null,
  jobId: null,
  active: false,
  selectedMetric: 'people',
  history: null,
};

/* Metric definitions for the live rail.  `read` pulls the value out of a KPI
   payload, `fmt` renders it for the tile, and `el` is the tile's value node —
   one table so the tiles, the inspector chart and the tick animation all
   agree on what a metric is. */
const LIVE_METRICS = {
  people:    { label: 'People Count',        el: '#live-kpi-people',
               read: k => k.person_count,  fmt: v => String(Math.round(v)) },
  positives: { label: 'Positive Alerts',     el: '#live-kpi-events',
               read: k => k.positives,     fmt: v => String(Math.round(v)) },
  crush:     { label: 'Crush Risk',          el: '#live-kpi-crush',
               read: k => k.crush_risk,    fmt: v => Number(v).toFixed(2) },
  flow:      { label: 'Flow Rate (pax/min)', el: '#live-kpi-flow',
               read: k => k.flow_rate,     fmt: v => Number(v).toFixed(1) },
  time:      { label: 'Video Timestamp',     el: '#live-kpi-time',
               read: k => k.timestamp_sec, fmt: v => fmtClock(v) },
};

/* Frames of history kept per metric for the inspector chart.  Bounded: a
   live camera runs for a shift, and an unbounded array here would grow for
   as long as it does. */
const LIVE_HISTORY_LEN = 240;

function fmtClock(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function resetLiveHistory() {
  liveState.history = {};
  Object.keys(LIVE_METRICS).forEach(k => { liveState.history[k] = []; });
}

function selectLiveMetric(key) {
  if (!LIVE_METRICS[key]) return;
  liveState.selectedMetric = key;
  document.querySelectorAll('.live-metrics-rail .hud-item').forEach(btn => {
    const on = btn.dataset.metric === key;
    btn.classList.toggle('is-selected', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
  const nameEl = $('#metric-detail-name');
  if (nameEl) nameEl.textContent = LIVE_METRICS[key].label;
  drawMetricSpark();
}

/* Live trace for the selected metric.  This is what makes "is it actually
   detecting right now?" answerable: the line extends by one point per frame
   the model returns, so a stalled pipeline shows a flat tail rather than a
   number that merely looks plausible. */
function drawMetricSpark() {
  const canvas = $('#metric-spark');
  if (!canvas || !liveState.history) return;
  const key = liveState.selectedMetric;
  const series = liveState.history[key] || [];
  const metric = LIVE_METRICS[key];

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 240;
  const cssH = canvas.clientHeight || 64;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const statNow = $('#metric-stat-now');
  const statMin = $('#metric-stat-min');
  const statMax = $('#metric-stat-max');
  const statN = $('#metric-stat-n');
  if (statN) statN.textContent = String(series.length);

  if (!series.length) {
    if (statNow) statNow.textContent = '–';
    if (statMin) statMin.textContent = '–';
    if (statMax) statMax.textContent = '–';
    return;
  }

  const min = Math.min(...series);
  const max = Math.max(...series);
  if (statNow) statNow.textContent = metric.fmt(series[series.length - 1]);
  if (statMin) statMin.textContent = metric.fmt(min);
  if (statMax) statMax.textContent = metric.fmt(max);

  // A flat series still needs a band to draw in, or every point lands on the
  // same pixel row and the chart reads as broken rather than as steady.
  const span = (max - min) || 1;
  const pad = 4;
  const plotW = cssW - pad * 2;
  const plotH = cssH - pad * 2;
  const xAt = i => pad + (series.length === 1 ? plotW : (i / (series.length - 1)) * plotW);
  const yAt = v => pad + plotH - ((v - min) / span) * plotH;

  const stroke = 'rgba(99, 102, 241, 0.95)';
  ctx.beginPath();
  series.forEach((v, i) => (i ? ctx.lineTo(xAt(i), yAt(v)) : ctx.moveTo(xAt(i), yAt(v))));
  const fill = ctx.createLinearGradient(0, pad, 0, pad + plotH);
  fill.addColorStop(0, 'rgba(99, 102, 241, 0.35)');
  fill.addColorStop(1, 'rgba(99, 102, 241, 0.02)');
  ctx.lineTo(xAt(series.length - 1), pad + plotH);
  ctx.lineTo(xAt(0), pad + plotH);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.beginPath();
  series.forEach((v, i) => (i ? ctx.lineTo(xAt(i), yAt(v)) : ctx.moveTo(xAt(i), yAt(v))));
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Leading point: the frame that just arrived.
  const lx = xAt(series.length - 1);
  const ly = yAt(series[series.length - 1]);
  ctx.beginPath();
  ctx.arc(lx, ly, 2.8, 0, Math.PI * 2);
  ctx.fillStyle = '#22d3ee';
  ctx.fill();
}

function updateLiveKPIs(kpis) {
  if (!kpis) return;
  if (!liveState.history) resetLiveHistory();

  // One pass over the metric table: record history, paint the tile, and
  // flash it when the number actually moved.
  Object.entries(LIVE_METRICS).forEach(([key, metric]) => {
    const raw = metric.read(kpis);
    if (raw === undefined || raw === null) return;

    const value = Number(raw);
    if (!Number.isFinite(value)) return;

    const series = liveState.history[key];
    series.push(value);
    if (series.length > LIVE_HISTORY_LEN) series.shift();

    const el = $(metric.el);
    if (!el) return;
    const text = metric.fmt(value);
    if (el.textContent !== text) {
      el.textContent = text;
      const tile = el.closest('.hud-item');
      if (tile) {
        tile.classList.remove('just-changed');
        // Reflow so the animation restarts on consecutive changes.
        void tile.offsetWidth;
        tile.classList.add('just-changed');
      }
    }
  });

  // Crush risk keeps its severity colouring.
  const crushEl = $('#live-kpi-crush');
  if (crushEl && kpis.crush_risk !== undefined) {
    const risk = Number(kpis.crush_risk);
    crushEl.style.color = risk > 0.5 ? 'var(--bad, #f43f5e)' : (risk > 0.2 ? 'var(--warn, #f59e0b)' : 'var(--text, #f8fafc)');
  }

  const fpsEl = $('#live-fps');
  if (fpsEl && kpis.fps !== undefined) {
    fpsEl.textContent = `FPS: ${kpis.fps > 0 ? kpis.fps.toFixed(1) : '--'}`;
  }

  drawMetricSpark();
  updateDashboardLiveKPIs(kpis);
}

/* The dashboard KPI bar keeps counting while a live job streams, so the
   numbers move whether or not the live modal is the thing on screen. */
function updateDashboardLiveKPIs(kpis) {
  const mainEventsEl = $('#kpi-events');
  if (mainEventsEl && kpis.positives !== undefined) {
    const baseEvents = state.historyData.reduce((acc, g) => acc + (g.total_positives || 0), 0);
    const text = String(baseEvents + (kpis.positives || 0));
    if (mainEventsEl.textContent !== text) {
      mainEventsEl.textContent = text;
      flashKpiCard(mainEventsEl);
    }
  }

  const livePeopleEl = $('#kpi-live-people');
  if (livePeopleEl && kpis.person_count !== undefined) {
    const text = String(Math.round(kpis.person_count));
    if (livePeopleEl.textContent !== text) {
      livePeopleEl.textContent = text;
      flashKpiCard(livePeopleEl);
    }
  }
}

function flashKpiCard(valEl) {
  const card = valEl.closest('.kpi-card');
  if (!card) return;
  card.classList.remove('just-changed');
  void card.offsetWidth;
  card.classList.add('just-changed');
}

/* Show/hide the dashboard's live-only card and mark the KPI bar as live. */
function setDashboardLiveState(active) {
  const card = $('#kpi-card-live');
  if (card) card.classList.toggle('hidden', !active);
  const bar = document.querySelector('.kpi-bar');
  if (bar) bar.classList.toggle('is-live', active);
}

/* Adds a "Replay saved video" action to the finished-stream overlay, or a
   plain note when the source was a camera and no video was written. */
function renderLiveReplayAction(annotatedPath) {
  const overlay = $('#live-status-overlay');
  if (!overlay) return;
  overlay.querySelector('.live-replay-action')?.remove();

  const wrap = document.createElement('div');
  wrap.className = 'live-replay-action';

  if (annotatedPath) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-primary';
    btn.textContent = '▶ Replay saved video';
    btn.addEventListener('click', () => renderModalTab('video'));
    wrap.appendChild(btn);
  } else {
    const note = document.createElement('div');
    note.className = 'hint';
    note.textContent = 'No annotated video for this source — a live camera has no end to record.';
    wrap.appendChild(note);
  }
  overlay.appendChild(wrap);
}

function b64toBlob(b64Data, contentType = 'image/jpeg') {
  const byteCharacters = atob(b64Data);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: contentType });
}

/* Show or hide the live-only chrome: the Live Stream tab, the FPS badge and
   the Stop button.  All three belong to a run that is happening now, and a
   finished run should not keep offering to stop it. */
function setLiveChrome({ tabVisible, streaming }) {
  const tabBtn = document.querySelector('#modal-nav .modal-tab-btn[data-mtab="live"]');
  if (tabBtn) tabBtn.classList.toggle('hidden', !tabVisible);
  const fpsEl = $('#live-fps');
  if (fpsEl) fpsEl.classList.toggle('hidden', !streaming);
  const stopBtn = $('#live-btn-stop');
  if (stopBtn) stopBtn.classList.toggle('hidden', !streaming);
}

/* Pull the job's stages and detections into the shape every detail tab
   already reads, so a live run's Overview / Timeline / Video / Validation /
   Raw JSON tabs are rendered by exactly the same code that renders a batch
   run's.  Tolerant of missing artifacts: mid-run there are none yet. */
async function loadLiveJobDetail(jobId, modelKey, videoName) {
  let job = null;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch { /* job record not readable yet */ }

  const stages = (job && job.stages) ? job.stages : [];
  const stage = stages.find(st => st.model_key === modelKey)
    || stages[0]
    || { model_key: modelKey, model_label: modelKey };

  let detections = { rows: [], total: 0 };
  try {
    detections = await api(`/api/jobs/${jobId}/detections/${stage.model_key}?limit=500`);
  } catch { /* detections.json is only written when the run finishes */ }

  const annotatedVideos = stages
    .filter(st => st.annotated)
    .map(st => ({ label: st.model_key, key: st.model_key, file: st.annotated }));

  state.currentDetail = {
    videoName: (job && job.video_name) ? job.video_name : (videoName || jobId),
    jobId,
    group: null,
    primaryStage: stage,
    allStages: stages.length ? stages : [stage],
    activeTimelineModelKey: stage.model_key,
    detections,
    allAnnotatedVideos: annotatedVideos,
    activeAnnotatedVideo: annotatedVideos.length ? annotatedVideos[0].file : null,
    isLiveJob: true,
  };
  return state.currentDetail;
}

/* Re-read the job once the stream ends, so the tabs that were empty during
   the run (annotated video, timeline, raw JSON) pick up the files the export
   just wrote — without the operator having to close and reopen anything. */
async function refreshLiveDetailAfterRun() {
  if (!liveState.jobId) return;
  const tab = state.activeModalTab || 'live';
  try {
    await loadLiveJobDetail(liveState.jobId, liveState.modelKey, liveState.videoName);
    renderModalTab(tab);
  } catch { /* leave the last-rendered view in place */ }
}

async function openLivePlayer(jobId, videoName, modelName, modelKey) {
  liveState.jobId = jobId;
  liveState.active = true;
  liveState.videoName = videoName;
  liveState.modelKey = modelKey || (modelName || '').split(',')[0].trim();

  state.modalOpener = document.activeElement;
  $('#modal-title').textContent = `Live Run — ${videoName || jobId}`;
  $('#modal').classList.remove('hidden');
  setLiveChrome({ tabVisible: true, streaming: true });

  const fpsEl = $('#live-fps');
  if (fpsEl) fpsEl.textContent = 'FPS: --';

  // Reset live KPIs and the metric traces — a new run starts a new history,
  // otherwise the inspector chart would splice two videos into one line.
  const stageEl = document.querySelector('#live-stage');
  if (stageEl) stageEl.classList.remove('is-idle');
  resetLiveHistory();
  setDashboardLiveState(true);
  updateLiveKPIs({ person_count: 0, positives: 0, crush_risk: 0, flow_rate: 0, timestamp_sec: 0, fps: 0 });
  resetLiveHistory();
  selectLiveMetric(liveState.selectedMetric || 'people');

  // Show the stream immediately; the rest of the tabs fill in behind it.
  state.currentDetail = state.currentDetail || {};
  renderModalTab('live');
  loadLiveJobDetail(jobId, liveState.modelKey, videoName)
    .then(() => { if (state.activeModalTab === 'live') renderModalTab('live'); })
    .catch(() => {});

  const statusOverlay = $('#live-status-overlay');
  const statusMsg = $('#live-status-msg');
  if (statusOverlay) {
    statusOverlay.classList.remove('hidden');
    statusOverlay.querySelector('.live-replay-action')?.remove();
    const spinner = statusOverlay.querySelector('.live-status-spinner');
    if (spinner) spinner.style.display = 'block';
    if (statusMsg) statusMsg.textContent = 'Connecting live stream…';
  }

  // Connect WebSocket
  if (liveState.ws) {
    try { liveState.ws.close(); } catch { }
  }

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const token = state.token || '';
  const tokenParam = token ? `?token=${encodeURIComponent(token)}` : '';
  const wsUrl = `${protocol}//${location.host}/ws/live/${encodeURIComponent(jobId)}${tokenParam}`;

  try {
    const ws = new WebSocket(wsUrl);
    liveState.ws = ws;

    ws.onopen = () => {
      if (statusMsg) statusMsg.textContent = 'Waiting for model frames…';
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.event === 'init') {
          if (statusMsg) {
            statusMsg.textContent = msg.message || 'Stream initialized…';
          }
        } else if (msg.event === 'status') {
          // Startup progress (weight loading) — keep the spinner up but say
          // what is actually happening instead of a generic message.
          if (statusOverlay) statusOverlay.classList.remove('hidden');
          if (statusMsg && msg.message) statusMsg.textContent = msg.message;
        } else if (msg.event === 'frame') {
          if (statusOverlay && !statusOverlay.classList.contains('hidden')) {
            statusOverlay.classList.add('hidden');
          }
          if (msg.kpis) {
            updateLiveKPIs(msg.kpis);
          }
          if (msg.jpeg_b64) {
            const blob = b64toBlob(msg.jpeg_b64, 'image/jpeg');
            createImageBitmap(blob).then((bitmap) => {
              const canvas = $('#live-canvas');
              if (canvas) {
                if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
                  canvas.width = bitmap.width;
                  canvas.height = bitmap.height;
                }
                const ctx = canvas.getContext('2d');
                ctx.drawImage(bitmap, 0, 0);
              }
              bitmap.close();
            }).catch(() => {});
          }
        } else if (msg.event === 'error') {
          if (statusOverlay) {
            statusOverlay.classList.remove('hidden');
            const spinner = statusOverlay.querySelector('.live-status-spinner');
            if (spinner) spinner.style.display = 'none';
          }
          if (statusMsg) statusMsg.textContent = `Error: ${msg.error || 'Pipeline error'}`;
        } else if (msg.event === 'done') {
          if (statusOverlay) {
            statusOverlay.classList.remove('hidden');
            const spinner = statusOverlay.querySelector('.live-status-spinner');
            if (spinner) spinner.style.display = 'none';
          }
          if (statusMsg) {
            statusMsg.textContent = msg.status === 'cancelled'
              ? 'Live stream cancelled.'
              : (msg.export_error
                  ? `Stream finished, but saving outputs failed: ${msg.export_error}`
                  : 'Live stream completed — outputs saved.');
          }
          // The stream cannot be rewound (it was never a video, just frames
          // as they were computed), but the run wrote one — so offer that
          // rather than leaving a frozen last frame as the only thing here.
          renderLiveReplayAction(msg.artifacts && msg.artifacts.annotated);
          const doneStage = document.querySelector('#live-stage');
          if (doneStage) doneStage.classList.add('is-idle');
          setDashboardLiveState(false);
          // The run is over: keep the recorded stream viewable, but stop
          // offering to stop a job that is no longer running.
          setLiveChrome({ tabVisible: true, streaming: false });
          liveState.active = false;
          refreshLiveDetailAfterRun();
          refreshJobs();
        }
      } catch (e) {
        console.error('Error processing live ws message:', e);
      }
    };

    ws.onerror = () => {
      if (statusOverlay) statusOverlay.classList.remove('hidden');
      if (statusMsg) statusMsg.textContent = 'WebSocket connection error.';
    };

    ws.onclose = () => {
      // ws closed
    };
  } catch (err) {
    if (statusOverlay) statusOverlay.classList.remove('hidden');
    if (statusMsg) statusMsg.textContent = `Could not connect: ${err.message}`;
  }
}

function closeLivePlayer() {
  if (liveState.ws) {
    try { liveState.ws.close(); } catch { }
    liveState.ws = null;
  }
  liveState.active = false;
  liveState.jobId = null;
  setDashboardLiveState(false);
  setLiveChrome({ tabVisible: false, streaming: false });
  parkLiveStage();
  refreshJobs();
}

/* Clicking a metric tile opens its live trace in the inspector.  Delegated,
   because the rail is in the modal markup and this binds once at startup. */
function initLiveMetricRail() {
  const rail = document.querySelector('.live-metrics-rail');
  if (!rail) return;
  rail.addEventListener('click', (ev) => {
    const tile = ev.target.closest('.hud-item[data-metric]');
    if (tile) selectLiveMetric(tile.dataset.metric);
  });
  window.addEventListener('resize', () => drawMetricSpark());
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

  const isLive = $('#live-mode')?.checked || false;
  const btn = $('#run-btn');
  btn.disabled = true;
  btn.textContent = isLive ? 'Starting live preview…' : 'Starting job…';

  try {
    const jobRes = await postJSON('/api/jobs', {
      source,
      models: [...state.selected],
      sample_every_n_frames: parseInt($('#stride').value, 10),
      device: $('#device').value || null,
      export_video: $('#export-video').checked,
      threshold: state.threshold,
      mode: isLive ? 'live' : 'batch',
    });
    state.openJobs.clear();
    refreshJobs();

    if (isLive && jobRes && jobRes.id) {
      openLivePlayer(jobRes.id, jobRes.video_name || source,
                     [...state.selected].join(', '), [...state.selected][0]);
    }
  } catch (e) {
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = isLive ? '▶ Run Live Preview' : '▶ Run selected models';
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
  document.addEventListener('keydown', (e) => {
    // One modal now, live or not, so Escape means the same thing either way:
    // closeModal tears down an active stream on its way out.
    if (e.key === 'Escape') closeModal();
  });

  // Live preview controls wiring
  const liveModeCb = $('#live-mode');
  const liveNote = $('#live-mode-note');
  const runBtn = $('#run-btn');
  if (liveModeCb) {
    liveModeCb.addEventListener('change', () => {
      if (liveModeCb.checked) {
        if (liveNote) liveNote.classList.remove('hidden');
        if (runBtn) runBtn.textContent = '▶ Run Live Preview';
      } else {
        if (liveNote) liveNote.classList.add('hidden');
        if (runBtn) runBtn.textContent = '▶ Run selected models';
      }
    });
  }

  const liveStopBtn = $('#live-btn-stop');
  if (liveStopBtn) {
    liveStopBtn.addEventListener('click', async () => {
      if (liveState.jobId) {
        try {
          await postJSON(`/api/jobs/${liveState.jobId}/cancel`, {});
        } catch { }
      }
    });
  }

  $('#clear-outputs').addEventListener('click', async () => {
    if (!confirm('Delete every generated log and annotated video in outputs/?\nSource videos in test_videos/ are not touched.')) return;
    const r = await api('/api/outputs', { method: 'DELETE' });
    $('#run-error').textContent = `Deleted ${r.removed} item(s).`;
    state.openHistory.clear();
    state.openAnpr.clear();
    state.openJobs.clear();
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
      state.openJobs.clear();
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

  $('#video-file')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const status = $('#upload-status');
    if (status) status.textContent = `Uploading ${file.name}…`;
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await api('/api/videos/upload', { method: 'POST', body: fd });
      e.target.dataset.uploaded = r.name;
      if (status) status.textContent = `Ready: ${r.name} (${r.size_mb} MB)`;
      loadVideos();
    } catch (err) {
      if (status) status.textContent = err.message;
    }
  });
}

/* ------------------------------------------------------------------ Route View Controller */

const routeState = {
  topology: null,
  metrics: null,
  alerts: [],
  sparklines: {},
  ws: null,
  wsTimer: null,
  selectedEdge: null,
};

async function loadRouteTopology() {
  try {
    const data = await api('/api/topology');
    routeState.topology = data;
  } catch (err) {
    console.error('Failed to load topology:', err);
  }
}

async function loadRouteMetrics() {
  try {
    const data = await api('/api/fusion/metrics');
    routeState.metrics = data;
  } catch (err) {
    console.error('Failed to load fusion metrics:', err);
  }
}

async function loadRouteAlerts() {
  try {
    const data = await api('/api/fusion/alerts?active=true');
    routeState.alerts = data.alerts || [];
    renderFusionAlerts();
  } catch (err) {
    console.error('Failed to load fusion alerts:', err);
  }
}

async function loadRouteSparklines() {
  try {
    const data = await api('/api/fusion/sparklines');
    routeState.sparklines = data.sparklines || {};
    renderSparklines();
  } catch (err) {
    console.error('Failed to load sparklines:', err);
  }
}

async function refreshRouteView() {
  await Promise.all([loadRouteTopology(), loadRouteMetrics(), loadRouteAlerts(), loadRouteSparklines()]);
  renderRouteGraph();
  renderFusionAlerts();
  renderSparklines();
}

function renderRouteGraph() {
  const edgesLayer = $('#route-edges-layer');
  const nodesLayer = $('#route-nodes-layer');
  if (!edgesLayer || !nodesLayer) return;

  if (!routeState.topology) {
    // Show placeholder until data loads
    nodesLayer.innerHTML = '';
    edgesLayer.innerHTML = '';
    const placeholder = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    placeholder.setAttribute('x', '350');
    placeholder.setAttribute('y', '230');
    placeholder.setAttribute('text-anchor', 'middle');
    placeholder.setAttribute('fill', 'var(--muted)');
    placeholder.setAttribute('font-size', '14');
    placeholder.setAttribute('font-family', 'var(--font-main)');
    placeholder.textContent = 'Loading topology…';
    nodesLayer.appendChild(placeholder);
    return;
  }

  edgesLayer.innerHTML = '';
  nodesLayer.innerHTML = '';

  const { cameras, edges } = routeState.topology;
  const metricsData = routeState.metrics?.cameras || {};

  // 1. Render Edges (Directed Paths with capacity indicators)
  edges.forEach((edge, idx) => {
    const uNode = cameras[edge.from];
    const vNode = cameras[edge.to];
    if (!uNode || !vNode) return;

    // Center coordinates for bounding boxes
    const nodeW = 190, nodeH = 76;
    const x1 = uNode.position.x + nodeW;
    const y1 = uNode.position.y + nodeH / 2;
    const x2 = vNode.position.x;
    const y2 = vNode.position.y + nodeH / 2;

    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;

    // Downstream predicted inflow & capacity.
    //
    // `predicted_inflow` is null when the forecast could not be made (every
    // upstream source stale, or no sample near t - travel_time). `|| 0` would
    // turn that into a 0/400 ratio and paint the edge GREEN -- rendering
    // "we cannot see this corridor" as "this corridor is clear", which is the
    // most dangerous thing this view could say. Unknown gets its own state.
    const rawInflow = metricsData[edge.to]?.predicted_inflow;
    const inflowKnown = rawInflow !== null && rawInflow !== undefined;
    const targetInflow = inflowKnown ? rawInflow : 0;
    const targetCapacity = vNode.corridor_capacity_pax_min || 400;
    const ratio = (inflowKnown && targetCapacity > 0) ? (targetInflow / targetCapacity) : 0;

    let tierClass = 'edge-ok';
    let markerId = 'arrow-neutral';
    if (!inflowKnown) {
      tierClass = 'edge-unknown';
      markerId = 'arrow-neutral';
    } else if (ratio > 1.0) {
      tierClass = 'edge-bad';
      markerId = 'arrow-bad';
    } else if (ratio >= 0.7) {
      tierClass = 'edge-warn';
      markerId = 'arrow-warn';
    }

    // SVG Curved Path
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const d = `M ${x1} ${y1} C ${x1 + 40} ${y1}, ${x2 - 40} ${y2}, ${x2} ${y2}`;
    path.setAttribute('d', d);
    path.setAttribute('class', `svg-edge-path ${tierClass}`);
    path.setAttribute('marker-end', `url(#${markerId})`);
    path.addEventListener('click', () => showEdgeBreakout(edge, uNode, vNode));

    const tooltipText = `Crowd Flow: ${uNode.name} (${edge.from}) ➔ ${vNode.name} (${edge.to})\nTransit Lead Time: ${Math.round(edge.travel_time_sec)}s\nPredicted Inflow: ${inflowKnown ? Math.round(targetInflow) : 'Unknown'} pax/min\nDownstream Capacity: ${Math.round(targetCapacity)} pax/min (${Math.round(ratio * 100)}% load)`;
    const titleEl1 = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    titleEl1.textContent = tooltipText;
    path.appendChild(titleEl1);
    edgesLayer.appendChild(path);

    // Midpoint Label Pill
    const gPill = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gPill.setAttribute('class', 'svg-edge-pill');
    gPill.setAttribute('transform', `translate(${midX}, ${midY})`);
    gPill.addEventListener('click', () => showEdgeBreakout(edge, uNode, vNode));

    const inflowDisplay = inflowKnown ? `${Math.round(targetInflow)}` : 'n/a';
    const labelText = `➔ ${Math.round(edge.travel_time_sec)}s · ${inflowDisplay}/${Math.round(targetCapacity)} pax/min`;
    const pillW = labelText.length * 7.5 + 20;

    const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bgRect.setAttribute('x', -pillW / 2);
    bgRect.setAttribute('y', -12);
    bgRect.setAttribute('width', pillW);
    bgRect.setAttribute('height', 24);
    bgRect.setAttribute('rx', 6);
    bgRect.setAttribute('ry', 6);
    bgRect.setAttribute('class', `svg-edge-bg ${tierClass}`);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('y', 4);
    text.setAttribute('class', 'svg-edge-text');
    text.textContent = labelText;

    const titleEl2 = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    titleEl2.textContent = tooltipText;
    gPill.appendChild(titleEl2);

    gPill.appendChild(bgRect);
    gPill.appendChild(text);
    edgesLayer.appendChild(gPill);
  });

  // 2. Render Nodes (Cameras)
  Object.values(cameras).forEach((cam) => {
    const camMetrics = metricsData[cam.id]?.snapshot;
    const isStale = metricsData[cam.id]?.is_stale ?? true;

    const density = camMetrics ? camMetrics.density : 0.0;
    const flow = camMetrics ? camMetrics.flow_rate_pax_min : 0.0;

    let tierClass = 'tier-ok';
    if (density >= 2.5) tierClass = 'tier-bad';
    else if (density >= 1.5) tierClass = 'tier-warn';

    const gNode = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gNode.setAttribute('class', 'svg-cam-node');
    gNode.setAttribute('transform', `translate(${cam.position.x}, ${cam.position.y})`);

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', 190);
    rect.setAttribute('height', 76);
    rect.setAttribute('rx', 10);
    rect.setAttribute('ry', 10);
    rect.setAttribute('class', `svg-cam-card ${tierClass}`);

    const nameText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    nameText.setAttribute('x', 14);
    nameText.setAttribute('y', 24);
    nameText.setAttribute('class', 'svg-node-title');
    nameText.textContent = cam.name;

    const idText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    idText.setAttribute('x', 14);
    idText.setAttribute('y', 40);
    idText.setAttribute('class', 'svg-node-id');
    idText.textContent = `${cam.id} · Cap: ${cam.corridor_capacity_pax_min} pax/m`;

    const statsText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    statsText.setAttribute('x', 14);
    statsText.setAttribute('y', 60);
    statsText.setAttribute('class', 'svg-node-stats');
    statsText.textContent = isStale
      ? '⚠️ Telemetry Offline'
      : `ρ: ${density.toFixed(2)} p/m²  |  Q: ${Math.round(flow)} p/m`;

    gNode.appendChild(rect);
    gNode.appendChild(nameText);
    gNode.appendChild(idText);
    gNode.appendChild(statsText);

    gNode.addEventListener('click', () => {
      // Find matching saved result card or switch to pipeline tab
      const cards = $$('.card');
      const targetCard = cards.find(c => c.textContent.toLowerCase().includes(cam.id.toLowerCase()));
      if (targetCard) {
        switchView('pipeline');
        targetCard.scrollIntoView({ behavior: 'smooth' });
        targetCard.click();
      } else {
        alert(`Camera: ${cam.name} (${cam.id})\nCapacity: ${cam.corridor_capacity_pax_min} pax/min\nDensity: ${density.toFixed(2)} pax/m²\nFlow Rate: ${Math.round(flow)} pax/min`);
      }
    });

    nodesLayer.appendChild(gNode);
  });

  // ── Resize SVG to fit all nodes ──────────────────────────────────────────
  // Compute the bounding box across every camera position so the SVG grows
  // to accommodate any number of cameras and the wrapper can scroll to show them.
  const NODE_W = 190, NODE_H = 76, PAD = 60;
  let maxX = 0, maxY = 0;
  Object.values(cameras).forEach(cam => {
    maxX = Math.max(maxX, (cam.position?.x || 0) + NODE_W);
    maxY = Math.max(maxY, (cam.position?.y || 0) + NODE_H);
  });
  const svgW = Math.max(maxX + PAD, 700);
  const svgH = Math.max(maxY + PAD, 300);

  const svg = $('#route-svg-canvas');
  if (svg) {
    svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
  }
}

function showEdgeBreakout(edge, uNode, vNode) {
  const panel = $('#edge-breakout-panel');
  const title = $('#breakout-title');
  const body = $('#breakout-body');
  if (!panel || !title || !body) return;

  const targetMetrics = routeState.metrics?.cameras?.[edge.to];
  const uMetrics = routeState.metrics?.cameras?.[edge.from];

  title.textContent = `${uNode.name} ➔ ${vNode.name}`;
  const uFlow = uMetrics?.snapshot?.flow_rate_pax_min || 0;
  // null = no forecast. Rendered as "unknown", never as zero (see above).
  const rawTargetInflow = targetMetrics?.predicted_inflow;
  const inflowKnown = rawTargetInflow !== null && rawTargetInflow !== undefined;
  const targetInflow = inflowKnown ? rawTargetInflow : 0;
  const inflowText = inflowKnown ? `${Math.round(targetInflow)} pax/min`
                                 : 'unknown - forecast unavailable';
  // Uncalibrated flow is not in pax/min at all; saying so beats printing a
  // number whose units are a scaled pixel count.
  const uCal = uMetrics?.snapshot?.flow_is_calibrated;
  const uFlowText = uCal ? `${Math.round(uFlow)} pax/min`
                         : `${Math.round(uFlow)} (uncalibrated - relative only)`;
  const fStatus = targetMetrics?.forecast_status;
  const incompleteNote = (fStatus && fStatus.complete === false && (fStatus.missing || []).length)
    ? `<div class="breakout-row"><span class="name">⚠ Forecast incomplete</span>`
      + `<span class="val">missing: ${esc((fStatus.missing || []).join(', '))}</span></div>`
    : '';
  const targetCap = vNode.corridor_capacity_pax_min || 400;

  body.innerHTML = `
    <div class="breakout-row">
      <span class="name">Crowd Movement</span>
      <span class="val" style="color: #38bdf8; font-weight: 700;">${esc(uNode.name)} ➔ ${esc(vNode.name)}</span>
    </div>
    <div class="breakout-row">
      <span class="name">Transit Lead Time</span>
      <span class="val">${Math.round(edge.travel_time_sec)} seconds</span>
    </div>
    <div class="breakout-row">
      <span class="name">${esc(uNode.name)} (${edge.from}) Flow</span>
      <span class="val">${uFlowText}</span>
    </div>
    <div class="breakout-row">
      <span class="name">Combined Corridor Inflow</span>
      <span class="val">${inflowText}</span>
    </div>
    ${incompleteNote}
    <div class="breakout-row">
      <span class="name">${esc(vNode.name)} Corridor Capacity</span>
      <span class="val">${Math.round(targetCap)} pax/min</span>
    </div>
    <div class="breakout-total" style="color: ${inflowKnown ? (targetInflow > targetCap ? 'var(--bad)' : 'var(--ok)') : 'var(--text-dim)'}">
      <span>Capacity Ratio:</span>
      <span>${inflowKnown ? `${Math.round((targetInflow / targetCap) * 100)}%` : 'n/a (forecast unavailable)'}</span>
    </div>
  `;
  panel.classList.remove('hidden');
}

function renderFusionAlerts() {
  const tbody = $('#fusion-alerts-body');
  const badge = $('#fusion-alert-count-badge');
  if (!tbody) return;

  const alerts = routeState.alerts || [];

  if (alerts.length === 0) {
    const camMetrics = routeState.metrics?.cameras || {};
    const cams = Object.values(camMetrics);

    let emptyMessage = 'No active cross-camera alerts. Corridor flow is currently within safe limits.';
    let emptyIcon = '✓';
    let badgeText = '0 active';

    // 1. Check if all telemetry is offline or uninitialized
    const allStaleOrOffline = cams.length === 0 || cams.every(c => !c.snapshot || c.is_stale);

    if (allStaleOrOffline) {
      emptyIcon = '⚠️';
      emptyMessage = 'Telemetry offline — no active camera streams received. Predictive cross-camera alerts cannot be evaluated.';
      badgeText = '0 active (offline)';
    } else {
      // 2. Check if any downstream bottleneck alert is blocked due to uncalibrated camera flow
      const uncalibratedCams = cams.filter(c =>
        c.forecast_status?.blocked === 'uncalibrated_flow' ||
        (c.snapshot && c.snapshot.flow_is_calibrated === false)
      );
      if (uncalibratedCams.length > 0) {
        emptyIcon = '⚠️';
        const uncalNames = uncalibratedCams.map(c => c.name || 'camera').join(', ');
        emptyMessage = `Predictive bottleneck alerts disabled — ${uncalNames} has no ground-plane calibration (flow is not pax/min).`;
        badgeText = '0 active (uncalibrated)';
      } else {
        // 3. Check if any corridor forecast is incomplete due to missing/stale upstream sources
        const incompleteCams = cams.filter(c => c.forecast_status && c.forecast_status.complete === false);
        if (incompleteCams.length > 0) {
          emptyIcon = '⚠️';
          emptyMessage = 'Incomplete upstream telemetry — corridor prediction is partially blind.';
          badgeText = '0 active (incomplete)';
        }
      }
    }

    if (badge) badge.textContent = badgeText;
    tbody.innerHTML = `<tr><td colspan="6" class="empty">${emptyIcon} ${esc(emptyMessage)}</td></tr>`;
    return;
  }

  if (badge) badge.textContent = `${alerts.length} active`;

  tbody.innerHTML = alerts.map(a => {
    const timeStr = new Date(a.timestamp_epoch_ms).toISOString().substr(11, 8);
    const badgeClass = a.level === 'BOTTLENECK_PREDICTED'
      ? 'bottleneck'
      : a.level === 'ACCUMULATION_RISING'
      ? 'accumulation'
      : 'crush';
    return `
      <tr>
        <td style="font-family: var(--font-code);">${esc(timeStr)}</td>
        <td><span class="level-badge ${badgeClass}">${esc(a.level)}</span></td>
        <td><strong>${esc(a.target_name)}</strong> <small class="subtle">(${esc(a.camera_id)})</small></td>
        <td>${esc(a.source_names?.join(' + ') || a.source_cameras?.join(' + '))}</td>
        <td><span class="lead-time-pill">${Math.round(a.lead_time_sec)}s</span></td>
        <td style="font-size: 12px; color: var(--text-dim);">${esc(a.detail)}</td>
      </tr>
    `;
  }).join('');
}

function renderSparklines() {
  const container = $('#camera-sparklines-grid');
  if (!container || !routeState.topology) return;

  const cameras = routeState.topology.cameras || {};
  const sparkData = routeState.sparklines || {};

  container.innerHTML = Object.values(cameras).map(cam => {
    const history = sparkData[cam.id] || [];
    const hasHistory = history.length > 0;
    const latest = hasHistory ? history[history.length - 1] : null;

    const densityVal = latest?.density;
    const flowVal = latest?.flow_rate_pax_min;

    const densityStr = (densityVal !== undefined && densityVal !== null)
      ? densityVal.toFixed(2)
      : 'n/a';
    const flowStr = (flowVal !== undefined && flowVal !== null)
      ? Math.round(flowVal)
      : 'n/a';

    // The labels below are rho (density) and Q (flow), which imply persons/m2
    // and pax/min. On an uncalibrated camera they are neither: density is an
    // image-plane proxy and flow is a scaled detection count. Printing the
    // symbols without saying so invites the numbers being read as physical
    // quantities and compared against published crowd-safety thresholds.
    const calibrated = latest ? latest.flow_is_calibrated === true : null;
    const unitTag = (latest === null) ? ''
      : (calibrated ? '<span class="unit-ok" title="Calibrated: persons/m² and pax/min">✓</span>'
                    : '<span class="unit-warn" title="UNCALIBRATED: relative units only, not persons/m² or pax/min">~</span>');

    let pointsSvg = '';
    if (history.length > 1) {
      const maxFlow = Math.max(...history.map(h => h.flow_rate_pax_min || 0), 100);
      const pts = history.map((h, i) => {
        const x = (i / (history.length - 1)) * 260;
        const val = h.flow_rate_pax_min || 0;
        const y = 50 - (val / maxFlow) * 45;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');
      pointsSvg = `<polyline fill="none" stroke="var(--accent-light)" stroke-width="2" points="${pts}" />`;
    } else {
      pointsSvg = `<text x="130" y="32" text-anchor="middle" fill="var(--muted)" font-size="11" font-style="italic">No telemetry recorded yet</text>`;
    }

    return `
      <div class="sparkline-card">
        <div class="sparkline-head">
          <span class="sparkline-cam-name">${esc(cam.name)} <small class="subtle">(${esc(cam.id)})</small></span>
          <span class="sparkline-metrics-summary">ρ: ${densityStr} | Q: ${flowStr} ${unitTag}</span>
        </div>
        <svg class="sparkline-svg" viewBox="0 0 260 55">
          <line x1="0" y1="50" x2="260" y2="50" stroke="var(--border)" stroke-width="1" />
          ${pointsSvg}
        </svg>
      </div>
    `;
  }).join('');
}

function connectFusionWS() {
  if (routeState.ws) {
    try { routeState.ws.close(); } catch { }
  }

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/fusion`;
  const statusEl = $('#route-ws-status');

  try {
    const ws = new WebSocket(wsUrl);
    routeState.ws = ws;

    ws.onopen = () => {
      if (statusEl) {
        statusEl.textContent = '🟢 Live Stream';
        statusEl.className = 'status-pill connected';
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'init' || msg.event === 'fusion_tick') {
          if (msg.topology) routeState.topology = msg.topology;
          if (msg.data?.forecast_status || msg.forecast_status) {
            if (!routeState.metrics) routeState.metrics = { cameras: {} };
            const statuses = msg.data?.forecast_status || msg.forecast_status;
            for (const [cid, st] of Object.entries(statuses)) {
              if (!routeState.metrics.cameras[cid]) routeState.metrics.cameras[cid] = {};
              routeState.metrics.cameras[cid].forecast_status = st;
            }
          }
          if (msg.data?.alerts || msg.alerts) {
            routeState.alerts = msg.data?.alerts || msg.alerts;
          }
          if (msg.data?.predicted_inflows || msg.predicted_inflows) {
            if (!routeState.metrics) routeState.metrics = { cameras: {} };
            const inflows = msg.data?.predicted_inflows || msg.predicted_inflows;
            for (const [cid, val] of Object.entries(inflows)) {
              if (!routeState.metrics.cameras[cid]) routeState.metrics.cameras[cid] = {};
              routeState.metrics.cameras[cid].predicted_inflow = val;
            }
            renderRouteGraph();
          }
          renderFusionAlerts();
        }
      } catch (e) { }
    };

    ws.onclose = () => {
      if (statusEl) {
        statusEl.textContent = '🔴 Disconnected';
        statusEl.className = 'status-pill disconnected';
      }
      clearTimeout(routeState.wsTimer);
      routeState.wsTimer = setTimeout(connectFusionWS, 3000);
    };
  } catch (err) {
    clearTimeout(routeState.wsTimer);
    routeState.wsTimer = setTimeout(connectFusionWS, 3000);
  }
}

/* ==========================================================================
   Multi-Camera Route Sessions Controller
   ========================================================================== */

let sessionPollTimer = null;

const INITIAL_CAMERA_CORRIDORS = [
  { cid: 'CCTV1', name: 'Cam 1' },
  { cid: 'CCTV2', name: 'Cam 2' },
];

async function loadSessionSubmitPanel(customSlots = null) {
  const container = $('#session-camera-slots');
  if (!container) return;

  try {
    const videosData = await api('/api/videos');
    const videos = videosData?.videos || [];

    // Default to 2 Camera Corridors (Cam 1 & Cam 2) unless custom slots passed from Route Builder
    const slots = (customSlots && customSlots.length > 0) ? customSlots : INITIAL_CAMERA_CORRIDORS;

    // Empty initial selection: NO example video pre-selected
    const optHtml = `
      <option value="" selected disabled>— Select Video File or Upload —</option>
      <option value="">— Unassigned (None) —</option>
      ${videos.map(v => `<option value="${esc(v.name)}">${esc(v.name)} (${v.size_mb} MB)</option>`).join('')}
    `;

    container.innerHTML = slots.map(slot => {
      const cid = slot.cid || slot.camera_id || slot.id;
      const cname = slot.name || slot.camera_name || cid;

      return `
        <div class="slot-row" data-cam-id="${esc(cid)}" data-cam-name="${esc(cname)}">
          <div class="slot-cam-badge">
            <span class="cam-code-tag">${esc(cid)}</span>
            <span class="cam-name-tag">${esc(cname)}</span>
          </div>
          <div class="slot-input-wrap">
            <select class="slot-select" data-slot-cam="${esc(cid)}">
              ${optHtml}
            </select>
            <label class="slot-file-btn" title="Upload custom video for this camera">
              📁 Upload Video
              <input type="file" accept="video/*" class="slot-file-input" data-slot-cam="${esc(cid)}" />
            </label>
          </div>
          <label class="slot-relate-toggle">
            <input type="checkbox" class="slot-relate-cb" checked data-slot-cam="${esc(cid)}" />
            Relate in Session
          </label>
        </div>
      `;
    }).join('');

    // Wire upload inputs for each slot
    container.querySelectorAll('.slot-file-input').forEach(input => {
      input.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const cid = input.dataset.slotCam;
        const select = container.querySelector(`.slot-select[data-slot-cam="${cid}"]`);
        const statusEl = $('#session-submit-status');
        if (statusEl) statusEl.textContent = `Uploading ${file.name} for ${cid}…`;

        const fd = new FormData();
        fd.append('file', file);
        try {
          const res = await fetch('/api/videos/upload', { method: 'POST', body: fd });
          if (!res.ok) throw new Error(await res.text());
          const json = await res.json();
          if (statusEl) statusEl.textContent = `Uploaded ${json.name} successfully.`;
          // Add to select and select it
          const opt = document.createElement('option');
          opt.value = json.name;
          opt.textContent = `${json.name} (${json.size_mb} MB)`;
          opt.selected = true;
          select.appendChild(opt);
          select.value = json.name;
        } catch (err) {
          if (statusEl) statusEl.textContent = `Upload failed: ${err.message}`;
        }
      });
    });

    // Wire relate checkboxes
    container.querySelectorAll('.slot-relate-cb').forEach(cb => {
      cb.addEventListener('change', () => {
        const row = cb.closest('.slot-row');
        if (row) row.classList.toggle('disabled', !cb.checked);
      });
    });

  } catch (err) {
    container.innerHTML = `<div class="err">Failed loading camera corridors: ${esc(err.message)}</div>`;
  }
}

function resetSessionSubmitPanelToInitial() {
  const nameInput = $('#session-name-input');
  if (nameInput) nameInput.value = '';

  const modelSelect = $('#session-model-select');
  if (modelSelect) modelSelect.value = 'crowd_motion_monitor';

  const runBtn = $('#btn-run-route-session');
  if (runBtn) runBtn.disabled = false;

  loadSessionSubmitPanel(INITIAL_CAMERA_CORRIDORS);
}

async function submitRouteSession() {
  const nameInput = $('#session-name-input');
  const modelSelect = $('#session-model-select');
  const statusEl = $('#session-submit-status');
  const runBtn = $('#btn-run-route-session');

  let sessionName = nameInput?.value?.trim();
  if (!sessionName) {
    const ts = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
    sessionName = `Route_Session_${ts}`;
  }

  const modelKey = modelSelect?.value || 'crowd_motion_monitor';
  const slotRows = $$('#session-camera-slots .slot-row');
  const slots = [];

  slotRows.forEach(row => {
    const cid = row.dataset.camId;
    const cname = row.dataset.camName;
    const cb = row.querySelector('.slot-relate-cb');
    const sel = row.querySelector('.slot-select');
    const isChecked = cb ? cb.checked : true;
    const videoSource = sel ? sel.value : '';

    if (isChecked && videoSource) {
      slots.push({
        camera_id: cid,
        camera_name: cname,
        video_source: videoSource,
        include_in_session: true,
      });
    }
  });

  if (slots.length === 0) {
    if (statusEl) statusEl.innerHTML = '<span style="color: var(--bad);">⚠️ Please select a video file for at least one camera corridor slot.</span>';
    return;
  }

  if (runBtn) runBtn.disabled = true;
  if (statusEl) statusEl.innerHTML = '<span style="color: var(--accent-cyan);">🚀 Initializing Route Session…</span>';

  try {
    const payload = {
      session_name: sessionName,
      slots,
      models: [modelKey],
      sample_every_n_frames: 5,
      export_video: true,
    };

    const res = await postJSON('/api/sessions', payload);

    if (statusEl) statusEl.innerHTML = `<span style="color: var(--ok);">▶ Running Route Session: <strong>${esc(res.session_name)}</strong></span>`;
    loadRouteSessionsList();
    pollRouteSession(res.session_name);
  } catch (err) {
    if (statusEl) statusEl.innerHTML = `<span style="color: var(--bad);">❌ Failed: ${esc(err.message)}</span>`;
    if (runBtn) runBtn.disabled = false;
  }
}

function pollRouteSession(sessionName) {
  clearTimeout(sessionPollTimer);
  const statusEl = $('#session-submit-status');
  const runBtn = $('#btn-run-route-session');

  sessionPollTimer = setTimeout(async () => {
    try {
      const sess = await api(`/api/sessions/${sessionName}`);
      if (!sess) return;

      const cams = sess.cameras || {};
      const camStatuses = Object.values(cams).map(c => `${c.camera_name || c.camera_id}: ${c.status} (${Math.round(c.progress * 100)}%)`).join(' · ');

      if (statusEl) {
        statusEl.innerHTML = `<span><strong>${esc(sess.status.toUpperCase())}</strong> · ${esc(camStatuses)}</span>`;
      }

      loadRouteSessionsList();

      if (sess.status === 'done' || sess.status === 'failed' || sess.status === 'partial') {
        if (runBtn) runBtn.disabled = false;
        if (sess.status === 'done' || sess.status === 'partial') {
          if (statusEl) {
            statusEl.innerHTML = `
              <span style="color: var(--ok);">✓ Session "${esc(sessionName)}" complete!</span>
              <a href="/api/files/session/${esc(sessionName)}/session_report.html" target="_blank" class="btn-fused-report" style="margin-left: 10px;">
                📄 View Fused Report
              </a>
            `;
          }
          refreshRouteView();
          // Automatically return panel to initial clean 2-camera empty state
          resetSessionSubmitPanelToInitial();
        }
        return;
      }

      pollRouteSession(sessionName);
    } catch (e) {
      pollRouteSession(sessionName);
    }
  }, 1500);
}

async function loadRouteSessionsList() {
  const container = $('#route-sessions-list');
  if (!container) return;

  try {
    const data = await api('/api/sessions');
    const sessions = data?.sessions || [];

    if (sessions.length === 0) {
      container.innerHTML = '<div class="empty">No route sessions created yet. Use the form above to launch a multi-camera session.</div>';
      return;
    }

    container.innerHTML = sessions.map(s => {
      const cams = s.cameras || {};
      const sum = s.summary;
      const num = (v, nd = 2, u = '') => v == null ? '—' : `${Number(v).toFixed(nd)}${u ? ' ' + u : ''}`;

      // 1. Model display identification
      const modelKey = (s.models && s.models.length > 0) ? s.models[0] : (s.model_key || 'crowd_motion_monitor');
      let modelDisplay = null;
      if (state.models && state.models.length) {
        const found = state.models.find(m => m.key === modelKey);
        if (found && found.label) modelDisplay = found.label;
      }
      if (!modelDisplay) {
        const modelLabels = {
          'crowd_motion_monitor': 'Crowd Motion Monitor (CMM)',
          'dm_count_crowd': 'DM-Count (Crowd Density)',
          'csrnet': 'CSRNet Crowd Density',
          'bayesian_crowd': 'Bayesian Crowd Counting',
          'apgcc': 'APGCC Head Count',
          'yolov8x': 'YOLOv8x Crowd Detector',
          'rtdetrv2': 'RT-DETRv2 Detector',
          'anpr': 'ANPR License Plate Reader',
          'movenet': 'MoveNet MultiPose',
          'blazepose': 'MediaPipe BlazePose',
          'optical_flow': 'Lucas-Kanade Optical Flow',
          'cctv_umbrella': 'Umbrella Detection',
        };
        modelDisplay = modelLabels[modelKey] || modelKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      }

      // 2. Camera summary counts
      const camCount = Object.keys(cams).length;
      const doneCount = Object.values(cams).filter(c => c.status === 'done').length;
      const totalDets = sum?.total_detections || Object.values(cams).reduce((acc, c) => acc + (c.detections || 0), 0);

      // 3. Compact 4-badge KPI strip (Intelligent: Flow metrics if available, else Detections/Tracks metrics)
      let kpiHtml = '';
      const hasFlowMetrics = sum && (sum.avg_density != null || sum.avg_speed != null || sum.max_crush_risk_pct != null);
      if (hasFlowMetrics) {
        const crushColor = (sum.max_crush_risk_pct || 0) >= 50 ? '#ef4444' : (sum.max_crush_risk_pct || 0) >= 20 ? '#f59e0b' : '#22c55e';
        kpiHtml = `
          <div class="session-kpi-strip">
            <div class="session-kpi-badge">
              <span class="skb-label">Density</span>
              <span class="skb-value" style="color:#f97316">${num(sum.avg_density, 1)} <small>p/m²</small></span>
            </div>
            <div class="session-kpi-badge">
              <span class="skb-label">Velocity</span>
              <span class="skb-value" style="color:#60a5fa">${num(sum.avg_speed, 2)} <small>px/f</small></span>
            </div>
            <div class="session-kpi-badge">
              <span class="skb-label">Crush Risk</span>
              <span class="skb-value" style="color:${crushColor}">${num(sum.max_crush_risk_pct, 1, '%')}</span>
            </div>
            <div class="session-kpi-badge">
              <span class="skb-label">Entropy</span>
              <span class="skb-value" style="color:#a78bfa">${num(sum.max_directional_entropy, 2)}</span>
            </div>
          </div>
        `;
      } else {
        // Counting / Detector model (e.g. DM-Count, APGCC, YOLOv8x)
        const totalTracks = sum?.total_tracks != null ? sum.total_tracks : '—';
        kpiHtml = `
          <div class="session-kpi-strip">
            <div class="session-kpi-badge">
              <span class="skb-label">Detections</span>
              <span class="skb-value" style="color:#38bdf8">${totalDets.toLocaleString()}</span>
            </div>
            <div class="session-kpi-badge">
              <span class="skb-label">Tracked</span>
              <span class="skb-value" style="color:#a78bfa">${typeof totalTracks === 'number' ? totalTracks.toLocaleString() : totalTracks}</span>
            </div>
            <div class="session-kpi-badge">
              <span class="skb-label">Coverage</span>
              <span class="skb-value" style="color:#22c55e">${doneCount}/${camCount} <small>cams</small></span>
            </div>
            <div class="session-kpi-badge">
              <span class="skb-label">Status</span>
              <span class="skb-value" style="color:#34d399">${esc(s.status.toUpperCase())}</span>
            </div>
          </div>
        `;
      }

      // Compact Corridor summary line
      const narratives = (sum && sum.transit_narratives) || [];
      let corridorLine = '';
      if (narratives.length > 0) {
        corridorLine = `
          <div class="session-corridor-compact-line">
            <span>🌊 <strong>${narratives.length} Corridor Segments</strong> · <strong>${totalDets.toLocaleString()}</strong> Detections</span>
            <span style="color: var(--accent); font-weight: 600; font-size: 11.5px;">Click card to inspect &rarr;</span>
          </div>
        `;
      } else if (camCount > 0) {
        corridorLine = `
          <div class="session-corridor-compact-line">
            <span>📹 <strong>${camCount} Cameras Active</strong> · <strong>${totalDets.toLocaleString()}</strong> Detections</span>
            <span style="color: var(--accent); font-weight: 600; font-size: 11.5px;">Click card to inspect &rarr;</span>
          </div>
        `;
      }

      const hasReport = s.report_html || s.status === 'done';
      const createdStr = s.created_at ? s.created_at.slice(0, 19).replace('T', ' ') : '';

      return `
        <div class="session-card u-clickable" data-session="${esc(s.session_name)}" title="Click anywhere on card to open output pages &amp; multi-camera videos">
          <div class="session-card-head">
            <div>
              <div class="session-card-title">${esc(s.session_name)}</div>
              <div class="session-card-date">${esc(createdStr)} UTC</div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="session-badge-status ${esc(s.status)}">${esc(s.status)}</span>
              <button class="link-btn danger-link btn-del-session" data-session="${esc(s.session_name)}" title="Delete session" style="padding: 2px 7px; font-size: 11px;">
                ✕
              </button>
            </div>
          </div>

          <!-- Model & Cameras Meta Strip -->
          <div class="session-meta-strip">
            <span class="session-model-pill" title="AI Model executed on this route session">
              <span class="smp-label">AI MODEL:</span>
              <strong>🤖 ${esc(modelDisplay)}</strong>
            </span>
            <span class="session-cams-summary-pill" title="${camCount} camera streams analyzed">
              📹 <strong>${camCount} Cams</strong> <small>(${doneCount} Done)</small>
            </span>
            <span class="session-dets-pill" title="Total detections across all cameras">
              📊 <strong>${totalDets.toLocaleString()}</strong> dets
            </span>
          </div>

          ${kpiHtml}
          ${corridorLine}

          <div class="session-card-foot">
            <button class="btn-card-detail btn-inspect-session" data-inspect-session="${esc(s.session_name)}">
              🔍 Inspect All Output Pages
            </button>
            ${hasReport ? `
              <a href="/api/files/session/${esc(s.session_name)}/session_report.html" target="_blank" class="btn-fused-report">
                📄 Fused Report
              </a>
            ` : `<span style="font-size: 12px; color: var(--muted);">${esc(s.status)}…</span>`}
          </div>
        </div>
      `;
    }).join('');

    // Wire entire session card click (clicking anywhere on card opens modal)
    container.querySelectorAll('.session-card').forEach(card => {
      card.addEventListener('click', (e) => {
        // Do not open modal if user clicked delete button or fused report link
        if (e.target.closest('.btn-del-session') || e.target.closest('.btn-fused-report')) {
          return;
        }
        openRouteSessionDetailModal(card.dataset.session);
      });
    });

    // Wire delete buttons
    container.querySelectorAll('.btn-del-session').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const sname = btn.dataset.session;
        if (!confirm(`Delete route session "${sname}" and all its outputs?`)) return;
        try {
          await api(`/api/sessions/${sname}`, { method: 'DELETE' });
          loadRouteSessionsList();
        } catch (err) {
          alert(`Failed deleting session: ${err.message}`);
        }
      });
    });

  } catch (err) {
    container.innerHTML = `<div class="err">Failed loading sessions: ${esc(err.message)}</div>`;
  }
}

function switchView(viewName) {
  const pipelineContainer = $('#pipeline-view-container');
  const routeContainer = $('#route-view-container');
  const navBtns = $$('[data-view]');

  navBtns.forEach(btn => {
    const isActive = btn.dataset.view === viewName;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive);
  });

  if (viewName === 'route') {
    pipelineContainer?.classList.add('hidden');
    routeContainer?.classList.remove('hidden');
    refreshRouteView();
    resetSessionSubmitPanelToInitial();
    loadRouteSessionsList();
    if (!routeState.ws) connectFusionWS();
  } else {
    routeContainer?.classList.add('hidden');
    pipelineContainer?.classList.remove('hidden');
  }
}


/* ==========================================================================
   Configure Route / Route Builder Modal Controller
   ========================================================================== */

function openRouteBuilderModal() {
  const modal = $('#route-builder-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  clearRouteBuilderErrors();

  const topo = routeState.topology || {};
  const cameras = topo.cameras || {};
  const edges = topo.edges || [];

  // Populate Default Capacity
  const defCapInput = $('#rb-default-capacity');
  if (defCapInput) {
    const firstCam = Object.values(cameras)[0];
    defCapInput.value = (firstCam && firstCam.corridor_capacity_pax_min) ? firstCam.corridor_capacity_pax_min : 400;
  }

  // Populate Cameras
  const camListEl = $('#rb-cameras-list');
  if (camListEl) {
    camListEl.innerHTML = '';
    const camEntries = Object.entries(cameras);
    if (camEntries.length >= 2) {
      camEntries.forEach(([cid, c]) => {
        addRouteBuilderCameraRow(cid, c.name || cid, c.corridor_capacity_pax_min, c.holding_capacity_pax, c.clock_offset_sec);
      });
    } else {
      addRouteBuilderCameraRow('CCTV1', 'Gate A', 400);
      addRouteBuilderCameraRow('CCTV2', 'Merge Point', 400);
    }
  }

  // Populate Corridors / Edges
  const edgesListEl = $('#rb-edges-list');
  if (edgesListEl) {
    edgesListEl.innerHTML = '';
    if (edges.length > 0) {
      edges.forEach(e => {
        addRouteBuilderEdgeRow(e.from, e.to, e.travel_time_sec);
      });
    } else {
      addRouteBuilderEdgeRow('CCTV1', 'CCTV2', 25);
    }
  }

  syncAllEdgeDropdowns();
}

function closeRouteBuilderModal() {
  const modal = $('#route-builder-modal');
  if (modal) modal.classList.add('hidden');
  clearRouteBuilderErrors();
}

function clearRouteBuilderErrors() {
  const banner = $('#rb-error-banner');
  if (banner) {
    banner.classList.add('hidden');
    banner.textContent = '';
  }
  $$('.rb-inline-error').forEach(el => el.remove());
  $$('.rb-input-error').forEach(el => el.classList.remove('rb-input-error'));
}

function showRouteBuilderError(msg, targetInput = null) {
  const banner = $('#rb-error-banner');
  if (banner) {
    banner.textContent = msg;
    banner.classList.remove('hidden');
  }
  if (targetInput) {
    targetInput.classList.add('rb-input-error');
    targetInput.focus();
    const row = targetInput.closest('.rb-row');
    if (row && !row.querySelector('.rb-inline-error')) {
      const errSpan = document.createElement('div');
      errSpan.className = 'rb-inline-error';
      errSpan.style.cssText = 'color: var(--bad); font-size: 12px; font-weight: 600; width: 100%; margin-top: 4px;';
      errSpan.textContent = msg;
      row.appendChild(errSpan);
    }
  }
}

function getDefinedCameraIds() {
  return $$('#rb-cameras-list .rb-camera-row').map(row => {
    const idInput = row.querySelector('.rb-cam-id');
    return idInput ? idInput.value.trim() : '';
  }).filter(Boolean);
}

function syncAllEdgeDropdowns() {
  const camIds = getDefinedCameraIds();
  $$('#rb-edges-list .rb-edge-row').forEach(row => {
    const fromSel = row.querySelector('.rb-edge-from');
    const toSel = row.querySelector('.rb-edge-to');
    const currFrom = fromSel ? fromSel.value : '';
    const currTo = toSel ? toSel.value : '';

    const optsHtml = camIds.length > 0
      ? camIds.map(id => `<option value="${esc(id)}">${esc(id)}</option>`).join('')
      : '<option value="">No cameras</option>';

    if (fromSel) {
      fromSel.innerHTML = optsHtml;
      if (camIds.includes(currFrom)) fromSel.value = currFrom;
      else if (camIds[0]) fromSel.value = camIds[0];
    }
    if (toSel) {
      toSel.innerHTML = optsHtml;
      if (camIds.includes(currTo)) toSel.value = currTo;
      else if (camIds[1]) toSel.value = camIds[1];
      else if (camIds[0]) toSel.value = camIds[0];
    }
  });
}

function addRouteBuilderCameraRow(cid = '', name = '', cap = '', holdingCap = '', clockOffset = '') {
  const container = $('#rb-cameras-list');
  if (!container) return;

  const row = document.createElement('div');
  row.className = 'rb-row rb-camera-row';
  row.style.cssText = 'display: flex; align-items: center; gap: 8px; flex-wrap: wrap; background: var(--panel-2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 12px;';

  row.innerHTML = `
    <div style="display: flex; flex-direction: column; width: 130px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;">Camera ID</label>
      <input type="text" class="rb-cam-id" value="${esc(cid)}" placeholder="e.g. CCTV4" style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-family: var(--font-code); font-size: 13px; font-weight: 700;" />
    </div>
    <div style="display: flex; flex-direction: column; flex: 1; min-width: 160px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;">Display Name</label>
      <input type="text" class="rb-cam-name" value="${esc(name)}" placeholder="e.g. Exit Pathway" style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-size: 13px;" />
    </div>
    <div style="display: flex; flex-direction: column; width: 140px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;">Capacity (optional)</label>
      <input type="number" class="rb-cam-cap" value="${cap || ''}" placeholder="Default" min="50" step="10" style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-family: var(--font-code); font-size: 13px; text-align: right;" />
    </div>
    <div style="display: flex; flex-direction: column; width: 140px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;">Holding capacity (pax)</label>
      <input type="number" class="rb-cam-holding-cap" value="${holdingCap || ''}" placeholder="Optional" min="1" step="10" style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-family: var(--font-code); font-size: 13px; text-align: right;" />
    </div>
    <div style="display: flex; flex-direction: column; width: 140px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;" title="Seconds this clip started AFTER the session reference time. Cross-camera fusion assumes every clip covers the same wall-clock window; enter the measured skew if they do not.">Clip offset (s)</label>
      <input type="number" class="rb-cam-clock-offset" value="${clockOffset === 0 || clockOffset ? clockOffset : ''}" placeholder="0" step="0.1" title="Seconds this clip started AFTER the session reference time. Negative if it started earlier. Leave blank if all clips start together." style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-family: var(--font-code); font-size: 13px; text-align: right;" />
    </div>
    <div style="display: flex; align-items: flex-end; padding-top: 14px;">
      <button class="link-btn danger-link rb-remove-cam-btn" type="button" style="padding: 6px 10px; font-size: 14px; font-weight: 700;" title="Remove camera">✕</button>
    </div>
  `;

  container.appendChild(row);

  const idInput = row.querySelector('.rb-cam-id');
  idInput?.addEventListener('input', () => {
    syncAllEdgeDropdowns();
  });

  row.querySelector('.rb-remove-cam-btn')?.addEventListener('click', () => {
    const totalRows = $$('#rb-cameras-list .rb-camera-row').length;
    if (totalRows <= 2) {
      showRouteBuilderError('A valid route must contain at least 2 cameras.');
      return;
    }
    row.remove();
    syncAllEdgeDropdowns();
  });

  syncAllEdgeDropdowns();
}

function addRouteBuilderEdgeRow(from = '', to = '', travelSec = 20) {
  const container = $('#rb-edges-list');
  if (!container) return;

  const camIds = getDefinedCameraIds();
  const optsHtml = camIds.length > 0
    ? camIds.map(id => `<option value="${esc(id)}">${esc(id)}</option>`).join('')
    : '<option value="">No cameras</option>';

  const row = document.createElement('div');
  row.className = 'rb-row rb-edge-row';
  row.style.cssText = 'display: flex; align-items: center; gap: 8px; flex-wrap: wrap; background: var(--panel-2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 12px;';

  row.innerHTML = `
    <div style="display: flex; flex-direction: column; flex: 1; min-width: 120px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;">From (Upstream)</label>
      <select class="rb-edge-from" style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-family: var(--font-code); font-size: 13px;">
        ${optsHtml}
      </select>
    </div>
    <div style="color: var(--muted); font-size: 18px; padding-top: 14px;">➔</div>
    <div style="display: flex; flex-direction: column; flex: 1; min-width: 120px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;">To (Downstream)</label>
      <select class="rb-edge-to" style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-family: var(--font-code); font-size: 13px;">
        ${optsHtml}
      </select>
    </div>
    <div style="display: flex; flex-direction: column; width: 140px;">
      <label style="font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700;">Travel Time (sec)</label>
      <input type="number" class="rb-edge-time" value="${travelSec || 20}" min="1" step="1" style="background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; color: var(--text); font-family: var(--font-code); font-size: 13px; text-align: right;" />
    </div>
    <div style="display: flex; align-items: flex-end; padding-top: 14px;">
      <button class="link-btn danger-link rb-remove-edge-btn" type="button" style="padding: 6px 10px; font-size: 14px; font-weight: 700;" title="Remove corridor">✕</button>
    </div>
  `;

  container.appendChild(row);

  const fromSel = row.querySelector('.rb-edge-from');
  const toSel = row.querySelector('.rb-edge-to');
  if (fromSel && from && camIds.includes(from)) fromSel.value = from;
  if (toSel && to && camIds.includes(to)) toSel.value = to;

  row.querySelector('.rb-remove-edge-btn')?.addEventListener('click', () => {
    row.remove();
  });
}

async function applyRouteBuilder() {
  clearRouteBuilderErrors();

  const defCapVal = parseFloat($('#rb-default-capacity')?.value || '400');
  const defaults = {
    corridor_capacity_pax_min: isNaN(defCapVal) || defCapVal <= 0 ? 400 : defCapVal,
  };

  const camRows = $$('#rb-cameras-list .rb-camera-row');
  if (camRows.length < 2) {
    showRouteBuilderError('A route must have at least 2 cameras.');
    return;
  }

  const cameras = [];
  const seenIds = new Set();

  for (const row of camRows) {
    const idInput = row.querySelector('.rb-cam-id');
    const nameInput = row.querySelector('.rb-cam-name');
    const capInput = row.querySelector('.rb-cam-cap');
    const holdingCapInput = row.querySelector('.rb-cam-holding-cap');
    const offsetInput = row.querySelector('.rb-cam-clock-offset');

    const cid = idInput ? idInput.value.trim() : '';
    const name = nameInput ? nameInput.value.trim() : '';
    const capStr = capInput ? capInput.value.trim() : '';
    const holdingCapStr = holdingCapInput ? holdingCapInput.value.trim() : '';
    const offsetStr = offsetInput ? offsetInput.value.trim() : '';

    if (!cid) {
      showRouteBuilderError('Camera ID cannot be empty.', idInput);
      return;
    }
    if (seenIds.has(cid)) {
      showRouteBuilderError(`Duplicate Camera ID "${cid}". Each camera ID must be unique.`, idInput);
      return;
    }
    seenIds.add(cid);

    const camObj = {
      camera_id: cid,
      name: name || cid,
    };
    if (capStr) {
      const capVal = parseFloat(capStr);
      if (!isNaN(capVal) && capVal > 0) {
        camObj.corridor_capacity_pax_min = capVal;
      }
    }
    if (holdingCapStr) {
      const hVal = parseFloat(holdingCapStr);
      if (!isNaN(hVal) && hVal > 0) {
        camObj.holding_capacity_pax = hVal;
      }
    }
    if (offsetStr) {
      const oVal = parseFloat(offsetStr);
      if (isNaN(oVal)) {
        showRouteBuilderError('Clip offset must be a number of seconds (0 if the clips start together).', offsetInput);
        return;
      }
      // Negative is legitimate: this clip started BEFORE the reference.
      camObj.clock_offset_sec = oVal;
    }
    cameras.push(camObj);
  }

  const edgeRows = $$('#rb-edges-list .rb-edge-row');
  const edges = [];
  const seenEdgePairs = new Set();

  for (const row of edgeRows) {
    const fromSel = row.querySelector('.rb-edge-from');
    const toSel = row.querySelector('.rb-edge-to');
    const timeInput = row.querySelector('.rb-edge-time');

    const from = fromSel ? fromSel.value.trim() : '';
    const to = toSel ? toSel.value.trim() : '';
    const tVal = parseFloat(timeInput ? timeInput.value : '');

    if (!from || !seenIds.has(from)) {
      showRouteBuilderError(`Upstream camera "${from}" does not exist in the defined cameras list.`, fromSel);
      return;
    }
    if (!to || !seenIds.has(to)) {
      showRouteBuilderError(`Downstream camera "${to}" does not exist in the defined cameras list.`, toSel);
      return;
    }
    if (from === to) {
      showRouteBuilderError(`Self-loop corridor detected (${from} → ${to}). A corridor must connect two distinct cameras.`, fromSel);
      return;
    }
    if (isNaN(tVal) || tVal <= 0) {
      showRouteBuilderError('Travel time must be greater than 0 seconds.', timeInput);
      return;
    }

    const pairKey = `${from}->${to}`;
    if (seenEdgePairs.has(pairKey)) {
      showRouteBuilderError(`Duplicate corridor (${from} → ${to}).`, fromSel);
      return;
    }
    seenEdgePairs.add(pairKey);

    edges.push({
      from,
      to,
      travel_time_sec: tVal,
    });
  }

  const applyBtn = $('#rb-apply-btn');
  if (applyBtn) {
    applyBtn.disabled = true;
    applyBtn.innerHTML = '<span>⏳</span> Applying…';
  }

  try {
    const payload = {
      cameras,
      edges,
      defaults,
    };
    const res = await postJSON('/api/topology/from-route', payload);
    closeRouteBuilderModal();

    // Reload active topology, live graph, and multi-camera upload slots immediately
    await refreshRouteView();
    const customSlots = cameras.map(c => ({ cid: c.camera_id, name: c.name || c.camera_id }));
    await loadSessionSubmitPanel(customSlots);

    const statusEl = $('#session-submit-status');
    if (statusEl) {
      statusEl.innerHTML = `<span style="color: var(--ok);">✓ Applied route with ${cameras.length} cameras.</span>`;
    }
  } catch (err) {
    showRouteBuilderError(err.message || 'Failed applying route topology.');
  } finally {
    if (applyBtn) {
      applyBtn.disabled = false;
      applyBtn.innerHTML = '<span>✓</span> Apply Route';
    }
  }
}

async function resetRouteBuilderToBaseline() {
  if (!confirm('Revert route topology to the hand-authored baseline (configs/topology.yaml)?\nThis will remove any custom route definitions.')) return;

  clearRouteBuilderErrors();
  const resetBtn = $('#rb-reset-baseline-btn');
  if (resetBtn) resetBtn.disabled = true;

  try {
    await postJSON('/api/topology/reset', {});
    closeRouteBuilderModal();

    // Reload active topology, live graph, and reset session slots to initial 2-camera state
    await refreshRouteView();
    resetSessionSubmitPanelToInitial();

    const statusEl = $('#session-submit-status');
    if (statusEl) {
      statusEl.innerHTML = '<span style="color: var(--ok);">✓ Reverted route topology to baseline.</span>';
    }
  } catch (err) {
    showRouteBuilderError(err.message || 'Failed resetting topology.');
  } finally {
    if (resetBtn) resetBtn.disabled = false;
  }
}

function initRouteView() {
  const resetPanelBtn = $('#btn-reset-session-panel');
  if (resetPanelBtn && !resetPanelBtn.dataset.bound) {
    resetPanelBtn.dataset.bound = 'true';
    resetPanelBtn.addEventListener('click', () => {
      resetSessionSubmitPanelToInitial();
      const statusEl = $('#session-submit-status');
      if (statusEl) statusEl.innerHTML = '<span style="color: var(--accent);">🔄 Reset to initial state: 2 Camera Corridors (Cam 1 &amp; Cam 2).</span>';
    });
  }

  $$('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  $('#refresh-topology')?.addEventListener('click', () => refreshRouteView());
  $('#close-breakout')?.addEventListener('click', () => $('#edge-breakout-panel')?.classList.add('hidden'));
  $('#btn-run-route-session')?.addEventListener('click', submitRouteSession);
  $('#refresh-sessions-btn')?.addEventListener('click', loadRouteSessionsList);

  // Route Builder Modal wiring
  $('#btn-configure-route')?.addEventListener('click', openRouteBuilderModal);
  $('#route-builder-close')?.addEventListener('click', closeRouteBuilderModal);
  $('#rb-cancel-btn')?.addEventListener('click', closeRouteBuilderModal);
  $('#rb-add-camera-btn')?.addEventListener('click', () => addRouteBuilderCameraRow());
  $('#rb-add-edge-btn')?.addEventListener('click', () => addRouteBuilderEdgeRow());
  $('#rb-apply-btn')?.addEventListener('click', applyRouteBuilder);
  $('#rb-reset-baseline-btn')?.addEventListener('click', resetRouteBuilderToBaseline);
  $('#route-builder-modal')?.addEventListener('click', (e) => {
    if (e.target.id === 'route-builder-modal') closeRouteBuilderModal();
  });
}

/* -------------------------------------------------------------------- go */
(async function init() {
  wire();
  loadDevice();
  initRouteView();
  initLiveMetricRail();
  await loadModels();
  loadVideos();
  refreshJobs();
  refreshHistory();
  refreshAnpr();
  refreshValidation();
})();


