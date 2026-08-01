/* Crowd Safety Testbed - Frontend Controller & Detail Inspector */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  models: [],
  categories: {},
  selected: new Set(),
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
  const order = ['fall', 'violence', 'traffic', 'anpr', 'umbrella', 'fire', 'crush', 'other'];
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
               ${m.status === 'blocked' ? 'disabled' : ''} data-key="${m.key}" />
        <div>
          <div class="name">${esc(m.label)}
            <span class="pill ${m.status}">${m.status}</span></div>
          <div class="blurb">${esc(m.blurb)}</div>
          ${m.note ? `<div class="note ${m.status}">${esc(m.note)}</div>` : ''}
        </div>`;

      const cb = el.querySelector('input');
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
    ? '<div class="scoring-tag">geometric fallback</div>'
    : (s.scoring_modes && s.scoring_modes.kinetics_zeroshot
      ? '<div class="scoring-tag">kinetics zero-shot</div>' : '');

  const posClass = s.positives > 0 ? 'pos' : 'zero';
  return `<tr>
    <td>${esc(name)}${fallbackTag}</td>
    <td>${progressCell}</td>
    <td class="num ${posClass}">${s.positives}</td>
    <td class="num">${s.detections}</td>
    <td class="num">${fmtDuration(s.elapsed_sec)}</td>
    <td>
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
      <span class="status ${job.status}">${job.status}</span>
      <span class="title">${esc(job.video_name || job.source)}</span>
      <span class="meta">${done}/${job.stages.length} models · stride ${job.sample_every_n_frames} · ${fmtDuration(job.elapsed_sec)}</span>
      ${cancellable ? `<button class="link-btn" data-cancel="${job.id}">cancel</button>` : ''}
      <span class="meta">${open ? '▾' : '▸'}</span>
    </div>
    ${open ? `<div class="job-body">
      <div class="job-msg">${esc(job.message || '')}${job.error ? ` — ${esc(job.error)}` : ''}</div>
      <table class="stages">
        <thead><tr>
          <th>Model</th><th>Progress</th><th style="text-align:right">Events</th>
          <th style="text-align:right">Rows</th><th style="text-align:right">Time</th><th>Actions</th>
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
  else if (group.stages.some(s => s.model_key.includes('crush') || s.model_key.includes('traffic'))) primaryCat = 'traffic';

  const catLabel = primaryCat === 'anpr' ? 'ANPR Read' : (primaryCat.charAt(0).toUpperCase() + primaryCat.slice(1));

  // Build model summaries
  const modelPills = group.stages.map((s) => {
    const tag = s.scoring_modes && s.scoring_modes.geometric_fallback
      ? '<span class="pill fallback">Geometric Fallback</span>'
      : (s.scoring_modes && s.scoring_modes.kinetics_zeroshot
        ? '<span class="pill ready">Kinetics Zero-Shot</span>' : '');
    return `<div style="display:flex; align-items:center; gap:6px; margin-top:3px;">
      <span style="font-weight:700; color:var(--text); font-size:12.5px;">${esc(s.model_label)}</span>
      <span class="${s.positives > 0 ? 'pos' : 'zero'}" style="font-size:11.5px; font-weight:700;">(${s.positives} alerts)</span>
      ${tag}
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
      <div style="font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); font-weight:700; margin-bottom:4px;">Models & Classifications</div>
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
      <div class="vmeta">${v.first_seen_sec}s–${v.last_seen_sec}s · ${v.frames_seen} frames${
        v.plate ? ` · ${Math.round(v.plate_agreement * 100)}% agree` : ''}</div>
    </figcaption>
  </figure>`;
}

function anprCard(g) {
  const c = g.counts || {};
  const withPlate = g.vehicles.filter((v) => v.plate);
  const without   = g.vehicles.filter((v) => !v.plate);
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
  $('#modal-title').textContent = `Job Details — ${jobId}`;
  $('#modal-body').innerHTML = '<div class="loading">Loading detections…</div>';
  $('#modal').classList.remove('hidden');

  try {
    const detections = await api(`/api/jobs/${jobId}/detections/${modelKey}?limit=500`);
    state.currentDetail = {
      videoName: jobId,
      group: null,
      primaryStage: { model_key: modelKey, model_label: modelKey },
      detections,
      annotatedVideo: null,
    };
    renderModalTab('overview');
  } catch (err) {
    $('#modal-body').innerHTML = `<div class="err">${esc(err.message)}</div>`;
  }
}

function renderModalTab(tabName) {
  state.activeModalTab = tabName;
  $$('#modal-nav .modal-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mtab === tabName);
  });

  const detail = state.currentDetail;
  if (!detail) return;

  const host = $('#modal-body');

  if (tabName === 'overview') {
    const g = detail.group;
    const s = detail.primaryStage || {};
    const d = detail.detections || {};
    const totalPositives = g ? g.total_positives : (d.rows ? d.rows.filter(r => r.confidence > 0.5).length : 0);
    const maxConf = d.rows && d.rows.length ? Math.max(...d.rows.map(r => r.confidence || 0)) : 0;
    const mainLabel = d.rows && d.rows.length ? (d.rows[0].label || 'detection') : 'N/A';

    const labelsList = s.label_counts
      ? Object.entries(s.label_counts).map(([k, v]) => `<span class="plate-badge" style="margin-right:6px;">${esc(k)}: ${v}</span>`).join(' ')
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

      <div class="detail-section-title">Classifications & Metadata</div>
      <table class="detail-info-table">
        <tbody>
          <tr><td class="key">Source Video</td><td class="val">${esc(detail.videoName)}</td></tr>
          <tr><td class="key">Primary Model</td><td class="val">${esc(s.model_label || s.model_key || '—')}</td></tr>
          <tr><td class="key">Category</td><td class="val">${esc(s.model_key ? s.model_key.split('_')[0] : 'general')}</td></tr>
          <tr><td class="key">Detected Labels</td><td class="val">${labelsList}</td></tr>
          ${s.modified_at ? `<tr><td class="key">Run Date</td><td class="val">${new Date(s.modified_at * 1000).toLocaleString()}</td></tr>` : ''}
          ${s.log_json ? `<tr><td class="key">Log Artifact</td><td class="val"><a href="/api/files/logs/${esc(s.log_json)}" target="_blank" class="link-btn">Download ${esc(s.log_json)}</a></td></tr>` : ''}
        </tbody>
      </table>
    `;
  } else if (tabName === 'timeline') {
    const allStages = detail.allStages;

    // Single model or job context — show detections directly
    if (!allStages || allStages.length <= 1) {
      renderDetections(detail.detections);
      return;
    }

    // Multiple models — show selector table + detections below
    const activeKey = detail.activeTimelineModelKey || allStages[0].model_key;

    const selectorRows = allStages.map(s => {
      const isActive = s.model_key === activeKey;
      return `<tr class="video-select-row${isActive ? ' active-video-row' : ''}" data-load-model="${esc(s.model_key)}" data-model-label="${esc(s.model_label)}" style="cursor:pointer;${isActive ? ' background:rgba(99,102,241,0.15);' : ''}">
        <td style="font-weight:700; color:var(--text);">${esc(s.model_label)}</td>
        <td class="${s.positives > 0 ? 'pos' : 'zero'}" style="font-weight:700;">${s.positives} alerts</td>
        <td>${s.detections} rows</td>
        <td><span style="color:var(--accent); font-weight:700; font-size:12px;">${isActive ? '⬤ Viewing' : '▶ Load'}</span></td>
      </tr>`;
    }).join('');

    host.innerHTML = `
      <div class="detail-section-title" style="margin-top:0;">⏱️ Select a Model to View Detections</div>
      <p class="hint" style="margin-bottom:10px;">${allStages.length} models scored this video — click any row to load its detection rows below.</p>
      <table style="margin-bottom:20px; border:1px solid var(--border); border-radius:var(--radius); overflow:hidden;">
        <thead><tr><th>Model</th><th style="text-align:right">Alerts</th><th style="text-align:right">Total Rows</th><th></th></tr></thead>
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
        // Temporarily swap detections so renderDetections renders into the right container
        const prevBody = $('#modal-body');
        // Render into the sub-host
        if (!d.rows.length) {
          detectHost.innerHTML = '<div class="empty">No positive detections for this model.</div>';
          return;
        }
        const hasPlates = d.rows.some(r => r.extra && (r.extra.plate || r.extra.plate_display || r.extra.plate_status));
        const rows = d.rows.map(r => {
          const extra = r.extra || {};
          const plateStr = extra.plate_display || extra.plate || '';
          const plateStatus = extra.plate_status || '';
          let plateCell = '—';
          if (plateStr) plateCell = `<span class="plate-badge">${esc(plateStr)}</span>`;
          else if (plateStatus) plateCell = `<span class="subtle">${esc(statusText(plateStatus, extra.plate_width_px))}</span>`;
          const details = extra.scoring ? esc(extra.scoring)
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
        detectHost.innerHTML = `
          <div class="hint" style="margin-bottom:10px;">Showing top ${d.rows.length} of ${d.total} positive detection rows for <strong>${esc(modelLabel)}</strong>.</div>
          <table><thead><tr><th>Timestamp</th><th>Class Label</th><th>Conf</th>
            <th>Track ID</th>${hasPlates ? '<th>Plate Read</th>' : ''}<th>Details</th></tr></thead>
            <tbody>${rows}</tbody></table>`;
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
        <video controls autoplay src="/api/files/annotated/${esc(allVids[0].file)}"></video>
        <p class="hint" style="margin-top:12px;"><strong>${esc(allVids[0].label)}</strong> — annotated video output with bounding box overlays.</p>
      `;
    } else {
      // Multiple videos — show selection table then player
      const currentFile = detail.activeAnnotatedVideo || allVids[0].file;
      const currentEntry = allVids.find(v => v.file === currentFile) || allVids[0];

      const tableRows = allVids.map((v) => {
        const isActive = v.file === currentFile;
        return `<tr class="video-select-row${isActive ? ' active-video-row' : ''}" data-play-video="${esc(v.file)}" data-play-label="${esc(v.label)}" style="cursor:pointer;${isActive ? ' background:rgba(99,102,241,0.15);' : ''}">
          <td style="font-weight:700; color:var(--text);">${esc(detail.videoName)}</td>
          <td style="color:var(--muted);">${esc(v.label)}</td>
          <td style="font-size:11px; color:var(--subtle); font-family:var(--font-code);">${esc(v.file)}</td>
          <td><span style="color:var(--accent); font-weight:700; font-size:12px;">${isActive ? '▶ Playing' : '▶ Play'}</span></td>
        </tr>`;
      }).join('');

      host.innerHTML = `
        <div class="detail-section-title" style="margin-top:0;">🎬 Select a Model Output Video</div>
        <p class="hint" style="margin-bottom:10px;">${allVids.length} annotated outputs available — click any row to switch playback.</p>
        <table style="margin-bottom:20px; border:1px solid var(--border); border-radius:var(--radius); overflow:hidden;">
          <thead><tr>
            <th>Video File</th>
            <th>Model</th>
            <th>Filename</th>
            <th></th>
          </tr></thead>
          <tbody>${tableRows}</tbody>
        </table>
        <div style="margin-bottom:8px; font-size:12px; color:var(--muted);">Now playing: <strong style="color:var(--text);" id="video-now-playing">${esc(currentEntry.label)}</strong></div>
        <video id="modal-video-player" controls autoplay src="/api/files/annotated/${esc(currentFile)}"></video>
      `;

      // Wire up row clicks to switch video
      host.querySelectorAll('.video-select-row').forEach(row => {
        row.addEventListener('click', () => {
          const file = row.dataset.playVideo;
          const label = row.dataset.playLabel;
          detail.activeAnnotatedVideo = file;

          const player = $('#modal-video-player');
          if (player) { player.src = `/api/files/annotated/${file}`; player.play(); }

          const nowPlaying = $('#video-now-playing');
          if (nowPlaying) nowPlaying.textContent = label;

          // Update row highlighting
          host.querySelectorAll('.video-select-row').forEach(r => {
            const isNowActive = r.dataset.playVideo === file;
            r.style.background = isNowActive ? 'rgba(99,102,241,0.15)' : '';
            const actionCell = r.querySelector('span');
            if (actionCell) actionCell.textContent = isNowActive ? '▶ Playing' : '▶ Play';
          });
        });
      });
    }
  } else if (tabName === 'raw') {
    const rawJsonStr = JSON.stringify(detail.detections, null, 2);
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

function renderDetections(d) {
  if (!d || !d.rows || !d.rows.length) {
    $('#modal-body').innerHTML = '<div class="empty">No positive detections recorded in this run.</div>';
    return;
  }
  const hasPlates = d.rows.some((r) => r.extra && (r.extra.plate || r.extra.plate_display || r.extra.plate_status));
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

  $('#modal-body').innerHTML = `
    <div class="hint" style="margin-bottom:12px;">Total positive detections: ${d.total}; showing top ${d.rows.length} rows.</div>
    <table><thead><tr><th>Timestamp</th><th>Class Label</th><th>Conf</th>
      <th>Track ID</th>${hasPlates ? '<th>Plate Read</th>' : ''}<th>Details</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
}

function closeModal() {
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
    await postJSON('/api/jobs', {
      source,
      models: [...state.selected],
      sample_every_n_frames: parseInt($('#stride').value, 10),
      device: $('#device').value || null,
      export_video: $('#export-video').checked,
      pose_size: $('#pose-size').value,
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
function wire() {
  $$('#source-tabs .tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('#source-tabs .tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
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

  $('#run-btn').addEventListener('click', runJob);
  $('#refresh-jobs').addEventListener('click', refreshJobs);
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
        try { await api(`/api/anpr/${encodeURIComponent(g.video)}`, { method: 'DELETE' }); } catch {}
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
})();
