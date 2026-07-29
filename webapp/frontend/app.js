/* Crowd Safety Testbed - frontend controller.
   Plain JS, no build step: the whole point is that `python -m webapp` is
   the only command needed to get a working UI. */

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
  lastActive: false,   // were any jobs running on the previous poll?
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

/* ------------------------------------------------------------------ device */
async function loadDevice() {
  const badge = $('#device-badge');
  try {
    const d = await api('/api/device');
    if (d.cuda) {
      badge.textContent = `GPU · ${d.name} (${d.total_gb} GB)`;
      badge.className = 'device-badge cuda';
      if (d.total_gb < 6) {
        badge.title = 'Under 6 GB: SlowFast and I3D may run out of memory. ' +
          'Force CPU for those if a run fails.';
        badge.textContent += ' ⚠';
      }
    } else {
      badge.textContent = 'CPU only — runs will be slow';
      badge.className = 'device-badge cpu';
    }
  } catch {
    badge.textContent = 'device unknown';
  }
}

/* ------------------------------------------------------------------ models */
async function loadModels() {
  const data = await api('/api/models');
  state.models = data.models;
  state.categories = data.categories;

  // Preselect the models that run at full capability with no extra setup.
  state.models.forEach((m) => { if (m.status === 'ready') state.selected.add(m.key); });

  renderModels();
}

function renderModels() {
  const host = $('#model-list');
  const order = ['fall', 'violence', 'traffic', 'anpr', 'other'];
  host.innerHTML = '';

  order.forEach((cat) => {
    const items = state.models.filter((m) => m.category === cat);
    if (!items.length) return;

    const title = document.createElement('div');
    title.className = 'cat-title';
    title.textContent = state.categories[cat] || cat;
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
      host.appendChild(el);
    });
  });

  host.querySelectorAll('input[data-key]').forEach((cb) => {
    cb.addEventListener('change', () => {
      if (cb.checked) state.selected.add(cb.dataset.key);
      else state.selected.delete(cb.dataset.key);
      renderModels();
    });
  });

  $('#selected-count').textContent =
    `${state.selected.size} selected of ${state.models.length}`;
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
  if (s.status === 'running' || s.status === 'loading') {
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

  const links = [];
  if (s.annotated) {
    links.push(`<a href="#" class="link-btn" data-video="${esc(s.annotated)}"
                   data-title="${esc(name)}">video</a>`);
  }
  if (s.log_json) {
    links.push(`<a href="#" class="link-btn" data-detections="${esc(s.model_key)}"
                   data-job="${esc(job.id)}" data-title="${esc(name)}">rows</a>`);
    links.push(`<a class="link-btn" href="/api/files/logs/${esc(s.log_json)}">json</a>`);
    links.push(`<a class="link-btn" href="/api/files/logs/${esc(s.log_csv)}">csv</a>`);
  }

  const posClass = s.positives > 0 ? 'pos' : 'zero';
  return `<tr>
    <td>${esc(name)}${fallbackTag}</td>
    <td>${progressCell}</td>
    <td class="num ${posClass}">${s.positives}</td>
    <td class="num">${s.detections}</td>
    <td class="num">${fmtDuration(s.elapsed_sec)}</td>
    <td>${links.join(' · ') || '—'}</td>
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
      <span class="meta">${done}/${job.stages.length} models · every ${job.sample_every_n_frames} frames · ${fmtDuration(job.elapsed_sec)}</span>
      ${cancellable ? `<button class="link-btn" data-cancel="${job.id}">cancel</button>` : ''}
      <span class="meta">${open ? '▾' : '▸'}</span>
    </div>
    ${open ? `<div class="job-body">
      <div class="job-msg">${esc(job.message || '')}${job.error ? ` — ${esc(job.error)}` : ''}</div>
      <table class="stages">
        <thead><tr>
          <th>Model</th><th>Progress</th><th style="text-align:right">Events</th>
          <th style="text-align:right">Rows</th><th style="text-align:right">Time</th><th>Output</th>
        </tr></thead>
        <tbody>${job.stages.map((s) => stageRow(job, s)).join('')}</tbody>
      </table>
    </div>` : ''}
  </div>`;
}

async function refreshJobs() {
  let jobs;
  try { ({ jobs } = await api('/api/jobs')); } catch { return; }

  const host = $('#jobs');
  if (!jobs.length) {
    host.innerHTML = '<div class="empty">No runs yet.</div>';
    return;
  }
  // Newest job starts expanded so progress is visible without a click.
  if (jobs.length && !state.openJobs.size) state.openJobs.add(jobs[0].id);

  host.innerHTML = jobs.map(jobCard).join('');

  host.querySelectorAll('[data-toggle]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      if (ev.target.dataset.cancel || ev.target.dataset.video ||
        ev.target.dataset.detections) return;
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

  host.querySelectorAll('[data-video]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      openModal(el.dataset.title,
        `<video controls autoplay src="/api/files/annotated/${el.dataset.video}"></video>`);
    });
  });

  host.querySelectorAll('[data-detections]').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      openModal(el.dataset.title, '<div class="loading">Loading detections…</div>');
      try {
        const d = await api(
          `/api/jobs/${el.dataset.job}/detections/${el.dataset.detections}?limit=500`);
        renderDetections(d);
      } catch (e) {
        $('#modal-body').innerHTML = `<div class="err">${esc(e.message)}</div>`;
      }
    });
  });

  const active = jobs.some((j) => !['done', 'failed', 'cancelled'].includes(j.status));
  // A job just finished: pick up the files it wrote.
  if (state.lastActive && !active) { refreshHistory(); refreshAnpr(); }
  state.lastActive = active;

  clearTimeout(state.pollTimer);
  if (active) state.pollTimer = setTimeout(refreshJobs, 1200);
}

/* ---------------------------------------------------------------- history */
function historyCard(group) {
  const when = new Date(group.modified_at * 1000).toLocaleString();
  const rows = group.stages.map((s) => {
    const links = [];
    if (s.annotated) {
      links.push(`<a href="#" class="link-btn" data-video="${esc(s.annotated)}"
                     data-title="${esc(s.model_label)}">video</a>`);
    }
    links.push(`<a href="#" class="link-btn" data-hist-video="${esc(group.video)}"
                   data-hist-model="${esc(s.model_key)}"
                   data-title="${esc(s.model_label)}">rows</a>`);
    links.push(`<a class="link-btn" href="/api/files/logs/${esc(s.log_json)}">json</a>`);
    if (s.log_csv) {
      links.push(`<a class="link-btn" href="/api/files/logs/${esc(s.log_csv)}">csv</a>`);
    }

    const tag = s.scoring_modes && s.scoring_modes.geometric_fallback
      ? '<div class="scoring-tag">geometric fallback</div>'
      : (s.scoring_modes && s.scoring_modes.kinetics_zeroshot
        ? '<div class="scoring-tag">kinetics zero-shot</div>' : '');

    const labels = Object.entries(s.label_counts || {})
      .map(([k, v]) => `${esc(k)} ${v}`).join(', ') || '—';

    return `<tr>
      <td>${esc(s.model_label)}${tag}</td>
      <td class="num ${s.positives > 0 ? 'pos' : 'zero'}">${s.positives}</td>
      <td class="num">${s.detections}</td>
      <td class="subtle">${labels}</td>
      <td>${links.join(' · ')}</td>
    </tr>`;
  }).join('');

  return `<div class="job">
    <div class="job-head" data-hist-toggle="${esc(group.video)}">
      <span class="status done">saved</span>
      <span class="title">${esc(group.video)}</span>
      <span class="meta">${group.stages.length} model(s) · ${group.total_positives} event(s) · ${esc(when)}</span>
      <span class="meta">${state.openHistory.has(group.video) ? '▾' : '▸'}</span>
    </div>
    ${state.openHistory.has(group.video) ? `<div class="job-body">
      <table class="stages">
        <thead><tr><th>Model</th><th style="text-align:right">Events</th>
          <th style="text-align:right">Rows</th><th>Labels</th><th>Output</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>` : ''}
  </div>`;
}

async function refreshHistory() {
  const host = $('#history');
  let history;
  try { ({ history } = await api('/api/history')); }
  catch (e) { host.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }

  if (!history.length) {
    host.innerHTML = '<div class="empty">Nothing in outputs/ yet.</div>';
    return;
  }
  // Most recent video expanded by default — that's what you just ran.
  if (!state.openHistory.size) state.openHistory.add(history[0].video);

  host.innerHTML = history.map(historyCard).join('');

  host.querySelectorAll('[data-hist-toggle]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      if (ev.target.dataset.video || ev.target.dataset.histModel) return;
      const v = el.dataset.histToggle;
      state.openHistory.has(v) ? state.openHistory.delete(v) : state.openHistory.add(v);
      refreshHistory();
    });
  });

  host.querySelectorAll('[data-video]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      openModal(el.dataset.title,
        `<video controls autoplay src="/api/files/annotated/${el.dataset.video}"></video>`);
    });
  });

  host.querySelectorAll('[data-hist-model]').forEach((el) => {
    el.addEventListener('click', async (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      openModal(el.dataset.title, '<div class="loading">Loading detections…</div>');
      try {
        const d = await api(`/api/history/${encodeURIComponent(el.dataset.histVideo)}` +
          `/${encodeURIComponent(el.dataset.histModel)}/detections?limit=500`);
        renderDetections(d);
      } catch (e) {
        $('#modal-body').innerHTML = `<div class="err">${esc(e.message)}</div>`;
      }
    });
  });
}

function renderDetections(d) {
  if (!d.rows.length) {
    $('#modal-body').innerHTML =
      '<div class="empty">No positive detections in this run.</div>';
    return;
  }
  const rows = d.rows.map((r) => `<tr>
    <td>${r.timestamp_sec.toFixed(2)}s</td>
    <td>${esc(r.label)}</td>
    <td>${r.confidence.toFixed(3)}</td>
    <td>${r.extra && r.extra.track_id != null ? r.extra.track_id : '—'}</td>
    <td>${r.extra && r.extra.scoring ? esc(r.extra.scoring) : '—'}</td>
  </tr>`).join('');
  $('#modal-body').innerHTML = `
    <div class="hint">${d.total} positive detection(s); showing ${d.rows.length}.</div>
    <table><thead><tr><th>Time</th><th>Label</th><th>Conf</th>
      <th>Track</th><th>Scoring</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/* ------------------------------------------------------------------- anpr */
function anprCard(g) {
  const c = g.counts || {};
  const withPlate = g.vehicles.filter((v) => v.plate);
  const without = g.vehicles.filter((v) => !v.plate);
  // Readable plates first — that's what you came to see.
  const ordered = [...withPlate, ...without];

  const tiles = ordered.map((v) => {
    const img = v.image
      ? `<img src="/api/anpr/${encodeURIComponent(g.video)}/vehicles/${encodeURIComponent(v.image)}" alt="" />`
      : '<div class="noimg">no image</div>';
    const plate = v.plate
      ? `<div class="plate">${esc(v.plate_display || v.plate)}</div>`
      : `<div class="plate none">${esc(statusText(v.plate_status, v.plate_width_px))}</div>`;
    return `<figure class="vcard${v.plate ? '' : ' unread'}">
      ${img}
      <figcaption>
        <div class="vname">${esc(v.caption || v.vehicle_class)}</div>
        ${plate}
        <div class="vmeta">${v.first_seen_sec}s–${v.last_seen_sec}s · ${v.frames_seen} frames${
          v.plate ? ` · ${Math.round(v.plate_agreement * 100)}% agree` : ''}</div>
      </figcaption>
    </figure>`;
  }).join('');

  return `<div class="job">
    <div class="job-head" data-anpr-toggle="${esc(g.video)}">
      <span class="status ${withPlate.length ? 'done' : 'cancelled'}">${withPlate.length} read</span>
      <span class="title">${esc(g.video)}</span>
      <span class="meta">${c.total || 0} vehicle(s) captured · ${withPlate.length} with a plate</span>
      <span class="meta">${state.openAnpr.has(g.video) ? '▾' : '▸'}</span>
    </div>
    ${state.openAnpr.has(g.video) ? `<div class="job-body">
      ${withPlate.length === 0 ? `<div class="warnbox">No plate was legible in this video.
        ${c.too_small ? `${c.too_small} plate(s) were detected but too small to read` : ''}
        — ANPR needs roughly 90px of plate width. Wide or distant traffic shots
        can't resolve the characters, regardless of model.</div>` : ''}
      <div class="gallery">${tiles}</div>
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
  const host = $('#anpr');
  let galleries;
  try { ({ galleries } = await api('/api/anpr')); }
  catch (e) { host.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }

  if (!galleries.length) {
    host.innerHTML = '<div class="empty">No ANPR runs yet. Select the ANPR model and run a video.</div>';
    return;
  }
  if (!state.openAnpr.size) state.openAnpr.add(galleries[0].video);

  host.innerHTML = galleries.map(anprCard).join('');
  host.querySelectorAll('[data-anpr-toggle]').forEach((el) => {
    el.addEventListener('click', () => {
      const v = el.dataset.anprToggle;
      state.openAnpr.has(v) ? state.openAnpr.delete(v) : state.openAnpr.add(v);
      refreshAnpr();
    });
  });
}

/* ------------------------------------------------------------------ modal */
function openModal(title, html) {
  $('#modal-title').textContent = title;
  $('#modal-body').innerHTML = html;
  $('#modal').classList.remove('hidden');
}
function closeModal() {
  $('#modal').classList.add('hidden');
  $('#modal-body').innerHTML = '';   // stops any playing video
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
  btn.textContent = 'Starting…';

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
    btn.textContent = 'Run selected models';
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

  $('#run-btn').addEventListener('click', runJob);
  $('#refresh-jobs').addEventListener('click', refreshJobs);
  $('#refresh-history').addEventListener('click', refreshHistory);
  $('#refresh-anpr').addEventListener('click', refreshAnpr);
  $('#modal-close').addEventListener('click', closeModal);
  $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  $('#clear-outputs').addEventListener('click', async () => {
    if (!confirm('Delete every generated log and annotated video in outputs/?\n' +
      'Source videos in test_videos/ are not touched.')) return;
    const r = await api('/api/outputs', { method: 'DELETE' });
    $('#run-error').textContent = `Deleted ${r.removed} file(s).`;
    state.openHistory.clear();
    state.openAnpr.clear();
    refreshJobs();
    refreshHistory();
  });

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
