const app = document.getElementById('app');

const state = {
  config: null,
  task: null,
  draftId: null,
  splits: null,
  activeSplit: 'specialization',
  result: null,
  runName: null,
  modelCatalog: [],
  modelResults: [],
};

const splitMeta = {
  specialization: ['Specialization', 'Builds the specialization. Never used for testing.'],
  validation: ['Validation', 'Used only to select settings.'],
  hidden: ['Hidden test', 'Primary measure, never visible while fitting.'],
  paraphrase: ['Paraphrases', 'Same meaning with substantially different wording.'],
  hard: ['Hard cases', 'Ambiguity, negation, and irrelevant information.'],
};

const methods = [
  ['baseline', 'Original Laya', 'Reference without specialization'],
  ['prompt_only', 'Improved prompt', 'Rewritten policy without geometric intervention'],
  ['nearest_prototype', 'Prototypes', 'Centers computed from Laya embeddings'],
  ['multiclass_centroids', 'Multiclass centroids', 'One normalized direction per class'],
  ['contrastive_vector', 'Contrastive vector', 'Binary only: positive minus negative'],
  ['whitened_prototypes', 'Whitened prototypes', 'Corrects embedding covariance'],
  ['residual_embedding_transform', 'Residual transform', 'Small analytical transform'],
  ['activation_steering', 'Activation steering', 'True injection before the decision head'],
  ['multi_vector_steering', 'Multi-vector steering', 'Several semantic directions per class'],
  ['pairwise_ranking', 'Pairwise ranking', 'Deterministic Copeland aggregation'],
];

function esc(value = '') {
  return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function fmt(value, digits = 1) {
  return value == null || Number.isNaN(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(digits)} %`;
}

function toast(message, error = false) {
  const node = document.createElement('div');
  node.className = `toast${error ? ' error' : ''}`;
  node.textContent = message;
  document.getElementById('toast-region').append(node);
  setTimeout(() => node.remove(), 4200);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {'Content-Type': 'application/json', ...(options.headers || {})},
  });
  if (!response.ok) {
    let detail = `HTTP error ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function stepper(active) {
  const names = ['Describe', 'Review', 'Data', 'Results'];
  return `<div class="stepper">${names.map((name, i) => `
    <div class="step ${i < active ? 'done' : i === active ? 'active' : ''}">
      <b>${i < active ? '✓' : i + 1}</b><span>${name}</span>
    </div>`).join('')}</div>`;
}

function setNav(name) {
  document.querySelectorAll('[data-nav]').forEach(link => link.classList.toggle('active', link.dataset.nav === name));
}

function navigate(path) {
  history.pushState({}, '', path);
  route();
}

document.addEventListener('click', event => {
  const link = event.target.closest('a[data-nav]');
  if (link) {
    event.preventDefault();
    navigate(link.getAttribute('href'));
  }
});
window.addEventListener('popstate', route);

async function route() {
  window.scrollTo({top: 0, behavior: 'instant'});
  if (location.pathname === '/create') {
    setNav('create');
    if (state.result) renderResults();
    else if (state.splits) renderDatasetReview();
    else if (state.task) renderPolicy();
    else await renderCreate();
  } else if (location.pathname === '/runs') {
    setNav('runs');
    await renderRuns();
  } else if (location.pathname === '/library') {
    setNav('library');
    const runId = new URLSearchParams(location.search).get('run');
    if (runId) await openLibraryRun(runId);
    else await renderLibrary();
  } else if (location.pathname === '/playground') {
    setNav('playground');
    await renderPlayground();
  } else {
    setNav('home');
    await renderHome();
  }
  app.focus({preventScroll: true});
}

async function renderHome() {
  app.innerHTML = `<div class="page-loading">Loading experiments…</div>`;
  let runs = [];
  try { runs = await api('/api/runs'); } catch (error) { toast(error.message, true); }
  const real = runs.filter(run => run.backend !== 'fake');
  const completed = runs.filter(run => run.status === 'completed');
  app.innerHTML = `<div class="page">
    <section class="hero">
      <div class="eyebrow">Training-free specialization studio</div>
      <h1>Teach Laya a new decision, simply.</h1>
      <p class="lede">Describe your goal in plain language. The studio builds a policy, generates independent data, compares methods, and prepares the best artifact.</p>
      <div class="hero-actions">
        <button class="btn btn-lime btn-lg" id="hero-create">Create a specialization <span>→</span></button>
        <a class="btn btn-ghost btn-lg" href="/runs" data-nav="runs" style="color:white;border-color:#365052">View results</a>
      </div>
    </section>
    <div class="stats-grid">
      <div class="stat-card"><small>Completed experiments</small><strong>${completed.length}</strong></div>
      <div class="stat-card"><small>Real Laya tests</small><strong>${real.length}</strong></div>
      <div class="stat-card"><small>Available strategies</small><strong>10</strong></div>
      <div class="stat-card"><small>Modified weights</small><strong>0</strong><span class="trend">frozen</span></div>
    </div>
    <div class="two-col">
      <section class="card">
        <div class="section-head"><div><h2>A guided workflow</h2><p>Every scientific boundary remains visible.</p></div></div>
        <div class="ranking">
          ${[
            ['1', 'Describe the decision', 'One sentence is enough. Labels can be inferred.'],
            ['2', 'Review the policy', 'Classes, concepts, and edge cases remain editable.'],
            ['3', 'Review the examples', 'Specialization and hidden tests stay separate and editable.'],
            ['4', 'Compare and export', 'The best result is selected on the hidden test.'],
          ].map(x => `<div class="rank-row" style="grid-template-columns:32px 1fr"><span class="rank-num">${x[0]}</span><div class="rank-name"><strong>${x[1]}</strong><small>${x[2]}</small></div></div>`).join('')}
        </div>
      </section>
      <section class="card">
        <div class="eyebrow">Latest real run</div>
        ${real[0] ? `<h2 style="margin-top:9px">${esc(real[0].id)}</h2>
          <p class="lede" style="font-size:14px">${esc(real[0].base_model)} · ${real[0].result_count} results</p>
          <button class="btn btn-secondary" id="latest-run" style="margin-top:18px">Open experiment</button>` : `
          <div class="empty"><p>No real run yet.</p></div>`}
      </section>
    </div>
  </div>`;
  document.getElementById('hero-create').onclick = () => navigate('/create');
  if (real[0]) document.getElementById('latest-run').onclick = () => { navigate('/runs'); setTimeout(() => openRun(real[0].id), 80); };
}

async function renderCreate() {
  let settings = {base_url: localStorage.getItem('laya.endpoint') || 'https://openrouter.ai/api/v1', model: localStorage.getItem('laya.model') || '', has_api_key: false};
  try { settings = {...settings, ...await api('/api/studio/settings')}; } catch (_) {}
  try { state.modelCatalog = await api('/api/studio/models'); } catch (_) {}
  const savedEndpoint = settings.base_url;
  const savedModel = settings.model;
  app.innerHTML = `<div class="page">
    ${stepper(0)}
    <div class="page-head compact"><div><div class="eyebrow">New specialization</div><h1>What should Laya learn?</h1><p class="lede">Describe the decision as you would to a colleague. The LLM will turn it into a testable policy.</p></div></div>
    <form class="card" id="create-form">
      <div class="form-grid">
        <div class="field full"><label for="description">Your goal, in plain language</label>
          <textarea id="description" required minlength="10" placeholder="E.g. Classify support tickets by urgency. A widespread outage or data leak requires immediate action; a usage question can wait."></textarea>
          <span class="hint">Include business rules, exceptions, and what matters most.</span>
        </div>
        <div class="field full"><label for="labels">Desired labels <span class="hint">— optional</span></label>
          <input id="labels" placeholder="URGENT, NORMAL, IGNORE">
          <span class="hint">Leave empty to let the LLM propose classes.</span>
        </div>
      </div>
      <div class="provider-box" style="margin-top:22px">
        <div class="section-head"><div><h3>LLM connection</h3><p>OpenRouter, OpenAI, or any compatible endpoint.</p></div><span class="badge">OpenAI compatible</span></div>
        <div class="form-grid">
          <div class="field full"><label for="endpoint">Endpoint</label><div class="input-with-icon"><span>↗</span><input id="endpoint" type="url" value="${esc(savedEndpoint)}" required></div></div>
          <div class="field"><label for="model">Model</label><input id="model" value="${esc(savedModel)}" placeholder="openai/gpt-4.1-mini" required></div>
          <div class="field"><label for="api-key">API key</label><input id="api-key" type="password" autocomplete="off" placeholder="${settings.has_api_key ? 'Saved key ••••••••' : 'sk-or-…'}" ${settings.has_api_key ? '' : 'required'}></div>
        </div>
        <label class="method-option" style="margin-top:14px"><input type="checkbox" id="save-connection"><span><strong>Save this connection to .env</strong><small>Stored only on this machine, protected with restrictive permissions, and ignored by Git.</small></span></label>
      </div>
      <div class="form-actions"><span class="hint">The first short step generates only the policy.</span><button class="btn btn-primary btn-lg" type="submit">Design policy <span>→</span></button></div>
    </form>
  </div>`;
  document.getElementById('create-form').onsubmit = startPlan;
}

async function startPlan(event) {
  event.preventDefault();
  const labels = document.getElementById('labels').value.split(',').map(x => x.trim()).filter(Boolean);
  state.config = {
    description: document.getElementById('description').value.trim(),
    labels: labels.length ? labels : null,
    provider: {
      base_url: document.getElementById('endpoint').value.trim().replace(/\/$/, ''),
      model: document.getElementById('model').value.trim(),
      api_key: document.getElementById('api-key').value,
    },
    seed: Math.floor(Math.random() * 1_000_000),
  };
  localStorage.setItem('laya.endpoint', state.config.provider.base_url);
  localStorage.setItem('laya.model', state.config.provider.model);
  try {
    if (document.getElementById('save-connection').checked) {
      await api('/api/studio/settings', {method: 'POST', body: JSON.stringify({provider: state.config.provider})});
      toast('Connection saved to .env');
    }
    const job = await api('/api/studio/plan', {method: 'POST', body: JSON.stringify(state.config)});
    await watchJob(job.id, 0, result => { state.task = result.task; renderPolicy(); });
  } catch (error) { toast(error.message, true); renderCreate(); }
}

async function watchJob(jobId, step, onDone) {
  renderProgress(step, {progress: 2, phase: 'Starting', message: 'Preparing the job…'});
  let lastSignature = '';
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 700));
    let job;
    try { job = await api(`/api/studio/jobs/${jobId}`); }
    catch (error) { renderProgress(step, {status: 'failed', error: error.message}); return; }
    const signature = JSON.stringify([job.status, job.progress, job.phase, job.message, job.error]);
    if (signature !== lastSignature) {
      renderProgress(step, job);
      lastSignature = signature;
    }
    if (job.status === 'completed') { onDone(job.result); return; }
    if (job.status === 'failed') return;
  }
}

function renderProgress(step, job) {
  app.innerHTML = `<div class="page">${stepper(step)}<section class="card progress-card">
    <div class="orb">✦</div><div class="eyebrow">${job.status === 'failed' ? 'Action required' : 'Laya Studio is working'}</div>
    <h2>${esc(job.phase || 'Processing')}</h2><p>${esc(job.message || '')}</p>
    ${job.status === 'failed' ? `<div class="error-panel"><strong>Unable to complete</strong><br>${esc(job.error || 'Unknown error')}</div><button class="btn btn-secondary" style="margin-top:18px" onclick="location.reload()">Try again</button>` : `
      <div class="progress-track"><div class="progress-bar" style="width:${job.progress || 0}%"></div></div><span class="progress-percent">${job.progress || 0} %</span>`}
  </section></div>`;
}

function renderPolicy() {
  const task = state.task;
  const descriptions = task.decision.class_descriptions || {};
  const benchmarkSame = true;
  app.innerHTML = `<div class="page">${stepper(1)}
    <div class="page-head compact"><div><div class="eyebrow">Proposed policy</div><h1>Review what Laya should decide.</h1><p class="lede">Everything remains editable before generating examples.</p></div></div>
    <section class="card">
      <div class="policy-title"><div class="field"><label for="task-name">Name</label><input id="task-name" value="${esc(task.name)}"></div><div class="field"><label for="task-domain">Domain</label><input id="task-domain" value="${esc(task.domain)}"></div></div>
      <div class="field" style="margin-top:15px"><label for="task-description">Detailed description</label><textarea id="task-description">${esc(task.description)}</textarea></div>
      <div class="field" style="margin-top:15px"><label for="task-policy">Rules and edge cases</label><textarea id="task-policy">${esc(task.policy || '')}</textarea></div>
      <div class="section-head" style="margin-top:22px"><div><h3>Decision classes</h3><p>Rename or refine boundaries if needed.</p></div></div>
      <div class="label-grid" id="label-grid">${task.decision.labels.map((label, i) => `<div class="label-card"><input data-label-index="${i}" value="${esc(label)}" aria-label="Class name ${i+1}"><textarea data-desc-index="${i}" aria-label="Class description ${i+1}">${esc(descriptions[label] || '')}</textarea></div>`).join('')}</div>
      ${task.semantic_concepts?.length ? `<div style="margin-top:20px"><label style="font-size:13px;font-weight:700">Detected concepts</label><div class="chips" style="margin-top:9px">${task.semantic_concepts.map(x => `<span class="chip">${esc(x)}</span>`).join('')}</div></div>` : ''}
    </section>
    <section class="card">
      <div class="section-head"><div><h2>Data generation</h2><p>Two prompts and seed spaces remain strictly separated.</p></div></div>
      <div class="form-grid">
        <div class="field"><label for="count-spec">Specialization examples</label><input id="count-spec" type="number" min="6" max="500" value="30"></div>
        <div class="field"><label for="count-hidden">Hidden test cases</label><input id="count-hidden" type="number" min="4" max="2000" value="40"></div>
        <div class="field"><label for="count-validation">Validation</label><input id="count-validation" type="number" min="4" value="18"></div>
        <div class="field"><label for="count-robust">Paraphrases / hard cases</label><input id="count-robust" type="number" min="4" value="16"></div>
      </div>
      <label class="method-option" style="margin-top:18px"><input type="checkbox" id="same-provider" checked><span><strong>Use the same endpoint with strict isolation</strong><small>Uncheck to generate the benchmark with another model or provider.</small></span></label>
      <div id="benchmark-provider" class="provider-box" style="display:none;margin-top:12px">
        <div class="form-grid"><div class="field full"><label>Benchmark endpoint</label><input id="bench-endpoint" value="${esc(state.config.provider.base_url)}"></div><div class="field"><label>Benchmark model</label><input id="bench-model" value="${esc(state.config.provider.model)}"></div><div class="field"><label>Benchmark API key</label><input id="bench-key" type="password" autocomplete="off"></div></div>
      </div>
      <div class="form-actions"><button class="btn btn-ghost" id="back-description">← Edit request</button><button class="btn btn-primary btn-lg" id="generate-data">Generate data <span>→</span></button></div>
    </section>
  </div>`;
  document.getElementById('same-provider').onchange = e => document.getElementById('benchmark-provider').style.display = e.target.checked ? 'none' : 'block';
  document.getElementById('back-description').onclick = () => { state.task = null; renderCreate(); };
  document.getElementById('generate-data').onclick = startGeneration;
}

function syncTaskFromPolicy() {
  const oldLabels = state.task.decision.labels;
  const labels = [...document.querySelectorAll('[data-label-index]')].map(input => input.value.trim()).filter(Boolean);
  const descriptions = {};
  [...document.querySelectorAll('[data-desc-index]')].forEach((input, i) => { if (labels[i]) descriptions[labels[i]] = input.value.trim(); });
  state.task.name = document.getElementById('task-name').value.trim().replace(/[^A-Za-z0-9_.-]+/g, '_');
  state.task.domain = document.getElementById('task-domain').value.trim();
  state.task.description = document.getElementById('task-description').value.trim();
  state.task.policy = document.getElementById('task-policy').value.trim();
  state.task.decision.labels = labels;
  state.task.decision.class_descriptions = descriptions;
  return oldLabels;
}

async function startGeneration() {
  syncTaskFromPolicy();
  const same = document.getElementById('same-provider').checked;
  const benchmarkProvider = same ? state.config.provider : {
    base_url: document.getElementById('bench-endpoint').value.trim().replace(/\/$/, ''),
    model: document.getElementById('bench-model').value.trim(),
    api_key: document.getElementById('bench-key').value,
  };
  const payload = {
    task: state.task,
    specialization_provider: state.config.provider,
    benchmark_provider: benchmarkProvider,
    specialization_examples: Number(document.getElementById('count-spec').value),
    validation_examples: Number(document.getElementById('count-validation').value),
    hidden_examples: Number(document.getElementById('count-hidden').value),
    paraphrase_examples: Number(document.getElementById('count-robust').value),
    hard_examples: Number(document.getElementById('count-robust').value),
    seed: state.config.seed,
  };
  try {
    const job = await api('/api/studio/generate', {method: 'POST', body: JSON.stringify(payload)});
    await watchJob(job.id, 2, result => {
      state.draftId = result.draft_id; state.splits = result.splits; state.task = result.task; renderDatasetReview();
    });
  } catch (error) { toast(error.message, true); renderPolicy(); }
}

function renderDatasetReview() {
  const split = state.activeSplit;
  const rows = state.splits[split] || [];
  const binary = state.task.decision.labels.length === 2;
  state.runName ||= state.task.name.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
  app.innerHTML = `<div class="page">${stepper(2)}
    <div class="page-head compact"><div><div class="eyebrow">Human review</div><h1>Your data is ready.</h1><p class="lede">Edit any text or label if needed. Hidden tests remain separate from specialization data.</p></div></div>
    <section class="card">
      <div class="dataset-toolbar"><div class="tabs">${Object.entries(splitMeta).map(([key, meta]) => `<button class="tab ${key === split ? 'active' : ''}" data-split="${key}">${meta[0]} <span class="count">${state.splits[key]?.length || 0}</span></button>`).join('')}</div><button class="btn btn-secondary" id="add-example">＋ Add</button></div>
      <div class="review-note"><strong>${splitMeta[split][0]} :</strong> ${splitMeta[split][1]}</div>
      <div class="dataset-list">${rows.map((row, i) => exampleRow(row, i)).join('') || '<div class="empty">No examples in this split.</div>'}</div>
    </section>
    <section class="card">
      <div class="section-head"><div><h2>Methods to compare</h2><p>The best hidden-test result will be exported automatically.</p></div><span class="badge">Frozen Laya weights</span></div>
      <div class="field" style="margin-bottom:18px"><label for="run-name">Variant name</label><input id="run-name" value="${esc(state.runName)}" maxlength="120" placeholder="e.g. Customer sentiment — balanced"><span class="hint">This is the friendly name shown in your model library.</span></div>
      <div class="section-head"><div><h3>Vanilla checkpoint</h3><p>Select one model, or several to compare every vanilla and specialization on this exact benchmark.</p></div><span class="badge">Auto-download</span></div>
      <div class="method-grid" id="model-grid">${(state.modelCatalog || []).map((model, i) => `<label class="method-option"><input type="checkbox" data-base-model="${esc(model.model_id)}" data-backend="${esc(model.backend)}" ${i === 0 ? 'checked' : ''}><span><strong>${esc(model.title)}</strong><small>${esc(model.summary)}</small></span></label>`).join('')}</div>
      <label class="method-option" style="margin:14px 0 20px"><input type="checkbox" id="custom-model-toggle"><span><strong>Custom Hugging Face ID</strong><small>Use any compatible Laya checkpoint, pinned to its resolved Hub commit.</small></span></label>
      <div class="form-grid" id="custom-model-fields" style="display:none;margin-bottom:20px"><div class="field"><label for="custom-backend">Backend</label><select id="custom-backend"><option value="mlx">MLX</option><option value="pytorch">PyTorch</option></select></div><div class="field"><label for="custom-model">Hugging Face ID or local path</label><input id="custom-model" placeholder="owner/checkpoint"></div></div>
      <div class="method-grid">${methods.map(([id, title, help]) => {
        const disabled = (id === 'contrastive_vector' || id === 'activation_steering') && !binary;
        const checked = !disabled && ['baseline','prompt_only','nearest_prototype','multiclass_centroids','multi_vector_steering','activation_steering'].includes(id);
        return `<label class="method-option" style="${disabled ? 'opacity:.45' : ''}"><input type="checkbox" data-method="${id}" ${checked ? 'checked' : ''} ${disabled ? 'disabled' : ''}><span><strong>${title}</strong><small>${disabled ? 'Not applicable to this multiclass task' : help}</small></span></label>`;
      }).join('')}</div>
      <div class="form-actions"><button class="btn btn-ghost" id="back-policy">← Policy</button><button class="btn btn-primary btn-lg" id="run-experiment">Test selected models <span>→</span></button></div>
    </section>
  </div>`;
  bindDatasetEvents();
}

function exampleRow(row, index) {
  return `<div class="example-row" data-index="${index}">
    <input data-example-text value="${esc(row.input)}" aria-label="Example text ${index+1}">
    <select data-example-label aria-label="Example label ${index+1}">${state.task.decision.labels.map(label => `<option ${label === row.label ? 'selected' : ''}>${esc(label)}</option>`).join('')}</select>
    <button class="icon-btn" data-remove title="Delete" aria-label="Delete example">×</button>
  </div>`;
}

function syncDatasetRows() {
  document.querySelectorAll('.example-row').forEach(node => {
    const row = state.splits[state.activeSplit][Number(node.dataset.index)];
    row.input = node.querySelector('[data-example-text]').value.trim();
    row.label = node.querySelector('[data-example-label]').value;
    delete row.content_hash;
  });
}

function bindDatasetEvents() {
  document.getElementById('run-name').oninput = event => { state.runName = event.target.value; };
  document.querySelectorAll('[data-split]').forEach(button => button.onclick = () => { syncDatasetRows(); state.activeSplit = button.dataset.split; renderDatasetReview(); });
  document.querySelectorAll('[data-remove]').forEach(button => button.onclick = () => {
    syncDatasetRows(); const index = Number(button.closest('.example-row').dataset.index); state.splits[state.activeSplit].splice(index, 1); renderDatasetReview();
  });
  document.getElementById('add-example').onclick = () => {
    syncDatasetRows();
    const i = state.splits[state.activeSplit].length;
    state.splits[state.activeSplit].push({id:`${state.task.name}-${state.activeSplit}-manual-${Date.now()}-${i}`,input:'New example',label:state.task.decision.labels[0],split:state.activeSplit,task_name:state.task.name,domain:state.task.domain,tags:['manual'],metadata:{}});
    renderDatasetReview();
    document.querySelector('.dataset-list').scrollTop = 999999;
  };
  document.getElementById('back-policy').onclick = () => { syncDatasetRows(); renderPolicy(); };
  document.getElementById('run-experiment').onclick = startRun;
  document.getElementById('custom-model-toggle').onchange = event => {
    document.getElementById('custom-model-fields').style.display = event.target.checked ? 'grid' : 'none';
  };
}

async function startRun() {
  syncDatasetRows();
  const selected = [...document.querySelectorAll('[data-method]:checked')].map(x => x.dataset.method);
  if (!selected.includes('baseline')) selected.unshift('baseline');
  const basePayload = {
    name: (document.getElementById('run-name')?.value || state.runName || state.task.name).trim(),
    task: state.task,
    examples: Object.values(state.splits).flat(),
    methods: selected,
    seed: state.config?.seed || 0,
  };
  const selections = [...document.querySelectorAll('[data-base-model]:checked')].map(node => ({backend: node.dataset.backend, model: node.dataset.baseModel}));
  if (document.getElementById('custom-model-toggle')?.checked && document.getElementById('custom-model').value.trim()) {
    selections.push({backend: document.getElementById('custom-backend').value, model: document.getElementById('custom-model').value.trim()});
  }
  if (!selections.length) { toast('Select at least one vanilla checkpoint.', true); return; }
  try {
    state.modelResults = [];
    for (const selection of selections) {
      const payload = {...basePayload, ...selection, device: selection.backend === 'pytorch' ? 'mps' : null};
      const job = await api(`/api/studio/drafts/${state.draftId}/run`, {method:'POST',body:JSON.stringify(payload)});
      await watchJob(job.id, 3, result => { state.modelResults.push({...result, checkpoint: selection}); });
    }
    state.result = state.modelResults[0];
    renderResults();
  } catch (error) { toast(error.message, true); renderDatasetReview(); }
}

function renderResults() {
  const result = state.result;
  const best = result.best;
  const exportName = result.export_path.split('/').pop();
  app.innerHTML = `<div class="page">${stepper(3)}
    <div class="page-head compact"><div><div class="eyebrow">Experiment complete</div><h1>Your specialization is ready.</h1><p class="lede">Method selection used validation only. Hidden, paraphrase and hard scores are final measurements.</p></div></div>
    ${state.modelResults.length > 1 ? `<section class="card"><div class="section-head"><div><h2>Checkpoint comparison</h2><p>Checkpoint gains and same-checkpoint specialization gains are kept separate.</p></div></div><div class="ranking">${state.modelResults.map(item => { const baseline = item.ranking.find(row => row.strategy === 'baseline'); return `<div class="rank-row"><span class="rank-num">L</span><div class="rank-name"><strong>${esc(item.checkpoint.model)}</strong><small>${esc(item.checkpoint.backend)} · vanilla ${fmt(baseline?.accuracy)}</small></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2,(item.best.accuracy || 0)*100)}%"></div></div><strong>${fmt(item.best.accuracy)}</strong><span class="latency">${item.best.delta >= 0 ? '+' : ''}${((item.best.delta || 0)*100).toFixed(1)} pts</span></div>`; }).join('')}</div></section>` : ''}
    <section class="card winner">
      <div class="eyebrow">Best result</div><h2>${humanMethod(best.strategy)}</h2><p>${componentLabel(best.decision_component)}</p>
      <div class="winner-metrics"><div><strong>${fmt(best.accuracy)}</strong><small>Hidden accuracy</small></div><div><strong class="${best.delta >= 0 ? 'delta-up' : 'delta-down'}">${best.delta == null ? '—' : `${best.delta >= 0 ? '+' : ''}${(best.delta*100).toFixed(1)} pts`}</strong><small>vs original Laya</small></div><div><strong>${best.latency_ms?.toFixed(1) || '—'} ms</strong><small>Latency / example</small></div></div>
      <a class="btn btn-lime btn-lg" href="/api/studio/exports/${encodeURIComponent(exportName)}">Download specialized model ↓</a>
    </section>
    <section class="card">
      <div class="section-head"><div><h2>Full comparison</h2><p>One improvement is not enough: inspect robustness and latency too.</p></div><span class="badge">Run ${esc(result.run_id)}</span></div>
      <div class="ranking">${result.ranking.map((row, i) => `<div class="rank-row"><span class="rank-num">${i+1}</span><div class="rank-name"><strong>${humanMethod(row.strategy)}</strong><small>${componentLabel(row.decision_component)}</small></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2,(row.accuracy || 0)*100)}%"></div></div><strong>${fmt(row.accuracy)}</strong><span class="latency">${row.latency_ms?.toFixed(1) || '—'} ms</span></div>`).join('')}</div>
    </section>
    <div class="form-actions"><button class="btn btn-secondary" id="view-run">Inspect experiment</button><button class="btn btn-secondary" id="open-library">Open model library</button><button class="btn btn-primary" id="new-specialization">Create another specialization</button></div>
  </div>`;
  document.getElementById('view-run').onclick = () => { navigate('/runs'); setTimeout(() => openRun(result.run_id), 80); };
  document.getElementById('open-library').onclick = () => navigate(`/library?run=${encodeURIComponent(result.run_id)}`);
  document.getElementById('new-specialization').onclick = () => { Object.assign(state,{config:null,task:null,draftId:null,splits:null,result:null}); renderCreate(); };
}

function humanMethod(id) {
  return methods.find(method => method[0] === id)?.[1] || id.replaceAll('_', ' ');
}

function componentLabel(component) {
  return ({laya_head:'Final decision from the Laya head',embedding_classifier:'Embedding-space classifier',pairwise_laya_head:'Pairwise comparisons through the Laya head'})[component] || component;
}

function friendlyRunName(run) {
  return run.display_name || (run.primary_task || run.id).replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function hiddenScore(row) {
  return row.metrics?.splits?.hidden?.accuracy ?? row.metrics?.overall?.accuracy ?? 0;
}

async function renderLibrary() {
  app.innerHTML = `<div class="page-loading">Loading model library…</div>`;
  try {
    const runs = (await api('/api/runs')).filter(run => run.status === 'completed' && run.task_count === 1);
    app.innerHTML = `<div class="page">
      <div class="page-head compact"><div><div class="eyebrow">Your Laya variants</div><h1>Model library</h1><p class="lede">Name each specialization, choose its default method, and open it directly in the Playground.</p></div><button class="btn btn-primary" id="library-create">＋ New specialization</button></div>
      <div class="library-grid">${runs.map(run => `<article class="model-card" data-library-run="${esc(run.id)}">
        <div class="model-card-top"><span class="model-icon">L</span><span class="badge">${run.kept_count || 1} saved</span></div>
        <h2>${esc(friendlyRunName(run))}</h2><p>${esc(run.primary_task || 'Laya specialization').replaceAll('_',' ')}</p>
        <div class="default-method"><small>Default method</small><strong>${humanMethod(run.default_strategy || 'baseline')}</strong></div>
        <div class="model-meta"><span>${esc(run.backend)}</span><span>${run.result_count} tested methods</span></div>
        <div class="model-actions"><button class="btn btn-secondary" data-manage-run="${esc(run.id)}">Manage</button><button class="btn btn-primary" data-play-run="${esc(run.id)}" data-strategy="${esc(run.default_strategy || '')}">Open in Playground</button></div>
      </article>`).join('') || '<div class="card empty">No variants yet. Create your first specialization to start the library.</div>'}</div>
    </div>`;
    document.getElementById('library-create').onclick = () => navigate('/create');
    document.querySelectorAll('[data-manage-run]').forEach(button => button.onclick = () => navigate(`/library?run=${encodeURIComponent(button.dataset.manageRun)}`));
    document.querySelectorAll('[data-play-run]').forEach(button => button.onclick = () => navigate(`/playground?run=${encodeURIComponent(button.dataset.playRun)}&strategy=${encodeURIComponent(button.dataset.strategy)}`));
  } catch (error) { app.innerHTML = `<div class="page"><div class="error-panel">${esc(error.message)}</div></div>`; }
}

async function openLibraryRun(runId) {
  app.innerHTML = `<div class="page-loading">Opening variant…</div>`;
  try {
    const data = await api(`/api/library/runs/${encodeURIComponent(runId)}`);
    const runName = data.run.display_name || (data.tasks[0]?.name || runId).replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    const ranked = [...data.results].sort((a,b) => hiddenScore(b) - hiddenScore(a));
    const currentDefault = ranked.find(row => row.is_default) || ranked[0];
    app.innerHTML = `<div class="page">
      <div class="page-head compact"><div><button class="back-link" id="back-library">← Model library</button><div class="eyebrow">Specialization settings</div><h1>${esc(runName)}</h1><p class="lede">Choose which tested methods matter and which one should be used by default.</p></div><button class="btn btn-primary" id="detail-play">Open in Playground →</button></div>
      <section class="card library-header"><div><label for="library-name">Specialization name</label><div class="rename-row"><input id="library-name" value="${esc(runName)}" maxlength="120"><button class="btn btn-secondary" id="save-library-name">Save name</button></div><p>${esc(data.run.base_model)} · ${esc(data.run.backend)}</p></div></section>
      <section class="card method-panel"><div class="section-head"><div><h2>Tested methods</h2><p>The best hidden-test score is selected by default. Change it explicitly whenever another method is a better fit.</p></div><span class="badge">${ranked.length} methods</span></div>
          <div class="method-library">${ranked.map((row, index) => `<article class="library-method ${row.kept ? 'kept' : ''} ${row.is_default ? 'default' : ''}">
            <div class="method-score"><strong>${fmt(hiddenScore(row))}</strong><small>hidden accuracy</small></div>
            <div><h3>${humanMethod(row.strategy)}</h3><p>${componentLabel(row.decision_component)}</p></div>
            <div class="method-status">${row.is_default ? '<span class="badge">Default</span>' : index === 0 ? '<span class="badge">Best score</span>' : ''}</div>
            ${row.is_default ? '<span class="action-placeholder"></span>' : `<button class="btn btn-secondary" data-default-method="${esc(row.strategy)}" data-task="${esc(row.task_name)}">Set as default</button>`}
            ${row.is_default ? '<span class="action-placeholder"></span>' : `<button class="btn ${row.kept ? 'btn-danger' : 'btn-secondary'}" data-keep-method="${esc(row.strategy)}" data-task="${esc(row.task_name)}" data-kept="${row.kept}">${row.kept ? 'Remove' : 'Keep'}</button>`}
            <button class="btn btn-ghost" data-export-method="${esc(row.strategy)}" data-task="${esc(row.task_name)}">Export</button>
          </article>`).join('')}</div>
      </section>
    </div>`;
    document.getElementById('back-library').onclick = () => navigate('/library');
    document.getElementById('detail-play').onclick = () => navigate(`/playground?run=${encodeURIComponent(runId)}&strategy=${encodeURIComponent(currentDefault.strategy)}`);
    document.getElementById('save-library-name').onclick = async () => {
      const name = document.getElementById('library-name').value.trim();
      if (!name) return;
      await api(`/api/library/runs/${encodeURIComponent(runId)}/name`, {method:'POST',body:JSON.stringify({name})});
      toast('Variant name saved');
      document.querySelector('.page-head h1').textContent = name;
    };
    document.querySelectorAll('[data-default-method]').forEach(button => button.onclick = async () => {
      await api(`/api/library/runs/${encodeURIComponent(runId)}/default`, {method:'POST',body:JSON.stringify({task:button.dataset.task,strategy:button.dataset.defaultMethod})});
      toast(`${humanMethod(button.dataset.defaultMethod)} is now the default`);
      await openLibraryRun(runId);
    });
    document.querySelectorAll('[data-keep-method]').forEach(button => button.onclick = async () => {
      const kept = button.dataset.kept !== 'true';
      try {
        await api(`/api/library/runs/${encodeURIComponent(runId)}/keep`, {method:'POST',body:JSON.stringify({task:button.dataset.task,strategy:button.dataset.keepMethod,kept})});
        toast(kept ? 'Method kept' : 'Method removed');
        await openLibraryRun(runId);
      } catch (error) { toast(error.message, true); }
    });
    document.querySelectorAll('[data-export-method]').forEach(button => button.onclick = async () => {
      button.disabled = true; button.textContent = 'Preparing…';
      try {
        const result = await api(`/api/library/runs/${encodeURIComponent(runId)}/export`, {method:'POST',body:JSON.stringify({task:button.dataset.task,strategy:button.dataset.exportMethod,name:document.getElementById('library-name').value.trim()})});
        location.href = `/api/studio/exports/${encodeURIComponent(result.export_name)}`;
      } catch (error) { toast(error.message, true); button.disabled = false; button.textContent = 'Export'; }
    });
  } catch (error) { app.innerHTML = `<div class="page"><div class="error-panel">${esc(error.message)}</div></div>`; }
}

async function renderPlayground() {
  app.innerHTML = `<div class="page-loading">Preparing Playground…</div>`;
  try {
    const runs = (await api('/api/runs')).filter(run => run.status === 'completed' && run.task_count === 1);
    if (!runs.length) {
      app.innerHTML = `<div class="page"><div class="page-head compact"><div><div class="eyebrow">Try your specialization</div><h1>Playground</h1></div></div><div class="card empty">Create a specialization before opening the Playground.</div></div>`;
      return;
    }
    const params = new URLSearchParams(location.search);
    const selectedRun = runs.find(run => run.id === params.get('run')) || runs[0];
    const data = await api(`/api/library/runs/${encodeURIComponent(selectedRun.id)}`);
    const ranked = [...data.results].sort((a,b) => hiddenScore(b) - hiddenScore(a));
    const selectedMethod = ranked.find(row => row.strategy === params.get('strategy')) || ranked.find(row => row.is_default) || ranked[0];
    const examples = data.examples.filter(row => row.task_name === selectedMethod.task_name);
    const firstExample = examples[0];
    app.innerHTML = `<div class="page playground-page">
      <div class="page-head compact"><div><div class="eyebrow">Live experimentation</div><h1>Playground</h1><p class="lede">Change a benchmark example or write your own text. Scratchpad edits are never saved.</p></div><button class="btn btn-secondary" id="manage-play-model">Manage specialization</button></div>
      <section class="card playground-controls"><div class="field"><label for="play-run-select">Specialization</label><select id="play-run-select">${runs.map(run => `<option value="${esc(run.id)}" ${run.id === selectedRun.id ? 'selected' : ''}>${esc(friendlyRunName(run))}</option>`).join('')}</select></div><div class="field"><label for="play-method">Method</label><select id="play-method">${ranked.map(row => `<option value="${esc(row.strategy)}" ${row.strategy === selectedMethod.strategy ? 'selected' : ''}>${humanMethod(row.strategy)}${row.is_default ? ' — Default' : ''}</option>`).join('')}</select></div><div class="default-callout"><span>Using</span><strong>${humanMethod(selectedMethod.strategy)}</strong><small>${selectedMethod.is_default ? 'Default for this specialization' : 'Temporary selection'}</small></div></section>
      <section class="card playground-workspace">
        <div class="play-editor"><div class="field"><label for="play-example">Start from a benchmark example</label><select id="play-example">${examples.map((row,i) => `<option value="${i}">${esc(splitMeta[row.split]?.[0] || row.split)} · ${esc(row.label)} · ${esc(row.input.slice(0,80))}</option>`).join('')}</select></div><div class="field"><label for="play-text">Input to Laya</label><textarea id="play-text">${esc(firstExample?.input || '')}</textarea><span class="hint" id="play-expected">Expected in benchmark: ${esc(firstExample?.label || '—')}</span></div><button class="btn btn-primary btn-lg" id="play-run">Run this example →</button></div>
        <div id="play-result" class="play-result empty"><div><strong>No prediction yet</strong><p>Edit the input if you want, then run it through Laya.</p></div></div>
      </section>
    </div>`;
    document.getElementById('manage-play-model').onclick = () => navigate(`/library?run=${encodeURIComponent(selectedRun.id)}`);
    document.getElementById('play-run-select').onchange = event => navigate(`/playground?run=${encodeURIComponent(event.target.value)}`);
    document.getElementById('play-method').onchange = event => navigate(`/playground?run=${encodeURIComponent(selectedRun.id)}&strategy=${encodeURIComponent(event.target.value)}`);
    document.getElementById('play-example').onchange = event => {
      const row = examples[Number(event.target.value)];
      document.getElementById('play-text').value = row?.input || '';
      document.getElementById('play-expected').textContent = `Expected in benchmark: ${row?.label || '—'}`;
      document.getElementById('play-result').className = 'play-result empty';
      document.getElementById('play-result').innerHTML = '<div><strong>No prediction yet</strong><p>Edit the input if you want, then run it through Laya.</p></div>';
    };
    document.getElementById('play-run').onclick = async () => {
      const button = document.getElementById('play-run');
      const text = document.getElementById('play-text').value.trim();
      if (!text) return;
      button.disabled = true; button.textContent = 'Laya is deciding…';
      try {
        const result = await api(`/api/library/runs/${encodeURIComponent(selectedRun.id)}/predict`, {method:'POST',body:JSON.stringify({task:selectedMethod.task_name,strategy:selectedMethod.strategy,text})});
        document.getElementById('play-result').className = 'play-result';
        document.getElementById('play-result').innerHTML = predictionMarkup(result);
      } catch (error) { toast(error.message, true); }
      finally { button.disabled = false; button.textContent = 'Run this example →'; }
    };
  } catch (error) { app.innerHTML = `<div class="page"><div class="error-panel">${esc(error.message)}</div></div>`; }
}

function predictionMarkup(result) {
  return `<div class="prediction-head"><span>Prediction</span><strong>${esc(result.label)}</strong></div>${Object.entries(result.probabilities || {}).sort((a,b)=>b[1]-a[1]).map(([label,value]) => `<div class="prob-row"><span>${esc(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${value*100}%"></div></div><strong>${fmt(value)}</strong></div>`).join('')}`;
}

async function renderRuns() {
  app.innerHTML = `<div class="page-loading">Loading experiments…</div>`;
  try {
    const runs = await api('/api/runs');
    app.innerHTML = `<div class="page"><div class="page-head compact"><div><div class="eyebrow">Scientific history</div><h1>Experiments</h1><p class="lede">Review models, domains, metrics, and mistakes for every run.</p></div><button class="btn btn-primary" id="runs-create">＋ New specialization</button></div>
      <div class="run-list">${runs.map(run => `<article class="run-card" data-run="${esc(run.id)}"><div><h3>${esc(friendlyRunName(run))}</h3><div class="run-meta"><span>${esc(run.id)}</span><span>${esc(run.base_model)}</span><span>seed ${run.seed}</span></div></div><span class="badge ${run.backend === 'fake' ? 'fake' : ''}">${esc(run.backend)}</span><div><strong>${run.result_count}</strong><small style="display:block">results</small></div></article>`).join('') || '<div class="card empty">No experiments yet.</div>'}</div>
      <div id="run-detail"></div></div>`;
    document.getElementById('runs-create').onclick = () => navigate('/create');
    document.querySelectorAll('[data-run]').forEach(card => card.onclick = () => openRun(card.dataset.run));
  } catch (error) { app.innerHTML = `<div class="page"><div class="error-panel">${esc(error.message)}</div></div>`; }
}

async function openRun(runId) {
  const target = document.getElementById('run-detail');
  if (!target) return;
  target.innerHTML = `<div class="card progress-card"><div class="orb">◫</div><p>Loading run…</p></div>`;
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}`);
    const baseline = Object.fromEntries(data.results.filter(x => x.strategy === 'baseline').map(x => [x.task_name, x.metrics.overall.accuracy]));
    target.innerHTML = `<section class="card" style="margin-top:20px"><div class="section-head"><div><h2>${esc(runId)}</h2><p>${esc(data.run.base_model)} · ${esc(data.run.backend)}</p></div><button class="icon-btn" id="close-run">×</button></div>
      <div class="ranking">${data.results.map((row,i) => { const score=row.metrics.overall.accuracy, delta=score-(baseline[row.task_name] ?? score); return `<div class="rank-row"><span class="rank-num">${i+1}</span><div class="rank-name"><strong>${esc(row.task_name)} · ${humanMethod(row.strategy)}</strong><small>${componentLabel(row.decision_component)}</small></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2,score*100)}%"></div></div><strong>${fmt(score)}</strong><span class="${delta>=0?'delta-up':'delta-down'}">${delta>=0?'+':''}${(delta*100).toFixed(1)}</span></div>`; }).join('')}</div></section>`;
    document.getElementById('close-run').onclick = () => target.innerHTML = '';
    target.scrollIntoView({behavior:'smooth',block:'start'});
  } catch (error) { target.innerHTML = `<div class="error-panel">${esc(error.message)}</div>`; }
}

route();
