const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  originalCode: '',
  filename: 'code.py',
  lastAnalysis: null,
  proposedCode: ''
};

function setStatus(text) {
  const el = $('#status');
  if (el) el.textContent = text;
}

function escapeHtml(s) {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function renderCode(id, code, lang = 'python') {
  const pre = $(id);
  if (!pre) return;
  const codeEl = pre.querySelector('code') || document.createElement('code');
  codeEl.className = `language-${lang}`;
  codeEl.textContent = code || '';
  if (!pre.contains(codeEl)) pre.appendChild(codeEl);
  if (window.Prism && window.Prism.highlightElement) {
    Prism.highlightElement(codeEl);
  }
}

function renderDiagnostics(items) {
  const wrap = $('#diagnostics');
  if (!wrap) return;
  wrap.innerHTML = '';
  if (!items || !items.length) {
    wrap.innerHTML = '<div class="diag-item"><span class="badge info">info</span><div>No issues found.</div></div>';
    return;
  }
  for (const d of items) {
    const level = (d.type || 'info').toLowerCase();
    const badge = `<span class="badge ${level}">${level}</span>`;
    const where = `<div class="where">line ${d.line || 1}, col ${d.column || 0}${d.symbol ? ` · ${d.symbol}` : ''}</div>`;
    const message = `<div>${escapeHtml(d.message || '')}</div>`;
    const item = document.createElement('div');
    item.className = 'diag-item';
    item.innerHTML = `${badge}<div>${message}${where}</div>`;
    wrap.appendChild(item);
  }
}

function activateTeachingStep(step) {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('teach-step-' + i);
    if (el) el.classList.remove('active');
  }
  const activeEl = document.getElementById('teach-step-' + step);
  if (activeEl) activeEl.classList.add('active');
}

async function analyze() {
  const code = $('#code').value;
  const language = 'python';
  if (!code.trim()) {
    setStatus('Please paste some code.');
    return;
  }
  activateTeachingStep(1); // Step 1: Code uploaded
  setStatus('Analyzing...');
  try {
    activateTeachingStep(2); // Step 2: Analyzing
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, language })
    });
    const ct = res.headers.get('content-type') || '';
    const text = await res.text();
    if (!ct.includes('application/json')) {
      throw new Error(`Unexpected response: ${text.slice(0, 200)}`);
    }
    const data = JSON.parse(text);
    if (!data.success) throw new Error(data.error || 'Unknown error');

    state.originalCode = data.original_code || '';
    state.lastAnalysis = data;
    state.proposedCode = data.proposed_code || '';

    renderCode('#original', state.originalCode, data.language || 'python');
    renderDiagnostics(data.diagnostics || []);
    renderSummary(data.summary || {});
    renderMetrics(data.metrics || {});
    renderSuggestions(data.suggestions || []);
    const wrap = $('#proposed-wrap');
    if (wrap) wrap.style.display = state.proposedCode ? '' : 'none';
    if (state.proposedCode) renderCode('#proposed', state.proposedCode, 'python');
    setStatus(`Done · ${data.language}`);
    activateTeachingStep(4); // Step 4: AI explains (after analysis)
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

function downloadReport() {
  const data = state.lastAnalysis || {};
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'analysis_report.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function setTheme(dark) {
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
}

function toggleTheme() {
  const darkNow = document.documentElement.dataset.theme !== 'dark';
  setTheme(darkNow);
  localStorage.setItem('theme', darkNow ? 'dark' : 'light');
}

function handleFile(file) {
  if (!file) return;
  const name = file.name || 'code.py';
  const ext = name.split('.').pop().toLowerCase();
  state.filename = `code.${ext || 'py'}`;
  const reader = new FileReader();
  reader.onload = () => {
    $('#code').value = reader.result || '';
  };
  reader.readAsText(file);
}

function renderSummary(summary) {
  const el = $('#summary');
  if (!el) return;
  const { errors = 0, warnings = 0, conventions = 0, refactors = 0 } = summary || {};
  el.innerHTML = `<div class="summary-grid">
    <div><span class="badge error">errors</span> ${errors}</div>
    <div><span class="badge warning">warnings</span> ${warnings}</div>
    <div><span class="badge info">conventions</span> ${conventions}</div>
    <div><span class="badge info">refactors</span> ${refactors}</div>
  </div>`;
}

function renderMetrics(metrics) {
  const el = $('#metrics');
  if (!el) return;
  const mi = metrics?.maintainability_index;
  const cyclo = metrics?.cyclomatic || {};
  const avg = cyclo.average;
  const max = cyclo.max;
  const functions = Array.isArray(cyclo.functions) ? cyclo.functions : [];
  const list = functions.slice(0, 10).map(f => `<li>${escapeHtml(f.name || '<fn>')} · C=${f.complexity} · line ${f.line}</li>`).join('');
  el.innerHTML = `
    <div class="metrics-grid">
      <div>Maintainability Index: <strong>${mi ?? 'n/a'}</strong></div>
      <div>Cyclomatic Avg: <strong>${avg ?? 'n/a'}</strong></div>
      <div>Cyclomatic Max: <strong>${max ?? 'n/a'}</strong></div>
    </div>
    <details ${functions.length ? 'open' : ''}><summary>Top functions by complexity</summary><ul>${list || '<li>n/a</li>'}</ul></details>
  `;
}

function renderSuggestions(items) {
  const wrap = $('#suggestions');
  if (!wrap) return;
  wrap.innerHTML = '';
  if (!items.length) {
    wrap.innerHTML = '<div class="diag-item"><span class="badge info">info</span><div>No suggestions.</div></div>';
    return;
  }
  for (const s of items) {
    const item = document.createElement('div');
    item.className = 'diag-item';
    const where = s.line ? `<div class="where">line ${s.line}</div>` : '';
    const metaPieces = [];
    if (s.category) metaPieces.push(escapeHtml(String(s.category)));
    if (typeof s.confidence === 'number') metaPieces.push(`conf ${(Math.round(s.confidence * 100))}%`);
    const meta = metaPieces.length ? `<div class="where">${metaPieces.join(' · ')}</div>` : '';
    item.innerHTML = `<span class="badge info">suggestion</span><div><div><strong>${escapeHtml(s.title || 'Suggestion')}</strong></div><div>${escapeHtml(s.detail || '')}</div>${where}${meta}</div>`;
    wrap.appendChild(item);
  }
}

async function runCode() {
  const code = state.originalCode || $('#code').value || '';
  const language = 'python';
  if (!code.trim()) { setStatus('Paste code first.'); return; }
  activateTeachingStep(3); // Step 3: Executing code
  setStatus('Running code...');
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, language })
    });
    const ct = res.headers.get('content-type') || '';
    const text = await res.text();
    if (!ct.includes('application/json')) {
      throw new Error(`Unexpected response: ${text.slice(0, 200)}`);
    }
    const data = JSON.parse(text);
    if (!data.success) throw new Error(data.error || 'Run failed');
    $('#run-stdout').textContent = data.stdout || '';
    $('#run-stderr').textContent = data.stderr || '';
    $('#run-exit').textContent = `exit ${data.exit_code}`;
    $('#run-time').textContent = `${data.runtime_ms} ms`;
    const wrap = $('#run-wrap');
    if (wrap) wrap.style.display = '';
    setStatus('Run complete.');
    // Show explanation and activate step 5
    if (typeof showRunExplanation === 'function' && data.explanation) {
      showRunExplanation(data.explanation);
    } else {
      activateTeachingStep(5);
    }
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

function copyProposed() {
  if (!state.proposedCode) { setStatus('No proposed code'); return; }
  navigator.clipboard.writeText(state.proposedCode);
  setStatus('Proposed code copied.');
}

function init() {
  // Analyze button
  const analyzeBtn = document.getElementById('analyze');
  if (analyzeBtn) analyzeBtn.onclick = analyze;

  // Run code button
  const runBtn = document.getElementById('run-code');
  if (runBtn) runBtn.onclick = runCode;

  // Clear button
  const clearBtn = document.getElementById('clear');
  if (clearBtn) clearBtn.onclick = function() {
    const codeArea = document.getElementById('code');
    if (codeArea) codeArea.value = '';
    // Clear results
    const original = document.getElementById('original');
    if (original) original.querySelector('code').textContent = '';
    const proposed = document.getElementById('proposed');
    if (proposed) proposed.querySelector('code').textContent = '';
    const runStdout = document.getElementById('run-stdout');
    if (runStdout) runStdout.textContent = '';
    const runStderr = document.getElementById('run-stderr');
    if (runStderr) runStderr.textContent = '';
    const runExplanation = document.getElementById('run-explanation');
    if (runExplanation) runExplanation.textContent = '';
  };

  // Download report button
  const downloadBtn = document.getElementById('download-report');
  if (downloadBtn) downloadBtn.onclick = downloadReport;

  // Copy proposed code button
  const copyBtn = document.getElementById('copy-proposed');
  if (copyBtn) copyBtn.onclick = copyProposed;

  // File upload
  const fileInput = document.getElementById('file-input');
  if (fileInput) {
    fileInput.onchange = function(e) {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    };
  }
}

document.addEventListener('DOMContentLoaded', init);

function showRunExplanation(explanation) {
    const el = document.getElementById('run-explanation');
    el.innerHTML = marked.parse(explanation || '');
    el.classList.add('animate__animated', 'animate__fadeIn');
    activateTeachingStep(5);
}
