#!/usr/bin/env python3
"""
SAM3 Config Editor — visual tool for editing config.json classes.

Standalone single-file HTTP server with embedded UI. Zero external dependencies.
Draw bounding boxes on reference images, manage text/image/both prompt types.

Usage:
    python3 config_editor.py --config config.json [--port 8080]
"""

import json
import argparse
import mimetypes
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# HTML / CSS / JS — single embedded page
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAM3 Config Editor</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #1a1a2e; --bg2: #16213e; --bg3: #0f3460;
  --fg: #e0e0e0; --fg2: #a0a0b0; --accent: #e94560;
  --accent2: #0f3460; --green: #4ecca3; --red: #e94560;
  --border: #2a2a4a; --input-bg: #0d1b2a; --radius: 6px;
}
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--fg); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
button { cursor: pointer; border: none; border-radius: var(--radius); padding: 6px 14px; font-size: 13px; transition: background 0.15s; }
input, select { background: var(--input-bg); color: var(--fg); border: 1px solid var(--border); border-radius: var(--radius); padding: 6px 10px; font-size: 13px; outline: none; }
input:focus, select:focus { border-color: var(--green); }

/* Header */
.header { display: flex; align-items: center; justify-content: space-between; padding: 10px 20px; background: var(--bg2); border-bottom: 1px solid var(--border); flex-shrink: 0; }
.header h1 { font-size: 16px; font-weight: 600; color: var(--green); }
.header-actions { display: flex; gap: 10px; align-items: center; }
.btn-save { background: var(--green); color: #000; font-weight: 600; padding: 8px 20px; }
.btn-save:hover { background: #3db892; }
.btn-save.dirty { animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
.save-status { font-size: 12px; color: var(--fg2); }

/* Layout */
.main { display: flex; flex: 1; overflow: hidden; }

/* Sidebar */
.sidebar { width: 220px; min-width: 220px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
.sidebar-header { padding: 12px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.sidebar-header span { font-size: 13px; font-weight: 600; color: var(--fg2); text-transform: uppercase; letter-spacing: 0.5px; }
.btn-add { background: var(--green); color: #000; font-size: 18px; width: 28px; height: 28px; padding: 0; display: flex; align-items: center; justify-content: center; border-radius: 50%; }
.btn-add:hover { background: #3db892; }
.class-list { flex: 1; overflow-y: auto; padding: 6px 0; }
.class-item { display: flex; align-items: center; padding: 8px 12px; cursor: pointer; border-left: 3px solid transparent; transition: all 0.1s; }
.class-item:hover { background: rgba(255,255,255,0.04); }
.class-item.active { background: rgba(78,204,163,0.1); border-left-color: var(--green); }
.class-item .dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 10px; flex-shrink: 0; }
.class-item .dot.text { background: #5dade2; }
.class-item .dot.image { background: #f39c12; }
.class-item .dot.both { background: #9b59b6; }
.class-item .name { flex: 1; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.class-item .btn-del { background: none; color: var(--fg2); font-size: 14px; padding: 2px 6px; opacity: 0; transition: opacity 0.1s; }
.class-item:hover .btn-del { opacity: 1; }
.class-item .btn-del:hover { color: var(--red); }
.sidebar-footer { padding: 12px; border-top: 1px solid var(--border); }
.sidebar-footer label { font-size: 12px; color: var(--fg2); display: block; margin-bottom: 4px; }
.sidebar-footer input { width: 100%; }
.class-count { font-size: 11px; color: var(--fg2); margin-top: 6px; }

/* Editor */
.editor { flex: 1; overflow-y: auto; padding: 24px 32px; }
.editor-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--fg2); font-size: 14px; }
.field-group { margin-bottom: 18px; }
.field-group label { display: block; font-size: 12px; color: var(--fg2); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.3px; }
.field-group input, .field-group select { width: 100%; max-width: 400px; }
.field-group .name-input { font-size: 18px; font-weight: 600; background: none; border: none; border-bottom: 2px solid var(--border); border-radius: 0; padding: 4px 0; max-width: 400px; }
.field-group .name-input:focus { border-bottom-color: var(--green); }

/* References section */
.refs-section { margin-top: 24px; }
.refs-section h3 { font-size: 14px; color: var(--fg2); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px; }
.ref-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 16px; overflow: hidden; }
.ref-header { display: flex; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--border); gap: 10px; }
.ref-header select { flex: 1; }
.ref-header .btn-del-ref { background: none; color: var(--fg2); font-size: 16px; padding: 2px 8px; }
.ref-header .btn-del-ref:hover { color: var(--red); }
.ref-body { display: flex; gap: 16px; padding: 14px; }
.canvas-wrap { position: relative; flex-shrink: 0; background: #000; border-radius: var(--radius); overflow: hidden; }
.canvas-wrap canvas { display: block; cursor: crosshair; }
.box-list { flex: 1; min-width: 200px; max-height: 360px; overflow-y: auto; }
.box-item { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: var(--radius); margin-bottom: 4px; font-size: 12px; font-family: 'Cascadia Code', 'Fira Code', monospace; background: var(--input-bg); }
.box-item .coords { flex: 1; color: var(--fg2); }
.box-item .lbl-toggle { padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
.box-item .lbl-toggle.pos { background: rgba(78,204,163,0.2); color: var(--green); }
.box-item .lbl-toggle.neg { background: rgba(233,69,96,0.2); color: var(--red); }
.box-item .btn-del-box { background: none; color: var(--fg2); padding: 2px 6px; font-size: 13px; }
.box-item .btn-del-box:hover { color: var(--red); }
.btn-add-ref { background: var(--bg3); color: var(--fg); padding: 8px 16px; margin-top: 4px; }
.btn-add-ref:hover { background: #1a4a7a; }
.box-hint { font-size: 11px; color: var(--fg2); margin-top: 6px; padding: 0 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a5a; }

/* Toast */
.toast { position: fixed; bottom: 20px; right: 20px; padding: 12px 20px; border-radius: var(--radius); font-size: 13px; z-index: 999; transition: opacity 0.3s; }
.toast.success { background: var(--green); color: #000; }
.toast.error { background: var(--red); color: #fff; }
.toast.hidden { opacity: 0; pointer-events: none; }
</style>
</head>
<body>

<div class="header">
  <h1>SAM3 Config Editor</h1>
  <div class="header-actions">
    <span class="save-status" id="saveStatus"></span>
    <button class="btn-save" id="btnSave" onclick="saveConfig()">Save</button>
  </div>
</div>

<div class="main">
  <div class="sidebar">
    <div class="sidebar-header">
      <span>Classes</span>
      <button class="btn-add" onclick="addClass()" title="Add class">+</button>
    </div>
    <div class="class-list" id="classList"></div>
    <div class="sidebar-footer">
      <label>Confidence threshold</label>
      <input type="number" id="confInput" min="0" max="1" step="0.05" value="0.3"
             onchange="config.confidence=parseFloat(this.value); markDirty()">
      <div class="class-count" id="classCount"></div>
    </div>
  </div>
  <div class="editor" id="editor">
    <div class="editor-empty">Select or add a class to begin editing</div>
  </div>
</div>

<div class="toast hidden" id="toast"></div>

<script>
// ─── State ───────────────────────────────────────────────────────────
let config = { engines: "engines", tokenizer: "engines/tokenizer.json", features: "features", confidence: 0.3, classes: [] };
let refImages = [];       // available reference image paths
let selectedIdx = -1;     // selected class index
let dirty = false;
let canvasStates = {};    // per-ref canvas state: { img, drawing, startX, startY }

const MAX_CLASSES = 8;
const MAX_BOXES = 20;
const CANVAS_MAX_W = 520;
const CANVAS_MAX_H = 360;

// ─── Init ────────────────────────────────────────────────────────────
async function init() {
  const [cfgResp, refResp] = await Promise.all([
    fetch('/api/config'), fetch('/api/references')
  ]);
  config = await cfgResp.json();
  refImages = await refResp.json();
  document.getElementById('confInput').value = config.confidence ?? 0.3;
  renderClassList();
  if (config.classes.length > 0) selectClass(0);
}

// ─── Toast ───────────────────────────────────────────────────────────
function toast(msg, type='success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast ' + type;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add('hidden'), 2500);
}

// ─── Dirty tracking ─────────────────────────────────────────────────
function markDirty() {
  dirty = true;
  document.getElementById('btnSave').classList.add('dirty');
  document.getElementById('saveStatus').textContent = 'Unsaved changes';
}

// ─── Save ────────────────────────────────────────────────────────────
async function saveConfig() {
  try {
    const resp = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config, null, 2)
    });
    if (!resp.ok) throw new Error(await resp.text());
    dirty = false;
    document.getElementById('btnSave').classList.remove('dirty');
    document.getElementById('saveStatus').textContent = 'Saved';
    toast('Config saved');
    setTimeout(() => {
      if (!dirty) document.getElementById('saveStatus').textContent = '';
    }, 3000);
  } catch (e) {
    toast('Save failed: ' + e.message, 'error');
  }
}

// ─── Class list ──────────────────────────────────────────────────────
function renderClassList() {
  const el = document.getElementById('classList');
  el.innerHTML = config.classes.map((c, i) => `
    <div class="class-item ${i === selectedIdx ? 'active' : ''}" onclick="selectClass(${i})">
      <div class="dot ${c.prompt_type}"></div>
      <span class="name">${esc(c.name)}</span>
      <button class="btn-del" onclick="event.stopPropagation(); deleteClass(${i})" title="Delete">&times;</button>
    </div>
  `).join('');
  document.getElementById('classCount').textContent =
    config.classes.length + '/' + MAX_CLASSES + ' classes';
}

function addClass() {
  if (config.classes.length >= MAX_CLASSES) {
    toast('Max ' + MAX_CLASSES + ' classes', 'error');
    return;
  }
  let name = 'new_class';
  let n = 1;
  const names = new Set(config.classes.map(c => c.name));
  while (names.has(name)) name = 'new_class_' + (n++);
  config.classes.push({ name, prompt_type: 'text', text: name });
  markDirty();
  selectClass(config.classes.length - 1);
}

function deleteClass(idx) {
  if (!confirm('Delete class "' + config.classes[idx].name + '"?')) return;
  config.classes.splice(idx, 1);
  markDirty();
  if (selectedIdx >= config.classes.length) selectedIdx = config.classes.length - 1;
  renderClassList();
  if (selectedIdx >= 0) selectClass(selectedIdx);
  else {
    selectedIdx = -1;
    document.getElementById('editor').innerHTML = '<div class="editor-empty">Select or add a class to begin editing</div>';
  }
}

function selectClass(idx) {
  selectedIdx = idx;
  renderClassList();
  renderEditor();
}

// ─── Editor ──────────────────────────────────────────────────────────
function renderEditor() {
  if (selectedIdx < 0) return;
  const c = config.classes[selectedIdx];
  const showText = c.prompt_type === 'text' || c.prompt_type === 'both';
  const showRefs = c.prompt_type === 'image' || c.prompt_type === 'both';

  let html = `
    <div class="field-group">
      <label>Name</label>
      <input class="name-input" value="${esc(c.name)}" onchange="updateName(this.value)">
    </div>
    <div class="field-group">
      <label>Prompt type</label>
      <select onchange="updateType(this.value)">
        <option value="text" ${c.prompt_type==='text'?'selected':''}>text</option>
        <option value="image" ${c.prompt_type==='image'?'selected':''}>image</option>
        <option value="both" ${c.prompt_type==='both'?'selected':''}>both</option>
      </select>
    </div>`;

  if (showText) {
    html += `
    <div class="field-group">
      <label>Text prompt</label>
      <input value="${esc(c.text || '')}" onchange="updateField('text', this.value)" placeholder="e.g. person">
    </div>`;
  }

  if (showRefs) {
    if (!c.references) c.references = [];
    html += `<div class="refs-section"><h3>References (${c.references.length})</h3>`;
    c.references.forEach((ref, ri) => {
      html += renderRefCard(ri, ref);
    });
    html += `<button class="btn-add-ref" onclick="addReference()">+ Add Reference</button></div>`;
  }

  document.getElementById('editor').innerHTML = html;

  // Load canvas images
  if (showRefs && c.references) {
    c.references.forEach((ref, ri) => {
      if (ref.image) loadCanvasImage(ri);
    });
  }
}

function renderRefCard(ri, ref) {
  const opts = refImages.map(p =>
    `<option value="${esc(p)}" ${p===ref.image?'selected':''}>${esc(p)}</option>`
  ).join('');

  let boxesHtml = '';
  if (ref.boxes && ref.boxes.length > 0) {
    ref.boxes.forEach((box, bi) => {
      const lbl = (ref.labels && ref.labels[bi] !== undefined) ? ref.labels[bi] : 1;
      const isPos = lbl === 1;
      boxesHtml += `
        <div class="box-item">
          <span class="coords">[${box.map(v=>v.toFixed(3)).join(', ')}]</span>
          <button class="lbl-toggle ${isPos?'pos':'neg'}" onclick="toggleLabel(${ri},${bi})">${isPos?'pos':'neg'}</button>
          <button class="btn-del-box" onclick="deleteBox(${ri},${bi})" title="Delete box">&times;</button>
        </div>`;
    });
  }

  return `
    <div class="ref-card" id="ref-${ri}">
      <div class="ref-header">
        <select onchange="updateRefImage(${ri}, this.value)">
          <option value="">-- select image --</option>
          ${opts}
        </select>
        <button class="btn-del-ref" onclick="deleteReference(${ri})" title="Delete reference">&times;</button>
      </div>
      <div class="ref-body">
        <div class="canvas-wrap">
          <canvas id="canvas-${ri}" width="${CANVAS_MAX_W}" height="${CANVAS_MAX_H}"></canvas>
        </div>
        <div>
          <div class="box-list" id="boxes-${ri}">${boxesHtml}</div>
          <div class="box-hint">Click and drag on image to draw a box. Max ${MAX_BOXES} boxes.</div>
        </div>
      </div>
    </div>`;
}

// ─── Field updates ───────────────────────────────────────────────────
function updateName(val) {
  val = val.trim();
  if (!val) return;
  const names = config.classes.map((c,i) => i !== selectedIdx ? c.name : null).filter(Boolean);
  if (names.includes(val)) { toast('Duplicate name', 'error'); renderEditor(); return; }
  config.classes[selectedIdx].name = val;
  markDirty();
  renderClassList();
}

function updateType(val) {
  const c = config.classes[selectedIdx];
  c.prompt_type = val;
  // Ensure required fields
  if ((val === 'text' || val === 'both') && !c.text) c.text = c.name;
  if ((val === 'image' || val === 'both') && !c.references) c.references = [];
  // Clean up unnecessary fields
  if (val === 'text') delete c.references;
  if (val === 'image') delete c.text;
  markDirty();
  renderEditor();
}

function updateField(field, val) {
  config.classes[selectedIdx][field] = val;
  markDirty();
}

// ─── References ──────────────────────────────────────────────────────
function addReference() {
  const c = config.classes[selectedIdx];
  if (!c.references) c.references = [];
  c.references.push({ image: '', boxes: [], labels: [] });
  markDirty();
  renderEditor();
}

function deleteReference(ri) {
  config.classes[selectedIdx].references.splice(ri, 1);
  markDirty();
  renderEditor();
}

function updateRefImage(ri, path) {
  config.classes[selectedIdx].references[ri].image = path;
  markDirty();
  if (path) loadCanvasImage(ri);
  else redrawCanvas(ri, null);
}

// ─── Canvas / bbox drawing ───────────────────────────────────────────
function loadCanvasImage(ri) {
  const ref = config.classes[selectedIdx].references[ri];
  if (!ref || !ref.image) return;
  const img = new Image();
  img.onload = () => {
    canvasStates[ri] = { img, naturalW: img.naturalWidth, naturalH: img.naturalHeight };
    setupCanvas(ri);
    redrawCanvas(ri, img);
  };
  img.onerror = () => {
    canvasStates[ri] = null;
    const cv = document.getElementById('canvas-' + ri);
    if (cv) { const ctx = cv.getContext('2d'); ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0,0,cv.width,cv.height); ctx.fillStyle = '#e94560'; ctx.font = '14px sans-serif'; ctx.fillText('Failed to load image', 20, 30); }
  };
  img.src = '/file/' + encodeURIComponent(ref.image);
}

function setupCanvas(ri) {
  const cv = document.getElementById('canvas-' + ri);
  if (!cv || !canvasStates[ri]) return;
  const { naturalW, naturalH } = canvasStates[ri];
  const scale = Math.min(CANVAS_MAX_W / naturalW, CANVAS_MAX_H / naturalH, 1);
  const dw = Math.round(naturalW * scale);
  const dh = Math.round(naturalH * scale);
  cv.width = dw;
  cv.height = dh;
  canvasStates[ri].dispW = dw;
  canvasStates[ri].dispH = dh;

  // Drawing state
  let drawing = false, sx = 0, sy = 0;

  cv.onmousedown = (e) => {
    const r = cv.getBoundingClientRect();
    sx = e.clientX - r.left;
    sy = e.clientY - r.top;
    drawing = true;
  };
  cv.onmousemove = (e) => {
    if (!drawing) return;
    const r = cv.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    redrawCanvas(ri, canvasStates[ri].img);
    const ctx = cv.getContext('2d');
    ctx.strokeStyle = 'rgba(78,204,163,0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(sx, sy, mx - sx, my - sy);
    ctx.setLineDash([]);
  };
  cv.onmouseup = (e) => {
    if (!drawing) return;
    drawing = false;
    const r = cv.getBoundingClientRect();
    const ex = e.clientX - r.left;
    const ey = e.clientY - r.top;
    // Min size check (5px)
    if (Math.abs(ex - sx) < 5 || Math.abs(ey - sy) < 5) {
      redrawCanvas(ri, canvasStates[ri].img);
      return;
    }
    addBox(ri, sx, sy, ex, ey);
  };
  cv.onmouseleave = (e) => {
    if (drawing) {
      drawing = false;
      redrawCanvas(ri, canvasStates[ri].img);
    }
  };
}

function addBox(ri, px1, py1, px2, py2) {
  const ref = config.classes[selectedIdx].references[ri];
  if (!ref) return;
  if (ref.boxes.length >= MAX_BOXES) { toast('Max ' + MAX_BOXES + ' boxes per reference', 'error'); return; }
  const st = canvasStates[ri];
  if (!st) return;

  // Normalise to 0-1, clamp
  const x1 = Math.max(0, Math.min(px1, px2)) / st.dispW;
  const y1 = Math.max(0, Math.min(py1, py2)) / st.dispH;
  const x2 = Math.min(1, Math.max(px1, px2) / st.dispW);
  const y2 = Math.min(1, Math.max(py1, py2) / st.dispH);

  // cxcywh
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const w  = x2 - x1;
  const h  = y2 - y1;

  ref.boxes.push([round4(cx), round4(cy), round4(w), round4(h)]);
  if (!ref.labels) ref.labels = [];
  ref.labels.push(1);
  markDirty();
  renderEditor();
}

function deleteBox(ri, bi) {
  const ref = config.classes[selectedIdx].references[ri];
  ref.boxes.splice(bi, 1);
  ref.labels.splice(bi, 1);
  markDirty();
  renderEditor();
}

function toggleLabel(ri, bi) {
  const ref = config.classes[selectedIdx].references[ri];
  ref.labels[bi] = ref.labels[bi] === 1 ? 0 : 1;
  markDirty();
  renderEditor();
}

function redrawCanvas(ri, img) {
  const cv = document.getElementById('canvas-' + ri);
  if (!cv) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);

  if (img) {
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
  } else {
    ctx.fillStyle = '#0d1b2a';
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = '#a0a0b0';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Select an image', cv.width / 2, cv.height / 2);
    ctx.textAlign = 'start';
    return;
  }

  // Draw existing boxes
  const ref = config.classes[selectedIdx]?.references?.[ri];
  if (!ref || !ref.boxes) return;
  ref.boxes.forEach((box, bi) => {
    const [cx, cy, w, h] = box;
    const lbl = ref.labels?.[bi] ?? 1;
    const x = (cx - w / 2) * cv.width;
    const y = (cy - h / 2) * cv.height;
    const bw = w * cv.width;
    const bh = h * cv.height;

    ctx.strokeStyle = lbl === 1 ? '#4ecca3' : '#e94560';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);

    // Label badge
    const tag = lbl === 1 ? 'pos' : 'neg';
    ctx.font = 'bold 11px sans-serif';
    const tw = ctx.measureText(tag).width + 8;
    ctx.fillStyle = lbl === 1 ? 'rgba(78,204,163,0.85)' : 'rgba(233,69,96,0.85)';
    ctx.fillRect(x, y - 16, tw, 16);
    ctx.fillStyle = lbl === 1 ? '#000' : '#fff';
    ctx.fillText(tag, x + 4, y - 4);
  });
}

// ─── Utils ───────────────────────────────────────────────────────────
function round4(v) { return Math.round(v * 10000) / 10000; }
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ─── Keyboard shortcut ──────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveConfig(); }
});

// ─── Boot ────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class ConfigEditorHandler(BaseHTTPRequestHandler):
    """Request handler for the config editor."""

    config_path: Path = None   # set by main()
    base_dir: Path = None

    def log_message(self, fmt, *args):
        # Quieter logging: just method + path
        print(f"  {args[0]}")

    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, msg):
        body = msg.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    # ── Routes ────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_html(HTML_PAGE)

        elif path == "/api/config":
            try:
                cfg = json.loads(self.config_path.read_text())
                self._send_json(cfg)
            except Exception as e:
                self._send_error(500, str(e))

        elif path == "/api/references":
            refs_dir = self.base_dir / "references"
            images = []
            if refs_dir.is_dir():
                for f in sorted(refs_dir.rglob("*")):
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                        images.append(str(f.relative_to(self.base_dir)))
            self._send_json(images)

        elif path.startswith("/file/"):
            self._serve_file(path[6:])

        else:
            self._send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/config":
            try:
                body = self._read_body()
                cfg = json.loads(body)
                # Validate basic structure
                if not isinstance(cfg.get("classes"), list):
                    self._send_error(400, "Missing 'classes' array")
                    return
                if len(cfg["classes"]) > 4:
                    self._send_error(400, "Max 4 classes")
                    return
                # Pretty-print save
                self.config_path.write_text(
                    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
                )
                self._send_json({"ok": True})
            except json.JSONDecodeError as e:
                self._send_error(400, f"Invalid JSON: {e}")
            except Exception as e:
                self._send_error(500, str(e))
        else:
            self._send_error(404, "Not found")

    def _serve_file(self, raw_path):
        """Serve an image file — only under references/ for security."""
        decoded = urllib.parse.unquote(raw_path)
        # Resolve relative to base_dir, then check it's under references/
        try:
            target = (self.base_dir / decoded).resolve()
            refs_dir = (self.base_dir / "references").resolve()
            if not str(target).startswith(str(refs_dir)):
                self._send_error(403, "Access denied: outside references/")
                return
            if not target.is_file():
                self._send_error(404, "File not found")
                return
        except Exception:
            self._send_error(400, "Bad path")
            return

        mime, _ = mimetypes.guess_type(str(target))
        if mime is None:
            mime = "application/octet-stream"

        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=300")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 Config Editor — visual config.json editor")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (default: config.json in script directory)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = (Path(__file__).parent / "config.json").resolve()

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        raise SystemExit(1)

    base_dir = config_path.parent

    # Verify config is valid JSON
    try:
        json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {config_path}: {e}")
        raise SystemExit(1)

    ConfigEditorHandler.config_path = config_path
    ConfigEditorHandler.base_dir = base_dir

    server = HTTPServer(("0.0.0.0", args.port), ConfigEditorHandler)
    print(f"SAM3 Config Editor")
    print(f"  Config: {config_path}")
    print(f"  Base:   {base_dir}")
    print(f"  URL:    http://localhost:{args.port}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
