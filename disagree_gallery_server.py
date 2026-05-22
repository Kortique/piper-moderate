#!/usr/bin/env python3
"""
disagree_gallery_server.py
--------------------------
HTTP gallery for labeling disagree images from data/disagree_pool.json.

Labels: child (≤14) | teen (15–17) | adult (≥18)

Usage:
    python3 disagree_gallery_server.py          # port 7824
    python3 disagree_gallery_server.py --port 7825
"""

import json
import os
import sys
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

BASE_DIR    = Path(__file__).resolve().parent
POOL_FILE   = BASE_DIR / "data" / "disagree_pool.json"
IMAGES_DIR  = BASE_DIR / "data" / "disagree_images"
PORT        = 7824

GALLERY_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Disagree Pool — разметка</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; font-size: 12px; background: #0f0f0f; color: #e0e0e0; }
header {
  padding: 10px 20px; background: #1a1a1a; border-bottom: 1px solid #333;
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  position: sticky; top: 0; z-index: 100;
}
header h1 { font-size: 13px; font-weight: normal; color: #aaa; white-space: nowrap; }
select, input[type=text] {
  background: #222; border: 1px solid #444; color: #e0e0e0;
  padding: 3px 7px; border-radius: 3px; font-size: 12px; font-family: monospace;
}
#save-btn {
  background: #1a2e1a; border: 1px solid #3a6a3a; color: #8fda8f;
  padding: 5px 16px; border-radius: 4px; cursor: pointer;
  font-size: 13px; font-family: monospace; margin-left: auto;
}
#save-btn:hover { background: #22401e; }
#save-btn.dirty { background: #3a1a1a; border-color: #cc4444; color: #ff8888; }
#status { font-size: 11px; color: #666; white-space: nowrap; }
.stats-bar {
  font-size: 11px; color: #666;
  display: flex; gap: 14px; align-items: center; padding: 6px 20px;
  background: #111; border-bottom: 1px solid #2a2a2a;
}
.stats-bar span { white-space: nowrap; }
.stats-bar .s-total  { color: #888; }
.stats-bar .s-child  { color: #ff7070; }
.stats-bar .s-teen   { color: #ffd040; }
.stats-bar .s-adult  { color: #6fda72; }
.stats-bar .s-none   { color: #555; }
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px; padding: 12px;
}
.card {
  background: #181818; border: 1px solid #2a2a2a;
  border-radius: 6px; overflow: hidden;
  display: flex; flex-direction: column;
  transition: border-color .12s;
}
.card:hover { border-color: #555; }
.card.modified  { border-color: #4a7a4a; }
.card.lbl-child { border-color: #883333; }
.card.lbl-teen  { border-color: #887733; }
.card.lbl-adult { border-color: #337744; }

.img-wrap {
  width: 100%; aspect-ratio: 1;
  overflow: hidden; background: #111;
  display: flex; align-items: center; justify-content: center;
  cursor: zoom-in; position: relative;
}
.img-wrap img {
  width: 100%; height: 100%; object-fit: contain;
  display: block; transition: transform .18s;
}
.img-wrap:hover img { transform: scale(1.04); }

/* Pipeline result badge */
.pipe-badge {
  position: absolute; bottom: 5px; left: 5px;
  font-size: 10px; padding: 2px 5px; border-radius: 3px;
  pointer-events: none;
}
.pipe-badge.underage { background: rgba(136,40,40,.85); color: #ffa0a0; }
.pipe-badge.other    { background: rgba(80,50,10,.85);  color: #ffd080; }
.pipe-badge.passed   { background: rgba(20,60,20,.85);  color: #80e080; }

.info {
  padding: 5px 8px 3px;
  border-bottom: 1px solid #222;
  display: flex; justify-content: space-between; align-items: center;
}
.info .img-id { color: #888; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 120px; }
.info .date   { color: #555; font-size: 10px; }

.prompt-row {
  padding: 3px 8px 4px;
  font-size: 10px; color: #444;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  cursor: help;
  border-bottom: 1px solid #1f1f1f;
}
.prompt-row:hover { color: #888; }

.radios {
  display: flex; padding: 5px 8px 6px;
}
.radios label {
  flex: 1; text-align: center; padding: 5px 2px;
  cursor: pointer; border: 1px solid #333;
  font-size: 11px; color: #888;
  transition: background .1s;
}
.radios label:first-child { border-radius: 4px 0 0 4px; }
.radios label:last-child  { border-radius: 0 4px 4px 0; }
.radios label:not(:first-child) { border-left: none; }
.radios input { display: none; }
.radios label.cat-child:has(input:checked)  { background: #3a1010; border-color: #883333; color: #ff8080; font-weight: bold; }
.radios label.cat-teen:has(input:checked)   { background: #3a2e00; border-color: #887733; color: #ffd040; font-weight: bold; }
.radios label.cat-adult:has(input:checked)  { background: #0f2210; border-color: #337744; color: #6fda72; font-weight: bold; }

#lb {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.9); z-index: 999;
  align-items: center; justify-content: center; cursor: zoom-out;
  flex-direction: column; gap: 10px;
}
#lb.open { display: flex; }
#lb img  { max-width: 90vw; max-height: 80vh; object-fit: contain; border-radius: 4px; }
#lb-prompt {
  max-width: 80vw; font-size: 11px; color: #888;
  text-align: center; line-height: 1.5; padding: 0 20px;
}
#pagination {
  display: flex; align-items: center; justify-content: center;
  gap: 6px; padding: 14px 20px; flex-wrap: wrap;
}
#pagination button {
  background: #1e1e1e; border: 1px solid #444; color: #aaa;
  padding: 4px 10px; border-radius: 3px; cursor: pointer;
  font-size: 12px; font-family: monospace; min-width: 32px;
}
#pagination button:hover { background: #2a2a2a; color: #fff; }
#pagination button.active { background: #2a3a5a; border-color: #4a6aaa; color: #88aaff; font-weight: bold; }
#pagination button:disabled { opacity: .3; cursor: default; }
#pagination .pg-info { color: #666; font-size: 11px; padding: 0 6px; }
</style>
</head>
<body>

<header>
  <h1>Disagree Pool <span id="subtitle">…</span></h1>

  <label>Разметка:
    <select id="f-label">
      <option value="unlabeled">Без разметки</option>
      <option value="all">Все</option>
      <option value="child">Дети (≤14)</option>
      <option value="teen">Подростки (15–17)</option>
      <option value="adult">Взрослые (≥18)</option>
    </select>
  </label>

  <label>Pipeline:
    <select id="f-pipe">
      <option value="all">Все</option>
      <option value="underage">blocked: underage</option>
      <option value="other">blocked: other</option>
      <option value="passed">passed</option>
      <option value="unprocessed">не прогнан</option>
    </select>
  </label>

  <label>На стр.:
    <select id="f-pgsize">
      <option value="50">50</option>
      <option value="100" selected>100</option>
      <option value="250">250</option>
    </select>
  </label>

  <span id="status"></span>
  <button id="save-btn" onclick="saveChanges()">💾 Сохранить</button>
</header>

<div class="stats-bar" id="stats-bar">…</div>

<div class="gallery" id="gallery"></div>
<div id="pagination"></div>

<div id="lb" onclick="closeLb()">
  <img id="lb-img" src="" onclick="event.stopPropagation()">
  <div id="lb-prompt"></div>
</div>

<script>
let allData = [], changes = {}, filteredData = [], currentPage = 1;

async function loadData() {
  const r = await fetch('/api/data');
  allData = await r.json();
  updateStats();
  applyFilter();
}

function pipeStatus(r) {
  const res = r.piper_result;
  if (!res || res.error) return 'unprocessed';
  const labels = res.siglip2_labels || [];
  if (labels.includes('underage')) return 'underage';
  if (!res.siglip2_passed) return 'other';
  return 'passed';
}

function effectiveLabel(r) {
  return changes[r.id] ? changes[r.id].label : r.label;
}

function updateStats() {
  const totals = {total: allData.length, unlabeled: 0, child: 0, teen: 0, adult: 0};
  allData.forEach(r => {
    const lbl = effectiveLabel(r) || 'unlabeled';
    totals[lbl] = (totals[lbl] || 0) + 1;
  });
  document.getElementById('stats-bar').innerHTML =
    `<span class="s-total">всего: ${totals.total}</span>` +
    `<span class="s-none">без разметки: ${totals.unlabeled}</span>` +
    `<span class="s-child">child: ${totals.child || 0}</span>` +
    `<span class="s-teen">teen: ${totals.teen || 0}</span>` +
    `<span class="s-adult">adult: ${totals.adult || 0}</span>`;
}

function applyFilter() {
  const lf  = document.getElementById('f-label').value;
  const pf  = document.getElementById('f-pipe').value;
  let d = allData;
  if (lf !== 'all') {
    if (lf === 'unlabeled') d = d.filter(r => !effectiveLabel(r));
    else d = d.filter(r => effectiveLabel(r) === lf);
  }
  if (pf !== 'all') d = d.filter(r => pipeStatus(r) === pf);
  filteredData = d;
  currentPage = 1;
  renderPage();
}

function renderCard(r) {
  const curLabel = effectiveLabel(r);
  const ps = pipeStatus(r);

  const card = document.createElement('div');
  card.className = 'card' + (changes[r.id] ? ' modified' : '') + (curLabel ? ` lbl-${curLabel}` : '');
  card.id = 'card-' + r.id;

  // Image wrap
  const wrap = document.createElement('div');
  wrap.className = 'img-wrap';

  const img = document.createElement('img');
  img.loading = 'lazy';
  img.src = r._serve_url || r.thumb_url;
  img.onerror = () => { wrap.innerHTML = '<span style="color:#444;font-size:10px">no image</span>'; };
  img.onclick = () => openLb(r);
  wrap.appendChild(img);

  // Pipeline badge
  if (ps !== 'unprocessed') {
    const badge = document.createElement('div');
    badge.className = 'pipe-badge ' + (ps === 'passed' ? 'passed' : ps === 'underage' ? 'underage' : 'other');
    badge.textContent = ps === 'passed' ? '✓ ok'
      : ps === 'underage' ? '⛔ underage'
      : '⚠ other';
    wrap.appendChild(badge);
  }

  // Info row
  const info = document.createElement('div');
  info.className = 'info';
  const idEl = document.createElement('span');
  idEl.className = 'img-id';
  idEl.title = r.id;
  idEl.textContent = r.id.substring(0, 8) + '…';
  const dateEl = document.createElement('span');
  dateEl.className = 'date';
  if (r.exported_at) {
    const dt = new Date(r.exported_at);
    dateEl.textContent = dt.toLocaleDateString('ru', {day:'2-digit',month:'2-digit'});
  }
  info.append(idEl, dateEl);

  // Prompt row
  const promptRow = document.createElement('div');
  promptRow.className = 'prompt-row';
  promptRow.title = r.prompt || '';
  promptRow.textContent = (r.prompt || '').substring(0, 80) || '—';

  // Label radios
  const radios = document.createElement('div');
  radios.className = 'radios';
  [['child','Дети'],['teen','Подр.'],['adult','Взрослые']].forEach(([lbl, name]) => {
    const label = document.createElement('label');
    label.className = 'cat-' + lbl;
    const inp = document.createElement('input');
    inp.type = 'radio';
    inp.name = 'lbl-' + r.id;
    inp.value = lbl;
    inp.checked = (curLabel === lbl);
    inp.addEventListener('change', () => onLabelChange(r.id, lbl));
    label.append(inp, Object.assign(document.createElement('span'), {textContent: name}));
    radios.appendChild(label);
  });

  card.append(wrap, info, promptRow, radios);
  return card;
}

function onLabelChange(id, newLabel) {
  changes[id] = {label: newLabel};
  const cardEl = document.getElementById('card-' + id);
  if (cardEl) {
    cardEl.className = cardEl.className
      .replace(/\blbl-\w+/g, '')
      .replace(/\bmodified\b/g, '')
      .trim();
    cardEl.classList.add('modified', 'lbl-' + newLabel);
  }
  updateDirty();
  updateStats();
}

function updateDirty() {
  const n = Object.keys(changes).length;
  const btn = document.getElementById('save-btn');
  if (n > 0) {
    btn.className = 'dirty';
    btn.textContent = '💾 Сохранить (' + n + ')';
    document.getElementById('status').textContent = 'изменено: ' + n;
  } else {
    btn.className = '';
    btn.textContent = '💾 Сохранить';
    document.getElementById('status').textContent = '';
  }
}

async function saveChanges() {
  if (!Object.keys(changes).length) return;
  document.getElementById('save-btn').textContent = '⏳ Сохранение…';
  try {
    const r = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({changes}),
    });
    const res = await r.json();
    changes = {};
    document.getElementById('save-btn').className = '';
    document.getElementById('save-btn').textContent = `✓ Сохранено (${res.saved})`;
    document.getElementById('status').textContent = '';
    await loadData();
  } catch(e) {
    document.getElementById('save-btn').textContent = '❌ Ошибка';
    console.error(e);
  }
}

function openLb(r) {
  document.getElementById('lb-img').src = r._serve_url || r.thumb_url;
  document.getElementById('lb-prompt').textContent = r.prompt || '';
  document.getElementById('lb').classList.add('open');
}
function closeLb() { document.getElementById('lb').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLb(); });

function renderPage() {
  const pgSize = parseInt(document.getElementById('f-pgsize').value);
  const total  = filteredData.length;
  const pages  = Math.max(1, Math.ceil(total / pgSize));
  if (currentPage > pages) currentPage = pages;

  const start = (currentPage - 1) * pgSize;
  const slice = filteredData.slice(start, start + pgSize);

  const g = document.getElementById('gallery');
  g.innerHTML = '';
  slice.forEach(r => g.appendChild(renderCard(r)));
  window.scrollTo(0, 0);

  document.getElementById('subtitle').textContent =
    `— ${total} / ${allData.length}`;

  renderPagination(pages, pgSize, total);
}

function renderPagination(pages, pgSize, total) {
  const pg = document.getElementById('pagination');
  pg.innerHTML = '';
  if (pages <= 1) return;
  const start = (currentPage - 1) * pgSize + 1;
  const end   = Math.min(currentPage * pgSize, total);

  function btn(label, page, disabled, active) {
    const b = document.createElement('button');
    b.textContent = label;
    if (active) b.className = 'active';
    if (disabled) b.disabled = true;
    else b.onclick = () => { currentPage = page; renderPage(); };
    return b;
  }

  pg.appendChild(btn('«', 1, currentPage === 1));
  pg.appendChild(btn('‹', currentPage - 1, currentPage === 1));
  let lo = Math.max(1, currentPage - 3);
  let hi = Math.min(pages, lo + 6);
  lo = Math.max(1, hi - 6);
  if (lo > 1) { pg.appendChild(btn('1', 1)); if (lo > 2) pg.appendChild(Object.assign(document.createElement('span'),{textContent:'…',className:'pg-info'})); }
  for (let p = lo; p <= hi; p++) pg.appendChild(btn(p, p, false, p === currentPage));
  if (hi < pages) { if (hi < pages-1) pg.appendChild(Object.assign(document.createElement('span'),{textContent:'…',className:'pg-info'})); pg.appendChild(btn(pages, pages)); }
  pg.appendChild(btn('›', currentPage + 1, currentPage === pages));
  pg.appendChild(btn('»', pages, currentPage === pages));
  const info = document.createElement('span');
  info.className = 'pg-info';
  info.textContent = `${start}–${end} из ${total}`;
  pg.appendChild(info);
}

document.getElementById('f-label').addEventListener('change', applyFilter);
document.getElementById('f-pipe').addEventListener('change', applyFilter);
document.getElementById('f-pgsize').addEventListener('change', () => { currentPage = 1; renderPage(); });

// Keyboard shortcuts: 1=child, 2=teen, 3=adult for current-hover card
let hoveredId = null;
document.addEventListener('mouseover', e => {
  const card = e.target.closest('.card');
  hoveredId = card ? card.id.replace('card-', '') : null;
});
document.addEventListener('keydown', e => {
  if (!hoveredId || document.getElementById('lb').classList.contains('open')) return;
  if (['1','2','3'].includes(e.key)) {
    const map = {'1':'child','2':'teen','3':'adult'};
    const lbl = map[e.key];
    const inp = document.querySelector(`input[name="lbl-${hoveredId}"][value="${lbl}"]`);
    if (inp) { inp.checked = true; inp.dispatchEvent(new Event('change')); }
    e.preventDefault();
  }
});

loadData();
</script>
</body>
</html>"""


def load_pool() -> dict:
    if POOL_FILE.exists():
        with open(POOL_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_pool(pool: dict):
    tmp = str(POOL_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    os.replace(tmp, POOL_FILE)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Suppress noisy request logging; print minimal info
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = GALLERY_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/data":
            pool = load_pool()
            rows = sorted(pool.values(), key=lambda v: v.get("exported_at") or "", reverse=True)
            # Prefer local file path served via /img/ route
            for r in rows:
                if r.get("local_path"):
                    r["_serve_url"] = "/img/" + Path(r["local_path"]).name
                else:
                    r["_serve_url"] = r["thumb_url"]
            self.send_json(rows)

        elif path.startswith("/img/"):
            fname = path[5:]  # strip /img/
            fpath = IMAGES_DIR / fname
            if fpath.exists() and fpath.resolve().parent == IMAGES_DIR.resolve():
                data = fpath.read_bytes()
                ext  = fpath.suffix.lower()
                mime = {"webp": "image/webp", "jpg": "image/jpeg",
                        "jpeg": "image/jpeg", "png": "image/png"}.get(ext.lstrip("."), "image/webp")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/api/stats":
            pool = load_pool()
            labels = [v["label"] for v in pool.values()]
            self.send_json({
                "total":     len(pool),
                "unlabeled": labels.count(None),
                "child":     labels.count("child"),
                "teen":      labels.count("teen"),
                "adult":     labels.count("adult"),
            })

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/save":
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            updates = body.get("changes", {})

            pool = load_pool()
            saved = 0
            now   = datetime.now(timezone.utc).isoformat()
            for gen_id, upd in updates.items():
                if gen_id in pool and upd.get("label") in ("child", "teen", "adult"):
                    pool[gen_id]["label"]      = upd["label"]
                    pool[gen_id]["labeled_at"] = now
                    saved += 1

            save_pool(pool)
            print(f"  Labeled {saved} images")
            self.send_json({"saved": saved})

        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    pool = load_pool()
    labels = [v["label"] for v in pool.values()]
    print(f"Pool file : {POOL_FILE}")
    print(f"Pool size : {len(pool)} images  "
          f"(unlabeled={labels.count(None)}  child={labels.count('child')}  "
          f"teen={labels.count('teen')}  adult={labels.count('adult')})")
    print(f"Starting  : http://localhost:{args.port}")
    print(f"Hotkeys   : hover over image, press 1=child  2=teen  3=adult")
    print(f"Press Ctrl+C to stop\n")

    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
