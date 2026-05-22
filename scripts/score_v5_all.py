#!/usr/bin/env python3
"""
Apply V5-new LGBM to ALL gallery items with siglip2_details in DB.
V4 score is taken from siglip2_details.underage.lgbm.score (stored by Piper).
V5 score is computed locally using the V5-new LGBM model (312 features).
Outputs: data/eval_v5_db_all.json
"""
import json, re, math, sqlite3, struct
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "gallery.db"
JS_PATH  = BASE_DIR / "data" / "lgbm_evaluate_v5_piper.js"
OUT_PATH = BASE_DIR / "data" / "eval_v5_db_all.json"

def _open_db(src: Path) -> sqlite3.Connection:
    """Open gallery.db, patching the page-count header if it mismatches file size."""
    data = src.read_bytes()
    actual_pages = len(data) // 4096
    header_pages = struct.unpack('>I', data[28:32])[0]
    if header_pages != actual_pages:
        data = bytearray(data)
        struct.pack_into('>I', data, 28, actual_pages)
        tmp = Path('/tmp/_gal_patched.db')
        tmp.write_bytes(data)
        conn = sqlite3.connect(str(tmp))
    else:
        conn = sqlite3.connect(str(src))
    conn.row_factory = sqlite3.Row
    return conn

# ── Load V5 LGBM from JS ──────────────────────────────────────────────────────
js = JS_PATH.read_text()
features = json.loads(re.search(r'const LGBM_FEATURES = (\[.*?\]);', js, re.DOTALL).group(1))
trees    = json.loads(re.search(r'const LGBM_TREES = (\[.*?\]);',    js, re.DOTALL).group(1))
feat_idx = {f: i for i, f in enumerate(features)}

def lgbm_predict(vec):
    score = 0.0
    for t in trees:
        node = t['r']
        while node >= 0:
            fi, thr = t['s'][node]
            l, r = t['c'][node]
            node = l if vec[fi] <= thr else r
        score += t['l'][-(node + 1)]
    return 1.0 / (1.0 + math.exp(-score))

def build_vec(underage_labels: dict, adult_labels: dict) -> list:
    """Build 312-dim feature vector from siglip2 labels dicts."""
    vec = [0.0] * len(features)
    # underage labels: key is bare name (stored without 'underage_' prefix)
    # JS strips 'underage_' from the full tag name; in DB they're already stripped
    for k, v in underage_labels.items():
        fname = k  # already bare
        if fname in feat_idx:
            vec[feat_idx[fname]] = v
    # adult labels: key is bare name (without 'adult_' prefix)
    # JS expects 'adult__' + bare_name
    for k, v in adult_labels.items():
        fname = 'adult__' + k
        if fname in feat_idx:
            vec[feat_idx[fname]] = v
    return vec

def score_item(siglip2_details: dict) -> dict | None:
    """Return {lgbm_v4, lgbm_v5, minor, blocked_v4, blocked_v5} or None if no data."""
    und = siglip2_details.get('underage', {})
    if not und:
        return None
    minor   = und.get('minor', 0.0)
    lgbm_v4 = und.get('lgbm', {}).get('score', None)
    labels  = und.get('labels', {})
    underage_lbl = labels.get('underage', {})
    adult_lbl    = labels.get('adult',    {})
    vec          = build_vec(underage_lbl, adult_lbl)
    lgbm_v5      = lgbm_predict(vec)
    return {
        'lgbm_v4':     round(lgbm_v4,  4) if lgbm_v4 is not None else None,
        'lgbm_v5':     round(lgbm_v5,  4),
        'minor':       round(minor,    4),
        'blocked_v4':  (lgbm_v4 >= 0.80 if lgbm_v4 is not None else None) or (minor >= 0.72),
        'blocked_v5':  (lgbm_v5 >= 0.80) or (minor >= 0.72),
    }

# ── Read DB ───────────────────────────────────────────────────────────────────
conn = _open_db(DB_PATH)

def ls_cat(age_from):
    if age_from is None: return None
    if age_from < 15: return 'child'
    if age_from <= 17: return 'teen'
    return 'adult'

results = []

# ── ls_images ─────────────────────────────────────────────────────────────────
print("Processing ls_images...")
n_ok = n_skip = 0
for offset in range(0, 2500, 200):
    try:
        rows = conn.execute(
            "SELECT task_id, age_from, siglip2_details FROM ls_images LIMIT 200 OFFSET ?",
            (offset,)
        ).fetchall()
        if not rows:
            break
        for row in rows:
            task_id = str(row['task_id'])
            age_from = row['age_from']
            raw = row['siglip2_details']
            if not raw:
                n_skip += 1
                continue
            try:
                det = json.loads(raw)
                sc = score_item(det)
                if sc is None:
                    n_skip += 1
                    continue
                human_label = ls_cat(age_from)
                results.append({
                    'id':          'ls_' + task_id,
                    'human_label': human_label,
                    'lgbm_v4':     sc['lgbm_v4'],
                    'lgbm_v5':     sc['lgbm_v5'],
                    'minor':       sc['minor'],
                    'blocked_v4':  sc['blocked_v4'],
                    'blocked_v5':  sc['blocked_v5'],
                    'source':      'db',
                })
                n_ok += 1
            except Exception as e:
                n_skip += 1
    except Exception as e:
        print(f"  Batch error at offset {offset}: {e}")

print(f"  ls_images: {n_ok} scored, {n_skip} skipped")

# ── grafana_pool ──────────────────────────────────────────────────────────────
print("Processing grafana_pool...")
gn_ok = gn_skip = 0
for offset in [0, 100, 200, 300, 500, 600]:
    try:
        rows = conn.execute(
            "SELECT id, label, piper_result FROM grafana_pool LIMIT 100 OFFSET ?",
            (offset,)
        ).fetchall()
        for row in rows:
            item_id = str(row['id'])
            label   = row['label']
            raw     = row['piper_result']
            if not raw:
                gn_skip += 1
                continue
            try:
                pr = json.loads(raw)
                # siglip2_details is inside piper_result
                det = pr.get('siglip2_details') or pr.get('siglip2_result', {}).get('siglip2_details')
                if not det:
                    # Try nested structure
                    for k, v in pr.items():
                        if isinstance(v, dict) and 'underage' in v:
                            det = v
                            break
                if not det:
                    gn_skip += 1
                    continue
                sc = score_item(det)
                if sc is None:
                    gn_skip += 1
                    continue
                results.append({
                    'id':          item_id,  # bare UUID for grafana
                    'human_label': label,
                    'lgbm_v4':     sc['lgbm_v4'],
                    'lgbm_v5':     sc['lgbm_v5'],
                    'minor':       sc['minor'],
                    'blocked_v4':  sc['blocked_v4'],
                    'blocked_v5':  sc['blocked_v5'],
                    'source':      'db',
                })
                gn_ok += 1
            except Exception as e:
                gn_skip += 1
    except Exception as e:
        print(f"  Grafana batch error at offset {offset}: {e}")

print(f"  grafana_pool: {gn_ok} scored, {gn_skip} skipped")

conn.close()

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH.write_text(json.dumps(results, indent=2))
print(f"\nSaved {len(results)} items → {OUT_PATH}")

# Quick stats
v4_blocked = sum(1 for r in results if r['blocked_v4'])
v5_blocked = sum(1 for r in results if r['blocked_v5'])
labeled    = [r for r in results if r['human_label'] in ('child','teen','adult')]
minors     = [r for r in labeled  if r['human_label'] in ('child','teen')]
adults     = [r for r in labeled  if r['human_label'] == 'adult']
print(f"\nQuick stats (labeled items only: {len(labeled)}):")
print(f"  V4 blocked: {v4_blocked} total | recall {sum(1 for r in minors if r['blocked_v4'])}/{len(minors)} | FPR {sum(1 for r in adults if r['blocked_v4'])}/{len(adults)}")
print(f"  V5 blocked: {v5_blocked} total | recall {sum(1 for r in minors if r['blocked_v5'])}/{len(minors)} | FPR {sum(1 for r in adults if r['blocked_v5'])}/{len(adults)}")
