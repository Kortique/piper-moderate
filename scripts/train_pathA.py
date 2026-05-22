#!/usr/bin/env python3
"""
Path A retrain:
  - В training data применить INVERSE getAdjustedScores: для ключей с :x20/:x5
    раз-усилить (divide by multiplier). Все остальные ключи остаются raw.
  - Имена features в LGBM — без :x20 суффикса (так buildVec будет matchить
    после strip).
  - V7pa: trained на old siglip + no merge.
  - V10pa: trained на new siglip + merge no_underage → adult.

После этого нужно изменить getAdjustedScores в Piper чтобы не делал multiplier.
"""
import json, re, struct, os, tempfile, datetime, ast, sqlite3
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')


def unmultiply(key, val):
    """Reverse adjusted = min(raw*mult, 0.999)."""
    m = X20_RE.search(key)
    if not m:
        return float(val)
    mult = float(m.group(1))
    if val >= 0.999:
        # capped — best estimate: lower bound (raw = 0.999/mult)
        return 0.999 / mult
    return float(val) / mult


def strip_mult(k):
    return X20_RE.sub('', k)


def sanitize(n):
    return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')


def open_db():
    raw = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', raw, 28, len(raw)//4096)
    fd, tmp = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd); Path(tmp).write_bytes(bytes(raw))
    conn = sqlite3.connect(tmp); conn.row_factory = sqlite3.Row
    return conn, tmp


def transform_for_pathA(under, adult, no_und, merge_no_und=False):
    """Apply path A transformation:
       - underage: keys with :x20 → divide value by multiplier; strip :x suffix
       - adult: keep; if merge_no_und, also merge no_und keys with max
    """
    u = {}
    for k, v in (under or {}).items():
        raw = unmultiply(k, v)
        k2 = strip_mult(k)
        u[k2] = max(u.get(k2, 0.0), raw)
    a = dict((k, float(v)) for k, v in (adult or {}).items())
    if merge_no_und:
        for k, v in (no_und or {}).items():
            a[k] = max(a.get(k, 0.0), float(v))
        nu = {}
    else:
        nu = dict((k, float(v)) for k, v in (no_und or {}).items())
    return u, a, nu


def load_v7_items():
    items = []
    raw = (BASE / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    qd = json.loads(raw)
    for v in qd.values():
        cat = v.get('category')
        if cat not in ('child','teen','adult'): continue
        sd = v.get('siglip2_details')
        if isinstance(sd, str):
            try: sd = ast.literal_eval(sd)
            except: sd = None
        det = (sd or {}).get('underage') or {}
        if not det: continue
        labels = det.get('labels') or {}
        items.append({'label': cat, 'weight': 1.0,
                      'underage': labels.get('underage') or {},
                      'adult': labels.get('adult') or {},
                      'no_underage': {}})
    conn, tmp = open_db()
    for r in conn.execute("""SELECT label,piper_result,export_batch FROM grafana_pool
                             WHERE (deleted IS NULL OR deleted=0) AND label IN ('child','teen','adult')
                             AND piper_result IS NOT NULL""").fetchall():
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage') or {}
            if not det: continue
            labels = det.get('labels') or {}
            w = 3.0 if r['export_batch'] == '2026-05-20 UTC' else 1.0
            items.append({'label': r['label'], 'weight': w,
                          'underage': labels.get('underage') or {},
                          'adult': labels.get('adult') or {},
                          'no_underage': {}})
        except: pass
    conn.close(); os.unlink(tmp)
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'label': 'adult', 'weight': 1.0,
                          'underage': r.get('underage_labels', {}),
                          'adult': r.get('adult_labels', {}),
                          'no_underage': {}})
    return items


def load_v10_items():
    items = []
    for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'label': r['label'], 'weight': 1.0,
                      'underage': r.get('underage_labels') or {},
                      'adult': r.get('adult_labels') or {},
                      'no_underage': r.get('no_underage_labels') or {}})
    for r in json.loads((DATA / 'v9_317_scores.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'label': r['label'], 'weight': 3.0,
                      'underage': r.get('underage_labels') or {},
                      'adult': r.get('adult_labels') or {},
                      'no_underage': r.get('no_underage_labels') or {}})
    for r in json.loads((DATA / 'eval616_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        if r.get('kind') == 'ls': continue
        items.append({'label': r['label'], 'weight': 1.0,
                      'underage': r.get('underage_labels') or {},
                      'adult': r.get('adult_labels') or {},
                      'no_underage': r.get('no_underage_labels') or {}})
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'label': 'adult', 'weight': 1.0,
                          'underage': r.get('underage_labels', {}),
                          'adult': r.get('adult_labels', {}),
                          'no_underage': {}})
    return items


def build_xy(items, merge_no_und):
    feat_set = set()
    rows = []
    for it in items:
        u, a, nu = transform_for_pathA(it['underage'], it['adult'], it['no_underage'],
                                       merge_no_und=merge_no_und)
        for k in u: feat_set.add(sanitize(k))
        for k in a: feat_set.add('adult__' + sanitize(k))
        for k in nu: feat_set.add('no_underage__' + sanitize(k))
        rows.append((u, a, nu, 1 if it['label'] in ('child','teen') else 0, it['weight']))
    feats = sorted(feat_set)
    idx = {f: i for i, f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    w = np.ones(len(items), dtype=np.float32)
    for i, (u, a, nu, yi, wi) in enumerate(rows):
        for k, v in u.items():
            f = sanitize(k)
            if f in idx: X[i, idx[f]] = v
        for k, v in a.items():
            f = 'adult__' + sanitize(k)
            if f in idx: X[i, idx[f]] = v
        for k, v in nu.items():
            f = 'no_underage__' + sanitize(k)
            if f in idx: X[i, idx[f]] = v
        y[i] = yi
        w[i] = wi
    return X, y, w, feats


def train_save(name, X, y, w, feats, parent, note):
    params = dict(objective='binary', metric='auc', num_leaves=15,
                  learning_rate=0.04, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=3, lambda_l1=0.1, lambda_l2=0.1,
                  is_unbalance=True, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        ds = lgb.Dataset(X[tr], label=y[tr], weight=w[tr])
        m = lgb.train(params, ds, num_boost_round=150)
        aucs.append(roc_auc_score(y[va], m.predict(X[va])))
    auc, std = float(np.mean(aucs)), float(np.std(aucs))
    final = lgb.train(params, lgb.Dataset(X, label=y, weight=w), num_boost_round=150)
    final.save_model(str(DATA / f'lgbm_underage_{name}.txt'))
    (DATA / f'lgbm_{name}_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / f'lgbm_{name}_meta.json').write_text(json.dumps({
        'version': name, 'parent': parent, 'note': note,
        'n_features': len(feats), 'cv_auc': auc, 'cv_std': std,
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))
    print(f'  {name}: features={len(feats)}  CV AUC={auc:.4f} ± {std:.4f}', flush=True)


def main():
    print('Loading V7 items + applying path A transformation...', flush=True)
    items_v7 = load_v7_items()
    print(f'  items: {len(items_v7)}', flush=True)
    X, y, w, feats = build_xy(items_v7, merge_no_und=False)
    train_save('v7pa', X, y, w, feats, 'v7',
               'Path A: raw scores (multiplier removed), :x20 stripped from feature names, no_underage NOT merged')

    print('\nLoading V10 items + applying path A + merge...', flush=True)
    items_v10 = load_v10_items()
    print(f'  items: {len(items_v10)}', flush=True)
    X, y, w, feats = build_xy(items_v10, merge_no_und=True)
    train_save('v10pa', X, y, w, feats, 'v10',
               'Path A: raw scores (multiplier removed), :x20 stripped, no_underage MERGED into adult')

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
