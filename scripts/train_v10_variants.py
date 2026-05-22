#!/usr/bin/env python3
"""
Train V10 variants for ablation:
  V10r  — rename :x20/:x5 features (strip suffix). Signal preserved.
  V10m  — merge no_underage_X → adult_X (sum on collisions).
  V10rm — both rename + merge.
  V7r   — V7 with same rename (for d2911d10bb pipeline).

Output: data/lgbm_underage_<name>.txt, lgbm_<name>_features.json, lgbm_<name>_meta.json
"""
import json, struct, os, tempfile, datetime, ast, sqlite3
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

X20_SUFFIXES = (':x20', ':x5')


def strip_multiplier(key):
    for sfx in X20_SUFFIXES:
        if key.endswith(sfx):
            return key[:-len(sfx)]
    return key


def sanitize(n):
    return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')


def open_db():
    raw = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', raw, 28, len(raw)//4096)
    fd, tmp = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd); Path(tmp).write_bytes(bytes(raw))
    conn = sqlite3.connect(tmp); conn.row_factory = sqlite3.Row
    return conn, tmp


def load_v10_data():
    items = []
    # rescored LS
    for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': 'ls_'+str(r['task_id']), 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 1.0})
    # 317 rescored
    for r in json.loads((DATA / 'v9_317_scores.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': r['id'], 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 3.0})
    # eval616 rescored (Grafana 17-May)
    for r in json.loads((DATA / 'eval616_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        if r.get('kind') == 'ls': continue
        items.append({'id': r['id'], 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 1.0})
    # negatives
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult',
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {}),
                          'no_underage_labels': {},
                          'weight': 1.0})
    return items


def load_v7_data():
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
        items.append({'id': 'ls_'+str(v['task_id']), 'label': cat,
                      'underage_labels': labels.get('underage') or {},
                      'adult_labels': labels.get('adult') or {},
                      'no_underage_labels': {},
                      'weight': 1.0})
    conn, tmp = open_db()
    for r in conn.execute("""SELECT id,label,piper_result,export_batch FROM grafana_pool
                             WHERE (deleted IS NULL OR deleted=0) AND label IN ('child','teen','adult')
                             AND piper_result IS NOT NULL""").fetchall():
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage') or {}
            if not det: continue
            labels = det.get('labels') or {}
            w = 3.0 if r['export_batch'] == '2026-05-20 UTC' else 1.0
            items.append({'id': r['id'], 'label': r['label'],
                          'underage_labels': labels.get('underage') or {},
                          'adult_labels': labels.get('adult') or {},
                          'no_underage_labels': {},
                          'weight': w})
        except: pass
    conn.close(); os.unlink(tmp)
    # negatives
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult',
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {}),
                          'no_underage_labels': {},
                          'weight': 1.0})
    return items


def transform_labels(item, rename_mult=False, merge_no_underage=False):
    """Return (underage_dict, adult_dict, no_underage_dict) — transformed."""
    u = {}
    for k, v in (item.get('underage_labels') or {}).items():
        k2 = strip_multiplier(k) if rename_mult else k
        # max on collisions
        u[k2] = max(u.get(k2, 0.0), float(v))

    if merge_no_underage:
        a = dict((k, float(v)) for k, v in (item.get('adult_labels') or {}).items())
        for k, v in (item.get('no_underage_labels') or {}).items():
            a[k] = max(a.get(k, 0.0), float(v))
        nu = {}  # empty after merge
    else:
        a = dict((k, float(v)) for k, v in (item.get('adult_labels') or {}).items())
        nu = dict((k, float(v)) for k, v in (item.get('no_underage_labels') or {}).items())
    return u, a, nu


def build_features(items, rename_mult=False, merge_no_underage=False):
    feat_set = set()
    rows = []
    for it in items:
        u, a, nu = transform_labels(it, rename_mult, merge_no_underage)
        for k in u: feat_set.add(sanitize(k))
        for k in a: feat_set.add('adult__' + sanitize(k))
        for k in nu: feat_set.add('no_underage__' + sanitize(k))
        rows.append((u, a, nu, 1 if it['label'] in ('child','teen') else 0, it.get('weight', 1.0)))
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


def train_and_save(name, X, y, w, feats, parent, note):
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
    print(f'  {name}: features={len(feats)} CV AUC={auc:.4f} ± {std:.4f}', flush=True)


def main():
    print('Loading V10 data...', flush=True)
    items_v10 = load_v10_data()
    print(f'  items: {len(items_v10)}', flush=True)

    # V10r: rename only
    X, y, w, feats = build_features(items_v10, rename_mult=True, merge_no_underage=False)
    train_and_save('v10r', X, y, w, feats, parent='v10', note='renamed :x20/:x5 features (suffix stripped)')

    # V10m: merge only
    X, y, w, feats = build_features(items_v10, rename_mult=False, merge_no_underage=True)
    train_and_save('v10m', X, y, w, feats, parent='v10', note='merged no_underage_ into adult_ (max on collision)')

    # V10rm: both
    X, y, w, feats = build_features(items_v10, rename_mult=True, merge_no_underage=True)
    train_and_save('v10rm', X, y, w, feats, parent='v10', note='renamed :x20 AND merged no_underage into adult')

    print('\nLoading V7 data...', flush=True)
    items_v7 = load_v7_data()
    print(f'  items: {len(items_v7)}', flush=True)

    # V7r: rename only (V7 has no no_underage to merge)
    X, y, w, feats = build_features(items_v7, rename_mult=True, merge_no_underage=False)
    train_and_save('v7r', X, y, w, feats, parent='v7', note='renamed :x20/:x5 features')

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
