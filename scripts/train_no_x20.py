#!/usr/bin/env python3
"""
Train V7nx and V10nx — same as V7/V10 but with all :x20/:x5 features excluded.
Тестирует гипотезу: эти 12 «зашумлённых» multiplier-named tags вносят искажение.

Output:
  data/lgbm_underage_v7nx.txt + v7nx_features.json + v7nx_meta.json
  data/lgbm_underage_v10nx.txt + v10nx_features.json + v10nx_meta.json
"""
import json, sys, struct, os, tempfile, datetime, ast
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

X20_KEYS = {
    'adult_with_child:x20', 'adult_with_girl:x20', 'adult_with_boy:x20',
    'man_with_young_girl:x20', 'man_with_young_boy:x20',
    'woman_with_young_girl:x20', 'woman_with_young_boy:x20',
    'maternal_contrast:x20', 'older_person_with_child:x20',
    'adult_child_together:x20', 'adult_child_sitting:x20',
    'boy_mature_woman:x5',
}
# sanitized variants used in V10
X20_KEYS_SAN = {k.replace(':', '_x') for k in X20_KEYS}


def sanitize(n):
    return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')


def is_x20(feat):
    """Check if feature key corresponds to one of the :x20 / :x5 multiplier tags."""
    # V7-style: stored as 'man_with_young_girl:x20' (raw with colon)
    if feat in X20_KEYS:
        return True
    # V10-style sanitized: 'man_with_young_girl_xx20'
    if feat in X20_KEYS_SAN:
        return True
    # robust check
    if feat.endswith('_xx20') or feat.endswith('_xx5') or feat.endswith(':x20') or feat.endswith(':x5'):
        return True
    return False


def open_db():
    raw = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', raw, 28, len(raw) // 4096)
    fd, tmp = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd)
    Path(tmp).write_bytes(bytes(raw))
    import sqlite3
    conn = sqlite3.connect(tmp)
    conn.row_factory = sqlite3.Row
    return conn, tmp


def load_v7_data():
    """Load same training data as V7: LS + grafana + V6 FN + existing negatives + 317-hard."""
    items = []
    # LS
    raw = (BASE / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    qd = json.loads(raw)
    ls_seen = set()
    for v in qd.values():
        try:
            cat = v.get('category')
            if cat not in ('child','teen','adult'): continue
            age = (v.get('age') or {}).get('ageFrom')
            # V7 used ageFrom-based label; we use category
            lbl = cat
            sd = v.get('siglip2_details')
            if isinstance(sd, str):
                try: sd = ast.literal_eval(sd)
                except: sd = None
            det = (sd or {}).get('underage') or {}
            if not det: continue
            labels = det.get('labels') or {}
            gid = 'ls_' + str(v['task_id'])
            if gid in ls_seen: continue
            ls_seen.add(gid)
            items.append({'id': gid, 'label': lbl,
                          'underage_labels': labels.get('underage') or {},
                          'adult_labels': labels.get('adult') or {},
                          'no_underage_labels': {},
                          'weight': 1.0, 'source': 'ls'})
        except: pass

    # Grafana ALL (как в train_lgbm_v7)
    conn, tmp = open_db()
    rows = conn.execute("""
        SELECT id, label, piper_result, export_batch FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
    """).fetchall()
    conn.close(); os.unlink(tmp)
    for r in rows:
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
                          'weight': w, 'source': 'graf'})
        except: pass

    # V6 hard FN positives
    p = DATA / 'v6_fn_hard_positives.json'
    if p.exists():
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': r.get('label', 'child'),
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {}),
                          'no_underage_labels': {},
                          'weight': 1.0, 'source': 'v6fn'})

    # Existing negatives
    for fname in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fname
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult',
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {}),
                          'no_underage_labels': {},
                          'weight': 1.0, 'source': 'neg'})
    return items


def load_v10_data():
    """Same as V10: rescored data + extras."""
    items = []
    # rescored LS
    for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()):
        if not r.get('done'): continue
        if r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': 'ls_'+str(r['task_id']), 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 1.0, 'source': 'ls'})
    # V9 317 scores (rescored Grafana 317-session, weight 3x)
    for r in json.loads((DATA / 'v9_317_scores.json').read_text()):
        if not r.get('done'): continue
        if r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': r['id'], 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 3.0, 'source': 'graf317'})
    # eval616_rescored (Grafana 17-May only as additional data)
    e616 = json.loads((DATA / 'eval616_rescored.json').read_text())
    for r in e616:
        if not r.get('done'): continue
        if r.get('label') not in ('child','teen','adult'): continue
        if r.get('kind') == 'ls': continue   # уже в ls_holdout
        items.append({'id': r['id'], 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 1.0, 'source': 'graf17'})
    # Existing negatives (no_underage missing — that's ok, they'll be zero)
    for fname in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fname
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult',
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {}),
                          'no_underage_labels': {},
                          'weight': 1.0, 'source': 'neg'})
    return items


def build_features(items, exclude_x20=False, include_no_underage=False):
    feat_set = set()
    for it in items:
        for k in (it.get('underage_labels') or {}):
            f = sanitize(k)
            if exclude_x20 and is_x20(f): continue
            feat_set.add(f)
        for k in (it.get('adult_labels') or {}):
            feat_set.add('adult__' + sanitize(k))
        if include_no_underage:
            for k in (it.get('no_underage_labels') or {}):
                feat_set.add('no_underage__' + sanitize(k))
    feats = sorted(feat_set)
    idx = {f: i for i, f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    w = np.ones(len(items), dtype=np.float32)
    for i, it in enumerate(items):
        for k, v in (it.get('underage_labels') or {}).items():
            f = sanitize(k)
            if exclude_x20 and is_x20(f): continue
            if f in idx: X[i, idx[f]] = float(v)
        for k, v in (it.get('adult_labels') or {}).items():
            f = 'adult__' + sanitize(k)
            if f in idx: X[i, idx[f]] = float(v)
        if include_no_underage:
            for k, v in (it.get('no_underage_labels') or {}).items():
                f = 'no_underage__' + sanitize(k)
                if f in idx: X[i, idx[f]] = float(v)
        y[i] = 1 if it['label'] in ('child','teen') else 0
        w[i] = it.get('weight', 1.0)
    return X, y, w, feats


def train_lgbm(X, y, w, n_trees=150, lr=0.04, num_leaves=15):
    params = dict(objective='binary', metric='auc', num_leaves=num_leaves,
                  learning_rate=lr, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=3, lambda_l1=0.1, lambda_l2=0.1,
                  is_unbalance=True, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        ds = lgb.Dataset(X[tr], label=y[tr], weight=w[tr])
        m = lgb.train(params, ds, num_boost_round=n_trees)
        aucs.append(roc_auc_score(y[va], m.predict(X[va])))
    final = lgb.train(params, lgb.Dataset(X, label=y, weight=w), num_boost_round=n_trees)
    return final, float(np.mean(aucs)), float(np.std(aucs))


def main():
    print('=== V7nx: same as V7, exclude :x20/:x5 features ===', flush=True)
    items_v7 = load_v7_data()
    print(f'  items: {len(items_v7)}', flush=True)
    X, y, w, feats = build_features(items_v7, exclude_x20=True, include_no_underage=False)
    print(f'  features: {len(feats)} (V7 had 314 — we should be ~302)', flush=True)
    model, auc, std = train_lgbm(X, y, w, n_trees=150, lr=0.04)
    print(f'  CV AUC: {auc:.4f} ± {std:.4f}', flush=True)
    model.save_model(str(DATA / 'lgbm_underage_v7nx.txt'))
    (DATA / 'lgbm_v7nx_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / 'lgbm_v7nx_meta.json').write_text(json.dumps({
        'version': 'v7nx', 'parent': 'v7', 'excluded': ':x20/:x5 multiplier tags',
        'n_samples': len(items_v7), 'n_features': len(feats),
        'cv_auc': auc, 'cv_std': std, 'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print('\n=== V10nx: same as V10, exclude :x20/:x5 ===', flush=True)
    items_v10 = load_v10_data()
    print(f'  items: {len(items_v10)}', flush=True)
    X, y, w, feats = build_features(items_v10, exclude_x20=True, include_no_underage=True)
    print(f'  features: {len(feats)} (V10 had 510 — we should be ~498)', flush=True)
    model, auc, std = train_lgbm(X, y, w, n_trees=150, lr=0.04)
    print(f'  CV AUC: {auc:.4f} ± {std:.4f}', flush=True)
    model.save_model(str(DATA / 'lgbm_underage_v10nx.txt'))
    (DATA / 'lgbm_v10nx_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / 'lgbm_v10nx_meta.json').write_text(json.dumps({
        'version': 'v10nx', 'parent': 'v10', 'excluded': ':x20/:x5 multiplier tags',
        'n_samples': len(items_v10), 'n_features': len(feats),
        'cv_auc': auc, 'cv_std': std, 'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print('\nDone. Now run benchmark to compare.', flush=True)


if __name__ == '__main__':
    main()
