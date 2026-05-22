#!/usr/bin/env python3
"""
V11 = V10pa + Tom-style BCI feature split + hard-neg mining x20.

Changes over V10pa:
  1. Add 4 BCI aggregates (computed on raw underage scores):
       _child_body          noisy-OR over BODY labels (anatomy/sex-act)
       _child_context       noisy-OR over CONTEXT labels (scene/clothing/pose)
       _child_interaction   noisy-OR over INTERACTION labels (:x adult+child)
       _body_vs_context     body / (body + context)
  2. Hard-neg mining: 71 V10-introduced adult FPs upweighted x20 in training.
  3. Same Path A treatment: :x20 multiplier OFF, no_underage merged into adult.

Output:
  data/lgbm_underage_v11.txt
  data/lgbm_v11_features.json
  data/lgbm_v11_meta.json
"""
import json, re, struct, os, tempfile, datetime, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')


def unmultiply(key, val):
    """Reverse adjusted = min(raw * mult, 0.999)."""
    m = X20_RE.search(key)
    if not m: return float(val)
    mult = float(m.group(1))
    if val >= 0.999: return 0.999 / mult
    return float(val) / mult


def strip_mult(k):
    return X20_RE.sub('', k)


def sanitize(n):
    return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')


def noisy_or(scores_dict, key_filter):
    """Compute noisy-OR over keys matching filter."""
    p = 1.0
    for k, v in scores_dict.items():
        if key_filter(k):
            p *= 1.0 - float(v)
    return 1.0 - p


def transform_item(item):
    """Apply path A transform: strip :x20, unmultiply, merge no_underage→adult.
       Returns (underage_dict, adult_dict) where adult includes merged no_underage."""
    u = {}
    for k, v in (item.get('underage_labels') or {}).items():
        raw = unmultiply(k, v)
        k2 = strip_mult(k)
        u[k2] = max(u.get(k2, 0.0), raw)
    a = dict((k, float(v)) for k, v in (item.get('adult_labels') or {}).items())
    for k, v in (item.get('no_underage_labels') or {}).items():
        a[k] = max(a.get(k, 0.0), float(v))
    return u, a


def load_data():
    """Load V10pa training items."""
    items = []
    for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': 'ls_'+str(r['task_id']), 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 1.0, 'source': 'ls'})
    for r in json.loads((DATA / 'v9_317_scores.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': r['id'], 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 3.0, 'source': 'graf317'})
    for r in json.loads((DATA / 'eval616_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        if r.get('kind') == 'ls': continue
        items.append({'id': r['id'], 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 1.0, 'source': 'graf17'})
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult',
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {}),
                          'no_underage_labels': {},
                          'weight': 1.0, 'source': 'neg'})
    return items


def add_hard_negatives(items, weight_x=20):
    """Add 71 V10-introduced FPs from v10_diff_analysis.json with x20 weight."""
    diff = json.loads((DATA / 'v10_diff_analysis.json').read_text())
    fps = diff.get('v10_fps_adults', [])
    print(f'  hard-neg pool: {len(fps)} V10 adult FPs', flush=True)
    # These items have top_underage_new / top_adult_new / top_no_underage as lists of (key, val) tuples.
    # We need to reconstruct the FULL underage_labels / adult_labels dicts.
    # The diff file has only top-8 features but for training we need everything.
    # → Load from ls_holdout_rescored.json by task_id.
    ls = {f"ls_{r['task_id']}": r for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()) if r.get('done')}
    added = 0
    for fp in fps:
        full = ls.get(fp['id'])
        if not full:
            continue
        items.append({'id': fp['id'], 'label': 'adult',
                      'underage_labels': full.get('underage_labels') or {},
                      'adult_labels': full.get('adult_labels') or {},
                      'no_underage_labels': full.get('no_underage_labels') or {},
                      'weight': float(weight_x), 'source': 'v10_fp_hardneg'})
        added += 1
    print(f'  added {added} hard negatives with weight={weight_x}', flush=True)
    return items


def build_features(items):
    """Build feature matrix: per-tag scores + 4 BCI aggregates."""
    # First pass: collect all feature names
    feat_set = set()
    for it in items:
        u, a = transform_item(it)
        for k in u: feat_set.add(sanitize(k))
        for k in a: feat_set.add('adult__' + sanitize(k))
    siglip_feats = sorted(feat_set)
    bci_feats = ['_child_body', '_child_context', '_child_interaction', '_body_vs_context']
    all_feats = siglip_feats + bci_feats
    idx = {f: i for i, f in enumerate(all_feats)}

    X = np.zeros((len(items), len(all_feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    w = np.ones(len(items), dtype=np.float32)
    body_match = lambda k: k in BODY_LABELS
    ctx_match  = lambda k: k in CONTEXT_LABELS
    int_match  = lambda k: k in INTERACTION_LABELS
    for i, it in enumerate(items):
        u, a = transform_item(it)
        for k, v in u.items():
            f = sanitize(k)
            if f in idx: X[i, idx[f]] = v
        for k, v in a.items():
            f = 'adult__' + sanitize(k)
            if f in idx: X[i, idx[f]] = float(v)
        # BCI aggregates on `u` (underage scores, raw, :x stripped)
        body = noisy_or(u, body_match)
        ctx  = noisy_or(u, ctx_match)
        inter = noisy_or(u, int_match)
        bc_tot = body + ctx
        body_vs_ctx = (body / bc_tot) if bc_tot > 0 else 0.0
        X[i, idx['_child_body']] = body
        X[i, idx['_child_context']] = ctx
        X[i, idx['_child_interaction']] = inter
        X[i, idx['_body_vs_context']] = body_vs_ctx
        y[i] = 1 if it['label'] in ('child','teen') else 0
        w[i] = it['weight']
    return X, y, w, all_feats


def main():
    print('Loading V10pa-style data...', flush=True)
    items = load_data()
    print(f'  base items: {len(items)}', flush=True)

    print('Adding hard negatives (x20)...', flush=True)
    items = add_hard_negatives(items, weight_x=20)
    print(f'  total items: {len(items)}', flush=True)

    print('Building feature matrix with BCI...', flush=True)
    X, y, w, feats = build_features(items)
    print(f'  X shape: {X.shape}  positives: {int(y.sum())}  negatives: {len(y) - int(y.sum())}', flush=True)
    print(f'  features: {len(feats)} (siglip: {len(feats)-4}, BCI: 4)', flush=True)
    # Check BCI feature stats
    bci_idx = {f: i for i, f in enumerate(feats) if f.startswith('_')}
    for bf, bi in bci_idx.items():
        nz = (X[:, bi] > 0).sum()
        mn = X[:, bi].mean()
        print(f'    {bf}: nonzero={nz}/{len(items)} ({100*nz/len(items):.0f}%) mean={mn:.3f}', flush=True)

    print('\n5-fold CV...', flush=True)
    params = dict(objective='binary', metric='auc', num_leaves=15,
                  learning_rate=0.04, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=3, lambda_l1=0.1, lambda_l2=0.1,
                  is_unbalance=True, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        ds = lgb.Dataset(X[tr], label=y[tr], weight=w[tr])
        m = lgb.train(params, ds, num_boost_round=150)
        auc = roc_auc_score(y[va], m.predict(X[va]))
        aucs.append(auc)
        print(f'  fold {fold+1}: AUC={auc:.4f}', flush=True)
    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    print(f'\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}', flush=True)

    print('\nTraining final model...', flush=True)
    final = lgb.train(params, lgb.Dataset(X, label=y, weight=w), num_boost_round=150)
    final.save_model(str(DATA / 'lgbm_underage_v11.txt'))
    (DATA / 'lgbm_v11_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / 'lgbm_v11_meta.json').write_text(json.dumps({
        'version': 'v11', 'parent': 'v10pa',
        'note': 'V10pa + BCI feature split + hard-neg mining x20 on 71 V10 FPs',
        'n_samples': len(items), 'n_features': len(feats),
        'cv_auc': mean_auc, 'cv_std': std_auc,
        'bci_aggregates': ['_child_body', '_child_context', '_child_interaction', '_body_vs_context'],
        'hard_neg_count': 71, 'hard_neg_weight': 20,
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))
    print(f'  saved: lgbm_underage_v11.txt + meta', flush=True)

    # Feature importance — посмотрим использует ли модель BCI features
    print('\nTop-30 features by gain:', flush=True)
    imp = final.feature_importance('gain')
    ranked = sorted(zip(feats, imp), key=lambda x: -x[1])
    for i, (f, g) in enumerate(ranked[:30]):
        marker = '*** BCI' if f.startswith('_') else ('  adlt' if f.startswith('adult__') else '   und')
        print(f'  {i+1:>2}. {marker}  {f[:50]:<52} gain={g:>8.0f}', flush=True)

    bci_gain = sum(g for f, g in zip(feats, imp) if f.startswith('_'))
    total_gain = sum(imp)
    print(f'\nBCI aggregates total gain: {bci_gain:.0f} ({100*bci_gain/total_gain:.1f}% of total)', flush=True)


if __name__ == '__main__':
    main()
