#!/usr/bin/env python3
"""
train_v11b_holdout.py
---------------------
Re-train V11 with an explicit, fixed 80/20 train/test split so the test set
gives us an honest out-of-sample evaluation.

Key difference from train_v11.py:
  * BEFORE adding hard-negatives, we set aside 20% (stratified by label) as TEST.
  * V11b trains on the remaining 80% (+ hard-neg pool ×20, same as V11).
  * Test items are EXCLUDED from training. We save their IDs to
    data/v11_test_split.json so the gallery can show a "Test only" view.
  * Final test-set metrics (AUC, per-class recall, adult FPR) saved to
    data/lgbm_v11bs80_holdout_meta.json.

Outputs:
  data/lgbm_underage_v11b.txt        — full 510-feat model trained on 80%
  data/lgbm_underage_v11bs80.txt     — slim 80-feat model (top-N by gain)
  data/lgbm_v11bs80_features.json
  data/lgbm_v11bs80_meta.json        — incl. honest test_auc / recall / fpr
  data/v11_test_split.json           — ids of 20% held-out items
"""
import json, re, datetime, sys, os
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')
BCI_FEATS = ['_child_body', '_child_context', '_child_interaction', '_body_vs_context']

SEED = 1337           # fixed for reproducible split (test set never changes)
LGB_SEEDS = [1, 42, 314]   # vary training seed across 3 runs to estimate stability


def unmultiply(key, val):
    m = X20_RE.search(key)
    if not m: return float(val)
    mult = float(m.group(1))
    if val >= 0.999: return 0.999 / mult
    return float(val) / mult


def strip_mult(k): return X20_RE.sub('', k)
def sanitize(n): return n.replace(':', '_x').replace('"','').replace("'",'').replace('{','').replace('}','')


def noisy_or(d, f):
    p = 1.0
    for k, v in d.items():
        if f(k): p *= 1.0 - float(v)
    return 1.0 - p


def transform_item(item):
    u = {}
    for k, v in (item.get('underage_labels') or {}).items():
        raw = unmultiply(k, v); k2 = strip_mult(k)
        u[k2] = max(u.get(k2, 0.0), raw)
    a = dict((k, float(v)) for k, v in (item.get('adult_labels') or {}).items())
    for k, v in (item.get('no_underage_labels') or {}).items():
        a[k] = max(a.get(k, 0.0), float(v))
    return u, a


def load_base_items():
    """Mirror of train_v11.load_data — but returns items keyed by id (no hard-neg yet)."""
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
        if r.get('kind') == 'ls': continue   # original train_v11 skipped these
        items.append({'id': r['id'], 'label': r['label'],
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {},
                      'weight': 1.0, 'source': 'graf17'})
    # de-dup on id (graf17 set sometimes overlaps with v317)
    seen = set(); uniq = []
    for it in items:
        if it['id'] in seen: continue
        seen.add(it['id']); uniq.append(it)
    return uniq


def add_hard_negatives(items, weight_x=20):
    """Same as train_v11 — add 71 V10-introduced FPs from v10_diff_analysis.json."""
    diff = json.loads((DATA / 'v10_diff_analysis.json').read_text())
    fps = diff.get('v10_fps_adults', [])
    ls_full = {f"ls_{r['task_id']}": r for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()) if r.get('done')}
    added = 0
    for fp in fps:
        full = ls_full.get(fp['id'])
        if not full: continue
        items.append({'id': fp['id']+'#hardneg', 'label': 'adult',
                      'underage_labels': full.get('underage_labels') or {},
                      'adult_labels': full.get('adult_labels') or {},
                      'no_underage_labels': full.get('no_underage_labels') or {},
                      'weight': float(weight_x), 'source': 'v10_fp_hardneg'})
        added += 1
    print(f'  added {added} hard negatives ×{weight_x}', flush=True)
    return items


def build_features(items):
    """Same as train_v11.build_features."""
    feat_set = set()
    for it in items:
        u, a = transform_item(it)
        for k in u: feat_set.add(sanitize(k))
        for k in a: feat_set.add('adult__' + sanitize(k))
    siglip_feats = sorted(feat_set)
    all_feats = siglip_feats + BCI_FEATS
    idx = {f: i for i, f in enumerate(all_feats)}
    X = np.zeros((len(items), len(all_feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    w = np.ones(len(items), dtype=np.float32)
    for i, it in enumerate(items):
        u, a = transform_item(it)
        for k, v in u.items():
            f = sanitize(k)
            if f in idx: X[i, idx[f]] = v
        for k, v in a.items():
            f = 'adult__' + sanitize(k)
            if f in idx: X[i, idx[f]] = float(v)
        body = noisy_or(u, lambda k: k in BODY_LABELS)
        ctx  = noisy_or(u, lambda k: k in CONTEXT_LABELS)
        inter = noisy_or(u, lambda k: k in INTERACTION_LABELS)
        bc = body + ctx
        X[i, idx['_child_body']] = body
        X[i, idx['_child_context']] = ctx
        X[i, idx['_child_interaction']] = inter
        X[i, idx['_body_vs_context']] = (body/bc) if bc>0 else 0.0
        y[i] = 1 if it['label'] in ('child','teen') else 0
        w[i] = it['weight']
    return X, y, w, all_feats


def metrics_at_thr(scores, labels, thr=0.30):
    n = {'child':0,'teen':0,'adult':0}
    blk = {'child':0,'teen':0,'adult':0}
    for s, l in zip(scores, labels):
        if l not in n: continue
        n[l] += 1
        if s >= thr: blk[l] += 1
    return {
        'child_recall': blk['child']/n['child'] if n['child'] else 0,
        'teen_recall':  blk['teen']/n['teen']  if n['teen']  else 0,
        'adult_fpr':    blk['adult']/n['adult']if n['adult'] else 0,
        'counts':       {**{f'n_{k}':v for k,v in n.items()}, **{f'blk_{k}':v for k,v in blk.items()}},
    }


def main():
    print('=== V11b — train with 80/20 holdout split ===\n', flush=True)
    base = load_base_items()
    print(f'  base items (after de-dup): {len(base)}', flush=True)
    by_label = {l: sum(1 for it in base if it['label']==l) for l in ['child','teen','adult']}
    print(f'  by label: {by_label}\n', flush=True)

    # STRATIFIED SPLIT — set aside 20% as test BEFORE adding hard-negatives
    base_idx = list(range(len(base)))
    base_labels = [it['label'] for it in base]
    train_idx, test_idx = train_test_split(
        base_idx, test_size=0.20, stratify=base_labels, random_state=SEED
    )
    train_items_base = [base[i] for i in train_idx]
    test_items       = [base[i] for i in test_idx]
    train_lab = {l: sum(1 for it in train_items_base if it['label']==l) for l in ['child','teen','adult']}
    test_lab  = {l: sum(1 for it in test_items if it['label']==l) for l in ['child','teen','adult']}
    print(f'  split (SEED={SEED}, stratified):', flush=True)
    print(f'    train: {len(train_items_base)} items  by_label={train_lab}', flush=True)
    print(f'    test : {len(test_items)} items  by_label={test_lab}\n', flush=True)

    # Save test split ids
    test_ids = [it['id'] for it in test_items]
    (DATA / 'v11_test_split.json').write_text(json.dumps({
        'seed': SEED,
        'test_ids': test_ids,
        'by_label': test_lab,
        'created_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))
    print(f'  saved test split → data/v11_test_split.json ({len(test_ids)} ids)\n', flush=True)

    # Add hard-negatives ONLY to train (test has none — they would double-dip)
    train_items = add_hard_negatives(train_items_base[:], weight_x=20)
    print(f'  train total (after hard-neg): {len(train_items)}\n', flush=True)

    # Build matrices — keep same feature set across train & test
    # First build TRAIN to derive feature set
    X_tr, y_tr, w_tr, feats = build_features(train_items)
    print(f'  X_tr shape: {X_tr.shape}  pos: {int(y_tr.sum())}  neg: {len(y_tr)-int(y_tr.sum())}', flush=True)
    print(f'  features: {len(feats)} (siglip: {len(feats)-4}, BCI: 4)\n', flush=True)

    # Build X_test with the SAME feature index (no new features)
    idx = {f: i for i, f in enumerate(feats)}
    X_te = np.zeros((len(test_items), len(feats)), dtype=np.float32)
    test_labels = []
    for i, it in enumerate(test_items):
        u, a = transform_item(it)
        for k, v in u.items():
            f = sanitize(k)
            if f in idx: X_te[i, idx[f]] = v
        for k, v in a.items():
            f = 'adult__' + sanitize(k)
            if f in idx: X_te[i, idx[f]] = float(v)
        body = noisy_or(u, lambda k: k in BODY_LABELS)
        ctx  = noisy_or(u, lambda k: k in CONTEXT_LABELS)
        inter = noisy_or(u, lambda k: k in INTERACTION_LABELS)
        bc = body + ctx
        if '_child_body' in idx: X_te[i, idx['_child_body']] = body
        if '_child_context' in idx: X_te[i, idx['_child_context']] = ctx
        if '_child_interaction' in idx: X_te[i, idx['_child_interaction']] = inter
        if '_body_vs_context' in idx: X_te[i, idx['_body_vs_context']] = (body/bc) if bc>0 else 0.0
        test_labels.append(it['label'])
    y_te = np.array([1 if l in ('child','teen') else 0 for l in test_labels], dtype=np.int8)

    # === Multi-seed training (full + slim) ===
    print(f'Training V11b (full {X_tr.shape[1]} features) with {len(LGB_SEEDS)} seeds...', flush=True)
    base_params = dict(objective='binary', metric='auc', num_leaves=15,
                       learning_rate=0.04, feature_fraction=0.8, bagging_fraction=0.8,
                       bagging_freq=5, min_child_samples=3, lambda_l1=0.1, lambda_l2=0.1,
                       is_unbalance=True, verbose=-1)
    results_full = []; results_slim = []
    last_full = None; last_slim = None; last_slim_feats = None
    for sd in LGB_SEEDS:
        params = {**base_params, 'seed': sd, 'data_random_seed': sd,
                  'feature_fraction_seed': sd, 'bagging_seed': sd}
        # full
        m_full = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, weight=w_tr), num_boost_round=150)
        sc_full = m_full.predict(X_te)
        auc_full = roc_auc_score(y_te, sc_full)
        mm_full = metrics_at_thr(sc_full, test_labels, thr=0.30)
        results_full.append({'seed': sd, 'auc': auc_full, **mm_full})
        last_full = m_full
        # slim — prune top-80 by this seed's gain
        imp = m_full.feature_importance('gain')
        bci_set = set(BCI_FEATS)
        ranked = sorted([(f, g) for f, g in zip(feats, imp) if f not in bci_set], key=lambda x: -x[1])
        slim_feats = [f for f, _ in ranked[:80 - len(BCI_FEATS)]] + list(BCI_FEATS)
        keep_idx = [feats.index(f) for f in slim_feats]
        X_tr_s = X_tr[:, keep_idx]; X_te_s = X_te[:, keep_idx]
        m_slim = lgb.train(params, lgb.Dataset(X_tr_s, label=y_tr, weight=w_tr), num_boost_round=150)
        sc_slim = m_slim.predict(X_te_s)
        auc_slim = roc_auc_score(y_te, sc_slim)
        mm_slim = metrics_at_thr(sc_slim, test_labels, thr=0.30)
        results_slim.append({'seed': sd, 'auc': auc_slim, **mm_slim})
        last_slim = m_slim; last_slim_feats = slim_feats
        print(f'  seed={sd}: FULL AUC={auc_full:.4f}  SLIM AUC={auc_slim:.4f}', flush=True)

    def aggregate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    summary_full = {k: aggregate(results_full, k) for k in ('auc','child_recall','teen_recall','adult_fpr')}
    summary_full['per_seed'] = results_full
    summary_slim = {k: aggregate(results_slim, k) for k in ('auc','child_recall','teen_recall','adult_fpr')}
    summary_slim['per_seed'] = results_slim

    # Per-source breakdown — use last_slim on subsets
    sources_te = [('ls' if it['id'].startswith('ls_') else 'grafana') for it in test_items]
    by_source = {}
    for src in set(sources_te):
        idx_subset = [i for i, s in enumerate(sources_te) if s == src]
        if not idx_subset: continue
        sub_labels = [test_labels[i] for i in idx_subset]
        sub_X = X_te[:, [feats.index(f) for f in last_slim_feats]][idx_subset]
        sub_sc = last_slim.predict(sub_X)
        sub_y = np.array([1 if l in ('child','teen') else 0 for l in sub_labels])
        try: sub_auc = float(roc_auc_score(sub_y, sub_sc))
        except Exception: sub_auc = None
        sub_m = metrics_at_thr(sub_sc, sub_labels, thr=0.30)
        by_source[src] = {'n': len(idx_subset), 'auc': sub_auc, **sub_m}

    # Save artifacts (use LAST seed's models — multi-seed numbers go in meta)
    last_full.save_model(str(DATA / 'lgbm_underage_v11b.txt'))
    (DATA / 'lgbm_v11b_features.json').write_text(json.dumps(feats, indent=2))
    last_slim.save_model(str(DATA / 'lgbm_underage_v11bs80.txt'))
    (DATA / 'lgbm_v11bs80_features.json').write_text(json.dumps(last_slim_feats, indent=2))

    print('\n=== V11b HONEST METRICS on 618-item universal holdout ===', flush=True)
    print(f'  FULL: AUC={summary_full["auc"]["mean"]:.4f} ± {summary_full["auc"]["std"]:.4f}', flush=True)
    print(f'  SLIM: AUC={summary_slim["auc"]["mean"]:.4f} ± {summary_slim["auc"]["std"]:.4f}', flush=True)
    print(f'        child={summary_slim["child_recall"]["mean"]:.3f} ± {summary_slim["child_recall"]["std"]:.3f}', flush=True)
    print(f'        teen ={summary_slim["teen_recall"]["mean"]:.3f} ± {summary_slim["teen_recall"]["std"]:.3f}', flush=True)
    print(f'        fpr  ={summary_slim["adult_fpr"]["mean"]:.3f} ± {summary_slim["adult_fpr"]["std"]:.3f}', flush=True)
    for src, m in by_source.items():
        print(f'    [{src:8}] n={m["n"]:>3} AUC={m["auc"]:.4f} child={m["child_recall"]:.3f} teen={m["teen_recall"]:.3f} fpr={m["adult_fpr"]:.3f}', flush=True)

    # rename for downstream usage
    auc_slim_mean = summary_slim['auc']['mean']
    auc_full_mean = summary_full['auc']['mean']
    metr_full = {'_multi_seed': summary_full}
    metr_slim = {'_multi_seed': summary_slim}
    auc_full = auc_full_mean
    auc_slim = auc_slim_mean

    # Comparison vs current V11s80 in-sample (loaded from existing meta)
    cur_meta_p = DATA / 'lgbm_v11s80_meta.json'
    cur_cv_auc = None
    if cur_meta_p.exists():
        cur_cv_auc = json.loads(cur_meta_p.read_text()).get('cv_auc')

    # Honest meta
    out_meta = {
        'version':    'v11bs80',
        'parent':     'v11b',
        'note':       'V11 retrained with explicit 80/20 stratified holdout split (SEED=1337). '
                      'Multi-seed lgb training. test_* below are HONEST out-of-sample metrics.',
        'split_seed': SEED,
        'lgb_seeds':  LGB_SEEDS,
        'split':      {'train_base': len(train_items_base), 'train_with_hardneg': len(train_items),
                       'test': len(test_items),
                       'train_by_label': train_lab, 'test_by_label': test_lab},
        'multi_seed_full': summary_full,
        'multi_seed_slim': summary_slim,
        'by_source_slim':  by_source,
        'compare_to_v11s80_cv_auc': cur_cv_auc,
        'n_features_full': len(feats),
        'n_features_slim': len(last_slim_feats),
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }
    (DATA / 'lgbm_v11bs80_meta.json').write_text(json.dumps(out_meta, indent=2))
    print('=== Summary ===', flush=True)
    print(f'  V11s80 (current) CV AUC = {cur_cv_auc:.4f}' if cur_cv_auc else '  (no prior CV AUC)', flush=True)
    print(f'  V11b   (full)    TEST AUC = {auc_full:.4f}  (honest 20% holdout)', flush=True)
    print(f'  V11bs80 (slim)   TEST AUC = {auc_slim:.4f}', flush=True)
    print(f'\n  ALL ARTIFACTS:', flush=True)
    print(f'    data/lgbm_underage_v11b.txt        — full model (80% train)', flush=True)
    print(f'    data/lgbm_underage_v11bs80.txt     — slim 80-feat model', flush=True)
    print(f'    data/lgbm_v11bs80_features.json    — slim feature list', flush=True)
    print(f'    data/lgbm_v11bs80_meta.json        — honest test metrics', flush=True)
    print(f'    data/v11_test_split.json           — test ids ({len(test_ids)})', flush=True)


if __name__ == '__main__':
    main()
