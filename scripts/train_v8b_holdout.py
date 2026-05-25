#!/usr/bin/env python3
"""
train_v8b_holdout.py
--------------------
Re-train V8pas80-v2 excluding the 618 universal holdout items
(data/v11_test_split.json). Multi-seed (3 lgb seeds) for stability.
Outputs honest test metrics: AUC, per-class recall, adult FPR, plus
per-source breakdown (LS vs Grafana).

Outputs:
  data/lgbm_underage_v8b.txt
  data/lgbm_underage_v8bs80.txt
  data/lgbm_v8bs80_features.json
  data/lgbm_v8bs80_meta.json  — incl. honest test_auc, per-source, multi-seed
"""
import json, re, ast, sys, datetime
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')
BCI = ['_child_body','_child_context','_child_interaction','_body_vs_context']
LGB_SEEDS = [1, 42, 314]


def unmult(k, v):
    m = X20_RE.search(k)
    if not m: return float(v)
    mu = float(m.group(1))
    return 0.999/mu if v >= 0.999 else float(v)/mu


def strip_mult(k): return X20_RE.sub('', k)
def sanitize(n): return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')


def noisy_or(d, fn):
    p = 1.0
    for k, v in d.items():
        if fn(k): p *= 1.0 - float(v)
    return 1.0 - p


def transform(item):
    u = {}
    for k, v in (item.get('underage_labels') or {}).items():
        raw = unmult(k, v); k2 = strip_mult(k)
        u[k2] = max(u.get(k2, 0.0), raw)
    a = dict((k, float(v)) for k, v in (item.get('adult_labels') or {}).items())
    return u, a


def build_X(items, feats):
    idx = {f: i for i, f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    labels = []
    for i, it in enumerate(items):
        u, a = transform(it)
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
        if '_child_body' in idx: X[i, idx['_child_body']] = body
        if '_child_context' in idx: X[i, idx['_child_context']] = ctx
        if '_child_interaction' in idx: X[i, idx['_child_interaction']] = inter
        if '_body_vs_context' in idx: X[i, idx['_body_vs_context']] = (body/bc) if bc>0 else 0.0
        y[i] = 1 if it['label'] in ('child','teen') else 0
        labels.append(it['label'])
    return X, y, labels


def load_ls_items():
    raw = (BASE / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    qd = json.loads(raw)
    items = []
    for v in qd.values():
        cat = v.get('category')
        if cat not in ('child','teen','adult'): continue
        sd = v.get('siglip2_details')
        if isinstance(sd, str):
            try: sd = ast.literal_eval(sd)
            except: continue
        det = (sd or {}).get('underage') or {}
        if not det: continue
        labels = det.get('labels') or {}
        items.append({
            'id': f"ls_{v['task_id']}", 'label': cat, 'source': 'ls',
            'underage_labels': labels.get('underage') or {},
            'adult_labels':    labels.get('adult')    or {},
        })
    return items


def load_hard_neg():
    """Same hard-neg pool as train_v8pas80_v2.py.add_hard_negs_v2().

    v7pa_fps.json structure: {"v7pa_fps_adults": [{...underage_labels, adult_labels...}]}
    v8pas80_fps_grok_confirmed.json structure: list[{id, scene_type, ...}]
      → feature vectors looked up in v8pas80_top100_fps.json by id.
    """
    items = []
    fps_v7pa = json.loads((DATA / 'v7pa_fps.json').read_text()).get('v7pa_fps_adults', [])
    fps_grok = json.loads((DATA / 'v8pas80_fps_grok_confirmed.json').read_text())
    grok_map = {f['id']: f for f in json.loads((DATA / 'v8pas80_top100_fps.json').read_text())}
    seen = set()
    for fp in fps_v7pa:
        fid = fp.get('id')
        if not fid or fid in seen: continue
        items.append({'id': fid, 'label': 'adult', 'source': 'hardneg',
                      'underage_labels': fp.get('underage_labels') or {},
                      'adult_labels':    fp.get('adult_labels')    or {}})
        seen.add(fid)
    for fp in fps_grok:
        fid = fp.get('id')
        if not fid or fid in seen: continue
        feat = grok_map.get(fid)
        if not feat: continue
        items.append({'id': fid, 'label': 'adult', 'source': 'hardneg',
                      'underage_labels': feat.get('underage_labels') or {},
                      'adult_labels':    feat.get('adult_labels')    or {}})
        seen.add(fid)
    return items


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
    print('=== V8b — train on V8 sources EXCLUDING universal v11_test_split ===\n', flush=True)
    test_ids = set(json.loads((DATA / 'v11_test_split.json').read_text())['test_ids'])
    print(f'  universal test set: {len(test_ids)} items\n', flush=True)

    ls_items = load_ls_items()
    print(f'  LS items: {len(ls_items)}', flush=True)
    train_ls = [it for it in ls_items if it['id'] not in test_ids]
    test_ls  = [it for it in ls_items if it['id'] in test_ids]
    print(f'    → train (excluding test): {len(train_ls)}', flush=True)
    print(f'    → test (in universal holdout): {len(test_ls)}', flush=True)

    hard_neg = load_hard_neg()
    # Filter hard-neg out of training if any id is in test set
    hard_neg_in_test = [h for h in hard_neg if h['id'] in test_ids]
    hard_neg_clean   = [h for h in hard_neg if h['id'] not in test_ids]
    print(f'\n  hard-neg pool: {len(hard_neg)}  in test: {len(hard_neg_in_test)}  clean: {len(hard_neg_clean)}', flush=True)

    train_items = train_ls + hard_neg_clean  # ×20 weight applied below via dataset weight
    print(f'\n  train total: {len(train_items)}  (LS={len(train_ls)} + hard-neg={len(hard_neg_clean)})', flush=True)

    # Build feature universe from train items
    feat_set = set()
    for it in train_items:
        u, a = transform(it)
        for k in u: feat_set.add(sanitize(k))
        for k in a: feat_set.add('adult__' + sanitize(k))
    feats = sorted(feat_set) + BCI
    print(f'  features: {len(feats)} (siglip: {len(feats)-4}, BCI: 4)\n', flush=True)

    X_tr, y_tr, _ = build_X(train_items, feats)
    weight = np.ones(len(train_items), dtype=np.float32)
    for i, it in enumerate(train_items):
        if it['source'] == 'hardneg':
            weight[i] = 20.0

    X_te, y_te, labels_te = build_X(test_ls, feats)
    sources_te = [it['source'] for it in test_ls]  # all 'ls' for V8

    # ── Multi-seed training (full + slim) ────────────────────────────────────
    base_params = dict(objective='binary', metric='auc', num_leaves=31,
                       learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.9,
                       min_child_samples=5, is_unbalance=True, verbose=-1)

    results_full = []
    results_slim = []
    last_full_model = None
    last_slim_feats = None
    last_slim_model = None

    for sd in LGB_SEEDS:
        params = {**base_params, 'seed': sd, 'data_random_seed': sd, 'feature_fraction_seed': sd, 'bagging_seed': sd}
        print(f'--- seed={sd} ---', flush=True)

        # full
        m_full = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, weight=weight), num_boost_round=200)
        sc_full = m_full.predict(X_te)
        auc_full = roc_auc_score(y_te, sc_full) if len(set(y_te)) > 1 else None
        m_full_metrics = metrics_at_thr(sc_full, labels_te, thr=0.30)
        results_full.append({'seed': sd, 'auc': auc_full, **m_full_metrics})
        last_full_model = m_full
        print(f'  FULL  AUC={auc_full:.4f}  child={m_full_metrics["child_recall"]:.3f}  teen={m_full_metrics["teen_recall"]:.3f}  fpr={m_full_metrics["adult_fpr"]:.3f}', flush=True)

        # slim — prune top-80 by gain
        imp = m_full.feature_importance('gain')
        bci_set = set(BCI)
        ranked = sorted([(f, g) for f, g in zip(feats, imp) if f not in bci_set], key=lambda x: -x[1])
        slim_feats = [f for f, _ in ranked[:80 - len(BCI)]] + BCI
        keep_idx = [feats.index(f) for f in slim_feats]
        X_tr_s = X_tr[:, keep_idx]
        X_te_s = X_te[:, keep_idx]
        m_slim = lgb.train(params, lgb.Dataset(X_tr_s, label=y_tr, weight=weight), num_boost_round=200)
        sc_slim = m_slim.predict(X_te_s)
        auc_slim = roc_auc_score(y_te, sc_slim) if len(set(y_te)) > 1 else None
        m_slim_metrics = metrics_at_thr(sc_slim, labels_te, thr=0.30)
        results_slim.append({'seed': sd, 'auc': auc_slim, **m_slim_metrics})
        last_slim_model = m_slim
        last_slim_feats = slim_feats
        print(f'  SLIM  AUC={auc_slim:.4f}  child={m_slim_metrics["child_recall"]:.3f}  teen={m_slim_metrics["teen_recall"]:.3f}  fpr={m_slim_metrics["adult_fpr"]:.3f}\n', flush=True)

    # Aggregate multi-seed stats
    def aggregate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    summary_full = {
        'auc':          aggregate(results_full, 'auc'),
        'child_recall': aggregate(results_full, 'child_recall'),
        'teen_recall':  aggregate(results_full, 'teen_recall'),
        'adult_fpr':    aggregate(results_full, 'adult_fpr'),
        'per_seed':     results_full,
    }
    summary_slim = {
        'auc':          aggregate(results_slim, 'auc'),
        'child_recall': aggregate(results_slim, 'child_recall'),
        'teen_recall':  aggregate(results_slim, 'teen_recall'),
        'adult_fpr':    aggregate(results_slim, 'adult_fpr'),
        'per_seed':     results_slim,
    }

    # Per-source breakdown (V8 holdout = LS only, so just LS bucket)
    by_source = {}
    for src in set(sources_te):
        idx = [i for i, s in enumerate(sources_te) if s == src]
        if not idx: continue
        sub_labels = [labels_te[i] for i in idx]
        # use last_slim_model on subset
        sub_X = X_te[:, [feats.index(f) for f in last_slim_feats]][idx]
        sub_sc = last_slim_model.predict(sub_X)
        sub_y = np.array([1 if l in ('child','teen') else 0 for l in sub_labels])
        try:
            sub_auc = float(roc_auc_score(sub_y, sub_sc))
        except Exception:
            sub_auc = None
        sub_m = metrics_at_thr(sub_sc, sub_labels, thr=0.30)
        by_source[src] = {'n': len(idx), 'auc': sub_auc, **sub_m}

    # Save model + meta
    last_full_model.save_model(str(DATA / 'lgbm_underage_v8b.txt'))
    last_slim_model.save_model(str(DATA / 'lgbm_underage_v8bs80.txt'))
    (DATA / 'lgbm_v8bs80_features.json').write_text(json.dumps(last_slim_feats, indent=2))
    (DATA / 'lgbm_v8bs80_meta.json').write_text(json.dumps({
        'version': 'v8bs80',
        'parent':  'v8b (V8pas80-v2 re-trained excluding v11_test_split)',
        'split':   'data/v11_test_split.json (SEED=1337, stratified 80/20 by V11 training data)',
        'n_train': len(train_items),
        'n_test':  len(test_ls),
        'test_source_breakdown': {'ls': len(test_ls), 'grafana': 0},
        'note':    'V8 trains on qwen3 LS only — the universal holdout contains 438 LS items '
                   '(V8 saw these in original training; this re-train excludes them).',
        'multi_seed_full': summary_full,
        'multi_seed_slim': summary_slim,
        'by_source_slim':  by_source,
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print('=== V8bs80 (slim) HONEST METRICS on 438-LS holdout ===', flush=True)
    print(f'  AUC:        {summary_slim["auc"]["mean"]:.4f} ± {summary_slim["auc"]["std"]:.4f}', flush=True)
    print(f'  child rec:  {summary_slim["child_recall"]["mean"]:.3f} ± {summary_slim["child_recall"]["std"]:.3f}', flush=True)
    print(f'  teen rec:   {summary_slim["teen_recall"]["mean"]:.3f} ± {summary_slim["teen_recall"]["std"]:.3f}', flush=True)
    print(f'  adult FPR:  {summary_slim["adult_fpr"]["mean"]:.3f} ± {summary_slim["adult_fpr"]["std"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
