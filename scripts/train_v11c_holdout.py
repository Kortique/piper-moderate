#!/usr/bin/env python3
"""
train_v11c_holdout.py
---------------------
V11 retrained on the EXTENDED scope: LS + Grafana + K30 (~7300 train items).
Uses data/v11_native_scores.json as the unified 317-tag input source for all
items (rescored via V11 native pipeline ce79f7e299).
Labels come from gallery.db (latest confirmed labels).
Honest holdout = data/v11_test_split_2026.json (1828 items).

Outputs:
  data/lgbm_underage_v11c.txt       — full model
  data/lgbm_underage_v11cs80.txt    — slim 80-feat (top by gain + 4 BCI)
  data/lgbm_v11cs80_features.json
  data/lgbm_v11cs80_meta.json
"""
import json, re, sys, sqlite3, datetime, struct
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
    for k, v in (item.get('no_underage_labels') or {}).items():
        a[k] = max(a.get(k, 0.0), float(v))
    return u, a


def open_db():
    db = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', db, 28, len(db)//4096)
    tmp = Path('/tmp/_v11c.db'); tmp.write_bytes(bytes(db))
    return sqlite3.connect(str(tmp))


def load_items():
    """Use v11_native_scores.json for features; overlay labels from gallery.db."""
    # Get current confirmed labels from gallery.db
    labels = {}
    conn = open_db()
    for r in conn.execute("SELECT task_id, age_from FROM ls_images WHERE age_from IS NOT NULL"):
        af = r[1]
        labels[f'ls_{r[0]}'] = 'child' if af <= 14 else ('teen' if af <= 17 else 'adult')
    for r in conn.execute("""SELECT id, label FROM grafana_pool
                              WHERE label_confirmed=1 AND label IN ('child','teen','adult')
                                AND (deleted IS NULL OR deleted=0)"""):
        labels[r[0]] = r[1]
    for r in conn.execute("""SELECT id, label FROM k30_pool
                              WHERE label_confirmed=1 AND label IN ('child','teen','adult')
                                AND (deleted IS NULL OR deleted=0)"""):
        labels[r[0]] = r[1]
    conn.close()
    print(f'  loaded labels from db: {len(labels)} items')

    # Pull features from v11_native_scores.json
    items = []
    src_map = {'labelstudio': 'ls', 'grafana': 'grafana', 'k30': 'k30'}
    skipped_nf = skipped_nolbl = 0
    for r in json.loads((DATA / 'v11_native_scores.json').read_text()):
        if not r.get('done'): continue
        if r.get('no_face'): skipped_nf += 1; continue
        lbl = labels.get(r['id'])
        if not lbl: skipped_nolbl += 1; continue
        u = r.get('underage_labels') or {}
        a = r.get('adult_labels') or {}
        nu = r.get('no_underage_labels') or {}
        if not u and not a: continue
        items.append({
            'id': r['id'], 'label': lbl, 'weight': 1.0,
            'source': src_map.get(r.get('source'), r.get('source') or 'unknown'),
            'underage_labels': u, 'adult_labels': a, 'no_underage_labels': nu,
        })
    print(f'  items with features+label: {len(items)}  skipped: no_face={skipped_nf} no_label={skipped_nolbl}')
    return items


def build_features(items, feat_names=None):
    if feat_names is None:
        feat_set = set()
        for it in items:
            u, a = transform(it)
            for k in u: feat_set.add(sanitize(k))
            for k in a: feat_set.add('adult__' + sanitize(k))
        feat_names = sorted(feat_set) + BCI
    idx = {f: i for i, f in enumerate(feat_names)}
    X = np.zeros((len(items), len(feat_names)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    w = np.ones(len(items), dtype=np.float32)
    labels_out = []
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
        w[i] = it['weight']
        labels_out.append(it['label'])
    return X, y, w, labels_out, feat_names


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
    print('=== V11c — extended scope (LS + Grafana + K30) ===\n', flush=True)
    split = json.loads((DATA / 'v11_test_split_2026.json').read_text())
    test_ids = set(split['test_ids'])
    print(f'  test split: {len(test_ids)} items (from data/v11_test_split_2026.json)', flush=True)

    items = load_items()
    train_items = [it for it in items if it['id'] not in test_ids]
    test_items  = [it for it in items if it['id'] in test_ids]
    print(f'\n  train: {len(train_items)}  test: {len(test_items)}', flush=True)

    from collections import Counter
    print(f'  train by source: {dict(Counter(it["source"] for it in train_items))}', flush=True)
    print(f'  test  by source: {dict(Counter(it["source"] for it in test_items))}', flush=True)
    print(f'  train by label:  {dict(Counter(it["label"]  for it in train_items))}', flush=True)
    print(f'  test  by label:  {dict(Counter(it["label"]  for it in test_items))}', flush=True)

    X_tr, y_tr, w_tr, _, feats = build_features(train_items)
    X_te, y_te, _, labels_te, _ = build_features(test_items, feats)
    sources_te = [it['source'] for it in test_items]
    print(f'\n  X_tr shape: {X_tr.shape}  pos: {int(y_tr.sum())}  neg: {len(y_tr)-int(y_tr.sum())}', flush=True)

    base_params = dict(objective='binary', metric='auc', num_leaves=15,
                       learning_rate=0.04, feature_fraction=0.8, bagging_fraction=0.8,
                       bagging_freq=5, min_child_samples=3, lambda_l1=0.1, lambda_l2=0.1,
                       is_unbalance=True, verbose=-1)

    results_full = []; results_slim = []
    last_full = None; last_slim = None; last_slim_feats = None
    for sd in LGB_SEEDS:
        params = {**base_params, 'seed': sd, 'data_random_seed': sd,
                  'feature_fraction_seed': sd, 'bagging_seed': sd}
        m_full = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, weight=w_tr), num_boost_round=200)
        sc_full = m_full.predict(X_te)
        auc_full = roc_auc_score(y_te, sc_full)
        mm_full = metrics_at_thr(sc_full, labels_te, thr=0.30)
        results_full.append({'seed': sd, 'auc': auc_full, **mm_full})
        last_full = m_full

        imp = m_full.feature_importance('gain')
        bci_set = set(BCI)
        ranked = sorted([(f, g) for f, g in zip(feats, imp) if f not in bci_set], key=lambda x: -x[1])
        slim_feats = [f for f, _ in ranked[:80 - len(BCI)]] + list(BCI)
        keep_idx = [feats.index(f) for f in slim_feats]
        X_tr_s = X_tr[:, keep_idx]; X_te_s = X_te[:, keep_idx]
        m_slim = lgb.train(params, lgb.Dataset(X_tr_s, label=y_tr, weight=w_tr), num_boost_round=200)
        sc_slim = m_slim.predict(X_te_s)
        auc_slim = roc_auc_score(y_te, sc_slim)
        mm_slim = metrics_at_thr(sc_slim, labels_te, thr=0.30)
        results_slim.append({'seed': sd, 'auc': auc_slim, **mm_slim})
        last_slim = m_slim; last_slim_feats = slim_feats
        print(f'  seed={sd}: FULL AUC={auc_full:.4f}  SLIM AUC={auc_slim:.4f}  '
              f'child={mm_slim["child_recall"]:.3f} teen={mm_slim["teen_recall"]:.3f} fpr={mm_slim["adult_fpr"]:.3f}', flush=True)

    def aggregate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    sf = {k: aggregate(results_full, k) for k in ('auc','child_recall','teen_recall','adult_fpr')}
    sf['per_seed'] = results_full
    ss = {k: aggregate(results_slim, k) for k in ('auc','child_recall','teen_recall','adult_fpr')}
    ss['per_seed'] = results_slim

    # per-source breakdown via last_slim
    by_source = {}
    for src in set(sources_te):
        idx_sub = [i for i, s in enumerate(sources_te) if s == src]
        if not idx_sub: continue
        sub_X = X_te[:, [feats.index(f) for f in last_slim_feats]][idx_sub]
        sub_lbl = [labels_te[i] for i in idx_sub]
        sub_y = np.array([1 if l in ('child','teen') else 0 for l in sub_lbl])
        sub_sc = last_slim.predict(sub_X)
        try: sub_auc = float(roc_auc_score(sub_y, sub_sc))
        except Exception: sub_auc = None
        sub_m = metrics_at_thr(sub_sc, sub_lbl, thr=0.30)
        by_source[src] = {'n': len(idx_sub), 'auc': sub_auc, **sub_m}

    last_full.save_model(str(DATA / 'lgbm_underage_v11c.txt'))
    last_slim.save_model(str(DATA / 'lgbm_underage_v11cs80.txt'))
    (DATA / 'lgbm_v11c_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / 'lgbm_v11cs80_features.json').write_text(json.dumps(last_slim_feats, indent=2))
    (DATA / 'lgbm_v11cs80_meta.json').write_text(json.dumps({
        'version': 'v11cs80',
        'parent':  'v11c (extended scope LS+Grafana+K30, v11 native input)',
        'split':   'data/v11_test_split_2026.json (SEED=1337, stratified by label+source)',
        'n_train': len(train_items),
        'n_test':  len(test_items),
        'multi_seed_full': sf,
        'multi_seed_slim': ss,
        'by_source_slim':  by_source,
        'n_features_full': len(feats),
        'n_features_slim': len(last_slim_feats),
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print('\n=== V11cs80 HONEST METRICS on extended holdout ===', flush=True)
    print(f'  AUC:        {ss["auc"]["mean"]:.4f} ± {ss["auc"]["std"]:.4f}', flush=True)
    print(f'  child rec:  {ss["child_recall"]["mean"]:.3f} ± {ss["child_recall"]["std"]:.3f}', flush=True)
    print(f'  teen rec:   {ss["teen_recall"]["mean"]:.3f} ± {ss["teen_recall"]["std"]:.3f}', flush=True)
    print(f'  adult FPR:  {ss["adult_fpr"]["mean"]:.3f} ± {ss["adult_fpr"]["std"]:.3f}', flush=True)
    print('\n  per-source:')
    for src, m in by_source.items():
        print(f'    {src:8} n={m["n"]:>4}  AUC={m["auc"]:.4f}  child={m["child_recall"]:.3f}  teen={m["teen_recall"]:.3f}  fpr={m["adult_fpr"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
