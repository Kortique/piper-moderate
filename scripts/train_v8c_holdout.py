#!/usr/bin/env python3
"""
train_v8c_holdout.py
--------------------
V8 retrained on extended scope: LS + Grafana + K30 (180-tag input, BCI aggregates,
hard-negs ×20). Honest holdout = data/v11_test_split_2026.json (1828 items).

Outputs:
  data/lgbm_underage_v8c.txt          — full model
  data/lgbm_underage_v8cs80.txt       — slim 80-feat (top by gain + 4 BCI)
  data/lgbm_v8cs80_features.json
  data/lgbm_v8cs80_meta.json
"""
import json, re, ast, struct, sqlite3, sys, datetime
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


def open_db():
    db = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', db, 28, len(db)//4096)
    tmp = Path('/tmp/_v8c.db'); tmp.write_bytes(bytes(db))
    return sqlite3.connect(str(tmp))


def load_ls():
    items = []
    raw = (BASE / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    qd = json.loads(raw)
    for v in qd.values():
        age = (v.get('age') or {}).get('ageFrom')
        if age is None: continue
        sd = v.get('siglip2_details')
        if isinstance(sd, str):
            try: sd = ast.literal_eval(sd)
            except: continue
        det = (sd or {}).get('underage') or {}
        if not det: continue
        labels = det.get('labels') or {}
        u = labels.get('underage') or {}; a = labels.get('adult') or {}
        if not u and not a: continue
        items.append({
            'id': f"ls_{v['task_id']}", 'source': 'ls',
            'label': 'child' if age <= 14 else ('teen' if age <= 17 else 'adult'),
            'underage_labels': u, 'adult_labels': a,
        })
    return items


def load_grafana():
    items = []
    conn = open_db()
    for row in conn.execute("""SELECT id, label, piper_result FROM grafana_pool
                                WHERE label_confirmed=1 AND label IN ('child','teen','adult')
                                  AND (deleted IS NULL OR deleted=0)"""):
        try:
            pr = json.loads(row[2]) if row[2] else {}
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            labels = det.get('labels', {})
            u = labels.get('underage') or {}; a = labels.get('adult') or {}
            if not u and not a: continue
            items.append({'id': row[0], 'source': 'grafana', 'label': row[1],
                          'underage_labels': u, 'adult_labels': a})
        except Exception: continue
    conn.close()
    return items


def load_k30():
    items = []
    labels = {}
    conn = open_db()
    for row in conn.execute("""SELECT id, label FROM k30_pool
                                WHERE label_confirmed=1 AND label IN ('child','teen','adult')
                                  AND (deleted IS NULL OR deleted=0)"""):
        labels[row[0]] = row[1]
    conn.close()
    for r in json.loads((DATA / 'k30_rescored.json').read_text()):
        if not r.get('done') or r.get('no_face'): continue
        lbl = labels.get(r['id'])
        if not lbl: continue
        u = r.get('underage_labels') or {}; a = r.get('adult_labels') or {}
        if not u and not a: continue
        items.append({'id': r['id'], 'source': 'k30', 'label': lbl,
                      'underage_labels': u, 'adult_labels': a})
    return items


def load_hard_neg():
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


def build_X(items, feats):
    idx = {f: i for i, f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
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
        labels_out.append(it['label'])
    return X, y, labels_out


def metrics_at_thr(scores, labels, thr=0.30):
    n = {'child':0,'teen':0,'adult':0}; blk = {'child':0,'teen':0,'adult':0}
    for s, l in zip(scores, labels):
        if l not in n: continue
        n[l] += 1
        if s >= thr: blk[l] += 1
    return {'child_recall': blk['child']/n['child'] if n['child'] else 0,
            'teen_recall':  blk['teen']/n['teen']  if n['teen']  else 0,
            'adult_fpr':    blk['adult']/n['adult']if n['adult'] else 0,
            'counts': {**{f'n_{k}':v for k,v in n.items()}, **{f'blk_{k}':v for k,v in blk.items()}}}


def main():
    print('=== V8c — extended scope (LS + Grafana + K30) ===\n', flush=True)
    split = json.loads((DATA / 'v11_test_split_2026.json').read_text())
    test_ids = set(split['test_ids'])

    ls = load_ls();  print(f'  LS:      {len(ls)}', flush=True)
    gr = load_grafana(); print(f'  Grafana: {len(gr)}', flush=True)
    k30 = load_k30(); print(f'  K30:     {len(k30)}', flush=True)
    all_main = ls + gr + k30
    train_main = [it for it in all_main if it['id'] not in test_ids]
    test_items = [it for it in all_main if it['id'] in test_ids]

    hard_neg = load_hard_neg()
    hard_neg_clean = [h for h in hard_neg if h['id'] not in test_ids]
    print(f'\n  hard-neg pool: {len(hard_neg)}  clean: {len(hard_neg_clean)}', flush=True)
    train_items = train_main + hard_neg_clean
    print(f'  train: {len(train_items)} (main={len(train_main)} + hardneg={len(hard_neg_clean)})', flush=True)
    print(f'  test:  {len(test_items)}', flush=True)

    from collections import Counter
    print(f'  train by source: {dict(Counter(it["source"] for it in train_items))}', flush=True)
    print(f'  test  by source: {dict(Counter(it["source"] for it in test_items))}', flush=True)

    feat_set = set()
    for it in train_items:
        u, a = transform(it)
        for k in u: feat_set.add(sanitize(k))
        for k in a: feat_set.add('adult__' + sanitize(k))
    feats = sorted(feat_set) + BCI
    print(f'  features: {len(feats)} (siglip: {len(feats)-4}, BCI: 4)\n', flush=True)

    X_tr, y_tr, _ = build_X(train_items, feats)
    w_tr = np.ones(len(train_items), dtype=np.float32)
    for i, it in enumerate(train_items):
        if it['source'] == 'hardneg':
            w_tr[i] = 20.0

    X_te, y_te, labels_te = build_X(test_items, feats)
    sources_te = [it['source'] for it in test_items]

    base_params = dict(objective='binary', metric='auc', num_leaves=31,
                       learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.9,
                       min_child_samples=5, is_unbalance=True, verbose=-1)
    results_full = []; results_slim = []
    last_full = None; last_slim = None; last_slim_feats = None
    for sd in LGB_SEEDS:
        params = {**base_params, 'seed': sd, 'data_random_seed': sd,
                  'feature_fraction_seed': sd, 'bagging_seed': sd}
        m_full = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, weight=w_tr), num_boost_round=250)
        sc_full = m_full.predict(X_te)
        auc_full = roc_auc_score(y_te, sc_full)
        mm_full = metrics_at_thr(sc_full, labels_te, thr=0.30)
        results_full.append({'seed': sd, 'auc': auc_full, **mm_full})
        last_full = m_full

        imp = m_full.feature_importance('gain')
        bci_set = set(BCI)
        ranked = sorted([(f, g) for f, g in zip(feats, imp) if f not in bci_set], key=lambda x: -x[1])
        slim_feats = [f for f, _ in ranked[:80 - len(BCI)]] + BCI
        keep_idx = [feats.index(f) for f in slim_feats]
        X_tr_s = X_tr[:, keep_idx]; X_te_s = X_te[:, keep_idx]
        m_slim = lgb.train(params, lgb.Dataset(X_tr_s, label=y_tr, weight=w_tr), num_boost_round=250)
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

    last_full.save_model(str(DATA / 'lgbm_underage_v8c.txt'))
    last_slim.save_model(str(DATA / 'lgbm_underage_v8cs80.txt'))
    (DATA / 'lgbm_v8c_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / 'lgbm_v8cs80_features.json').write_text(json.dumps(last_slim_feats, indent=2))
    (DATA / 'lgbm_v8cs80_meta.json').write_text(json.dumps({
        'version': 'v8cs80',
        'parent':  'v8 (extended scope LS+Grafana+K30+hardneg×20, BCI, slim80)',
        'split': 'data/v11_test_split_2026.json',
        'n_train': len(train_items), 'n_test': len(test_items),
        'multi_seed_full': sf, 'multi_seed_slim': ss,
        'by_source_slim': by_source,
        'n_features_full': len(feats), 'n_features_slim': len(last_slim_feats),
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print(f'\n=== V8cs80 HONEST METRICS ===', flush=True)
    print(f'  AUC: {ss["auc"]["mean"]:.4f} ± {ss["auc"]["std"]:.4f}', flush=True)
    print(f'  child: {ss["child_recall"]["mean"]:.3f}  teen: {ss["teen_recall"]["mean"]:.3f}  fpr: {ss["adult_fpr"]["mean"]:.3f}', flush=True)
    print('\n  per-source:')
    for src, m in by_source.items():
        print(f'    {src:8} n={m["n"]:>4}  AUC={m["auc"]:.4f}  child={m["child_recall"]:.3f}  teen={m["teen_recall"]:.3f}  fpr={m["adult_fpr"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
