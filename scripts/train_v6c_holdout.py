#!/usr/bin/env python3
"""
train_v6c_holdout.py
--------------------
V6 retrained on extended scope: LS + Grafana + K30 (180-tag input).
Uses qwen3_age_results.json (LS), gallery.db.piper_result (Grafana),
data/k30_rescored.json (K30, rescored via d2911d10bb).
Honest holdout = data/v11_test_split_2026.json (1828 items).

Outputs:
  data/lgbm_underage_v6c.txt
  data/lgbm_v6c_features.json
  data/lgbm_v6c_meta.json
"""
import json, ast, struct, sqlite3, datetime, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
LGB_SEEDS = [1, 42, 314]


def open_db():
    db = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', db, 28, len(db)//4096)
    tmp = Path('/tmp/_v6c.db'); tmp.write_bytes(bytes(db))
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
            items.append({
                'id': row[0], 'source': 'grafana', 'label': row[1],
                'underage_labels': u, 'adult_labels': a,
            })
        except Exception:
            continue
    conn.close()
    return items


def load_k30():
    """K30: labels from db, features from k30_rescored.json (180-tag via d2911d10bb)."""
    items = []
    labels = {}
    conn = open_db()
    for row in conn.execute("""SELECT id, label FROM k30_pool
                                WHERE label_confirmed=1 AND label IN ('child','teen','adult')
                                  AND (deleted IS NULL OR deleted=0)"""):
        labels[row[0]] = row[1]
    conn.close()
    for r in json.loads((DATA / 'k30_rescored.json').read_text()):
        if not r.get('done'): continue
        if r.get('no_face'): continue
        lbl = labels.get(r['id'])
        if not lbl: continue
        u = r.get('underage_labels') or {}; a = r.get('adult_labels') or {}
        if not u and not a: continue
        items.append({
            'id': r['id'], 'source': 'k30', 'label': lbl,
            'underage_labels': u, 'adult_labels': a,
        })
    return items


def build_features(items, feat_names=None):
    """V6 raw features: underage_labels + adult__-prefixed adult_labels."""
    if feat_names is None:
        feat_set = set()
        for it in items:
            feat_set.update(it['underage_labels'].keys())
            for k in it['adult_labels'].keys():
                feat_set.add(f'adult__{k}')
        feat_names = sorted(feat_set)
    idx = {f: i for i, f in enumerate(feat_names)}
    X = np.zeros((len(items), len(feat_names)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    labels_out = []
    for i, it in enumerate(items):
        for k, v in it['underage_labels'].items():
            if k in idx: X[i, idx[k]] = float(v)
        for k, v in it['adult_labels'].items():
            fk = f'adult__{k}'
            if fk in idx: X[i, idx[fk]] = float(v)
        y[i] = 1 if it['label'] in ('child','teen') else 0
        labels_out.append(it['label'])
    return X, y, labels_out, feat_names


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
    print('=== V6c — extended scope (LS + Grafana + K30) ===\n', flush=True)
    split = json.loads((DATA / 'v11_test_split_2026.json').read_text())
    test_ids = set(split['test_ids'])
    print(f'  test split: {len(test_ids)} items\n', flush=True)

    ls = load_ls();      print(f'  LS:      {len(ls)}', flush=True)
    gr = load_grafana(); print(f'  Grafana: {len(gr)}', flush=True)
    k30 = load_k30();    print(f'  K30:     {len(k30)}', flush=True)
    all_items = ls + gr + k30
    train_items = [it for it in all_items if it['id'] not in test_ids]
    test_items  = [it for it in all_items if it['id'] in test_ids]
    print(f'\n  train: {len(train_items)}  test: {len(test_items)}', flush=True)

    from collections import Counter
    print(f'  train by source: {dict(Counter(it["source"] for it in train_items))}', flush=True)
    print(f'  test  by source: {dict(Counter(it["source"] for it in test_items))}', flush=True)

    X_tr, y_tr, _, feats = build_features(train_items)
    X_te, y_te, labels_te, _ = build_features(test_items, feats)
    sources_te = [it['source'] for it in test_items]
    print(f'  feature matrix: {X_tr.shape}\n', flush=True)

    base_params = dict(objective='binary', metric='auc', num_leaves=31,
                       learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.9,
                       min_child_samples=10, verbose=-1)
    results = []; last = None
    for sd in LGB_SEEDS:
        params = {**base_params, 'seed': sd, 'data_random_seed': sd,
                  'feature_fraction_seed': sd, 'bagging_seed': sd}
        m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=200)
        sc = m.predict(X_te)
        auc = roc_auc_score(y_te, sc) if len(set(y_te)) > 1 else None
        mm = metrics_at_thr(sc, labels_te, thr=0.30)
        results.append({'seed': sd, 'auc': auc, **mm})
        last = m
        print(f'  seed={sd}: AUC={auc:.4f} child={mm["child_recall"]:.3f} teen={mm["teen_recall"]:.3f} fpr={mm["adult_fpr"]:.3f}', flush=True)

    def aggregate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    summary = {k: aggregate(results, k) for k in ('auc','child_recall','teen_recall','adult_fpr')}
    summary['per_seed'] = results

    by_source = {}
    for src in set(sources_te):
        idx_sub = [i for i, s in enumerate(sources_te) if s == src]
        if not idx_sub: continue
        sub_X = X_te[idx_sub]
        sub_lbl = [labels_te[i] for i in idx_sub]
        sub_y = np.array([1 if l in ('child','teen') else 0 for l in sub_lbl])
        sub_sc = last.predict(sub_X)
        try: sub_auc = float(roc_auc_score(sub_y, sub_sc))
        except Exception: sub_auc = None
        sub_m = metrics_at_thr(sub_sc, sub_lbl, thr=0.30)
        by_source[src] = {'n': len(idx_sub), 'auc': sub_auc, **sub_m}

    last.save_model(str(DATA / 'lgbm_underage_v6c.txt'))
    (DATA / 'lgbm_v6c_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / 'lgbm_v6c_meta.json').write_text(json.dumps({
        'version': 'v6c', 'parent': 'v6 (extended scope LS+Grafana+K30)',
        'split': 'data/v11_test_split_2026.json',
        'n_train': len(train_items), 'n_test': len(test_items),
        'multi_seed': summary, 'by_source': by_source,
        'n_features': len(feats),
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print(f'\n=== V6c HONEST METRICS ===', flush=True)
    print(f'  AUC: {summary["auc"]["mean"]:.4f} ± {summary["auc"]["std"]:.4f}', flush=True)
    print(f'  child: {summary["child_recall"]["mean"]:.3f}  teen: {summary["teen_recall"]["mean"]:.3f}  fpr: {summary["adult_fpr"]["mean"]:.3f}', flush=True)
    print(f'\n  per-source:')
    for src, m in by_source.items():
        print(f'    {src:8} n={m["n"]:>4}  AUC={m["auc"]:.4f}  child={m["child_recall"]:.3f}  teen={m["teen_recall"]:.3f}  fpr={m["adult_fpr"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
