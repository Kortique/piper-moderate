#!/usr/bin/env python3
"""
train_v6b_holdout.py
--------------------
Re-train V6 (legacy 312-feature LGBM) excluding the 618 universal holdout items
(data/v11_test_split.json). Multi-seed for stability. Honest test metrics,
per-source breakdown.

Outputs:
  data/lgbm_underage_v6b.txt
  data/lgbm_v6b_features.json
  data/lgbm_v6b_meta.json
"""
import json, re, ast, sys, struct, sqlite3, datetime
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

LGB_SEEDS = [1, 42, 314]


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
        u = labels.get('underage') or {}
        a = labels.get('adult') or {}
        if not u and not a: continue
        items.append({
            'id': f"ls_{v['task_id']}", 'source': 'ls',
            'label': 'child' if age <= 14 else 'teen' if age <= 17 else 'adult',
            'underage_labels': u, 'adult_labels': a,
        })
    return items


def load_grafana():
    items = []
    db = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', db, 28, len(db)//4096)
    tmp = Path('/tmp/_v6b.db'); tmp.write_bytes(bytes(db))
    conn = sqlite3.connect(str(tmp)); conn.row_factory = sqlite3.Row
    for row in conn.execute("""SELECT id, label, piper_result
                               FROM grafana_pool
                               WHERE label_confirmed=1
                                 AND label IS NOT NULL
                                 AND (deleted IS NULL OR deleted=0)"""):
        try:
            pr = json.loads(row['piper_result']) if row['piper_result'] else {}
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            labels = det.get('labels', {})
            u = labels.get('underage') or {}
            a = labels.get('adult') or {}
            if not u and not a: continue
            items.append({
                'id': row['id'], 'source': 'grafana', 'label': row['label'],
                'underage_labels': u, 'adult_labels': a,
            })
        except Exception:
            continue
    conn.close()
    return items


def build_features_v6(items, feature_names=None):
    """V6 feature pipeline: just raw underage_labels + adult__ prefix for adult_labels."""
    if feature_names is None:
        all_feats = set()
        for it in items:
            all_feats.update(it['underage_labels'].keys())
            for k in it['adult_labels'].keys():
                all_feats.add(f'adult__{k}')
        feature_names = sorted(all_feats)
    idx = {f: i for i, f in enumerate(feature_names)}
    X = np.zeros((len(items), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    labels = []
    for i, it in enumerate(items):
        for k, v in it['underage_labels'].items():
            if k in idx: X[i, idx[k]] = float(v)
        for k, v in it['adult_labels'].items():
            fk = f'adult__{k}'
            if fk in idx: X[i, idx[fk]] = float(v)
        y[i] = 1 if it['label'] in ('child','teen') else 0
        labels.append(it['label'])
    return X, y, labels, feature_names


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
    print('=== V6b — train on V6 sources EXCLUDING universal v11_test_split ===\n', flush=True)
    test_ids = set(json.loads((DATA / 'v11_test_split.json').read_text())['test_ids'])
    print(f'  universal test set: {len(test_ids)} items\n', flush=True)

    ls = load_ls(); print(f'  LS items: {len(ls)}', flush=True)
    gr = load_grafana(); print(f'  Grafana items (confirmed): {len(gr)}', flush=True)
    all_items = ls + gr
    train_items = [it for it in all_items if it['id'] not in test_ids]
    test_items  = [it for it in all_items if it['id'] in test_ids]
    print(f'\n  train (excluding test): {len(train_items)}', flush=True)
    print(f'  test  (in universal holdout): {len(test_items)}', flush=True)
    test_src = {'ls': sum(1 for it in test_items if it['source']=='ls'),
                'grafana': sum(1 for it in test_items if it['source']=='grafana')}
    print(f'  test by source: {test_src}', flush=True)
    test_lab = {l: sum(1 for it in test_items if it['label']==l) for l in ['child','teen','adult']}
    print(f'  test by label:  {test_lab}\n', flush=True)

    # Build features from TRAIN only (V6 trained on its own taxonomy)
    X_tr, y_tr, _, feature_names = build_features_v6(train_items)
    print(f'  feature matrix: {X_tr.shape}', flush=True)
    X_te, y_te, labels_te, _ = build_features_v6(test_items, feature_names)
    sources_te = [it['source'] for it in test_items]

    # ── Multi-seed training ─────────────────────────────────────────────────
    base_params = dict(objective='binary', metric='auc', num_leaves=31,
                       learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.9,
                       min_child_samples=10, verbose=-1)
    results = []
    last_model = None
    for sd in LGB_SEEDS:
        params = {**base_params, 'seed': sd, 'data_random_seed': sd, 'feature_fraction_seed': sd, 'bagging_seed': sd}
        m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=150)
        sc = m.predict(X_te)
        auc = roc_auc_score(y_te, sc) if len(set(y_te)) > 1 else None
        mm = metrics_at_thr(sc, labels_te, thr=0.30)
        results.append({'seed': sd, 'auc': auc, **mm})
        last_model = m
        print(f'  seed={sd}: AUC={auc:.4f} child={mm["child_recall"]:.3f} teen={mm["teen_recall"]:.3f} fpr={mm["adult_fpr"]:.3f}', flush=True)

    def aggregate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    summary = {
        'auc':          aggregate(results, 'auc'),
        'child_recall': aggregate(results, 'child_recall'),
        'teen_recall':  aggregate(results, 'teen_recall'),
        'adult_fpr':    aggregate(results, 'adult_fpr'),
        'per_seed':     results,
    }

    # Per-source breakdown using last model
    by_source = {}
    for src in set(sources_te):
        idx = [i for i, s in enumerate(sources_te) if s == src]
        if not idx: continue
        sub_X = X_te[idx]
        sub_labels = [labels_te[i] for i in idx]
        sub_sc = last_model.predict(sub_X)
        sub_y = np.array([1 if l in ('child','teen') else 0 for l in sub_labels])
        try:
            sub_auc = float(roc_auc_score(sub_y, sub_sc))
        except Exception:
            sub_auc = None
        sub_m = metrics_at_thr(sub_sc, sub_labels, thr=0.30)
        by_source[src] = {'n': len(idx), 'auc': sub_auc, **sub_m}

    last_model.save_model(str(DATA / 'lgbm_underage_v6b.txt'))
    (DATA / 'lgbm_v6b_features.json').write_text(json.dumps(feature_names, indent=2))
    (DATA / 'lgbm_v6b_meta.json').write_text(json.dumps({
        'version': 'v6b',
        'parent':  'v6 (V6 LGBM re-trained excluding v11_test_split)',
        'split':   'data/v11_test_split.json (SEED=1337, stratified 80/20 by V11 training data)',
        'n_train': len(train_items),
        'n_test':  len(test_items),
        'test_source_breakdown': test_src,
        'test_label_breakdown':  test_lab,
        'multi_seed':  summary,
        'by_source':   by_source,
        'trained_at':  datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print('\n=== V6b HONEST METRICS on 618-item universal holdout ===', flush=True)
    print(f'  AUC:        {summary["auc"]["mean"]:.4f} ± {summary["auc"]["std"]:.4f}', flush=True)
    print(f'  child rec:  {summary["child_recall"]["mean"]:.3f} ± {summary["child_recall"]["std"]:.3f}', flush=True)
    print(f'  teen rec:   {summary["teen_recall"]["mean"]:.3f} ± {summary["teen_recall"]["std"]:.3f}', flush=True)
    print(f'  adult FPR:  {summary["adult_fpr"]["mean"]:.3f} ± {summary["adult_fpr"]["std"]:.3f}', flush=True)
    print(f'\n  per-source breakdown:', flush=True)
    for src, m in by_source.items():
        print(f'    {src:8} n={m["n"]:>3}  AUC={m["auc"]:.4f}  child={m["child_recall"]:.3f}  teen={m["teen_recall"]:.3f}  fpr={m["adult_fpr"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
