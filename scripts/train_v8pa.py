#!/usr/bin/env python3
"""
V8pa = V7pa + BCI feature split + hard-neg mining x20.

V7pa source: old siglip data (no_underage absent).
BCI taxonomy (Tom's): BODY/CONTEXT/INTERACTION applies to underage_ keys.
Hard-negatives: 71 V7pa adult FPs from data/v7pa_fps.json.
Path A: :x20 multiplier OFF.
"""
import json, re, struct, os, tempfile, datetime, ast, sqlite3, sys
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
    p = 1.0
    for k, v in scores_dict.items():
        if key_filter(k): p *= 1.0 - float(v)
    return 1.0 - p


def transform_item(item):
    """Path A: strip :x, unmultiply. No no_underage merge (V7pa source has no no_underage)."""
    u = {}
    for k, v in (item.get('underage_labels') or {}).items():
        raw = unmultiply(k, v); k2 = strip_mult(k)
        u[k2] = max(u.get(k2, 0.0), raw)
    a = dict((k, float(v)) for k, v in (item.get('adult_labels') or {}).items())
    return u, a


def open_db():
    raw = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', raw, 28, len(raw)//4096)
    fd, tmp = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd); Path(tmp).write_bytes(bytes(raw))
    conn = sqlite3.connect(tmp); conn.row_factory = sqlite3.Row
    return conn, tmp


def load_data():
    items = []
    # LS
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
        items.append({'id': 'ls_'+str(v['task_id']), 'label': cat, 'weight': 1.0,
                      'underage_labels': labels.get('underage') or {},
                      'adult_labels': labels.get('adult') or {}})
    # Grafana (all)
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
            items.append({'id': r['id'], 'label': r['label'], 'weight': w,
                          'underage_labels': labels.get('underage') or {},
                          'adult_labels': labels.get('adult') or {}})
        except: pass
    conn.close(); os.unlink(tmp)
    # Existing negatives
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult', 'weight': 1.0,
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {})})
    return items


def add_hard_negatives(items, weight_x=20):
    fps = json.loads((DATA / 'v7pa_fps.json').read_text()).get('v7pa_fps_adults', [])
    print(f'  hard-neg pool: {len(fps)} V7pa adult FPs', flush=True)
    for fp in fps:
        items.append({'id': fp['id'], 'label': 'adult', 'weight': float(weight_x),
                      'underage_labels': fp.get('underage_labels') or {},
                      'adult_labels': fp.get('adult_labels') or {}})
    return items


def build_features(items):
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
        inter= noisy_or(u, lambda k: k in INTERACTION_LABELS)
        bc = body + ctx
        X[i, idx['_child_body']] = body
        X[i, idx['_child_context']] = ctx
        X[i, idx['_child_interaction']] = inter
        X[i, idx['_body_vs_context']] = (body / bc) if bc > 0 else 0.0
        y[i] = 1 if it['label'] in ('child','teen') else 0
        w[i] = it['weight']
    return X, y, w, all_feats


def main():
    print('Loading V7pa-style data...', flush=True)
    items = load_data()
    print(f'  base items: {len(items)}', flush=True)
    items = add_hard_negatives(items, weight_x=20)
    print(f'  total items: {len(items)}', flush=True)

    print('Building features with BCI...', flush=True)
    X, y, w, feats = build_features(items)
    print(f'  X={X.shape}  pos={int(y.sum())}  neg={len(y)-int(y.sum())}  features={len(feats)}', flush=True)

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
    print(f'\nCV AUC: {mean_auc:.4f} ± {std_auc:.4f}', flush=True)

    final = lgb.train(params, lgb.Dataset(X, label=y, weight=w), num_boost_round=150)
    final.save_model(str(DATA / 'lgbm_underage_v8pa.txt'))
    (DATA / 'lgbm_v8pa_features.json').write_text(json.dumps(feats, indent=2))
    (DATA / 'lgbm_v8pa_meta.json').write_text(json.dumps({
        'version': 'v8pa', 'parent': 'v7pa',
        'note': 'V7pa + BCI feature split + hard-neg mining x20 on 71 V7pa FPs',
        'n_samples': len(items), 'n_features': len(feats),
        'cv_auc': mean_auc, 'cv_std': std_auc,
        'bci_aggregates': ['_child_body','_child_context','_child_interaction','_body_vs_context'],
        'hard_neg_count': 71, 'hard_neg_weight': 20,
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))
    print(f'  saved: lgbm_underage_v8pa.txt', flush=True)

    imp = final.feature_importance('gain')
    bci_gain = sum(g for f, g in zip(feats, imp) if f.startswith('_'))
    total_gain = sum(imp)
    print(f'BCI total gain: {bci_gain:.0f} ({100*bci_gain/total_gain:.1f}%)', flush=True)
    print('Top-15 features:', flush=True)
    for i, (f, g) in enumerate(sorted(zip(feats, imp), key=lambda x: -x[1])[:15]):
        mark = '*** BCI' if f.startswith('_') else ('  adlt' if f.startswith('adult__') else '   und')
        print(f'  {i+1:>2}. {mark}  {f[:45]:<46} gain={g:>7.0f}', flush=True)


if __name__ == '__main__':
    main()
