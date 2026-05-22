#!/usr/bin/env python3
"""
Train slim variants of V11 and V8pa using top-N features by gain.
Two cuts per model: top-50 (aggressive) and top-80 (balanced).

Sequence:
  1. Identify top-N features from existing model by gain.
  2. Make sure all 4 BCI aggregates are kept regardless (always relevant).
  3. Retrain on same data with the reduced feature set.
"""
import json, re, struct, os, tempfile, datetime, ast, sqlite3, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'scripts')
from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

BASE = Path('.')
DATA = BASE / 'data'

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')
BCI_FEATS = ['_child_body', '_child_context', '_child_interaction', '_body_vs_context']

def unmultiply(key, val):
    m = X20_RE.search(key)
    if not m: return float(val)
    mult = float(m.group(1))
    if val >= 0.999: return 0.999 / mult
    return float(val) / mult
def strip_mult(k): return X20_RE.sub('', k)
def sanitize(n): return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')
def noisy_or(d, f):
    p = 1.0
    for k, v in d.items():
        if f(k): p *= 1.0 - float(v)
    return 1.0 - p


def transform(item, merge_no_und):
    u = {}
    for k, v in (item.get('underage_labels') or {}).items():
        raw = unmultiply(k, v); k2 = strip_mult(k)
        u[k2] = max(u.get(k2, 0.0), raw)
    a = dict((k, float(v)) for k, v in (item.get('adult_labels') or {}).items())
    if merge_no_und:
        for k, v in (item.get('no_underage_labels') or {}).items():
            a[k] = max(a.get(k, 0.0), float(v))
    return u, a


def open_db():
    raw = bytearray(Path('gallery.db').read_bytes())
    struct.pack_into('>I', raw, 28, len(raw)//4096)
    fd, tmp = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd); Path(tmp).write_bytes(bytes(raw))
    conn = sqlite3.connect(tmp); conn.row_factory = sqlite3.Row
    return conn, tmp


def load_v11_data():
    items = []
    for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': 'ls_'+str(r['task_id']), 'label': r['label'], 'weight': 1.0,
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {}})
    for r in json.loads((DATA / 'v9_317_scores.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        items.append({'id': r['id'], 'label': r['label'], 'weight': 3.0,
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {}})
    for r in json.loads((DATA / 'eval616_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        if r.get('kind') == 'ls': continue
        items.append({'id': r['id'], 'label': r['label'], 'weight': 1.0,
                      'underage_labels': r.get('underage_labels') or {},
                      'adult_labels': r.get('adult_labels') or {},
                      'no_underage_labels': r.get('no_underage_labels') or {}})
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult', 'weight': 1.0,
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {}),
                          'no_underage_labels': {}})
    # Hard-negs x20
    fps = json.loads((DATA / 'v10_diff_analysis.json').read_text()).get('v10_fps_adults', [])
    ls_map = {f"ls_{r['task_id']}": r for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()) if r.get('done')}
    for fp in fps:
        full = ls_map.get(fp['id'])
        if not full: continue
        items.append({'id': fp['id'], 'label': 'adult', 'weight': 20.0,
                      'underage_labels': full.get('underage_labels') or {},
                      'adult_labels': full.get('adult_labels') or {},
                      'no_underage_labels': full.get('no_underage_labels') or {}})
    return items


def load_v8pa_data():
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
        items.append({'id': 'ls_'+str(v['task_id']), 'label': cat, 'weight': 1.0,
                      'underage_labels': labels.get('underage') or {},
                      'adult_labels': labels.get('adult') or {}})
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
    for fn in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json', 'v6_fp_hard_negatives.json']:
        p = DATA / fn
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            items.append({'id': r.get('id'), 'label': 'adult', 'weight': 1.0,
                          'underage_labels': r.get('underage_labels', {}),
                          'adult_labels': r.get('adult_labels', {})})
    # hard-negs x20
    for fp in json.loads((DATA / 'v7pa_fps.json').read_text()).get('v7pa_fps_adults', []):
        items.append({'id': fp['id'], 'label': 'adult', 'weight': 20.0,
                      'underage_labels': fp.get('underage_labels') or {},
                      'adult_labels': fp.get('adult_labels') or {}})
    return items


def build_X_subset(items, feats, merge_no_und):
    """Build X with given subset of features (only those features are computed)."""
    idx = {f: i for i, f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    w = np.ones(len(items), dtype=np.float32)
    for i, it in enumerate(items):
        u, a = transform(it, merge_no_und)
        for k, v in u.items():
            f = sanitize(k)
            if f in idx: X[i, idx[f]] = v
        for k, v in a.items():
            f = 'adult__' + sanitize(k)
            if f in idx: X[i, idx[f]] = float(v)
        # BCI
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
    return X, y, w


def select_top_features(parent_model_name, top_n):
    """Select top-N features by gain from existing model. Always include 4 BCI."""
    feats = json.load(open(f'data/lgbm_{parent_model_name}_features.json'))
    booster = lgb.Booster(model_file=f'data/lgbm_underage_{parent_model_name}.txt')
    gain = booster.feature_importance('gain')
    ranked = sorted(zip(feats, gain), key=lambda x: -x[1])
    keep = [f for f, g in ranked[:top_n]]
    # Always include BCI even if missing
    for bf in BCI_FEATS:
        if bf not in keep: keep.append(bf)
    return keep


def train_save(name, items, feats, merge_no_und, parent, top_n, note):
    X, y, w = build_X_subset(items, feats, merge_no_und)
    print(f'  X={X.shape}  pos={int(y.sum())}  neg={len(y)-int(y.sum())}', flush=True)
    params = dict(objective='binary', metric='auc', num_leaves=15,
                  learning_rate=0.04, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=3, lambda_l1=0.1, lambda_l2=0.1,
                  is_unbalance=True, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], weight=w[tr]), num_boost_round=150)
        aucs.append(roc_auc_score(y[va], m.predict(X[va])))
    auc, std = float(np.mean(aucs)), float(np.std(aucs))
    print(f'  CV AUC: {auc:.4f} ± {std:.4f}', flush=True)
    final = lgb.train(params, lgb.Dataset(X, label=y, weight=w), num_boost_round=150)
    final.save_model(f'data/lgbm_underage_{name}.txt')
    json.dump(feats, open(f'data/lgbm_{name}_features.json', 'w'), indent=2)
    json.dump({'version': name, 'parent': parent, 'note': note, 'top_n_target': top_n,
               'n_features': len(feats), 'cv_auc': auc, 'cv_std': std,
               'trained_at': datetime.datetime.utcnow().isoformat()},
              open(f'data/lgbm_{name}_meta.json', 'w'), indent=2)
    print(f'  saved: lgbm_underage_{name}.txt', flush=True)


def main():
    print('=== Slim V11 variants ===', flush=True)
    items_v11 = load_v11_data()
    print(f'V11 items: {len(items_v11)}', flush=True)
    for top_n in [50, 80]:
        print(f'\nV11-slim{top_n}:', flush=True)
        feats = select_top_features('v11', top_n)
        print(f'  features: {len(feats)} (top-{top_n} by gain + 4 BCI)', flush=True)
        train_save(f'v11s{top_n}', items_v11, feats, merge_no_und=True,
                   parent='v11', top_n=top_n, note=f'V11 pruned to top-{top_n} features by gain')

    print('\n=== Slim V8pa variants ===', flush=True)
    items_v8pa = load_v8pa_data()
    print(f'V8pa items: {len(items_v8pa)}', flush=True)
    for top_n in [50, 80]:
        print(f'\nV8pa-slim{top_n}:', flush=True)
        feats = select_top_features('v8pa', top_n)
        print(f'  features: {len(feats)} (top-{top_n} by gain + 4 BCI)', flush=True)
        train_save(f'v8pas{top_n}', items_v8pa, feats, merge_no_und=False,
                   parent='v8pa', top_n=top_n, note=f'V8pa pruned to top-{top_n}')


if __name__ == '__main__':
    main()
