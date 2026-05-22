#!/usr/bin/env python3
"""
V8pas80-v2 = V8pa retrained with extended hard-neg pool.

Hard-neg pool (weight ×20):
  • 71 original V7pa adult FPs (data/v7pa_fps.json)
  • 67 new Grok-confirmed V8pas80 adult FPs (data/v8pas80_fps_grok_confirmed.json)
  Total: 138 hard-negatives ×20 = effective weight 2760

Steps:
  1. Train V8pa-v2 (full feature set) — gain ranking source
  2. Prune to top-80 features + 4 BCI aggregates → V8pas80-v2
  3. Reports CV AUC + per-class recall/FPR before/after on LS holdout

Backwards compatible with build_X_subset from train_slim.py.
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
BCI_FEATS = ['_child_body', '_child_context', '_child_interaction', '_body_vs_context']


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


def transform(item):
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


def load_base_items():
    """Same as train_v8pa.py / train_slim.py load_v8pa_data — without hard-negs."""
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
    return items


def add_hard_negs_v2(items, weight_x=20):
    """71 V7pa-FPs + 67 Grok-confirmed V8pas80-FPs as hard-negatives ×20.

    For each FP, drop the corresponding base item (same id) — we don't want
    the same media as both label='adult' weight=1 and label='adult' weight=20.
    """
    fps_v7pa = json.loads((DATA / 'v7pa_fps.json').read_text()).get('v7pa_fps_adults', [])
    fps_grok = json.loads((DATA / 'v8pas80_fps_grok_confirmed.json').read_text())
    # Get original feature vectors for grok FPs — from v8pas80_top100_fps.json
    grok_map = {f['id']: f for f in json.loads((DATA / 'v8pas80_top100_fps.json').read_text())}

    hard_negs = []
    seen_ids = set()
    for fp in fps_v7pa:
        if fp['id'] in seen_ids: continue
        hard_negs.append({'id': fp['id'], 'label': 'adult', 'weight': float(weight_x),
                          'underage_labels': fp.get('underage_labels') or {},
                          'adult_labels': fp.get('adult_labels') or {},
                          '_source': 'v7pa_fp'})
        seen_ids.add(fp['id'])
    for fp in fps_grok:
        if fp['id'] in seen_ids: continue
        src = grok_map.get(fp['id'])
        if not src: continue  # no feature vector available
        hard_negs.append({'id': fp['id'], 'label': 'adult', 'weight': float(weight_x),
                          'underage_labels': src.get('underage_labels') or {},
                          'adult_labels': src.get('adult_labels') or {},
                          '_source': 'grok_fp',
                          '_grok_scene': fp.get('scene_type', '')})
        seen_ids.add(fp['id'])

    # Drop base-items whose id matches a hard-neg id (avoid duplicate signal)
    filtered = [it for it in items if it.get('id') not in seen_ids]
    dropped = len(items) - len(filtered)
    print(f'  hard-neg pool: {len(hard_negs)} ({len(fps_v7pa)} V7pa + {len(fps_grok)} Grok-confirmed)', flush=True)
    print(f'  dropped {dropped} base-items overlapping with hard-neg pool', flush=True)
    return filtered + hard_negs


def build_features_full(items):
    """Full feature set (everything seen in data) + BCI."""
    feat_set = set()
    for it in items:
        u, a = transform(it)
        for k in u: feat_set.add(sanitize(k))
        for k in a: feat_set.add('adult__' + sanitize(k))
    feats = sorted(feat_set) + BCI_FEATS
    return build_X_subset(items, feats), feats


def build_X_subset(items, feats):
    idx = {f: i for i, f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)
    w = np.ones(len(items), dtype=np.float32)
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
        if '_child_body' in idx:        X[i, idx['_child_body']] = body
        if '_child_context' in idx:     X[i, idx['_child_context']] = ctx
        if '_child_interaction' in idx: X[i, idx['_child_interaction']] = inter
        if '_body_vs_context' in idx:   X[i, idx['_body_vs_context']] = (body/bc) if bc>0 else 0.0
        y[i] = 1 if it['label'] in ('child','teen') else 0
        w[i] = it['weight']
    return X, y, w


def train_cv(X, y, w, n_rounds=150):
    params = dict(objective='binary', metric='auc', num_leaves=15,
                  learning_rate=0.04, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=3, lambda_l1=0.1, lambda_l2=0.1,
                  is_unbalance=True, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], weight=w[tr]), num_boost_round=n_rounds)
        aucs.append(roc_auc_score(y[va], m.predict(X[va])))
        print(f'    fold {fold+1}: AUC={aucs[-1]:.4f}', flush=True)
    final = lgb.train(params, lgb.Dataset(X, label=y, weight=w), num_boost_round=n_rounds)
    return final, float(np.mean(aucs)), float(np.std(aucs))


def select_top_features(model, feats, top_n):
    gain = model.feature_importance('gain')
    ranked = sorted(zip(feats, gain), key=lambda x: -x[1])
    keep = [f for f, _ in ranked[:top_n]]
    for bf in BCI_FEATS:
        if bf not in keep: keep.append(bf)
    return keep


def main():
    print('=== V8pas80-v2 training ===', flush=True)
    print('\n[1/3] Loading base items...', flush=True)
    base = load_base_items()
    print(f'  base items: {len(base)}', flush=True)

    print('\n[2/3] Add hard-negatives ×20 (extended pool)...', flush=True)
    items = add_hard_negs_v2(base, weight_x=20)
    n_pos = sum(1 for it in items if it['label'] in ('child','teen'))
    n_neg = len(items) - n_pos
    print(f'  total items: {len(items)} (pos={n_pos}, neg={n_neg})', flush=True)

    print('\n[3/3] Train V8pa-v2 (full features) — gain source...', flush=True)
    (X, y, w), feats_full = build_features_full(items)
    print(f'  X={X.shape}  features={len(feats_full)}', flush=True)
    m_full, auc_full, std_full = train_cv(X, y, w)
    print(f'  V8pa-v2 CV AUC: {auc_full:.4f} ± {std_full:.4f}', flush=True)
    m_full.save_model(str(DATA / 'lgbm_underage_v8pa_v2.txt'))
    (DATA / 'lgbm_v8pa_v2_features.json').write_text(json.dumps(feats_full, indent=2))

    print('\nTop-80 features by gain (V8pas80-v2):', flush=True)
    feats_top80 = select_top_features(m_full, feats_full, 80)
    print(f'  features: {len(feats_top80)} (top-80 + 4 BCI)', flush=True)
    X80, y80, w80 = build_X_subset(items, feats_top80)
    m_slim, auc_slim, std_slim = train_cv(X80, y80, w80)
    print(f'  V8pas80-v2 CV AUC: {auc_slim:.4f} ± {std_slim:.4f}', flush=True)
    m_slim.save_model(str(DATA / 'lgbm_underage_v8pas80_v2.txt'))
    (DATA / 'lgbm_v8pas80_v2_features.json').write_text(json.dumps(feats_top80, indent=2))
    (DATA / 'lgbm_v8pas80_v2_meta.json').write_text(json.dumps({
        'version': 'v8pas80_v2',
        'parent': 'v8pa_v2',
        'note': 'V8pa retrained with extended hard-neg pool (71 V7pa + 67 Grok-confirmed FPs), pruned to top-80 by gain',
        'n_samples': len(items), 'n_features': len(feats_top80),
        'cv_auc_full': auc_full, 'cv_auc_slim': auc_slim,
        'cv_std_full': std_full, 'cv_std_slim': std_slim,
        'hard_neg_count': 138, 'hard_neg_weight': 20,
        'hard_neg_v7pa': 71, 'hard_neg_grok': 67,
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    # Top-15 features
    imp = m_slim.feature_importance('gain')
    print('\nTop-15 V8pas80-v2 features:', flush=True)
    for i, (f, g) in enumerate(sorted(zip(feats_top80, imp), key=lambda x: -x[1])[:15]):
        mark = '*** BCI' if f.startswith('_') else ('  adlt' if f.startswith('adult__') else '   und')
        print(f'  {i+1:>2}. {mark}  {f[:45]:<46} gain={g:>7.0f}', flush=True)

    print(f'\n✓ V8pas80-v2 saved: data/lgbm_underage_v8pas80_v2.txt', flush=True)


if __name__ == '__main__':
    main()
