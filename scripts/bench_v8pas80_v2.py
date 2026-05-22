#!/usr/bin/env python3
"""
Benchmark V8pas80 vs V8pas80-v2 on LS holdout (qwen3_age_results.json).

Critical: no-regression gates (set by user):
  • child recall  ≥ 95 %  (must catch real underage)
  • teen recall   ≥ 80 %  (must catch real teens)
  • adult FPR     ≤ 20 %  (don't over-block adults)

Outputs report at fixed and best thresholds.
"""
import json, re, ast, sys, struct, os, tempfile, sqlite3
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

def unmult(k, v):
    m = X20_RE.search(k)
    if not m: return float(v)
    mult = float(m.group(1))
    return 0.999/mult if v >= 0.999 else float(v)/mult
def strip_mult(k): return X20_RE.sub('', k)
def sanitize(n): return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')
def noisy_or(d, f):
    p = 1.0
    for k, v in d.items():
        if f(k): p *= 1.0 - float(v)
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


def load_ls():
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
        items.append({'id': 'ls_'+str(v['task_id']), 'label': cat,
                      'underage_labels': labels.get('underage') or {},
                      'adult_labels': labels.get('adult') or {}})
    return items


def metrics_at_thr(scores, labels, thr):
    """Per-class recall and FPR at given threshold."""
    n_child = n_teen = n_adult = 0
    tp_child = tp_teen = fp_adult = 0
    for s, l in zip(scores, labels):
        flagged = s >= thr
        if l == 'child':
            n_child += 1
            if flagged: tp_child += 1
        elif l == 'teen':
            n_teen += 1
            if flagged: tp_teen += 1
        elif l == 'adult':
            n_adult += 1
            if flagged: fp_adult += 1
    return {
        'child_recall': tp_child/n_child if n_child else 0,
        'teen_recall':  tp_teen/n_teen if n_teen else 0,
        'adult_fpr':    fp_adult/n_adult if n_adult else 0,
        'counts': {'child': n_child, 'teen': n_teen, 'adult': n_adult,
                   'tp_child': tp_child, 'tp_teen': tp_teen, 'fp_adult': fp_adult},
    }


def main():
    print('=== V8pas80 vs V8pas80-v2 — no-regression bench ===\n', flush=True)
    items = load_ls()
    n_by_label = {l: sum(1 for it in items if it['label']==l) for l in ['child','teen','adult']}
    print(f'LS holdout: {len(items)} items', flush=True)
    print(f'  by label: {n_by_label}', flush=True)

    models = {
        'V8pas80':     ('lgbm_underage_v8pas80.txt',    'lgbm_v8pas80_features.json'),
        'V8pas80-v2':  ('lgbm_underage_v8pas80_v2.txt', 'lgbm_v8pas80_v2_features.json'),
    }
    results = {}
    for name, (mfile, ffile) in models.items():
        if not (DATA / mfile).exists():
            print(f'\n[{name}] model not found, skip', flush=True); continue
        booster = lgb.Booster(model_file=str(DATA / mfile))
        feats = json.loads((DATA / ffile).read_text())
        X, y, labels = build_X(items, feats)
        scores = booster.predict(X)
        auc = roc_auc_score(y, scores)
        print(f'\n[{name}]  features={len(feats)}  AUC={auc:.4f}', flush=True)
        per_thr = {}
        for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            m = metrics_at_thr(scores, labels, thr)
            per_thr[thr] = m
            print(f'  thr={thr:.2f}  child={m["child_recall"]:.3f}  teen={m["teen_recall"]:.3f}  fpr_adult={m["adult_fpr"]:.3f}', flush=True)
        results[name] = {'auc': auc, 'per_thr': per_thr, 'scores': scores.tolist()}

    if 'V8pas80' in results and 'V8pas80-v2' in results:
        print('\n=== DELTA (v2 minus v1) ===', flush=True)
        print(f'  AUC: {results["V8pas80-v2"]["auc"] - results["V8pas80"]["auc"]:+.4f}', flush=True)
        for thr in [0.20, 0.25, 0.30, 0.35, 0.40]:
            v1 = results['V8pas80']['per_thr'][thr]
            v2 = results['V8pas80-v2']['per_thr'][thr]
            print(f'  thr={thr:.2f}  '
                  f'child={v2["child_recall"]-v1["child_recall"]:+.3f}  '
                  f'teen={v2["teen_recall"]-v1["teen_recall"]:+.3f}  '
                  f'fpr={v2["adult_fpr"]-v1["adult_fpr"]:+.3f}', flush=True)

        # Gates @ thr=0.30 (production default)
        thr_prod = 0.30
        v2 = results['V8pas80-v2']['per_thr'][thr_prod]
        v1 = results['V8pas80']['per_thr'][thr_prod]
        print(f'\n=== REGRESSION GATES @ thr={thr_prod:.2f} ===', flush=True)
        gates = [
            ('child_recall ≥ 0.95',  v2['child_recall'] >= 0.95, f'{v2["child_recall"]:.3f} (v1: {v1["child_recall"]:.3f})'),
            ('teen_recall  ≥ 0.80',  v2['teen_recall'] >= 0.80, f'{v2["teen_recall"]:.3f} (v1: {v1["teen_recall"]:.3f})'),
            ('adult_fpr    ≤ 0.20',  v2['adult_fpr'] <= 0.20, f'{v2["adult_fpr"]:.3f} (v1: {v1["adult_fpr"]:.3f})'),
        ]
        for label, ok, val in gates:
            print(f'  {"✓" if ok else "✗"} {label}: {val}', flush=True)
        all_pass = all(ok for _, ok, _ in gates)
        print(f'\n  {"✓ PASS — safe to deploy" if all_pass else "✗ REGRESSION — DO NOT deploy"}', flush=True)

        # Save full results
        (DATA / 'v8pas80_v2_bench.json').write_text(json.dumps({
            'V8pas80': {'auc': results['V8pas80']['auc'],
                        'per_thr': {str(k): v for k, v in results['V8pas80']['per_thr'].items()}},
            'V8pas80_v2': {'auc': results['V8pas80-v2']['auc'],
                           'per_thr': {str(k): v for k, v in results['V8pas80-v2']['per_thr'].items()}},
            'gates_pass': all_pass,
        }, indent=2))


if __name__ == '__main__':
    main()
