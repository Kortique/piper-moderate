#!/usr/bin/env python3
"""
compare_v7_v8_v9_v10.py
-----------------------
Сравнивает V7, V8, V9, V10 на:
  1. LS holdout rescored (ls_holdout_rescored.json) — консистентные 867-тегов
  2. LS holdout original (qwen3_age_results.json)   — для сравнения с priors
  3. 317-session hard examples (v9_317_scores.json)

Blocking rule: lgbm_score >= THR  OR  minor >= 0.72
Target: Child recall >= 95%

Usage:
    python scripts/compare_v7_v8_v9_v10.py
"""
import json, ast, struct, sqlite3, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def sanitize_feat(name: str) -> str:
    return name.replace(':', '_x').replace('"', '').replace("'", '').replace('{', '').replace('}', '')


def build_X(items, feat_names):
    feat_idx = {f: i for i, f in enumerate(feat_names)}
    X = np.zeros((len(items), len(feat_names)), dtype=np.float32)
    for i, item in enumerate(items):
        for k, v in item.get('underage_labels', {}).items():
            fk = sanitize_feat(k)
            if fk in feat_idx: X[i, feat_idx[fk]] = float(v)
        for k, v in item.get('adult_labels', {}).items():
            fk = 'adult__' + sanitize_feat(k)
            if fk in feat_idx: X[i, feat_idx[fk]] = float(v)
        for k, v in item.get('no_underage_labels', {}).items():
            fk = 'no_underage__' + sanitize_feat(k)
            if fk in feat_idx: X[i, feat_idx[fk]] = float(v)
    return X


def score_items(items, model_path, feat_path, key):
    model = lgb.Booster(model_file=str(model_path))
    feat_names = json.loads(Path(feat_path).read_text())
    X = build_X(items, feat_names)
    preds = model.predict(X)
    for item, p in zip(items, preds):
        item[key] = float(p)


def load_ls_rescored():
    path = BASE_DIR / 'data' / 'ls_holdout_rescored.json'
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    items = []
    for r in data:
        if not r.get('done'): continue
        lbl = r.get('label')
        if lbl not in ('child', 'teen', 'adult'): continue
        items.append({
            'id': f"ls_{r['task_id']}",
            'label': lbl,
            'minor': r.get('minor', 0),
            'adult': r.get('adult', 0),
            'underage_labels':    r.get('underage_labels', {}),
            'adult_labels':       r.get('adult_labels', {}),
            'no_underage_labels': r.get('no_underage_labels', {}),
            'lgbm_v7': 0, 'lgbm_v8': 0, 'lgbm_v9': 0, 'lgbm_v10': 0,
        })
    return items


def load_ls_original():
    json_path = BASE_DIR / 'qwen3_age_results.json'
    raw = json_path.read_bytes().rstrip(b'\x00').decode('utf-8')
    data = json.loads(raw)
    items = []
    for v in data.values():
        try:
            lbl = v.get('category')
            if lbl not in ('child', 'teen', 'adult'): continue
            siglip_raw = v.get('siglip2_details')
            siglip2 = ast.literal_eval(siglip_raw) if isinstance(siglip_raw, str) else (siglip_raw or {})
            det = siglip2.get('underage', {})
            labels = det.get('labels', {})
            items.append({
                'id': f"ls_{v['task_id']}",
                'label': lbl,
                'minor': det.get('minor', 0),
                'adult': det.get('adult', 0),
                'underage_labels':    labels.get('underage') or {},
                'adult_labels':       labels.get('adult') or {},
                'no_underage_labels': labels.get('no_underage') or {},
                'lgbm_v7': 0, 'lgbm_v8': 0, 'lgbm_v9': 0, 'lgbm_v10': 0,
            })
        except Exception:
            pass
    return items


def load_317_session():
    path = BASE_DIR / 'data' / 'v9_317_scores.json'
    if not path.exists(): return []
    data = json.loads(path.read_text())
    items = []
    for r in data:
        if not r.get('done'): continue
        lbl = r.get('label')
        if lbl not in ('child', 'teen', 'adult'): continue
        items.append({
            'id': r['id'], 'label': lbl,
            'minor': r.get('minor', 0),
            'adult': r.get('adult', 0),
            'underage_labels':    r.get('underage_labels', {}),
            'adult_labels':       r.get('adult_labels', {}),
            'no_underage_labels': r.get('no_underage_labels', {}),
            'lgbm_v7': 0, 'lgbm_v8': 0, 'lgbm_v9': 0, 'lgbm_v10': 0,
        })
    return items


def evaluate_at_threshold(items, score_key, thr):
    child_items = [x for x in items if x['label'] == 'child']
    teen_items  = [x for x in items if x['label'] == 'teen']
    adult_items = [x for x in items if x['label'] == 'adult']

    def blocked(item):
        return item.get(score_key, 0) >= thr  # LGBM-only, no minor rule

    cr  = 100 * sum(1 for x in child_items if blocked(x)) / max(len(child_items), 1)
    tr  = 100 * sum(1 for x in teen_items  if blocked(x)) / max(len(teen_items), 1)
    fpr = 100 * sum(1 for x in adult_items if blocked(x)) / max(len(adult_items), 1)
    return cr, tr, fpr


def print_comparison(title, items, versions):
    thresholds = [0.55, 0.50, 0.45, 0.42, 0.40, 0.38, 0.35, 0.30]
    child_n = sum(1 for x in items if x['label'] == 'child')
    teen_n  = sum(1 for x in items if x['label'] == 'teen')
    adult_n = sum(1 for x in items if x['label'] == 'adult')

    print(f'\n{"="*70}')
    print(f'{title}')
    print(f'  child={child_n}, teen={teen_n}, adult={adult_n}, total={len(items)}')
    print(f'  Blocking rule: lgbm >= THR  (LGBM-only, no minor rule)')
    print(f'{"="*70}')

    hdr = f"{'THR':>5}"
    for v, _ in versions:
        hdr += f"  {v:>24}"
    print(hdr)
    sep = '-' * (5 + 26 * len(versions))
    print(sep)
    sub = ' ' * 5
    for _, _ in versions:
        sub += f"  {'Child%':>8} {'Teen%':>7} {'FPR%':>6}"
    print(sub)
    print(sep)

    for thr in thresholds:
        row = f'{thr:>5.2f}'
        for _, sk in versions:
            cr, tr, fpr = evaluate_at_threshold(items, sk, thr)
            flag = '✓' if cr >= 95.0 else ' '
            row += f"  {cr:>7.1f}{flag} {tr:>7.1f} {fpr:>6.1f}"
        print(row)

    print(sep)
    print('  ✓ = child recall ≥ 95% (target)  |  LGBM-only blocking')


def main():
    # Models
    models = [
        ('V7 (314f)', 'lgbm_v7', 'lgbm_underage_v7.txt', 'lgbm_v7_features.json'),
        ('V8 (355f)', 'lgbm_v8', 'lgbm_underage_v8.txt', 'lgbm_v8_features.json'),
        ('V9 (422f)', 'lgbm_v9', 'lgbm_underage_v9.txt', 'lgbm_v9_features.json'),
        ('V10(rescored)', 'lgbm_v10', 'lgbm_underage_v10.txt', 'lgbm_v10_features.json'),
    ]

    available = [(label, key, mf, ff) for label, key, mf, ff in models
                 if (BASE_DIR / 'data' / mf).exists() and (BASE_DIR / 'data' / ff).exists()]
    if not available:
        print('No models found!')
        return

    versions = [(label, key) for label, key, _, _ in available]

    # ── LS holdout RESCORED ────────────────────────────────────────────────────
    print('Loading LS holdout (rescored)...')
    ls_rescored = load_ls_rescored()
    if ls_rescored:
        print(f'  {len(ls_rescored)} items')
        for label, key, mf, ff in available:
            print(f'  Scoring {label}...')
            score_items(ls_rescored, BASE_DIR/'data'/mf, BASE_DIR/'data'/ff, key)
        print_comparison('LS HOLDOUT — RESCORED (867 tags, no_underage реальные)', ls_rescored, versions)
    else:
        print('  ls_holdout_rescored.json not found or empty — пропускаем')

    # ── LS holdout ORIGINAL ────────────────────────────────────────────────────
    print('\nLoading LS holdout (original, for reference)...')
    ls_orig = load_ls_original()
    if ls_orig:
        print(f'  {len(ls_orig)} items')
        for label, key, mf, ff in available:
            score_items(ls_orig, BASE_DIR/'data'/mf, BASE_DIR/'data'/ff, key)
        print_comparison('LS HOLDOUT — ORIGINAL (siglip5, no_underage=0)', ls_orig, versions)

    # ── 317-session ────────────────────────────────────────────────────────────
    print('\nLoading 317-session...')
    s317 = load_317_session()
    if s317:
        print(f'  {len(s317)} items')
        for label, key, mf, ff in available:
            score_items(s317, BASE_DIR/'data'/mf, BASE_DIR/'data'/ff, key)
        print_comparison('317-SESSION HARD EXAMPLES (ce79f7e299, 867 tags)', s317, versions)

    print('\nDone.')


if __name__ == '__main__':
    main()
