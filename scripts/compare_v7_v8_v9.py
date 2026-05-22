#!/usr/bin/env python3
"""
compare_v7_v8_v9.py
-------------------
Сравнительный анализ V7, V8, V9 на LS holdout (qwen3_age_results.json, n=2216).
Показывает sweep порогов 0.30–0.55 для blocking_rule: lgbm >= thr OR minor >= 0.72

Usage:
    python scripts/compare_v7_v8_v9.py
"""
import json, ast, struct, sqlite3, sys, os
from pathlib import Path
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


# ── Data loading ──────────────────────────────────────────────────────────────

def sanitize_feat(name: str) -> str:
    return name.replace(':', '_x').replace('"', '').replace("'", '').replace('{', '').replace('}', '')


def load_ls_items():
    """Load LS items from qwen3_age_results.json."""
    raw = (BASE_DIR / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    data = json.loads(raw)
    items = []
    for v in data.values():
        try:
            lbl = v.get('category')
            if lbl not in ('child', 'teen', 'adult'):
                continue
            siglip_raw = v.get('siglip2_details')
            siglip2 = ast.literal_eval(siglip_raw) if isinstance(siglip_raw, str) else (siglip_raw or {})
            det = siglip2.get('underage', {})
            labels = det.get('labels', {})
            items.append({
                'id': f"ls_{v['task_id']}",
                'label': lbl,
                'minor': det.get('minor', 0),
                'adult': det.get('adult', 0),
                'underage_labels': labels.get('underage') or {},
                'adult_labels': labels.get('adult') or {},
                'no_underage_labels': labels.get('no_underage') or {},
                'lgbm_v7': 0,  # filled below
                'lgbm_v8': 0,
                'lgbm_v9': 0,
            })
        except Exception:
            pass
    return items


def _open_db():
    candidates = sorted((BASE_DIR / 'backups').glob('gallery_*.db'), reverse=True)
    for db_path in candidates:
        try:
            data = bytearray(db_path.read_bytes())
            struct.pack_into('>I', data, 28, len(data) // 4096)
            tmp = Path('/tmp/_cmp.db')
            tmp.write_bytes(bytes(data))
            conn = sqlite3.connect(str(tmp))
            conn.row_factory = sqlite3.Row
            conn.execute('SELECT id FROM grafana_pool LIMIT 1').fetchall()
            return conn
        except Exception:
            continue
    return None


# ── Feature building ──────────────────────────────────────────────────────────

def build_X(items, feat_names):
    """Build feature matrix for a list of items using given feature names."""
    feat_idx = {f: i for i, f in enumerate(feat_names)}
    n, m = len(items), len(feat_names)
    X = np.zeros((n, m), dtype=np.float32)
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


def score_items(items, model_path, feat_path, version_key):
    """Score items with LGBM model, store result in item[version_key]."""
    model = lgb.Booster(model_file=str(model_path))
    feat_names = json.loads(Path(feat_path).read_text())
    X = build_X(items, feat_names)
    preds = model.predict(X)
    for item, score in zip(items, preds):
        item[version_key] = float(score)
    return feat_names


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_at_threshold(items, score_key, thr, minor_thr=0.72):
    """Return (child_recall, teen_recall, adult_fpr) for blocking_rule: score >= thr OR minor >= minor_thr."""
    child_items = [x for x in items if x['label'] == 'child']
    teen_items  = [x for x in items if x['label'] == 'teen']
    adult_items = [x for x in items if x['label'] == 'adult']

    def blocked(item):
        return item.get(score_key, 0) >= thr or item.get('minor', 0) >= minor_thr

    child_recall = 100 * sum(1 for x in child_items if blocked(x)) / max(len(child_items), 1)
    teen_recall  = 100 * sum(1 for x in teen_items  if blocked(x)) / max(len(teen_items), 1)
    adult_fpr    = 100 * sum(1 for x in adult_items if blocked(x)) / max(len(adult_items), 1)

    return child_recall, teen_recall, adult_fpr


def print_comparison(items, versions):
    """Print threshold sweep table for multiple versions."""
    thresholds = [0.55, 0.50, 0.45, 0.42, 0.40, 0.38, 0.35, 0.30]

    child_n  = sum(1 for x in items if x['label'] == 'child')
    teen_n   = sum(1 for x in items if x['label'] == 'teen')
    adult_n  = sum(1 for x in items if x['label'] == 'adult')
    print(f'\nEval set: child={child_n}, teen={teen_n}, adult={adult_n}, total={len(items)}')
    print(f'Blocking rule: lgbm_score >= THR  OR  minor >= 0.72')
    print()

    # Header
    hdr = f"{'THR':>5}"
    for v, _ in versions:
        hdr += f"  {v:>22}"
    print(hdr)
    sep = '-' * (5 + 24 * len(versions))
    print(sep)
    sub = ' ' * 5
    for _, _ in versions:
        sub += f"  {'Child%':>7} {'Teen%':>7} {'FPR%':>6}"
    print(sub)
    print(sep)

    for thr in thresholds:
        row = f'{thr:>5.2f}'
        for _, sk in versions:
            cr, tr, fpr = evaluate_at_threshold(items, sk, thr)
            # Highlight if child >= 95 target
            cr_flag = '✓' if cr >= 95.0 else ' '
            row += f"  {cr:>6.1f}{cr_flag} {tr:>7.1f} {fpr:>6.1f}"
        print(row)

    print(sep)
    print('  ✓ = child recall ≥ 95% (target)')


def main():
    print('Loading LS data...')
    items = load_ls_items()
    print(f'  Loaded {len(items)} items')

    # Score with V7
    v7_model = BASE_DIR / 'data' / 'lgbm_underage_v7.txt'
    v7_feats = BASE_DIR / 'data' / 'lgbm_v7_features.json'
    if v7_model.exists() and v7_feats.exists():
        print('\nScoring V7...')
        score_items(items, v7_model, v7_feats, 'lgbm_v7')
        print('  Done')
    else:
        print('  V7 model not found — skipping')

    # Score with V8
    v8_model = BASE_DIR / 'data' / 'lgbm_underage_v8.txt'
    v8_feats = BASE_DIR / 'data' / 'lgbm_v8_features.json'
    if v8_model.exists() and v8_feats.exists():
        print('Scoring V8...')
        score_items(items, v8_model, v8_feats, 'lgbm_v8')
        print('  Done')
    else:
        print('  V8 model not found — skipping')

    # Score with V9
    v9_model = BASE_DIR / 'data' / 'lgbm_underage_v9.txt'
    v9_feats = BASE_DIR / 'data' / 'lgbm_v9_features.json'
    if v9_model.exists() and v9_feats.exists():
        print('Scoring V9...')
        score_items(items, v9_model, v9_feats, 'lgbm_v9')
        print('  Done')
    else:
        print('  V9 model not found — skipping')

    # Print comparison
    versions = []
    if v7_model.exists(): versions.append(('V7 (314f)', 'lgbm_v7'))
    if v8_model.exists(): versions.append(('V8 (155f)', 'lgbm_v8'))
    if v9_model.exists(): versions.append(('V9 (314f)', 'lgbm_v9'))

    if not versions:
        print('No models found!')
        return

    print_comparison(items, versions)

    # Also evaluate on 317-session hard examples using V9 scores
    v9_scores_path = BASE_DIR / 'data' / 'v9_317_scores.json'
    if v9_scores_path.exists():
        print('\n' + '='*60)
        print('317-SESSION HARD EXAMPLES (V9 scores):')
        v9_data = json.loads(v9_scores_path.read_text())
        hard_items = [r for r in v9_data if r.get('done')]

        # Need to score V7/V8 on these too
        hard_items_feat = []
        for r in hard_items:
            hard_items_feat.append({
                'id': r['id'],
                'label': r['label'],
                'minor': r.get('minor', 0),
                'adult': r.get('adult', 0),
                'underage_labels': r.get('underage_labels', {}),
                'adult_labels': r.get('adult_labels', {}),
                'no_underage_labels': r.get('no_underage_labels', {}),
                'lgbm_v7': 0, 'lgbm_v8': 0, 'lgbm_v9': 0,
            })

        if v7_model.exists():
            score_items(hard_items_feat, v7_model, v7_feats, 'lgbm_v7')
        if v8_model.exists():
            score_items(hard_items_feat, v8_model, v8_feats, 'lgbm_v8')
        if v9_model.exists():
            score_items(hard_items_feat, v9_model, v9_feats, 'lgbm_v9')

        print_comparison(hard_items_feat, versions)

    print('\nDone.')


if __name__ == '__main__':
    main()
