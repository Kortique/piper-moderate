#!/usr/bin/env python3
"""
train_lgbm_v10.py
-----------------
Train LGBM V10: LS holdout items rescored through ce79f7e299 (867 tags).
Eliminates the distribution mismatch of V9 (LS had no_underage=0 for all).

Key differences from V9:
  - LS holdout items use ls_holdout_rescored.json (real no_underage features)
  - 317-session items still weight=3.0
  - Both data sources now have consistent 867-tag features
  - Extra negatives still have no_underage=0 (older pipeline)

Input:
  data/ls_holdout_rescored.json   — LS items rescored with ce79f7e299
  data/v9_317_scores.json         — 317-session with new tags (weight×3)

Output:
  data/lgbm_underage_v10.txt
  data/lgbm_v10_features.json
  data/lgbm_v10_meta.json

Usage:
    python scripts/train_lgbm_v10.py
"""
import sys, os, json, datetime, struct
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _open_db():
    candidates = sorted((BASE_DIR / 'backups').glob('gallery_*.db'), reverse=True)
    for db_path in candidates:
        try:
            data = bytearray(db_path.read_bytes())
            struct.pack_into('>I', data, 28, len(data) // 4096)
            tmp = Path('/tmp/_gal_v10_train.db')
            tmp.write_bytes(bytes(data))
            import sqlite3
            conn = sqlite3.connect(str(tmp))
            conn.row_factory = sqlite3.Row
            conn.execute('SELECT id FROM grafana_pool LIMIT 1').fetchall()
            return conn
        except Exception:
            continue
    raise RuntimeError('No DB found')


def sanitize_feat(name: str) -> str:
    return name.replace(':', '_x').replace('"', '').replace("'", '').replace('{', '').replace('}', '')


def load_ls_rescored():
    """Load LS holdout from ls_holdout_rescored.json (new tags)."""
    path = BASE_DIR / 'data' / 'ls_holdout_rescored.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found — run rescore_ls_holdout.py first')
    data = json.loads(path.read_text())
    done = [r for r in data if r.get('done')]

    items = []
    for r in done:
        lbl = r.get('label')
        if lbl not in ('child', 'teen', 'adult'):
            continue
        items.append({
            'id': f"ls_{r['task_id']}",
            'label': lbl,
            'minor': r.get('minor', 0),
            'adult': r.get('adult', 0),
            'underage_labels':    r.get('underage_labels', {}),
            'adult_labels':       r.get('adult_labels', {}),
            'no_underage_labels': r.get('no_underage_labels', {}),
        })
    by_lbl = {lb: sum(1 for x in items if x['label'] == lb) for lb in ('child', 'teen', 'adult')}
    print(f'  LS rescored: {len(items)} — child={by_lbl["child"]}, teen={by_lbl["teen"]}, adult={by_lbl["adult"]}')
    return items


def load_317_session():
    """Load 317-session items from v9_317_scores.json (weight × 3)."""
    path = BASE_DIR / 'data' / 'v9_317_scores.json'
    if not path.exists():
        print('  317-session scores not found — skipping')
        return []
    data = json.loads(path.read_text())
    done = [r for r in data if r.get('done')]
    items = []
    for r in done:
        lbl = r.get('label')
        if lbl not in ('child', 'teen', 'adult'):
            continue
        items.append({
            'id': r['id'],
            'label': lbl,
            'minor': r.get('minor', 0),
            'adult': r.get('adult', 0),
            'underage_labels':    r.get('underage_labels', {}),
            'adult_labels':       r.get('adult_labels', {}),
            'no_underage_labels': r.get('no_underage_labels', {}),
            'export_batch': '2026-05-20 UTC',  # mark for weight×3
        })
    by_lbl = {lb: sum(1 for x in items if x['label'] == lb) for lb in ('child', 'teen', 'adult')}
    print(f'  317-session: {len(items)} — child={by_lbl["child"]}, teen={by_lbl["teen"]}, adult={by_lbl["adult"]}')
    return items


def load_grafana_pool_extra():
    """Load grafana_pool items NOT in 317-session (older data, no no_underage)."""
    conn = _open_db()
    import sqlite3
    rows = conn.execute("""
        SELECT id, label, piper_result, export_batch
        FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
        AND (export_batch IS NULL OR export_batch != '2026-05-20 UTC')
    """).fetchall()
    conn.close()

    items = []
    for r in rows:
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            lbl = (det.get('labels') or {})
            items.append({
                'id': r['id'],
                'label': r['label'],
                'minor': det.get('minor', 0),
                'adult': det.get('adult', 0),
                'underage_labels':    lbl.get('underage') or {},
                'adult_labels':       lbl.get('adult') or {},
                'no_underage_labels': lbl.get('no_underage') or {},
                'export_batch': r['export_batch'],
            })
        except Exception:
            continue
    print(f'  Grafana pool (non-317): {len(items)}')
    return items


def load_extra_negatives():
    neg_items = []
    for neg_file in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json']:
        npath = BASE_DIR / 'data' / neg_file
        if npath.exists():
            for r in json.loads(npath.read_text()):
                lbl = r.get('label')
                if lbl not in ('child', 'teen', 'adult'):
                    continue
                neg_items.append({
                    'id': r.get('id'),
                    'label': lbl,
                    'underage_labels':    r.get('underage_labels', {}),
                    'adult_labels':       r.get('adult_labels', {}),
                    'no_underage_labels': {},
                })
    print(f'  Extra negatives: {len(neg_items)}')
    return neg_items


def build_feature_matrix(items, feature_names=None):
    if feature_names is None:
        feat_set = set()
        for item in items:
            feat_set.update(sanitize_feat(k) for k in item.get('underage_labels', {}).keys())
            feat_set.update('adult__' + sanitize_feat(k) for k in item.get('adult_labels', {}).keys())
            feat_set.update('no_underage__' + sanitize_feat(k) for k in item.get('no_underage_labels', {}).keys())
        feature_names = sorted(feat_set)

    feat_idx = {f: i for i, f in enumerate(feature_names)}
    X = np.zeros((len(items), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int8)

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
        y[i] = 1 if item['label'] in ('child', 'teen') else 0

    return X, y, feature_names


def main():
    print('=== V10 Training (LS rescored + 317-session) ===\n')

    ls_items       = load_ls_rescored()
    s317_items     = load_317_session()
    pool_items     = load_grafana_pool_extra()
    neg_items      = load_extra_negatives()

    all_items = ls_items + s317_items + pool_items + neg_items
    # Deduplicate by id (prefer first occurrence = ls_rescored has priority)
    seen = set()
    deduped = []
    for it in all_items:
        iid = it.get('id')
        if iid and iid in seen:
            continue
        if iid:
            seen.add(iid)
        deduped.append(it)
    all_items = deduped

    by_lbl = {lb: sum(1 for x in all_items if x['label'] == lb) for lb in ('child', 'teen', 'adult')}
    print(f'\nTotal: {len(all_items)} — child={by_lbl["child"]}, teen={by_lbl["teen"]}, adult={by_lbl["adult"]}')

    X, y, feat_names = build_feature_matrix(all_items)
    nu_feats = sum(1 for f in feat_names if f.startswith('no_underage__'))
    print(f'Features: {len(feat_names)} (incl. {nu_feats} no_underage__*)')
    print(f'Labels: pos={int(y.sum())}, neg={int((1-y).sum())}')

    # Sample weights
    weights = np.ones(len(all_items))
    for i, item in enumerate(all_items):
        if item.get('export_batch') == '2026-05-20 UTC':
            weights[i] = 3.0

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 3,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
    }

    # 5-fold CV
    print('\nCross-validation...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=weights[tr], feature_name=feat_names)
        ds_va = lgb.Dataset(X[va], label=y[va], reference=ds_tr, feature_name=feat_names)
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)])
        auc = roc_auc_score(y[va], model.predict(X[va]))
        cv_aucs.append(auc)
        print(f'  Fold {fold+1}/5: AUC={auc:.4f}')

    mean_auc = float(np.mean(cv_aucs))
    std_auc  = float(np.std(cv_aucs))
    print(f'CV AUC: {mean_auc:.4f} ± {std_auc:.4f}')

    # Final model
    print('\nFinal model (all data)...')
    ds_all = lgb.Dataset(X, label=y, weight=weights, feature_name=feat_names)
    final_model = lgb.train(params, ds_all, num_boost_round=150)

    MODEL_PATH = BASE_DIR / 'data' / 'lgbm_underage_v10.txt'
    FEAT_PATH  = BASE_DIR / 'data' / 'lgbm_v10_features.json'
    META_PATH  = BASE_DIR / 'data' / 'lgbm_v10_meta.json'

    final_model.save_model(str(MODEL_PATH))
    FEAT_PATH.write_text(json.dumps(feat_names, indent=2))
    META_PATH.write_text(json.dumps({
        'version': 'v10',
        'cv_auc': round(mean_auc, 4),
        'cv_std': round(std_auc, 4),
        'n_samples': len(all_items),
        'n_features': len(feat_names),
        'no_underage_features': nu_feats,
        'blocking_rule': 'lgbm_score >= 0.45 OR minor >= 0.72',
        'tag_set': 'tags.json (867 tags)',
        'ls_rescored': True,
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))

    print(f'\nSaved:')
    print(f'  {MODEL_PATH}')
    print(f'  {FEAT_PATH}')
    print(f'  {META_PATH}')
    print(f'\nV10 training complete! Run compare_v7_v8_v9_v10.py to evaluate.')


if __name__ == '__main__':
    main()
