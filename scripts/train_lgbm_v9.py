#!/usr/bin/env python3
"""
train_lgbm_v9.py
----------------
Train LGBM V9 using rescored data with new tags (including no_underage_* features).

Key differences from V8:
  - Uses rescored images with 867-tag SigLIP-2 (tags.json minus roleplay)
  - Adds no_underage_* as new feature group (prefix: no_underage__)
  - Removes 5 deprecated tags if not present in new scores
  - Same training data structure, same CV strategy

Input:
  data/v9_317_scores.json    — 317-session rescored with new tags
  
Output:
  data/lgbm_underage_v9.txt     — LightGBM model
  data/lgbm_evaluate_v9.js      — JS evaluator for Piper deployment
  data/lgbm_v9_meta.json        — version metadata

Usage:
    python scripts/train_lgbm_v9.py
"""
import sys, os, json, re, math, datetime, struct, shutil
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
            tmp = Path('/tmp/_gal_v9_train.db')
            tmp.write_bytes(bytes(data))
            import sqlite3
            conn = sqlite3.connect(str(tmp))
            conn.row_factory = sqlite3.Row
            conn.execute('SELECT id FROM grafana_pool LIMIT 1').fetchall()
            return conn
        except:
            continue
    raise RuntimeError('No DB found')

def extract_item_v9(lbl, det, gid):
    """Extract features including no_underage_* from V9 scored piper_result."""
    labels = (det or {}).get('labels', {})
    return {
        'id': gid, 'label': lbl,
        'minor': (det or {}).get('minor', 0),
        'adult': (det or {}).get('adult', 0),
        'underage_labels': labels.get('underage') or {},
        'adult_labels':    labels.get('adult') or {},
        'no_underage_labels': labels.get('no_underage') or {},
    }

def load_ls_all_v9():
    """Load LS items from qwen3_age_results.json (old SigLIP scores, dict by task_id)."""
    import ast
    json_path = BASE_DIR / 'qwen3_age_results.json'
    raw = json_path.read_bytes().rstrip(b'\x00').decode('utf-8')
    data = json.loads(raw)
    items = []
    seen = set()
    for v in data.values():
        try:
            lbl = v.get('category')
            if lbl not in ('child', 'teen', 'adult'):
                continue
            siglip_raw = v.get('siglip2_details')
            if isinstance(siglip_raw, str):
                siglip2 = ast.literal_eval(siglip_raw)
            else:
                siglip2 = siglip_raw or {}
            det = siglip2.get('underage', {})
            gid = f"ls_{v['task_id']}"
            if gid in seen:
                continue
            seen.add(gid)
            items.append(extract_item_v9(lbl, det, gid))
        except Exception:
            pass
    by_lbl = {lb: sum(1 for x in items if x['label']==lb) for lb in ('child','teen','adult')}
    print(f'  LS items: {len(items)} — child={by_lbl["child"]}, teen={by_lbl["teen"]}, adult={by_lbl["adult"]}')
    return items

def load_grafana_pool_v9(v9_scores_path):
    """Load grafana pool, using V9 scores where available."""
    v9_scores = {}
    if Path(v9_scores_path).exists():
        for r in json.loads(Path(v9_scores_path).read_text()):
            if r.get('done'):
                v9_scores[r['id']] = r
        print(f'  V9 scores loaded: {len(v9_scores)} items')

    conn = _open_db()
    import sqlite3
    rows = conn.execute("""
        SELECT id, label, piper_result, export_batch
        FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
    """).fetchall()
    conn.close()

    items = []
    v9_used = 0
    for r in rows:
        lbl = r['label']
        gid = r['id']
        
        # Use V9 scores if available for this item
        if gid in v9_scores:
            v9 = v9_scores[gid]
            item = {
                'id': gid, 'label': lbl,
                'minor': v9.get('minor', 0),
                'adult': v9.get('adult', 0),
                'underage_labels': v9.get('underage_labels', {}),
                'adult_labels': v9.get('adult_labels', {}),
                'no_underage_labels': v9.get('no_underage_labels', {}),
                'export_batch': r['export_batch'],
            }
            v9_used += 1
        else:
            # Fall back to DB scores (no no_underage_labels)
            try:
                pr = json.loads(r['piper_result'])
                det = (pr.get('siglip2_details') or {}).get('underage', {})
                item = extract_item_v9(lbl, det, gid)
                item['export_batch'] = r['export_batch']
            except:
                continue
        items.append(item)

    print(f'  Total grafana pool: {len(items)}, V9-rescored: {v9_used}')
    return items

def sanitize_feat(name: str) -> str:
    """Replace JSON-unsafe chars in feature name (e.g. ':' from :x20 multipliers)."""
    return name.replace(':', '_x').replace('"', '').replace("'", '').replace('{', '').replace('}', '')


def build_feature_matrix_v9(items, feature_names=None):
    """Build feature matrix including no_underage__ features."""
    if feature_names is None:
        feat_set = set()
        for item in items:
            feat_set.update(sanitize_feat(k) for k in item.get('underage_labels', {}).keys())
            feat_set.update('adult__' + sanitize_feat(k) for k in item.get('adult_labels', {}).keys())
            feat_set.update('no_underage__' + sanitize_feat(k) for k in item.get('no_underage_labels', {}).keys())
        feature_names = sorted(feat_set)

    feat_idx = {f: i for i, f in enumerate(feature_names)}
    n = len(items)
    m = len(feature_names)
    X = np.zeros((n, m), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)

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

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    V9_SCORES = BASE_DIR / 'data' / 'v9_317_scores.json'
    
    print('Loading data...')
    ls_items = load_ls_all_v9()
    print(f'  LS items: {len(ls_items)}')
    pool_items = load_grafana_pool_v9(V9_SCORES)
    
    # Retraining negatives
    neg_items = []
    for neg_file in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json']:
        npath = BASE_DIR / 'data' / neg_file
        if npath.exists():
            for r in json.loads(npath.read_text()):
                lbl = r.get('label')
                if lbl not in ('child','teen','adult'): continue
                neg_items.append({
                    'id': r.get('id'), 'label': lbl,
                    'underage_labels': r.get('underage_labels', {}),
                    'adult_labels': r.get('adult_labels', {}),
                    'no_underage_labels': {},  # no V9 scores yet
                })
    print(f'  Extra negatives: {len(neg_items)}')
    
    all_items = ls_items + pool_items + neg_items
    print(f'  Total: {len(all_items)}')
    
    # Build features
    X, y, feat_names = build_feature_matrix_v9(all_items)
    print(f'Features: {len(feat_names)} (incl. {sum(1 for f in feat_names if f.startswith("no_underage__"))} no_underage__*)')
    print(f'Labels: pos={y.sum()}, neg={(1-y).sum()}')
    
    # Sample weights: 317-session hard examples × 3
    weights = np.ones(len(all_items))
    for i, item in enumerate(all_items):
        if item.get('export_batch') == '2026-05-20 UTC':
            weights[i] = 3.0
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
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
    
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        wtr = weights[tr]
        
        ds_tr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feat_names)
        ds_va = lgb.Dataset(Xva, label=yva, reference=ds_tr, feature_name=feat_names)
        
        model = lgb.train(params, ds_tr, num_boost_round=200,
                         valid_sets=[ds_va],
                         callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=0)])
        
        va_pred = model.predict(Xva)
        auc = roc_auc_score(yva, va_pred)
        cv_aucs.append(auc)
        print(f'  Fold {fold+1}/5: AUC={auc:.4f}')
    
    mean_auc = np.mean(cv_aucs)
    std_auc = np.std(cv_aucs)
    print(f'CV AUC: {mean_auc:.4f} ± {std_auc:.4f}')
    
    # Final model on all data
    ds_all = lgb.Dataset(X, label=y, weight=weights, feature_name=feat_names)
    final_model = lgb.train(params, ds_all, num_boost_round=150)
    
    # Save model
    MODEL_PATH = BASE_DIR / 'data' / 'lgbm_underage_v9.txt'
    final_model.save_model(str(MODEL_PATH))
    print(f'Model saved: {MODEL_PATH}')
    
    # Save feature list
    FEAT_PATH = BASE_DIR / 'data' / 'lgbm_v9_features.json'
    FEAT_PATH.write_text(json.dumps(feat_names, indent=2))
    
    # Meta
    META_PATH = BASE_DIR / 'data' / 'lgbm_v9_meta.json'
    META_PATH.write_text(json.dumps({
        'version': 'v9', 'threshold': 0.45,
        'cv_auc': round(mean_auc, 4), 'cv_std': round(std_auc, 4),
        'n_samples': len(all_items), 'n_features': len(feat_names),
        'no_underage_features': sum(1 for f in feat_names if f.startswith('no_underage__')),
        'blocking_rule': 'lgbm_score >= 0.45 OR minor >= 0.72',
        'tag_set': 'tags.json (867 tags, no roleplay)',
        'trained_at': datetime.datetime.utcnow().isoformat(),
    }, indent=2))
    
    print(f'\nV9 training complete!')
    print(f'Next: generate lgbm_evaluate_v9.js from the model')
    print(f'Run: python scripts/export_lgbm_v9_piper.py')

if __name__ == '__main__':
    main()
