"""
train_lgbm_v6.py
----------------
Retrain LGBM underage classifier (V6) incorporating:
  - All existing LS training data (positives + adults from DB)
  - 40 FN hard positives (child images missed by V5)
  - 123 FP hard negatives (adult images incorrectly blocked by V5)
  - All existing retraining negatives

New in V6:
  - 22 new adult__ counter-tag features
  - 9 new underage tag features
  - Total: 174 features (vs 312 in V5)

Output:
  data/lgbm_underage_v6.txt     — LightGBM model file
  data/lgbm_evaluate_v6.js      — compact JS evaluator for Piper (same format as V5)

Usage:
    pip install lightgbm scikit-learn pandas --break-system-packages
    python scripts/train_lgbm_v6.py
"""
import sys, os, json, re, math, datetime, struct
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── DB helper ─────────────────────────────────────────────────────────────────

def _open_db():
    data = bytearray((BASE_DIR / 'gallery.db').read_bytes())
    struct.pack_into('>I', data, 28, len(data) // 4096)
    tmp = Path('/tmp/_gal_v6_train.db')
    tmp.write_bytes(bytes(data))
    import sqlite3
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ls_all():
    """Load ALL LS items — tries JSON first (DB may have page corruption)."""
    json_path = BASE_DIR / 'qwen3_age_results.json'
    if json_path.exists():
        raw = json_path.read_bytes().rstrip(b'\x00').decode('utf-8')
        data = json.loads(raw)
        items = []
        for v in data.values():
            try:
                age = (v.get('age') or {}).get('ageFrom')
                if age is None:
                    continue
                det = (v.get('siglip2_details') or {}).get('underage', {})
                labels = det.get('labels', {})
                label = 'child' if age <= 14 else 'teen' if age <= 17 else 'adult'
                items.append({
                    'id': f"ls_{v['task_id']}",
                    'label': label,
                    'minor': det.get('minor', 0),
                    'adult': det.get('adult', 0),
                    'underage_labels': labels.get('underage') or {},
                    'adult_labels': labels.get('adult') or {},
                })
            except: pass
        print(f"  LS items (from JSON): {len(items)} total ({sum(1 for x in items if x['label']=='child')} child, "
              f"{sum(1 for x in items if x['label']=='teen')} teen, "
              f"{sum(1 for x in items if x['label']=='adult')} adult)")
        return items

    # Fallback: read from DB (may fail with page corruption)
    import sqlite3
    conn = _open_db()
    items = []
    for offset in range(0, 2500, 200):
        try:
            rows = conn.execute(
                "SELECT task_id, age_from, siglip2_details FROM ls_images "
                "WHERE siglip2_details IS NOT NULL AND age_from IS NOT NULL "
                "LIMIT 200 OFFSET ?", (offset,)
            ).fetchall()
            if not rows: break
            for r in rows:
                try:
                    det = json.loads(r['siglip2_details']).get('underage', {})
                    labels = det.get('labels', {})
                    age = r['age_from']
                    label = 'child' if age <= 14 else 'teen' if age <= 17 else 'adult'
                    items.append({
                        'id': f"ls_{r['task_id']}",
                        'label': label,
                        'minor': det.get('minor', 0),
                        'adult': det.get('adult', 0),
                        'underage_labels': labels.get('underage') or {},
                        'adult_labels': labels.get('adult') or {},
                    })
                except: pass
        except: pass
    conn.close()
    print(f"  LS items (from DB): {len(items)} total ({sum(1 for x in items if x['label']=='child')} child, "
          f"{sum(1 for x in items if x['label']=='teen')} teen, "
          f"{sum(1 for x in items if x['label']=='adult')} adult)")
    return items


def load_negatives():
    """Load existing retraining negatives, skipping any that appear in FP hard negatives."""
    fp_hard = json.loads((BASE_DIR / 'data' / 'v6_fp_hard_negatives.json').read_text())
    fp_ids = {r['id'] for r in fp_hard}

    items = list(fp_hard)  # Start with FP hard negatives
    seen = set(fp_ids)

    for fname in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json']:
        path = BASE_DIR / 'data' / fname
        if not path.exists(): continue
        data = json.loads(path.read_text())
        added = 0
        for r in data:
            gid = r.get('id', '')
            if gid in seen: continue
            items.append({
                'id': gid,
                'label': 'adult',
                'minor': r.get('minor_score') or r.get('minor', 0),
                'adult': r.get('adult_score') or r.get('adult', 0),
                'underage_labels': r.get('underage_labels') or {},
                'adult_labels': r.get('adult_labels') or {},
            })
            seen.add(gid)
            added += 1
        print(f"  {added} negatives from {fname}")

    print(f"  Total negatives: {len(items)} ({len(fp_hard)} FP hard + existing)")
    return items


def load_grafana_pool():
    """Load labeled grafana_pool items from disagree_pool.json."""
    pool_path = BASE_DIR / 'data' / 'disagree_pool.json'
    if not pool_path.exists():
        return []
    raw = pool_path.read_bytes().rstrip(b'\x00').decode('utf-8')
    data = json.loads(raw)
    items = []
    for v in data.values():
        lbl = v.get('label')
        if lbl not in ('child', 'teen', 'adult'):
            continue
        pr = v.get('piper_result') or {}
        det = (pr.get('siglip2_details') or {}).get('underage', {})
        if not det:
            continue
        labels = det.get('labels', {})
        items.append({
            'id': v['id'],
            'label': lbl,
            'minor': det.get('minor', 0),
            'adult': det.get('adult', 0),
            'underage_labels': labels.get('underage') or {},
            'adult_labels': labels.get('adult') or {},
        })
    print(f"  Grafana pool items: {len(items)} ({sum(1 for x in items if x['label']=='child')} child, "
          f"{sum(1 for x in items if x['label']=='teen')} teen, "
          f"{sum(1 for x in items if x['label']=='adult')} adult)")
    return items


def load_fn_hard_positives():
    """Load FN hard positives (child images missed by V5)."""
    fn = json.loads((BASE_DIR / 'data' / 'v6_fn_hard_positives.json').read_text())
    print(f"  FN hard positives: {len(fn)}")
    return fn


# ── Feature engineering ───────────────────────────────────────────────────────

def build_feature_matrix(items, feature_names=None):
    """Convert items to (X, y, feature_names)."""
    if feature_names is None:
        all_feats = set()
        for item in items:
            all_feats.update(item.get('underage_labels', {}).keys())
            for k in item.get('adult_labels', {}).keys():
                all_feats.add(f'adult__{k}')
        feature_names = sorted(all_feats)

    feat_idx = {f: i for i, f in enumerate(feature_names)}
    X = np.zeros((len(items), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int32)

    for i, item in enumerate(items):
        for k, v in item.get('underage_labels', {}).items():
            if k in feat_idx:
                X[i, feat_idx[k]] = float(v)
        for k, v in item.get('adult_labels', {}).items():
            fk = f'adult__{k}'
            if fk in feat_idx:
                X[i, feat_idx[fk]] = float(v)
        y[i] = 1 if item['label'] in ('child', 'teen') else 0

    return X, y, feature_names


# ── Training ──────────────────────────────────────────────────────────────────

def train_lgbm(X_train, y_train, X_test=None, y_test=None):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 15,
        'n_estimators': 150,
        'learning_rate': 0.04,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'is_unbalance': True,
    }
    dataset = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dataset, num_boost_round=150)
    if X_test is not None:
        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, pred)
        print(f"    AUC={auc:.4f}")
    return model


def cross_validate(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        model = train_lgbm(X[ti], y[ti], X[vi], y[vi])
        pred = model.predict(X[vi])
        auc = roc_auc_score(y[vi], pred)
        aucs.append(auc)
        print(f"  Fold {fold+1}: AUC={auc:.4f}")
    return aucs


def find_threshold(model, X, y, target_recall=0.90):
    """Find LGBM threshold that achieves target recall with best precision."""
    preds = model.predict(X)
    prec, rec, thresholds = precision_recall_curve(y, preds)
    # Find thresholds where recall >= target
    valid = [(t, p) for t, p, r in zip(thresholds, prec[:-1], rec[:-1]) if r >= target_recall]
    if valid:
        best = max(valid, key=lambda x: x[1])  # max precision at recall >= target
        return best[0]
    return 0.80  # fallback


# ── Model export ──────────────────────────────────────────────────────────────

def export_compact_js(model, feature_names, out_path: Path, version="v6", n_samples=0, cv_auc=0):
    """Export model as compact JS (same format as lgbm_evaluate_v5_piper.js)."""
    today = datetime.date.today().isoformat()
    model_str = model.model_to_string()

    # Parse trees
    trees = []
    tree_blocks = re.split(r'Tree=\d+', model_str)
    for block in tree_blocks[1:]:
        # Parse leaf values
        lv_match = re.search(r'leaf_value=([^\n]+)', block)
        nf_match = re.search(r'split_feature=([^\n]+)', block)
        nt_match = re.search(r'threshold=([^\n]+)', block)
        lc_match = re.search(r'left_child=([^\n]+)', block)
        rc_match = re.search(r'right_child=([^\n]+)', block)
        nc_match = re.search(r'num_leaves=(\d+)', block)

        if not (lv_match and nf_match):
            continue

        leaf_values = [float(x) for x in lv_match.group(1).split()]
        split_features = [int(x) for x in nf_match.group(1).split()]
        thresholds = [float(x) for x in nt_match.group(1).split()]
        left_children = [int(x) for x in lc_match.group(1).split()]
        right_children = [int(x) for x in rc_match.group(1).split()]

        num_internal = len(split_features)
        root = 0

        # Find root (node not referenced as child of any other node)
        all_children = set(left_children + right_children)
        for ni in range(num_internal):
            if ni not in all_children:
                root = ni
                break

        splits = [[split_features[i], thresholds[i]] for i in range(num_internal)]
        children = [[left_children[i], right_children[i]] for i in range(num_internal)]
        trees.append({'r': root, 's': splits, 'c': children, 'l': leaf_values})

    feats_js = json.dumps(feature_names)
    trees_js = json.dumps(trees, separators=(',', ':'))

    js = f"""// lgbm_evaluate_v6 — LGBM Underage Scorer V6
// Model: 150 trees, 15 leaves, CV AUC={cv_auc:.3f}, trained {today} on {n_samples} samples
// V6.1: retrained on new labeling scheme (child/teen/adult), clean LS+grafana only
// Input: `labels` object from siglip2 underage classifier
// Outputs: score 0-1 (underage probability)

const LGBM_FEATURES = {feats_js};

const LGBM_TREES = {trees_js};

function lgbm_predict_v6(vec) {{
  let score = 0;
  for (const t of LGBM_TREES) {{
    let node = t.r;
    while (node >= 0) {{
      const [fi, thr] = t.s[node];
      const [l, r] = t.c[node];
      node = vec[fi] <= thr ? l : r;
    }}
    score += t.l[-(node + 1)];
  }}
  return 1 / (1 + Math.exp(-score));
}}

function lgbm_evaluate_v6(labelsObj) {{
  const vec = LGBM_FEATURES.map(f => {{
    if (f.startsWith('adult__')) {{
      const key = f.slice(7);
      return (labelsObj.adult && labelsObj.adult[key]) || 0;
    }}
    return (labelsObj.underage && labelsObj.underage[f]) || 0;
  }});
  return lgbm_predict_v6(vec);
}}
"""
    out_path.write_text(js, encoding='utf-8')
    print(f"  Exported JS: {out_path} ({len(feature_names)} features, {len(trees)} trees)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Training LGBM V6 ===")

    print("\n[1] Loading data...")
    ls_items = load_ls_all()
    grafana_items = load_grafana_pool()
    # NOTE: v6_fn_hard_positives and v6_fp_hard_negatives excluded —
    #       compiled under old labeling criteria, incompatible with new scheme.
    # lgbm_retraining_negatives also excluded — 317/330 overlap with grafana_pool
    # which now has authoritative new labels.

    # Simple merge: LS + grafana, dedup by ID (LS has priority)
    seen = {}
    all_items = []
    for item in ls_items + grafana_items:
        if item['id'] not in seen:
            seen[item['id']] = item
            all_items.append(item)

    pos = sum(1 for x in all_items if x['label'] in ('child', 'teen'))
    neg = sum(1 for x in all_items if x['label'] == 'adult')
    child_n = sum(1 for x in all_items if x['label'] == 'child')
    teen_n  = sum(1 for x in all_items if x['label'] == 'teen')
    print(f"\n  Combined dataset: {len(all_items)} items")
    print(f"  child={child_n}, teen={teen_n}, adult={neg}")

    print("\n[2] Building feature matrix...")
    X, y, feature_names = build_feature_matrix(all_items)
    print(f"  Feature matrix: {X.shape}, features: {len(feature_names)}")

    # Show feature distribution by category
    und_feats = [f for f in feature_names if not f.startswith('adult__')]
    adult_feats = [f for f in feature_names if f.startswith('adult__')]
    print(f"  Underage features: {len(und_feats)}, Adult counter-features: {len(adult_feats)}")

    print("\n[3] Cross-validation (5-fold)...")
    aucs = cross_validate(X, y)
    mean_auc = np.mean(aucs)
    print(f"  Mean CV AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")

    print("\n[4] Training final model on all data...")
    final_model = train_lgbm(X, y)

    print("\n[5] Threshold analysis (targets: child≥95% teen≥80% adult_FP≤20%)...")
    preds = final_model.predict(X)
    # also apply minor score rule (independent of LGBM)
    minors = np.array([item.get('minor', 0.0) for item in all_items])
    MINOR_THR = 0.72

    print(f"  {'thr':>5} | {'child TP':>9} {'c-FN':>6} | {'teen TP':>9} {'t-FN':>6} | {'adult FP':>9} {'a-FP':>6}")
    print("  " + "-"*65)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        blocked = (preds >= thr) | (minors >= MINOR_THR)
        cat_res = {'child': [0,0], 'teen': [0,0], 'adult': [0,0]}
        for i, item in enumerate(all_items):
            cat = item['label']
            if cat not in cat_res: continue
            if blocked[i]: cat_res[cat][0] += 1
            cat_res[cat][1] += 1
        c = cat_res['child']; t = cat_res['teen']; a = cat_res['adult']
        c_tp = c[0]/c[1]*100; t_tp = t[0]/t[1]*100; a_fp = a[0]/a[1]*100
        ok = ('✓' if c_tp >= 95 else '✗') + ('✓' if t_tp >= 80 else '✗') + ('✓' if a_fp <= 20 else '✗')
        print(f"  {thr:>5.2f} | {c_tp:>8.1f}% {c[1]-c[0]:>6} | {t_tp:>8.1f}% {t[1]-t[0]:>6} | {a_fp:>8.1f}% {a[0]:>6}  {ok}")

    # Feature importance
    print("\n[6] Top features by importance:")
    importance = final_model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    for name, imp in feat_imp[:20]:
        print(f"  {imp:8.1f}  {name}")

    print("\n[7] Exporting model...")
    model_path = BASE_DIR / 'data' / 'lgbm_underage_v6.txt'
    js_path = BASE_DIR / 'data' / 'lgbm_evaluate_v6.js'
    final_model.save_model(str(model_path))
    print(f"  Saved: {model_path}")
    export_compact_js(final_model, feature_names, js_path, version='v6',
                      n_samples=len(all_items), cv_auc=mean_auc)

    # Also save feature list
    feat_list_path = BASE_DIR / 'data' / 'lgbm_underage_v6.txt'
    print(f"\nDone! Model: {model_path}")
    print(f"JS:    {js_path}")
