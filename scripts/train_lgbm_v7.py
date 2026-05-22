"""
train_lgbm_v7.py
----------------
Retrain LGBM underage classifier V7 incorporating:
  - All existing LS training data (from qwen3_age_results.json)
  - All labeled grafana_pool items (disagree_pool.json, all sessions)
  - NEW: 317-session hard positives (1 child + 29 teen FNs)
  - NEW: 317-session hard negatives (39 adult FPs)
  - Existing retraining negatives (lgbm_retraining_negatives.json)
  - V6 hard positives / negatives

Key improvement over V6:
  - Adds 317-session FP/FN hard examples (new session, same feature space)
  - These images had ambiguous LGBM scores (0.40-0.80) — model needs to learn them
  - No tag changes needed (siglip2 already has good 153-feature representation)

Output:
  data/lgbm_underage_v7.txt     — LightGBM model file
  data/lgbm_evaluate_v7.js      — compact JS evaluator for Piper deployment

Usage:
    pip install lightgbm scikit-learn pandas --break-system-packages
    python scripts/train_lgbm_v7.py
"""
import sys, os, json, re, math, datetime, struct, shutil
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
    db_path = BASE_DIR / 'gallery.db'
    if not db_path.exists():
        # try latest backup
        backups = sorted((BASE_DIR / 'backups').glob('gallery_*.db'))
        if not backups:
            raise FileNotFoundError("gallery.db not found")
        db_path = backups[-1]
        print(f"  Using backup: {db_path.name}")
    data = bytearray(db_path.read_bytes())
    struct.pack_into('>I', data, 28, len(data) // 4096)
    tmp = Path('/tmp/_gal_v7_train.db')
    tmp.write_bytes(bytes(data))
    import sqlite3
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


# ── Data loading ──────────────────────────────────────────────────────────────

def extract_item(lbl, det, gid):
    labels = (det or {}).get('labels', {})
    return {
        'id': gid,
        'label': lbl,
        'minor': (det or {}).get('minor', 0),
        'adult': (det or {}).get('adult', 0),
        'underage_labels': labels.get('underage') or {},
        'adult_labels':    labels.get('adult') or {},
    }


def load_ls_all():
    """Load ALL LS items from qwen3_age_results.json."""
    json_path = BASE_DIR / 'qwen3_age_results.json'
    raw = json_path.read_bytes().rstrip(b'\x00').decode('utf-8')
    data = json.loads(raw)
    items = []
    seen = set()
    for v in data.values():
        try:
            age = (v.get('age') or {}).get('ageFrom')
            if age is None:
                continue
            det = (v.get('siglip2_details') or {}).get('underage', {})
            label = 'child' if age <= 14 else 'teen' if age <= 17 else 'adult'
            gid = f"ls_{v['task_id']}"
            if gid in seen: continue
            seen.add(gid)
            items.append(extract_item(label, det, gid))
        except:
            pass
    print(f"  LS (qwen3_age_results): {len(items)} "
          f"({sum(1 for x in items if x['label']=='child')} child, "
          f"{sum(1 for x in items if x['label']=='teen')} teen, "
          f"{sum(1 for x in items if x['label']=='adult')} adult)")
    return items, seen


def load_grafana_all(exclude_ids=None):
    """Load ALL labeled grafana_pool items from the DB."""
    if exclude_ids is None:
        exclude_ids = set()
    conn = _open_db()
    rows = conn.execute("""
        SELECT id, label, piper_result
        FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
    """).fetchall()
    conn.close()
    
    items = []
    seen = set()
    for r in rows:
        gid = r['id']
        if gid in exclude_ids or gid in seen:
            continue
        seen.add(gid)
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            if not det:
                continue
            items.append(extract_item(r['label'], det, gid))
        except:
            pass
    print(f"  Grafana pool: {len(items)} "
          f"({sum(1 for x in items if x['label']=='child')} child, "
          f"{sum(1 for x in items if x['label']=='teen')} teen, "
          f"{sum(1 for x in items if x['label']=='adult')} adult)")
    return items


def load_317_hard_examples():
    """Load the 317-session FP/FN items as hard examples (will be weighted more)."""
    conn = _open_db()
    rows = conn.execute("""
        SELECT id, label, piper_result
        FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND export_batch = '2026-05-20 UTC'
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
    """).fetchall()
    conn.close()
    
    LGBM_THR = 0.55
    MINOR_THR = 0.72
    
    hard_pos = []  # FN: labeled minor but not blocked
    hard_neg = []  # FP: labeled adult but blocked
    
    for r in rows:
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            if not det:
                continue
            lgbm = (det.get('lgbm') or {}).get('score', 0)
            minor = det.get('minor', 0)
            blocked = lgbm >= LGBM_THR or minor >= MINOR_THR
            lbl = r['label']
            
            item = extract_item(lbl, det, r['id'])
            
            if not blocked and lbl in ('child', 'teen'):
                hard_pos.append(item)
            elif blocked and lbl == 'adult':
                hard_neg.append(item)
        except:
            pass
    
    print(f"  317-session hard positives (FN): {len(hard_pos)} "
          f"({sum(1 for x in hard_pos if x['label']=='child')} child, "
          f"{sum(1 for x in hard_pos if x['label']=='teen')} teen)")
    print(f"  317-session hard negatives (FP): {len(hard_neg)}")
    return hard_pos, hard_neg


def load_existing_negatives(exclude_ids=None):
    """Load existing hard negatives from prior retraining sessions."""
    if exclude_ids is None:
        exclude_ids = set()
    items = []
    seen = set(exclude_ids)
    
    for fname in ['lgbm_retraining_negatives.json', 'lgbm_new_negatives.json',
                  'v6_fp_hard_negatives.json']:
        path = BASE_DIR / 'data' / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        added = 0
        for r in data:
            gid = r.get('id', '')
            if gid in seen:
                continue
            seen.add(gid)
            items.append({
                'id': gid,
                'label': 'adult',
                'minor': r.get('minor_score') or r.get('minor', 0),
                'adult': r.get('adult_score') or r.get('adult', 0),
                'underage_labels': r.get('underage_labels') or {},
                'adult_labels': r.get('adult_labels') or {},
            })
            added += 1
        print(f"  {added} existing negatives from {fname}")
    
    print(f"  Total existing negatives: {len(items)}")
    return items


def load_v6_fn_hard_positives():
    """Load V6 child FN hard positives."""
    path = BASE_DIR / 'data' / 'v6_fn_hard_positives.json'
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    print(f"  V6 child FN hard positives: {len(data)}")
    return data


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

def train_lgbm(X_train, y_train, sample_weight=None, X_test=None, y_test=None):
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
    dataset = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    model = lgb.train(params, dataset, num_boost_round=150)
    if X_test is not None:
        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, pred)
        print(f"    Test AUC={auc:.4f}")
    return model


def cross_validate(X, y, weight, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        w_train = weight[ti] if weight is not None else None
        model = train_lgbm(X[ti], y[ti], w_train)
        pred = model.predict(X[vi])
        auc = roc_auc_score(y[vi], pred)
        aucs.append(auc)
        print(f"  Fold {fold+1}: AUC={auc:.4f}")
    print(f"  CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return aucs


# ── Eval on 317-session ────────────────────────────────────────────────────────

def eval_317_session(model, feature_names, threshold):
    """Evaluate on the 317-session data and print recall/FPR."""
    conn = _open_db()
    rows = conn.execute("""
        SELECT id, label, piper_result
        FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND export_batch = '2026-05-20 UTC'
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
    """).fetchall()
    conn.close()
    
    items = []
    for r in rows:
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            if not det:
                continue
            items.append(extract_item(r['label'], det, r['id']))
        except:
            pass
    
    X, y, _ = build_feature_matrix(items, feature_names)
    preds = model.predict(X)
    
    # Also check minor score
    minors = np.array([it.get('minor', 0) for it in items])
    labels = [it['label'] for it in items]
    
    # V7 blocking: lgbm_score >= threshold OR minor >= 0.72
    blocked = (preds >= threshold) | (minors >= 0.72)
    
    children = [(b, l) for b, l in zip(blocked, labels) if l == 'child']
    teens = [(b, l) for b, l in zip(blocked, labels) if l == 'teen']
    adults = [(b, l) for b, l in zip(blocked, labels) if l == 'adult']
    
    cr = sum(1 for b, _ in children if b) / len(children) * 100 if children else 0
    tr = sum(1 for b, _ in teens if b) / len(teens) * 100 if teens else 0
    afpr = sum(1 for b, _ in adults if b) / len(adults) * 100 if adults else 0
    
    print(f"  317-session eval (thr={threshold:.2f} OR minor>=0.72):")
    print(f"    Child recall: {sum(1 for b,_ in children if b)}/{len(children)} = {cr:.1f}% (target ≥98%) {'✓' if cr>=98 else '✗'}")
    print(f"    Teen recall:  {sum(1 for b,_ in teens if b)}/{len(teens)} = {tr:.1f}% (target ≥80%) {'✓' if tr>=80 else '✗'}")
    print(f"    Adult FPR:    {sum(1 for b,_ in adults if b)}/{len(adults)} = {afpr:.1f}% (target ≤20%) {'✓' if afpr<=20 else '✗'}")
    
    return cr, tr, afpr


# ── JS export ─────────────────────────────────────────────────────────────────

def export_compact_js(model, feature_names, out_path: Path, threshold: float, n_samples=0, cv_auc=0):
    today = datetime.date.today().isoformat()
    model_str = model.model_to_string()

    trees = []
    tree_blocks = re.split(r'Tree=\d+', model_str)
    for block in tree_blocks[1:]:
        lv_match = re.search(r'leaf_value=([^\n]+)', block)
        nf_match = re.search(r'split_feature=([^\n]+)', block)
        nt_match = re.search(r'threshold=([^\n]+)', block)
        lc_match = re.search(r'left_child=([^\n]+)', block)
        rc_match = re.search(r'right_child=([^\n]+)', block)
        if not (lv_match and nf_match):
            continue
        leaf_values = [float(x) for x in lv_match.group(1).split()]
        split_features = [int(x) for x in nf_match.group(1).split()]
        thresholds = [float(x) for x in nt_match.group(1).split()]
        left_children = [int(x) for x in lc_match.group(1).split()]
        right_children = [int(x) for x in rc_match.group(1).split()]
        num_internal = len(split_features)
        all_children = set(left_children + right_children)
        root = next(ni for ni in range(num_internal) if ni not in all_children)
        splits = [[split_features[i], thresholds[i]] for i in range(num_internal)]
        children = [[left_children[i], right_children[i]] for i in range(num_internal)]
        trees.append({'r': root, 's': splits, 'c': children, 'l': leaf_values})

    feats_js = json.dumps(feature_names)
    trees_js = json.dumps(trees, separators=(',', ':'))
    thr_str = f"{threshold:.2f}"

    js = f"""// lgbm_evaluate_v7 — LGBM Underage Scorer V7
// Model: 150 trees, 15 leaves, CV AUC={cv_auc:.3f}, trained {today} on {n_samples} samples
// V7: added 317-session hard FP/FN examples (2026-05-20 session)
// Blocking rule in Piper: lgbm_score >= {thr_str} OR minor >= 0.72
// Input: `labels` object from siglip2 underage classifier
// Outputs: score 0-1 (underage probability)

const LGBM_FEATURES = {feats_js};

const LGBM_TREES = {trees_js};

function lgbm_predict_v7(vec) {{
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

function lgbm_evaluate_v7(labelsObj) {{
  const vec = LGBM_FEATURES.map(f => {{
    if (f.startsWith('adult__')) {{
      const key = f.slice(7);
      return (labelsObj.adult && labelsObj.adult[key]) || 0;
    }}
    return (labelsObj.underage && labelsObj.underage[f]) || 0;
  }});
  return lgbm_predict_v7(vec);
}}
"""
    out_path.write_text(js, encoding='utf-8')
    print(f"  Exported: {out_path} ({len(feature_names)} features, {len(trees)} trees)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LGBM V7 Training")
    print("=" * 60)

    # 1. Load all data
    print("\n[1/5] Loading training data...")
    ls_items, ls_ids = load_ls_all()
    grafana_items = load_grafana_all(exclude_ids=ls_ids)
    v6_fn = load_v6_fn_hard_positives()
    existing_negs = load_existing_negatives()
    hard_pos_317, hard_neg_317 = load_317_hard_examples()

    # Combine
    # Deduplicate: grafana_items already excludes ls_ids
    # Treat 317-session FP/FN as separate (may already be in grafana_items — that's OK, will just appear twice with higher weight)
    all_items = ls_items + grafana_items + v6_fn + existing_negs
    total_before = len(all_items)
    print(f"\n  Base dataset: {total_before} items")

    # Hard examples get weight=3x (forcing model to learn them)
    base_weights = np.ones(total_before, dtype=np.float32)

    # Add 317 hard examples with high weight
    hard_items_317 = hard_pos_317 + hard_neg_317
    hard_weight_317 = np.full(len(hard_items_317), 3.0, dtype=np.float32)

    all_items = all_items + hard_items_317
    weights = np.concatenate([base_weights, hard_weight_317])

    print(f"  Total with 317-hard examples (3x weight): {len(all_items)}")

    # 2. Build feature matrix
    print("\n[2/5] Building feature matrix...")
    X, y, feature_names = build_feature_matrix(all_items)
    print(f"  Shape: {X.shape}  positives={y.sum()}  negatives={(1-y).sum()}")
    print(f"  Features: {len(feature_names)}")

    # 3. Cross-validation
    print("\n[3/5] Cross-validation (5-fold)...")
    aucs = cross_validate(X, y, weights)
    cv_auc = np.mean(aucs)

    # 4. Full model training
    print("\n[4/5] Training final model on full data...")
    model = train_lgbm(X, y, weights)

    # Threshold selection — evaluate on 317-session
    print("\n[4b] Threshold sweep on 317-session...")
    conn = _open_db()
    rows317 = conn.execute("""
        SELECT id, label, piper_result
        FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND export_batch = '2026-05-20 UTC'
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
    """).fetchall()
    conn.close()
    
    items317 = []
    for r in rows317:
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            if not det:
                continue
            items317.append(extract_item(r['label'], det, r['id']))
        except:
            pass
    
    X317, y317, _ = build_feature_matrix(items317, feature_names)
    preds317 = model.predict(X317)
    minors317 = np.array([it.get('minor', 0) for it in items317])
    labels317 = [it['label'] for it in items317]
    
    print(f"\n  {'thr':>6} | {'child_r':>8} | {'teen_r':>8} | {'adult_fpr':>10} | OK?")
    print("  " + "-" * 50)
    best_thr = 0.55
    for thr in [0.40, 0.45, 0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.65]:
        blocked = (preds317 >= thr) | (minors317 >= 0.72)
        children = [(b, l) for b, l in zip(blocked, labels317) if l == 'child']
        teens = [(b, l) for b, l in zip(blocked, labels317) if l == 'teen']
        adults = [(b, l) for b, l in zip(blocked, labels317) if l == 'adult']
        cr = sum(1 for b, _ in children if b) / len(children) * 100 if children else 0
        tr = sum(1 for b, _ in teens if b) / len(teens) * 100 if teens else 0
        afpr = sum(1 for b, _ in adults if b) / len(adults) * 100 if adults else 0
        ok = 'ALL ✓' if cr>=98 and tr>=80 and afpr<=20 else ('child✓' if cr>=98 else '') + (' teen✓' if tr>=80 else '') + (' fpr✓' if afpr<=20 else '')
        print(f"  {thr:>6.2f} | {cr:>7.1f}% | {tr:>7.1f}% | {afpr:>9.1f}% | {ok}")
        if cr >= 98 and tr >= 80 and afpr <= 20:
            best_thr = thr
            break
    
    # Pick best threshold (highest recall that keeps FPR ≤20%)
    best_thr_overall = None
    for thr in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.60]:
        blocked = (preds317 >= thr) | (minors317 >= 0.72)
        adults = [(b, l) for b, l in zip(blocked, labels317) if l == 'adult']
        afpr = sum(1 for b, _ in adults if b) / len(adults) * 100 if adults else 0
        if afpr <= 20:
            best_thr_overall = thr
            break
    
    chosen_thr = best_thr_overall if best_thr_overall else 0.55
    print(f"\n  Chosen threshold: {chosen_thr:.2f} (lowest thr with adult FPR ≤20%)")

    # 5. Final eval and export
    print(f"\n[5/5] Final eval and export...")
    cr, tr, afpr = eval_317_session(model, feature_names, chosen_thr)

    data_dir = BASE_DIR / 'data'
    model.save_model(str(data_dir / 'lgbm_underage_v7.txt'))
    print(f"  Saved: data/lgbm_underage_v7.txt")

    export_compact_js(
        model, feature_names,
        data_dir / 'lgbm_evaluate_v7.js',
        threshold=chosen_thr,
        n_samples=len(all_items),
        cv_auc=cv_auc
    )

    # Save threshold metadata
    meta = {
        'version': 'v7',
        'trained_at': datetime.datetime.utcnow().isoformat(),
        'cv_auc': cv_auc,
        'threshold': chosen_thr,
        'n_samples': len(all_items),
        'n_features': len(feature_names),
        'eval_317_child_recall': cr,
        'eval_317_teen_recall': tr,
        'eval_317_adult_fpr': afpr,
        'training_data': {
            'ls_items': len(ls_items),
            'grafana_pool': len(grafana_items),
            'v6_fn_hard_positives': len(v6_fn),
            'existing_negatives': len(existing_negs),
            'hard_pos_317': len(hard_pos_317),
            'hard_neg_317': len(hard_neg_317),
        }
    }
    (data_dir / 'lgbm_v7_meta.json').write_text(json.dumps(meta, indent=2))
    print(f"\nV7 Summary:")
    print(f"  CV AUC: {cv_auc:.4f}")
    print(f"  Threshold: {chosen_thr:.2f}")
    print(f"  317-session: child={cr:.1f}% teen={tr:.1f}% adult_fpr={afpr:.1f}%")


if __name__ == '__main__':
    main()
