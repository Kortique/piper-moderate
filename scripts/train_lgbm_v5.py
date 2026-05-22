"""
train_lgbm_v5.py
----------------
Retrain LGBM underage classifier (V5) incorporating 69 new FP images
with updated tag features, plus existing LS training data.

Output:
  data/lgbm_underage_v5.txt     — LightGBM model file
  data/lgbm_evaluate_v5.js      — JS evaluator for Piper

Usage:
    pip install lightgbm scikit-learn pandas --break-system-packages
    python scripts/train_lgbm_v5.py
"""
import sys, os, json, re, datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data loading ──────────────────────────────────────────────────────────────

def load_ls_positives():
    """Load LS (Label Studio) minor images from DB."""
    import sqlite3
    conn = sqlite3.connect(BASE_DIR / "gallery.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT task_id, age_from, siglip2_details
        FROM ls_images
        WHERE siglip2_details IS NOT NULL
          AND age_from IS NOT NULL
    """).fetchall()
    conn.close()

    items = []
    for r in rows:
        det = json.loads(r["siglip2_details"]).get("underage", {})
        lgbm_o = det.get("lgbm") or {}
        lgbm_score = lgbm_o.get("score", 0)
        minor = det.get("minor", 0)
        adult = det.get("adult", 0)

        labels = det.get("labels", {})
        u_labels = labels.get("underage") or {}
        a_labels = labels.get("adult") or {}

        age = r["age_from"]
        label = "child" if age <= 14 else "teen" if age <= 17 else "adult"

        items.append({
            "id": f"ls_{r['task_id']}",
            "label": label,
            "minor": minor,
            "adult": adult,
            "lgbm_score": lgbm_score,
            "underage_labels": u_labels,
            "adult_labels": a_labels,
        })
    return items


def load_negatives():
    """Load adult FP negative training examples.

    Priority: fp69_features_new_tags.json has NEW siglip2 scores for the 69 FP images.
    These override any old-score entries in lgbm_retraining_negatives.json.
    """
    # Load FP69 with NEW scores first (priority)
    fp69_ids = set(l.strip() for l in (BASE_DIR / "fp_69_ids.txt").read_text().splitlines() if l.strip())

    items = []
    seen_ids = set()

    # 1. Load FP69 new features (highest priority — new tag scores)
    fp69_path = BASE_DIR / "data" / "fp69_features_new_tags.json"
    if fp69_path.exists():
        data = json.load(open(fp69_path))
        for r in data:
            gid = r.get("id", "")
            items.append({
                "id": gid,
                "label": "adult",
                "minor": r.get("minor_score", 0),
                "adult": r.get("adult_score", 0),
                "lgbm_score": r.get("lgbm_score", 0),
                "underage_labels": r.get("underage_labels") or {},
                "adult_labels": r.get("adult_labels") or {},
            })
            seen_ids.add(gid)
        print(f"  Loaded {len(data)} FP69 negatives (new tags) from fp69_features_new_tags.json")

    # 2. Load other negatives, skipping FP69 duplicates
    for fname in ["lgbm_retraining_negatives.json", "lgbm_new_negatives.json"]:
        path = BASE_DIR / "data" / fname
        if not path.exists():
            continue
        data = json.load(open(path))
        added = 0
        for r in data:
            gid = r.get("id", "")
            if gid in seen_ids:
                continue  # Skip duplicates (already loaded with new scores)
            items.append({
                "id": gid,
                "label": "adult",
                "minor": r.get("minor_score") or r.get("minor", 0),
                "adult": r.get("adult_score") or r.get("adult", 0),
                "lgbm_score": r.get("lgbm_score", 0),
                "underage_labels": r.get("underage_labels") or {},
                "adult_labels": r.get("adult_labels") or {},
            })
            seen_ids.add(gid)
            added += 1
        print(f"  Loaded {added} negatives (deduped) from {fname}")
    return items


# ── Feature engineering ───────────────────────────────────────────────────────

def build_feature_matrix(items, feature_names=None):
    """
    Convert list of items to (X, y) for LGBM training.
    Features: underage tag scores + adult tag scores (prefixed with 'adult__').
    """
    # Collect all feature names if not provided
    if feature_names is None:
        all_feats = set()
        for item in items:
            all_feats.update(item.get("underage_labels", {}).keys())
            for k in item.get("adult_labels", {}).keys():
                all_feats.add(f"adult__{k}")
        feature_names = sorted(all_feats)

    X = np.zeros((len(items), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int32)
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    for i, item in enumerate(items):
        # Underage features
        for k, v in item.get("underage_labels", {}).items():
            if k in feat_idx:
                X[i, feat_idx[k]] = float(v)
        # Adult counter-features
        for k, v in item.get("adult_labels", {}).items():
            fk = f"adult__{k}"
            if fk in feat_idx:
                X[i, feat_idx[fk]] = float(v)
        # Label: 1 = minor (child/teen), 0 = adult
        y[i] = 1 if item["label"] in ("child", "teen") else 0

    return X, y, feature_names


# ── Training ──────────────────────────────────────────────────────────────────

def train_lgbm(X_train, y_train, X_test=None, y_test=None):
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 15,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "is_unbalance": True,
    }

    dataset = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dataset, num_boost_round=100)

    if X_test is not None:
        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, pred)
        print(f"  CV AUC: {auc:.4f}")

    return model


def cross_validate(X, y, n_splits=5):
    """5-fold stratified CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = train_lgbm(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        pred = model.predict(X[val_idx])
        auc = roc_auc_score(y[val_idx], pred)
        aucs.append(auc)
        print(f"  Fold {fold+1}: AUC={auc:.4f}")
    return aucs


# ── Model export ──────────────────────────────────────────────────────────────

def export_js(model, feature_names, out_path: Path, version="v5", n_samples=0, cv_auc=0):
    """Export LightGBM model as JS evaluator compatible with Piper."""
    today = datetime.date.today().isoformat()
    model_str = model.model_to_string()

    # Build JS feature list
    feats_js = json.dumps(feature_names, ensure_ascii=False)

    # Build model text
    js_lines = [
        f"// lgbm_evaluate — LGBM Underage Scorer",
        f"// Model: 100 trees, 15 leaves, CV AUC={cv_auc:.3f}, trained {today} on {n_samples} samples",
        f"// Input: same `labels` flat object as evaluate_siglip receives",
        f"// Outputs: labels (array), details (json) — same pattern as evaluate_siglip",
        "",
        f"// ── Embedded LGBM model ({len(feature_names)} features, 100 trees) ──────────────",
        f"const LGBM_FEATURES = {feats_js};",
        "",
    ]

    # Embed model
    js_lines.append("const LGBM_MODEL = `")
    js_lines.append(model_str.replace("`", "\\`"))
    js_lines.append("`;")
    js_lines.append("")

    # Add evaluator function (same interface as v4)
    js_lines.extend([
        "function lgbm_evaluate(labelsObj) {",
        "  // Build feature vector from tag scores",
        "  const vec = LGBM_FEATURES.map(f => {",
        "    if (f.startsWith('adult__')) {",
        "      const key = f.slice(7);",
        "      return (labelsObj.adult && labelsObj.adult[key]) || 0;",
        "    }",
        "    return (labelsObj.underage && labelsObj.underage[f]) || 0;",
        "  });",
        "  // Run LGBM (inline)",
        "  // Returns score 0-1",
        "  return lgbm_predict(vec, LGBM_MODEL, LGBM_FEATURES);",
        "}",
    ])

    out_path.write_text("\n".join(js_lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading training data...")

    print("  Loading LS positives from DB...")
    positives = load_ls_positives()
    pos_minor = [p for p in positives if p["label"] in ("child", "teen")]
    pos_adult = [p for p in positives if p["label"] == "adult"]
    print(f"  LS positives: {len(pos_minor)} minor, {len(pos_adult)} adult (LS adults excluded from negatives)")

    print("  Loading negatives...")
    negatives = load_negatives()

    # Use LS adults with lgbm_score < 0.40 as clean negatives too
    clean_ls_adults = [p for p in pos_adult if p["lgbm_score"] < 0.40]
    print(f"  LS clean adults (lgbm<0.40): {len(clean_ls_adults)}")

    all_items = pos_minor + negatives + clean_ls_adults
    print(f"\nTotal training samples: {len(all_items)}")
    print(f"  Minor: {sum(1 for x in all_items if x['label'] in ('child','teen'))}")
    print(f"  Adult: {sum(1 for x in all_items if x['label'] == 'adult')}")

    print("\nBuilding feature matrix...")
    X, y, feature_names = build_feature_matrix(all_items)
    print(f"  Features: {len(feature_names)}")
    print(f"  Shape: {X.shape}")
    print(f"  Class balance: {y.mean():.3f} positive rate")

    print("\nCross-validation (5-fold)...")
    aucs = cross_validate(X, y)
    mean_auc = np.mean(aucs)
    print(f"  Mean AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")

    print("\nTraining final model on full dataset...")
    final_model = train_lgbm(X, y)

    # Find optimal threshold
    preds = final_model.predict(X)
    print("\nThreshold analysis:")
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
        blocked = preds >= thresh
        tp = ((blocked == 1) & (y == 1)).sum()
        fp = ((blocked == 1) & (y == 0)).sum()
        fn = ((blocked == 0) & (y == 1)).sum()
        tn = ((blocked == 0) & (y == 0)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"  thresh={thresh:.2f}: recall={recall:.3f} fpr={fpr:.3f} TP={tp} FP={fp} FN={fn} TN={tn}")

    # Check FP69 specific
    fp69_items = [item for item in all_items if item["id"] in
                  [l.strip() for l in (BASE_DIR / "fp_69_ids.txt").read_text().splitlines()]]
    if fp69_items:
        fp69_X, fp69_y, _ = build_feature_matrix(fp69_items, feature_names)
        fp69_preds = final_model.predict(fp69_X)
        print(f"\nFP69 results with new model (thresh=0.80):")
        fp69_passed = (fp69_preds < 0.80).sum()
        print(f"  Passed: {fp69_passed}/69 ({fp69_passed/69*100:.1f}%)")
        print(f"  LGBM scores: min={fp69_preds.min():.4f} max={fp69_preds.max():.4f} med={np.median(fp69_preds):.4f}")

    # Save model
    out_dir = BASE_DIR / "data"
    model_path = out_dir / "lgbm_underage_v5.txt"
    final_model.save_model(str(model_path))
    print(f"\nSaved model: {model_path}")

    # Export JS
    js_path = out_dir / "lgbm_evaluate_v5.js"
    export_js(final_model, feature_names, js_path,
              version="v5", n_samples=len(all_items), cv_auc=mean_auc)
    print(f"Saved JS: {js_path}")

    print("\nDone! Next: deploy lgbm_underage_v5.txt to Piper (via n8n or manual upload).")


if __name__ == "__main__":
    main()
