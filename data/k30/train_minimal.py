"""Train the MINIMAL CSAM-only model.

The key difference from train_pruned_hardneg.py: aggregates are computed from
ONLY the kept SigLIP labels (not the full set). This matches what will happen
at inference when we cut the SigLIP payload — SigLIP only returns scores for
the labels we ask about, so the JS aggregates will only include those.

Two-stage training:
  1. First pass on full label set to get importance ranking.
  2. Pick top N SigLIP labels (default 30).
  3. Rebuild features computing aggregates from ONLY those labels.
  4. Retrain on the cut-down feature matrix.
  5. Apply hardneg w=20 weighting on adult FPs at thr 0.05.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from label_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS
from train_for_piper import siglip_labels_raw, combine_scores, load_rows, build_features, parse_lgbm_dump_trees

ROOT = Path(__file__).resolve().parent

# How many top-importance SigLIP labels to retain
TOP_N_SIGLIP = 30
HARDNEG_WEIGHT = 20.0
# Pass `--n N` to override; useful for sweeping.
if "--n" in sys.argv:
    TOP_N_SIGLIP = int(sys.argv[sys.argv.index("--n") + 1])


def rebuild_with_subset(rows, keep_siglip: set[str]):
    """Recompute aggregates using ONLY the labels in keep_siglip. This mirrors
    what the JS will compute at inference when the SigLIP payload is pruned."""
    all_features = sorted(keep_siglip) + ["_minor", "_adult", "_confidence", "_child_body", "_child_context", "_child_interaction", "_body_vs_context"]
    feat_idx = {n: i for i, n in enumerate(all_features)}
    X = np.zeros((len(rows), len(all_features)), dtype=np.float32)
    for i, (gid, sg, prompt, cp, tp, yy, s) in enumerate(rows):
        # Subset the label dict to the kept ones
        sg_kept = {k: v for k, v in sg.items() if k in keep_siglip}
        for fn, val in sg_kept.items():
            X[i, feat_idx[fn]] = float(val)
        minor = combine_scores(sg_kept, lambda k: not k.startswith("adult__"))
        adult = combine_scores(sg_kept, lambda k: k.startswith("adult__"))
        body  = combine_scores(sg_kept, lambda k: k in BODY_LABELS)
        ctx   = combine_scores(sg_kept, lambda k: k in CONTEXT_LABELS)
        inter = combine_scores(sg_kept, lambda k: k in INTERACTION_LABELS)
        total = minor + adult; bc = body + ctx
        X[i, feat_idx["_minor"]]              = minor
        X[i, feat_idx["_adult"]]              = adult
        X[i, feat_idx["_confidence"]]         = (minor / total) if total > 0 else 0
        X[i, feat_idx["_child_body"]]         = body
        X[i, feat_idx["_child_context"]]      = ctx
        X[i, feat_idx["_child_interaction"]]  = inter
        X[i, feat_idx["_body_vs_context"]]    = (body / bc) if bc > 0 else 0
    return X, all_features


def cv(X, y, sw):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1,
                                min_child_samples=20, reg_lambda=1.0,
                                n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(X[tr], y[tr], sample_weight=sw[tr] if sw is not None else None)
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def main():
    rows = load_rows()
    X_full, y, src, full_features = build_features(rows)
    n = len(rows); n_pos = int(y.sum())
    print(f"n={n} pos={n_pos}", file=sys.stderr)

    sc = sqlite3.connect(ROOT / "scored.sqlite"); sc.row_factory = sqlite3.Row
    bn = sqlite3.connect(ROOT / "scored_benign.sqlite"); bn.row_factory = sqlite3.Row
    min_ages = []
    for db in (sc, bn):
        for r in db.execute("SELECT qwen3_min_age FROM scored WHERE error IS NULL AND raw_siglip IS NOT NULL"):
            min_ages.append(r["qwen3_min_age"])
    adult_mask = np.array([a is not None and a >= 18 for a in min_ages])
    pos_mask = (y == 1)

    # Step 1: get importance ranking from a full-feature baseline
    m0 = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1,
                            min_child_samples=20, reg_lambda=1.0,
                            n_jobs=-1, verbosity=-1, random_state=42)
    m0.fit(X_full, y)
    imp = m0.feature_importances_
    siglip_feats = [(f, imp[i]) for i, f in enumerate(full_features) if not f.startswith("_")]
    siglip_feats.sort(key=lambda x: -x[1])
    keep_siglip = set(f for f, _ in siglip_feats[:TOP_N_SIGLIP])
    print(f"\nKeeping top {TOP_N_SIGLIP} SigLIP labels (by importance from baseline)", file=sys.stderr)

    # Step 2: rebuild X with aggregates from ONLY kept labels
    X_min, min_features = rebuild_with_subset(rows, keep_siglip)
    print(f"Minimal feature set: {len(min_features)} ({TOP_N_SIGLIP} SigLIP + 7 derived)", file=sys.stderr)

    # Step 3: find hardnegs on the minimal feature set
    oof_base = cv(X_min, y, None)
    hn = adult_mask & (oof_base >= 0.05)
    print(f"hardnegs (adults flagged at thr 0.05 by baseline minimal model): {hn.sum()}", file=sys.stderr)
    sw = np.ones(n, dtype=np.float32); sw[hn] = HARDNEG_WEIGHT

    # Step 4: retrain with hardneg weighting
    oof = cv(X_min, y, sw)
    auc = roc_auc_score(y[pos_mask | adult_mask], oof[pos_mask | adult_mask])
    print(f"\n=== Minimal model: top {TOP_N_SIGLIP} SigLIP + 7 derived, hardneg w={HARDNEG_WEIGHT} ===")
    print(f"AUC vs adults: {auc:.4f}")
    print(f"  {'thr':>6} {'recall':>8} {'aFP':>8} {'aFP_cnt':>9}")
    for thr in [0.20, 0.10, 0.05]:
        rec = (oof[pos_mask] >= thr).sum() / pos_mask.sum()
        afp = (oof[adult_mask] >= thr).sum() / adult_mask.sum()
        afp_c = int((oof[adult_mask] >= thr).sum())
        print(f"  {thr:>6.3f} {rec:>7.1%} {afp:>7.2%} {afp_c:>8d}")

    # Final fit + dump
    final = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1,
                                min_child_samples=20, reg_lambda=1.0,
                                n_jobs=-1, verbosity=-1, random_state=42)
    final.fit(X_min, y, sample_weight=sw)
    trees = parse_lgbm_dump_trees(final.booster_)

    out = {
        "features": min_features,
        "trees": trees,
        "suggested_threshold": 0.10,
        "trained_on": {
            "n_rows": n, "n_pos": n_pos, "n_neg": n - n_pos,
            "auc_vs_adults": float(auc),
            "n_hardnegs_upweighted": int(hn.sum()),
            "hardneg_weight": HARDNEG_WEIGHT,
            "n_siglip_labels": TOP_N_SIGLIP,
            "comment": "MINIMAL CSAM-only: aggregates computed from ONLY the kept SigLIP labels",
        },
        "hyperparams": {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20, "reg_lambda": 1.0},
    }
    (ROOT / "piper_lgbm_model.json").write_text(json.dumps(out))
    print(f"\nwrote piper_lgbm_model.json ({(ROOT/'piper_lgbm_model.json').stat().st_size} bytes)")
    print(f"  features: {len(min_features)}  (SigLIP labels: {TOP_N_SIGLIP})")
    (ROOT / "feature_keepers.json").write_text(json.dumps({
        "siglip_keepers": sorted(keep_siglip),
        "derived_keepers": ["_minor", "_adult", "_confidence", "_child_body", "_child_context", "_child_interaction", "_body_vs_context"],
    }, indent=2))


if __name__ == "__main__":
    main()
