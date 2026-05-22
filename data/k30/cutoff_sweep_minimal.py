"""Sweep cutoff in {13, 14, 15} with the CURRENT minimal model setup:
- top 30 SigLIP labels by importance (re-ranked per cutoff)
- hardneg weight 20× on adult FPs at thr 0.05
- aggregates computed from ONLY the kept SigLIP labels (proper train-inference parity)
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
from train_for_piper import siglip_labels_raw, combine_scores, load_rows, build_features

ROOT = Path(__file__).resolve().parent
TOP_N_SIGLIP = 30
HARDNEG_WEIGHT = 20.0


def rebuild_with_subset(rows, keep_siglip):
    all_features = sorted(keep_siglip) + ["_minor", "_adult", "_confidence", "_child_body", "_child_context", "_child_interaction", "_body_vs_context"]
    feat_idx = {n: i for i, n in enumerate(all_features)}
    X = np.zeros((len(rows), len(all_features)), dtype=np.float32)
    for i, (gid, sg, prompt, cp, tp, yy, s) in enumerate(rows):
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
    return X


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
    X_full, _, src, full_features = build_features(rows)

    sc = sqlite3.connect(ROOT / "scored.sqlite"); sc.row_factory = sqlite3.Row
    bn = sqlite3.connect(ROOT / "scored_benign.sqlite"); bn.row_factory = sqlite3.Row
    min_ages = []
    for db in (sc, bn):
        for r in db.execute("SELECT qwen3_min_age FROM scored WHERE error IS NULL AND raw_siglip IS NOT NULL"):
            min_ages.append(r["qwen3_min_age"])
    min_ages = np.array([a if a is not None else -1 for a in min_ages])
    adult_mask = min_ages >= 18

    print(f"\n{'cutoff':>6} {'n_pos':>6} {'thr':>6} {'recall':>8} {'aFP':>8} {'aFP_cnt':>8}")
    print("-" * 60)
    for CUTOFF in [13, 14, 15]:
        y = (min_ages <= CUTOFF) & (min_ages >= 0)
        y = y.astype(np.int8)
        n_pos = int(y.sum())

        # Re-rank features per cutoff
        m0 = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1,
                                min_child_samples=20, reg_lambda=1.0,
                                n_jobs=-1, verbosity=-1, random_state=42)
        m0.fit(X_full, y)
        imp = m0.feature_importances_
        siglip_feats = [(f, imp[i]) for i, f in enumerate(full_features) if not f.startswith("_")]
        siglip_feats.sort(key=lambda x: -x[1])
        keep_siglip = set(f for f, _ in siglip_feats[:TOP_N_SIGLIP])

        # Rebuild + hardneg + retrain
        X_min = rebuild_with_subset(rows, keep_siglip)
        oof_base = cv(X_min, y, None)
        hn = adult_mask & (oof_base >= 0.05)
        sw = np.ones(len(y), dtype=np.float32); sw[hn] = HARDNEG_WEIGHT
        oof = cv(X_min, y, sw)

        pos_mask = y == 1
        for thr in [0.20, 0.10, 0.05]:
            rec = (oof[pos_mask] >= thr).sum() / pos_mask.sum()
            afp = (oof[adult_mask] >= thr).sum() / adult_mask.sum()
            afp_c = int((oof[adult_mask] >= thr).sum())
            print(f"  ≤{CUTOFF:>2}   {n_pos:>5}  {thr:>5.2f}  {rec:>6.1%}  {afp:>6.2%}  {afp_c:>6d}")
        print()


if __name__ == "__main__":
    main()
