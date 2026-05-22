"""Honest apples-to-apples comparison:
 - A. Production (detect.ts noisy-OR: ratio≥0.85 AND risk≥0.01)
 - B. K=25 (existing piper LGBM in violations-detector-test-v3)
 - C. K=30 (BCI) — current deployed minimal model (5-fold OOF)

All three scored on the SAME dataset:
 - 6,680 Qwen3-VL-scored images
 - positive = qwen3_min_age ≤ 14 (current production cutoff)
 - same denominator for FPR (all negatives, not just adults)

Outputs metrics + breakdown by Qwen3 age band so we can see WHERE each model fires.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from label_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS
from train_for_piper import siglip_labels_raw, combine_scores, predict_compact, load_rows, build_features

ROOT = Path(__file__).resolve().parent
CUTOFF = 14


def parse_existing_lgbm():
    d = json.load(open(ROOT / "piper_pipeline_v3.json"))
    script = d["pipeline"]["nodes"]["lgbm_evaluate"]["script"]
    feats = json.loads(re.search(r"const LGBM_FEATURES = (\[[^\n]+\]);", script).group(1))
    start = script.find("const LGBM_TREES = [") + len("const LGBM_TREES = ")
    depth = 0; in_str = False; esc = False
    for i in range(start, len(script)):
        ch = script[i]
        if esc: esc = False; continue
        if in_str:
            if ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True; continue
        if ch == '[': depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0: end = i + 1; break
    trees = json.loads(script[start:end])
    thr = float(re.search(r"LGBM_THRESHOLD = ([\d\.]+)", script).group(1))
    return feats, trees, thr


def detect_ts_score(raw_siglip: str) -> tuple[float, bool]:
    """Mirror detect.ts. Returns (continuous score, block decision)."""
    if not raw_siglip:
        return 0.0, False
    sl = json.loads(raw_siglip)
    under = (((sl.get("underage") or {}).get("labels") or {}).get("underage")) or {}
    adult_l = (((sl.get("underage") or {}).get("labels") or {}).get("adult")) or {}
    def apply_mult(d):
        out = {}
        for k, v in d.items():
            m = re.search(r":x(\d+(?:\.\d+)?)$", k)
            if m:
                out[k[:m.start()]] = min(float(v) * float(m.group(1)), 0.999)
            else:
                out[k] = float(v)
        return out
    risk = combine_scores(apply_mult(under), lambda k: True)
    adult = combine_scores(apply_mult(adult_l), lambda k: True)
    total = risk + adult
    ratio = (risk / total) if total > 0 else 0.0
    score = ratio if risk >= 0.01 else 0.0
    blocked = (ratio >= 0.85) and (risk >= 0.01)
    return score, blocked


def k25_vec(raw_siglip: str, features: list[str]) -> np.ndarray:
    vec = np.zeros(len(features), dtype=np.float32)
    fi = {n: i for i, n in enumerate(features)}
    if not raw_siglip: return vec
    sl = json.loads(raw_siglip)
    under = (((sl.get("underage") or {}).get("labels") or {}).get("underage")) or {}
    adult_l = (((sl.get("underage") or {}).get("labels") or {}).get("adult")) or {}
    # Existing model: feature names preserve :xN, values are RAW (un-multiplied).
    # Our storage has POST-multiplier values, so divide by N to recover raw.
    for k, v in under.items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        raw_v = float(v) / float(m.group(1)) if m else float(v)
        if k in fi: vec[fi[k]] = raw_v
    for k, v in adult_l.items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        raw_v = float(v) / float(m.group(1)) if m else float(v)
        fn = "adult__" + k
        if fn in fi: vec[fi[fn]] = raw_v
    # Derived for K=25: _minor, _adult, _confidence (no multiplier)
    raw_under = {k: (float(v) / float(re.search(r":x(\d+(?:\.\d+)?)$", k).group(1)) if ":x" in k else float(v)) for k, v in under.items()}
    raw_adult = {k: (float(v) / float(re.search(r":x(\d+(?:\.\d+)?)$", k).group(1)) if ":x" in k else float(v)) for k, v in adult_l.items()}
    minor = combine_scores(raw_under, lambda k: True)
    adult = combine_scores(raw_adult, lambda k: True)
    total = minor + adult
    conf = (minor / total) if total > 0 else 0.0
    if "_minor" in fi:      vec[fi["_minor"]] = minor
    if "_adult" in fi:      vec[fi["_adult"]] = adult
    if "_confidence" in fi: vec[fi["_confidence"]] = conf
    return vec


def main():
    # Load data
    rows = []
    min_ages = []
    raws = []
    for db_path, src in [("scored.sqlite", "cand"), ("scored_benign.sqlite", "benign")]:
        c = sqlite3.connect(ROOT / db_path); c.row_factory = sqlite3.Row
        for r in c.execute("SELECT generation_id, qwen3_min_age, raw_siglip FROM scored WHERE error IS NULL AND raw_siglip IS NOT NULL"):
            rows.append(r["generation_id"])
            min_ages.append(r["qwen3_min_age"])
            raws.append(r["raw_siglip"])
    min_ages = np.array([a if a is not None else -1 for a in min_ages])
    y = ((min_ages >= 0) & (min_ages <= CUTOFF)).astype(np.int8)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    print(f"n_rows={len(rows)} pos (min_age ≤ {CUTOFF})={n_pos} neg={n_neg}", file=sys.stderr)

    # === A. Production ===
    print("scoring A (production detect.ts)...", file=sys.stderr)
    score_A = np.zeros(len(rows), dtype=np.float32)
    A_blocked = np.zeros(len(rows), dtype=bool)
    for i, raw in enumerate(raws):
        s, b = detect_ts_score(raw)
        score_A[i] = s
        A_blocked[i] = b

    # === B. K=25 (existing piper LGBM) ===
    print("scoring B (K=25)...", file=sys.stderr)
    ex_feats, ex_trees, ex_thr = parse_existing_lgbm()
    score_B = np.zeros(len(rows), dtype=np.float32)
    for i, raw in enumerate(raws):
        vec = k25_vec(raw, ex_feats)
        score_B[i] = predict_compact(ex_trees, vec)

    # === C. K=30 (BCI), 5-fold OOF on this data ===
    print("scoring C (K=30 BCI) with 5-fold OOF...", file=sys.stderr)
    # Build features the same way train_minimal.py does
    rows_full = load_rows()
    X_full, _, src, full_features = build_features(rows_full)
    # Get importance ranking with our cutoff labeling
    y_train = y  # same labels as everywhere
    m0 = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1, min_child_samples=20, reg_lambda=1.0, n_jobs=-1, verbosity=-1, random_state=42)
    m0.fit(X_full, y_train)
    imp = m0.feature_importances_
    siglip_feats = [(f, imp[i]) for i, f in enumerate(full_features) if not f.startswith("_")]
    siglip_feats.sort(key=lambda x: -x[1])
    keep_siglip = set(f for f, _ in siglip_feats[:30])
    # Rebuild X with aggregates from kept labels only
    all_features = sorted(keep_siglip) + ["_minor","_adult","_confidence","_child_body","_child_context","_child_interaction","_body_vs_context"]
    fi = {n: i for i, n in enumerate(all_features)}
    X_min = np.zeros((len(rows_full), len(all_features)), dtype=np.float32)
    for i, (gid, sg, prompt, cp, tp, yy, s) in enumerate(rows_full):
        sg_kept = {k: v for k, v in sg.items() if k in keep_siglip}
        for fn, val in sg_kept.items():
            X_min[i, fi[fn]] = float(val)
        minor = combine_scores(sg_kept, lambda k: not k.startswith("adult__"))
        adult = combine_scores(sg_kept, lambda k: k.startswith("adult__"))
        body  = combine_scores(sg_kept, lambda k: k in BODY_LABELS)
        ctx   = combine_scores(sg_kept, lambda k: k in CONTEXT_LABELS)
        inter = combine_scores(sg_kept, lambda k: k in INTERACTION_LABELS)
        total = minor + adult; bc = body + ctx
        X_min[i, fi["_minor"]] = minor
        X_min[i, fi["_adult"]] = adult
        X_min[i, fi["_confidence"]] = (minor / total) if total > 0 else 0
        X_min[i, fi["_child_body"]] = body
        X_min[i, fi["_child_context"]] = ctx
        X_min[i, fi["_child_interaction"]] = inter
        X_min[i, fi["_body_vs_context"]] = (body / bc) if bc > 0 else 0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    adult_mask_arr = min_ages >= 18
    # baseline for hardneg detection
    oof_base = np.zeros(len(y_train), dtype=np.float32)
    for tr, va in skf.split(X_min, y_train):
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1, min_child_samples=20, reg_lambda=1.0, n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(X_min[tr], y_train[tr]); oof_base[va] = m.predict_proba(X_min[va])[:, 1]
    hn = adult_mask_arr & (oof_base >= 0.05)
    sw = np.ones(len(y_train), dtype=np.float32); sw[hn] = 20.0
    score_C = np.zeros(len(y_train), dtype=np.float32)
    for tr, va in skf.split(X_min, y_train):
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1, min_child_samples=20, reg_lambda=1.0, n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(X_min[tr], y_train[tr], sample_weight=sw[tr]); score_C[va] = m.predict_proba(X_min[va])[:, 1]

    pos_mask = y == 1
    neg_mask = y == 0
    adult_mask = adult_mask_arr

    def report(scores, label, thresholds, prod_decision=None):
        print(f"\n[{label}]  AUC vs all negatives: {roc_auc_score(y, scores):.4f}")
        print(f"  {'thr':>9} {'recall':>8} {'FPR_all':>9} {'aFP':>8} {'aFP_cnt':>9} {'prec_sample':>13} {'F1_sample':>10} {'prec@4.64%':>12} {'F1@4.64%':>10}")
        if prod_decision is not None:
            tp = int(((prod_decision) & pos_mask).sum())
            fp_total = int(((prod_decision) & neg_mask).sum())
            afp = int(((prod_decision) & adult_mask).sum())
            recall = tp / max(1, pos_mask.sum())
            fpr_all = fp_total / max(1, neg_mask.sum())
            afp_rate = afp / max(1, adult_mask.sum())
            prec_sample = tp / max(1, tp + fp_total)
            f1_sample = 2*prec_sample*recall/(prec_sample+recall) if (prec_sample+recall)>0 else 0
            CP = 0.0464
            p_corr = recall * CP / max(1e-9, recall*CP + fpr_all*(1-CP))
            f1_corr = 2*p_corr*recall/(p_corr+recall) if (p_corr+recall)>0 else 0
            print(f"  {'(prod)':>9} {recall:>7.1%} {fpr_all:>8.2%} {afp_rate:>7.2%} {afp:>8d} {prec_sample:>12.1%} {f1_sample:>9.3f} {p_corr:>11.1%} {f1_corr:>9.3f}")
        for thr in thresholds:
            tp = int(((scores >= thr) & pos_mask).sum())
            fp_total = int(((scores >= thr) & neg_mask).sum())
            afp = int(((scores >= thr) & adult_mask).sum())
            recall = tp / max(1, pos_mask.sum())
            fpr_all = fp_total / max(1, neg_mask.sum())
            afp_rate = afp / max(1, adult_mask.sum())
            prec_sample = tp / max(1, tp + fp_total)
            f1_sample = 2*prec_sample*recall/(prec_sample+recall) if (prec_sample+recall)>0 else 0
            CP = 0.0464
            p_corr = recall * CP / max(1e-9, recall*CP + fpr_all*(1-CP))
            f1_corr = 2*p_corr*recall/(p_corr+recall) if (p_corr+recall)>0 else 0
            print(f"  {thr:>9.3f} {recall:>7.1%} {fpr_all:>8.2%} {afp_rate:>7.2%} {afp:>8d} {prec_sample:>12.1%} {f1_sample:>9.3f} {p_corr:>11.1%} {f1_corr:>9.3f}")

    report(score_A, "A. Production detect.ts (binary @ ratio≥0.85, risk≥0.01)", [], prod_decision=A_blocked)
    report(score_B, "B. K=25 (existing piper LGBM)", [0.85, 0.50, 0.20, 0.10])
    report(score_C, "C. K=30 (BCI) — ours", [0.20, 0.10, 0.05])

    # Breakdown by age band at the recommended operating points
    bands = [("≤10", (min_ages >= 0) & (min_ages <= 10)),
             ("11-14", (min_ages >= 11) & (min_ages <= 14)),
             ("15-17", (min_ages >= 15) & (min_ages <= 17)),
             ("18+",   min_ages >= 18),
             ("no_age", min_ages < 0)]
    print(f"\n=== Fire-rate breakdown by Qwen3 age band ===")
    print(f"{'band':>7} {'n':>5}  {'A(prod)':>9} {'B(K25@0.5)':>11} {'C(K30@0.10)':>13} {'C(K30@0.20)':>13}")
    for name, mask in bands:
        n = int(mask.sum())
        if n == 0: continue
        a_fire = int((A_blocked & mask).sum())
        b_fire = int(((score_B >= 0.5) & mask).sum())
        c_10   = int(((score_C >= 0.10) & mask).sum())
        c_20   = int(((score_C >= 0.20) & mask).sum())
        print(f"  {name:>6} {n:>5}  {a_fire:>4d}/{n} ({a_fire/n:>4.0%})  {b_fire:>4d}/{n} ({b_fire/n:>4.0%})  {c_10:>4d}/{n} ({c_10/n:>4.0%})  {c_20:>4d}/{n} ({c_20/n:>4.0%})")


if __name__ == "__main__":
    main()
