"""3-way comparison:
  A. detect.ts noisy-OR (CURRENT PRODUCTION in the backend `detect.ts`)
  B. existing piper LGBM in violations-detector-test-v3 (other team's WIP, not yet activated)
  C. ours — SigLIP-only LGBM (prompt-free, body/context split, :x cap=0.1)

All three are evaluated on the same 6,680 rows (477 children + 6,203 negatives)
with labels corrected (y = qwen3_min_age <= 10).
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from label_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS
from train_for_piper import (
    siglip_labels_raw,
    combine_scores,
    predict_compact,
    XMULT_CAP,
)

ROOT = Path(__file__).resolve().parent


def parse_existing_lgbm():
    """Read existing piper lgbm_evaluate JS source, extract features + trees + threshold."""
    d = json.load(open(ROOT / "piper_pipeline_v3.json"))
    script = d["pipeline"]["nodes"]["lgbm_evaluate"]["script"]
    # Extract LGBM_FEATURES (single-line)
    m = re.search(r"const LGBM_FEATURES = (\[[^\n]+\]);", script)
    feats = json.loads(m.group(1))
    # Extract LGBM_TREES via balanced bracket count (it's huge multiline JSON)
    start = script.find("const LGBM_TREES = [") + len("const LGBM_TREES = ")
    depth = 0
    end = start
    in_str = False; esc = False
    for i in range(start, len(script)):
        ch = script[i]
        if esc:
            esc = False; continue
        if in_str:
            if ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"':
            in_str = True; continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    trees = json.loads(script[start:end])
    thr = float(re.search(r"LGBM_THRESHOLD = ([\d\.]+)", script).group(1))
    return feats, trees, thr


def detect_ts_noisy_or_score(raw_siglip: str) -> float:
    """Mirror detect.ts underage decision — but return the riskScoreInTotal
    (used as a continuous score for ROC). The detect.ts decision is
    `riskScoreInTotal >= 0.85 AND risk >= 0.01`. We approximate as a score by
    returning riskScoreInTotal multiplied by an indicator that risk meets the
    min_score floor (otherwise 0)."""
    if not raw_siglip:
        return 0.0
    sl = json.loads(raw_siglip)
    under = (((sl.get("underage") or {}).get("labels") or {}).get("underage")) or {}
    adult_l = (((sl.get("underage") or {}).get("labels") or {}).get("adult")) or {}
    # Apply :x multiplier (matches detect.ts's `affected` rule)
    def apply_mult(d):
        out = {}
        for k, v in d.items():
            m = re.search(r":x(\d+(?:\.\d+)?)$", k)
            if m:
                out[k[:m.start()]] = min(float(v) * float(m.group(1)), 0.999)
            else:
                out[k] = float(v)
        return out
    risk_dict = apply_mult(under)
    adult_dict = apply_mult(adult_l)
    risk = combine_scores(risk_dict, lambda k: True)
    adult = combine_scores(adult_dict, lambda k: True)
    total = risk + adult
    ratio = (risk / total) if total > 0 else 0.0
    # detect.ts boolean: ratio >= 0.85 AND risk >= 0.01 (since underage_min_score = 0.01)
    # For a continuous score, we'll use ratio when risk meets the floor.
    if risk < 0.01:
        return 0.0
    return ratio


def existing_piper_build_vec(raw_siglip: str, features: list[str]) -> np.ndarray:
    """Existing piper LGBM expects features keyed by post-:x-multiplier-stripped
    name (with :x suffix preserved when present) and RAW (unmultiplied) values."""
    vec = np.zeros(len(features), dtype=np.float32)
    feat_idx = {n: i for i, n in enumerate(features)}
    if not raw_siglip:
        return vec
    sl = json.loads(raw_siglip)
    under = (((sl.get("underage") or {}).get("labels") or {}).get("underage")) or {}
    adult_l = (((sl.get("underage") or {}).get("labels") or {}).get("adult")) or {}
    # Existing model stores feature names with :x preserved + RAW values (unmultiplied).
    # scored.sqlite contains POST-multiplier values, so divide by N to recover raw.
    for k, v in under.items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        raw = float(v) / float(m.group(1)) if m else float(v)
        if k in feat_idx:
            vec[feat_idx[k]] = raw
    for k, v in adult_l.items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        raw = float(v) / float(m.group(1)) if m else float(v)
        fn = "adult__" + k
        if fn in feat_idx:
            vec[feat_idx[fn]] = raw
    # Existing model also has _minor / _adult / _confidence derived (no multiplier)
    raw_under = {k: (float(v) / float(re.search(r":x(\d+(?:\.\d+)?)$", k).group(1)) if ":x" in k else float(v)) for k, v in under.items()}
    raw_adult = {k: (float(v) / float(re.search(r":x(\d+(?:\.\d+)?)$", k).group(1)) if ":x" in k else float(v)) for k, v in adult_l.items()}
    minor = combine_scores(raw_under, lambda k: True)
    adult = combine_scores(raw_adult, lambda k: True)
    total = minor + adult
    conf = (minor / total) if total > 0 else 0.0
    if "_minor" in feat_idx:      vec[feat_idx["_minor"]] = minor
    if "_adult" in feat_idx:      vec[feat_idx["_adult"]] = adult
    if "_confidence" in feat_idx: vec[feat_idx["_confidence"]] = conf
    return vec


def ours_build_vec(raw_siglip: str, features: list[str]) -> np.ndarray:
    """Our SigLIP-only model: body/context/interaction split + :x cap at XMULT_CAP."""
    vec = np.zeros(len(features), dtype=np.float32)
    feat_idx = {n: i for i, n in enumerate(features)}
    labels = siglip_labels_raw(raw_siglip)  # already strips :x and caps
    for fn, val in labels.items():
        if fn in feat_idx: vec[feat_idx[fn]] = float(val)
    minor = combine_scores(labels, lambda k: not k.startswith("adult__"))
    adult = combine_scores(labels, lambda k: k.startswith("adult__"))
    total = minor + adult
    conf = (minor / total) if total > 0 else 0.0
    body  = combine_scores(labels, lambda k: k in BODY_LABELS)
    ctx   = combine_scores(labels, lambda k: k in CONTEXT_LABELS)
    inter = combine_scores(labels, lambda k: k in INTERACTION_LABELS)
    bc = body + ctx
    body_vs_ctx = (body / bc) if bc > 0 else 0.0
    for name, val in [("_minor", minor), ("_adult", adult), ("_confidence", conf),
                      ("_child_body", body), ("_child_context", ctx),
                      ("_child_interaction", inter), ("_body_vs_context", body_vs_ctx)]:
        if name in feat_idx: vec[feat_idx[name]] = val
    return vec


def main():
    # Load all rows with corrected labels
    rows = []
    for db_path, src in [("scored.sqlite", "cand"), ("scored_benign.sqlite", "benign")]:
        c = sqlite3.connect(ROOT / db_path); c.row_factory = sqlite3.Row
        for r in c.execute("SELECT generation_id, qwen3_min_age, raw_siglip FROM scored WHERE error IS NULL AND raw_siglip IS NOT NULL"):
            y = 1 if (r["qwen3_min_age"] is not None and r["qwen3_min_age"] <= 10) else 0
            rows.append((r["generation_id"], src, y, r["raw_siglip"]))
    n = len(rows)
    n_pos = sum(1 for r in rows if r[2] == 1)
    print(f"rows={n}, pos={n_pos}, neg={n - n_pos}", file=sys.stderr)

    # A. detect.ts noisy-OR
    print("scoring detect.ts noisy-OR...", file=sys.stderr)
    score_A = np.array([detect_ts_noisy_or_score(r[3]) for r in rows], dtype=np.float32)

    # B. existing piper LGBM
    print("loading existing piper LGBM...", file=sys.stderr)
    ex_feats, ex_trees, ex_thr = parse_existing_lgbm()
    print(f"  existing: {len(ex_feats)} features, {len(ex_trees)} trees, thr={ex_thr}", file=sys.stderr)
    print("scoring existing piper LGBM (direct prediction)...", file=sys.stderr)
    score_B = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(rows):
        vec = existing_piper_build_vec(r[3], ex_feats)
        score_B[i] = predict_compact(ex_trees, vec)

    # C. ours
    print("loading ours...", file=sys.stderr)
    ours = json.loads((ROOT / "piper_lgbm_model.json").read_text())
    our_feats = ours["features"]
    print(f"  ours: {len(our_feats)} features, {len(ours['trees'])} trees", file=sys.stderr)
    # OOF for unbiased estimate
    print("scoring ours with 5-fold OOF...", file=sys.stderr)
    y_arr = np.array([r[2] for r in rows], dtype=np.int8)
    X_ours = np.zeros((n, len(our_feats)), dtype=np.float32)
    for i, r in enumerate(rows):
        X_ours[i] = ours_build_vec(r[3], our_feats)
    score_C = np.zeros(n, dtype=np.float32)
    n_pos_total = int(y_arr.sum()); n_neg_total = n - n_pos_total
    for fold, (tr, va) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_ours, y_arr), 1):
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1,
                                min_child_samples=20, reg_lambda=1.0,
                                n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(X_ours[tr], y_arr[tr])
        score_C[va] = m.predict_proba(X_ours[va])[:, 1]

    src = np.array([r[1] for r in rows])
    y = y_arr
    benign_mask = src == "benign"
    children_mask = y == 1

    # AUC (benign-only denominator — deployment-relevant)
    sel = benign_mask | children_mask
    print("\n" + "="*60)
    print("3-WAY COMPARISON  (n=477 children vs n=3000 benign)")
    print("="*60)
    print(f"AUC (children vs benign-only):")
    print(f"  A. detect.ts noisy-OR:    {roc_auc_score(y[sel], score_A[sel]):.4f}")
    print(f"  B. existing piper LGBM:   {roc_auc_score(y[sel], score_B[sel]):.4f}")
    print(f"  C. ours (SigLIP-only):    {roc_auc_score(y[sel], score_C[sel]):.4f}")

    def op_table(scores, label, thrs):
        print(f"\n[{label}]")
        print(f"  {'thr':>8} {'recall':>9} {'fp':>8} {'precision':>10}")
        for thr in thrs:
            tp = (scores[children_mask] >= thr).sum() / max(1, children_mask.sum())
            fp = (scores[benign_mask] >= thr).sum() / max(1, benign_mask.sum())
            tp_count = int((scores[children_mask] >= thr).sum())
            all_flagged = int((scores[benign_mask] >= thr).sum()) + tp_count
            prec = (tp_count / all_flagged) if all_flagged else 0.0
            print(f"  {thr:>8.4f} {tp:>8.1%} {fp:>7.2%} {prec:>9.1%}")

    # A operating points — detect.ts only has one binary decision point (ratio >= 0.85)
    op_table(score_A, "A. detect.ts noisy-OR (production)", [0.85, 0.50, 0.30, 0.10, 0.01])
    op_table(score_B, "B. existing piper LGBM (other team)", [0.85, 0.50, 0.30, 0.10, 0.05, 0.02])
    op_table(score_C, "C. OURS (SigLIP-only)",                [0.20, 0.10, 0.075, 0.05, 0.02, 0.01, 0.005])

    # Save raw
    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append({
            "gen_id": r[0], "source": r[1], "y": int(r[2]),
            "score_A_detect_ts": float(score_A[i]),
            "score_B_existing_piper": float(score_B[i]),
            "score_C_ours_siglip_only": float(score_C[i]),
        })
    (ROOT / "three_way_results.json").write_text(json.dumps(out_rows))
    print(f"\nwrote three_way_results.json")


if __name__ == "__main__":
    main()
