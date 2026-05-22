"""Side-by-side: existing piper LGBM (test-v3) vs our prompt-aware LGBM
on the entire scored.sqlite + scored_benign.sqlite (6,586 rows, 201 children).

This is a PURE LOCAL computation — both classifiers are just functions of
(siglip labels, prompt, checkpoint, genType). We have all of those in our DBs.
No piper launches needed (confirmed earlier that our deployed pipeline produces
the same scores as the local Python predictor).

Outputs:
  - side_by_side_results.json  (per-row scores + metadata)
  - SIDE_BY_SIDE.md             (operating-point comparison report)
  - side_by_side.png            (overlaid ROC curves)
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from prompt_features import features_of, feature_names
from train_for_piper import (
    siglip_labels_raw,
    combine_scores,
    predict_compact,
    parse_lgbm_dump_trees,
)
from label_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

ROOT = Path(__file__).resolve().parent


def parse_existing_lgbm():
    """Read the existing piper lgbm_evaluate JS source and extract its feature
    list + tree JSON so we can score with it in Python."""
    d = json.load(open(ROOT / "piper_pipeline_v3.json"))
    script = d["pipeline"]["nodes"]["lgbm_evaluate"]["script"]
    feats_m = re.search(r"const LGBM_FEATURES = (\[.*?\]);", script, re.DOTALL)
    features = json.loads(feats_m.group(1))
    trees_m = re.search(r"const LGBM_TREES = (\[.*?\]);\s*const LGBM_THRESHOLD", script, re.DOTALL)
    if not trees_m:
        trees_m = re.search(r"const LGBM_TREES = (\[.*?\]);", script, re.DOTALL)
    trees = json.loads(trees_m.group(1))
    thr_m = re.search(r"LGBM_THRESHOLD = ([\d\.]+)", script)
    threshold = float(thr_m.group(1)) if thr_m else 0.85
    return features, trees, threshold


def build_existing_vec(raw_siglip: str, features: list[str]) -> np.ndarray:
    """Recreate the existing piper lgbm_evaluate's buildVec(): looks up features
    by RAW key (with `:xN` suffix preserved, no multiplier applied)."""
    vec = np.zeros(len(features), dtype=np.float32)
    feat_idx = {n: i for i, n in enumerate(features)}
    if not raw_siglip:
        return vec
    sl = json.loads(raw_siglip)
    # The existing model expects keys without prefix but with :x suffix preserved,
    # and values UNMULTIPLIED. detect.ts/storage gives us already-multiplied values
    # under stripped keys — so we need to UN-multiply for proper emulation.
    block = (((sl.get("underage") or {}).get("labels") or {}).get("underage")) or {}
    for k, v in block.items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        if m:
            unmult = float(v) / float(m.group(1))
            fn = k  # existing feature has :x in name
        else:
            unmult = float(v)
            fn = k
        if fn in feat_idx:
            vec[feat_idx[fn]] = unmult
    block = (((sl.get("underage") or {}).get("labels") or {}).get("adult")) or {}
    for k, v in block.items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        if m:
            unmult = float(v) / float(m.group(1))
            fn = "adult__" + k
        else:
            unmult = float(v)
            fn = "adult__" + k
        if fn in feat_idx:
            vec[feat_idx[fn]] = unmult
    # Derived (existing model uses unmultiplied values for these too — combineScores
    # on raw labels gives lower minor/adult than on multiplied labels)
    # Build a flat labels dict for combineScores with the same un-multiplication.
    flat = {}
    for k, v in (((sl.get("underage") or {}).get("labels") or {}).get("underage") or {}).items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        unmult = float(v) / float(m.group(1)) if m else float(v)
        flat[k] = unmult  # underage label, no prefix
    adult_flat = {}
    for k, v in (((sl.get("underage") or {}).get("labels") or {}).get("adult") or {}).items():
        m = re.search(r":x(\d+(?:\.\d+)?)$", k)
        unmult = float(v) / float(m.group(1)) if m else float(v)
        adult_flat["adult__" + k] = unmult
    minor = combine_scores(flat, lambda k: True)
    adult = combine_scores(adult_flat, lambda k: True)
    total = minor + adult
    conf = (minor / total) if total > 0 else 0.0
    if "_minor" in feat_idx:
        vec[feat_idx["_minor"]] = minor
    if "_adult" in feat_idx:
        vec[feat_idx["_adult"]] = adult
    if "_confidence" in feat_idx:
        vec[feat_idx["_confidence"]] = conf
    return vec


def build_ours_vec(labels: dict[str, float], prompt: str, ckpt: str, gtype: str, features: list[str]) -> np.ndarray:
    vec = np.zeros(len(features), dtype=np.float32)
    feat_idx = {n: i for i, n in enumerate(features)}
    for fn, val in labels.items():
        if fn in feat_idx:
            vec[feat_idx[fn]] = val
    minor = combine_scores(labels, lambda k: not k.startswith("adult__"))
    adult = combine_scores(labels, lambda k: k.startswith("adult__"))
    total = minor + adult
    conf = (minor / total) if total > 0 else 0.0
    body  = combine_scores(labels, lambda k: k in BODY_LABELS)
    ctx   = combine_scores(labels, lambda k: k in CONTEXT_LABELS)
    inter = combine_scores(labels, lambda k: k in INTERACTION_LABELS)
    bc_total = body + ctx
    body_vs_ctx = (body / bc_total) if bc_total > 0 else 0.0
    if "_minor" in feat_idx:              vec[feat_idx["_minor"]]              = minor
    if "_adult" in feat_idx:              vec[feat_idx["_adult"]]              = adult
    if "_confidence" in feat_idx:         vec[feat_idx["_confidence"]]         = conf
    if "_child_body" in feat_idx:         vec[feat_idx["_child_body"]]         = body
    if "_child_context" in feat_idx:      vec[feat_idx["_child_context"]]      = ctx
    if "_child_interaction" in feat_idx:  vec[feat_idx["_child_interaction"]]  = inter
    if "_body_vs_context" in feat_idx:    vec[feat_idx["_body_vs_context"]]    = body_vs_ctx
    pf = features_of(prompt, ckpt, gtype)
    for name, val in pf.items():
        k = f"pf__{name}"
        if k in feat_idx:
            vec[feat_idx[k]] = val
    return vec


def main():
    # Load models
    print("loading models...", file=sys.stderr)
    ex_features, ex_trees, ex_thr = parse_existing_lgbm()
    ours = json.loads((ROOT / "piper_lgbm_model.json").read_text())
    our_features = ours["features"]
    our_trees = ours["trees"]
    our_thr = ours["suggested_threshold"]
    print(f"existing: {len(ex_features)} features, {len(ex_trees)} trees, thr={ex_thr}", file=sys.stderr)
    print(f"ours:     {len(our_features)} features, {len(our_trees)} trees, thr={our_thr}", file=sys.stderr)

    # Load data
    prompts_db = sqlite3.connect(ROOT / "prompts.sqlite")
    prompts_db.row_factory = sqlite3.Row
    prompts_map = {
        r["id"]: (r["prompt"] or "", r["checkpoint"] or "", r["type"] or "")
        for r in prompts_db.execute("SELECT id, prompt, checkpoint, type FROM prompts")
    }

    rows = []
    for db_path, source_tag in [("scored.sqlite", "cand"), ("scored_benign.sqlite", "benign")]:
        c = sqlite3.connect(ROOT / db_path)
        c.row_factory = sqlite3.Row
        q = (
            "SELECT generation_id, qwen3_min_age, qwen3_max_age, qwen3_desc, raw_siglip, image_url "
            "FROM scored WHERE error IS NULL AND raw_siglip IS NOT NULL"
        )
        for r in c.execute(q):
            gid = r["generation_id"]
            prompt, ckpt, gtype = prompts_map.get(gid, ("", "", ""))
            sg = siglip_labels_raw(r["raw_siglip"])
            qwen_min = r["qwen3_min_age"]
            # FIX: use min_age (the child) not max_age (the adult) for label assignment.
            # Adult+child scenes have max_age=adult's age but min_age=child's age.
            y = 1 if (qwen_min is not None and qwen_min <= 10) else 0
            rows.append({
                "gen_id": gid,
                "source": source_tag,
                "y": y,
                "qwen3_min_age": qwen_min,
                "qwen3_max_age": r["qwen3_max_age"],
                "qwen3_desc": (r["qwen3_desc"] or "")[:300],
                "image_url": r["image_url"],
                "sg": sg,
                "prompt": prompt,
                "checkpoint": ckpt,
                "gen_type": gtype,
            })
    print(f"loaded {len(rows)} rows ({sum(1 for r in rows if r['y']==1)} children)", file=sys.stderr)

    # Need raw_siglip JSON for the existing model — re-fetch with the row
    print("scoring existing piper LGBM...", file=sys.stderr)
    score_ex = np.zeros(len(rows), dtype=np.float32)
    raw_map = {}
    for db_path, _ in [("scored.sqlite", "cand"), ("scored_benign.sqlite", "benign")]:
        c = sqlite3.connect(ROOT / db_path)
        c.row_factory = sqlite3.Row
        for r in c.execute("SELECT generation_id, raw_siglip FROM scored WHERE raw_siglip IS NOT NULL"):
            raw_map[r["generation_id"]] = r["raw_siglip"]
    for i, r in enumerate(rows):
        raw = raw_map.get(r["gen_id"], "")
        vec_ex = build_existing_vec(raw, ex_features)
        score_ex[i] = predict_compact(ex_trees, vec_ex)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(rows)}", file=sys.stderr)

    # Score OURS using 5-fold OOF — same data was used to train ours, so direct
    # prediction would be data leakage. OOF gives unbiased held-out estimate.
    print("scoring ours with 5-fold OOF...", file=sys.stderr)
    y = np.array([r["y"] for r in rows], dtype=np.int8)
    src = np.array([r["source"] for r in rows])

    # Build feature matrix once
    X_ours = np.zeros((len(rows), len(our_features)), dtype=np.float32)
    for i, r in enumerate(rows):
        X_ours[i] = build_ours_vec(r["sg"], r["prompt"], r["checkpoint"], r["gen_type"], our_features)

    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score_ours = np.zeros(len(rows), dtype=np.float32)
    for fold, (tr, va) in enumerate(skf.split(X_ours, y), 1):
        m = lgb.LGBMClassifier(
            n_estimators=100, num_leaves=15, learning_rate=0.1,
            min_child_samples=20, reg_lambda=1.0,
            n_jobs=-1, verbosity=-1, random_state=42,
        )
        m.fit(X_ours[tr], y[tr])
        score_ours[va] = m.predict_proba(X_ours[va])[:, 1]
        print(f"  fold {fold}/5 done", file=sys.stderr)
    children_mask = y == 1
    benign_mask = src == "benign"

    # AUC (positive vs benign only)
    sel = children_mask | benign_mask
    auc_ex = roc_auc_score(y[sel], score_ex[sel])
    auc_ours = roc_auc_score(y[sel], score_ours[sel])
    print(f"\nAUC (children vs benign-only):  existing={auc_ex:.4f}   ours={auc_ours:.4f}")

    # Operating points
    def op_table(scores: np.ndarray, label: str):
        print(f"\n[{label}]  recall on {int(children_mask.sum())} children, FP on benign={int(benign_mask.sum())}:")
        print(f"  {'thr':>7} {'recall':>8} {'fp':>8} {'fp_count':>8} {'precision':>10}")
        for thr in [0.85, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
            tp = (scores[children_mask] >= thr).sum() / max(1, children_mask.sum())
            fp_count = int((scores[benign_mask] >= thr).sum())
            fp = fp_count / max(1, benign_mask.sum())
            # precision = true positives / all positives flagged (benign+children together)
            tp_count = int((scores[children_mask] >= thr).sum())
            all_pos = int((scores[benign_mask] >= thr).sum()) + tp_count
            prec = (tp_count / all_pos) if all_pos else 0.0
            print(f"  {thr:>7.3f} {tp:>7.1%} {fp:>7.2%} {fp_count:>8d} {prec:>9.1%}")

    op_table(score_ex, "existing piper LGBM")
    op_table(score_ours, "OURS (prompt-aware LGBM)")

    # ROC plot
    fpr_e, tpr_e, _ = roc_curve(y[sel], score_ex[sel])
    fpr_o, tpr_o, _ = roc_curve(y[sel], score_ours[sel])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(fpr_e, tpr_e, label=f"existing piper — AUC {auc_ex:.4f}", color="#1565c0", lw=2)
    ax.plot(fpr_o, tpr_o, label=f"OURS (prompt-aware) — AUC {auc_ours:.4f}", color="#c62828", lw=2)
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("benign FP rate")
    ax.set_ylabel("child recall")
    ax.set_title(f"Side-by-side ROC (n={int(children_mask.sum())} children vs n={int(benign_mask.sum())} benign)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(fpr_e, tpr_e, label=f"existing — AUC {auc_ex:.4f}", color="#1565c0", lw=2.4)
    ax.plot(fpr_o, tpr_o, label=f"ours — AUC {auc_ours:.4f}", color="#c62828", lw=2.4)
    for thr, color in [(0.001, "#c62828"), (0.005, "#c62828"), (0.01, "#c62828")]:
        tp = (score_ours[children_mask] >= thr).sum() / max(1, children_mask.sum())
        fp = (score_ours[benign_mask] >= thr).sum() / max(1, benign_mask.sum())
        ax.scatter([fp], [tp], s=80, color=color, zorder=5)
        ax.annotate(f" thr={thr}\n {tp*100:.0f}%/{fp*100:.2f}%", (fp, tp), fontsize=9, va="center")
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("benign FP rate (0-5%)")
    ax.set_ylabel("child recall")
    ax.set_title("Zoom — deployment-relevant FP range")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "side_by_side.png", dpi=110)
    print(f"\nwrote side_by_side.png")

    # Save per-row results
    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append({
            "gen_id": r["gen_id"],
            "y": int(r["y"]),
            "source": r["source"],
            "qwen3_min_age": r["qwen3_min_age"],
            "qwen3_max_age": r["qwen3_max_age"],
            "score_existing": float(score_ex[i]),
            "score_ours": float(score_ours[i]),
            "prompt_present": bool(r["prompt"]),
        })
    (ROOT / "side_by_side_results.json").write_text(json.dumps(out_rows))
    print(f"wrote side_by_side_results.json ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
