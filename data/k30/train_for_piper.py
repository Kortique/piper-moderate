"""Train a LightGBM classifier compatible with piper's lgbm_evaluate node
embedded-JS format.

Mirrors the existing piper node's hyperparameters (100 trees, 15 leaves) but
trains on our larger dataset (6,586 samples) WITH 22 prompt features added.
Outputs a JSON file containing:
  - LGBM_FEATURES: ordered feature names (mirrors existing convention for siglip)
  - LGBM_TREES:    compact {s, c, l, r} tree representation
  - LGBM_THRESHOLD: suggested operating threshold

Their feature naming convention (so the JS node code can stay near-identical):
  - SigLIP underage labels: strip "underage_" prefix → feature name
  - SigLIP adult labels:    strip "adult_" prefix → "adult__<rest>"
  - Aggregate derived:       _minor, _adult, _confidence
  - NEW: prompt features:    "pf__<name>" (double underscore prefix for namespace)

Output:  piper_lgbm_model.json
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from prompt_features import features_of, feature_names
from label_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS, categorize

ROOT = Path(__file__).resolve().parent


# Cap for :x-multiplied features. detect.ts caps at 0.999 which is too aggressive
# — it lets weak raw signal (0.05) blow up to 0.999 and dominate. Re-cap at 0.3
# here. Both training (this Python) AND inference (JS) must use the same cap.
XMULT_CAP = 0.1


def siglip_labels_raw(raw):
    """Extract underage + adult labels in piper's feature-name convention.

    Storage format in scored.sqlite is the per-category detect.ts output. CRITICAL:
    detect.ts already applies the `:xN` multiplier in `affected` BEFORE storing;
    storage values are POST-multiplied (capped at 0.999). We strip the `:xN`
    suffix from the key but ALSO re-cap the value at XMULT_CAP (0.3) — without
    this cap, weak raw SigLIP signal on size-contrast labels (maternal_contrast,
    man_with_young_girl, adult_with_girl) gets amplified to 0.999 and causes
    false positives on adult-only images with size/age contrast.

    Mapping to piper feature-name convention:
      sl['underage']['labels']['underage'][<rest>:xN]  →  feature `<rest>`,         val_capped
      sl['underage']['labels']['adult'][<rest>:xN]     →  feature `adult__<rest>`, val_capped
    """
    if not raw:
        return {}
    sl = json.loads(raw)
    out = {}
    block = (((sl.get("underage") or {}).get("labels") or {}).get("underage")) or {}
    for k, v in block.items():
        fn, val = _strip_and_cap(k, float(v))
        out[fn] = val
    block = (((sl.get("underage") or {}).get("labels") or {}).get("adult")) or {}
    for k, v in block.items():
        fn, val = _strip_and_cap(k, float(v))
        out[f"adult__{fn}"] = val
    return out


def _strip_and_cap(name: str, val: float) -> tuple[str, float]:
    """Strip `:xN` suffix. If suffix was present, re-cap value at XMULT_CAP."""
    m = re.search(r":x(\d+(?:\.\d+)?)$", name)
    if m:
        return name[: m.start()], min(val, XMULT_CAP)
    return name, val


def combine_scores(label_vals: dict[str, float], prefix_match) -> float:
    """Noisy-OR over labels matching prefix_match (callable taking key → bool)."""
    p = 1.0
    for k, v in label_vals.items():
        if prefix_match(k):
            p *= 1 - v
    return 1 - p


def load_rows():
    """Yield (gen_id, raw_label_dict, prompt, checkpoint, gen_type, y, source)."""
    rows = []
    p = sqlite3.connect(ROOT / "prompts.sqlite")
    p.row_factory = sqlite3.Row
    prompts_map = {
        r["id"]: (r["prompt"] or "", r["checkpoint"] or "", r["type"] or "")
        for r in p.execute("SELECT id, prompt, checkpoint, type FROM prompts")
    }
    c = sqlite3.connect(ROOT / "scored.sqlite")
    c.row_factory = sqlite3.Row
    # Positive = qwen3 saw a minor (min_age ≤ POSITIVE_AGE_CUTOFF) anywhere in
    # the image. min_age catches the child even when an adult is also present.
    # Cutoff chosen 2026-05-21 after sweeping 10/11/12/13/14/15 — the cliff in
    # adult-FP rate is between 13 and 14 (0.99% → 1.72%); past 14 the model
    # genuinely can't tell mature 15yo from young 18yo.
    POSITIVE_AGE_CUTOFF = 14
    for r in c.execute(
        "SELECT generation_id, qwen3_min_age, qwen3_max_age, raw_siglip FROM scored WHERE error IS NULL AND raw_siglip IS NOT NULL"
    ):
        gid = r["generation_id"]
        pr, cp, tp = prompts_map.get(gid, ("", "", ""))
        sg = siglip_labels_raw(r["raw_siglip"])
        y = 1 if (r["qwen3_min_age"] is not None and r["qwen3_min_age"] <= POSITIVE_AGE_CUTOFF) else 0
        rows.append((gid, sg, pr, cp, tp, y, "cand"))
    b = sqlite3.connect(ROOT / "scored_benign.sqlite")
    b.row_factory = sqlite3.Row
    for r in b.execute(
        "SELECT generation_id, qwen3_min_age, qwen3_max_age, raw_siglip FROM scored WHERE error IS NULL AND raw_siglip IS NOT NULL"
    ):
        gid = r["generation_id"]
        pr, cp, tp = prompts_map.get(gid, ("", "", ""))
        y = 1 if (r["qwen3_min_age"] is not None and r["qwen3_min_age"] <= POSITIVE_AGE_CUTOFF) else 0
        rows.append((gid, siglip_labels_raw(r["raw_siglip"]), pr, cp, tp, y, "benign"))
    return rows


def build_features(rows):
    """Build features matching piper's convention + add prompt features.

    NEW: replaces single `_minor` aggregate with three:
       `_child_body`         noisy-OR over BODY labels (explicit anatomy / sex acts)
       `_child_context`      noisy-OR over CONTEXT labels (clothing / scene / pose)
       `_child_interaction`  noisy-OR over INTERACTION labels (`:x`-multiplied adult-with-child)
    Plus keeps `_minor` for backward compatibility (still total noisy-OR of all underage).
    """
    seen_siglip_names = set()
    for r in rows:
        for fn in r[1].keys():
            seen_siglip_names.add(fn)
    siglip_names = sorted(seen_siglip_names)

    # Derived aggregate features — split into body / context / interaction
    derived = [
        "_minor",                # all underage labels combined (kept for compat)
        "_adult",                # all adult labels combined
        "_confidence",           # _minor / (_minor + _adult)
        "_child_body",           # noisy-OR over body-labels only
        "_child_context",        # noisy-OR over context-labels only
        "_child_interaction",    # noisy-OR over `:x` interaction-labels only
        "_body_vs_context",      # _child_body / (_child_body + _child_context)
    ]

    # Prompt features removed 2026-05-21 — they were hand-written regex /
    # substring checks (no language model), trivially defeated by leet-speak
    # ("sch00lg1rl"), and added recall only when SigLIP was undertrained on
    # adult+child cases. After the body/context split + min_age labeling fix,
    # SigLIP-only is at parity with prompt-aware. Cleaner architecture.
    all_features = siglip_names + derived
    feat_idx = {n: i for i, n in enumerate(all_features)}

    underage_match = lambda k: not k.startswith("adult__")
    adult_match = lambda k: k.startswith("adult__")
    body_match = lambda k: k in BODY_LABELS
    context_match = lambda k: k in CONTEXT_LABELS
    interaction_match = lambda k: k in INTERACTION_LABELS

    X = np.zeros((len(rows), len(all_features)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int8)
    src = []
    for i, (gid, sg, prompt, cp, tp, yy, s) in enumerate(rows):
        for fn, val in sg.items():
            if fn in feat_idx:
                X[i, feat_idx[fn]] = float(val)
        # Aggregate features
        minor = combine_scores(sg, underage_match)
        adult = combine_scores(sg, adult_match)
        total = minor + adult
        conf = (minor / total) if total > 0 else 0.0
        body = combine_scores(sg, body_match)
        ctx  = combine_scores(sg, context_match)
        inter = combine_scores(sg, interaction_match)
        body_ctx_total = body + ctx
        body_vs_ctx = (body / body_ctx_total) if body_ctx_total > 0 else 0.0
        X[i, feat_idx["_minor"]]              = minor
        X[i, feat_idx["_adult"]]              = adult
        X[i, feat_idx["_confidence"]]         = conf
        X[i, feat_idx["_child_body"]]         = body
        X[i, feat_idx["_child_context"]]      = ctx
        X[i, feat_idx["_child_interaction"]]  = inter
        X[i, feat_idx["_body_vs_context"]]    = body_vs_ctx
        y[i] = yy
        src.append(s)
    return X, y, np.array(src), all_features


def parse_lgbm_dump_trees(booster: lgb.Booster) -> list[dict]:
    """Convert lgbm dump_model() tree info to piper's compact {s, c, l, r} format.

    Each LightGBM tree's `tree_structure` is a recursive dict with either:
      - {split_feature, threshold, left_child, right_child, ...} (internal)
      - {leaf_index, leaf_value} (leaf)

    Compact format:
      s[i] = [feature_index, threshold]   for internal node i (0-indexed)
      c[i] = [left, right]                  positive = internal index, negative = -(leaf_index + 1)
      l[k] = leaf value at leaf_index k
      r    = root internal-node index (always 0 after our re-indexing)
    """
    dump = booster.dump_model()
    out = []
    for tree in dump["tree_info"]:
        struct = tree["tree_structure"]
        internals = []  # list of {feature, threshold, left, right}
        leaves = []     # list of leaf values, indexed by leaf_index

        # First pass: collect leaves in their numeric order
        def collect_leaves(node):
            if "leaf_index" in node:
                idx = node["leaf_index"]
                while len(leaves) <= idx:
                    leaves.append(0.0)
                leaves[idx] = node["leaf_value"]
                return
            collect_leaves(node["left_child"])
            collect_leaves(node["right_child"])
        collect_leaves(struct)

        # Second pass: number the internal nodes in pre-order and emit compact arrays
        def emit(node):
            if "leaf_index" in node:
                return -(node["leaf_index"] + 1)
            my_idx = len(internals)
            internals.append(None)  # placeholder
            left_ref = emit(node["left_child"])
            right_ref = emit(node["right_child"])
            internals[my_idx] = {
                "feature": node["split_feature"],
                "threshold": node["threshold"],
                "left": left_ref,
                "right": right_ref,
            }
            return my_idx
        emit(struct)

        s = [[n["feature"], n["threshold"]] for n in internals]
        c = [[n["left"], n["right"]] for n in internals]
        out.append({"s": s, "l": leaves, "c": c, "r": 0})
    return out


def predict_compact(trees: list[dict], vec: np.ndarray) -> float:
    """Reference predictor in pure Python — mirror of the JS lgbmPredict."""
    score = 0.0
    for t in trees:
        node = t["r"]
        while node >= 0:
            fi, thr = t["s"][node]
            l, r = t["c"][node]
            node = l if vec[fi] <= thr else r
        score += t["l"][-(node + 1)]
    return 1.0 / (1.0 + np.exp(-score))


def main():
    rows = load_rows()
    X, y, src, all_features = build_features(rows)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    print(f"rows={len(rows)} positives={n_pos} negatives={n_neg} features={len(all_features)}")

    # 5-fold OOF AUC for sanity
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=15,
            learning_rate=0.1,
            min_child_samples=20,
            reg_lambda=1.0,
            n_jobs=-1, verbosity=-1, random_state=42,
        )
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        fold_aucs.append(roc_auc_score(y[va], oof[va]))
    auc = roc_auc_score(y, oof)
    print(f"5-fold OOF AUC: {auc:.4f}  (folds: {[f'{a:.3f}' for a in fold_aucs]})")

    # Operating points on benign-only denominator
    benign_mask = (src == "benign")
    children_mask = (y == 1)
    print(f"\nOperating points (recall on {n_pos} children, FP on benign={int(benign_mask.sum())}):")
    print(f"  {'thr':>6} {'recall':>8} {'fp':>8} {'fp_count':>8}")
    for thr in [0.85, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        tp = (oof[children_mask] >= thr).sum() / max(1, children_mask.sum())
        fp_count = int((oof[benign_mask] >= thr).sum())
        fp = fp_count / max(1, benign_mask.sum())
        print(f"  {thr:>6.3f} {tp:>7.1%} {fp:>7.2%} {fp_count:>8d}")

    # Refit on all data, dump as JS-compatible format
    final = lgb.LGBMClassifier(
        n_estimators=100, num_leaves=15, learning_rate=0.1,
        min_child_samples=20, reg_lambda=1.0,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    final.fit(X, y)
    trees = parse_lgbm_dump_trees(final.booster_)
    print(f"\nDumped {len(trees)} trees, total leaves: {sum(len(t['l']) for t in trees)}")

    # Sanity: compare our compact predictor vs lightgbm on a few rows
    pick = np.random.RandomState(0).choice(len(X), size=5, replace=False)
    lgbm_p = final.predict_proba(X[pick])[:, 1]
    compact_p = np.array([predict_compact(trees, X[i]) for i in pick])
    print(f"lgbm vs compact max abs diff: {np.abs(lgbm_p - compact_p).max():.3e}")
    assert np.abs(lgbm_p - compact_p).max() < 1e-5, "compact predictor drifted from lightgbm"

    out = {
        "features": all_features,
        "trees": trees,
        "suggested_threshold": 0.005,
        "trained_on": {"n_rows": len(rows), "n_pos": n_pos, "n_neg": n_neg, "auc_oof": float(auc)},
        "hyperparams": {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20, "reg_lambda": 1.0},
    }
    (ROOT / "piper_lgbm_model.json").write_text(json.dumps(out))
    print(f"\nwrote piper_lgbm_model.json  ({(ROOT/'piper_lgbm_model.json').stat().st_size} bytes)")


if __name__ == "__main__":
    main()
