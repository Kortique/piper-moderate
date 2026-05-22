"""Generate the JS source for a replacement `lgbm_evaluate` piper node.

The node consumes the SigLIP labels from `ask_siglip2.labels` and runs our
prompt-FREE LightGBM trees. Reads piper_lgbm_model.json and emits
piper_lgbm_evaluate_node.js for pasting into the node's `script` field.

Dropped 2026-05-21: the 22 hand-written prompt features. Empirically they
added <0.5pp recall after the labeling fix + body/context aggregate split,
and they were trivially defeated by leet-speak in prompts (e.g. "sch00lg1rl").
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    m = json.loads((ROOT / "piper_lgbm_model.json").read_text())
    features = m["features"]
    trees = m["trees"]
    thr = m["suggested_threshold"]
    info = m["trained_on"]

    # XMULT_CAP must match train_for_piper.py's XMULT_CAP. Re-caps :x-multiplied
    # feature values to prevent weak SigLIP signal from being amplified to 0.999.
    XMULT_CAP = 0.1

    features_js = json.dumps(features)
    trees_js = json.dumps(trees, separators=(',', ':'))

    js = r"""// lgbm_evaluate_siglip — LGBM Underage Scorer (SigLIP image-only)
// Source: moderation-eval/face_age_sample/train_for_piper.py
// Hyperparams: """ + json.dumps(m["hyperparams"]) + r"""
// Trained on: """ + json.dumps(info) + r"""
// Threshold for blocking: """ + f"{thr}" + r"""
//
// Decisions are purely based on the SigLIP-2 image labels. NO prompt features.
// Output schema matches the upstream node so callers don't change:
//   outputs.labels  = ["underage"] | []
//   outputs.details = { score, blocked, threshold, top_features }

const LGBM_FEATURES = """ + features_js + r""";

const LGBM_TREES = """ + trees_js + r""";

const LGBM_THRESHOLD = """ + f"{thr}" + r""";

// ── Label taxonomy — body vs context vs interaction.
// Must stay aligned with moderation-eval/face_age_sample/label_taxonomy.py.
const BODY_LABELS = new Set([
  "baby","infant","toddler","toddler_literal","toddler_group","preschooler","kid",
  "child","small_child","little_boy","little_girl","young_boy","young_girl",
  "preteen_boy","preteen_girl","preteen_slim","preteen_model","preteen_smile",
  "tween","teen_girl","chubby_cheeks","child_proportions","smooth_child_skin",
  "child_genitals","girl_genitals","explicit_genitals","explicit_spread",
  "naked_child","naked_minor","naked_teen","flat_chest_child","flat_chest_nude",
  "flat_chest_teen","frail_body","fluids_face","cum_child_face","fluids_macro_face",
  "face_fluids_macro","juvenile_face_fluids","fluids_undeveloped_body",
  "fragile_crawling","fragile_boy_anatomy","petite_side_profile",
  "slender_childish_frame","slender_minor_group","youthful_face_flat_body",
  "explicit_expression_juvenile","lying_down_nude_girl","kneeling_submissive",
  "kneeling_teen_girl","intercourse_scene_with_girl","intercourse_boy_body",
  "two_girls_explicit","shota_bed_scene","boy_explicit_size","boy_macro_breasts",
  "boy_overpowered_breasts","macro_monster_anatomy","monster_explicit",
  "interracial_flat_anatomy","cosplay_minor_explicit","cosplay_bodysuit_minor",
  "girl_multiple_adults","anime_child","anime_loli","anime_explicit","3d_doll_child",
  "3d_doll_explicit","3d_preteen_model","explicit_3d_minor","digital_art_loli",
  "stylized_big_eyes","explicit_universal_anchor","universal_anchor",
  "group_naked_teens","bent_over_narrow_hips","bent_over_strict",
  "pov_face_close_up","pov_kneeling",
]);
const CONTEXT_LABELS = new Set([
  "school","kindergarten","playground","classroom","child_in_car","child_bedroom",
  "seatbelt","stroller","crib","sandbox","toys","pacifier","bathing_child",
  "child_swimsuit","child_swimwear_model","preteen_swimwear","girl_swimsuit_posing",
  "girl_bikini","flat_chest_swimwear_portrait","swimsuit","girl_in_dress",
  "girl_pretty_dress","minor_tight_dress","girl_portrait","girl_posing_photo",
  "preteen_posing","teen_selfie","child_portrait","teen_girl_casual",
  "teen_girl_sporty","teen_school","denim_casual_fashion","casual_bedroom_pose",
  "bedroom_pose","panties_casual","crop_top_panties","bare_legs_indoor",
  "fitness_body_underwear","childish_panties_bows","revealing_child_clothes",
  "cosplay","gymnastics_leotard","stripping_clothes","glasses_girl_nude",
  "glasses_child_person","teen_acne","pigtails","child_bangs","child_freckles",
  "curly_child","short_hair_boy","missing_teeth","bridal_child","group",
  "group_of_children","children_party","children_playing","classroom_children",
  "extreme_size_difference","giant_adult_small_child",
]);
const INTERACTION_LABELS = new Set([
  "adult_with_child","adult_with_girl","adult_with_boy","adult_child_together",
  "adult_child_sitting","man_with_young_girl","man_with_young_boy",
  "woman_with_young_girl","woman_with_young_boy","older_person_with_child",
  "maternal_contrast","boy_mature_woman",
]);

// ── LGBM PREDICTOR (identical to upstream) ────────────────────────────────────
function lgbmPredict(vec) {
  let score = 0.0;
  for (const t of LGBM_TREES) {
    let node = t.r;
    while (node >= 0) {
      const [fi, thr] = t.s[node];
      const [l, r] = t.c[node];
      node = vec[fi] <= thr ? l : r;
    }
    score += t.l[-(node + 1)];
  }
  return 1.0 / (1.0 + Math.exp(-score));
}

// ── Combine scores (noisy-OR over a key-predicate) ────────────────────────────
function combineScores(labels, matchFn) {
  let p = 1.0;
  for (const [k, v] of Object.entries(labels)) if (matchFn(k)) p *= 1 - v;
  return 1 - p;
}

// ── Build feature vector
// Convention: siglip underage labels named `<rest>`; siglip adult labels `adult__<rest>`;
// _minor / _adult / _confidence + _child_body / _child_context / _child_interaction /
// _body_vs_context derived aggregates.
function buildVec(affected) {
  const vec = new Array(LGBM_FEATURES.length).fill(0);
  const featIdx = {};
  LGBM_FEATURES.forEach((name, i) => { featIdx[name] = i; });

  const sigOnly = {};
  for (const [key, val] of Object.entries(affected)) {
    let fn;
    if (key.startsWith("underage_")) fn = key.slice(9);
    else if (key.startsWith("adult_")) fn = "adult__" + key.slice(6);
    else continue;
    sigOnly[fn] = val;
    if (fn in featIdx) vec[featIdx[fn]] = val;
  }

  const minor = combineScores(sigOnly, (k) => !k.startsWith("adult__"));
  const adult = combineScores(sigOnly, (k) => k.startsWith("adult__"));
  const total = minor + adult;
  const conf = total > 0 ? minor / total : 0;
  const body  = combineScores(sigOnly, (k) => BODY_LABELS.has(k));
  const ctx   = combineScores(sigOnly, (k) => CONTEXT_LABELS.has(k));
  const inter = combineScores(sigOnly, (k) => INTERACTION_LABELS.has(k));
  const bcTotal = body + ctx;
  const bodyVsCtx = bcTotal > 0 ? body / bcTotal : 0;
  if ("_minor"             in featIdx) vec[featIdx["_minor"]]             = minor;
  if ("_adult"             in featIdx) vec[featIdx["_adult"]]             = adult;
  if ("_confidence"        in featIdx) vec[featIdx["_confidence"]]        = conf;
  if ("_child_body"        in featIdx) vec[featIdx["_child_body"]]        = body;
  if ("_child_context"     in featIdx) vec[featIdx["_child_context"]]     = ctx;
  if ("_child_interaction" in featIdx) vec[featIdx["_child_interaction"]] = inter;
  if ("_body_vs_context"   in featIdx) vec[featIdx["_body_vs_context"]]   = bodyVsCtx;
  return vec;
}

// ── Main ──────────────────────────────────────────────────────────────────────
export async function run({ inputs }) {
  const { NextNode } = DEFINITIONS;

  const labels = inputs.labels || {};

  // Apply :xN multipliers (mirror evaluate_siglip's `affected`) BUT re-cap at
  // XMULT_CAP (0.1) instead of 0.999. Match training-side cap.
  const XMULT_CAP = """ + f"{XMULT_CAP}" + r""";
  const affected = {};
  for (const [key, val] of Object.entries(labels)) {
    if (typeof val !== "number" || val <= 0) continue;
    const m = key.match(/:x(\d+(?:\.\d+)?)$/);
    if (m) {
      affected[key.slice(0, m.index)] = Math.min(val * parseFloat(m[1]), XMULT_CAP);
    } else {
      affected[key] = val;
    }
  }

  const vec = buildVec(affected);
  const score = lgbmPredict(vec);
  const blocked = score >= LGBM_THRESHOLD;

  const topFeats = LGBM_FEATURES
    .map((name, i) => ({ name, val: vec[i] }))
    .filter(f => f.val > 0.001)
    .sort((a, b) => b.val - a.val)
    .slice(0, 6)
    .map(f => `${f.name}=${f.val.toFixed(3)}`)
    .join(", ");

  return NextNode.from({
    outputs: {
      labels:  blocked ? ["underage"] : [],
      details: {
        score:        Math.round(score * 100000) / 100000,
        blocked:      blocked,
        threshold:    LGBM_THRESHOLD,
        n_features:   LGBM_FEATURES.length,
        top_features: topFeats,
      },
    },
  });
}
"""
    out = ROOT / "piper_lgbm_evaluate_node.js"
    out.write_text(js)
    print(f"wrote {out}  ({len(js)} bytes / {len(js)/1024:.1f} KiB)")
    print(f"  LGBM_FEATURES: {len(features)}")
    print(f"  LGBM_TREES: {len(trees)}")
    print(f"  threshold: {thr}")


if __name__ == "__main__":
    main()
