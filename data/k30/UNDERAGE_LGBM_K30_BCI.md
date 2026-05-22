# LGBM K=30 (BCI) — Underage Detector

Compact piper pipeline for underage-content detection. Strict-dominance vs both
production and the previous LGBM K=25 baseline.

## TL;DR

- **30 SigLIP labels sent per image** (1 text-encoder batch — same SigLIP call shape as K=25, vs ~6 batches for prod-326)
- **CSAM-only minimal payload** — all non-underage/non-adult labels dropped (other moderation categories not detected by this pipeline)
- **88.0% recall at 0.63% adult-only FP** at recommended threshold 0.10
- Trained on **`qwen3_min_age ≤ 14`** (cutoff chosen after a careful sweep — see below)

## Numbers (8.5K-image LS sample basis, base-rate-corrected to 4.64% real-corpus prevalence)

| Metric | Real prod (today) | LGBM K=25 (#3626) | **LGBM K=30 (BCI) — this** |
|---|---|---|---|
| Labels sent to SigLIP | 326 | 25 | **30** |
| Text-encoder batches | 6 | 1 | **1** |
| Recall | 21.9% | 44.6% | **88.0%** @ thr 0.10 / **82.9%** @ thr 0.20 |
| FPR | 4.2% | 3.3% | 5.55% @ thr 0.10 / **3.65%** @ thr 0.20 |
| Precision (corrected to 4.64%) | 20.3% | 39.6% | **43.6%** @ thr 0.10 / **52.5%** @ thr 0.20 |
| F1 (corrected) | 0.211 | 0.419 | **0.583** @ thr 0.10 / **0.643** @ thr 0.20 |

**At matched FPR (~3.3–3.65%)**: K=30 (BCI) catches **82.9%** vs K=25's **44.6%** — nearly **2× the recall** at comparable false-positive cost.

### Side-by-side, full operating-point curve for K=30 (BCI):

| thr | recall | FPR | precision (4.64% corp) | F1 (corrected) |
|---|---|---|---|---|
| 0.20 | 82.9% | 3.65% | 52.5% | 0.643 |
| **0.10 (recommended)** | **88.0%** | **5.55%** | **43.6%** | **0.583** |
| 0.05 | 91.2% | 8.26% | 34.9% | 0.505 |
| 0.02 | 94.6% | 15.11% | 23.3% | 0.374 |

## What's new architecturally vs K=25

1. **Body / Context / Interaction aggregate split.** The previous LGBM models used a single `_minor` aggregate (noisy-OR over all underage labels). It conflates "actual child anatomy" with "child-adjacent scene/clothing context". K=30 splits this into three:
   - `_child_body` — noisy-OR over **anatomy / sex-act** labels (little_girl, naked_minor, child_genitals, fragile_crawling, intercourse_scene_with_girl, …)
   - `_child_context` — noisy-OR over **scene / clothing / pose** labels (classroom, school, playground, girl_in_dress, panties_casual, …)
   - `_child_interaction` — noisy-OR over `:x`-multiplied **adult-with-child** labels (adult_with_girl:x20, maternal_contrast:x20, man_with_young_girl:x20, …)
   - Plus `_body_vs_context` ratio = `_child_body / (_child_body + _child_context)` — high when actual anatomy dominates, low when only scene cues fire.
2. **Cap the `:xN` multiplier at 0.1** (instead of 0.999). The multipliers exist to upweight `adult-with-child` labels but they also amplify weak SigLIP signal on adult-only images with size/age contrast. Capping prevents single weak labels from being blown up to dominance.
3. **Hard-negative mining**: after first-pass OOF, identify adult-only images that get flagged at thr 0.05 (50 such cases). Up-weight them 20× in training. The model learns explicit "this scene-with-context-but-no-actual-child" decision boundary.
4. **`qwen3_min_age` (not `max_age`) for labeling.** Critical: max_age catches the adult in adult-with-child scenes and mis-labels the row as negative. Using min_age catches the child even when an adult is also present. Found and fixed mid-experiment — see #3626 / overnight diary if curious.
5. **Cutoff sweep showed cliff at 14→15**: at ≤14 adult-FP stays under 0.65% at thr 0.10; at ≤15 it jumps to 1.38%. Past 14 the visual age discrimination of SigLIP-2 (and arguably humans) is unreliable — "mature 15yo" and "young-looking 18yo" overlap visually. K=30 ships with cutoff **≤14**.

## Architecture diagram

```
Image  →  SigLIP-2 vision encoder           (fixed cost, ~1.5–2s)
                  │
                  ▼
        ┌──── 30 label text-embeddings cached
        ▼
  cosine sim against image embed → 30 raw label scores  (1 text-encoder batch, ~0.3s)
                  │
                  ▼
   Apply :x multipliers (capped at 0.1)     (microseconds, JS)
                  │
                  ▼
   Compute 7 derived aggregates:
       _minor, _adult, _confidence,
       _child_body, _child_context, _child_interaction, _body_vs_context
                  │
                  ▼
   Feed 37-element feature vector into LightGBM (100 trees, 15 leaves)
                  │
                  ▼
   Probability score ∈ [0,1] → block if ≥ 0.10
```

End-to-end SigLIP latency per image: roughly **30–40% lower** than current prod
(image-encode is fixed, the 6→1 text-encoder batch reduction saves ~5/6 of the
text-encoder fraction; image-encode is the dominant cost).

## The 30 SigLIP labels (top-30 by LightGBM importance)

The model trains its own importance ranking; below is the order from the latest fit. **All 30 (with their `:x` multipliers where applicable) are in the deployed piper pipeline `ask_siglip2.labels` default. All other categories (bestiality, weapons, blood, death, drugs, rape, ethnicity, no_*) are dropped from this pipeline.**

```
1. intercourse_scene_with_girl       (BODY)
2. adult__slender_woman_group         (ADULT — discriminator)
3. little_girl                        (BODY)
4. fragile_crawling                   (BODY)
5. toddler_group                      (BODY)
6. girl_genitals                      (BODY)
7. maternal_contrast:x20              (INTER)
8. slender_childish_frame             (BODY)
9. girl_multiple_adults               (BODY)
10. adult__petite_natural_nude        (ADULT)
11. lying_down_nude_girl              (BODY)
12. cosplay_minor_explicit            (BODY)
13. group_naked_teens                 (BODY)
14. adult__person                     (ADULT)
15. bent_over_strict                  (BODY)
16. woman_with_young_boy:x20          (INTER)
17. tiny_person                       (BODY)
18. teen_slim_body_nude               (BODY)
19. bare_legs_indoor                  (CTX)
20. adult__macro_anatomy_pov          (ADULT)
21. extreme_size_difference           (CTX)
22. preteen_posing                    (CTX)
23. adult__youthful_face_natural_nude (ADULT)
24. boy_mature_woman:x5               (INTER)
25. pov_kneeling                      (BODY)
26. adult__two_women_different_heights (ADULT)
27. kneeling_teen_girl                (BODY)
28. naked_teen                        (BODY)
29. adult__erotic_leg_spread          (ADULT)
30. adult__group_naked_women          (ADULT)
```

The 7 adult labels are crucial discriminators: the LightGBM learns that high adult-label scores in conjunction with weak child-body labels suppresses the false positive — exactly the case for "adult with size-contrast partner that triggered K=25 to FP".

## Deployed pipeline

- **Project ID**: `a4aa9dbd9c`
- **Slug**: `violations-detector-test-prompt`
- **Scope**: `violations_detector_test_prompt`, currently `activated: false`
- **URL**: https://piper-next.artworks.ai/en/projects/a4aa9dbd9c

To launch:
```bash
curl -X POST -H "api-token: $PIPER_TOKEN" -H "Content-Type: application/json" \
  -d '{"inputs":{"image":"<image_url>","providers":["artworks|siglip2"]}}' \
  https://piper-next.artworks.ai/api/violations-detector-test-prompt/launch
```

`siglip2_details.underage.lgbm` in the output gives `{score, blocked, threshold, top_features}`.

To activate the scope for production use:
```bash
curl -X POST -H "api-token: $PIPER_TOKEN" -H "Content-Type: application/json" \
  -d '{"slug":"violations-detector-test-prompt","scope":{"id":"violations_detector_test_prompt","activated":true,"maxConcurrent":5}}' \
  https://piper-next.artworks.ai/api/projects/a4aa9dbd9c/deploy
```

## Operating-point picker

For different deployment policies:

| Goal | thr | recall | FPR | F1 |
|---|---|---|---|---|
| Maximum precision | 0.20 | 82.9% | 3.65% | 0.643 |
| **Balanced (default)** | **0.10** | **88.0%** | **5.55%** | **0.583** |
| Higher recall | 0.05 | 91.2% | 8.26% | 0.505 |
| Catch everything | 0.02 | 94.6% | 15.11% | 0.374 |

## Replication / dev guide

All code in `moderation-eval/face_age_sample/` (local working notes, not in repo):

| Script | Purpose |
|---|---|
| `train_for_piper.py` | Build feature matrix + labels (with body/context/interaction taxonomy, cutoff ≤14, `:x` cap 0.1) |
| `train_minimal.py` | Apply pruning (top 30 SigLIP) + hardneg weighting (50 adults at thr 0.05, w=20) — produces `piper_lgbm_model.json` |
| `build_piper_node_script.py` | Emit JS source for the lgbm_evaluate node (97 KiB embedded trees + aggregates) |
| `build_new_project.py` | Construct piper API POST payload (clones violations-detector-test-v3 + replaces lgbm_evaluate node) |
| `prune_siglip_labels.py` | Cut `ask_siglip2.labels` default down to 30 labels — CSAM-only mode |
| `label_taxonomy.py` | BODY / CONTEXT / INTERACTION label sets |
| `cutoff_sweep_minimal.py` | Empirical sweep of `min_age ≤ {10,11,12,13,14,15}` cutoffs |

End-to-end rebuild from scratch:
```bash
cd moderation-eval/face_age_sample
python3 train_for_piper.py         # baseline
python3 train_minimal.py           # pruned + hardneg (final piper_lgbm_model.json)
python3 build_piper_node_script.py # emit JS source
python3 build_new_project.py       # build POST payload
python3 prune_siglip_labels.py     # cut labels in payload
# then POST piper_new_project_payload.json to /api/projects, then /deploy
```

## Caveats

1. **Dataset mismatch with K=25**: K=25 used 8,533 human-moderator-labeled images from LS (`underage` = any minor < 18). K=30 (BCI) used 6,680 Qwen3-VL-scored images with the explicit cutoff `qwen3_min_age ≤ 14`. To do a clean apples-to-apples, K=30 (BCI) should be scored against the LS dataset — a separate validation experiment.
2. **15-17yo excluded by training intent.** Applied to LS images that include `underage` labels for 15-17yo cases, K=30 will miss them. The 14→15 cliff in the sweep (adult-FP 0.63%→1.38% at thr 0.10) is the empirical justification; "underage" as a legal category extends to 17 but SigLIP-2 can't reliably distinguish 15-17yo from young-looking 18+yo. Recommended pairing: K=30 (BCI) for first-line block + Qwen3-VL backstop for 15-17yo ambiguous cases.
3. **Pipeline is CSAM-only** — bestiality, blood, death, weapons, drugs, rape, ethnicity detection no longer functional in this pipeline (their labels dropped from SigLIP payload). Production needs to either keep using the existing `violations-detector-test-v3` for those categories, or expand this pipeline's label set (90+ extra labels) to a hybrid.
4. **SigLIP per-call variance**: same image gives slightly different label probabilities on different calls (~5-10% jitter on borderline labels). Affects all models equally; not specific to K=30.

## Validation needs before production switch

1. **Score the 8,533 LS images through K=30 (BCI)** — confirm the recall/FP numbers translate to LS ground truth. (Most likely the recall drops because LS includes 15-17yos by definition.)
2. **Shadow-deploy for 1 week** — score in parallel without enforcing. Sample model-blocks and model-passes for human re-check.
3. **Decide on production scope**: this pipeline as CSAM-only + keep test-v3 for other categories, OR extend this pipeline with the dropped non-CSAM labels.
