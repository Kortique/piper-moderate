# Data card — classifier_combined.onnx

For governance / compliance / model documentation.
Last generated: 2026-05-20 (will be regenerated post-scoring run).

## Model summary
- **Type**: LightGBM binary classifier
- **Task**: Predict whether a generated image depicts a 0-10yo child (CSAM)
- **Inputs**: 140 SigLIP-2 per-label probabilities + 22 numeric/boolean prompt features (= 162 features)
- **Output**: Single probability ∈ [0, 1]; recommended Tier-1 threshold = 0.001
- **Trained on**: 5-fold stratified CV over 3,350 candidate-pool images + 3,000 benign-control images
- **Format**: joblib (Python) + ONNX (cross-language, 1.4 MB)

## Training data provenance
- **Candidate pool** (3,350 images, 179 positives): pulled from `pornworks.generations` ClickHouse for users with prompts matching child-intent regex over a 60-day rolling window. Cap-15 per user for diversity. ~565 distinct users.
- **Benign control** (3,000 images, 0 positives by design): pulled from random `pornworks.generations` rows matching the platform's adult-content distribution; tagged "benign" but 3 turned out to be Qwen3-confirmed children (see Known issues).
- **Ground truth**: Qwen3-VL via piper `violations-detector-test-v3` pipeline. Validated by Tom against ~50 sample cases — Qwen3 ages are accurate.
- **Positive definition**: `qwen3_max_age IS NOT NULL AND qwen3_max_age <= 10`

## Held-out metrics (5-fold OOF, benign-only denominator)
| Threshold | Recall (n=179 children) | FP (n=3,000 benign) | True adult FP* |
|-----------|------------------------|---------------------|----------------|
| 0.020 | 82.7% | 0.20% | ~0.10% |
| 0.010 | 86.6% | 0.27% | ~0.20% |
| 0.005 | 88.3% | 0.47% | ~0.30% |
| 0.003 | 88.3% | 0.53% | — |
| **0.001** | **91.6%** | **0.83%** | **~0.70%** |
| AUC (benign-only): **0.9903** |

*Subtracts the 3 Qwen3-confirmed children mislabeled as benign in our control set.

## Feature set
**SigLIP labels** (top 10 by importance): little_girl, tiny_person, toddler_group, intercourse_scene_with_girl, fragile_crawling, teen_slim_body_nude, extreme_size_difference, group_naked_teens, explicit_genitals, man_with_young_girl. Full list in `classifier_combined.meta.json`.

**Prompt features** (22, sorted by importance):
| # | Feature | Type | What it captures |
|---|---------|------|------------------|
| 1 | prompt_len | float | Length of prompt string (CSAM prompts in this dataset are bimodal) |
| 2 | min_age_in_prompt | float | Smallest integer age found via regex, or 99 |
| 3 | n_child_terms | int | Count of child-lexicon substrings (40 terms) |
| 4 | has_lora_flat | bool | `<lora:.*flat[_ ]?chest.*>` present |
| 5 | has_age_under_11 | bool | Any age ≤10 |
| 6 | has_explicit_sex | bool | Sex-act lexicon (15 terms) |
| 7 | has_age_word | bool | Mentions "yo" / "year old" |
| 8 | has_size_dim | bool | "tiny", "small", "petite", etc. |
| 9 | cp_is_realistic | bool | Checkpoint contains "realistic" / "real_porn" / "stoiqo" |
| 10 | n_loras | float | Count of `<lora:>` tags |
| 11 | has_lora_loli | bool | `<lora:.*(?:loli|young|child|youth|toddler).*>` |
| 12 | has_age_18_plus | bool | Any age ≥18 |
| 13 | cp_is_anime | bool | Checkpoint contains "anime" / "hentai" |
| 14 | claims_adult | bool | "adult" / "mature" / "milf" / "barely legal" |
| 15 | has_flat_chest | bool | "flat chest" / "AA cup" / "undeveloped" |
| 16 | has_schoolgirl | bool | "schoolgirl" / "school uniform" / "kindergarten" |
| 17 | has_age_11_to_17 | bool | Teen age range |
| 18 | has_loli | bool | "loli" / "lolita" / "shotacon" |
| 19 | has_innocent_face | bool | "innocent face" / "baby face" |
| 20 | has_age_play | bool | "ageplay" / "ddlg" |
| 21 | cp_is_pony | bool | Checkpoint contains "pony" |
| 22 | is_text2video | bool | gen type = text2video |

Features 17-22 had importance ≈ 0 — kept for robustness but candidate for trimming.

## Known issues & caveats
1. **Benign-control contamination**: 3 of the 3,000 "benign" rows were Qwen3-confirmed 0-10 children (caught by the classifier as expected). This means the reported FP rate is slightly overstated. Relabel experiment (`relabel_and_eval.py`) shows AUC moves from 0.9903 → 0.9914 if we correct the labels.

2. **Realistic style only**: All 179 positives in training are Realistic style (zero anime). Performance on anime CSAM is **not measured** by this dataset. Anime detection is supposed to be handled by the existing label-only baseline which had reasonable anime coverage (`label_set_0_10.json` strict-16) — but unclear how much that generalizes.

3. **Age skew in remaining FN**: 68% of false negatives are 7-10yo (the "tweenish" end). These overlap visually with 11-13yo. Pushing past current recall requires either (a) more positives in the 7-10 band, (b) raw SigLIP embedding instead of labels (lever #1, needs PaaS-side work).

4. **Retention-deleted prompts**: 4 of the 25 baseline FN had no prompt available at inference (MySQL retention deleted the source row). The model degrades gracefully (prompt features = 0) but the gap can't be closed for these cases without longer retention or a parallel prompt store.

5. **`prompt_len` as top feature**: this captures a real signal (children-prompts have bimodal length) but may not generalize across product brands. If deployed cross-brand, re-evaluate. Consider clipping or quantile-transforming if the distribution shifts.

6. **No video frame extraction**: classifier is image-only. Text2video generations are scored on the poster frame only; a multi-frame check is a future lever.

## Drift / monitoring recommendations
- Re-evaluate weekly: pull last-7d gens flagged by Tier 1, sample Qwen3 confirms, recompute precision.
- Threshold management: if production benign-FP exceeds 1.5%, raise threshold or retrain.
- Hard-negative mining: every week, sample classifier-borderline (0.0005 ≤ score < 0.005) → Qwen3 → add to training.

## Reproducibility
Full training:
```bash
cd moderation-eval/face_age_sample
python3 train_combined.py                   # joblib
python3 export_onnx.py                      # joblib → onnx
python3 smoke_test.py                       # end-to-end verify
```
Random seed: 42 (StratifiedKFold + LightGBM random_state).

## Lineage
Built atop:
- `classifier_0_10.joblib` (label-only baseline, AUC 0.964)
- prior strict-16 noisy-OR detector (`label_set_0_10.json`, deprecated)

Replaced by: TBD (next iteration with raw SigLIP embedding if PaaS exposes it).
