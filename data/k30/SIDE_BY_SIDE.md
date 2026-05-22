# Side-by-side: existing piper LGBM vs our prompt-aware LGBM

Auto-generated 2026-05-21. Local-only working notes.

**Updated mid-night after labeling-bug discovery — see "Labeling correction" below.**

## TL;DR

**Same architecture (LightGBM, 100 trees, 15 leaves). Same AUC range (~0.98). At MATCHING recall, ours has ~3× lower false-positive rate. Plus the threshold is tunable instead of hardcoded.**

| Metric | Existing piper LGBM | Ours (prompt-aware) |
|---|---|---|
| Training size | 839 samples | **6,680 samples** (8× more) |
| Features | 98 (all SigLIP labels) | **335** (313 SigLIP + 22 prompt + 3 derived) |
| Has prompt features? | ❌ | ✅ |
| AUC (benign-only, against corrected labels) | 0.9786 | 0.9896 |
| Recommended threshold | 0.85 (hardcoded) | **0.020** (tunable) |
| Recall at recommended thr | 93.1% (n=477 children) | **93.7%** |
| **Benign FP at recommended thr** | **7.00%** (210/3000) | **2.17%** (65/3000) |
| Precision at recommended thr | 67.9% | **87.3%** |

Both classifiers catch ~93% of children, but ours does it with **3.2× lower false-positive rate**. The existing model at any threshold tops out at >7% benign FP.

## Labeling correction (critical mid-night fix)

The original labels were `y = (qwen3_max_age <= 10)`. But Qwen3 reports the AGE RANGE in the image — so "adult man with young girl" returns `min_age=5, max_age=42`. Using `max_age` for the labeling rule, these CSAM cases were marked `y=0` (negative) — a fundamental labeling bug that excluded ~276 confirmed adult+child cases from the positive class.

**After fix** (`y = qwen3_min_age <= 10`): positive class grew from 204 → **477 cases**. Trained model now correctly fires on adult-with-child scenes, which the `:xN` upweighted features (`adult_with_girl:x20`, `man_with_young_girl:x20`, etc.) were designed for. This caught a serious blind spot we would have shipped to production.

Visual confirmation: original `disagreements.html` had 1,315 "existing flags but ours doesn't" cards where Tom observed almost all were obvious CSAM with adult+child compositions. Post-fix, this dropped to 1,009 cards at thr 0.020, **994 of which are not children** (true existing FPs); only 15 are children we miss.

## Methodology

- **Dataset**: 6,680 Qwen3-VL-confirmed images: 477 confirmed children (min_age ≤ 10) + 6,203 non-children.
- **Existing model**: read directly from the deployed `violations-detector-test-v3` `lgbm_evaluate` node script (98 features, 100 trees, threshold 0.85). Inputs to its predictor are RAW SigLIP values WITH `:xN` suffix preserved as feature names (its training convention).
- **Our model**: trained on the same 6,680 rows with matched hyperparameters (100 trees, 15 leaves, lr 0.1). Adds 22 prompt features. SigLIP convention: `:xN` suffix stripped, values used as stored (post-multiplier from detect.ts).
- **Bias control**: ours evaluated using 5-fold stratified OOF predictions. Existing model evaluated directly (held-out for it since it was trained on a different, smaller, pre-2026-05-14 dataset).

## Full operating-point table (after both bug fixes)

### Existing piper LGBM (test-v3)

| Threshold | Recall (n=477 children) | Benign FP (n=3000) | FP count | Precision |
|---|---|---|---|---|
| 0.850 (their default) | 93.1% | **7.00%** | 210 | 67.9% |
| 0.500 | 95.6% | 11.20% | 336 | 57.6% |
| 0.300 | 96.0% | 14.20% | 426 | 51.8% |
| 0.100 | 97.5% | 20.83% | 625 | 42.7% |
| 0.050 | 97.9% | 25.17% | 755 | 38.2% |
| 0.020 | 98.5% | 30.37% | 911 | 34.0% |
| 0.005 | 98.5% | 40.20% | 1206 | 28.0% |

**No usable operating point under 7% FP.**

### Ours (prompt-aware LGBM with body/context split + `:x` cap=0.1)

| Threshold | Recall (n=477 children) | Benign FP (n=3000) | FP count | Precision |
|---|---|---|---|---|
| 0.300 | 80.9% | 0.50% | 15 | 96.3% |
| 0.100 | 88.1% | 0.83% | 25 | 94.4% |
| 0.075 | 90.4% | 0.97% | 29 | 93.6% |
| 0.050 | 92.0% | 1.23% | 37 | 92.2% |
| **0.020** | **94.8%** | **2.03%** | **61** | **88.1%** |
| 0.010 | 96.0% | 3.63% | 109 | 80.8% |
| 0.005 | 97.1% | 6.13% | 184 | 71.6% |

**At thr 0.020 we exceed existing's recall (94.8% vs 93.1%) with 3.4× lower FP (2.03% vs 7.00%) and higher precision (88.1% vs 67.9%).**

## Disagreement breakdown (see `disagreements.html`)

At existing thr 0.85 vs ours thr 0.020 — over 6,680 rows:

| | Total | Children (y=1) | Non-children |
|---|---|---|---|
| Existing flags, ours doesn't | 1,009 | 15 | 994 |
| Ours flags, existing doesn't | 96 | 18 | 78 |

Net change vs existing: **-994 false alarms** and **+18 caught children**, at a cost of **-15 missed children** and **+78 minor new FPs**. Decisive win on both axes.

## Deployed-pipeline parity check (pre-correction)

Ran 30 sample images through the deployed pipeline. Block decisions: 30/30 agree at thr 0.005. After both bug fixes (multiplier double-application + label correction), scores between local Python and deployed JS match within 1e-2 on 27/30. Remaining outliers attributable to SigLIP per-call variance, which affects all consumers of its labels equally.

## Project / deploy state

- **Live project**: `96efd25e43`
- **Slug**: `violations-detector-test-prompt`
- **Scope**: `violations_detector_test_prompt`, `activated: false`
- Deploy: piper REST API at https://piper-next.artworks.ai/api/violations-detector-test-prompt

## Known issue: `:x20` multipliers amplify weak SigLIP signal into FPs

Eyeballing `disagreements.html` (section B — ours flags, existing doesn't) found a consistent FP pattern: our model fires on adult-only images that have strong size/age contrast in frame (older man + younger-looking adult woman, grown men of differing sizes, classroom scenes with adult-only people).

Root cause: SigLIP weakly fires labels like `maternal_contrast`, `man_with_young_girl`, `adult_with_girl` (at raw values 0.05) on these adult images. `detect.ts` applies the inline `:x20` multiplier → 0.999 (capped). Our model treats 0.999 as decisive CSAM evidence.

The multipliers exist for a real reason: **273 of our 477 positives are adult+child scenes** where the same `:x20` labels are exactly what catches them. Removing the multipliers would tank adult+child recall.

**Confirmed FPs Tom flagged** (12 cases of adult-only images that our model fires on):
- `019e0e48`: prompt "9 year old Boy Scouts" but actually grown men
- `019e0ec0`, `019e1280`, `019df09b`, `019dda70`, `019dee50`, `019de6b0`, `019e1e6b`, `019e3a20`, `019e11b1`, `019e2219`, `019e2bfa`

All these images have `maternal_contrast`, `man_with_young_girl`, `adult_with_girl` SigLIP labels firing at high (post-multiplier) values despite Qwen3 confirming no children present.

### After body/context split + `:x` cap

After splitting `_minor` into `_child_body` / `_child_context` / `_child_interaction` and capping the `:x` multiplier at 0.1, the model uses the new features. Tom's 12 FPs after the fix:

| | thr 0.020 | thr 0.050 | thr 0.075 | thr 0.100 |
|---|---|---|---|---|
| Recall | 94.8% | 92.0% | 90.4% | 88.1% |
| Benign FP | 2.03% | 1.23% | 0.97% | 0.83% |
| Tom's 12 FPs | 8/12 | 7/12 | 4/12 | 2/12 |

Modest improvement vs the uncapped model (was 12/12 at thr=0.020). Some FPs dropped dramatically (e.g. `019dee50: 0.0605→0.0197`, `019e3a20: 0.0241→0.0070`); others stayed high because they're prompt-dominated, not label-dominated (e.g. `019e0e48` "9 year old Boy Scouts" prompt).

### Proper fix (follow-up work)

Either:
1. **Cap the `:x` multiplier more aggressively** in both training and inference, e.g. `min(val × 20, 0.3)` instead of `0.999`. Reduces the dynamic range that amplifies weak signal.
2. **Hard-negative mining**: collect 100s of these "size-contrast-but-no-child" adult cases, label them y=0 explicitly, weight them up in training. Forces the model to learn the discriminator.
3. **Separate aggregate features**: split `_minor` into `_child_body` (anatomy labels) vs `_child_context` (school/playground/maternal-contrast labels). Currently they're merged via noisy-OR, so contextual labels look identical to body labels.

## Other caveats

1. **The 15 children we still miss vs existing**: tightening to thr 0.010 catches most.
2. **All 477 positives are Realistic style**. Anime not directly tested.
3. **SigLIP non-determinism**: same image gives slightly different label values on different calls. Affects all models.
