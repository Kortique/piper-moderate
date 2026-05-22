# 3-way comparison — production vs other-team's WIP vs ours

Same eval set (6,680 rows, 477 children using `qwen3_min_age <= 10`).

## TL;DR

| Model | AUC | Best deployable operating point |
|---|---|---|
| **A. detect.ts noisy-OR** (current production) | 0.8998 | **44.7% recall / 6.93% FP** |
| **B. existing piper LGBM** in `test-v3` (other team, not activated) | 0.9786 | 93.1% / 7.00% |
| **C. ours** (SigLIP-only, `violations-detector-test-prompt`) | **0.9877** | **93.1% / 1.40%** or **90.8% / 1.10%** |

## Production is missing more than half of CSAM

The `detect.ts` decision rule:
```ts
risk >= 1.0  // never
|| (riskScoreInTotal >= 0.85 && risk >= 0.01)
```

Where `riskScoreInTotal = risk / (risk + adult)` — the ratio of underage signal to combined underage+adult signal.

For "adult man with young girl" scenes, BOTH underage_* and adult_* labels fire strongly. The ratio is then around 0.3-0.6, well below 0.85. Production never blocks these.

That's why production catches only 44.7% of confirmed CSAM in our eval — it's biased to block ONLY "pure child" scenes and lets "adult with child" through. 273 of our 477 positives (57%) are adult+child cases.

## How each model is constructed

### A. detect.ts (`backend/utils/moderation/detect.ts`)
- Compute noisy-OR over all `underage_*` SigLIP labels = `minor_risk`
- Compute noisy-OR over all `adult_*` SigLIP labels = `adult_conf`
- Block if `minor_risk / (minor_risk + adult_conf) ≥ 0.85` AND `minor_risk ≥ 0.01`
- No learned model. Just two aggregates and a ratio threshold.

### B. existing piper LGBM (`violations-detector-test-v3`)
- 98 features (all SigLIP labels, with `:x` suffix preserved as feature names)
- Trained on 839 samples (per source comment)
- 100 trees, 15 leaves
- Single hardcoded threshold 0.85
- No prompt features

### C. ours (`violations-detector-test-prompt`)
- 317 features (313 SigLIP labels + 4 derived: `_child_body`, `_child_context`, `_child_interaction`, `_body_vs_context` + `_minor`, `_adult`, `_confidence`)
- Trained on 6,680 samples (8× more)
- 100 trees, 15 leaves (matching their hyperparams for fair comparison)
- Tunable threshold; default 0.05 → 93.1% / 1.40%
- **No prompt features** (removed 2026-05-21 after Tom pointed out the architectural concern — they were hand-written regex with no semantic understanding, defeated by leet-speak, didn't add meaningful recall once labeling was correct)
- `:x` multipliers capped at 0.1 (instead of 0.999) to prevent weak SigLIP signal from being amplified to dominance
- Labels partitioned into BODY (anatomy/sex acts), CONTEXT (clothing/scene/pose), INTERACTION (`:x` adult+child) so the model can distinguish "actual child in image" from "child-adjacent context"

## Full operating-point tables

### A. detect.ts (production)
| thr | recall | FP | precision |
|---|---|---|---|
| 0.85 (production setting) | **44.7%** | **6.93%** | 50.6% |

(Other rows in `three_way_compare.py` output are NOT real production states — production has a single binary decision.)

### B. existing piper LGBM
| thr | recall | FP | precision |
|---|---|---|---|
| 0.85 (default) | 93.1% | 7.00% | 67.9% |
| 0.50 | 95.6% | 11.20% | 57.6% |
| 0.30 | 96.0% | 14.20% | 51.8% |
| 0.10 | 97.5% | 20.83% | 42.7% |

### C. ours (SigLIP-only)
| thr | recall | FP | precision |
|---|---|---|---|
| 0.20 | 84.3% | 0.70% | 95.0% |
| **0.10** | **89.5%** | **1.00%** | **93.4%** |
| 0.075 | 90.8% | 1.10% | 92.9% |
| **0.05** | **93.1%** | **1.40%** | **91.4%** |
| 0.02 | 94.8% | 2.33% | 86.6% |
| 0.01 | 96.0% | 3.53% | 81.2% |
| 0.005 | 96.6% | 7.17% | 68.2% |

## Per-image score artifact

`three_way_results.json` — for each of the 6,680 rows, the three model scores. Use for further analysis.

## Recommendation

Switch from A (production) to C (ours). Recommended threshold **0.05** → 93.1% recall, 1.40% FP, 91.4% precision. Roughly **doubles CSAM recall** vs production while **cutting false-positive rate in 5×**.

If conservative: **0.10** → 89.5% / 1.00% / 93.4% precision. Still 2× the recall of production at 7× lower FP.

## Limitations carried over

1. **The 12 confirmed FPs Tom flagged**: ~7 still fire at thr 0.05; ~4 at thr 0.075; ~2 at thr 0.10. Closing these would need hard-negative mining of size-contrast adult cases.
2. **SigLIP non-determinism**: same image scored twice gives slightly different raw label probabilities (affects all three models equally).
3. **All 477 positives are Realistic style** — anime CSAM detection not directly tested.
4. **204 of 477 are pure-child; 273 are adult+child** — the body/context/interaction split is key for distinguishing these.
