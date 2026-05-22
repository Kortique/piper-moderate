# moderation-eval-k30

SigLIP-2 + LightGBM **K=30 (BCI)** underage-content detector. Dataset, training scripts, trained model, comparison results.

Companion to GitLab issue [#3626](https://gitlab.artworks.ai/realistic-ai/fullstack/-/issues/3626).

> ⚠️ **Confidential.** This repo contains image URLs to CSAM-candidate generations, real user IDs, user prompts, and Qwen3-VL visual descriptions. Internal-only. Do not export, mirror, or reference outside artworks.

> 🏷️ **Human labeling in progress** at Label Studio [project 5](https://ls.artworks.ai/projects/5) — see [`LABEL_STUDIO.md`](LABEL_STUDIO.md). Images are mirrored at `https://fsn1.your-objectstorage.com/artworks-assets/g-<gid>.webp` (WEBP q=96) for fast UI loading.

## Headline result

Apples-to-apples on this dataset (6,680 Qwen3-scored images, ground truth `qwen3_min_age ≤ 14`):

| Model | Recall | FPR | Adult FP count | AUC |
|---|---|---|---|---|
| Production `detect.ts` noisy-OR | 44.1% | 15.4% | 301 / 3,832 | 0.81 |
| LGBM K=25 (deployed in `violations-detector-test-v3`) @ thr 0.85 | 90.3% | 19.0% | 218 / 3,832 | 0.93 |
| **LGBM K=30 (BCI) @ thr 0.10** | **88.0%** | **5.55%** | **24 / 3,832** | **0.97** |
| **LGBM K=30 (BCI) @ thr 0.20** | 82.9% | **3.65%** | **11 / 3,832** | 0.97 |

At comparable recall (~88–90%), K=30 (BCI) has **~9× fewer adult false positives** than K=25. K=25 cannot reach K=30's FPR at any threshold.

## What's in here

```
README.md                       ← you are here
UNDERAGE_LGBM_K30_BCI.md        ← main dev doc — architecture, 30-label list, deployment
LABEL_STUDIO.md                 ← Label Studio project 5 — labeling UI + export instructions
DATA_CARD.md                    ← dataset description (sources, fields, sampling)
THREE_WAY_COMPARISON.md         ← analysis writeup: prod vs K=25 vs K=30
SIDE_BY_SIDE.md                 ← K=25 vs K=30 head-to-head
MORNING_2026-05-21.md           ← full chronological log of how we got here

ls/                             ← scripts that moved the dataset into Label Studio
  move_to_label_studio.py         download → WEBP → fsn1 S3 → LS import (initial pipeline)
  reencode_q96.py                 re-encode all 6,675 webps at q=96 to match existing media
  import_to_project5.py           move tasks to dedicated project 5
  delete_from_project3.py         remove tasks from the original Gallery NSFW project

Data (sqlite):
  scored.sqlite                 ← 3,680 CSAM-candidate generations (Qwen3-VL + SigLIP-2 scored)
  scored_benign.sqlite          ← 3,000 confirmed-adult negatives (Qwen3-VL + SigLIP-2 scored)
  prompts.sqlite                ← prompt source corpus (~8MB)

Training / scoring scripts:
  pull_sample.py                ← assemble candidate set from MySQL (replica DB)
  build_dataset.py              ← curate + ground-truth labeling pipeline
  score_dataset.py              ← run Qwen3-VL + SigLIP-2 over a URL list
  label_taxonomy.py             ← BODY / CONTEXT / INTERACTION label sets
  train_for_piper.py            ← baseline LGBM training pipeline (cutoff ≤14, :x cap 0.1)
  train_minimal.py              ← final pruned + hardneg model → piper_lgbm_model.json
  cutoff_sweep_minimal.py       ← cutoff sweep (≤13 / ≤14 / ≤15) that justified ≤14
  honest_comparison.py          ← apples-to-apples 3-way comparison (table above)
  three_way_compare.py          ← richer 3-way (per-image scores in three_way_results.json)
  side_by_side_compare.py       ← K=25 vs K=30 head-to-head

Piper pipeline build:
  build_piper_node_script.py    ← emit lgbm_evaluate JS node from piper_lgbm_model.json
  build_new_project.py          ← construct piper API POST payload (clone + replace lgbm node)
  prune_siglip_labels.py        ← cut ask_siglip2.labels default to CSAM-only (30 labels)
  piper_parity_check.py         ← launch piper N times and check JS == Python predictions

Artifacts:
  piper_lgbm_model.json         ← trained LGBM (100 trees, 15 leaves) — the deployed model
  feature_keepers.json          ← 30 SigLIP labels + 7 derived features
  three_way_results.json        ← per-image scores for all 3 models (1.4 MB)
  side_by_side_results.json     ← per-image K=25 vs K=30 (1.4 MB)
```

## Quickstart — reproduce the apples-to-apples table

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install lightgbm scikit-learn numpy
python3 honest_comparison.py
```

Reads `scored.sqlite` + `scored_benign.sqlite`, prints the metrics table verbatim.

## Retrain from scratch

```bash
python3 train_for_piper.py      # baseline training (writes to stdout)
python3 train_minimal.py        # apply pruning + hardneg → piper_lgbm_model.json
python3 build_piper_node_script.py  # emit JS source for lgbm_evaluate node
```

## Deployed piper pipeline

- **Project**: [violations-detector-test-prompt](https://piper-next.artworks.ai/en/projects/a4aa9dbd9c) (project `a4aa9dbd9c`)
- **Scope**: `violations_detector_test_prompt`, currently `activated: false` (not routing prod traffic)
- **Launch**:
  ```bash
  export PIPER_TOKEN=...
  curl -X POST -H "api-token: $PIPER_TOKEN" -H "Content-Type: application/json" \
    -d '{"inputs":{"image":"<image_url>","providers":["artworks|siglip2"]}}' \
    https://piper-next.artworks.ai/api/violations-detector-test-prompt/launch
  ```
- Block decision: `siglip2_details.underage.lgbm.blocked` (boolean); raw score in `.score`; threshold in `.threshold`.

## Architecture (1-min version)

```
Image → SigLIP-2 (1 text-encoder batch, 30 labels) → raw label scores
  → Apply :x multipliers (capped at 0.1)
  → Compute 7 aggregates: _minor, _adult, _confidence,
                          _child_body, _child_context, _child_interaction,
                          _body_vs_context
  → 37-element feature vector → LightGBM (100 trees, 15 leaves) → P(underage)
  → block if P ≥ 0.10
```

The **BCI split** (Body / Context / Interaction) is the key change vs K=25: the model can now separate "actual child anatomy" signal from "child-adjacent scene/clothing" signal and adjust accordingly. See `UNDERAGE_LGBM_K30_BCI.md` § "What's new architecturally vs K=25".

## Caveats

1. **K=30 (BCI) is trained with cutoff ≤14 only.** Empirical 14→15 cliff: adult-FP doubles at ≤15 because SigLIP-2 can't reliably distinguish young-looking 18+yo from 15-17yo. To catch 15-17yo, pair with a Qwen3-VL async backstop (or extend the training set with high-quality 15-17yo labels).
2. **Pipeline is CSAM-only.** Non-underage/non-adult labels (bestiality, weapons, blood, drugs, rape, ethnicity) are dropped from the SigLIP request to enable 1-batch inference. Other moderation categories not detected — keep `violations-detector-test-v3` alongside, or extend K=30 with ~325 more labels.
3. **K=25 vs K=30 numbers from #3626 vs this README**: #3626's K=25 published numbers (44.6% recall / 3.3% FPR) are on the LS dataset with the human "underage" label (any minor <18). The numbers in this README are K=25 evaluated on our Qwen3-≤14 dataset. Different distribution, different label definition.

## Open items for collaboration

- **Score the LS 8,533-image set through K=30 (BCI)** for a flip-side apples-to-apples in K=25's home court.
- **Decide deployment policy**: CSAM-only + keep test-v3 for other categories, or extend K=30 with non-CSAM labels into a unified pipeline.
- **Shadow-deploy K=30 (BCI) for 1 week** before flipping `activated: true`, sample model-blocks and model-passes for human re-check.

## Contact

Tom Renneberg — internal Slack / `markus.b.schmidt@gmail.com`. File issues against #3626 or open MRs on this repo.
