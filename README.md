# piper-moderate

Underage-content moderation pipeline built on top of SigLIP-2 image-text classifier + LightGBM
gradient-boosting scorer. Includes training scripts, gallery for live model comparison,
and cross-validation tooling against external benchmarks.

**Current production model:** `V8pas80-v2` — deployed in Piper project `d2911d10bb`.

## How it works

```
Image  →  SigLIP-2 (180 underage tags, 1 batch)  →  LGBM (80 features + 4 BCI aggregates)  →  threshold 0.30  →  block / pass
```

The LGBM classifier learns from per-image SigLIP scores plus three semantic aggregates
(Body / Context / Interaction) that disambiguate "real child in frame" from
"child-adjacent context without an actual child" (cosplay, anime classroom, etc).

On full LS + Grafana set (3,120 items):

| Stage | child recall | teen recall | adult FPR |
|---|---|---|---|
| Pure SigLIP (rule-based) | 96.9% | 79.1% | 46.6% |
| + LGBM (V6) | ≈96% | ≈78% | ≈22% |
| + BCI feature split (V8pa) | ≈98% | ≈88% | ≈13% |
| **+ Hard-neg ×20 (V8pas80-v2)** | **97.7%** | **86.2%** | **6.1%** |

Cross-validation against [Tom Renneberg's K=30 (BCI)](https://gitlab.artworks.ai/realistic-ai/fullstack/-/issues/3626) model
confirms the architecture: V8pas80-v2 catches +9.7pp more children and 4× more teens
on his dataset, K=30 wins on adult FPR by virtue of targeting only `≤14`.

## Repository layout

```
piper-moderate/
├── gallery_server.py             # live model-comparison gallery (V8 / V11 / V6 inline)
├── simulator/index.html          # standalone SigLIP threshold tuner
├── config.yaml
│
├── scripts/
│   ├── bci_taxonomy.py           # Body / Context / Interaction label sets
│   ├── train_v8pa.py             # V8pa: V7pa + BCI + hard-neg ×20
│   ├── train_v8pas80_v2.py       # current production model
│   ├── train_v11.py              # V11 candidate (with no_underage_ features)
│   ├── train_slim.py             # feature pruning (top-N by gain + always-keep 4 BCI)
│   ├── train_pathA.py            # V7pa / V10pa — :x multipliers stripped
│   │
│   ├── export_v8pas80_v2_js.py   # build JS LGBM evaluate-script for Piper
│   ├── export_v8pa_js.py
│   ├── export_v11_js.py
│   │
│   ├── deploy_v8pas80_v2.py      # PATCH Piper d2911d10bb with new model
│   ├── deploy_phase4.py
│   ├── rollback_piper.py
│   │
│   ├── bench_v8pas80_v2.py       # regression gates on LS holdout
│   ├── grok_validate_fps.py      # Grok-4.3 validation of FP candidates
│   ├── run_k30_dataset.py        # run Tom's 6,675 dataset through d2911d10bb
│   ├── compare_k30_vs_ours.py    # binary head-to-head
│   ├── compare_k30_3class.py     # 3-class breakdown by child / teen / adult
│   │
│   ├── export_disagree.py        # fetch disagree images from Grafana
│   ├── moderate_disagree.py      # auto-moderate via Piper
│   ├── run_disagree_pipeline.py
│   ├── analyze_failed.py         # Grok-based tag suggestion for FP/FN
│   └── update_tags.py            # apply tag suggestions to data/tags.json
│
├── data/
│   ├── lgbm_underage_v8pas80_v2.txt        # trained model (binary)
│   ├── lgbm_v8pas80_v2_features.json       # feature list
│   ├── lgbm_v8pas80_v2_meta.json           # training metadata
│   ├── lgbm_evaluate_v8pas80_v2.js         # ready-to-deploy Piper LGBM node
│   ├── lgbm_underage_v11s80.txt            # V11 candidate
│   ├── d2911d10bb_slim_labels.json         # 180-label slim taxonomy
│   ├── k30_ls_export_full.json             # Tom's dataset exported
│   ├── k30_ours.jsonl                      # our pipeline applied to it
│   ├── k30_3class_report.json              # 3-class comparison
│   ├── k30_vs_v8pas80_v2_report.json
│   ├── v8pas80_top100_fps.json             # FP candidates
│   ├── v8pas80_fps_grok_confirmed.json     # 67 Grok-validated confirmed adults
│   ├── v8pas80_v2_full_thr_sweep.json      # threshold sweep on full dataset
│   └── tags.json                           # SigLIP tag library (versioned)
│
└── docs/
    └── workflow.md
```

## Setup

### Clone + install

```bash
git clone https://github.com/Kortique/piper-moderate.git
cd piper-moderate

python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### Configure secrets

Copy `.env.example` to `.env` and fill in your credentials.

**Required keys:**

| Key | Used by |
|-----|---------|
| `PIPER_TOKEN` | all `deploy_*` / `run_*` scripts (Piper API auth) |
| `PIPER_PROJECT` | default project ID for moderation runs |