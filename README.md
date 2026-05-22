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
| `OPENROUTER_API_KEY` | `analyze_failed.py`, `grok_validate_fps.py` (vision LLM analysis) |
| `GRAFANA_USER`, `GRAFANA_PASSWORD` | `export_disagree.py` (Grafana login) |
| `GRAFANA_SESSION` | auto-refreshed by `grafana_login.py` |
| `EXPORT_WATCH_DIR` | folder where mod.artworks.ai exports land |
| `LS_TOKEN` (optional) | Label Studio API token (refresh token for JWT auth) |

## Common workflows

### Re-train V8pas80-v2 with new hard-negatives

```bash
# 1. Identify new FP candidates from gallery and run them through Grok validation
python scripts/grok_validate_fps.py --chunk 100

# 2. Re-train (uses bci_taxonomy.py for aggregates, hard-neg pool weight ×20)
python scripts/train_v8pas80_v2.py

# 3. Sanity-check regression gates on LS holdout
python scripts/bench_v8pas80_v2.py

# 4. Export JS for Piper and deploy
python scripts/export_v8pas80_v2_js.py
python scripts/deploy_v8pas80_v2.py
```

Rollback: `python scripts/rollback_piper.py d2911d10bb <previous_revision>`

### Cross-validate against Tom's K=30 dataset

```bash
# 1. Pull dataset from Label Studio (project 5)
# - dataset already in data/k30_ls_export_full.json

# 2. Run our pipeline against his 6,675 images
python scripts/run_k30_dataset.py --chunk 80 --workers 15
# (resumable, appends to data/k30_ours.jsonl)

# 3. Generate comparison reports
python scripts/compare_k30_vs_ours.py     # binary head-to-head
python scripts/compare_k30_3class.py      # 3-class child / teen / adult breakdown
```

Outputs `data/k30_vs_v8pas80_v2_report.json`, `data/k30_3class_report.json` and a sample
disagreement list at `data/k30_vs_v8pas80_v2_disagreements.json`.

### Disagree-image moderation loop

```bash
# 1. Fetch latest disagree images from Grafana, auto-moderate via Piper
python scripts/export_disagree.py

# 2. Open the gallery to review and confirm labels
python gallery_server.py            # http://localhost:7823
python gallery_server.py --port 7825
```

### Gallery features

- **Three models scored inline on every card** — V8pas80-v2 (production),
  V11s80 (candidate), V6 (legacy baseline). All three share the same SigLIP labels.
- **Dynamic per-model thresholds** (V8 ≥ 0.30, V11 ≥ 0.30, V6 ≥ 0.80) in a collapsible bar.
- **Filter by selected model and outcome** (TP / TN / FP / FN) — instantly find e.g. all
  V8 false-positives on adult.
- **Pipeline verdict badge** (`ok` / `underage`) — driven by currently-selected model at
  current threshold, recomputes on the fly.
- **Per-model breakdown panel** — child / teen / adult coverage with absolute-threshold
  colour indicators and best-in-row markers.
- **Hotkeys** on hover: `1` = child, `2` = teen, `3` = adult.

## Data model

`grafana_pool` (SQLite, `gallery.db`):

```
id, thumb_url, local_path, prompt, label, label_source, label_confirmed,
labeled_at, variant, export_batch, exported_at,
piper_result (siglip2_labels, siglip2_passed, siglip2_details, face_detect_result),
qwen3_result (label, faces, description, underage, status)
```

`ls_images` — separate table with Label-Studio-sourced items.

`label_source = "qwen3"` means automatic suggestion; `"human"` means confirmed in gallery.
Only confirmed records participate in training.

## Tag naming convention

- `{category}_{description}` — positive detection tag (`underage_kneeling_teen_girl`)
- `no_{category}_{description}` — counter-tag suppressing false positives
  (`no_underage_youthful_adult_face`)

**`:x20` / `:x5` multiplier suffixes were retired** in Sprint 2025-11 (Path A) —
multipliers caused systematic false positives on adult POV shots
(e.g. `man_with_young_girl:x20` firing on size-contrast adult-only scenes).
The LGBM scorer now sees raw cosine scores without inflation.

##