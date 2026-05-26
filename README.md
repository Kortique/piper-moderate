# piper-moderate

Underage-content moderation pipeline built on top of SigLIP-2 image-text classifier + LightGBM
gradient-boosting scorer. Includes training scripts, an interactive gallery for live model
comparison and moderation, and cross-validation tooling against external benchmarks.

**Currently deployed:** `V8pas80-v2` in Piper project `d2911d10bb` (production).
**Production candidate (Sprint 2026-05):** `V8cs80` — same architecture, extended scope, +11pp teen recall vs deployed.
**Candidate model:** `V11s80` in `ce79f7e299`. **Next candidate:** `V11cs80` (extended scope, 98% child / 90% teen recall).

## How it works

```
Image → SigLIP-2 (180 underage tags) → LGBM (80 features + 4 BCI aggregates) → threshold → block / pass
```

The LGBM scorer learns from per-image SigLIP cosine scores plus three semantic aggregates
(Body / Context / Interaction) that disambiguate "real child in frame" from
"child-adjacent context without an actual child" (cosplay, anime classroom, etc.).

### Honest holdout numbers (Sprint 2026-05 — extended scope)

After a full manual relabel of 9,150 items, each "c" model was re-trained on the
**extended scope** (LS + Grafana + K30 = ~7,300 train items) and evaluated on
`data/v11_test_split_2026.json` — a fresh 80/20 split stratified by label × source
(1,815 held-out items: 292 child / 357 teen / 1,179 adult; 433 LS + 169 Grafana + 1,226 K30),
averaged across 3 lgb seeds {1, 42, 314}.

| Model | Test AUC | child recall | teen recall | adult FPR | n_features | scope |
|---|---|---|---|---|---|---|
| V6c (legacy LGBM) | 0.948 ±0.001 | 97.4% | 79.1% | **12.6%** | 314 | LS + Grafana + K30 |
| V8cs80 (slim, production candidate) | 0.947 ±0.001 | 96.8% | 81.5% | 16.3% | 80 | LS + Grafana + K30 + hard-neg |
| **V11cs80 (slim, candidate)** | **0.946 ±0.001** | **98.0%** | **90.1%** | 22.9% | 80 | LS + Grafana + K30 + BCI |
| Tom K=30 (external reference) | 0.787 | 88.7% | 61.4% | 20.7% | 37 | targets `≤14` only |

All three "c" models converged to AUC ~0.946–0.948 — high-performance underage detection.
V11cs80 leads on **child + teen recall** (98% / 90%), V6c leads on **adult FPR** (12.6%),
V8cs80 sits in the middle as the production-ready compromise.

#### Previous models (Sprint 2026-04, narrower scope)

For reference — these were trained on LS+Grafana only (~2,500 items) before K30 was added:

| Model | Test AUC | child | teen | adult FPR | scope |
|---|---|---|---|---|---|
| V6b | 0.888 ±0.001 | 93.6% | 70.8% | 18.2% | LS + Grafana |
| V8bs80 | 0.844 ±0.001 | 92.0% | 58.2% | 18.4% | LS only |
| V11bs80 | 0.879 ±0.003 | 93.9% | 71.2% | 20.8% | LS + Grafana |

#### Full-CV numbers (training-set 5-fold stratified)

| Model | n_samples | CV AUC | n_features |
|---|---|---|---|
| V6 (recomputed retroactively) | 2,903 | 0.879 ±0.012 | 312 |
| V8pas80-v2 (slim, currently deployed) | 3,574 | 0.850 ±0.013 | 80 |
| V11s80 (slim) | 3,652 | 0.905 ±0.007 | 80 |

`:x20` and `:x5` multiplier suffixes were retired in Sprint 2025-11 (Path A) — they caused
systematic false positives on adult POV shots. The LGBM scorer now sees raw cosine scores.

## Repository layout

```
piper-moderate/
├── gallery_server.py             # interactive moderation gallery (live)
├── simulator/index.html          # standalone SigLIP threshold tuner
│
├── scripts/
│   │  ── training ──
│   ├── train_v8pas80_v2.py       # original production model retrain (LS only, ~3k items)
│   ├── train_v11.py              # V11 candidate (original)
│   ├── train_v6b_holdout.py      # honest holdout retrain: V6 (LS+Grafana)
│   ├── train_v8b_holdout.py      # honest holdout retrain: V8 (LS only)
│   ├── train_v11b_holdout.py     # honest holdout retrain: V11 (LS+Grafana)
│   ├── train_v6c_holdout.py      # extended scope V6 (LS+Grafana+K30)
│   ├── train_v8c_holdout.py      # extended scope V8 (LS+Grafana+K30+hardneg×20)
│   ├── train_v11c_holdout.py     # extended scope V11 (LS+Grafana+K30+BCI)
│   ├── make_holdout_split_2026.py # creates v11_test_split_2026.json (stratified by label×source)
│   ├── train_slim.py             # feature pruning (top-N by gain + always-keep 4 BCI)
│   │
│   │  ── rescore (resumable, retry/backoff on 5xx and timeouts) ──
│   ├── rescore_via_v11.py        # LS+Grafana → V11 native pipeline ce79f7e299
│   ├── rescore_k30_v6v8.py       # K30 → production d2911d10bb (180-tag for V6/V8)
│   ├── rescore_via_tom.py        # LS+Grafana → Tom K=30 pipeline a4aa9dbd9c
│   │
│   │  ── data ingestion ──
│   ├── import_k30.py             # import Tom's K=30 dataset into gallery.db
│   ├── export_disagree.py        # fetch disagree images from Grafana
│   ├── moderate_disagree.py      # auto-moderate via Piper
│   ├── run_category.py           # category-scoped runner
│   │
│   │  ── deploy + compare ──
│   ├── export_v8pas80_v2_js.py   # build JS LGBM evaluate-script for Piper
│   ├── export_v8cs80_js.py       # V8cs80 (extended scope) → Piper JS
│   ├── export_v11cs80_js.py      # V11cs80 (extended scope) → Piper JS
│   ├── extract_v8cs80_ls_fps.py  # next-wave hard-neg mining on V8cs80 LS adults
│   ├── deploy_v8pas80_v2.py      # PATCH Piper d2911d10bb with new model
│   ├── rollback_piper.py
│   ├── bench_v8pas80_v2.py       # regression gates on LS holdout
│   ├── grok_validate_fps.py      # Grok-4 validation of FP candidates
│   ├── compare_k30_vs_ours.py    # binary head-to-head with Tom's K=30
│   ├── compare_k30_3class.py     # 3-class child / teen / adult breakdown
│   └── analyze_failed.py         # Grok-based tag suggestion for FP / FN
│
├── data/
│   │  ── current production model ──
│   ├── lgbm_underage_v8pas80_v2.txt        # trained model (binary)
│   ├── lgbm_v8pas80_v2_features.json
│   ├── lgbm_v8pas80_v2_meta.json
│   ├── lgbm_evaluate_v8pas80_v2.js         # Piper LGBM node
│   │
│   │  ── candidate V11 + holdout retrains ──
│   ├── lgbm_underage_v11s80.txt
│   ├── lgbm_underage_v6b.txt    lgbm_v6b_meta.json
│   ├── lgbm_underage_v8bs80.txt lgbm_v8bs80_meta.json
│   ├── lgbm_underage_v11bs80.txt lgbm_v11bs80_meta.json
│   │
│   │  ── shared splits / configs ──
│   ├── v11_test_split.json                 # frozen 618-item holdout (SEED=1337)
│   ├── thresholds.json                     # persisted UI thresholds (v6/v8/v11/k30tom)
│   ├── tags.json                           # SigLIP tag library (versioned)
│   ├── d2911d10bb_slim_labels.json         # 180-label slim taxonomy
│   │
│   │  ── rescore dumps (gitignored, regenerated locally) ──
│   ├── v11_native_scores.json   # ~7 MB,  rescore_via_v11.py output
│   ├── tom_scores.json          #         rescore_via_tom.py output
│   ├── k30_rescored.json        #         rescore_k30_v6v8.py output
│   │
│   │  ── K30 study ──
│   └── k30/                                # Tom K=30 dataset, papers, scripts
│
└── docs/
    └── workflow.md
```

## Setup

```bash
git clone https://github.com/Kortique/piper-moderate.git
cd piper-moderate
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` → `.env` and fill in real values:

| Key | Used by |
|-----|---------|
| `PIPER_TOKEN` | all `deploy_*` / `rescore_*` scripts (Piper API auth) |
| `PIPER_PROJECT` | default project ID for moderation runs |
| `OPENROUTER_API_KEY` | `analyze_failed.py`, `grok_validate_fps.py` (vision LLM analysis) |
| `GRAFANA_USER`, `GRAFANA_PASSWORD` | `export_disagree.py` |
| `EXPORT_WATCH_DIR` | folder where mod.artworks.ai exports land |
| `LS_TOKEN` (optional) | Label Studio API token |

## Common workflows

### Honest holdout retrain (V6b / V8bs80 / V11bs80)

All three "b" models share the same frozen test set so their numbers are directly comparable.

```bash
# Train all three. Re-uses data/v11_test_split.json — never re-shuffles.
python scripts/train_v11b_holdout.py
python scripts/train_v8b_holdout.py
python scripts/train_v6b_holdout.py

# Outputs: data/lgbm_underage_v*b*.txt, data/lgbm_v*b*_meta.json
# Multi-seed metrics ({1, 42, 314}) and per-source breakdown land in *_meta.json.
```

### Rescore the dataset through a specific pipeline

V11 needs its native 317-tag input (with `:x20` style features) to score correctly.
Production `d2911d10bb` outputs 180-tag input that V6/V8 expect.
Tom's K=30 pipeline `a4aa9dbd9c` gives a third independent vote.

```bash
# V11 native rescore (run when V11 numbers in gallery look unrealistic)
python -u scripts/rescore_via_v11.py             # LS + Grafana (~3,120 items)
python -u scripts/rescore_via_v11.py --source k30 --workers 6   # opt-in for K30

# V6/V8 K30 rescore (180-tag for sparse-input K30 items)
python -u scripts/rescore_k30_v6v8.py --workers 6

# Tom K=30 cross-evaluation
python -u scripts/rescore_via_tom.py
```

All three scripts: resumable (skip already-done IDs), 4-attempt exponential backoff on
5xx/timeouts/network errors, atomic JSON writes via tmp+rename, terminal `no_face` errors
recorded once so they don't re-run.

### Re-train production V8pas80-v2 with new hard-negatives

```bash
# 1. Identify new FP candidates from gallery → Grok-validate
python scripts/grok_validate_fps.py --chunk 100

# 2. Re-train (bci_taxonomy.py for aggregates, hard-neg pool weight ×20)
python scripts/train_v8pas80_v2.py

# 3. Sanity regression on LS holdout
python scripts/bench_v8pas80_v2.py

# 4. Export and deploy
python scripts/export_v8pas80_v2_js.py
python scripts/deploy_v8pas80_v2.py
```

Rollback: `python scripts/rollback_piper.py d2911d10bb <previous_revision>`

### Cross-validate against Tom's K=30 dataset

```bash
# Tom's 6,675 items are already in data/k30_ls_export_full.json (gitignored).
# Run our pipeline against them:
python scripts/rescore_k30_v6v8.py                  # V6/V8 input
python scripts/rescore_via_v11.py --source k30      # V11 input

# Generate comparison reports
python scripts/compare_k30_vs_ours.py     # binary head-to-head
python scripts/compare_k30_3class.py      # 3-class child / teen / adult breakdown
```

### Disagree-image moderation loop

```bash
python scripts/export_disagree.py         # fetch from Grafana
python gallery_server.py                  # http://localhost:7823
```

## Gallery

Live model-comparison + moderation UI at `http://localhost:7823`.

### Four models side-by-side on every card

V6 (legacy) · V8pas80-v2 (production) · V11s80 (candidate) · Tom K=30 (external reference).

For each model: LGBM score, per-model UI threshold slider, CV AUC + honest test AUC chip,
per-source recall/FPR breakdown, fallback-taxonomy marker (`*`) when the model is fed
input from a non-native pipeline.

### Filters & sources

- **Source** — All / Label Studio / Grafana (with session) / K30 / **⭐ Отмеченные** (marked-only)
- **Label** — All / unlabeled / unconfirmed / child / teen / adult
- **Age** — q3 (qwen3 estimate) and fd (face_detect) range filters
- **LGBM panel** (collapsible) — drag any threshold, all badges + stats update live;
  "Holdout" toggle restricts stats to the frozen 618 test items;
  "Disagreement" toggle sorts by our-vs-Tom score delta (active-learning queue).

### Lightbox moderation mode

Click any card → full-screen lightbox with:

- **Image on the left**, **info sidebar on the right** (falls back to overlay on narrow viewports)
- **Verdict badge** at top: UNDERAGE / OK / UNKNOWN (effective label wins; otherwise model vote)
- **Per-model LGBM scores** with color grading (red ≥ thr, green < thr)
- **Age info**: q3, fd, Label-Studio source range
- **🔥 disagreement row** with low/mid/hi color
- **⭐ marked toggle row**
- **Thumbnails strip** below image, auto-centered on current item (cubic-bezier transition)
- **Position counter** "N / TOTAL" within current page

### Hotkeys

In lightbox:
- `1` `2` `3` → child / teen / adult (auto-advance after 120ms flash)
- `4` → mark for deletion (auto-advance)
- `5` or `m` → toggle ⭐ marked
- `Enter` → next item · `Esc` → close
- **Mouse wheel** → next/prev (180ms throttle)

On hover (outside lightbox): same `1` `2` `3` `5` / `m` work for the hovered card.

### Marked items workflow

A persistent ⭐ set (across browser sessions, via `localStorage`):

- ☆/★ button top-center of each card; click or hotkey toggles
- Golden outline around marked cards
- Source dropdown has `⭐ Отмеченные` to filter to marked-only
- **Export JSON** button in toolbar produces `marked_YYYYMMDDHHMMSS.json` containing
  id / source / label / qwen3+fd+LS ages / model scores (per-threshold blocked flags) /
  fallback markers for every marked item

### Card UX

- 👁 viewed-in-lightbox badge (bottom-center of preview, session-scoped)
- 🗑 deletion mark synced between card and lightbox
- Confirm/save flow: "Confirm page" auto-saves in one click;
  Save preserves current page (no jump to page 1) and auto-confirms items
  the user touched or viewed in lightbox during the session.

## Data model

`gallery.db` (SQLite) with three pools:

- `grafana_pool` — production moderation queue (Grafana exports)
- `ls_images`   — Label Studio items
- `k30_pool`    — Tom's external K=30 dataset (6,675 items)

Each row carries `piper_result` (siglip2_details, face_detect_result),
`qwen3_result` (min/max age, faces, description), label, label_confirmed, variant, etc.

`label_source = "qwen3"` is the auto-suggested draft; `"human"` is confirmed via the gallery.
Only confirmed records participate in training.

## Tag naming convention

- `{category}_{description}` — positive detection tag (`underage_kneeling_teen_girl`)
- `no_{category}_{description}` — counter-tag suppressing false positives
  (`no_underage_youthful_adult_face`)

`:x20` / `:x5` multiplier suffixes are not used after Sprint 2025-11 (Path A).
The LGBM scorer sees raw cosine scores without inflation.

## Cross-validation report

Versus Tom Renneberg's [K=30 BCI model](https://gitlab.artworks.ai/realistic-ai/fullstack/-/issues/3626).
The two models target different policies — Tom: strict `≤14`, ours: `≤17` —
so each dominates on its own scope.

| Axis | V11cs80 (ours, 2026-05) | Tom K=30 |
|---|---|---|
| holdout AUC | **0.946** | 0.787 |
| child recall | **98.0%** | 88.7% |
| teen recall (15–17) | **90.1%** | 61.4% |
| adult FPR | 22.9% | **20.7%** |
| target policy | `≤17` underage content | `≤14` CSAM |

`adult FPR` measured against our 18+ adult test set: Tom is correctly less aggressive on
15–17 (which are "adult" by his definition), so his FPR there is naturally lower.

## Contributing

1. Fork
2. `git checkout -b feature/{name}`
3. Edit tag library at `data/tags.json`; train via `scripts/train_*.py`; output models at `data/lgbm_*.txt`
4. Run regression bench: `python scripts/bench_v8pas80_v2.py` (all three gates must hold)
5. Commit: `git commit -m "feat(model): add anime-cosplay hard-negs"`
6. Open a PR
