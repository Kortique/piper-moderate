# piper-moderate

Automated content moderation pipeline for SigLIP-2 tag tuning and disagree image analysis.

Integrates with:
- **Piper** — image classification pipeline (SigLIP-2 + Qwen3-VL)
- **Grafana / ClickHouse** — disagree review stats (`stat.artworks.ai`)
- **mod.artworks.ai** — test result viewer & exporter
- **OpenRouter / Grok** — vision LLM for failed image analysis (tag suggestions only)

---

## How it works

### Main tag-tuning workflow

```
2. mod.artworks.ai   →  manually click Refresh + Export to download JSON
3. analyze_failed.py →  sends failed images to Grok, gets tag suggestions
4. update_tags.py    →  reviews & applies suggestions to data/tags.json
5. Repeat from step 1 to verify improvements
```

### Disagree image analysis workflow

Images that users marked "disagree" were previously **passed** by the old Siglip2 (without LGBM Underage). This workflow measures how much better the new pipeline catches them.

```
1. export_disagree.py   →  fetch latest disagree reviews from Grafana, download images
                            auto-calls moderate_disagree.py on newly added images
2. moderate_disagree.py →  run Piper pipeline d2911d10bb (Siglip2 + Qwen3-VL)
                            stores age label suggestion + siglip2 result
3. gallery_server.py    →  open http://localhost:7823 to review, confirm/correct labels
4. run_disagree_pipeline.py →  (optional) re-run labeled images through a specific pipeline
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Kortique/piper-moderate.git
cd piper-moderate
```

### 2. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure secrets

```bash
cp .env.example .env
# Edit .env with your credentials (never commit .env!)
```

`.env` keys:

| Key | Description |
|-----|-------------|
| `OPENROUTER_API_KEY` | OpenRouter key — used for Grok tag analysis only |
| `EXPORT_WATCH_DIR` | Folder where mod.artworks.ai saves exports |
| `GRAFANA_USER` | stat.artworks.ai login |
| `GRAFANA_PASSWORD` | stat.artworks.ai password |
| `GRAFANA_SESSION` | Auto-updated by grafana_login.py — do not edit manually |
| `PIPER_TOKEN` | Piper API user token |
| `PIPER_PROJECT` | Default Piper project ID (main pipeline) |

---

## Usage

All commands must be run from the **project root** (where `config.yaml` lives).

### Tag-tuning commands

#### List available categories

```bash
python scripts/run_category.py --list
```

#### Run a category (trigger re-moderation)

```bash
python scripts/run_category.py --category underage
```

After the agent responds "✅ Reset complete", go to [mod.artworks.ai](https://mod.artworks.ai), click **Refresh** → **Export**.

#### Analyze failed images

```bash
# Auto-detect latest export from EXPORT_WATCH_DIR:
python scripts/analyze_failed.py --category underage

# Specify file explicitly:
python scripts/analyze_failed.py --category underage --results results/export.json

# Analyze only first 10 failed items (for quick testing):
python scripts/analyze_failed.py --category underage --limit 10

# Only missed threats (Failed+):
python scripts/analyze_failed.py --category underage --type positive

# Only false alarms (Failed-):
python scripts/analyze_failed.py --category underage --type negative
```

Output: `suggestions/underage_YYYY-MM-DD.json`

#### Review & apply tag suggestions

```bash
python scripts/update_tags.py --suggestions suggestions/underage_2026-04-13.json

# Apply all suggestions without confirmation:
python scripts/update_tags.py --suggestions suggestions/underage_2026-04-13.json --auto

# Preview changes without writing:
python scripts/update_tags.py --suggestions suggestions/underage_2026-04-13.json --dry-run
```

Output: updated `data/tags.json` + backup `data/tags.YYYY-MM-DD.bak.json`

#### Open the simulator

Open `simulator/index.html` directly in any browser. No server needed.

Load your exported JSON to see Accuracy/Error per category and tune thresholds interactively.

---

### Disagree image analysis

#### Export disagree reviews from Grafana

```bash
# Fetch latest 500 disagree images, download locally, auto-moderate with Piper:
python scripts/export_disagree.py

# Options:
python scripts/export_disagree.py --limit 1000 --hours 72   # wider lookback
python scripts/export_disagree.py --no-download             # skip image download
python scripts/export_disagree.py --dry-run                 # preview only
```

Downloads images to `data/disagree_images/`. Deduplicates across sessions by UUID.
Auto-refreshes the Grafana session via Authentik SSO if expired.

After download, automatically calls `moderate_disagree.py` on newly added images.

#### Moderate images (Piper: Siglip2 + Qwen3-VL)

```bash
# Process all unmoderated images (Piper project d2911d10bb):
python scripts/moderate_disagree.py

# Options:
python scripts/moderate_disagree.py --workers 3       # parallel Piper launches
python scripts/moderate_disagree.py --limit 50        # process up to 50
python scripts/moderate_disagree.py --reprocess       # re-run already processed
python scripts/moderate_disagree.py --stats           # print stats only
```

Piper project `d2911d10bb` runs both Siglip2 and Qwen3-VL in one call.
- **Qwen3-VL** → age label from youngest face in image  
  - child: ageFrom 1–14, teen: 15–17, adult: 18+ (by minimum ageFrom)
- **Siglip2** → blocked/passed, siglip2_labels

Labels are stored as **suggestions** (`label_source: "qwen3"`, `label_confirmed: false`).
Human confirmation in the gallery sets `label_source: "human"`, `label_confirmed: true`.
Only confirmed records participate in training base operations.

#### Review in gallery

```bash
# Start gallery server:
python gallery_server.py                # http://localhost:7823
python gallery_server.py --port 7825   # custom port
```

Gallery features:
- **Two sources**: Label Studio (`qwen3_age_results.json`) + Grafana (`disagree_pool.json`)
- **Session filter**: filter Grafana images by export batch (e.g. `2026-05-17 08:48 UTC`)
- **Pipeline filter**: filter by siglip2 result (underage / other / passed / unprocessed)
- **Label filter**: includes `⚡ Не подтверждено` — shows all AI-labeled, not yet confirmed
- **Badges on cards**:
  - `⚡ AI` (orange) — auto-labeled by Qwen3, awaiting human confirmation
  - `✓` (green) — human-confirmed
  - `⛔ underage` / `✓ ok` — siglip2 result
- **Hotkeys**: hover over card → `1` = child, `2` = teen, `3` = adult
- **Save**: marks changed labels as `label_source: "human"`, `label_confirmed: true`

---

## Data model

### `data/disagree_pool.json`

```json
{
  "<uuid>": {
    "id": "<uuid>",
    "thumb_url": "https://s3.../thumbnail.webp",
    "prompt": "...",
    "exported_at": "2026-05-17T05:43:00Z",
    "export_batch": "2026-05-17 05:43 UTC",
    "local_path": "data/disagree_images/<uuid>.webp",

    "label": "child | teen | adult | null",
    "label_source": "qwen3 | human | null",
    "label_confirmed": false,
    "labeled_at": "ISO timestamp | null",

    "qwen3_result": {
      "label": "adult",
      "faces": [{"ageFrom": 18, "ageTo": 22}],
      "description": "...",
      "underage": false,
      "status": "PASS",
      "processed_at": "ISO timestamp"
    },

    "piper_result": {
      "siglip2_labels": ["underage"],
      "siglip2_passed": false,
      "siglip2_details": {...},
      "processed_at": "ISO timestamp"
    }
  }
}
```

---

## Project structure

```
piper-moderate/
├── .env.example              # secrets template (copy to .env)
├── .gitignore
├── config.yaml               # non-secret settings
├── requirements.txt
│
├── scripts/
│   ├── analyze_failed.py     # VLM analysis via OpenRouter/Grok
│   ├── analyze_contextual.py # contextual analysis with full tag history
│   ├── update_tags.py        # apply suggestions to tags.json
│   ├── patch_and_simulate.py # simulate tag changes before applying
│   ├── piper_check.py        # check single image against Piper
│   │
│   ├── export_disagree.py    # fetch disagree images from Grafana, download
│   ├── moderate_disagree.py  # auto-moderate via Piper d2911d10bb
│   ├── run_disagree_pipeline.py  # run labeled images through a pipeline
│   └── grafana_login.py      # Authentik SSO → Grafana session (auto-refresh)
│
├── gallery_server.py         # unified gallery (port 7823)
│
├── simulator/
│   └── index.html            # standalone SigLIP threshold tuner
│
├── data/
│   ├── tags.json             # current tag library (versioned in git)
│   ├── disagree_pool.json    # Grafana disagree image pool
│   └── disagree_images/      # downloaded thumbnails (gitignored)
│
├── results/                  # exported test JSONs (gitignored)
└── suggestions/              # Grok suggestion JSONs (versioned in git)
```

---

## Tag naming conventions

See [docs/tag-naming.md](docs/tag-naming.md) for full rules.

Quick reference:
- `<category>_<description>` — positive detection tag
- `no_<category>_<description>` — counter-tag to suppress false positives
- `<key>:x20` — tag with ×20 score multiplier (high-signal indicator)
- `<key>:x5`  — tag with ×5 score multiplier

---

## Grafana / Authentik SSO

Grafana session is managed automatically:
- `GRAFANA_SESSION` in `.env` is validated on each `export_disagree.py` run
- If expired, `grafana_login.py` performs the full Authentik OAuth flow and updates `.env`
- Manual refresh: `python scripts/grafana_login.py`

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/improve-underage-tags`
3. Make changes to `data/tags.json` or scripts
4. Commit with a clear message: `git commit -m "feat(tags): add underage group nude tags"`
5. Open a Pull Request

All tag changes must be tested by running the full category before merging.
