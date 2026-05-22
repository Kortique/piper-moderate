# Label Studio — Underage Detection Eval (Project 5)

The 6,675-image evaluation dataset is loaded into Label Studio for human ground-truth labeling.

- **Project**: [Underage detection — K=30 (BCI) eval set](https://ls.artworks.ai/projects/5)
- **Project ID**: 5
- **Title**: "Underage detection — K=30 (BCI) eval set"
- **Task count**: 6,675
- **Media storage**: `https://fsn1.your-objectstorage.com/artworks-assets/g-<gid>.webp` (internal S3, WEBP q=96, matches existing media quality)

## What each task shows the labeler

- **Image** (max 600 px wide for fast loading)
- **Original prompt** (`data.prompt`) — the prompt the user used to generate the image
- **Qwen3-VL prediction** — fully structured, see fields below
- **SigLIP-2 raw scores** — combined underage/adult scores + top-5 labels per category
- **Choice question**: `age_verdict` — one of:
  - `underage_le_14` — ≤14, clear minor
  - `borderline_15_17` — 15-17, borderline
  - `adult_18plus` — adult, 18+
  - `cannot_tell` — face hidden / ambiguous
- **Optional notes** textarea

## Full task data schema

Every task in project 5 has this structure (sortable/filterable in LS Data Manager):

```json
{
  "data": {
    "source":      "private",
    "external_id": "g/<gid>",
    "type":        "image",
    "media":       "https://fsn1.your-objectstorage.com/artworks-assets/g-<gid>.webp",
    "prompt":      "<original generation prompt>",

    "qwen3": {
      "desc":          "<full visual description>",
      "faces":         [{"ageFrom": 5, "ageTo": 7}],     // structured age ranges per face
      "faces_count":   1,                                  // int — for filtering
      "min_age":       5,                                  // int — min of all ageFrom
      "max_age":       7,                                  // int — max of all ageTo
      "underage":      true,                               // bool — Qwen3's binary verdict
      "nsfw":          "X",                                // X | M | PG-13 | SFW
      "style":         "Realistic",                        // Realistic | Anime | ...
      "gender":        "Straight",                         // Straight | Gay | Solo | ...
      "quality":       "Masterpiece",                      // Masterpiece | Good | ...
      "status":        "BLOCK",                            // BLOCK | PASS
      "rape":          true,                               // bool — Qwen3 detected rape
      "bestiality":    false,
      "weapons":       false,
      "drugs":         false,
      "blood":         false,
      "death":         false,
      "celebrities":   false,
      "block_reasons": "<free text>"
    },

    "siglip": {
      "underage_score": 0.6727,                            // noisy-OR over ~116 underage SigLIP labels (production formula)
      "adult_score":    0.0,                               // noisy-OR over ~108 adult SigLIP labels
      "confidence":     1.0,                               // underage / (underage + adult)
      "top_underage":   [                                  // top-5 firing underage labels (raw SigLIP-2 scores)
        {"label": "intercourse_scene_with_girl", "score": 0.1919},
        {"label": "macro_monster_anatomy",       "score": 0.0998},
        {"label": "group_naked_teens",           "score": 0.0984},
        {"label": "naked_teen",                  "score": 0.0983},
        {"label": "man_with_young_girl",         "score": 0.0680}
      ],
      "top_adult":      []                                 // top-5 firing adult labels
    },

    "models": {
      "prod_score":   1.0,                                 // detect.ts noisy-OR ratio (continuous)
      "prod_blocked": true,                                // detect.ts decision (ratio≥0.85 AND risk≥0.01)
      "k25_score":    0.9997,                              // existing piper LGBM (violations-detector-test-v3)
      "k30_score":    0.9957,                              // K=30 (BCI) — 5-fold OOF prediction
      "k30_blocked":  true                                 // K=30 (BCI) decision at threshold 0.10
    }
  }
}
```

> **What's the difference between `siglip.*` and `models.*`?**
> `siglip.*` are the **raw model outputs** — direct cosine similarities from SigLIP-2 between the image and each text label, combined via noisy-OR. These are deterministic and identical regardless of any downstream classifier. `models.*` are the **classifier verdicts** on top: production rule, K=25 LightGBM, K=30 (BCI) LightGBM. Two different layers of the stack.

### Useful Data Manager filters

In the LS UI → Data Manager → Filters, sort or filter by:

| Field | Use case |
|---|---|
| `data.models.k30_score` | Sort desc → review what K=30 (BCI) thinks is highest-risk first |
| `data.models.k25_score` | Same for K=25 |
| `data.models.prod_blocked` | Filter "= true" → what production currently blocks (44.1% recall, 15.4% FPR) |
| `data.models.k30_blocked` | Filter "= true" → what K=30 (BCI) blocks at thr 0.10 |
| `models.k30_score < 0.05 AND qwen3.min_age <= 14` | K=30's **false negatives** (children it missed) |
| `models.k30_score >= 0.10 AND qwen3.min_age >= 18` | K=30's **false positives** (adults it mis-flagged — should be ~24 of 3,832) |
| `abs(models.k30_score - models.k25_score) > 0.5` | Cases where K=30 (BCI) and K=25 strongly disagree |
| `data.siglip.underage_score` | Sort desc → raw SigLIP underage signal before classifier |
| `data.siglip.confidence` < 0.5 | Find ambiguous cases (SigLIP can't tell minor vs adult) |
| `data.qwen3.min_age` / `.max_age` | Filter by Qwen3-VL face age range |
| `data.qwen3.faces_count`     | Filter "= 0" for no-face images; "> 1" for multi-person scenes |
| `data.qwen3.status`          | Filter "BLOCK" vs "PASS" — Qwen3's binary verdict |
| `data.qwen3.rape` / `.bestiality` / etc. | Filter on specific safety categories |

## Label config XML (for reference)

```xml
<View>
  <Image name="img" value="$media" maxWidth="600px"/>
  <Header value="Prompt: $prompt"/>
  <View style="display: flex; gap: 20px; padding: 8px; background: #f5f5f7; border-radius: 4px;">
    <View>
      <Header value="Qwen3-VL says:" size="6"/>
      <Text name="q_age" value="ages: $qwen3.min_age – $qwen3.max_age  ($qwen3.faces_count faces detected)"/>
      <Text name="q_underage" value="underage flag: $qwen3.underage"/>
      <Text name="q_nsfw" value="nsfw: $qwen3.nsfw  •  style: $qwen3.style  •  status: $qwen3.status"/>
      <Text name="q_desc" value="$qwen3.desc"/>
    </View>
    <View>
      <Header value="SigLIP-2 says:" size="6"/>
      <Text name="s_under" value="underage_score: $siglip.underage_score"/>
      <Text name="s_adult" value="adult_score: $siglip.adult_score"/>
      <Text name="s_conf" value="confidence: $siglip.confidence"/>
    </View>
  </View>
  <Choices name="age_verdict" toName="img" choice="single" showInLine="true" required="true">
    <Choice value="underage_le_14" alias="≤14 — clear minor"/>
    <Choice value="borderline_15_17" alias="15-17 borderline"/>
    <Choice value="adult_18plus" alias="adult 18+"/>
    <Choice value="cannot_tell" alias="cannot tell (face hidden, etc)"/>
  </Choices>
  <TextArea name="notes" toName="img" placeholder="Optional notes" rows="2"/>
</View>
```

Adjust in the Label Studio UI: Settings → Labeling Interface.

## Exporting annotations

Once labelers have annotated, pull the results via API:

```bash
export LS_TOKEN=<your-token>
curl -sS -H "Authorization: Token $LS_TOKEN" \
  "https://ls.artworks.ai/api/projects/5/export?exportType=JSON" \
  -o annotations.json
```

Or via the LS UI: project → "Export" → format JSON / CSV.

## Joining annotations back to the dataset

Each task's `external_id` is `g/<generation_id>`. Strip the `g/` prefix to match `scored.sqlite` / `scored_benign.sqlite`:

```python
import sqlite3, json

ann = json.load(open("annotations.json"))
labels = {}
for task in ann:
    gid = task["data"]["external_id"].removeprefix("g/")
    verdict = next((r["value"]["choices"][0] for r in task["annotations"][0]["result"] if r["from_name"] == "age_verdict"), None)
    labels[gid] = verdict

# Then join with sqlite by generation_id
```

## The move pipeline (for reference)

The scripts that moved the data into Label Studio + fsn1 S3 are in this repo
(secrets read from env vars; do not commit secrets):

| Script | Purpose |
|---|---|
| `ls/move_to_label_studio.py` | Initial pipeline: download → WEBP q=96 → fsn1 → LS import |
| `ls/reencode_q96.py` | Quality-match re-encode (was q=85 first, raised to q=96 to match existing media at ~0.144 byte/pixel) |
| `ls/import_to_project5.py` | Move tasks from project 3 (Gallery NSFW) to dedicated project 5 |
| `ls/delete_from_project3.py` | Remove the 6,675 tasks from project 3 after moving |
| `ls/enrich_ls_tasks.py` | Bulk-PATCH every task's `data` with the full structured Qwen3 (`faces[]` array) + SigLIP (combined scores + top-5 labels per category). Run after the initial import to expose all model predictions for sort/filter in Data Manager. |
| `ls/compute_and_patch_model_scores.py` | Compute per-image scores for production (`detect.ts`), K=25, and K=30 (BCI) — the latter via 5-fold OOF — and PATCH each task's `data.models` field. Sources of truth for sort-by-model-prediction in the Data Manager UI. |

To re-run (env vars required):
```bash
export PIPER_TOKEN=...                # not used by these scripts but used by piper-build scripts
export S3_KEY_ID=...                  # fsn1 access key
export S3_SECRET=...                  # fsn1 secret
export LS_TOKEN=...                   # Label Studio API token
python3 ls/move_to_label_studio.py    # full pipeline (resumable)
```

State files in `/tmp/k30-share/` track resume-progress:
- `done.json` — set of generation_ids that completed initial upload
- `reencoded.json` — set of gids that completed q=96 re-encode
- `deleted_from_p3.json` — set of LS task IDs deleted from project 3
- `our_task_ids_in_p3.json` — cached list of our task IDs in project 3 (for re-deletion if needed)

## Caveats

1. **5 generations were unrecoverable** (HTTP 404 from realistic-media — images deleted between when we scored them and when we tried to move them). Their gids are in `ls/errors_404.json` if you need to investigate. Final count is 6,675 not 6,680.
2. **Image quality**: WEBP q=96 (~0.144 byte/pixel) was chosen to match existing media in the bucket. Lower than the source PNG (lossless), but visually indistinguishable for human labeling. If labelers report compression artifacts on borderline cases, raise to q=98 or keep PNG.
3. **External ID format**: `g/<uuid>` for the LS task `external_id`, `g-<uuid>.webp` for the S3 object key. The `g-` separator matches the existing convention shown in the import example.
