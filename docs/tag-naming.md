# Tag Naming Conventions

## Structure

```
<category>_<descriptive_name>[:<multiplier>]
```

## Rules

### Category prefixes

| Category | Positive prefix | Counter prefix |
|----------|----------------|----------------|
| underage | `underage_` | — |
| adult (counter) | `adult_` | — |
| bestiality | `bestiality_` | `no_bestiality_` |
| human_waste | `human_waste_` | `no_waste_` |
| blood | `blood_` | `no_blood_` |
| death | `death_` | `no_death_` |
| weapons | `weapons_` | `no_weapon_` |
| military | `military_` | — |
| drugs | `drugs_` | `no_drugs_` |
| rape | `rape_` | `no_rape_` |
| asian/ebony/arab/latin/indian | `<ethnic>_` | — |
| style | `fantasy_style`, `anime_style`, `realistic_style` | — |

### Multipliers (SigLIP-2 score boosters)

Used for high-signal tags that should be weighted more heavily.

| Suffix | Multiplier | Use case |
|--------|-----------|---------|
| `:x20` | ×20 | Very strong indicator (e.g. adult+child together in same scene) |
| `:x5` | ×5 | Moderate booster (e.g. male minor with adult woman) |

Example: `underage_adult_with_child:x20`

Custom multipliers are supported: `tagname:x10`, `tagname:x3`, etc.

### Description best practices

Good descriptions are **visual** — they describe what SigLIP-2 would literally see in the image.

✅ Good:
```
"a prepubescent minor child, literal juvenile face, completely childish body, early grade-school developmental stage"
```

❌ Bad:
```
"inappropriate content involving a young person"
```

Guidelines:
- Be specific about anatomical features, proportions, age indicators
- Include setting/context when relevant (playground, classroom, etc.)
- For counter-tags (`no_*`): describe what makes the content NOT in the category
  (e.g. "a fantasy werewolf, not a real animal")
- Avoid subjective terms ("inappropriate", "bad", "dangerous")
- Aim for 10–25 words

### Naming the key

Format: `<prefix>_<noun>_<modifier>` or `<prefix>_<descriptor>`

Examples:
- `underage_flat_chest_nude` — category + feature + context
- `no_bestiality_furry_wolf` — counter + sub-type
- `blood_severed_limb` — simple noun phrase
- `underage_adult_with_child:x20` — with multiplier

Keep keys lowercase with underscores. No spaces or special characters except `:xN`.

## Editing tags.json

Tags are stored in `data/tags.json` as a flat JSON object:

```json
{
  "category_key": "visual description for SigLIP-2",
  "category_other_key:x5": "description for boosted tag"
}
```

After editing, always run the full category to check for regressions:
```bash
python scripts/run_category.py --category <category>
```
