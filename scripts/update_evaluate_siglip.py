#!/usr/bin/env python3
"""
Update evaluate_siglip node:
  - Add input `categories: string[] enum` (same as ask_siglip2).
  - In script, build active set from categories and emit details/labels only for active.
  - Top-level pipeline input categories_e0 → wire into evaluate_siglip.categories via inline flows.

Mapping (category in UI → output detail keys):
  underage_slim / underage_full → details.underage
  bestiality                    → details.bestiality (includes furry-as-context)
  furry                         → details.bestiality (furry sub-keys only)
  death                         → details.death
  blood                         → details.blood
  weapons                       → details.weapons
  drugs                         → details.drugs
  rape                          → details.rape
  human                         → details.humanWaste
  ethnic                        → details.asian/ebony/arab/latin/indian (all 5)
  meta, copyright, celebrity    → no evaluate_siglip section; skipped
  all                           → everything
"""
import os, json, re, time, httpx
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
API = 'https://piper-next.artworks.ai/api'

ENUM = ['underage_slim','underage_full','bestiality','death','blood','weapons','drugs',
        'rape','ethnic','furry','copyright','celebrity','human','meta','all']


def build_new_script(old_script):
    """Inject a category-active helper + wrap each output section in `if active()` guard.

    Strategy:
      1. Insert helper functions right after first `function round(value) {...}` block.
      2. Read inputs.categories in run() body.
      3. Replace every `labels.push('X')` + `details.X = {...}` block with conditional emission.
      We do this with regex-based block replacement, anchored to comments like `// ===== UNDERAGE =====`.
    """
    helper_block = """
// === CATEGORY FILTER (added 2026-05-21) ===
const CATEGORY_TO_EVAL = {
  underage:   new Set(['underage_slim','underage_full','all']),
  bestiality: new Set(['bestiality','furry','all']),
  humanWaste: new Set(['human','all']),
  blood:      new Set(['blood','all']),
  death:      new Set(['death','all']),
  weapons:    new Set(['weapons','all']),
  drugs:      new Set(['drugs','all']),
  rape:       new Set(['rape','all']),
  asian:      new Set(['ethnic','all']),
  ebony:      new Set(['ethnic','all']),
  arab:       new Set(['ethnic','all']),
  latin:      new Set(['ethnic','all']),
  indian:     new Set(['ethnic','all']),
};
function evalActive(categories, evalKey) {
  if (!categories || !Array.isArray(categories) || categories.length === 0) {
    categories = ['underage_slim'];
  }
  const allow = CATEGORY_TO_EVAL[evalKey];
  if (!allow) return false;
  for (const c of categories) {
    if (allow.has(c)) return true;
  }
  return false;
}
"""

    # 1) Insert helper right after `function round` block end (before combineScores)
    new_script = old_script.replace(
        "// Probability combination",
        helper_block + "\n// Probability combination"
    )

    # 2) After cfg destructure, read categories input
    new_script = new_script.replace(
        "const cfg = inputs.config ?? {};",
        "const cfg = inputs.config ?? {};\n    const categories = inputs.categories;"
    )

    # 3) Wrap each `if (...) labels.push('X');` / `details.X = {...};` pair with conditional.
    # We'll just wrap each `details.<key> = { ... };` block — push lines are inside same `if` already.
    # Simplest: at the very end (before `return NextNode.from(...)`) prune `details` and `labels`.

    # Replace return statement
    new_script = new_script.replace(
        "return NextNode.from({\n        outputs: { labels, details }\n    });",
        """// Filter by active categories
    const activeLabels = [];
    const LABEL_TO_EVAL = {
      'underage': 'underage', 'bestiality': 'bestiality',
      'human_waste': 'humanWaste', 'blood': 'blood', 'death_and_murder': 'death',
      'weapons_and_military': 'weapons', 'drugs': 'drugs', 'rape': 'rape',
      'asian': 'asian', 'ebony': 'ebony', 'arab': 'arab', 'latin': 'latin', 'indian': 'indian',
    };
    for (const l of labels) {
      const ek = LABEL_TO_EVAL[l];
      if (ek === undefined) { activeLabels.push(l); continue; }
      if (evalActive(categories, ek)) activeLabels.push(l);
    }
    const activeDetails = {};
    for (const ek of Object.keys(details)) {
      if (evalActive(categories, ek)) activeDetails[ek] = details[ek];
    }
    return NextNode.from({
        outputs: { labels: activeLabels, details: activeDetails }
    });"""
    )

    # 4) Bump version comment if present
    if "// v3" in new_script:
        new_script = new_script.replace("// v3:", "// v4 (categories filter):")
    return new_script


def main():
    r = httpx.get(f'{API}/projects/d2911d10bb', headers=HDR, timeout=30).json()
    rev = r['revision']
    pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
    ev = pipe['nodes']['evaluate_siglip']
    print(f'current rev: {rev}')

    old_script = ev['script']
    new_script = build_new_script(old_script)
    if new_script == old_script:
        print('  ERROR: script not changed; replacements failed'); return
    print(f'  script: {len(old_script)} → {len(new_script)} chars')

    # Backup
    backup_dir = BASE / 'backups' / 'piper_evaluate_siglip_filter'
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    (backup_dir / f'd2911d10bb_{ts}.json').write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
    print(f'  backup: backups/piper_evaluate_siglip_filter/d2911d10bb_{ts}.json')

    # 2) Wire categories_e0 top-level input → evaluate_siglip.categories (via inline flow)
    cat_e0 = pipe['inputs'].get('categories_e0', {})
    old_cat = dict(cat_e0)
    new_cat = dict(cat_e0)
    flows = dict(new_cat.get('flows', {}))
    flows['to_evaluate_siglip'] = {'to': 'evaluate_siglip', 'input': 'categories'}
    new_cat['flows'] = flows

    delta = {
        'pipeline': {
            'nodes': {
                'evaluate_siglip': {
                    'script': [old_script, new_script],
                    'inputs': {
                        'categories': [{
                            'order': 6,
                            'title': 'en=Categories;ru=Категории',
                            'type': 'string[]',
                            'enum': ENUM,
                            'required': False,
                        }]
                    },
                }
            },
            'inputs': {
                'categories_e0': [old_cat, new_cat],   # replace
            },
        }
    }

    r2 = httpx.patch(f'{API}/projects/d2911d10bb/patch/{rev}', headers=HDR,
                     content=json.dumps(delta).encode(), timeout=60)
    if r2.status_code != 200:
        print(f'  PATCH failed: {r2.status_code}, {r2.text[:400]}'); return
    print(f'  ✓ new rev: {r2.json().get("revision")}')


if __name__ == '__main__':
    main()
