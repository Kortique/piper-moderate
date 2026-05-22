#!/usr/bin/env python3
"""
Build a new Piper node `prepare_siglip_labels` that selects SigLIP labels
based on a multi-select `categories` input — same UX as `providers` in prepare_params.

Categories (enum):
  underage_slim       (default — for V8pas80 production)
  underage_full
  bestiality, death, blood, weapons, drugs, rape
  ethnic              (asian + indian + latin + arab + ebony)
  furry, copyright, celebrity
  meta                (anime + fantasy + realistic + roleplay)
  all                 (everything)

After this script runs:
  - new node `prepare_siglip_labels` is added to d2911d10bb
  - flow prepare_siglip_labels.labels → ask_siglip2.labels added
  - top-level pipeline input `categories: string[]` added so UI shows it

The default in the new node is ['underage_slim'] — production behaviour unchanged.
"""
import os, json, re, time, sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
API = 'https://piper-next.artworks.ai/api'

DRY = '--dry-run' in sys.argv


# ── Build LABEL_GROUPS taxonomy ────────────────────────────────────────────
def build_taxonomy():
    """Return (ALL_LABELS dict, LABEL_GROUPS dict)."""
    full = json.load(open(BASE / 'data' / 'tags.json'))
    slim = json.load(open(BASE / 'data' / 'd2911d10bb_slim_labels.json'))

    # Make sure ALL labels we might ever need are in the master dict.
    # Combine tags.json + slim (slim has any extra :x20 keys which may be missing from current tags.json)
    all_labels = dict(full)
    for k, v in slim.items():
        if k not in all_labels:
            all_labels[k] = v

    # Helper: prefix-based grouping
    def has_prefix(k, *prefixes):
        return any(k.startswith(p) for p in prefixes)

    groups = {
        'underage_slim':  sorted(slim.keys()),
        'underage_full':  sorted([k for k in all_labels if has_prefix(k, 'underage_', 'adult_', 'no_underage_')]),
        'bestiality':     sorted([k for k in all_labels if has_prefix(k, 'bestiality_', 'no_bestiality_')]),
        'death':          sorted([k for k in all_labels if has_prefix(k, 'death_', 'no_death_')]),
        'blood':          sorted([k for k in all_labels if has_prefix(k, 'blood_', 'no_blood_')]),
        'weapons':        sorted([k for k in all_labels if has_prefix(k, 'weapons_', 'no_weapons_')]),
        'drugs':          sorted([k for k in all_labels if has_prefix(k, 'drugs_', 'no_drugs_')]),
        'rape':           sorted([k for k in all_labels if has_prefix(k, 'rape_', 'no_rape_')]),
        'ethnic':         sorted([k for k in all_labels if has_prefix(k, 'asian_', 'indian_', 'latin_', 'arab_', 'ebony_')]),
        'furry':          sorted([k for k in all_labels if has_prefix(k, 'furry_')]),
        'copyright':      sorted([k for k in all_labels if has_prefix(k, 'copyright_')]),
        'celebrity':      sorted([k for k in all_labels if has_prefix(k, 'celebrity_')]),
        'human':          sorted([k for k in all_labels if has_prefix(k, 'human_')]),
        'meta':           sorted([k for k in all_labels if has_prefix(k, 'anime_', 'fantasy_', 'realistic_', 'roleplay_')]),
    }
    groups['all'] = sorted(all_labels.keys())

    # Print stats
    print('LABEL_GROUPS sizes:', flush=True)
    for cat, keys in groups.items():
        print(f'  {cat:<14}: {len(keys)}', flush=True)
    print(f'\nALL_LABELS total: {len(all_labels)}', flush=True)

    return all_labels, groups


# ── JS for prepare_siglip_labels node ──────────────────────────────────────
def build_node_script(all_labels, groups):
    enum_list = list(groups.keys())
    return f"""// prepare_siglip_labels — selects SigLIP label subset based on `categories` input.
// Mimics the UX of `providers` in prepare_params.
// Default: ['underage_slim'] — production behaviour preserved.

const ALL_LABELS = {json.dumps(all_labels, ensure_ascii=False)};

const LABEL_GROUPS = {json.dumps(groups, ensure_ascii=False)};

export async function run({{ inputs }}) {{
  const {{ NextNode }} = DEFINITIONS;
  let {{ categories }} = inputs;
  if (!categories || !Array.isArray(categories) || categories.length === 0) {{
    categories = ['underage_slim'];
  }}
  const selected = new Set();
  for (const cat of categories) {{
    const keys = LABEL_GROUPS[cat] || [];
    for (const k of keys) selected.add(k);
  }}
  const labels = {{}};
  for (const k of selected) {{
    if (k in ALL_LABELS) labels[k] = ALL_LABELS[k];
  }}
  return NextNode.from({{ outputs: {{ labels, count: Object.keys(labels).length }} }});
}}
// v1
"""


# ── Build delta to add the node + flow + top-level input ───────────────────
def build_delta(all_labels, groups, current_pipe):
    enum_list = list(groups.keys())
    node_script = build_node_script(all_labels, groups)

    # New node spec
    new_node = {
        'version': 1,
        'title': 'Prepare SigLIP labels',
        'execution': 'rapid',
        'script': node_script,
        'arrange': {'x': 600, 'y': 280},
        'inputs': {
            'categories': {
                'order': 1,
                'title': 'en=Categories;ru=Категории',
                'type': 'string[]',
                'enum': enum_list,
                # No default — script falls back to ['underage_slim'] if empty
            },
        },
        'outputs': {
            'labels': {'title': 'Labels', 'type': 'json'},
            'count':  {'title': 'Count',  'type': 'integer'},
        },
        'environment': {},
    }

    # New flow: prepare_siglip_labels.labels → ask_siglip2.labels
    new_flow_key = 'prepare_siglip_labels_to_ask_siglip2_labels'
    new_flow = {'from': 'prepare_siglip_labels', 'output': 'labels',
                'to': 'ask_siglip2', 'input': 'labels'}

    # Top-level pipeline input `categories` (no default — pipeline schema disallows array defaults at top level)
    top_input = {
        'type': 'string[]',
        'required': False,
        'order': 5,
        'enum': enum_list,
    }

    delta = {
        'pipeline': {
            'nodes': {
                'prepare_siglip_labels': [new_node],  # add new node
            },
            'flows': {
                new_flow_key: [new_flow],  # add new flow
            },
            'inputs': {
                'categories': [top_input],   # add top-level input
            },
        }
    }
    return delta, new_flow_key


def main():
    print('Building taxonomy...', flush=True)
    all_labels, groups = build_taxonomy()

    print('\nFetching current pipeline...', flush=True)
    r = httpx.get(f'{API}/projects/d2911d10bb', headers=HDR, timeout=30).json()
    rev = r['revision']
    pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
    print(f'  current rev: {rev}', flush=True)

    # Check if our node already exists — if so, fail fast
    if 'prepare_siglip_labels' in pipe.get('nodes', {}):
        print('  Node prepare_siglip_labels already exists. Use update path.', flush=True)
        # Update path: replace script and inputs
        node = pipe['nodes']['prepare_siglip_labels']
        old_script = node.get('script', '')
        new_script = build_node_script(all_labels, groups)
        delta = {'pipeline': {'nodes': {'prepare_siglip_labels': {'script': [old_script, new_script]}}}}
    else:
        delta, _ = build_delta(all_labels, groups, pipe)

    # Backup
    backup_dir = BASE / 'backups' / 'piper_categories'
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    (backup_dir / f'd2911d10bb_{ts}.json').write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
    print(f'  backup: backups/piper_categories/d2911d10bb_{ts}.json', flush=True)

    # Save delta for inspection
    (backup_dir / f'delta_{ts}.json').write_text(json.dumps(delta, ensure_ascii=False, indent=2)[:5000])
    print(f'  delta preview saved (5KB head): backups/piper_categories/delta_{ts}.json', flush=True)

    if DRY:
        print('  [DRY-RUN]', flush=True)
        return

    r = httpx.patch(f'{API}/projects/d2911d10bb/patch/{rev}',
                    headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    if r.status_code != 200:
        print(f'  PATCH failed: HTTP {r.status_code}', flush=True)
        print(f'  Body: {r.text[:500]}', flush=True)
        return
    new_rev = r.json().get('revision')
    print(f'  ✓ new rev: {new_rev}', flush=True)


if __name__ == "__main__":
    main()
