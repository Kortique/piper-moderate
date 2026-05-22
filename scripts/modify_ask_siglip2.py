#!/usr/bin/env python3
"""
Modify ask_siglip2 node in d2911d10bb:
  - Add input `categories: string[] enum` (same pattern as `providers` in prepare_params).
  - Replace labels.default with FULL master tag set (873 keys).
  - Embed LABEL_GROUPS into ask_siglip2 script.
  - Script filters labels by selected categories before sending to PaaS.

Backwards compatibility:
  - If `categories` is empty/undefined → uses all labels (full master set).
  - Payload override `inputs.categories=[...]` works directly per-launch.

Result: UI in Piper Studio shows `Categories` multi-select on ask_siglip2 node.
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


def build_taxonomy():
    full = json.load(open(BASE / 'data' / 'tags.json'))
    slim = json.load(open(BASE / 'data' / 'd2911d10bb_slim_labels.json'))
    all_labels = dict(full)
    for k, v in slim.items():
        all_labels.setdefault(k, v)

    has_prefix = lambda k, *ps: any(k.startswith(p) for p in ps)
    groups = {
        'underage_slim': sorted(slim.keys()),
        'underage_full': sorted([k for k in all_labels if has_prefix(k, 'underage_', 'adult_', 'no_underage_')]),
        'bestiality':    sorted([k for k in all_labels if has_prefix(k, 'bestiality_', 'no_bestiality_')]),
        'death':         sorted([k for k in all_labels if has_prefix(k, 'death_', 'no_death_')]),
        'blood':         sorted([k for k in all_labels if has_prefix(k, 'blood_', 'no_blood_')]),
        'weapons':       sorted([k for k in all_labels if has_prefix(k, 'weapons_', 'no_weapons_')]),
        'drugs':         sorted([k for k in all_labels if has_prefix(k, 'drugs_', 'no_drugs_')]),
        'rape':          sorted([k for k in all_labels if has_prefix(k, 'rape_', 'no_rape_')]),
        'ethnic':        sorted([k for k in all_labels if has_prefix(k, 'asian_', 'indian_', 'latin_', 'arab_', 'ebony_')]),
        'furry':         sorted([k for k in all_labels if has_prefix(k, 'furry_')]),
        'copyright':     sorted([k for k in all_labels if has_prefix(k, 'copyright_')]),
        'celebrity':     sorted([k for k in all_labels if has_prefix(k, 'celebrity_')]),
        'human':         sorted([k for k in all_labels if has_prefix(k, 'human_')]),
        'meta':          sorted([k for k in all_labels if has_prefix(k, 'anime_', 'fantasy_', 'realistic_', 'roleplay_')]),
    }
    groups['all'] = sorted(all_labels.keys())
    return all_labels, groups


def build_script(groups, base_version):
    """Build new ask_siglip2 script with category filtering on top of original logic."""
    groups_js = json.dumps(groups, ensure_ascii=False)
    new_v = base_version + 1
    return f"""const CHECK_TASK_INTERVAL = 2000;
const MAX_ATTEMPTS = 60;

// Categories filter: when `inputs.categories` is non-empty, restrict the SigLIP
// labels.default master set down to keys in the selected groups. Default = all.
const LABEL_GROUPS = {groups_js};

function filterLabelsByCategories(labels, categories) {{
  if (!categories || !Array.isArray(categories) || categories.length === 0) {{
    return labels;  // no filter — use full master set
  }}
  const allow = new Set();
  for (const cat of categories) {{
    const keys = LABEL_GROUPS[cat] || [];
    for (const k of keys) allow.add(k);
  }}
  if (allow.size === 0) return labels;
  const out = {{}};
  for (const k of Object.keys(labels)) {{
    if (allow.has(k)) out[k] = labels[k];
  }}
  return out;
}}

export async function costs() {{
    return 0.00004;
}}

export async function run({{ inputs, state }}) {{

    const {{ FatalError, TimeoutError, RepeatNode, NextNode }} = DEFINITIONS;
    const {{ ArtWorks, FatalError: ArtWorksError }} = require('artworks');

    const PAAS_BASE_URL = env.variables.get('PAAS_BASE_URL');
    if (!PAAS_BASE_URL) {{
        throw new FatalError('Please, set PAAS_BASE_URL in environment');
    }}
    const OPEN_PAAS_USER = env.variables.get('OPEN_PAAS_USER');
    if (!OPEN_PAAS_USER) {{
        throw new FatalError('Please, set OPEN_PAAS_USER in environment');
    }}
    const OPEN_PAAS_PASSWORD = env.variables.get('OPEN_PAAS_PASSWORD');
    if (!OPEN_PAAS_PASSWORD) {{
        throw new FatalError('Please, set OPEN_PAAS_PASSWORD in environment');
    }}

    const artworks = new ArtWorks({{
        baseUrl: PAAS_BASE_URL,
        username: OPEN_PAAS_USER,
        password: OPEN_PAAS_PASSWORD
    }});

    if (!state) {{
        const {{ image, model, labels: rawLabels, categories }} = inputs;
        const labels = filterLabelsByCategories(rawLabels, categories);

        const payload = {{
            type: "classify-image",
            isFast: true,
            payload: {{
                base64: false,
                image,
                model,
                labels: Object.values(labels),
            }},
        }};
        try {{
            const task = await artworks.createTask(payload);
            console.log(`Task created ${{task}} (labels=${{Object.keys(labels).length}}/${{Object.keys(rawLabels).length}}, cats=${{(categories || []).join(',')}})`);
            return RepeatNode.from({{
                state: {{
                    payload,
                    task,
                    labelsKeys: Object.keys(labels),
                    attempt: 0,
                    startedAt: new Date()
                }},
                progress: {{
                    total: MAX_ATTEMPTS,
                    processed: 0
                }},
                delay: 2000
            }});
        }} catch (e) {{
            if (e instanceof ArtWorksError) {{
                throw new FatalError(e.message);
            }}
            throw e;
        }}
    }} else {{
        const {{
            payload,
            task,
            labelsKeys,
            attempt,
            startedAt
        }} = state;

        if (attempt > MAX_ATTEMPTS) {{
            try {{
                await artworks.cancelTask(task);
            }} catch (e) {{ }}

            const now = new Date();
            const time = (now - new Date(startedAt)) / 1000;
            throw new TimeoutError(`PaaS task for text to image ${{task}} timeout in ${{time}} sec`);
        }}

        console.log(`Check task ${{attempt}} ${{task}}`);

        try {{
            const results = await artworks.checkState(task);
            if (!results) {{
                return RepeatNode.from({{
                    delay: CHECK_TASK_INTERVAL,
                    state: {{
                        payload,
                        task,
                        labelsKeys,
                        attempt: attempt + 1,
                        startedAt,
                    }},
                    progress: {{
                        total: MAX_ATTEMPTS,
                        processed: attempt
                    }},
                }});
            }}
            let {{ probs }} = results;
            // Re-derive {{key: prob}} dict from the labels we actually sent (labelsKeys + payload.payload.labels in same order).
            const sentValues = payload.payload.labels;
            const tags = {{}};
            for (let i = 0; i < labelsKeys.length; i++) {{
                const k = labelsKeys[i];
                const desc = sentValues[i];
                tags[k] = probs[desc];
            }}
            return NextNode.from({{
                outputs: {{ labels: tags }},
                costs: costs({{ inputs }}),
            }});
        }} catch (e) {{
            if (e instanceof ArtWorksError) {{
                throw new FatalError(e.message);
            }}
            throw e;
        }}
    }}
}}
// v{new_v}
"""


def main():
    all_labels, groups = build_taxonomy()
    enum_list = list(groups.keys())
    print(f'LABEL_GROUPS: {len(groups)} groups, ALL_LABELS: {len(all_labels)}', flush=True)

    r = httpx.get(f'{API}/projects/d2911d10bb', headers=HDR, timeout=30).json()
    rev = r['revision']
    pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
    sig = pipe['nodes']['ask_siglip2']
    print(f'current rev: {rev}', flush=True)

    # Detect current version
    m = re.search(r'// v(\d+)\s*$', sig['script'])
    base_v = int(m.group(1)) if m else 20

    old_script = sig['script']
    new_script = build_script(groups, base_v)
    old_labels = sig['inputs']['labels']['default']
    if not isinstance(old_labels, str):
        old_labels = json.dumps(old_labels, ensure_ascii=False, indent=2)
    new_labels_str = json.dumps(all_labels, ensure_ascii=False, indent=2)

    # Backup
    backup_dir = BASE / 'backups' / 'piper_categories_v2'
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    (backup_dir / f'd2911d10bb_{ts}.json').write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
    print(f'backup: backups/piper_categories_v2/d2911d10bb_{ts}.json', flush=True)

    delta = {
        'pipeline': {
            'nodes': {
                'ask_siglip2': {
                    'script': [old_script, new_script],
                    'inputs': {
                        # ADD: new input categories
                        'categories': [{
                            'order': 4,
                            'title': 'en=Categories;ru=Категории',
                            'type': 'string[]',
                            'enum': enum_list,
                            'required': False,
                        }],
                        # REPLACE: master labels set
                        'labels': {
                            'default': [old_labels, new_labels_str],
                        },
                    },
                }
            }
        }
    }

    print(f'script: {len(old_script)} → {len(new_script)} chars', flush=True)
    print(f'labels.default: {len(json.loads(old_labels))} → {len(all_labels)} keys', flush=True)
    print(f'new input "categories" with enum: {enum_list}', flush=True)

    if DRY:
        print('  [DRY-RUN]'); return

    r = httpx.patch(f'{API}/projects/d2911d10bb/patch/{rev}',
                    headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    if r.status_code != 200:
        print(f'  PATCH failed HTTP {r.status_code}: {r.text[:500]}'); return
    print(f'  ✓ new rev: {r.json().get("revision")}')


if __name__ == '__main__':
    main()
