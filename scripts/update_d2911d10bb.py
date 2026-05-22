#!/usr/bin/env python3
"""
update_d2911d10bb.py
--------------------
Создаёт новый Piper проект на основе d2911d10bb с обновлёнными тегами из tags.json
(без roleplay_*) и патчем evaluate_siglip для хранения no_underage labels.

Usage:
    python scripts/update_d2911d10bb.py            # dry-run: показать что изменится
    python scripts/update_d2911d10bb.py --apply    # создать новый проект в Piper

Outputs:
    data/d2911d10bb_current.json   — текущий pipeline (сохранённый)
    data/d2911d10bb_updated.json   — модифицированный (для проверки)
    data/v9_test_project_id.txt    — ID нового проекта (при --apply)
"""
import os, sys, json, copy
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()
BASE_DIR   = Path(__file__).resolve().parent.parent
TOKEN      = os.getenv('PIPER_TOKEN', '')
PIPER_BASE = 'https://piper-next.artworks.ai/api'
SOURCE_ID  = 'd2911d10bb'
TAGS_FILE  = BASE_DIR / 'data' / 'tags.json'


def hdr():
    return {
        'User-Token': TOKEN,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }


def get_project(project_id):
    r = httpx.get(f'{PIPER_BASE}/projects/{project_id}', headers=hdr(), timeout=30)
    r.raise_for_status()
    return r.json()


def load_tags():
    """Load tags.json, remove roleplay_* entries."""
    with open(TAGS_FILE, encoding='utf-8') as f:
        tags = json.load(f)
    original_count = len(tags)
    tags = {k: v for k, v in tags.items() if not k.startswith('roleplay_')}
    removed = original_count - len(tags)
    print(f'  tags.json: {original_count} total → {len(tags)} after removing {removed} roleplay_* tags')
    return tags


def patch_evaluate_siglip(script: str) -> str:
    """
    Добавляет no_underage_* поддержку в секцию UNDERAGE скрипта evaluate_siglip.

    Изменения:
    1. После ADULT_LABELS/adultConfidence добавляем NO_UNDERAGE_LABELS + noUnderageConfidence
    2. noUnderage добавляется к totalUnderageScore (снижает underageConfidence при срабатывании)
    3. noUnderage score сохраняется в details.underage
    4. no_underage labels сохраняются в details.underage.labels
    """
    if 'NO_UNDERAGE_LABELS' in script:
        print('  evaluate_siglip already patched — skipping')
        return script

    changed = False

    # 1. Add NO_UNDERAGE_LABELS after adultConfidence line
    old_adult = (
        '    const ADULT_LABELS = Object.keys(affected).filter(k => k.startsWith("adult"));\n'
        '    const adultConfidence = combineScores(affected, ADULT_LABELS);'
    )
    new_adult = (
        '    const ADULT_LABELS = Object.keys(affected).filter(k => k.startsWith("adult"));\n'
        '    const adultConfidence = combineScores(affected, ADULT_LABELS);\n'
        '\n'
        '    const NO_UNDERAGE_LABELS = Object.keys(affected).filter(k => k.startsWith("no_underage"));\n'
        '    const noUnderageConfidence = combineScores(affected, NO_UNDERAGE_LABELS);'
    )
    if old_adult in script:
        script = script.replace(old_adult, new_adult, 1)
        changed = True
        print('  ✓ Added NO_UNDERAGE_LABELS + noUnderageConfidence')
    else:
        print('  ✗ ADULT_LABELS block not found — check script formatting')
        return script

    # 2. Add noUnderage to totalUnderageScore
    old_total = '    const totalUnderageScore = minorRisk + adultConfidence;'
    new_total  = '    const totalUnderageScore = minorRisk + adultConfidence + noUnderageConfidence;'
    if old_total in script:
        script = script.replace(old_total, new_total, 1)
        print('  ✓ totalUnderageScore includes noUnderageConfidence')
    else:
        print('  ✗ totalUnderageScore line not found')

    # 3. Add noUnderage score to details.underage object
    #    Find "minor: round(minorRisk)," and add noUnderage after it
    #    Live script uses aligned spacing: "minor:      round(minorRisk),"
    import re
    minor_match = re.search(r'(        minor:\s+round\(minorRisk\),)', script)
    if minor_match:
        old_minor = minor_match.group(1)
        new_minor  = old_minor + '\n        noUnderage: round(noUnderageConfidence),'
        script = script.replace(old_minor, new_minor, 1)
        print('  ✓ Added noUnderage score to details.underage')
    else:
        print('  ✗ minor score line not found in details.underage')

    # 4. Patch labels block (handles aligned formatting used in live script)
    #    Live script uses:
    #      "          underage: getLabels(...),\n            adult:    getLabels(...)"
    old_labels_aligned = (
        '          underage: getLabels(affected, MINOR_LABELS, "underage_", MIN_LABEL_DISPLAY_SCORE),\n'
        '            adult:    getLabels(affected, ADULT_LABELS, "adult_",    MIN_LABEL_DISPLAY_SCORE)'
    )
    new_labels_aligned = (
        '          underage:    getLabels(affected, MINOR_LABELS,       "underage_",    MIN_LABEL_DISPLAY_SCORE),\n'
        '            adult:       getLabels(affected, ADULT_LABELS,       "adult_",       MIN_LABEL_DISPLAY_SCORE),\n'
        '            no_underage: getLabels(affected, NO_UNDERAGE_LABELS, "no_underage_", MIN_LABEL_DISPLAY_SCORE)'
    )

    # Unaligned formatting (from evaluate_siglip_v3.js local file)
    old_labels_plain = (
        '            underage: getLabels(affected, MINOR_LABELS, "underage_", MIN_LABEL_DISPLAY_SCORE),\n'
        '            adult: getLabels(affected, ADULT_LABELS, "adult_", MIN_LABEL_DISPLAY_SCORE)'
    )
    new_labels_plain = (
        '            underage: getLabels(affected, MINOR_LABELS, "underage_", MIN_LABEL_DISPLAY_SCORE),\n'
        '            adult: getLabels(affected, ADULT_LABELS, "adult_", MIN_LABEL_DISPLAY_SCORE),\n'
        '            no_underage: getLabels(affected, NO_UNDERAGE_LABELS, "no_underage_", MIN_LABEL_DISPLAY_SCORE)'
    )

    if old_labels_aligned in script:
        script = script.replace(old_labels_aligned, new_labels_aligned, 1)
        print('  ✓ Patched labels block (aligned format)')
    elif old_labels_plain in script:
        script = script.replace(old_labels_plain, new_labels_plain, 1)
        print('  ✓ Patched labels block (plain format)')
    else:
        # Fallback: search for the adult getLabels line and append after it
        idx = script.find('getLabels(affected, ADULT_LABELS, "adult_"')
        if idx > 0:
            # Find end of that line
            end = script.find('\n', idx)
            # Check if line ends with }, or , — we need to add comma + new line
            line = script[idx:end]
            if not line.rstrip().endswith(','):
                # Need to add comma before inserting
                no_underage_line = '\n            no_underage: getLabels(affected, NO_UNDERAGE_LABELS, "no_underage_", MIN_LABEL_DISPLAY_SCORE)'
                script = script[:end] + ',' + no_underage_line + script[end:]
                print('  ✓ Patched labels block (fallback insert)')
            else:
                no_underage_line = '\n            no_underage: getLabels(affected, NO_UNDERAGE_LABELS, "no_underage_", MIN_LABEL_DISPLAY_SCORE)'
                script = script[:end] + no_underage_line + script[end:]
                print('  ✓ Patched labels block (fallback append)')
        else:
            print('  ✗ Could not find adult labels line to patch!')

    return script


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Create new project in Piper')
    args = parser.parse_args()

    print(f'1. Fetching project {SOURCE_ID}...')
    project = get_project(SOURCE_ID)
    (BASE_DIR / 'data' / f'{SOURCE_ID}_current.json').write_text(
        json.dumps(project, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'   Saved to data/{SOURCE_ID}_current.json')

    pipeline = project.get('pipeline', {})
    nodes = pipeline.get('nodes', {})
    print(f'   Nodes: {list(nodes.keys())}')

    print('\n2. Loading tags.json...')
    new_tags = load_tags()
    print(f'   New tag count: {len(new_tags)}')
    no_underage_count = sum(1 for k in new_tags if k.startswith('no_underage_'))
    print(f'   no_underage_* tags: {no_underage_count}')

    print('\n3. Updating project...')
    updated_project = copy.deepcopy(project)

    # Remove fields that must not be sent when creating a new project
    for field in ('_id', 'createdAt', 'createdBy', 'updatedAt', 'updatedBy', 'cursor', 'revision'):
        updated_project.pop(field, None)

    # Update title
    old_title = updated_project.get('title', '')
    new_title = f'{old_title} [V9-test]' if '[V9' not in (old_title or '') else old_title
    updated_project['title'] = new_title
    print(f'   Title: {repr(old_title)} → {repr(new_title)}')

    upd_pipeline = updated_project['pipeline']
    upd_nodes = upd_pipeline['nodes']

    # — Update ask_siglip2 tags —
    if 'ask_siglip2' in upd_nodes:
        labels_field = upd_nodes['ask_siglip2']['inputs']['labels']
        current_tags_str = labels_field.get('default', '{}')
        current_tags = json.loads(current_tags_str)
        print(f'   ask_siglip2 tags: {len(current_tags)} → {len(new_tags)}')
        # The default field is a JSON string
        labels_field['default'] = json.dumps(new_tags, ensure_ascii=False)
    else:
        print('   ✗ ask_siglip2 node not found!')

    # — Patch evaluate_siglip script —
    if 'evaluate_siglip' in upd_nodes:
        ev_node = upd_nodes['evaluate_siglip']
        for sk in ('script', 'source', 'code'):
            if sk in ev_node and ev_node[sk]:
                print(f'   Patching evaluate_siglip.{sk} ({len(ev_node[sk])} chars)...')
                ev_node[sk] = patch_evaluate_siglip(ev_node[sk])
                print(f'   After patch: {len(ev_node[sk])} chars')
                break
    else:
        print('   ✗ evaluate_siglip node not found!')

    # Save updated project for inspection
    out_path = BASE_DIR / 'data' / f'{SOURCE_ID}_updated.json'
    out_path.write_text(json.dumps(updated_project, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n4. Saved to data/{SOURCE_ID}_updated.json')

    # Verify patch in saved JSON
    saved = json.loads(out_path.read_text())
    saved_nodes = saved['pipeline']['nodes']
    saved_labels_count = len(json.loads(saved_nodes['ask_siglip2']['inputs']['labels']['default']))
    saved_script = saved_nodes['evaluate_siglip'].get('script', '')
    has_no_underage = 'NO_UNDERAGE_LABELS' in saved_script and 'no_underage: getLabels' in saved_script
    print(f'   Verify tags: {saved_labels_count} ✓' if saved_labels_count == len(new_tags) else f'   Verify tags: {saved_labels_count} ✗ (expected {len(new_tags)})')
    print(f'   Verify no_underage patch: {"✓" if has_no_underage else "✗"}')

    if not args.apply:
        print('\n[DRY RUN] Pass --apply to create the project in Piper')
        return

    print('\n5. Creating new project in Piper...')
    # Remove pipeline-level fields that may cause issues
    for f in ('deploy',):
        updated_project.get('pipeline', {}).pop(f, None)

    r = httpx.post(f'{PIPER_BASE}/projects', headers=hdr(),
                   json=updated_project, timeout=60)
    print(f'   Status: {r.status_code}')
    if r.status_code not in (200, 201):
        print(f'   Error body: {r.text[:1000]}')
        sys.exit(1)

    new_project = r.json()
    new_id = new_project.get('_id') or new_project.get('id')
    print(f'\n✅ New project created: {new_id}')
    print(f'   URL: https://piper-next.artworks.ai/en/projects/{new_id}')

    (BASE_DIR / 'data' / 'v9_test_project_id.txt').write_text(new_id, encoding='utf-8')
    print(f'   Saved to data/v9_test_project_id.txt')
    print(f'\nNext:')
    print(f'   python scripts/rescore_v9_317session.py --project {new_id}')


if __name__ == '__main__':
    main()
