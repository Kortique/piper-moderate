#!/usr/bin/env python3
"""
check_piper_pipeline.py — diagnostic for Piper pipeline structure.

Dumps everything our scripts care about for one or more Piper projects:
  • top-level pipeline inputs (with names, types, enum values, defaults)
  • prepare_params node (where the JS providers-splitter lives)
  • all input fields whose name or title looks like 'provider*'
  • current pipeline revision (needed for PATCH operations)

Use this BEFORE editing any provider/payload key in moderate_* / rescore_*
scripts — different pipelines name their top-level providers input differently
(d2911d10bb / ce79f7e299 use `providers_e0`, a4aa9dbd9c uses plain `providers`).

Usage:
    python scripts/check_piper_pipeline.py d2911d10bb
    python scripts/check_piper_pipeline.py d2911d10bb ce79f7e299 a4aa9dbd9c
    python scripts/check_piper_pipeline.py --all
"""
import argparse, json, os, sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')

PIPER_BASE = 'https://piper-next.artworks.ai/api'
TOKEN      = os.getenv('PIPER_TOKEN', '')

# Known projects we use across the repo. --all dumps these.
KNOWN_PROJECTS = {
    'd2911d10bb': 'moderation pipeline (siglip2 + qwen3 + face_detect → LGBM Underage)',
    'ce79f7e299': 'V11 native scoring (siglip2 only → V11 LGBM)',
    'a4aa9dbd9c': 'Tom K30 (siglip2 only → K30 LGBM)',
    'b2fb1af977': 'legacy moderation (older default)',
    '9cd1798843': 'V9 rescore (317-tag for V11 training)',
}


def _hdr():
    return {'User-Token': TOKEN, 'Accept': 'application/json'}


def fetch(project_id):
    r = httpx.get(f'{PIPER_BASE}/projects/{project_id}',
                  headers=_hdr(), timeout=30)
    r.raise_for_status()
    data = r.json()
    pipe = data.get('pipeline')
    if isinstance(pipe, str):
        pipe = json.loads(pipe)
    return data, pipe


def dump_project(project_id):
    print(f'\n{"="*70}\n{project_id}  —  {KNOWN_PROJECTS.get(project_id, "(unknown)")}\n{"="*70}')
    try:
        data, pipe = fetch(project_id)
    except Exception as e:
        print(f'  ERR: {type(e).__name__}: {str(e)[:200]}')
        return

    print(f'  revision: {data.get("revision")}')
    print(f'  pipeline title: {data.get("title")}')

    top_inputs = (pipe or {}).get('inputs') or {}
    nodes      = (pipe or {}).get('nodes')  or {}
    prep       = nodes.get('prepare_params') or {}

    # ── Provider-like inputs at top level ───────────────────────────────────
    print(f'\n  TOP-LEVEL INPUTS related to providers:')
    found_any_provider = False
    for k, v in top_inputs.items():
        title = (v.get('title') or '') if isinstance(v, dict) else ''
        if 'rovider' in k.lower() or 'rovider' in title.lower():
            found_any_provider = True
            enum = v.get('enum') if isinstance(v, dict) else None
            default = v.get('default') if isinstance(v, dict) else None
            ttype = v.get('type') if isinstance(v, dict) else None
            flows = (v.get('flows') if isinstance(v, dict) else None) or {}
            flow_target = next(
                (f'{f.get("to")}.{f.get("input")}' for f in flows.values()
                 if isinstance(f, dict)),
                None
            )
            print(f'    key:        {k!r}')
            print(f'      title:    {title!r}')
            print(f'      type:     {ttype!r}')
            print(f'      enum:     {enum}')
            print(f'      default:  {default!r}')
            print(f'      flows to: {flow_target}')
    if not found_any_provider:
        print('    (none found — pipeline may not expose providers as a top-level input)')

    # ── All other top-level inputs (names + types only, compact) ────────────
    other_keys = [k for k in top_inputs.keys()
                  if 'rovider' not in k.lower()
                  and 'rovider' not in (
                      (top_inputs[k].get('title') if isinstance(top_inputs[k], dict) else '') or ''
                  ).lower()]
    if other_keys:
        print(f'\n  OTHER top-level inputs ({len(other_keys)}):')
        for k in other_keys:
            v = top_inputs[k]
            ttype = v.get('type') if isinstance(v, dict) else type(v).__name__
            req   = (v.get('required') if isinstance(v, dict) else None)
            tag   = ' [required]' if req else ''
            print(f'    {k:<22} type={ttype}{tag}')

    # ── prepare_params node (the JS providers-splitter) ─────────────────────
    print(f'\n  prepare_params node:')
    if not prep:
        print('    (not found in this pipeline)')
    else:
        prep_inputs = (prep.get('inputs') or {})
        prep_providers = prep_inputs.get('providers')
        if isinstance(prep_providers, dict):
            print(f'    inputs.providers.enum: {prep_providers.get("enum")}')
            if 'default' in prep_providers:
                print(f'    inputs.providers.default: {prep_providers.get("default")!r}')
        # Try to extract the JS that handles providers
        script = prep.get('script') or ''
        if 'providers' in script:
            print(f'\n    --- script snippet (lines with "provider") ---')
            for ln in script.split('\n'):
                if 'rovider' in ln:
                    print(f'    | {ln.strip()[:140]}')

    # ── Summary recommendation ──────────────────────────────────────────────
    print(f'\n  RECOMMENDATION:')
    if 'providers_e0' in top_inputs:
        print('    → payload key: "providers_e0"  (top-level input uses _e0 suffix)')
    elif 'providers' in top_inputs:
        print('    → payload key: "providers"  (top-level input is plain)')
    else:
        print('    → no top-level providers input found; payload may not affect provider list.')
        print('      Pipeline will use whatever defaults the prepare_params script falls back to.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('projects', nargs='*',
                    help='Piper project IDs to inspect (e.g. d2911d10bb)')
    ap.add_argument('--all', action='store_true',
                    help=f'Dump all known projects: {list(KNOWN_PROJECTS)}')
    args = ap.parse_args()

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    projects = list(KNOWN_PROJECTS) if args.all else args.projects
    if not projects:
        ap.error('specify project IDs or --all')

    for pid in projects:
        dump_project(pid)
    print()


if __name__ == '__main__':
    main()
