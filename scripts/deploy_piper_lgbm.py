#!/usr/bin/env python3
"""
deploy_piper_lgbm.py
--------------------
Generic Piper LGBM deploy with verbose diagnostics on auth/network errors.

Usage:
    python scripts/deploy_piper_lgbm.py d2911d10bb data/lgbm_evaluate_v8cs80.js --threshold 0.51 --dry-run
    python scripts/deploy_piper_lgbm.py d2911d10bb data/lgbm_evaluate_v8cs80.js --threshold 0.51
"""
import argparse, json, os, sys, time
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
API = 'https://piper-next.artworks.ai/api'


def diag_response(r, what):
    """Print useful diagnostics when a response is not the expected JSON."""
    print(f'\n  ERR while {what}', file=sys.stderr)
    print(f'    HTTP status: {r.status_code}', file=sys.stderr)
    print(f'    Content-Type: {r.headers.get("content-type", "?")}', file=sys.stderr)
    body = r.text or ''
    head = body[:600]
    print(f'    body (first 600 chars):\n      {head!r}', file=sys.stderr)
    if r.status_code == 401:
        print(f'\n    -> PIPER_TOKEN is invalid or expired. Refresh it from piper-next UI.', file=sys.stderr)
    elif r.status_code == 404:
        print(f'\n    -> project not found or endpoint changed.', file=sys.stderr)
    elif r.status_code == 403:
        print(f'\n    -> token lacks permission on this project.', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('project_id')
    ap.add_argument('js_file')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    token = os.getenv('PIPER_TOKEN')
    if not token:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)
    hdr = {'User-Token': token, 'Content-Type': 'application/json',
           'Accept': 'application/json'}

    js_path = Path(args.js_file)
    if not js_path.exists():
        print(f'ERR: {js_path} not found', file=sys.stderr); sys.exit(1)
    new_js = js_path.read_text()

    print(f'=== Deploy {js_path.name} -> {args.project_id} ===', flush=True)
    url = f'{API}/projects/{args.project_id}'
    print(f'  GET {url}', flush=True)

    try:
        r = httpx.get(url, headers=hdr, timeout=30)
    except Exception as e:
        print(f'  NETWORK ERROR: {type(e).__name__}: {e}', file=sys.stderr); sys.exit(1)

    if r.status_code != 200:
        diag_response(r, 'fetching project')
        sys.exit(1)

    try:
        data = r.json()
    except Exception as e:
        diag_response(r, f'parsing project JSON ({e})')
        sys.exit(1)

    rev = data.get('revision')
    pipe_raw = data.get('pipeline')
    if rev is None or pipe_raw is None:
        print(f'  ERR: response has no revision/pipeline. Keys: {list(data.keys())}', file=sys.stderr)
        sys.exit(1)
    pipe = json.loads(pipe_raw) if isinstance(pipe_raw, str) else pipe_raw
    print(f'  current revision: {rev}', flush=True)

    backup_dir = BASE / 'backups' / f'piper_{args.project_id}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'{args.project_id}_{ts}.json'
    backup_path.write_text(json.dumps(pipe, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  backup -> {backup_path}', flush=True)

    nodes = pipe.get('nodes', {})
    if 'lgbm_evaluate' not in nodes:
        print(f'  ERR: pipeline nodes: {list(nodes.keys())}', file=sys.stderr); sys.exit(1)
    old_js = nodes['lgbm_evaluate'].get('script', '')
    print(f'  lgbm_evaluate.script: {len(old_js)} -> {len(new_js)} chars', flush=True)

    delta_nodes = {'lgbm_evaluate': {'script': [old_js, new_js]}}
    delta_inputs = {}
    if args.threshold is not None:
        # Path 1: legacy "lgbm" node with `threshold` field (older pipelines)
        if 'lgbm' in nodes and 'threshold' in nodes.get('lgbm', {}):
            old_thr = nodes['lgbm']['threshold']
            print(f'  lgbm.threshold (node): {old_thr} -> {args.threshold}', flush=True)
            delta_nodes['lgbm'] = {'threshold': [old_thr, args.threshold]}
        # Path 2: modern pipelines (d2911d10bb-style) expose threshold as a
        # pipeline-level input (`pipeline.inputs.lgbm_threshold.default`).
        # PATCH that field so default applies even when a launch payload omits
        # `inputs.lgbm_threshold`. Both paths are independent — apply whichever
        # exists.
        inputs = pipe.get('inputs', {})
        if 'lgbm_threshold' in inputs:
            old_thr = inputs['lgbm_threshold'].get('default')
            if old_thr != args.threshold:
                print(f'  inputs.lgbm_threshold.default: {old_thr} -> {args.threshold}', flush=True)
                delta_inputs['lgbm_threshold'] = {'default': [old_thr, args.threshold]}
            else:
                print(f'  inputs.lgbm_threshold.default already {args.threshold} — no change', flush=True)
        if not delta_inputs and 'lgbm' not in delta_nodes:
            print(f'  NOTE: no node "lgbm.threshold" or input "lgbm_threshold" found.', flush=True)
            print(f'    Pipeline nodes: {list(nodes.keys())}', flush=True)
            print(f'    Pipeline inputs: {list(inputs.keys())}', flush=True)

    delta = {'pipeline': {'nodes': delta_nodes}}
    if delta_inputs:
        delta['pipeline']['inputs'] = delta_inputs

    if args.dry_run:
        print(f'\n  [DRY-RUN] would PATCH revision {rev}', flush=True)
        print(f'    delta nodes:  {list(delta_nodes.keys())}', flush=True)
        if delta_inputs:
            print(f'    delta inputs: {list(delta_inputs.keys())}', flush=True)
        return

    patch_url = f'{API}/projects/{args.project_id}/patch/{rev}'
    print(f'  PATCH {patch_url}', flush=True)
    try:
        r = httpx.patch(patch_url, headers=hdr,
                        content=json.dumps(delta).encode(), timeout=60)
    except Exception as e:
        print(f'  NETWORK ERROR: {type(e).__name__}: {e}', file=sys.stderr); sys.exit(1)

    if r.status_code != 200:
        diag_response(r, 'PATCH')
        sys.exit(1)

    new_rev = (r.json() or {}).get('revision')
    print(f'\n  OK new revision: {new_rev}', flush=True)
    print(f'  Rollback: python scripts/rollback_piper.py {args.project_id} {rev}', flush=True)


if __name__ == '__main__':
    main()
