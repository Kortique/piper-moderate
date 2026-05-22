#!/usr/bin/env python3
"""
Deploy renamed tags + new LGBM models to both pipelines.

d2911d10bb:
  - labels.default: rename :x20 → no suffix (710 tags, was 713)
  - ask_siglip2.script: increment version comment (invalidates SigLIP cache)
  - lgbm_evaluate.script: replace with V7r

ce79f7e299:
  - labels.default: rename :x20 + merge no_underage → adult (857 tags, was 867)
  - ask_siglip2.script: increment version
  - lgbm_evaluate.script: replace with V10rm

Usage:
    python scripts/deploy_to_piper.py [--dry-run]
"""
import os, sys, json, re, time
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
API = 'https://piper-next.artworks.ai/api'

DRY = '--dry-run' in sys.argv


def fetch(proj):
    r = httpx.get(f'{API}/projects/{proj}', headers=HDR, timeout=30)
    r.raise_for_status()
    live = r.json()
    pipe = live['pipeline']
    if isinstance(pipe, str): pipe = json.loads(pipe)
    return live['revision'], pipe


def increment_version(script):
    m = re.search(r'// v(\d+)', script)
    if m:
        new_v = int(m.group(1)) + 1
        return re.sub(r'// v\d+', f'// v{new_v}', script, count=1)
    return script + f'\n// v1\n'


def patch(proj, revision, delta):
    if DRY:
        print(f'  [DRY-RUN] would PATCH {proj} @ rev={revision}', flush=True)
        # don't show delta — it's huge
        return None
    r = httpx.patch(f'{API}/projects/{proj}/patch/{revision}',
                    headers=HDR, content=json.dumps(delta).encode(),
                    timeout=60)
    r.raise_for_status()
    return r.json().get('revision')


def make_backup(proj, pipe):
    backup_dir = BASE / 'backups' / 'piper_pre_rename'
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'{proj}_{ts}.json'
    backup_path.write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
    print(f'  Backup: {backup_path}', flush=True)


def deploy_one(proj, new_labels_path, lgbm_js_path, name):
    print(f'\n=== Deploy {name} → {proj} ===', flush=True)
    rev, pipe = fetch(proj)
    print(f'  Live revision: {rev}', flush=True)
    make_backup(proj, pipe)

    # Read new state
    new_labels = json.loads(Path(new_labels_path).read_text())
    new_labels_str = json.dumps(new_labels, ensure_ascii=False, indent=2)

    # Current state
    sig_node = pipe['nodes']['ask_siglip2']
    old_labels = sig_node['inputs']['labels']['default']
    if not isinstance(old_labels, str):
        old_labels = json.dumps(old_labels, ensure_ascii=False, indent=2)
    old_sig_script = sig_node['script']
    new_sig_script = increment_version(old_sig_script)
    if new_sig_script == old_sig_script:
        new_sig_script = old_sig_script + '\n// renamed_x20\n'

    lgbm_node = pipe['nodes']['lgbm_evaluate']
    old_lgbm_script = lgbm_node['script']
    new_lgbm_script = Path(lgbm_js_path).read_text()

    print(f'  Labels: {len(json.loads(old_labels)) if isinstance(old_labels, str) else len(old_labels)} → {len(new_labels)} tags', flush=True)
    print(f'  ask_siglip2.script length: {len(old_sig_script)} → {len(new_sig_script)}', flush=True)
    print(f'  lgbm_evaluate.script length: {len(old_lgbm_script)} → {len(new_lgbm_script)}', flush=True)

    delta = {
        'pipeline': {
            'nodes': {
                'ask_siglip2': {
                    'script': [old_sig_script, new_sig_script],
                    'inputs': {'labels': {'default': [old_labels, new_labels_str]}},
                },
                'lgbm_evaluate': {
                    'script': [old_lgbm_script, new_lgbm_script],
                },
            }
        }
    }
    new_rev = patch(proj, rev, delta)
    if new_rev:
        print(f'  ✓ Patched. New revision: {new_rev}', flush=True)
    return new_rev


def main():
    if not TOKEN:
        print('ERROR: PIPER_TOKEN not set', flush=True)
        sys.exit(1)
    deploy_one('d2911d10bb', 'data/deploy_preview/d2911d10bb_new_labels.json',
               'data/lgbm_evaluate_v7r.js', 'V7r')
    deploy_one('ce79f7e299', 'data/deploy_preview/ce79f7e299_new_labels.json',
               'data/lgbm_evaluate_v10rm.js', 'V10rm')
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
