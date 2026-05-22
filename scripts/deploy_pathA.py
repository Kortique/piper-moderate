#!/usr/bin/env python3
"""
Path A deploy:
  d2911d10bb: keep labels as-is. Update only lgbm_evaluate.script (V7pa).
  ce79f7e299: merge no_underage_X → adult_X in labels.default. Invalidate SigLIP cache
              by incrementing ask_siglip2.script version. Update lgbm_evaluate.script (V10pa).

Both new lgbm_evaluate scripts have Path A behaviour: no :x20 multiplier in
getAdjustedScores; buildVec strips :x20 from feature lookups; for ce79f7e299
no_underage_X scores are routed into adult__X via buildVec mapping (no rename in tags needed).
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
    r = httpx.get(f'{API}/projects/{proj}', headers=HDR, timeout=30).json()
    pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
    return r['revision'], pipe


def bump_version(script):
    m = re.search(r'// v(\d+)', script)
    if m:
        new_v = int(m.group(1)) + 1
        return re.sub(r'// v\d+', f'// v{new_v}', script, count=1)
    return script + '\n// v1\n'


def backup(proj, pipe):
    d = BASE / 'backups' / 'piper_pathA'
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    p = d / f'{proj}_{ts}.json'
    p.write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
    print(f'  Backup: {p}', flush=True)


def patch(proj, rev, delta):
    if DRY:
        print(f'  [DRY-RUN] would PATCH {proj} @ {rev}', flush=True)
        return None
    r = httpx.patch(f'{API}/projects/{proj}/patch/{rev}',
                    headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    r.raise_for_status()
    return r.json().get('revision')


def deploy_d2911d10bb():
    print('\n=== Deploy V7pa → d2911d10bb (labels unchanged, lgbm_evaluate updated) ===', flush=True)
    rev, pipe = fetch('d2911d10bb')
    print(f'  rev: {rev}', flush=True)
    backup('d2911d10bb', pipe)
    old_lgbm = pipe['nodes']['lgbm_evaluate']['script']
    new_lgbm = (BASE / 'data' / 'lgbm_evaluate_v7pa.js').read_text()
    delta = {'pipeline': {'nodes': {'lgbm_evaluate': {'script': [old_lgbm, new_lgbm]}}}}
    nr = patch('d2911d10bb', rev, delta)
    if nr: print(f'  ✓ new rev: {nr}', flush=True)


def deploy_ce79f7e299():
    print('\n=== Deploy V10pa → ce79f7e299 (labels merged, lgbm_evaluate updated, SigLIP invalidated) ===', flush=True)
    rev, pipe = fetch('ce79f7e299')
    print(f'  rev: {rev}', flush=True)
    backup('ce79f7e299', pipe)

    # Merge no_underage → adult in labels
    old_labels = pipe['nodes']['ask_siglip2']['inputs']['labels']['default']
    if not isinstance(old_labels, str):
        old_labels = json.dumps(old_labels, ensure_ascii=False, indent=2)
    labels = json.loads(old_labels)
    new_labels = {}
    # pass 1: keep non-no_underage keys
    for k, v in labels.items():
        if not k.startswith('no_underage_'):
            new_labels[k] = v
    # pass 2: convert no_underage_X → adult_X (skip if adult_X already exists)
    for k, v in labels.items():
        if k.startswith('no_underage_'):
            new_k = 'adult_' + k[len('no_underage_'):]
            if new_k not in new_labels:
                new_labels[new_k] = v
    new_labels_str = json.dumps(new_labels, ensure_ascii=False, indent=2)
    print(f'  labels: {len(labels)} → {len(new_labels)} (merge no_underage into adult)', flush=True)

    old_sig_script = pipe['nodes']['ask_siglip2']['script']
    new_sig_script = bump_version(old_sig_script)

    old_lgbm = pipe['nodes']['lgbm_evaluate']['script']
    new_lgbm = (BASE / 'data' / 'lgbm_evaluate_v10pa.js').read_text()

    delta = {'pipeline': {'nodes': {
        'ask_siglip2': {
            'script': [old_sig_script, new_sig_script],
            'inputs': {'labels': {'default': [old_labels, new_labels_str]}},
        },
        'lgbm_evaluate': {'script': [old_lgbm, new_lgbm]},
    }}}
    nr = patch('ce79f7e299', rev, delta)
    if nr: print(f'  ✓ new rev: {nr}', flush=True)


if __name__ == '__main__':
    deploy_d2911d10bb()
    deploy_ce79f7e299()
    print('\nDone.', flush=True)
