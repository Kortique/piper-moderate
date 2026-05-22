#!/usr/bin/env python3
"""Rollback both pipelines to pre-deploy state using backups."""
import os, json, glob
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
API = 'https://piper-next.artworks.ai/api'


def rollback(proj):
    print(f'\n=== Rollback {proj} ===', flush=True)
    # find latest backup
    backups = sorted(glob.glob(str(BASE / 'backups' / 'piper_pre_rename' / f'{proj}_*.json')))
    if not backups:
        print('  No backups!', flush=True)
        return
    backup_file = backups[-1]
    print(f'  using backup: {backup_file}', flush=True)
    old_pipe = json.loads(Path(backup_file).read_text())
    old_sig = old_pipe['nodes']['ask_siglip2']
    old_lgbm = old_pipe['nodes']['lgbm_evaluate']
    old_sig_script = old_sig['script']
    old_sig_labels = old_sig['inputs']['labels']['default']
    if not isinstance(old_sig_labels, str):
        old_sig_labels = json.dumps(old_sig_labels, ensure_ascii=False, indent=2)
    old_lgbm_script = old_lgbm['script']

    # fetch current
    r = httpx.get(f'{API}/projects/{proj}', headers=HDR, timeout=30).json()
    rev = r['revision']
    cur_pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
    cur_sig_script = cur_pipe['nodes']['ask_siglip2']['script']
    cur_sig_labels = cur_pipe['nodes']['ask_siglip2']['inputs']['labels']['default']
    if not isinstance(cur_sig_labels, str):
        cur_sig_labels = json.dumps(cur_sig_labels, ensure_ascii=False, indent=2)
    cur_lgbm_script = cur_pipe['nodes']['lgbm_evaluate']['script']

    print(f'  current revision: {rev}', flush=True)

    delta = {
        'pipeline': {
            'nodes': {
                'ask_siglip2': {
                    'script': [cur_sig_script, old_sig_script],
                    'inputs': {'labels': {'default': [cur_sig_labels, old_sig_labels]}},
                },
                'lgbm_evaluate': {
                    'script': [cur_lgbm_script, old_lgbm_script],
                },
            }
        }
    }
    resp = httpx.patch(f'{API}/projects/{proj}/patch/{rev}',
                       headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    resp.raise_for_status()
    print(f'  rolled back. new revision: {resp.json().get("revision")}', flush=True)


if __name__ == '__main__':
    rollback('d2911d10bb')
    rollback('ce79f7e299')
