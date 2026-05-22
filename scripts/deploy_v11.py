#!/usr/bin/env python3
"""Deploy V11 to ce79f7e299: only update lgbm_evaluate.script (labels unchanged)."""
import os, sys, json, time
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
API = 'https://piper-next.artworks.ai/api'

DRY = '--dry-run' in sys.argv

print('Deploy V11 → ce79f7e299', flush=True)
r = httpx.get(f'{API}/projects/ce79f7e299', headers=HDR, timeout=30).json()
rev = r['revision']
pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
print(f'  current rev: {rev}', flush=True)

# Backup
backup_dir = BASE / 'backups' / 'piper_v11'
backup_dir.mkdir(parents=True, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
backup_path = backup_dir / f'ce79f7e299_{ts}.json'
backup_path.write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
print(f'  backup: {backup_path}', flush=True)

old_lgbm = pipe['nodes']['lgbm_evaluate']['script']
new_lgbm = (BASE / 'data' / 'lgbm_evaluate_v11.js').read_text()
print(f'  lgbm script: {len(old_lgbm)} → {len(new_lgbm)} chars', flush=True)

delta = {'pipeline': {'nodes': {'lgbm_evaluate': {'script': [old_lgbm, new_lgbm]}}}}

if DRY:
    print('  [DRY-RUN]')
else:
    r = httpx.patch(f'{API}/projects/ce79f7e299/patch/{rev}',
                    headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    r.raise_for_status()
    print(f'  ✓ new rev: {r.json().get("revision")}', flush=True)
