#!/usr/bin/env python3
"""Deploy V8pas80-v2 to d2911d10bb: replace lgbm_evaluate.script only.

Slim labels and ask_siglip2 stay as-is.
"""
import os, json, time, sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
API = 'https://piper-next.artworks.ai/api'

DRY = '--dry-run' in sys.argv

print('=== Deploy V8pas80-v2 to d2911d10bb ===', flush=True)
r = httpx.get(f'{API}/projects/d2911d10bb', headers=HDR, timeout=30).json()
rev = r['revision']
pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
print(f'  current rev: {rev}', flush=True)

# Backup
backup_dir = BASE / 'backups' / 'piper_v8pas80_v2'
backup_dir.mkdir(parents=True, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
backup_path = backup_dir / f'd2911d10bb_{ts}.json'
backup_path.write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
print(f'  backup: {backup_path}', flush=True)

old_lgbm = pipe['nodes']['lgbm_evaluate']['script']
new_lgbm = (BASE / 'data' / 'lgbm_evaluate_v8pas80_v2.js').read_text()
print(f'  lgbm_evaluate: {len(old_lgbm)} → {len(new_lgbm)} chars', flush=True)

# Check we're replacing v8pas80 (not something unexpected)
if 'v8pas80' not in old_lgbm:
    print(f'  WARNING: current lgbm_evaluate does not mention v8pas80!', flush=True)
    print(f'  first 200 chars: {old_lgbm[:200]}', flush=True)

delta = {
    'pipeline': {
        'nodes': {
            'lgbm_evaluate': {
                'script': [old_lgbm, new_lgbm],
            }
        }
    }
}

if DRY:
    print('  [DRY-RUN] would PATCH d2911d10bb', flush=True)
else:
    r = httpx.patch(f'{API}/projects/d2911d10bb/patch/{rev}',
                    headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    if r.status_code != 200:
        print(f'  PATCH failed HTTP {r.status_code}: {r.text[:500]}', flush=True); sys.exit(1)
    new_rev = r.json().get('revision')
    print(f'  ✓ new rev: {new_rev}', flush=True)
    print(f'\n  Model: V8pas80 → V8pas80-v2 (138 hard-negs ×20 vs 71)', flush=True)
    print(f'  Rollback: python scripts/rollback_piper.py d2911d10bb {rev}', flush=True)
