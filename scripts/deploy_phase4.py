#!/usr/bin/env python3
"""Phase 4 deploy: V8pas80 + slim labels in d2911d10bb.

Atomic PATCH:
  - ask_siglip2.inputs.labels.default: slim set (180 keys, was 713)
  - ask_siglip2.script: bump version to invalidate SigLIP cache
  - lgbm_evaluate.script: replace with V8pas80 JS
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

print('=== Phase 4 deploy: V8pas80 + slim labels in d2911d10bb ===', flush=True)
r = httpx.get(f'{API}/projects/d2911d10bb', headers=HDR, timeout=30).json()
rev = r['revision']
pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
print(f'  current rev: {rev}', flush=True)

# Backup
backup_dir = BASE / 'backups' / 'piper_phase4'
backup_dir.mkdir(parents=True, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
backup_path = backup_dir / f'd2911d10bb_{ts}.json'
backup_path.write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
print(f'  backup: {backup_path}', flush=True)

# Old state
sig_node = pipe['nodes']['ask_siglip2']
old_labels = sig_node['inputs']['labels']['default']
if not isinstance(old_labels, str):
    old_labels = json.dumps(old_labels, ensure_ascii=False, indent=2)
old_sig_script = sig_node['script']
old_lgbm = pipe['nodes']['lgbm_evaluate']['script']

# New state
new_labels = json.loads((BASE / 'data' / 'd2911d10bb_slim_labels.json').read_text())
new_labels_str = json.dumps(new_labels, ensure_ascii=False, indent=2)
print(f'  labels: {len(json.loads(old_labels))} → {len(new_labels)} keys', flush=True)
print(f'  SigLIP batches: 12 → {(len(new_labels)+63)//64}', flush=True)

# Bump SigLIP script version to invalidate cache
m = re.search(r'// v(\d+)', old_sig_script)
if m:
    new_v = int(m.group(1)) + 1
    new_sig_script = re.sub(r'// v\d+', f'// v{new_v}', old_sig_script, count=1)
else:
    new_sig_script = old_sig_script + '\n// v_slim\n'
print(f'  ask_siglip2 script: {len(old_sig_script)} → {len(new_sig_script)} (cache invalidated)', flush=True)

new_lgbm = (BASE / 'data' / 'lgbm_evaluate_v8pas80.js').read_text()
print(f'  lgbm_evaluate script: {len(old_lgbm)} → {len(new_lgbm)}', flush=True)

delta = {
    'pipeline': {
        'nodes': {
            'ask_siglip2': {
                'script': [old_sig_script, new_sig_script],
                'inputs': {'labels': {'default': [old_labels, new_labels_str]}},
            },
            'lgbm_evaluate': {
                'script': [old_lgbm, new_lgbm],
            },
        }
    }
}

if DRY:
    print('  [DRY-RUN] would PATCH d2911d10bb', flush=True)
else:
    r = httpx.patch(f'{API}/projects/d2911d10bb/patch/{rev}',
                    headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    r.raise_for_status()
    new_rev = r.json().get('revision')
    print(f'  ✓ new rev: {new_rev}', flush=True)
