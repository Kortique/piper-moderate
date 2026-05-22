#!/usr/bin/env python3
"""
Switch d2911d10bb default labels between slim and full mode.

Usage:
  python scripts/switch_labels.py slim     # → 180 keys (production default, ~3 SigLIP batches)
  python scripts/switch_labels.py full     # → 860 keys (data collection / training, ~14 batches)
  python scripts/switch_labels.py status   # → show current state without changing
  python scripts/switch_labels.py <path>   # → load labels from custom JSON file
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
PROJECT = 'd2911d10bb'

PRESETS = {
    'slim': BASE / 'data' / 'd2911d10bb_slim_labels.json',
    'full': BASE / 'data' / 'live867.json',
}


def fetch_pipeline():
    r = httpx.get(f'{API}/projects/{PROJECT}', headers=HDR, timeout=30).json()
    pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
    return r['revision'], pipe


def show_status():
    rev, pipe = fetch_pipeline()
    labels = pipe['nodes']['ask_siglip2']['inputs']['labels']['default']
    if isinstance(labels, str): labels = json.loads(labels)
    ver = re.search(r'// v(\d+)', pipe['nodes']['ask_siglip2']['script'])
    n_und = sum(1 for k in labels if k.startswith('underage_'))
    n_adlt = sum(1 for k in labels if k.startswith('adult_'))
    n_no_und = sum(1 for k in labels if k.startswith('no_underage_'))
    print(f'd2911d10bb:')
    print(f'  revision: {rev}')
    print(f'  script version: {ver.group(0) if ver else "?"}')
    print(f'  labels: {len(labels)} total (underage_={n_und}, adult_={n_adlt}, no_underage_={n_no_und})')
    if len(labels) < 250: print(f'  mode: SLIM')
    elif len(labels) > 700: print(f'  mode: FULL')
    else: print(f'  mode: CUSTOM ({len(labels)} keys)')


def switch_to(labels_path: Path):
    if not labels_path.exists():
        print(f'ERROR: {labels_path} not found'); sys.exit(1)
    new_labels = json.loads(labels_path.read_text())
    rev, pipe = fetch_pipeline()
    sig = pipe['nodes']['ask_siglip2']
    old_labels = sig['inputs']['labels']['default']
    if not isinstance(old_labels, str):
        old_labels = json.dumps(old_labels, ensure_ascii=False, indent=2)
    old_script = sig['script']
    # bump version
    m = re.search(r'// v(\d+)', old_script)
    if m:
        new_v = int(m.group(1)) + 1
        new_script = re.sub(r'// v\d+', f'// v{new_v}', old_script, count=1)
    else:
        new_script = old_script + '\n// v_switch\n'
    new_labels_str = json.dumps(new_labels, ensure_ascii=False, indent=2)

    # backup
    backup_dir = BASE / 'backups' / 'piper_switch'
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    (backup_dir / f'd2911d10bb_{ts}.json').write_text(json.dumps(pipe, ensure_ascii=False, indent=2))
    print(f'  backup → backups/piper_switch/d2911d10bb_{ts}.json')

    delta = {'pipeline': {'nodes': {'ask_siglip2': {
        'script': [old_script, new_script],
        'inputs': {'labels': {'default': [old_labels, new_labels_str]}},
    }}}}
    r = httpx.patch(f'{API}/projects/{PROJECT}/patch/{rev}',
                    headers=HDR, content=json.dumps(delta).encode(), timeout=60)
    r.raise_for_status()
    new_rev = r.json().get('revision')
    print(f'  labels: {len(json.loads(old_labels))} → {len(new_labels)} keys')
    print(f'  script ver: bumped (SigLIP cache invalidated)')
    print(f'  ✓ new rev: {new_rev}')


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if arg == 'status':
        show_status()
    elif arg in PRESETS:
        path = PRESETS[arg]
        print(f'Switching to {arg.upper()}: {path}')
        switch_to(path)
        print()
        show_status()
    else:
        path = Path(arg)
        print(f'Switching to custom: {path}')
        switch_to(path)
        print()
        show_status()


if __name__ == '__main__':
    main()
