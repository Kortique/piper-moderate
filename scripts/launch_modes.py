#!/usr/bin/env python3
"""
launch_modes.py — examples of slim/full launching against d2911d10bb.

Usage:
  python scripts/launch_modes.py slim <image_url>
  python scripts/launch_modes.py full <image_url>
  python scripts/launch_modes.py batch full < urls.txt   # read URLs from stdin, full mode
"""
import json, os, sys, time
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
API = 'https://piper-next.artworks.ai/api'
PROJECT = 'd2911d10bb'


def load_full_labels():
    """Load the full SigLIP tag set (ce79f7e299 superset, ~860 keys)."""
    p = BASE / 'data' / 'live867.json'
    if p.exists():
        return json.loads(p.read_text())
    # fallback: live fetch
    r = httpx.get(f'{API}/projects/ce79f7e299', headers=HDR, timeout=30).json()
    pipe = json.loads(r['pipeline']) if isinstance(r['pipeline'], str) else r['pipeline']
    labels = pipe['nodes']['ask_siglip2']['inputs']['labels']['default']
    return json.loads(labels) if isinstance(labels, str) else labels


def launch_slim(image_url):
    """Production mode — uses default 180 slim labels in d2911d10bb."""
    payload = {'inputs': {'image': image_url, 'providers_e0': ['siglip2']}}
    return _launch(payload)


def launch_full(image_url, full_labels=None):
    """Data collection mode — override labels with full set."""
    if full_labels is None:
        full_labels = load_full_labels()
    payload = {
        'inputs': {
            'image': image_url,
            'providers_e0': ['siglip2'],
            # IMPORTANT: labels must be passed as a JSON-encoded string, not as object
            'labels': json.dumps(full_labels, ensure_ascii=False),
        }
    }
    return _launch(payload)


def _launch(payload):
    r = httpx.post(f'{API}/projects/{PROJECT}/launch',
                   headers=HDR, content=json.dumps(payload).encode(), timeout=20)
    r.raise_for_status()
    return r.json()['_id']


def wait(run_id, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = httpx.get(f'{API}/launches/{run_id}/state', headers=HDR, timeout=10).json()
        if (s.get('outputs') or {}).get('siglip2_details'):
            return s
        if s.get('errors'):
            return s
        time.sleep(2)
    return None


def summarize(state):
    if not state: return 'TIMEOUT'
    if state.get('errors'): return f'ERROR: {state["errors"]}'
    det = state['outputs']['siglip2_details']['underage']
    lgbm = det.get('lgbm') or {}
    labels = det.get('labels') or {}
    return {
        'lgbm_score': lgbm.get('score'),
        'blocked': lgbm.get('blocked'),
        'minor': det.get('minor'),
        'adult': det.get('adult'),
        'top_features': lgbm.get('top_features'),
        'underage_keys_returned': len(labels.get('underage') or {}),
        'adult_keys_returned': len(labels.get('adult') or {}),
        'no_underage_keys_returned': len(labels.get('no_underage') or {}),
    }


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'slim'
    if mode == 'batch':
        sub_mode = sys.argv[2] if len(sys.argv) > 2 else 'full'
        full_labels = load_full_labels() if sub_mode == 'full' else None
        urls = [line.strip() for line in sys.stdin if line.strip()]
        print(f'batch {sub_mode}: {len(urls)} URLs', flush=True)
        for url in urls:
            rid = launch_full(url, full_labels) if sub_mode == 'full' else launch_slim(url)
            s = wait(rid)
            print(json.dumps({'url': url, 'rid': rid, 'result': summarize(s)}), flush=True)
    else:
        url = sys.argv[2] if len(sys.argv) > 2 else 'https://fsn1.your-objectstorage.com/artworks-assets/test-942.jpg'
        rid = launch_slim(url) if mode == 'slim' else launch_full(url)
        print(f'mode={mode} run_id={rid}', flush=True)
        s = wait(rid)
        print(json.dumps(summarize(s), indent=2))
