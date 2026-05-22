#!/usr/bin/env python3
"""
rescore_v9_317session.py
------------------------
Run the 317-session images through the V9 test pipeline (with new tags)
and save results to data/v9_317_scores.json.

Requires:
  - V9 test Piper project to be running (see v9_test_project_id.txt)
  - PIPER_TOKEN in .env

Usage:
    python scripts/rescore_v9_317session.py [--project ID]
"""
import os, sys, json, time, sqlite3, struct, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
TOKEN = os.getenv('PIPER_TOKEN', '')
PIPER_BASE = 'https://piper-next.artworks.ai/api'
WORKERS = 4
OUT_PATH = BASE_DIR / 'data' / 'v9_317_scores.json'

def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}

def _open_db():
    candidates = sorted((BASE_DIR / 'backups').glob('gallery_*.db'), reverse=True)
    for db_path in candidates:
        try:
            data = bytearray(db_path.read_bytes())
            struct.pack_into('>I', data, 28, len(data) // 4096)
            tmp = Path('/tmp/_v9_rescore.db')
            tmp.write_bytes(bytes(data))
            conn = sqlite3.connect(str(tmp))
            conn.row_factory = sqlite3.Row
            conn.execute('SELECT id FROM grafana_pool LIMIT 1').fetchall()
            return conn
        except:
            continue
    raise RuntimeError('No DB found')

def run_one(item_id, url, label, project_id):
    try:
        r = httpx.post(f'{PIPER_BASE}/projects/{project_id}/launch', headers=hdr(),
                       json={'inputs': {'image': url, 'providers': ['siglip2']}}, timeout=30)
        r.raise_for_status()
        run_id = r.json()['_id']

        for _ in range(50):
            time.sleep(3)
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state', headers=hdr(), timeout=15)
            if rs.status_code != 200: continue
            state = rs.json()
            errors = state.get('errors') or []
            outputs = state.get('outputs') or {}

            if errors:
                return {'id': item_id, 'label': label, 'error': str(errors[0])[:100]}

            if 'siglip2_details' in outputs:
                det = (outputs.get('siglip2_details') or {}).get('underage', {})
                labels = det.get('labels', {})
                return {
                    'id': item_id,
                    'label': label,
                    'minor': det.get('minor', 0),
                    'adult': det.get('adult', 0),
                    'lgbm_score': (det.get('lgbm') or {}).get('score', 0),
                    'underage_labels': labels.get('underage', {}),
                    'adult_labels': labels.get('adult', {}),
                    'no_underage_labels': labels.get('no_underage', {}),
                    'done': True
                }
        return {'id': item_id, 'label': label, 'error': 'timeout'}
    except Exception as e:
        return {'id': item_id, 'label': label, 'error': str(e)[:100]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Piper project ID for V9 test')
    parser.add_argument('--workers', type=int, default=WORKERS)
    args = parser.parse_args()

    conn = _open_db()
    rows = conn.execute("""
        SELECT id, label, thumb_url FROM grafana_pool
        WHERE export_batch = '2026-05-20 UTC'
        AND thumb_url IS NOT NULL
        AND (deleted IS NULL OR deleted=0)
    """).fetchall()
    conn.close()
    print(f'Total 317-session items: {len(rows)}')

    # Resume
    existing = {}
    if OUT_PATH.exists():
        for r in json.loads(OUT_PATH.read_text()):
            if r.get('done'): existing[r['id']] = r

    todo = [r for r in rows if r['id'] not in existing]
    print(f'Done: {len(existing)}, Todo: {len(todo)}')

    results = list(existing.values())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, r['id'], r['thumb_url'], r['label'], args.project): r for r in todo}
        n = len(existing)
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n += 1
            status = '✓' if res.get('done') else f'✗({res.get("error","?")[:20]})'
            print(f'[{n:3d}/{len(rows)}] {status} {res["id"][:16]} ({res["label"]})')

            # Incremental save
            if n % 10 == 0:
                OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    done = sum(1 for r in results if r.get('done'))
    print(f'\nCompleted: {done}/{len(rows)} saved to {OUT_PATH}')

if __name__ == '__main__':
    main()
