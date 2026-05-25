#!/usr/bin/env python3
"""
rescore_k30_v9.py
-----------------
Re-score all K30 items through the V9 test pipeline (project 9cd1798843 by default)
to obtain native 317-tag input with :x20 suffixes — the format V11 was trained on.

K30 items have only top-5 siglip outputs in 180-tag baseline taxonomy in their
stored data, which makes V11 essentially blind on them. After running this,
V11 will have full 317-tag rescored input for honest evaluation on K30.

Output: data/k30_rescored.json — list of records like:
  {id, label, minor, adult, lgbm_score, underage_labels, adult_labels, no_underage_labels, done}

Usage:
    python scripts/rescore_k30_v9.py                    # default project 9cd1798843
    python scripts/rescore_k30_v9.py --project XXX --workers 8
    python scripts/rescore_k30_v9.py --limit 100        # dry-run on first 100
"""
import os, sys, json, time, sqlite3, struct, argparse, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()
BASE_DIR   = Path(__file__).resolve().parent.parent
TOKEN      = os.getenv('PIPER_TOKEN', '')
PIPER_BASE = 'https://piper-next.artworks.ai/api'
DEFAULT_PROJECT = 'd2911d10bb'  # V9 (9cd1798843) was removed; production gives full 180-tag instead
WORKERS         = 8
OUT_PATH        = BASE_DIR / 'data' / 'k30_rescored.json'


def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}


def _open_db():
    src = BASE_DIR / 'gallery.db'
    tmp = Path('/tmp/_k30_rescore.db')
    shutil.copy2(src, tmp)
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def run_one(item_id, url, label, project_id, max_polls=50, poll_delay=3):
    try:
        r = httpx.post(f'{PIPER_BASE}/projects/{project_id}/launch', headers=hdr(),
                       json={'inputs': {'image': url, 'providers': ['siglip2']}}, timeout=30)
        r.raise_for_status()
        run_id = r.json()['_id']
        for _ in range(max_polls):
            time.sleep(poll_delay)
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state', headers=hdr(), timeout=15)
            if rs.status_code != 200: continue
            state = rs.json()
            errors = state.get('errors') or []
            outputs = state.get('outputs') or {}
            if errors:
                return {'id': item_id, 'label': label, 'error': str(errors[0])[:120]}
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
                    'done': True,
                }
        return {'id': item_id, 'label': label, 'error': 'timeout'}
    except Exception as e:
        return {'id': item_id, 'label': label, 'error': str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default=DEFAULT_PROJECT, help='Piper project ID for V9 rescoring')
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--limit', type=int, default=None, help='Process first N items only (testing)')
    args = ap.parse_args()

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    conn = _open_db()
    rows = conn.execute("""
        SELECT id, label, thumb_url FROM k30_pool
        WHERE thumb_url IS NOT NULL AND (deleted IS NULL OR deleted=0)
    """).fetchall()
    conn.close()
    if args.limit:
        rows = rows[:args.limit]
    print(f'Total K30 items: {len(rows)}', flush=True)

    # Resume from existing file
    existing = {}
    if OUT_PATH.exists():
        try:
            for r in json.loads(OUT_PATH.read_text()):
                if r.get('done'):
                    existing[r['id']] = r
        except Exception:
            pass

    todo = [r for r in rows if r['id'] not in existing]
    print(f'  already done: {len(existing)}', flush=True)
    print(f'  todo:         {len(todo)}', flush=True)
    print(f'  workers:      {args.workers}', flush=True)
    print(f'  project:      {args.project}\n', flush=True)

    results = list(existing.values())

    if not todo:
        tmp_p = OUT_PATH.with_suffix(OUT_PATH.suffix + '.tmp'); tmp_p.write_text(json.dumps(results, indent=2, ensure_ascii=False)); os.replace(tmp_p, OUT_PATH)
        print('Nothing to do, file already complete.')
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, r['id'], r['thumb_url'], r['label'], args.project): r for r in todo}
        n_total = len(existing)
        n_done  = 0
        n_err   = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n_total += 1
            if res.get('done'):
                n_done += 1
                status = '✓'
            else:
                n_err += 1
                status = f'✗({res.get("error","?")[:30]})'
            print(f'[{n_total:4d}/{len(rows)}] {status} {res["id"][:24]} ({res.get("label")})', flush=True)
            if n_total % 25 == 0:
                tmp_p = OUT_PATH.with_suffix(OUT_PATH.suffix + '.tmp'); tmp_p.write_text(json.dumps(results, indent=2, ensure_ascii=False)); os.replace(tmp_p, OUT_PATH)

    tmp_p = OUT_PATH.with_suffix(OUT_PATH.suffix + '.tmp'); tmp_p.write_text(json.dumps(results, indent=2, ensure_ascii=False)); os.replace(tmp_p, OUT_PATH)
    done = sum(1 for r in results if r.get('done'))
    err  = sum(1 for r in results if not r.get('done'))
    print(f'\n=== Done ===', flush=True)
    print(f'  total processed: {len(results)}', flush=True)
    print(f'  successful:      {done}', flush=True)
    print(f'  errors:          {err}', flush=True)
    print(f'  saved to:        {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
