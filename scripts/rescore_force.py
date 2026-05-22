#!/usr/bin/env python3
"""
rescore_force.py — force-rescore ALL gallery items through live Piper API.

Unlike scan_ls_images.py this ignores the siglip2_labels IS NULL check
and overwrites all existing siglip2_details in the DB.

Uses the live Piper project (d2911d10bb) — whatever pipeline is deployed there.

Usage:
    python scripts/rescore_force.py                    # all LS images
    python scripts/rescore_force.py --workers 8
    python scripts/rescore_force.py --limit 200        # test with first N items
    python scripts/rescore_force.py --skip_if_ok       # skip items that already have fresh data
"""
import os, sys, json, time, sqlite3, argparse, struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = BASE_DIR / 'gallery.db'
LS_JSON    = BASE_DIR / 'qwen3_age_results.json'
PIPER_BASE = 'https://piper-next.artworks.ai/api'
PROJECT    = 'd2911d10bb'
TOKEN      = os.getenv('PIPER_TOKEN', '')
WORKERS    = 6


def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}


def run_one(item_id, url):
    """Run one image through Piper siglip2. Returns (item_id, result_dict)."""
    try:
        r = httpx.post(
            f'{PIPER_BASE}/projects/{PROJECT}/launch',
            headers=hdr(),
            json={'inputs': {'image': url, 'providers': ['siglip2']}},
            timeout=30,
        )
        r.raise_for_status()
        run_id = r.json()['_id']

        for attempt in range(40):
            time.sleep(3)
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state', headers=hdr(), timeout=15)
            if rs.status_code != 200:
                continue
            state   = rs.json()
            outputs = state.get('outputs') or {}
            errors  = state.get('errors') or []

            if errors:
                return item_id, {'error': str(errors[0])[:120]}

            if 'siglip2_labels' in outputs:
                return item_id, {
                    'launch_id':       run_id,
                    'siglip2_labels':  outputs.get('siglip2_labels'),
                    'siglip2_passed':  outputs.get('siglip2_passed', True),
                    'siglip2_details': outputs.get('siglip2_details'),
                    'error': None,
                }

        return item_id, {'error': 'timeout'}

    except Exception as e:
        return item_id, {'error': str(e)[:120]}


def j(v):
    if v is None: return None
    if isinstance(v, (dict, list)): return json.dumps(v, ensure_ascii=False)
    return v


def _open_db() -> sqlite3.Connection:
    data = bytearray(DB_PATH.read_bytes())
    struct.pack_into('>I', data, 28, len(data) // 4096)
    tmp = Path('/tmp/_rescore_force.db')
    tmp.write_bytes(bytes(data))
    conn = sqlite3.connect(str(tmp))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.row_factory = sqlite3.Row
    return conn


def save_to_db(conn, task_id, result):
    conn.execute("""
        UPDATE ls_images SET
            launch_id       = ?,
            siglip2_labels  = ?,
            siglip2_passed  = ?,
            siglip2_details = ?,
            error           = ?,
            processed_at    = datetime('now')
        WHERE task_id = ?
    """, (
        result.get('launch_id'),
        j(result.get('siglip2_labels')),
        1 if result.get('siglip2_passed') else 0,
        j(result.get('siglip2_details')),
        result.get('error'),
        int(task_id),
    ))
    conn.commit()


def sync_ls_json(conn):
    """Rebuild qwen3_age_results.json from DB."""
    rows = conn.execute("""
        SELECT task_id, media, variant, category, age_from, age_to,
               launch_id, siglip2_labels, siglip2_passed, siglip2_details,
               face_detect, error, processed_at, extra
        FROM ls_images
    """).fetchall()
    data = {}
    for row in rows:
        (tid, media, variant, category, af, at,
         launch_id, labels, passed, details, face, error, proc_at, extra) = row
        item = {
            'task_id':            tid,
            'media':              media,
            'variant':            variant,
            'category':           category,
            'age':                {'ageFrom': af, 'ageTo': at} if af is not None else None,
            'launch_id':          launch_id,
            'siglip2_labels':     json.loads(labels) if labels else None,
            'siglip2_passed':     bool(passed) if passed is not None else None,
            'siglip2_details':    json.loads(details) if details else None,
            'face_detect_result': json.loads(face) if face else None,
            'error':              error,
            'piper_processed_at': proc_at,
        }
        if extra:
            item.update(json.loads(extra))
        data[str(tid)] = item
    tmp = str(LS_JSON) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(LS_JSON))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',    type=int,  default=WORKERS)
    parser.add_argument('--limit',      type=int,  default=0, help='Max items to process (0=all)')
    parser.add_argument('--skip_if_ok', action='store_true',
                        help='Skip items that already have valid siglip2_details')
    args = parser.parse_args()

    if not TOKEN:
        print('ERROR: PIPER_TOKEN not set in .env'); sys.exit(1)
    if not LS_JSON.exists():
        print('ERROR: qwen3_age_results.json not found'); sys.exit(1)

    # Load items from JSON (avoids DB corruption issues)
    raw = LS_JSON.read_bytes().rstrip(b'\x00').decode('utf-8')
    ls_data = json.loads(raw)

    todo = []
    for v in ls_data.values():
        url = v.get('media')
        if not url:
            continue
        if args.skip_if_ok and v.get('siglip2_details') and not v.get('error'):
            continue
        todo.append((str(v['task_id']), url))

    if args.limit:
        todo = todo[:args.limit]

    print(f'Items to rescore: {len(todo)} (workers={args.workers})')
    if not todo:
        print('Nothing to do.')
        return

    # Open DB for saving results
    conn = _open_db()
    n = 0
    since_sync = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task_id, url): task_id for task_id, url in todo}
        for fut in as_completed(futures):
            task_id = futures[fut]
            item_result = fut.result()[1]
            n += 1
            since_sync += 1

            # Save to DB (try — ignore if row not found due to corruption)
            try:
                save_to_db(conn, task_id, item_result)
            except Exception as e:
                pass  # DB write failed for this row; JSON sync will still work

            # Also update in-memory ls_data for JSON sync
            if str(task_id) in ls_data:
                ls_data[str(task_id)]['siglip2_labels']  = item_result.get('siglip2_labels')
                ls_data[str(task_id)]['siglip2_details'] = item_result.get('siglip2_details')
                ls_data[str(task_id)]['siglip2_passed']  = item_result.get('siglip2_passed')
                ls_data[str(task_id)]['launch_id']       = item_result.get('launch_id')
                ls_data[str(task_id)]['error']           = item_result.get('error')

            # Progress
            err    = item_result.get('error')
            labels = item_result.get('siglip2_labels')
            if err:
                status = f'✗ {err[:35]}'
            elif labels is None:
                status = '✗ no labels'
            elif not labels:
                status = '✓ passed'
            else:
                status = '⛔ ' + ','.join(labels)

            det   = (item_result.get('siglip2_details') or {}).get('underage', {})
            minor = det.get('minor', 0)
            adult_n = len((det.get('labels') or {}).get('adult', {}))
            und_n   = len((det.get('labels') or {}).get('underage', {}))
            print(f'  [{n:4d}/{len(todo)}] {status:40s}  minor={minor:.3f}  adult={adult_n} und={und_n}  id={task_id}')

            # Periodic JSON write every 50 items
            if since_sync >= 50:
                tmp = str(LS_JSON) + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(ls_data, f, indent=2, ensure_ascii=False)
                os.replace(tmp, str(LS_JSON))
                since_sync = 0

    # Final write
    tmp = str(LS_JSON) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(ls_data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(LS_JSON))

    conn.close()
    print(f'\nDone. Rescored {n} items. Updated: {LS_JSON}')


if __name__ == '__main__':
    main()
