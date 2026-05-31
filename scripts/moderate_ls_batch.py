#!/usr/bin/env python3
"""
moderate_ls_batch.py — get siglip2_details (180-tag taxonomy) for new LS-batch
items via Piper d2911d10bb. Needed for local V6/V8 LGBM scoring.

LS-specific note: we deliberately DO NOT request qwen3 or face_detect — those
were never used for LS items, and labels in LS are human-set. We only need
siglip2_details so that V6/V8 LGBM scoring works in the gallery. V11 and Tom
K30 scores come from their own pipelines via rescore_via_v11/tom.

Picks items where:
  - session = <--session arg>
  - siglip2_details IS NULL  (i.e. not yet moderated)

Usage:
    python scripts/moderate_ls_batch.py --session "2026-05-28_underage" --workers 6
    python scripts/moderate_ls_batch.py --session ... --limit 10        # smoke test
"""
import argparse, json, os, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')

DB_PATH = BASE / 'gallery.db'
PIPER_BASE = 'https://piper-next.artworks.ai/api'
MODERATE_PROJECT = 'd2911d10bb'
TOKEN = os.getenv('PIPER_TOKEN', '')


def _hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json',
            'Accept': 'application/json'}


def run_pipeline(image_url: str) -> dict:
    """Single-image siglip2-only call via d2911d10bb. Returns outputs (or {'error':…})."""
    # providers=['siglip2'] is what rescore_via_tom/v11 also use; faster than the
    # full siglip+qwen3+face_detect chain since qwen3 alone takes ~20-30s/image.
    payload = {'inputs': {'image': image_url, 'providers': ['siglip2']}}
    try:
        r = httpx.post(f'{PIPER_BASE}/projects/{MODERATE_PROJECT}/launch',
                       headers=_hdr(), json=payload, timeout=30)
        r.raise_for_status()
        run_id = r.json()['_id']
    except Exception as e:
        return {'error': f'launch: {type(e).__name__}: {str(e)[:120]}'}

    for _ in range(40):  # ~2 min max (siglip2 alone is fast)
        time.sleep(3)
        try:
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state',
                           headers=_hdr(), timeout=20)
            if rs.status_code != 200: continue
            st = rs.json()
        except Exception:
            continue
        outs = st.get('outputs') or {}
        errs = st.get('errors') or []
        if errs:
            return {'error': str(errs[0])[:200]}
        # Done as soon as siglip2_details is back
        if 'siglip2_details' in outs:
            return outs
    return {'error': 'timeout'}


def process_one(item_id: int, media: str):
    outs = run_pipeline(media)
    now = datetime.utcnow().isoformat()
    if 'error' in outs:
        return item_id, {'error': outs['error'], 'processed_at': now}

    siglip_labels  = outs.get('siglip2_labels') or []
    siglip_details = outs.get('siglip2_details')
    return item_id, {
        'siglip2_labels':  json.dumps(siglip_labels, ensure_ascii=False),
        'siglip2_passed':  1 if 'underage' not in siglip_labels else 0,
        'siglip2_details': json.dumps(siglip_details, ensure_ascii=False) if siglip_details else None,
        'processed_at':    now,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', required=True, help='Only process items with this session')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit',   type=int, default=0, help='Process at most N items (0 = all unprocessed)')
    ap.add_argument('--redo',    action='store_true',
                    help='Re-process even items that already have siglip2_details (default: skip)')
    args = ap.parse_args()

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)
    if not DB_PATH.exists():
        print(f'ERR: {DB_PATH} not found', file=sys.stderr); sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Pick items
    if args.redo:
        rows = conn.execute(
            "SELECT task_id, media FROM ls_images WHERE session=? AND media IS NOT NULL",
            (args.session,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT task_id, media FROM ls_images "
            "WHERE session=? AND media IS NOT NULL AND siglip2_details IS NULL",
            (args.session,)
        ).fetchall()

    if args.limit:
        rows = rows[:args.limit]

    print(f"=== Moderate LS batch ===")
    print(f"  session: {args.session!r}")
    print(f"  to process: {len(rows)}")
    print(f"  workers:    {args.workers}")
    print(f"  project:    {MODERATE_PROJECT}\n")

    if not rows:
        print("Nothing to do.")
        return

    saved = errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, r['task_id'], r['media']): r['task_id']
                   for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            task_id, res = fut.result()
            if 'error' in res:
                errors += 1
                print(f"  [{i:>5}/{len(rows)}] ✗ task_id={task_id}  {res['error'][:80]}", flush=True)
                continue
            # UPDATE row — LS items: we touch ONLY siglip2_* fields + processed_at.
            # age_from / age_to come from LS annotations on import and must not be
            # overwritten here. face_detect/qwen3 are intentionally not requested.
            conn.execute("""
                UPDATE ls_images SET
                    siglip2_labels=?, siglip2_passed=?, siglip2_details=?,
                    processed_at=?
                WHERE task_id=?
            """, (
                res['siglip2_labels'], res['siglip2_passed'], res['siglip2_details'],
                res['processed_at'], task_id
            ))
            saved += 1
            if saved % 25 == 0:
                conn.commit()
            status = '✓'
            tag = ', '.join(json.loads(res['siglip2_labels'] or '[]')[:3]) or 'passed'
            print(f"  [{i:>5}/{len(rows)}] {status} task_id={task_id:<8}  siglip2={tag[:50]}", flush=True)

    conn.commit()
    conn.close()
    dt = time.time() - t0
    print(f"\n=== Done in {dt:.0f}s ===")
    print(f"  saved:  {saved}")
    print(f"  errors: {errors}")
    print(f"\nNext:")
    print(f"  python scripts/rescore_via_v11.py --source ls --workers 8")
    print(f"  python scripts/rescore_via_tom.py --source ls --workers 8")
    print(f"\nThen restart gallery_server.py to see new items.")


if __name__ == '__main__':
    main()
