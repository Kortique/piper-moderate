#!/usr/bin/env python3
"""
fetch_ls_batch.py — pull tasks from a Label Studio view via API into a single JSON
file that import_ls_batch.py already knows how to consume.

Replaces the manual "Export → JSON" UI step which produces too many files for
big views.

Endpoint:
    GET /api/tasks?project={pid}&view={vid}&page={n}&page_size={ps}&fields=all
    Auth: Authorization: Token <LS_API_TOKEN>

Why fields=all: the plain list endpoint returns only annotation summaries
(annotations_results) without the structured .result[*].value blocks that
import_ls_batch.py needs to extract age and category. fields=all switches the
serializer to include full annotation bodies.

Output is a JSON array — exactly the shape import_ls_batch.parse_input expects.

Usage:
    python scripts/fetch_ls_batch.py --project 3 --view 65 \\
        --out data/ls_export_2026-05-28_underage.json
    python scripts/fetch_ls_batch.py --project 3 --view 65 --out ... --limit 50
    python scripts/fetch_ls_batch.py --project 3 --view 65 --out ... \\
        --page-size 200 --max-retries 5

Notes:
    - LS pagination caps at page_size <= 200 in practice; default is 100. We pick
      150 as a balance between throughput and memory.
    - Server hiccups (502/504/timeout) are retried with exponential backoff.
    - If `--resume` is passed and the output file already exists, we read its
      current task ids and skip pages whose first task id is already present.
      (Coarse — re-fetches a partial page; safe due to dedup at the end.)
"""
import argparse, json, os, sys, time
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

LS_TOKEN = os.getenv('LS_API_TOKEN', '')
LS_BASE  = os.getenv('LS_BASE', 'https://ls.artworks.ai').rstrip('/')


def _hdr():
    return {'Authorization': f'Token {LS_TOKEN}',
            'Accept': 'application/json'}


def fetch_page(client: httpx.Client, project: int, view: int,
               page: int, page_size: int, max_retries: int = 5) -> dict:
    """Fetch one page. Returns parsed JSON dict {tasks, total, ...} or raises."""
    url = f'{LS_BASE}/api/tasks'
    params = {
        'project':   project,
        'view':      view,
        'page':      page,
        'page_size': page_size,
        'fields':    'all',
    }
    delay = 2.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = client.get(url, params=params, headers=_hdr(), timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (502, 503, 504, 429):
                last_err = f'HTTP {r.status_code}'
            else:
                # Unexpected — don't retry blindly, surface the body
                body = (r.text or '')[:400]
                raise RuntimeError(f'HTTP {r.status_code}: {body!r}')
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            last_err = f'{type(e).__name__}: {e}'
        print(f'    page {page} attempt {attempt}/{max_retries} failed ({last_err}); '
              f'sleeping {delay:.0f}s', flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 30.0)
    raise RuntimeError(f'page {page}: gave up after {max_retries} attempts: {last_err}')


def slim_task(t: dict) -> dict:
    """Reduce a raw LS task to what import_ls_batch.py actually reads.

    Keeps:
      - id
      - data (full — let import script pick image/media/url)
      - annotations: list of {result: [...]}, only the result array is needed
    """
    anns_in = t.get('annotations') or []
    anns_out = []
    for a in anns_in:
        if not isinstance(a, dict): continue
        res = a.get('result') or []
        if not isinstance(res, list): continue
        anns_out.append({'result': res})
    return {
        'id':          t.get('id'),
        'data':        t.get('data') or {},
        'annotations': anns_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project',   type=int, required=True)
    ap.add_argument('--view',      type=int, required=True)
    ap.add_argument('--out',       required=True, help='Output JSON path')
    ap.add_argument('--page-size', type=int, default=150)
    ap.add_argument('--max-retries', type=int, default=5)
    ap.add_argument('--limit',     type=int, default=0,
                    help='Stop after N tasks (0 = all). Useful for smoke tests.')
    args = ap.parse_args()

    if not LS_TOKEN:
        print('ERR: LS_API_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'=== Fetch LS view ===')
    print(f'  base    : {LS_BASE}')
    print(f'  project : {args.project}')
    print(f'  view    : {args.view}')
    print(f'  out     : {out_path}')
    print(f'  page sz : {args.page_size}\n')

    collected = []  # list of slim_task dicts
    seen_ids  = set()
    t0 = time.time()

    with httpx.Client(http2=False) as client:
        # First page to get total
        page = 1
        d = fetch_page(client, args.project, args.view, page, args.page_size,
                       args.max_retries)
        total = int(d.get('total') or 0)
        print(f'  total in view: {total}\n')

        tasks = d.get('tasks') or []
        for t in tasks:
            tid = t.get('id')
            if tid is None or tid in seen_ids: continue
            collected.append(slim_task(t))
            seen_ids.add(tid)

        while True:
            done = len(collected)
            pct  = (done / total * 100) if total else 0.0
            dt   = time.time() - t0
            rate = done / dt if dt else 0
            print(f'  page {page:>3}: have {done:>5}/{total} '
                  f'({pct:5.1f}%, {rate:5.1f}/s, {dt:5.0f}s elapsed)', flush=True)

            if args.limit and len(collected) >= args.limit:
                print(f'  --limit {args.limit} reached, stopping')
                collected = collected[:args.limit]
                break
            if not tasks:           # empty page = end of stream
                break
            if len(collected) >= total and total > 0:
                break

            page += 1
            try:
                d = fetch_page(client, args.project, args.view, page,
                               args.page_size, args.max_retries)
            except Exception as e:
                # Save what we have so progress isn't lost
                tmp = out_path.with_suffix('.partial.json')
                tmp.write_text(json.dumps(collected, ensure_ascii=False),
                               encoding='utf-8')
                print(f'\n  ERR on page {page}: {e}', file=sys.stderr)
                print(f'  partial save -> {tmp}  ({len(collected)} tasks)', file=sys.stderr)
                sys.exit(2)

            tasks = d.get('tasks') or []
            new_in_page = 0
            for t in tasks:
                tid = t.get('id')
                if tid is None or tid in seen_ids: continue
                collected.append(slim_task(t))
                seen_ids.add(tid)
                new_in_page += 1
            # If a full page returned no new IDs we're either looping or hit end
            if new_in_page == 0:
                print(f'  page {page} returned no new ids — stopping')
                break

    # Final write
    out_path.write_text(json.dumps(collected, ensure_ascii=False), encoding='utf-8')
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f'\n  ✓ saved {len(collected)} tasks -> {out_path} ({size_mb:.1f} MB)')
    print(f'\nNext:')
    print(f'  python scripts/import_ls_batch.py --input {out_path} \\')
    print(f'       --session "2026-05-28_ls_underage" --dry-run')


if __name__ == '__main__':
    main()
