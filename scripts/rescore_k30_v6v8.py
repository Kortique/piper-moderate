#!/usr/bin/env python3
"""
rescore_k30_v6v8.py
-------------------
Run K30 (Tom's external dataset, 6675 items) through Piper's PRODUCTION
pipeline d2911d10bb to obtain full 180-tag input — the taxonomy V6 and V8
were trained on. K30 storage natively contains only a sparse top-5
extract, which makes V6 + V8 underperform on K30 in the gallery (V6 child
~65%, V8 child ~87% as of now). With this rescore those numbers should
rise to in-distribution levels.

V11 K30 scoring is handled separately by rescore_via_v11.py through
project ce79f7e299 (V11's native pipeline) — do NOT mix the two.

Output: data/k30_rescored.json — list of records:
  {id, label, minor, adult, lgbm_score, lgbm_blocked, lgbm_threshold,
   underage_labels, adult_labels, no_underage_labels, done}
  or for terminal pipeline failures: {id, label, no_face, error, done}

Resilience (same as rescore_via_v11.py):
  - POST /launch: up to 4 attempts with exponential backoff 2s→4s→8s→16s
    on 5xx / TimeoutException / ConnectError / NetworkError.
  - GET /launches/state: transient errors swallowed, polling continues.

Usage (run from PowerShell on Windows host, NOT from sandbox):
    python -u scripts/rescore_k30_v6v8.py                  # all K30 (resume)
    python -u scripts/rescore_k30_v6v8.py --workers 6      # safer parallelism
    python -u scripts/rescore_k30_v6v8.py --limit 50       # smoke test
"""
import os, sys, json, time, sqlite3, argparse, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()
BASE_DIR    = Path(__file__).resolve().parent.parent
TOKEN       = os.getenv('PIPER_TOKEN', '')
PIPER_BASE  = 'https://piper-next.artworks.ai/api'
K30_PROJECT = 'd2911d10bb'   # production pipeline (180-tag, what V6/V8 expect)
WORKERS     = 8
OUT_PATH    = BASE_DIR / 'data' / 'k30_rescored.json'

# Retry policy for transient infra errors (502/503/504, timeouts, conn reset).
LAUNCH_MAX_TRIES  = 4
LAUNCH_BASE_DELAY = 2.0    # 2s → 4s → 8s → 16s


def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}


_TRANSIENT_EXC = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.NetworkError,
)


def _post_launch_with_retry(url, payload):
    """POST with exponential backoff on 5xx / timeouts / network errors."""
    last_err = None
    for attempt in range(LAUNCH_MAX_TRIES):
        try:
            r = httpx.post(url, headers=hdr(), json=payload, timeout=30)
            if 500 <= r.status_code < 600:
                last_err = f'HTTP {r.status_code}'
                if attempt < LAUNCH_MAX_TRIES - 1:
                    time.sleep(LAUNCH_BASE_DELAY * (2 ** attempt))
                    continue
                r.raise_for_status()
            r.raise_for_status()
            return r
        except _TRANSIENT_EXC as e:
            last_err = f'{type(e).__name__}: {str(e)[:80]}'
            if attempt < LAUNCH_MAX_TRIES - 1:
                time.sleep(LAUNCH_BASE_DELAY * (2 ** attempt))
                continue
            raise
    raise httpx.HTTPError(f'launch failed after {LAUNCH_MAX_TRIES} attempts: {last_err}')


def _get_state_safe(url, timeout=15):
    """GET that swallows transient errors; returns (status_code, json_or_None)."""
    try:
        rs = httpx.get(url, headers=hdr(), timeout=timeout)
        if rs.status_code != 200:
            return rs.status_code, None
        try:
            return 200, rs.json()
        except Exception:
            return 200, None
    except _TRANSIENT_EXC:
        return None, None


def _open_db():
    src = BASE_DIR / 'gallery.db'
    if os.name == 'nt':
        tmp = Path(os.environ.get('TEMP', '.')) / '_k30_v6v8_rescore.db'
    else:
        tmp = Path('/tmp/_k30_v6v8_rescore.db')
    shutil.copy2(src, tmp)
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def _is_terminal_pipeline_error(err_msg):
    if not err_msg:
        return False
    em = err_msg.lower()
    return ('no face' in em
            or 'face_detect' in em
            or 'no faces detected' in em
            or 'face not found' in em)


def run_one(item_id, url, label, max_polls=50, poll_delay=2.5):
    """Submit one image to the production pipeline and wait for result."""
    try:
        r = _post_launch_with_retry(
            f'{PIPER_BASE}/projects/{K30_PROJECT}/launch',
            {'inputs': {'image': url, 'providers': ['siglip2']}},
        )
        run_id = r.json()['_id']
        for _ in range(max_polls):
            time.sleep(poll_delay)
            status, state = _get_state_safe(f'{PIPER_BASE}/launches/{run_id}/state')
            if status != 200 or not state:
                continue
            if state.get('errors'):
                err_msg = str(state['errors'][0])[:200]
                if _is_terminal_pipeline_error(err_msg):
                    return {
                        'id': item_id, 'label': label,
                        'no_face': True, 'error': err_msg, 'done': True,
                    }
                return {'id': item_id, 'label': label, 'error': err_msg}
            outputs = state.get('outputs') or {}
            if 'siglip2_details' in outputs:
                under  = (outputs.get('siglip2_details') or {}).get('underage', {})
                labels = under.get('labels', {})
                lgbm   = under.get('lgbm') or {}
                return {
                    'id':                 item_id,
                    'label':              label,
                    'lgbm_score':         lgbm.get('score'),
                    'lgbm_blocked':       lgbm.get('blocked'),
                    'lgbm_threshold':     lgbm.get('threshold'),
                    'lgbm_n_features':    lgbm.get('n_features'),
                    'minor':              under.get('minor'),
                    'adult':              under.get('adult'),
                    'confidence':         under.get('confidence'),
                    'underage_labels':    labels.get('underage', {}),
                    'adult_labels':       labels.get('adult', {}),
                    'no_underage_labels': labels.get('no_underage', {}),
                    'done': True,
                }
        return {'id': item_id, 'label': label, 'error': 'timeout'}
    except Exception as e:
        return {'id': item_id, 'label': label, 'error': f'{type(e).__name__}: {str(e)[:120]}'}


def _atomic_write(path, data):
    tmp_p = path.with_suffix(path.suffix + '.tmp')
    tmp_p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp_p, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--limit', type=int, default=None,
                    help='Process first N K30 items only (smoke test)')
    ap.add_argument('--out', default=None,
                    help='Override output path (use for smoke tests)')
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else OUT_PATH

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    conn = _open_db()
    rows = conn.execute("""SELECT id, label, thumb_url FROM k30_pool
                            WHERE thumb_url IS NOT NULL
                              AND (deleted IS NULL OR deleted=0)""").fetchall()
    conn.close()
    if args.limit:
        rows = rows[:args.limit]
    print(f'Total K30 items in DB: {len(rows)}', flush=True)

    existing = {}
    if out_path.exists():
        try:
            for r in json.loads(out_path.read_text()):
                if r.get('done'):
                    existing[r['id']] = r
        except Exception:
            pass

    todo = [r for r in rows if r['id'] not in existing]
    print(f'  already done: {len(existing)}', flush=True)
    print(f'  todo:         {len(todo)}', flush=True)
    print(f'  workers:      {args.workers}', flush=True)
    print(f'  project:      {K30_PROJECT}  (production 180-tag for V6/V8)', flush=True)
    print(f'  output:       {out_path}', flush=True)
    print(f'  retry policy: {LAUNCH_MAX_TRIES} attempts, backoff {LAUNCH_BASE_DELAY}s×2^n\n', flush=True)

    results = list(existing.values())

    if not todo:
        _atomic_write(out_path, results)
        print('Nothing to do, file already complete.')
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, r['id'], r['thumb_url'], r['label']): r for r in todo}
        n_total = len(existing)
        n_done  = 0
        n_err   = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n_total += 1
            if res.get('done'):
                n_done += 1
                status = '✓nf' if res.get('no_face') else '✓'
            else:
                n_err += 1
                status = f'✗({res.get("error", "?")[:30]})'
            print(f'[{n_total:5d}/{len(rows)}] {status} {res["id"][:24]:<26} ({res.get("label")})',
                  flush=True)
            if n_total % 25 == 0:
                _atomic_write(out_path, results)

    _atomic_write(out_path, results)
    done    = sum(1 for r in results if r.get('done'))
    err     = sum(1 for r in results if not r.get('done'))
    no_face = sum(1 for r in results if r.get('no_face'))
    scored  = done - no_face
    print(f'\n=== Done ===', flush=True)
    print(f'  total:                 {len(results)}', flush=True)
    print(f'  done (scored):         {scored}', flush=True)
    print(f'  done (no_face/skip):   {no_face}', flush=True)
    print(f'  retry-able errors:     {err}', flush=True)
    print(f'  saved to:              {out_path}', flush=True)


if __name__ == '__main__':
    main()
