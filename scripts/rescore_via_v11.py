#!/usr/bin/env python3
"""
rescore_via_v11.py
------------------
Run our LS + Grafana images through V11's NATIVE Piper pipeline
(project ce79f7e299) to obtain 317-tag input in the exact format V11 was
trained on. This is the right way to evaluate V11 — feeding it Tom's
pipeline output (a4aa9dbd9c) was wrong because the tag distribution
differs (~6% adult-feature overlap).

K30 is intentionally EXCLUDED by default because that dataset is not
yet manually labelled — rescoring it now would burn API calls on items
we cannot evaluate. Pass --source k30 explicitly if you want it anyway.

V11 pipeline has a face_detect stage: items without a detectable face
fail terminally with "no face..." error. Those are recorded with
{no_face: True, done: True} so resume skips them (and the gallery can
treat them as "V11 doesn't evaluate this item").

Resilience:
  - POST /launch is retried up to 4 times with exponential backoff
    (2s → 4s → 8s → 16s) on 5xx, timeouts, and network errors.
  - GET /launches/state polling tolerates per-call exceptions
    (TimeoutException, ConnectError, etc.) and just continues polling
    instead of failing the whole item.

Output: data/v11_native_scores.json — list of records:
  {id, source, label, minor, adult, lgbm_score, lgbm_blocked, lgbm_threshold,
   underage_labels, adult_labels, no_underage_labels, done}
  or for terminal pipeline failures: {id, source, label, no_face, error, done}

Usage (run from PowerShell on Windows host, NOT from sandbox):
    python scripts/rescore_via_v11.py                     # LS + Grafana (no K30)
    python scripts/rescore_via_v11.py --source ls         # LS only
    python scripts/rescore_via_v11.py --source grafana --workers 12
    python scripts/rescore_via_v11.py --source k30        # opt-in, K30 only
    python scripts/rescore_via_v11.py --limit 50          # smoke test
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
V11_PROJECT = 'ce79f7e299'   # V11 native pipeline
WORKERS     = 12
OUT_PATH    = BASE_DIR / 'data' / 'v11_native_scores.json'

# Retry policy for transient infra errors (502/503/504, timeouts, conn reset).
LAUNCH_MAX_TRIES   = 4         # initial + 3 retries
LAUNCH_BASE_DELAY  = 2.0       # 2s → 4s → 8s → 16s


def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}


# Transient exception types: network glitches, timeouts, malformed responses.
_TRANSIENT_EXC = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.NetworkError,
)


def _post_launch_with_retry(url, payload):
    """POST with exponential backoff on 5xx / timeouts / network errors.

    Returns the httpx.Response on success. Raises the last exception/error
    if all retries are exhausted.
    """
    last_err = None
    for attempt in range(LAUNCH_MAX_TRIES):
        try:
            r = httpx.post(url, headers=hdr(), json=payload, timeout=30)
            if 500 <= r.status_code < 600:
                last_err = f'HTTP {r.status_code}'
                if attempt < LAUNCH_MAX_TRIES - 1:
                    time.sleep(LAUNCH_BASE_DELAY * (2 ** attempt))
                    continue
                # Out of tries: surface the status.
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
    """GET that returns (status_code, json_or_None) and swallows transient errors.

    Returns (None, None) on transient failure — caller treats it as another
    polling miss and continues sleeping.
    """
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
        tmp = Path(os.environ.get('TEMP', '.')) / '_v11_rescore.db'
    else:
        tmp = Path('/tmp/_v11_rescore.db')
    shutil.copy2(src, tmp)
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def _is_terminal_pipeline_error(err_msg):
    """True if the pipeline failed deterministically (won't succeed on retry).

    V11 pipeline ce79f7e299 has a face_detect stage that fails on images
    without a detectable face. These items will never produce a V11 score —
    we record them as 'done with no_face=True' so resume skips them.
    """
    if not err_msg:
        return False
    em = err_msg.lower()
    return ('no face' in em
            or 'face_detect' in em
            or 'no faces detected' in em
            or 'face not found' in em)


def run_one(item_id, src, url, label, max_polls=50, poll_delay=2.5):
    """Submit one image to V11's native pipeline and wait for result."""
    try:
        r = _post_launch_with_retry(
            f'{PIPER_BASE}/projects/{V11_PROJECT}/launch',
            {'inputs': {'image': url, 'providers': ['siglip2']}},
        )
        run_id = r.json()['_id']
        for _ in range(max_polls):
            time.sleep(poll_delay)
            status, state = _get_state_safe(f'{PIPER_BASE}/launches/{run_id}/state')
            if status != 200 or not state:
                continue  # transient error — keep polling
            if state.get('errors'):
                err_msg = str(state['errors'][0])[:200]
                if _is_terminal_pipeline_error(err_msg):
                    return {
                        'id': item_id, 'source': src, 'label': label,
                        'no_face': True, 'error': err_msg, 'done': True,
                    }
                return {'id': item_id, 'source': src, 'label': label, 'error': err_msg}
            outputs = state.get('outputs') or {}
            if 'siglip2_details' in outputs:
                under  = (outputs.get('siglip2_details') or {}).get('underage', {})
                labels = under.get('labels', {})
                lgbm   = under.get('lgbm') or {}
                return {
                    'id':                 item_id,
                    'source':             src,
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
        return {'id': item_id, 'source': src, 'label': label, 'error': 'timeout'}
    except Exception as e:
        return {'id': item_id, 'source': src, 'label': label, 'error': f'{type(e).__name__}: {str(e)[:120]}'}


def load_items(source):
    """Return list of (id, source, label, url) tuples to score."""
    conn = _open_db()
    items = []

    # NB: 'all' = ls + grafana (K30 deliberately excluded — not yet labelled).
    # Pass --source k30 explicitly if you need to rescore K30 anyway.
    if source in ('all', 'grafana'):
        for r in conn.execute("""SELECT id, label, thumb_url FROM grafana_pool
                                  WHERE thumb_url IS NOT NULL
                                    AND (deleted IS NULL OR deleted=0)"""):
            items.append((r['id'], 'grafana', r['label'], r['thumb_url']))

    if source == 'k30':
        for r in conn.execute("""SELECT id, label, thumb_url FROM k30_pool
                                  WHERE thumb_url IS NOT NULL
                                    AND (deleted IS NULL OR deleted=0)"""):
            items.append((r['id'], 'k30', r['label'], r['thumb_url']))

    if source in ('all', 'ls'):
        ls_rows = []
        try:
            for r in conn.execute("""SELECT task_id, media, age_from
                                      FROM ls_images WHERE media IS NOT NULL"""):
                if r['media']:
                    af = r['age_from']
                    lbl = ('child' if af is not None and af <= 14
                           else 'teen' if af is not None and af <= 17
                           else 'adult' if af is not None else None)
                    ls_rows.append((f"ls_{r['task_id']}", 'labelstudio', lbl, r['media']))
        except Exception:
            pass
        if not ls_rows:
            try:
                qd = json.loads((BASE_DIR / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8'))
                for v in qd.values():
                    tid = v.get('task_id')
                    if tid is None: continue
                    media = v.get('media')
                    if not media: continue
                    af = (v.get('age') or {}).get('ageFrom')
                    lbl = ('child' if af is not None and af <= 14
                           else 'teen' if af is not None and af <= 17
                           else 'adult' if af is not None else None)
                    ls_rows.append((f"ls_{tid}", 'labelstudio', lbl, media))
            except Exception as e:
                print(f'WARN: LS load fallback failed: {e}', file=sys.stderr)
        items.extend(ls_rows)

    conn.close()
    return items


def _atomic_write(path, data):
    tmp_p = path.with_suffix(path.suffix + '.tmp')
    tmp_p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp_p, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='all', choices=['all', 'ls', 'grafana', 'k30'])
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--limit', type=int, default=None,
                    help='Process first N items only (smoke test)')
    ap.add_argument('--out', default=None,
                    help='Override output path (use for smoke tests)')
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else OUT_PATH

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    items = load_items(args.source)
    if args.limit:
        items = items[:args.limit]
    print(f'Total items for source={args.source}: {len(items)}', flush=True)

    existing = {}
    if out_path.exists():
        try:
            for r in json.loads(out_path.read_text()):
                if r.get('done'):
                    existing[r['id']] = r
        except Exception:
            pass

    todo = [it for it in items if it[0] not in existing]
    print(f'  already done: {len(existing)}', flush=True)
    print(f'  todo:         {len(todo)}', flush=True)
    print(f'  workers:      {args.workers}', flush=True)
    print(f'  project:      {V11_PROJECT}  (V11 native)', flush=True)
    print(f'  output:       {out_path}', flush=True)
    print(f'  retry policy: {LAUNCH_MAX_TRIES} attempts, backoff {LAUNCH_BASE_DELAY}s×2^n\n', flush=True)

    results = list(existing.values())

    if not todo:
        _atomic_write(out_path, results)
        print('Nothing to do, file already complete.')
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, iid, src, url, lbl): (iid, src)
                   for (iid, src, lbl, url) in todo}
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
            sid = (res.get('source') or '')[:7]
            print(f'[{n_total:5d}/{len(items)}] {status} {res["id"][:24]:<26} ({sid:<8} {res.get("label")})',
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
    print(f'  done (no_face/skip):   {no_face}  <- V11 cannot evaluate these', flush=True)
    print(f'  retry-able errors:     {err}', flush=True)
    print(f'  saved to:              {out_path}', flush=True)


if __name__ == '__main__':
    main()
