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

    Also covers corrupt/malformed source images (libvips decode errors, HTTP
    400/413 from Piper) — re-encoding through PIL in _resolve_url is best-effort
    but won't save genuinely truncated files. Marking them done avoids
    retrying the same dead file forever.
    """
    if not err_msg:
        return False
    em = err_msg.lower()
    if ('no face' in em
            or 'face_detect' in em
            or 'no faces detected' in em
            or 'face not found' in em):
        return True
    if any(s in em for s in (
            'premature end',
            'vipsjpeg',
            'unable to decode',
            'invalid image',
            'corrupt jpeg',
            'bad image data',
            'http 400',
            'http 413',
            "'400 bad request'",
            "'413 payload too large'")):
        return True
    return False


def _resolve_url(url):
    """If url is an http(s) URL or already a data URI, pass through. Otherwise
    treat it as a local file path (absolute, or relative to project root) and
    normalise it through Pillow into a JPEG data URI before base64-encoding.

    Why PIL-normalise: the Borderlands import contains a long tail of GIFs,
    corrupt JPEGs and oversize files. Sending them raw triggers libvips errors
    on Piper's side (500/400/VipsJpeg). Re-encoding through Pillow gives Piper
    a clean static RGB JPEG sized to fit under the request payload cap. PIL
    failures fall back to raw bytes so a corrupt file still surfaces a clean
    Piper error rather than crashing locally.
    """
    if not url:
        return url
    if url.startswith(('http://', 'https://', 'data:')):
        return url
    import base64, mimetypes, io
    from pathlib import Path as _P
    p = _P(url)
    if not p.is_absolute():
        p = BASE_DIR / url

    MAX_SIDE = 768
    MAX_BYTES = 3_500_000
    Q_HI, Q_LO = 88, 72

    try:
        from PIL import Image, ImageOps
    except ImportError:
        # No Pillow → original raw-bytes behaviour
        mime, _ = mimetypes.guess_type(p.name)
        if not mime or not mime.startswith('image/'):
            ext = p.suffix.lower().lstrip('.')
            mime = f'image/{"jpeg" if ext == "jpg" else (ext or "jpeg")}'
        return f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode('ascii')

    try:
        with Image.open(p) as im:
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            if im.mode in ('RGBA', 'LA', 'P'):
                bg = Image.new('RGB', im.size, (0, 0, 0))
                src = im.convert('RGBA') if im.mode == 'P' else im
                bg.paste(src, mask=src.split()[-1] if src.mode in ('RGBA', 'LA') else None)
                im = bg
            elif im.mode != 'RGB':
                im = im.convert('RGB')
            w, h = im.size
            longest = max(w, h)
            if longest > MAX_SIDE:
                ratio = MAX_SIDE / longest
                im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=Q_HI, optimize=True)
            if buf.tell() > MAX_BYTES:
                buf = io.BytesIO()
                im.save(buf, 'JPEG', quality=Q_LO, optimize=True)
        raw = buf.getvalue()
    except Exception:
        raw = p.read_bytes()
    return 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')


def run_one(item_id, src, url, label, max_polls=50, poll_delay=2.5):
    """Submit one image to V11's native pipeline and wait for result.

    V11 pipeline ce79f7e299 top-level input is `providers_e0` (verified via
    GET /api/projects/ce79f7e299 dump — same convention as d2911d10bb).
    Sending plain `providers` is silently ignored. The "historically working"
    state used the default fallback `[siglip2, hive]` (siglip2 was accidentally
    activated even though our explicit key was wrong).
    """
    try:
        r = _post_launch_with_retry(
            f'{PIPER_BASE}/projects/{V11_PROJECT}/launch',
            {'inputs': {'image': _resolve_url(url), 'providers_e0': ['siglip2']}},
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

    if source in ('all', 'borderlands'):
        try:
            for r in conn.execute(
                """SELECT id, label, local_path FROM borderlands_pool
                    WHERE local_path IS NOT NULL
                      AND (deleted IS NULL OR deleted=0)"""):
                # local_path is project-relative ("data/borderlands/<file>")
                # _resolve_url() will convert it to a data URI at launch time.
                items.append((r['id'], 'borderlands', r['label'], r['local_path']))
        except Exception:
            # borderlands_pool may not exist yet — silently skip
            pass

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
    """Atomic JSON write — UTF-8 explicit + fsync, then rename.

    CRITICAL: encoding='utf-8' must be explicit. Default on Windows is
    locale.getpreferredencoding (cp1251 for ru-RU), which silently mangles
    cyrillic characters in borderlands IDs (e.g. 'bl_..._эротические_') —
    every save corrupted the file mid-stream and broke subsequent reads.
    """
    tmp_p = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_p, path)


def _check_pipeline_providers(needed):
    """Fail-fast if the pipeline's prepare_params node enum doesn't include all
    providers we plan to request. (Pipeline JS silently drops unknown providers,
    so missing data would only surface as a half-broken result hours later.)"""
    try:
        r = httpx.get(f'{PIPER_BASE}/projects/{V11_PROJECT}',
                      headers=hdr(), timeout=20)
        if r.status_code != 200:
            print(f'WARN: pipeline config fetch failed (HTTP {r.status_code}); '
                  'skipping providers validation.', file=sys.stderr)
            return True
        data = r.json()
        pipe = data.get('pipeline')
        if isinstance(pipe, str):
            pipe = json.loads(pipe)
        top_inputs = (pipe or {}).get('inputs') or {}
        nodes = (pipe or {}).get('nodes') or {}
        prep = nodes.get('prepare_params') or {}
        # Top-level providers_e0 (preferred) → node-level providers (fallback)
        enum = (((top_inputs.get('providers_e0') or {}).get('enum'))
                or ((prep.get('inputs') or {}).get('providers') or {}).get('enum')
                or [])
        if not enum:
            return True   # No prepare_params or no enum — older pipeline shape; skip.
        missing = [p for p in needed if p not in enum]
        if missing:
            print('\n!!! Pipeline {} does NOT support providers: {}'
                  .format(V11_PROJECT, missing), file=sys.stderr)
            print('!!! Allowed: {}'.format(enum), file=sys.stderr)
            print('!!! Add them at https://piper-next.artworks.ai/en/projects/{}'
                  .format(V11_PROJECT), file=sys.stderr)
            return False
        print(f'  pipeline providers OK — enum: {enum}', flush=True)
        return True
    except Exception as e:
        print(f'WARN: pipeline validation failed ({e}); proceeding.', file=sys.stderr)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='all',
                    choices=['all', 'ls', 'grafana', 'k30', 'borderlands'])
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--limit', type=int, default=None,
                    help='Process first N items only (smoke test)')
    ap.add_argument('--out', default=None,
                    help='Override output path (use for smoke tests)')
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else OUT_PATH

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    # V11 pipeline calls inputs.providers=['siglip2'] (it has its own siglip2-→LGBM chain).
    if not _check_pipeline_providers(['siglip2']):
        sys.exit(2)

    items = load_items(args.source)
    if args.limit:
        items = items[:args.limit]
    print(f'Total items for source={args.source}: {len(items)}', flush=True)

    # Resume from existing file. Only done=True records count as "already
    # scored" — error-without-done records get retried.
    # Read raw bytes + errors='ignore' so a few corrupt bytes anywhere in the
    # file (legacy bug — see recover_json_caches.py) don't make us re-do the
    # entire 25k-record history.
    existing = {}
    if out_path.exists():
        try:
            raw = out_path.read_bytes()
            try:
                data = json.loads(raw.decode('utf-8'))
            except UnicodeDecodeError:
                data = json.loads(raw.decode('utf-8', errors='ignore'))
            for r in data:
                if r.get('done') and r.get('id'):
                    existing[r['id']] = r
        except Exception:
            pass

    todo = [it for it in items if it[0] not in existing]
    print(f'  already done: {len(existing)}', flush=True)
    print(f'  todo:         {len(todo)}', flush=True)
    print(f'  workers:      {args.workers}', flush=True)
    print(f'  project:      {V11_PROJECT}  (V11 native)', flush=True)
    print(f'  output:       {out_path}', flush=True)
    print(f'  retry policy: {LAUNCH_MAX_TRIES} attempts, backoff {LAUNCH_BASE_DELAY}s x2^n', flush=True)

    if not todo:
        _atomic_write(out_path, list(existing.values()))
        print('Nothing to do, file already complete.')
        return

    results = list(existing.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, iid, src, url, lbl): (iid, src)
                   for (iid, src, lbl, url) in todo}
        # Counter shows progress through THIS run (1..len(todo)). Previously it
        # started at len(existing) and divided by len(items), giving nonsense
        # like [25441/9829] when scoring only the borderlands subset of a file
        # that already had ls/grafana/k30 entries.
        n_done  = 0
        n_err   = 0
        n_seen  = 0
        n_todo  = len(todo)
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n_seen += 1
            if res.get('done'):
                n_done += 1
                status = 'OK/nf' if res.get('no_face') else 'OK'
            else:
                n_err += 1
                status = f'ERR({res.get("error", "?")[:30]})'
            sid = (res.get('source') or '')[:7]
            print(f'[{n_seen:4d}/{n_todo}] {status} {res["id"][:24]:<26} ({sid:<8} {res.get("label")})',
                  flush=True)
            if n_seen % 25 == 0:
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
