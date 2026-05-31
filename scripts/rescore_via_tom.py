#!/usr/bin/env python3
"""
rescore_via_tom.py
------------------
Run our LS + Grafana + K30 images through Tom's pipeline (project a4aa9dbd9c)
to obtain:
  1) Tom's K=30 model score (siglip2_details.underage.lgbm.score) → used as
     `k30tom` model in the gallery for disagreement comparison.
  2) 317-tag taxonomy with :x20 suffixes (siglip2_details.underage.labels.*)
     — this is V11's native input format, so we can score V11 properly on
     items previously stuck in fallback mode.

Output: data/tom_scores.json — list of records:
  {id, source, label, k30_score, k30_blocked, k30_threshold,
   minor, adult, underage_labels, adult_labels, no_underage_labels, done}

Usage:
    python scripts/rescore_via_tom.py                     # all sources, all items
    python scripts/rescore_via_tom.py --source ls         # only LS
    python scripts/rescore_via_tom.py --source grafana --workers 12
    python scripts/rescore_via_tom.py --limit 50          # smoke test
"""
import os, sys, json, time, sqlite3, argparse, shutil, ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()
BASE_DIR   = Path(__file__).resolve().parent.parent
TOKEN      = os.getenv('PIPER_TOKEN', '')
PIPER_BASE = 'https://piper-next.artworks.ai/api'
TOM_PROJECT = 'a4aa9dbd9c'
WORKERS     = 12
OUT_PATH    = BASE_DIR / 'data' / 'tom_scores.json'


def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}


def _open_db():
    src = BASE_DIR / 'gallery.db'
    tmp = Path('/tmp/_tom_rescore.db')
    shutil.copy2(src, tmp)
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_url(url):
    """If url is http(s) or already a data URI, pass through. Otherwise read the
    local file, normalise through Pillow (RGB → JPEG, ≤768px, ≤3.5 MB) and
    return as a base64 data URI. PIL normalisation is what makes the
    Borderlands long-tail of GIFs / corrupt JPEGs / oversize files actually
    decodable by Piper's libvips. PIL failure falls back to raw bytes."""
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


_PERMANENT_ERR_MARKERS = (
    'premature end', 'vipsjpeg', 'unable to decode', 'invalid image',
    'corrupt jpeg', 'bad image data',
    'no faces detected', 'face_detect',
)


def _is_permanent(err_msg: str) -> bool:
    """Errors that won't succeed on retry (bad source file or no-face content)."""
    if not err_msg:
        return False
    em = err_msg.lower()
    return any(m in em for m in _PERMANENT_ERR_MARKERS)


def run_one(item_id, src, url, label, max_polls=50, poll_delay=2.5):
    """Submit one image to Tom's pipeline and wait for result.

    NOTE: top-level input name varies per pipeline. Keeping `providers` here
    because Tom rescore was historically working with this key. If launches
    suddenly stop activating siglip2 — dump pipeline.inputs for a4aa9dbd9c
    and switch to the correct key.
    """
    try:
        r = httpx.post(f'{PIPER_BASE}/projects/{TOM_PROJECT}/launch', headers=hdr(),
                       json={'inputs': {'image': _resolve_url(url), 'providers': ['siglip2']}}, timeout=30)
        # Permanent client-side rejection — corrupt file or oversize payload.
        if r.status_code in (400, 413):
            return {'id': item_id, 'source': src, 'label': label,
                    'unprocessable': True,
                    'error': f'HTTP {r.status_code}',
                    'done': True}
        r.raise_for_status()
        run_id = r.json()['_id']
        for _ in range(max_polls):
            time.sleep(poll_delay)
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state', headers=hdr(), timeout=15)
            if rs.status_code != 200: continue
            state = rs.json()
            if state.get('errors'):
                err_msg = str(state['errors'][0])[:200]
                if _is_permanent(err_msg):
                    return {'id': item_id, 'source': src, 'label': label,
                            'unprocessable': True, 'error': err_msg, 'done': True}
                return {'id': item_id, 'source': src, 'label': label, 'error': err_msg[:120]}
            outputs = state.get('outputs') or {}
            if 'siglip2_details' in outputs:
                under = (outputs.get('siglip2_details') or {}).get('underage', {})
                labels = under.get('labels', {})
                lgbm = under.get('lgbm') or {}
                return {
                    'id':                item_id,
                    'source':            src,
                    'label':             label,
                    'k30_score':         lgbm.get('score'),
                    'k30_blocked':       lgbm.get('blocked'),
                    'k30_threshold':     lgbm.get('threshold'),
                    'k30_n_features':    lgbm.get('n_features'),
                    'minor':             under.get('minor'),
                    'adult':             under.get('adult'),
                    'confidence':        under.get('confidence'),
                    'underage_labels':   labels.get('underage', {}),
                    'adult_labels':      labels.get('adult', {}),
                    'no_underage_labels':labels.get('no_underage', {}),
                    'done': True,
                }
        return {'id': item_id, 'source': src, 'label': label, 'error': 'timeout'}
    except Exception as e:
        return {'id': item_id, 'source': src, 'label': label, 'error': str(e)[:120]}


def load_items(source: str):
    """Return list of (id, source, label, url) tuples to score."""
    conn = _open_db()
    items = []

    if source in ('all', 'grafana'):
        for r in conn.execute("""SELECT id, label, thumb_url FROM grafana_pool
                                  WHERE thumb_url IS NOT NULL
                                    AND (deleted IS NULL OR deleted=0)"""):
            items.append((r['id'], 'grafana', r['label'], r['thumb_url']))

    if source in ('all', 'k30'):
        for r in conn.execute("""SELECT id, label, thumb_url FROM k30_pool
                                  WHERE thumb_url IS NOT NULL
                                    AND (deleted IS NULL OR deleted=0)"""):
            items.append((r['id'], 'k30', r['label'], r['thumb_url']))

    if source in ('all', 'ls'):
        # LS items: prefer ls_images table; fall back to qwen3_age_results.json
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

    if source in ('all', 'borderlands'):
        try:
            for r in conn.execute(
                """SELECT id, label, local_path FROM borderlands_pool
                    WHERE local_path IS NOT NULL
                      AND (deleted IS NULL OR deleted=0)"""):
                items.append((r['id'], 'borderlands', r['label'], r['local_path']))
        except Exception:
            pass

    conn.close()
    return items


def _check_pipeline_providers(needed):
    """Fail-fast if pipeline's prepare_params doesn't list all needed providers."""
    try:
        r = httpx.get(f'{PIPER_BASE}/projects/{TOM_PROJECT}',
                      headers=hdr(), timeout=20)
        if r.status_code != 200:
            print(f'WARN: pipeline config fetch failed (HTTP {r.status_code}); '
                  'skipping validation.', file=sys.stderr)
            return True
        data = r.json()
        pipe = data.get('pipeline')
        if isinstance(pipe, str):
            pipe = json.loads(pipe)
        top_inputs = (pipe or {}).get('inputs') or {}
        nodes = (pipe or {}).get('nodes') or {}
        prep = nodes.get('prepare_params') or {}
        enum = (((top_inputs.get('providers_e0') or {}).get('enum'))
                or ((prep.get('inputs') or {}).get('providers') or {}).get('enum')
                or [])
        if not enum:
            return True
        missing = [p for p in needed if p not in enum]
        if missing:
            print('\n!!! Pipeline {} does NOT support providers: {}'
                  .format(TOM_PROJECT, missing), file=sys.stderr)
            print('!!! Allowed: {}'.format(enum), file=sys.stderr)
            print('!!! Add them at https://piper-next.artworks.ai/en/projects/{}'
                  .format(TOM_PROJECT), file=sys.stderr)
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
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    # Tom pipeline only feeds siglip2 (its built-in LGBM does the rest).
    if not _check_pipeline_providers(['siglip2']):
        sys.exit(2)

    items = load_items(args.source)
    if args.limit:
        items = items[:args.limit]
    print(f'Total items for source={args.source}: {len(items)}', flush=True)

    # Resume from existing file. CRITICAL: only treat records with done=True as
    # "already scored". Records that have an 'id' but no 'done' are previous
    # transient errors (HTTP 5xx, timeouts, terminal pipeline errors). We want
    # to retry those, not skip them — otherwise 5 failed items become a
    # permanent gap nobody will ever revisit.
    # Read raw bytes + errors='ignore' so a few corrupt bytes anywhere in the
    # file (legacy bug — see recover_json_caches.py) don't make us re-do the
    # whole 25k records.
    existing = {}
    if OUT_PATH.exists():
        try:
            raw = OUT_PATH.read_bytes()
            try:
                data = json.loads(raw.decode('utf-8'))
            except UnicodeDecodeError:
                data = json.loads(raw.decode('utf-8', errors='ignore'))
            for r in data:
                if r.get('id') and r.get('done'):
                    existing[r['id']] = r
        except Exception:
            pass

    todo = [it for it in items if it[0] not in existing]
    if not todo:
        _atomic_write(OUT_PATH, list(existing.values()))
        print('Nothing to do, all items already scored.')
        return
    print(f'  already done: {len(existing)}', flush=True)
    print(f'  todo:         {len(todo)}', flush=True)
    print(f'  workers:      {args.workers}', flush=True)
    print(f'  project:      {TOM_PROJECT}\n', flush=True)

    results = list(existing.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, iid, src, url, lbl): (iid, src)
                   for (iid, src, lbl, url) in todo}
        # Counter shows progress through THIS run (1..len(todo)), not (existing+1)..items_total.
        n_done = 0
        n_err  = 0
        n_seen = 0
        n_todo = len(todo)
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n_seen += 1
            if res.get('done'):
                n_done += 1
                status = 'OK'
            else:
                n_err += 1
                status = f'ERR({res.get("error", "?")[:30]})'
            sid = (res.get('source') or '')[:7]
            print(f'[{n_seen:4d}/{n_todo}] {status} {res["id"][:24]:<26} ({sid:<8} {res.get("label")})',
                  flush=True)
            if n_seen % 25 == 0:
                _atomic_write(OUT_PATH, results)

    _atomic_write(OUT_PATH, results)
    print(f'\n=== Done ===  total={len(results)}, done={n_done}, errors={n_err}', flush=True)


def _atomic_write(path, data):
    """Atomic JSON write — UTF-8 explicit + fsync, then rename.

    CRITICAL: encoding='utf-8' must be explicit. Default on Windows is
    locale.getpreferredencoding (cp1251 for ru-RU), which silently mangles
    cyrillic characters in borderlands IDs (e.g. 'bl_..._эротические_') —
    every save corrupted the file mid-stream and broke subsequent reads.
    """
    import os as _os
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        _os.fsync(f.fileno())
    _os.replace(tmp, path)


if __name__ == '__main__':
    main()
